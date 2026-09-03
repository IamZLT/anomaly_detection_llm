import os
import re
import shutil
import signal
import socket
import subprocess
import time
from typing import Optional

from tqdm.auto import tqdm
from transformers import TrainerCallback
from transformers.trainer_utils import has_length


class PerEpochProgressCallback(TrainerCallback):
    """
    HuggingFace 默认 ProgressCallback 用一条 tqdm 覆盖全程 max_steps。
    本回调改为每个 epoch 一条进度条（0%→100% 对应当轮 optimizer steps）。
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None
        self.current_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.current_step = int(state.global_step)

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        ne = max(1, int(getattr(state, "num_train_epochs", 1) or 1))
        ms = max(0, int(getattr(state, "max_steps", 0) or 0))
        gs = int(state.global_step)
        remaining = max(0, ms - gs)
        spe = max(1, (ms + ne - 1) // ne) if ms > 0 else 1
        total_this_epoch = max(1, min(spe, remaining))
        if self.training_bar is not None:
            self.training_bar.close()
        ep_label = int(state.epoch or 0) + 1
        ep_label = max(1, min(ep_label, ne))
        desc = f"Epoch {ep_label}/{ne}"
        self.training_bar = tqdm(total=total_this_epoch, dynamic_ncols=True, desc=desc, leave=True)
        self.current_step = gs

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            delta = int(state.global_step) - self.current_step
            if delta:
                self.training_bar.update(delta)
            self.current_step = int(state.global_step)

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader), leave=self.training_bar is None, dynamic_ncols=True
                )
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_predict(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 不往 tqdm 写 logs（否则会与 PrettyTrainLogCallback 重复一行 dict）
        pass

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            self.training_bar.close()
            self.training_bar = None


class PrettyTrainLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if not _is_main_process():
            return control
        if not logs:
            return control

        m = model.module if (model is not None and hasattr(model, "module")) else model
        extra = {}
        if m is not None and hasattr(m, "get_last_loss_stats"):
            try:
                extra = m.get_last_loss_stats() or {}
            except Exception:
                extra = {}

        # epoch is fractional progress (e.g. 0.02 means 2% of epoch 1)
        step = int(getattr(state, "global_step", 0) or 0)
        epoch = getattr(state, "epoch", None)
        epoch_s = "?"
        if epoch is not None:
            try:
                epoch_s = f"{float(epoch):.2f}"
            except Exception:
                epoch_s = str(epoch)

        loss_total = logs.get("loss", extra.get("loss_total", None))
        lr = logs.get("learning_rate", None)
        gnorm = logs.get("grad_norm", None)

        parts = []
        if loss_total is not None:
            parts.append(f"loss={float(loss_total):.4f}")
        if extra.get("loss_lm") is not None:
            parts.append(f"loss_lm={float(extra['loss_lm']):.4f}")
        if lr is not None:
            parts.append(f"lr={float(lr):.3e}")
        if gnorm is not None:
            parts.append(f"gnorm={float(gnorm):.2f}")

        _train_log(f"step={step} epoch={epoch_s} | " + " ".join(parts))
        return control


def _is_main_process() -> bool:
    """单进程或未设 RANK 时视为主进程；多卡以 RANK==0 为准，否则退化为 LOCAL_RANK==0。"""
    r = os.environ.get("RANK")
    if r is not None:
        return int(r) == 0
    lr = os.environ.get("LOCAL_RANK")
    if lr is not None:
        return int(lr) == 0
    return True


def _train_log(msg: str, main_only: bool = False) -> None:
    if main_only and not _is_main_process():
        return
    rank = os.environ.get("RANK", "?")
    lr = os.environ.get("LOCAL_RANK", "?")
    prefix = f"[train rank={rank} local={lr}] "
    print(prefix + msg, flush=True)


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def tensorboard_event_dir(output_dir: str) -> str:
    """All scalars / images / text go in one folder so TB shows curves and vis together."""
    d = os.path.abspath(os.path.join(output_dir, "tb"))
    os.makedirs(d, exist_ok=True)
    return d


def _tensorboard_logdir(cfg: dict, output_dir: str) -> str:
    tb_cfg = cfg.get("tensorboard") or {}
    override = tb_cfg.get("logdir")
    if override not in (None, "", "null", "None"):
        return os.path.abspath(os.path.expanduser(str(override)))
    return tensorboard_event_dir(output_dir)


def _pid_listening_on_port(port: int) -> Optional[int]:
    try:
        out = subprocess.check_output(["ss", "-lptn", f"sport = :{port}"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    m = re.search(r"pid=(\d+)", out)
    return int(m.group(1)) if m else None


def _proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except Exception:
        return ""


def _cmdline_logdir(cmdline: str) -> str:
    m = re.search(r"--logdir(?:\s+|=)(\S+)", cmdline)
    if not m:
        return ""
    return os.path.abspath(m.group(1))


def _auto_start_tensorboard(cfg: dict, output_dir: str) -> None:
    tb_cfg = cfg.get("tensorboard", {})
    if not bool(tb_cfg.get("auto_start", True)):
        return

    host = str(tb_cfg.get("host", "0.0.0.0"))
    port = int(tb_cfg.get("port", 5000))
    probe_host = "127.0.0.1" if host == "0.0.0.0" else host
    log_dir = _tensorboard_logdir(cfg, output_dir)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    if _is_port_in_use(probe_host, port):
        pid = _pid_listening_on_port(port)
        cmd_now = _proc_cmdline(pid) if pid else ""
        watching = _cmdline_logdir(cmd_now)
        if pid and "tensorboard" in cmd_now and watching == os.path.abspath(log_dir):
            print(f"[TensorBoard] 端口 {port} 已在看 {log_dir}: http://127.0.0.1:{port}")
            return
        if pid and "tensorboard" in cmd_now:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.8)
                print(f"[TensorBoard] 旧进程 pid={pid} 盯的是 {watching or '?'}，已重启为 {log_dir}")
            except Exception as e:
                print(f"[TensorBoard] 端口 {port} 被占用且无法重启 ({e}): http://127.0.0.1:{port}")
                return
        else:
            print(f"[TensorBoard] 端口 {port} 被其他进程占用，无法启动: http://127.0.0.1:{port}")
            return

    tb_bin = shutil.which("tensorboard")
    if tb_bin is None:
        print("[TensorBoard] 未找到 tensorboard 命令，跳过自动启动。")
        return

    cmd = [
        tb_bin,
        "--logdir",
        log_dir,
        "--host",
        host,
        "--port",
        str(port),
        "--reload_multifile",
        "true",
        "--reload_interval",
        "5",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"[TensorBoard] 已自动启动 (pid={proc.pid}) logdir={log_dir}: http://127.0.0.1:{port}")
    except Exception as e:
        print(f"[TensorBoard] 自动启动失败: {e}")


def _disable_hf_datasets_check() -> None:
    """
    项目根下本地目录 `datasets/` 会遮蔽 HuggingFace 的 `datasets` 包，
    导致 Trainer 里 `datasets.Dataset` 报 AttributeError。
    我们用的是自定义 torch Dataset，关闭该检查即可。
    """
    try:
        import transformers.trainer as hf_trainer

        hf_trainer.is_datasets_available = lambda: False  # type: ignore[assignment]
    except Exception:
        pass
    try:
        import transformers.utils.import_utils as import_utils

        import_utils._datasets_available = False  # type: ignore[attr-defined]
    except Exception:
        pass


def train_main(cfg: dict) -> None:
    raise RuntimeError(
        "旧 DINO/JSON 训练链路已移除。请用 python train.py --config configs/ad_llm_qwen35_2b_prior.yaml"
    )
