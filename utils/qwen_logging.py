import os
import sys
from typing import Any, Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback, TrainerControl, TrainerState


def _is_main_process() -> bool:
    r = os.environ.get("RANK")
    if r is not None:
        return int(r) == 0
    lr = os.environ.get("LOCAL_RANK")
    if lr is not None:
        return int(lr) == 0
    return True


def unwrap_training_model(model: Any) -> Any:
    m = model
    while m is not None and hasattr(m, "module"):
        m = m.module
    return m


def inject_visual_dims_into_logs(logs: Dict[str, Any], model_ref: Any) -> None:
    """把 dino_dim / clip_dim 写入 logs（来自 last_loss_stats 或模型 config 维数）。"""
    if model_ref is None:
        return
    if hasattr(model_ref, "get_last_loss_stats"):
        try:
            _st = model_ref.get_last_loss_stats() or {}
            if _st.get("dino_patch_dim") is not None:
                logs["dino_dim"] = int(_st["dino_patch_dim"])
            if _st.get("clip_patch_dim") is not None:
                logs["clip_dim"] = int(_st["clip_patch_dim"])
        except Exception:
            pass
    if "dino_dim" not in logs:
        _dh = getattr(model_ref, "dino_hidden", None)
        if _dh is not None:
            logs["dino_dim"] = int(_dh)
    if "clip_dim" not in logs:
        _ch = getattr(model_ref, "clip_hidden", None)
        if _ch is not None:
            logs["clip_dim"] = int(_ch)


def _maybe_compute_grad_norm(model: Any, logs: Dict[str, Any]) -> Optional[float]:
    if model is None or os.environ.get("DISABLE_GRAD_NORM", "false").lower() == "true":
        return None
    try:
        total_norm_sq = 0.0
        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                has_grad = True
                n = p.grad.data.norm(2).item()
                total_norm_sq += n * n
        if has_grad:
            gnorm_val = total_norm_sq ** 0.5
            logs["grad_norm"] = gnorm_val
            return float(gnorm_val)
    except Exception:
        pass
    return None


def format_train_log_line(
    state: TrainerState,
    logs: Dict[str, Any],
    model: Any,
    *,
    compute_gnorm_if_missing: bool = True,
    add_rank_prefix: bool = True,
) -> str:
    """
    与 ``EnhancedLoggingCallback`` 终端行一致的一行字符串（含 dino_dim / clip_dim）。
    供 ``utils.qwen_train`` 等在自定义 ``on_log`` 里 ``tqdm.write`` 时调用。

    ``add_rank_prefix=False``：只返回 ``step=... epoch=... | ...``，用于外层已带
    ``[train rank=...]`` 前缀的 ``_train_log``，避免重复。
    """
    model_ref = unwrap_training_model(model)
    inject_visual_dims_into_logs(logs, model_ref)

    step = state.global_step
    trainer_reported_loss = logs.get("loss")
    total_loss = trainer_reported_loss
    lm_loss = None
    mask_loss = None
    loss_stats: dict = {}

    if model_ref is not None and hasattr(model_ref, "get_last_loss_stats"):
        try:
            loss_stats = model_ref.get_last_loss_stats() or {}
            lm_loss = loss_stats.get("loss_lm")
            mask_loss = loss_stats.get("loss_mask")
            if loss_stats.get("loss_total") is not None:
                total_loss = loss_stats.get("loss_total")
        except Exception:
            loss_stats = {}

    gnorm_val = logs.get("grad_norm")
    if compute_gnorm_if_missing and gnorm_val is None:
        gnorm_val = _maybe_compute_grad_norm(model, logs)
    elif gnorm_val is not None:
        gnorm_val = float(gnorm_val)

    r_env, lr_env = os.environ.get("RANK"), os.environ.get("LOCAL_RANK")
    rank_s = r_env if r_env is not None else "?"
    local_s = lr_env if lr_env is not None else "?"
    try:
        epoch_f = float(state.epoch) if getattr(state, "epoch", None) is not None else 0.0
    except (TypeError, ValueError):
        epoch_f = 0.0
    if add_rank_prefix:
        msg = f"[train rank={rank_s} local={local_s}] step={step} epoch={epoch_f:.2f} |"
    else:
        msg = f"step={step} epoch={epoch_f:.2f} |"
    if total_loss is not None:
        msg += f" loss={float(total_loss):.4f}"
    if lm_loss is not None:
        msg += f" loss_lm={float(lm_loss):.4f}"
    if mask_loss is not None:
        msg += f" loss_mask={float(mask_loss):.4f}"
    if loss_stats.get("bridge_tokens") is not None:
        msg += f" bridge_tok={int(loss_stats['bridge_tokens'])}"
    if loss_stats.get("raw_visual_tokens") is not None:
        msg += f" raw_vis_tok={int(loss_stats['raw_visual_tokens'])}"
    if logs.get("dino_dim") is not None:
        msg += f" dino_dim={int(logs['dino_dim'])}"
    if logs.get("clip_dim") is not None:
        msg += f" clip_dim={int(logs['clip_dim'])}"
    if "learning_rate" in logs:
        msg += f" lr={float(logs['learning_rate']):.6g}"
    if gnorm_val is not None:
        msg += f" gnorm={float(gnorm_val):.2f}"
    return msg


class EnhancedLoggingCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.log_dir = os.path.join(output_dir, "logs")
        self.is_world_process_zero = _is_main_process()
        self.writer = None
        if self.is_world_process_zero:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(
                f"[EnhancedLoggingCallback] utils.qwen_logging loaded from {__file__}",
                file=sys.stderr,
                flush=True,
            )

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, model=None, **kwargs):
        if not self.is_world_process_zero or self.writer is None or not logs:
            return

        model_ref = unwrap_training_model(model)
        inject_visual_dims_into_logs(logs, model_ref)

        step = state.global_step
        trainer_reported_loss = logs.get("loss")
        total_loss = trainer_reported_loss
        lm_loss = None
        mask_loss = None
        loss_stats: dict = {}

        if hasattr(model_ref, "get_last_loss_stats"):
            try:
                loss_stats = model_ref.get_last_loss_stats() or {}
                lm_loss = loss_stats.get("loss_lm")
                mask_loss = loss_stats.get("loss_mask")
                if loss_stats.get("loss_total") is not None:
                    total_loss = loss_stats.get("loss_total")
            except Exception:
                loss_stats = {}

        if total_loss is not None:
            self.writer.add_scalar("train/loss_total", total_loss, step)
        if lm_loss is not None:
            self.writer.add_scalar("train/loss_lm", lm_loss, step)
        if mask_loss is not None:
            self.writer.add_scalar("train/loss_mask", mask_loss, step)
        if "learning_rate" in logs:
            self.writer.add_scalar("train/learning_rate", logs["learning_rate"], step)

        gnorm_val = logs.get("grad_norm")
        if gnorm_val is None and model is not None and os.environ.get("DISABLE_GRAD_NORM", "false").lower() != "true":
            try:
                total_norm_sq = 0.0
                has_grad = False
                for p in model.parameters():
                    if p.grad is not None:
                        has_grad = True
                        n = p.grad.data.norm(2).item()
                        total_norm_sq += n * n
                if has_grad:
                    gnorm_val = total_norm_sq ** 0.5
                    logs["grad_norm"] = gnorm_val
                    self.writer.add_scalar("train/grad_norm", gnorm_val, step)
            except Exception:
                pass
        elif gnorm_val is not None:
            self.writer.add_scalar("train/grad_norm", float(gnorm_val), step)

        msg = format_train_log_line(state, logs, model, compute_gnorm_if_missing=False)

        print(file=sys.stderr, flush=True)
        print(msg, file=sys.stderr, flush=True)

        self.writer.flush()
        if step % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.close()
