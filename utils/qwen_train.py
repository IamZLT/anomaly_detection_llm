import os
import json
import random
from contextlib import nullcontext
import shutil
import socket
import subprocess
import time

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from transformers.trainer_utils import has_length

try:
    from transformers.utils.notebook import NotebookProgressCallback
except ImportError:
    NotebookProgressCallback = None  # type: ignore[misc,assignment]
from PIL import Image
from transformers import AutoImageProcessor

from data.mvtec_json_loader import MVTecDataManager
from data.mvtec_grounding import MVTecQwenGroundingDataset
from models.avNet import (
    QwenDinoBridgeModel,
    setup_model_and_processor,
)
from utils.visualization import (
    attach_avnet_to_step1_shell,
    build_step1_shell,
    compute_step1_heat_up,
    heatmap_overlay,
    prepare_step1_image_tensor,
    restore_avnet_bridge_from_step1_shell,
    visual_prototypes_from_avnet,
)
from utils.qwen_common import prepare_output_dir, set_seed
from utils.qwen_infer import decode_generation_output


def _build_generation_inputs_for_eval(
    cfg: dict,
    processor,
    image: Image.Image,
    prompt: str,
) -> dict:
    """
    Minimal copy of utils.qwen_infer.build_generation_inputs to avoid circular imports during training.
    """
    use_dino_bridge = bool(cfg.get("dino", {}).get("enabled", True))
    local_files_only = cfg.get("model", {}).get("local_files_only", True)
    if use_dino_bridge:
        dino_processor = AutoImageProcessor.from_pretrained(
            cfg["dino"]["model_path"],
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        clip_processor = AutoImageProcessor.from_pretrained(
            cfg["clip"]["model_path"],
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        dino_inputs = dino_processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={
                "height": int(cfg.get("dino", {}).get("image_size", 512)),
                "width": int(cfg.get("dino", {}).get("image_size", 512)),
            },
        )
        inputs["dino_pixel_values"] = dino_inputs["pixel_values"]
        clip_inputs = clip_processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={
                "height": int(cfg.get("clip", {}).get("image_size", 224)),
                "width": int(cfg.get("clip", {}).get("image_size", 224)),
            },
        )
        inputs["clip_pixel_values"] = clip_inputs["pixel_values"]
        return inputs

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt", padding=True)


class BridgeAndProcessorCheckpointCallback(TrainerCallback):
    """
    Trainer 保存 checkpoint-* 时只写入 unwrap 后的 HF 基座权重，不会调用 QwenDinoBridgeModel.save_pretrained，
    因此缺 dino_bridge.bin；也不会自动 save processor。在 on_save 里补全，使中间 checkpoint 与 final_model 一样可推理。
    """

    def __init__(self, cfg: dict, processor, manager: MVTecDataManager):
        self.cfg = cfg
        self.processor = processor
        self.manager = manager
        self._last_eval_step: int | None = None
        self._seen_batches: int = 0
        # 与 test_ad_llm_step1 共用 DinoClipVisualPrototypeModel.forward；壳子懒加载后挂 avNet 子模块
        self._step1_vis_shell = None
        self._warned_eval_heatmap_no_proto = False

    def on_save(self, args, state, control, model=None, **kwargs):
        if not _is_main_process() or model is None:
            return control
        m = model.module if hasattr(model, "module") else model
        if not isinstance(m, QwenDinoBridgeModel):
            return control
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt):
            _train_log(f"[checkpoint] 目录不存在，跳过补全: {ckpt}")
            return control
        try:
            tr = self.cfg.get("training", {}) or {}

            # 1) 保存 checkpoint 为可直接推理目录（可配置开关）
            if bool(tr.get("save_pretrained_on_save", True)):
                m.save_pretrained(ckpt)
                self.processor.save_pretrained(ckpt)
                _train_log(f"[checkpoint] saved pretrained + dino_bridge.bin + processor → {ckpt}")

            # 2) 保存后随机可视化/推理若干样本（可配置开关与数量）
            if bool(tr.get("eval_on_save", True)):
                # on_save 时也允许跑一次（主要用于与 checkpoint 绑定的可视化产物）
                step = int(getattr(state, "global_step", 0) or 0)
                if self._last_eval_step != step:
                    self._last_eval_step = step
                    self._run_eval_after_save(m, ckpt, global_step=step)
        except Exception as e:
            _train_log(f"[checkpoint] 补全保存失败: {e}")
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        真正按“batch step”频率跑 eval（不依赖 save_steps）。
        注意：Trainer 的 global_step 在梯度累积时按 optimizer update 计数，不等于 batch 数；
        这里用 _seen_batches 统计 batch 次数，满足用户“每 N 个 batch 测一次”的需求。
        """
        if not _is_main_process() or model is None:
            return control
        tr = self.cfg.get("training", {}) or {}
        if not bool(tr.get("eval_on_save", True)):
            return control
        every = int(tr.get("eval_every_n_steps", 0))
        if every <= 0:
            return control

        self._seen_batches += 1
        if self._seen_batches % every != 0:
            return control

        m = model.module if hasattr(model, "module") else model
        if not isinstance(m, QwenDinoBridgeModel):
            return control

        step = int(getattr(state, "global_step", 0) or 0)
        out_dir = os.path.join(args.output_dir, "eval_steps", f"batch_{self._seen_batches:08d}_step_{step:08d}")
        os.makedirs(out_dir, exist_ok=True)
        self._run_eval_after_save(m, out_dir, global_step=step)
        return control

    @torch.no_grad()
    def _run_eval_after_save(self, model: QwenDinoBridgeModel, ckpt_dir: str, global_step: int) -> None:
        tr = self.cfg.get("training", {}) or {}
        prompt = str(tr.get("eval_prompt", "Does this image have any anomalies?"))
        k = int(tr.get("eval_num_samples", 5))
        if k <= 0:
            return
        out_dir = os.path.join(ckpt_dir, "eval_samples")
        os.makedirs(out_dir, exist_ok=True)

        # sample pool from manager json（整份 JSON，随机抽）
        pool = self.manager.get_all_grounding_samples(mode="test", anomaly_only=False)

        seed = int(self.cfg.get("training", {}).get("seed", 42))
        rng = random.Random(seed + int(global_step))
        picks = [pool[i] for i in rng.sample(range(len(pool)), k=min(k, len(pool)))]

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        if (
            _is_main_process()
            and not model.visual_prototypes_ready()
            and not self._warned_eval_heatmap_no_proto
        ):
            self._warned_eval_heatmap_no_proto = True
            _train_log(
                "[eval][heatmap] AVNet 未加载 Step1 visual prototypes，热力图为 cosdiff，"
                "与 test_ad_llm_step1 使用含 prototypes 的 ckpt 时不一致。"
                "请配置 bridge.bridge_ckpt_path 或 model.bridge_ckpt_path 为 Step1 的 epoch/best.pth（顶层含 prototypes），"
                "或另设 model/bridge.step1_prototypes_ckpt_path 指向该文件。",
                main_only=True,
            )

        results = []
        for j, s in enumerate(picks):
            img_path = s.get("full_img_path") or s.get("image")
            if not img_path or (not os.path.isfile(str(img_path))):
                continue

            try:
                img = Image.open(str(img_path)).convert("RGB")
            except Exception:
                continue

            # match inference preprocessing: smart_resize inside build_generation_inputs expects resized image
            from utils.qwen_common import smart_resize

            img_rs, orig_size, _ = smart_resize(
                img.copy(),
                max_size=int(self.cfg["data"]["max_image_size"]),
                factor=int(self.cfg["data"]["factor"]),
            )

            inputs = _build_generation_inputs_for_eval(self.cfg, self.processor, img_rs, prompt)
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            # feature visualization (overlay heatmap on image)
            dino_pv = inputs.get("dino_pixel_values")
            clip_pv = inputs.get("clip_pixel_values")
            dino_hm_path = None
            mapped_hm_path = None
            clip_hm_path = None
            mapped_vis_mode = None
            if dino_pv is not None and clip_pv is not None:
                # 与 test_ad_llm_step1 同一套 forward；attach 会暂时把子模块挂到 shell，必须在 generate 前 restore
                if self._step1_vis_shell is None:
                    self._step1_vis_shell = build_step1_shell(self.cfg)
                    self._step1_vis_shell.eval()
                _attached = False
                try:
                    attach_avnet_to_step1_shell(model, self._step1_vis_shell)
                    _attached = True
                    protos = visual_prototypes_from_avnet(model)
                    amp_ctx = (
                        torch.amp.autocast("cuda", enabled=False)
                        if device.type == "cuda"
                        else nullcontext()
                    )
                    with torch.no_grad():
                        with amp_ctx:
                            img_t = prepare_step1_image_tensor(
                                img_rs, int(self.cfg["dino"]["image_size"]), device
                            )
                            heat_up, mapped_vis_mode = compute_step1_heat_up(
                                model=self._step1_vis_shell,
                                img_t=img_t,
                                img_rs=img_rs,
                                cfg=self.cfg,
                                prototypes=protos,
                                device=device,
                            )
                    alpha = float(tr.get("eval_heatmap_alpha", 0.45))
                    vis = heatmap_overlay(img_rs, heat_up, alpha=alpha)
                    fname = (
                        "mapped_proto_abn_overlay"
                        if mapped_vis_mode == "prototype"
                        else "mapped_clip_cosdiff_overlay"
                    )
                    mapped_hm_path = os.path.join(out_dir, f"{global_step:08d}_{j:02d}_{fname}.png")
                    vis.save(mapped_hm_path)
                    dino_hm_path = None
                    clip_hm_path = None
                finally:
                    if _attached:
                        restore_avnet_bridge_from_step1_shell(model, self._step1_vis_shell)

            outputs = model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.get("inference", {}).get("max_new_tokens", 64)),
                temperature=float(self.cfg.get("inference", {}).get("temperature", 0.0)),
                top_p=float(self.cfg.get("inference", {}).get("top_p", 0.9)),
                do_sample=bool(self.cfg.get("inference", {}).get("do_sample", False)),
            )
            answer = decode_generation_output(self.processor, outputs, inputs, self.cfg)

            rec = {
                "global_step": int(global_step),
                "idx": int(j),
                "prompt": prompt,
                "image_path": str(img_path),
                "orig_size": list(orig_size),
                "answer": str(answer),
                "dino_feature_vis": dino_hm_path,
                "mapped_feature_vis": mapped_hm_path,
                "mapped_vis_mode": mapped_vis_mode,
                "clip_feature_vis": clip_hm_path,
                "heatmap_backend": "utils.visualization" if mapped_vis_mode is not None else None,
            }
            results.append(rec)

        with open(os.path.join(out_dir, f"eval_{global_step:08d}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if was_training:
            model.train()


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
        # bbox 辅助头：w=0 时不建 head、不算 loss_bbox；日志里仍打出状态避免误以为漏打
        bbox_w = float(getattr(m, "bbox_aux_loss_weight", 0.0) or 0.0)
        if extra.get("loss_bbox") is not None:
            parts.append(f"loss_bbox={float(extra['loss_bbox']):.4f}")
        elif bbox_w > 0.0 and getattr(m, "bbox_head", None) is not None:
            parts.append("loss_bbox=n/a")
        else:
            parts.append("loss_bbox=off")
        # helpful bridge stats
        if extra.get("bridge_tokens") is not None:
            parts.append(f"bridge_tok={int(extra['bridge_tokens'])}")
        if extra.get("raw_visual_tokens") is not None:
            parts.append(f"raw_vis_tok={int(extra['raw_visual_tokens'])}")
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


def _auto_start_tensorboard(cfg: dict, output_dir: str) -> None:
    tb_cfg = cfg.get("tensorboard", {})
    if not bool(tb_cfg.get("auto_start", True)):
        return

    host = str(tb_cfg.get("host", "0.0.0.0"))
    port = int(tb_cfg.get("port", 5000))
    probe_host = "127.0.0.1" if host == "0.0.0.0" else host
    if _is_port_in_use(probe_host, port):
        print(f"[TensorBoard] 端口 {port} 已被占用，复用现有服务: http://127.0.0.1:{port}")
        return

    tb_bin = shutil.which("tensorboard")
    if tb_bin is None:
        print("[TensorBoard] 未找到 tensorboard 命令，跳过自动启动。")
        return

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    cmd = [tb_bin, "--logdir", log_dir, "--host", host, "--port", str(port)]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"[TensorBoard] 已自动启动 (pid={proc.pid}): http://127.0.0.1:{port}")
    except Exception as e:
        print(f"[TensorBoard] 自动启动失败: {e}")


def train_main(cfg: dict) -> None:
    is_world_process_zero = _is_main_process()
    _train_log("进入 train_main …")
    set_seed(cfg["training"]["seed"])

    # 避免 DataLoader 多进程通过 /dev/shm 传 tensor 时写爆（常见于容器 shm 很小）
    # 可在 yaml 里设置 training.multiprocessing_sharing_strategy: file_system
    mp_strategy = str(cfg.get("training", {}).get("multiprocessing_sharing_strategy", "")).strip().lower()

    output_dir = prepare_output_dir(
        base_dir=cfg["paths"]["output_dir"],
        run_name=cfg["runtime"]["run_name"],
        auto_create=cfg["runtime"]["auto_create_output_dir"],
    )
    cfg["paths"]["output_dir"] = output_dir
    if is_world_process_zero:
        _train_log(f"输出目录: {output_dir}", main_only=True)

    manager = MVTecDataManager(
        dataset_root=cfg["paths"]["dataset_root"],
        conversation_json_path=cfg["paths"]["conversation_json_path"],
    )
    _train_log("正在加载 MVTec 元数据 …")
    manager.load_all()
    _train_log("MVTec 元数据加载完成")

    _train_log("正在 setup_model_and_processor（全量 Qwen + DINO/CLIP 桥接；DINO/CLIP 骨干冻结，可能数分钟无输出）…")
    model, processor = setup_model_and_processor(cfg, for_inference=False)
    _train_log("模型与 processor 构建完成")
    if isinstance(model, QwenDinoBridgeModel):
        if model.visual_prototypes_ready():
            g = float(getattr(model, "proto_modulation_gamma", 0.0) or 0.0)
            _train_log(
                "bridge：Step1 visual prototypes 已加载；eval 热力图走 "
                "DinoClipVisualPrototypeModel.forward + compute_step1_heat_up（与 test_ad_llm_step1 相同，"
                "仍为未调制的 mapped 打分）。"
                f"AVNet._build_visual_tokens：若 bridge.proto_modulation_gamma>0，进 LLM 前会对 mapped 做 proto 幅度调制（当前 gamma={g}）。"
                "对照排查可设 training.step1_visual_debug: true。"
            )
    train_dataset = MVTecQwenGroundingDataset(
        manager=manager,
        processor=processor,
        mode="train",
        max_length=cfg["training"]["max_length"],
        max_image_size=cfg["data"]["max_image_size"],
        factor=cfg["data"]["factor"],
        use_grounding_format=cfg["data"]["use_grounding_format"],
        dino_cfg=cfg.get("dino", {}),
        clip_cfg=cfg.get("clip", {}),
        local_files_only=cfg.get("model", {}).get("local_files_only", True),
        train_anomaly_only=bool(cfg["data"].get("train_anomaly_only", False)),
        normalize_bbox_01=bool(cfg["data"].get("bbox_normalize_01", False)),
        train_gt_bbox_only=bool(cfg["data"].get("train_gt_bbox_only", False)),
    )
    _train_log(f"训练集样本数 len(dataset)={len(train_dataset)}")
    if bool(cfg["data"].get("train_gt_bbox_only", False)):
        _train_log("data.train_gt_bbox_only=true：仅保留 metadata 含 bbox 的 grounding 样本")
    bw = float(cfg.get("training", {}).get("bbox_aux_loss_weight", 0.0))
    if bw > 0.0:
        _train_log(f"training.bbox_aux_loss_weight={bw}：除 LM 外叠加视觉前缀 bbox 回归 Smooth L1（推理仍走 LLM 文本）")
    else:
        _train_log("training.bbox_aux_loss_weight=0：无 bbox 辅助头，终端每步会显示 loss_bbox=off；需要数值请设 >0")

    # Save a small snapshot of random training samples + resized images for sanity check (configurable)
    if is_world_process_zero:
        tr = cfg.get("training", {}) or {}
        k = int(tr.get("data_snapshot_num_samples", 5))
        if k <= 0:
            k = 0
        _dump_dir = os.path.join(output_dir, "data_samples")
        os.makedirs(_dump_dir, exist_ok=True)
        seed = int(cfg.get("training", {}).get("seed", 42))
        rng = random.Random(seed)
        k = min(k, len(train_dataset.samples))
        picks = rng.sample(range(len(train_dataset.samples)), k=k) if k > 0 else []
        dumped = []
        for j, idx in enumerate(picks):
            s = train_dataset.samples[idx]
            img_path = s.get("full_img_path") or s.get("image")
            meta = s.get("metadata", {}) or {}
            conv = s.get("conversations", None)
            rec = {"idx": int(idx), "image_path": str(img_path), "metadata": meta, "conversations": conv}
            if img_path and os.path.isfile(str(img_path)):
                try:
                    img = Image.open(str(img_path)).convert("RGB")
                    from utils.qwen_common import smart_resize

                    img_rs, orig_size, _ = smart_resize(
                        img.copy(),
                        max_size=int(cfg["data"]["max_image_size"]),
                        factor=int(cfg["data"]["factor"]),
                    )
                    out_img = os.path.join(_dump_dir, f"sample_{j:02d}.png")
                    img_rs.save(out_img)
                    rec["saved_image"] = out_img
                    rec["orig_size"] = list(orig_size)
                    rec["resized_size"] = [img_rs.size[0], img_rs.size[1]]
                except Exception as e:
                    rec["image_error"] = str(e)
            dumped.append(rec)

        with open(os.path.join(_dump_dir, "samples.json"), "w", encoding="utf-8") as f:
            json.dump(dumped, f, ensure_ascii=False, indent=2)
        _train_log(f"[data] saved {len(dumped)} samples snapshot → {_dump_dir}", main_only=True)

    # torchrun 会设置 LOCAL_RANK；yaml 里 -1 时需从环境读取，否则多卡不会进 DDP
    _local_rank = int(cfg["distributed"]["local_rank"])
    if _local_rank < 0:
        _local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    tr = cfg["training"]
    num_workers = int(tr.get("num_workers", 0))
    ta_common = dict(
        output_dir=output_dir,
        num_train_epochs=tr["num_epochs"],
        per_device_train_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        learning_rate=tr["learning_rate"],
        weight_decay=tr["weight_decay"],
        warmup_ratio=tr["warmup_ratio"],
        logging_steps=tr["logging_steps"],
        logging_first_step=True,
        save_steps=tr["save_steps"],
        save_strategy="steps",
        save_total_limit=tr["save_total_limit"],
        save_safetensors=False,
        fp16=tr["fp16"],
        bf16=tr["bf16"],
        gradient_checkpointing=tr["gradient_checkpointing"],
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=False,
        report_to=[],
        remove_unused_columns=False,
        logging_dir=os.path.join(output_dir, "logs"),
        disable_tqdm=False,
        ddp_find_unused_parameters=cfg["distributed"]["ddp_find_unused_parameters"],
        local_rank=_local_rank,
        deepspeed=cfg["distributed"]["deepspeed"],
    )
    # transformers: prefetch_factor 只能在 num_workers>0 时设置
    if num_workers > 0:
        ta_common["dataloader_prefetch_factor"] = 2
    gckw = tr.get("gradient_checkpointing_kwargs")
    if gckw:
        ta_common["gradient_checkpointing_kwargs"] = gckw
    training_args = TrainingArguments(**ta_common)

    bridge_ckpt_cb = BridgeAndProcessorCheckpointCallback(cfg=cfg, processor=processor, manager=manager)
    pretty_log_cb = PrettyTrainLogCallback()
    per_epoch_pbar_cb = PerEpochProgressCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[bridge_ckpt_cb, pretty_log_cb, per_epoch_pbar_cb],
    )
    # 去掉默认进度条（终端 ProgressCallback / Notebook 下 NotebookProgressCallback），改用 PerEpochProgressCallback
    _skip_progress = (PrinterCallback, ProgressCallback)
    if NotebookProgressCallback is not None:
        _skip_progress = _skip_progress + (NotebookProgressCallback,)
    trainer.callback_handler.callbacks = [cb for cb in trainer.callback_handler.callbacks if not isinstance(cb, _skip_progress)]

    if is_world_process_zero:
        _auto_start_tensorboard(cfg, output_dir)
        print(f"开始训练，输出目录: {output_dir}")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _train_log("开始 trainer.train（首轮 forward / DataLoader 启动可能较慢）…")
    trainer.train(resume_from_checkpoint=cfg["training"]["resume_from_checkpoint"])

    final_model_path = os.path.join(output_dir, "final_model")
    if is_world_process_zero:
        model.save_pretrained(final_model_path)
        processor.save_pretrained(final_model_path)

    mins = (time.time() - t0) / 60
    if is_world_process_zero:
        print(f"训练完成，耗时: {mins:.2f} 分钟")
        print(f"模型已保存: {final_model_path}")

