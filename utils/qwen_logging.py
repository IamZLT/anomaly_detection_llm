import os

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback, TrainerControl, TrainerState


class EnhancedLoggingCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.log_dir = os.path.join(output_dir, "logs")
        rank = int(os.environ.get("RANK", "0"))
        self.is_world_process_zero = rank == 0
        self.writer = None
        if self.is_world_process_zero:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, model=None, **kwargs):
        if not self.is_world_process_zero or self.writer is None or not logs:
            return

        model_ref = model
        while hasattr(model_ref, "module"):
            model_ref = model_ref.module

        step = state.global_step
        total_loss = logs.get("loss")
        lm_loss = None
        mask_loss = None

        if hasattr(model_ref, "get_last_loss_stats"):
            try:
                loss_stats = model_ref.get_last_loss_stats() or {}
                lm_loss = loss_stats.get("loss_lm")
                mask_loss = loss_stats.get("loss_mask")
                if loss_stats.get("loss_total") is not None:
                    total_loss = loss_stats.get("loss_total")
            except Exception:
                pass

        if total_loss is not None:
            self.writer.add_scalar("train/loss_total", total_loss, step)
        if lm_loss is not None:
            self.writer.add_scalar("train/loss_lm", lm_loss, step)
        if mask_loss is not None:
            self.writer.add_scalar("train/loss_mask", mask_loss, step)
        if "learning_rate" in logs:
            self.writer.add_scalar("train/learning_rate", logs["learning_rate"], step)

        if model is not None and os.environ.get("DISABLE_GRAD_NORM", "false").lower() != "true":
            try:
                total_norm_sq = 0.0
                has_grad = False
                for p in model.parameters():
                    if p.grad is not None:
                        has_grad = True
                        n = p.grad.data.norm(2).item()
                        total_norm_sq += n * n
                if has_grad:
                    self.writer.add_scalar("train/grad_norm", total_norm_sq ** 0.5, step)
            except Exception:
                pass

        world_size = int(getattr(args, "world_size", 1) or 1)
        per_step_samples = int(args.per_device_train_batch_size) * int(args.gradient_accumulation_steps) * world_size
        seen_samples = step * per_step_samples
        max_steps = max(int(getattr(state, "max_steps", 0) or 0), 1)
        progress = min(100.0, (step / max_steps) * 100.0)
        msg = (
            f"[train] step {step}/{max_steps} ({progress:.1f}%) "
            f"samples~{seen_samples} "
            f"loss_total={float(total_loss):.4f}" if total_loss is not None else
            f"[train] step {step}/{max_steps} ({progress:.1f}%) samples~{seen_samples}"
        )
        if lm_loss is not None:
            msg += f" loss_lm={float(lm_loss):.4f}"
        if mask_loss is not None:
            msg += f" loss_mask={float(mask_loss):.4f}"
        if "learning_rate" in logs:
            msg += f" lr={float(logs['learning_rate']):.6g}"
        if "grad_norm" in logs:
            msg += f" grad_norm={float(logs['grad_norm']):.4f}"
        print(msg)

        self.writer.flush()
        if step % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.close()

