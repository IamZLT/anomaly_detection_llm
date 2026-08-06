#!/usr/bin/env python3
"""
Stage-2: 全量训练 Qwen3-VL（图像 + grounding 对话），并加载 Stage-1 的 dino_bridge.bin 作为初始化映射。

约束（按你的要求）：
- 训练 loss 只保留语言模型 cross-entropy（LM loss）
- 删除/不使用 DINO↔CLIP align_loss 与 mask_loss（已在 QwenDinoBridgeModel 中移除）
"""

import argparse
import os
import subprocess
import sys

import torch

from utils.config import apply_runtime_overrides, load_yaml_config
from utils.train import train_main


def _in_distributed_worker() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


def _maybe_relaunch_multi_gpu_train(cfg: dict) -> None:
    """
    避免 HF Trainer 在单进程多 GPU 下走 DataParallel（容易触发 NCCL broadcast 错误）。
    当 distributed.num_gpu>1 且当前不是 torchrun worker 时，自动用 torch.distributed.run 重启。
    """
    num_gpu = int(cfg.get("distributed", {}).get("num_gpu", 1))
    mode = cfg.get("runtime", {}).get("mode", "train")
    if mode != "train":
        return
    if num_gpu <= 1:
        return
    if _in_distributed_worker():
        return
    # best-effort diagnostics
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(
        f"[stage2] relaunch check: distributed.num_gpu={num_gpu} cuda_count={cuda_count} CUDA_VISIBLE_DEVICES={cvd}",
        flush=True,
    )
    script = os.path.abspath(sys.argv[0])
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpu}",
        script,
        *sys.argv[1:],
    ]
    print(f"[stage2] distributed.num_gpu={num_gpu}，正在启动: {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.call(cmd))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("AD-LLM Stage-2 (Qwen full training)")
    p.add_argument("--config", type=str, default="configs/ad_llm_step2.yaml")
    p.add_argument("--mode", type=str, choices=["train", "inference"], default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--num-gpu", type=int, default=None, help="覆盖 distributed.num_gpu（训练）")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_runtime_overrides(cfg, args)

    # banner (useful when users still hit DataParallel)
    print(
        "[stage2] env: "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"RANK={os.environ.get('RANK')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
        flush=True,
    )

    _maybe_relaunch_multi_gpu_train(cfg)

    # 将 Step1 桥接 ckpt 映射到 model.bridge_ckpt_path（setup_model_and_processor 只认此处）
    # 支持：model.bridge_ckpt_path、step2.bridge_ckpt_path、bridge.bridge_ckpt_path（与 YAML 中 bridge: 段一致）
    model_m = cfg.get("model", {}) or {}
    step2 = cfg.get("step2", {}) or {}
    bridge_sec = cfg.get("bridge", {}) or {}
    bridge_ckpt = (
        model_m.get("bridge_ckpt_path")
        or step2.get("bridge_ckpt_path")
        or bridge_sec.get("bridge_ckpt_path")
    )
    if bridge_ckpt:
        cfg.setdefault("model", {})
        cfg["model"]["bridge_ckpt_path"] = bridge_ckpt

    train_main(cfg)


if __name__ == "__main__":
    main()
