#!/usr/bin/env python3
"""
Stage-3：在 Stage-2 基础上可选只做「含 GT bbox」的 grounding 微调。

相对 Step-2：
- `data.train_gt_bbox_only=true` 时训练集只保留 metadata 带 bbox 的样本。
- `training.bbox_aux_loss_weight`：>0 时在 LM loss 外叠加视觉前缀池化后的 bbox 回归 Smooth L1（仅训练；推理仍只 decode LLM 文本里的 `<bbox>`）。
  设为 0 则与 Step-2 一样只靠 LM 学 `<bbox>` 标签。
"""

import argparse
import os
import subprocess
import sys

import torch

from utils.qwen_config import apply_runtime_overrides, load_yaml_config
from utils.qwen_train import train_main


def _in_distributed_worker() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


def _maybe_relaunch_multi_gpu_train(cfg: dict) -> None:
    num_gpu = int(cfg.get("distributed", {}).get("num_gpu", 1))
    mode = cfg.get("runtime", {}).get("mode", "train")
    if mode != "train":
        return
    if num_gpu <= 1:
        return
    if _in_distributed_worker():
        return
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(
        f"[stage3] relaunch check: distributed.num_gpu={num_gpu} cuda_count={cuda_count} CUDA_VISIBLE_DEVICES={cvd}",
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
    print(f"[stage3] distributed.num_gpu={num_gpu}，正在启动: {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.call(cmd))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("AD-LLM Stage-3（可选 bbox-only 数据 + 可选 bbox 辅助损失）")
    p.add_argument("--config", type=str, default="configs/ad_llm_step3.yaml")
    p.add_argument("--mode", type=str, choices=["train", "inference"], default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--num-gpu", type=int, default=None, help="覆盖 distributed.num_gpu（训练）")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_runtime_overrides(cfg, args)

    print(
        "[stage3] env: "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"RANK={os.environ.get('RANK')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
        flush=True,
    )

    _maybe_relaunch_multi_gpu_train(cfg)

    # Step2/3 桥接 ckpt → model.bridge_ckpt_path；支持 step3、model、bridge 三段
    model_m = cfg.get("model", {}) or {}
    step3 = cfg.get("step3", {}) or {}
    bridge_sec = cfg.get("bridge", {}) or {}
    bridge_ckpt = (
        model_m.get("bridge_ckpt_path")
        or step3.get("bridge_ckpt_path")
        or bridge_sec.get("bridge_ckpt_path")
    )
    if bridge_ckpt:
        cfg.setdefault("model", {})
        cfg["model"]["bridge_ckpt_path"] = bridge_ckpt

    train_main(cfg)


if __name__ == "__main__":
    main()
