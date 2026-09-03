#!/usr/bin/env python3
"""Train prior-guided process-aware spatial GRPO (VisA → MVTec)."""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from utils.config import apply_runtime_overrides, load_yaml_config


def _in_distributed_worker() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


def _maybe_relaunch_multi_gpu_train(cfg: dict) -> None:
    num_gpu = int(cfg.get("distributed", {}).get("num_gpu", 1))
    if num_gpu <= 1 or _in_distributed_worker():
        return
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(
        f"[train] relaunch check: distributed.num_gpu={num_gpu} cuda_count={cuda_count} CUDA_VISIBLE_DEVICES={cvd}",
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
    print(f"[train] distributed.num_gpu={num_gpu}，正在启动: {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.call(cmd))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Prior-guided Spatial GRPO")
    p.add_argument("--config", type=str, default="configs/qwen35_2b_grpo.yaml")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--num-gpu", type=int, default=None, help="覆盖 distributed.num_gpu")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_runtime_overrides(cfg, args)
    cfg.setdefault("runtime", {})["mode"] = "train"
    print(
        "[train] env: "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"RANK={os.environ.get('RANK')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
        flush=True,
    )
    _maybe_relaunch_multi_gpu_train(cfg)
    from rl.trainer import train_main

    train_main(cfg)


if __name__ == "__main__":
    main()
