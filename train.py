#!/usr/bin/env python3
"""
工业异常定位训练 / 推理入口。

默认配置 configs/ad_llm_qwen35_2b_prior.yaml：
  冻结 Qwen3.5 Vision Encoder 多层差异先验 H
  + 参考图 / 测试图 / H → Grounded Comparative CoT
  + LoRA Process-Aware Spatial GRPO（无 SFT）。

- runtime.pipeline=prior_cot → utils.train_prior.train_prior_main
- inference → utils.infer.inference_main
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from utils.config import apply_runtime_overrides, load_yaml_config
from utils.infer import inference_main


def _in_distributed_worker() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


def _maybe_relaunch_multi_gpu_train(cfg: dict) -> None:
    """
    避免 HF Trainer 在单进程多 GPU 下走 DataParallel。
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
    p = argparse.ArgumentParser("AD-LLM prior-guided CoT (Qwen3.5)")
    p.add_argument("--config", type=str, default="configs/ad_llm_qwen35_2b_prior.yaml")
    p.add_argument("--mode", type=str, choices=["train", "inference"], default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--num-gpu", type=int, default=None, help="覆盖 distributed.num_gpu（训练）")
    p.add_argument("--model_path", type=str, default=None, help="推理：覆盖 inference.model_path")
    p.add_argument("--image_path", type=str, default=None, help="推理：覆盖 inference.image_path")
    p.add_argument("--prompt", type=str, default=None, help="推理：覆盖 inference.prompt")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_runtime_overrides(cfg, args)

    mode = cfg.get("runtime", {}).get("mode", "train")
    print(
        "[train] env: "
        f"mode={mode} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"RANK={os.environ.get('RANK')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
        flush=True,
    )

    if mode == "inference":
        inference_main(cfg)
        return

    _maybe_relaunch_multi_gpu_train(cfg)
    pipeline = str(cfg.get("runtime", {}).get("pipeline", "prior_cot")).strip().lower()
    if pipeline not in ("prior_cot", "anomaly_prior"):
        raise ValueError(
            f"未知 pipeline={pipeline!r}。旧 DINO/fusion 训练已移除，请用 "
            "runtime.pipeline=prior_cot 与 configs/ad_llm_qwen35_2b_prior.yaml"
        )
    from utils.train_prior import train_prior_main

    train_prior_main(cfg)


if __name__ == "__main__":
    main()
