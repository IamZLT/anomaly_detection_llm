#!/usr/bin/env python3
"""Qwen3-VL MVTec 入口脚本（稳定版，配置驱动）。"""

import argparse
import os
import subprocess
import sys

from utils.config import apply_runtime_overrides, load_yaml_config
from utils.infer import inference_main
from utils.train import train_main


def _in_distributed_worker() -> bool:
    """已由 torchrun / torch.distributed.run 拉起时，环境里有 LOCAL_RANK。"""
    return os.environ.get("LOCAL_RANK") is not None


def _maybe_relaunch_multi_gpu_train(cfg: dict) -> None:
    """distributed.num_gpu > 1 时，用 torch.distributed.run 重启本脚本（父进程退出）。"""
    if cfg.get("runtime", {}).get("mode", "train") != "train":
        return
    num_gpu = int(cfg.get("distributed", {}).get("num_gpu", 1))
    if num_gpu <= 1:
        return
    if _in_distributed_worker():
        return
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
    parser = argparse.ArgumentParser(description="Qwen3-VL MVTec Grounding")
    parser.add_argument("--config", type=str, default="configs/qwen.yaml", help="YAML配置文件路径")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], help="运行模式，覆盖yaml")
    parser.add_argument("--output_dir", type=str, help="输出目录，覆盖yaml")
    parser.add_argument("--run_name", type=str, help="任务名，覆盖yaml")
    parser.add_argument("--model_path", type=str, help="推理模型路径，覆盖yaml")
    parser.add_argument("--image_path", type=str, help="推理图像路径，覆盖yaml")
    parser.add_argument("--prompt", type=str, help="推理提示词，覆盖yaml")
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=None,
        help="覆盖 yaml 中 distributed.num_gpu（仅训练；设为 1 可强制单进程）",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_runtime_overrides(cfg, args)
    mode = cfg.get("runtime", {}).get("mode", "train")

    if mode == "train":
        _maybe_relaunch_multi_gpu_train(cfg)
        train_main(cfg)
    elif mode == "inference":
        inference_main(cfg)
    else:
        raise ValueError(f"不支持的模式: {mode}")


if __name__ == "__main__":
    main()

