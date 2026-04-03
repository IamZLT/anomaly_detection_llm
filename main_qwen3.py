#!/usr/bin/env python3
"""Qwen3-VL MVTec 入口脚本（稳定版，配置驱动）。"""

import argparse

from utils.qwen_config import apply_runtime_overrides, load_yaml_config
from utils.qwen_infer import inference_main
from utils.qwen_train import train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-VL MVTec Grounding")
    parser.add_argument("--config", type=str, default="configs/qwen.yaml", help="YAML配置文件路径")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], help="运行模式，覆盖yaml")
    parser.add_argument("--output_dir", type=str, help="输出目录，覆盖yaml")
    parser.add_argument("--run_name", type=str, help="任务名，覆盖yaml")
    parser.add_argument("--model_path", type=str, help="推理模型路径，覆盖yaml")
    parser.add_argument("--image_path", type=str, help="推理图像路径，覆盖yaml")
    parser.add_argument("--prompt", type=str, help="推理提示词，覆盖yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_runtime_overrides(cfg, args)
    mode = cfg.get("runtime", {}).get("mode", "train")

    if mode == "train":
        train_main(cfg)
    elif mode == "inference":
        inference_main(cfg)
    else:
        raise ValueError(f"不支持的模式: {mode}")


if __name__ == "__main__":
    main()

