#!/usr/bin/env python3
"""
单图测试纯 Qwen-VL（从 YAML 读取参数）。

用法:
  python test.py --config configs/ad_llm_qwen35_9b_zeroshot.yaml \\
      --image_path /path/to/image.png
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import apply_runtime_overrides, load_yaml_config
from utils.infer import inference_main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="单图测试 AD-LLM（纯 Qwen-VL）")
    p.add_argument(
        "--config",
        type=str,
        default="configs/ad_llm_qwen35_9b_zeroshot.yaml",
        help="YAML 配置路径",
    )
    p.add_argument("--model_path", type=str, default=None, help="覆盖 inference.model_path")
    p.add_argument("--image_path", type=str, default=None, help="覆盖 inference.image_path")
    p.add_argument("--prompt", type=str, default=None, help="覆盖 inference.prompt")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_runtime_overrides(cfg, args)
    cfg.setdefault("runtime", {})["mode"] = "inference"

    inf = cfg.get("inference") or {}
    mp = inf.get("model_path")
    if not mp:
        raise ValueError("未设置 inference.model_path（微调输出目录或基座 model_card）。请在 YAML 中填写。")

    img = inf.get("image_path")
    if not img:
        raise ValueError("未设置 inference.image_path（单图路径）。请在 YAML 中填写。")
    img = os.path.abspath(os.path.expanduser(str(img)))
    if not os.path.isfile(img):
        raise FileNotFoundError(f"测试图像不存在: {img}")
    cfg.setdefault("inference", {})["image_path"] = img

    inf = cfg.setdefault("inference", {})
    if inf.get("max_new_tokens") is None:
        inf["max_new_tokens"] = 256

    print(f"[test] config: {os.path.abspath(args.config)}")
    print(f"[test] model_path: {mp}")
    print(f"[test] image_path: {img}")
    inference_main(cfg)


if __name__ == "__main__":
    main()
