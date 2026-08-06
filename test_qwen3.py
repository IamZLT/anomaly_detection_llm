#!/usr/bin/env python3
"""单张图像快速测试微调模型（与训练共用 configs/qwen.yaml）。

图像路径优先顺序：命令行 --image-path > yaml 中 test.image_path > yaml 中 inference.image_path
模型目录：inference.model_path（可用 --model-path 覆盖）

用法:
  python test_qwen3.py --config configs/qwen.yaml
  python test_qwen3.py --config configs/qwen.yaml --image-path /path/to/test.png
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
    p = argparse.ArgumentParser(description="单图测试微调 Qwen3-VL（配置 + inference_main）")
    p.add_argument("--config", type=str, default="configs/qwen.yaml", help="YAML 配置路径")
    p.add_argument("--model-path", type=str, default=None, dest="model_path", help="覆盖 inference.model_path")
    p.add_argument("--image-path", type=str, default=None, dest="image_path", help="覆盖测试图像路径")
    p.add_argument("--prompt", type=str, default=None, help="覆盖 inference.prompt")
    return p


def _resolve_test_image_path(cfg: dict, args: argparse.Namespace) -> str:
    if getattr(args, "image_path", None):
        return os.path.abspath(args.image_path)
    test_cfg = cfg.get("test") or {}
    if test_cfg.get("image_path"):
        return os.path.abspath(str(test_cfg["image_path"]))
    inf = cfg.get("inference") or {}
    if inf.get("image_path"):
        return os.path.abspath(str(inf["image_path"]))
    return ""


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_runtime_overrides(cfg, args)

    img = _resolve_test_image_path(cfg, args)
    if not img:
        raise ValueError(
            "未设置测试图像。请在 configs/qwen.yaml 的 test.image_path 填写单图路径，"
            "或填 inference.image_path，或使用: python test_qwen3.py --image-path /path/to/image.png"
        )
    if not os.path.isfile(img):
        raise FileNotFoundError(f"测试图像不存在: {img}")

    cfg.setdefault("inference", {})["image_path"] = img

    mp = cfg.get("inference", {}).get("model_path")
    if not mp:
        raise ValueError(
            "未设置 inference.model_path（微调输出目录，含权重与 dino_bridge.bin）。"
            "请在 yaml 中填写或使用 --model-path"
        )

    print(f"[test_qwen3] config: {os.path.abspath(args.config)}")
    print(f"[test_qwen3] model_path: {mp}")
    print(f"[test_qwen3] image_path: {img}")
    inference_main(cfg)


if __name__ == "__main__":
    main()
