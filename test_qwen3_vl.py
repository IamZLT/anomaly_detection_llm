#!/usr/bin/env python3
"""
单图测试：与当前训练一致，使用 QwenDinoBridgeModel + DINO/CLIP 桥接（configs/qwen.yaml）。

与 test_qwen3.py 的区别：可从 MVTec test 集随机抽一张图（--random）。

用法:
  python test_qwen3_vl.py --config configs/qwen.yaml --image-path /path/to/img.png
  python test_qwen3_vl.py --config configs/qwen.yaml --random
  python test_qwen3_vl.py --config configs/qwen.yaml --model-path ./logs/xxx/final_model --random
"""

import argparse
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.qwen_config import apply_runtime_overrides, load_yaml_config
from utils.qwen_infer import inference_main

# 默认可视化目录（可被 yaml inference.visual_output_dir 覆盖）
_DEFAULT_VIS_DIR = os.path.join(PROJECT_ROOT, "outputs", "qwen3_vl_test")


def _find_random_test_image(dataset_root: str) -> str:
    """在 MVTec 目录下递归收集 test/ 下图像，随机选一张。"""
    dataset_root = os.path.abspath(os.path.expanduser(dataset_root))
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"数据集根目录不存在: {dataset_root}")

    test_images: list[str] = []
    for category in os.listdir(dataset_root):
        category_path = os.path.join(dataset_root, category)
        if not os.path.isdir(category_path):
            continue
        test_dir = os.path.join(category_path, "test")
        if not os.path.exists(test_dir):
            continue
        for root, _, files in os.walk(test_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    test_images.append(os.path.join(root, f))

    if not test_images:
        raise RuntimeError(f"在 {dataset_root} 下未找到 test 图像")

    path = random.choice(test_images)
    print(f"[test_qwen3_vl] 随机选择: {path}")
    return path


def _resolve_image_path(cfg: dict, args: argparse.Namespace) -> str:
    if getattr(args, "random_image", False):
        root = args.dataset_root or (cfg.get("paths") or {}).get("dataset_root")
        if not root:
            raise ValueError("随机抽图需要 paths.dataset_root 或 --dataset-root")
        return _find_random_test_image(str(root))

    if getattr(args, "image_path", None):
        return os.path.abspath(os.path.expanduser(args.image_path))

    test_cfg = cfg.get("test") or {}
    if test_cfg.get("image_path"):
        return os.path.abspath(str(test_cfg["image_path"]))

    inf = cfg.get("inference") or {}
    if inf.get("image_path"):
        return os.path.abspath(str(inf["image_path"]))

    return ""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen3-VL + DINO 桥 单图测试（与训练配置一致）")
    p.add_argument("--config", type=str, default="configs/qwen.yaml", help="YAML 配置")
    p.add_argument(
        "--model-path",
        type=str,
        default=None,
        dest="model_path",
        help="覆盖 inference.model_path（微调输出 final_model 目录）",
    )
    p.add_argument("--image-path", type=str, default=None, dest="image_path", help="测试图像路径")
    p.add_argument(
        "--random",
        action="store_true",
        dest="random_image",
        help="从 paths.dataset_root 下 MVTec test 随机选一张图",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        dest="dataset_root",
        help="覆盖 dataset_root（仅 --random 时有用）",
    )
    p.add_argument("--prompt", type=str, default=None, help="覆盖 inference.prompt")
    p.add_argument(
        "--vis-dir",
        type=str,
        default=None,
        help="解析到 bbox 时保存可视化目录，默认项目下 outputs/qwen3_vl_test",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_runtime_overrides(cfg, args)

    img = _resolve_image_path(cfg, args)
    if not img:
        raise ValueError(
            "请指定图像: --image-path /path/to.png，或使用 --random，"
            "或在 yaml 中设置 test.image_path / inference.image_path"
        )
    if not os.path.isfile(img):
        raise FileNotFoundError(f"测试图像不存在: {img}")

    cfg.setdefault("inference", {})["image_path"] = img

    mp = cfg.get("inference", {}).get("model_path")
    if not mp:
        raise ValueError(
            "未设置 inference.model_path（final_model 目录，含 pytorch 权重与 dino_bridge.bin）。"
            "请在 configs/qwen.yaml 中填写或使用 --model-path"
        )

    if args.vis_dir:
        cfg.setdefault("inference", {})["visual_output_dir"] = os.path.abspath(
            os.path.expanduser(args.vis_dir)
        )
    elif not (cfg.get("inference") or {}).get("visual_output_dir"):
        cfg.setdefault("inference", {})["visual_output_dir"] = _DEFAULT_VIS_DIR

    # 相对路径相对项目根目录解析（inference_main 内也会对 model_path 做 abspath）
    if not os.path.isabs(mp):
        mp = os.path.join(PROJECT_ROOT, mp)
    cfg["inference"]["model_path"] = os.path.abspath(mp)

    print(f"[test_qwen3_vl] config: {os.path.abspath(args.config)}")
    print(f"[test_qwen3_vl] model_path: {cfg['inference']['model_path']}")
    print(f"[test_qwen3_vl] image_path: {img}")

    inference_main(cfg)


if __name__ == "__main__":
    main()
