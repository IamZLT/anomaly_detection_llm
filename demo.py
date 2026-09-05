#!/usr/bin/env python3
"""Single-image inference demo."""

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from PIL import Image

from evaluation.infer import build_generation_inputs, decode_generation_output
from models.qwen35 import setup_model_and_processor
from utils.common import (
    draw_bbox_on_image,
    infer_model_compute_device,
    parse_grounding_output,
    qwen_norm1000_to_original_pixels,
)
from utils.config import apply_runtime_overrides, load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Single-image anomaly grounding demo")
    p.add_argument("--config", type=str, default="configs/qwen35_2b_grpo.yaml")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--prompt", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_runtime_overrides(cfg, args)
    model_path = os.path.abspath(os.path.expanduser(args.model_path))
    image_path = os.path.abspath(os.path.expanduser(args.image_path))
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图像不存在: {image_path}")

    prompt = args.prompt or (cfg.get("inference") or {}).get("prompt")
    if not prompt:
        cls = "object"
        prompt = str((cfg.get("prompt") or {}).get("user") or "").replace("{class_name}", cls)
    print(f"[demo] model={model_path}")
    print(f"[demo] image={image_path}")
    model, processor = setup_model_and_processor(
        cfg, for_inference=True, model_name_override=model_path, freeze_vision=True
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    image_original = Image.open(image_path).convert("RGB")
    original_size = image_original.size
    inputs = build_generation_inputs(cfg, processor, image_original, prompt)
    device = infer_model_compute_device(model)
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    inf = cfg.get("inference") or {}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(inf.get("max_new_tokens", 768)),
            temperature=float(inf.get("temperature", 0.0)),
            top_p=float(inf.get("top_p", 0.9)),
            do_sample=bool(inf.get("do_sample", False)),
        )
    response = decode_generation_output(processor, outputs, inputs, cfg)
    print(response)
    bbox_data = parse_grounding_output(response)
    bbox = (bbox_data or {}).get("bbox_2d")
    if bbox and len(bbox) == 4:
        original_bbox = qwen_norm1000_to_original_pixels(list(map(float, bbox)), original_size)
        out_dir = (inf.get("visual_output_dir") or os.path.join(PROJECT_ROOT, "outputs", "eval", "qwen_infer_vis"))
        out_dir = os.path.abspath(os.path.expanduser(str(out_dir)))
        os.makedirs(out_dir, exist_ok=True)
        annotated = draw_bbox_on_image(image_original.copy(), original_bbox, "Bf")
        stem, ext = os.path.splitext(os.path.basename(image_path))
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            ext = ".png"
        out_path = os.path.join(out_dir, f"annotated_{stem}_{time.strftime('%Y%m%d_%H%M%S')}{ext}")
        annotated.save(out_path)
        print(f"\n可视化已保存: {out_path}")


if __name__ == "__main__":
    main()
