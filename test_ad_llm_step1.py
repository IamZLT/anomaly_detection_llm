#!/usr/bin/env python3
"""
单图测试：Stage-1（visual prototype 版本）推理 + anomaly heatmap

输入：
  - configs/ad_llm_step1.yaml
  - step1 训练输出目录中的 dino_bridge.bin
  - 一张图片

输出：
  - 控制台打印 bbox（JSON）
  - 可选保存可视化图（bbox + 热力图叠加）

支持两种测试方式：
1) 推荐：加载 `train_ad_llm_step1.py` 训练出的 `epoch_*.pth`（含 prototypes），
   用 `compute_patch_cls_loss` 输出 patch abnormal prob，再上采样得到热力图。
2) 退化：只加载 `dino_bridge*.bin`（只有映射权重，无 prototypes），
   用 `1 - cos(mapped_patches, resized_clip_patches)` 作为启发式热力图。

热力图与训练 eval 共用 ``utils.visualization``，保证像素级一致。
"""

import argparse
import json
import os
from typing import Dict, List

import torch
from PIL import Image

from models.visual_proto import DinoClipVisualPrototypeModel, VisualPrototypes
from utils.visualization import bbox_from_map, compute_step1_heat_up, heatmap_overlay, prepare_step1_image_tensor
from utils.qwen_common import smart_resize
from utils.qwen_config import load_yaml_config


def _load_state(path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return payload.get("state_dict", payload)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("test_ad_llm_step1 single image")
    p.add_argument("--config", type=str, default="configs/ad_llm_step1.yaml")
    p.add_argument("--ckpt_path", type=str, default=None, help="epoch_*.pth 或 dino_bridge*.bin")
    p.add_argument("--ckpt_dir", type=str, default=None, help="包含 dino_bridge_latest.bin/dino_bridge.bin 的目录（退化模式）")
    p.add_argument("--image_path", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_vis", action="store_true", help="覆盖 YAML：强制保存可视化")
    p.add_argument("--no_save_vis", action="store_true", help="覆盖 YAML：关闭保存可视化")
    p.add_argument("--vis_path", type=str, default=None)
    p.add_argument("--thr_q", type=float, default=None, help="bbox 阈值分位数")
    return p


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    test_cfg = (cfg.get("test", {}) or {})

    ckpt_path = args.ckpt_path or test_cfg.get("ckpt_path")
    ckpt_dir = args.ckpt_dir or test_cfg.get("ckpt_dir")
    image_path = args.image_path or test_cfg.get("image_path")
    if not image_path:
        raise SystemExit("请提供 --image_path，或在 YAML 的 test.image_path 中填写。")

    device_name = args.device or test_cfg.get("device", "cuda")
    device = torch.device(device_name if (device_name != "cuda" or torch.cuda.is_available()) else "cpu")
    thr_q = float(args.thr_q if args.thr_q is not None else test_cfg.get("thr_q", 0.98))
    vis_path = args.vis_path or test_cfg.get("vis_path", "outputs/step1_vis.png")
    save_vis = bool(test_cfg.get("save_vis", True))
    if args.save_vis:
        save_vis = True
    if args.no_save_vis:
        save_vis = False

    model = DinoClipVisualPrototypeModel(
        dino_model_path=cfg["dino"]["model_path"],
        clip_model_path=cfg["clip"]["model_path"],
        layer_indices=cfg["dino"]["layer_indices"],
        dino_image_size=int(cfg["dino"]["image_size"]),
        clip_image_size=int(cfg["clip"]["image_size"]),
        local_files_only=bool(cfg.get("model", {}).get("local_files_only", False)),
    ).to(device)
    model.eval()

    prototypes = VisualPrototypes()
    if ckpt_path:
        payload = torch.load(ckpt_path, map_location="cpu")
        if isinstance(payload, dict) and "model_state_dict" in payload:
            model.load_state_dict(payload["model_state_dict"], strict=False)
            p = (payload.get("prototypes") or {})
            if p.get("normal") is not None:
                prototypes.normal = p["normal"]
            if p.get("abnormal") is not None:
                prototypes.abnormal = p["abnormal"]
        else:
            sd = _load_state(ckpt_path)
            model.load_state_dict(sd, strict=False)
    else:
        if not ckpt_dir:
            raise SystemExit("请提供 --ckpt_path 或 --ckpt_dir（或在 YAML 的 test.ckpt_path/test.ckpt_dir 中填写）。")
        cand = os.path.join(ckpt_dir, "dino_bridge_latest.bin")
        if not os.path.exists(cand):
            cand = os.path.join(ckpt_dir, "dino_bridge.bin")
        sd = _load_state(cand)
        model.load_state_dict(sd, strict=False)

    img = Image.open(image_path).convert("RGB")
    img_rs, orig_size, _ = smart_resize(img, int(cfg["data"]["max_image_size"]), int(cfg["data"]["factor"]))

    img_t = prepare_step1_image_tensor(img_rs, int(cfg["dino"]["image_size"]), device)
    heat_up, _mode = compute_step1_heat_up(
        model=model,
        img_t=img_t,
        img_rs=img_rs,
        cfg=cfg,
        prototypes=prototypes,
        device=device,
    )

    bbox = bbox_from_map(heat_up, thr_quantile=thr_q)
    out_json = {"bbox_2d": bbox, "label": "anomaly" if bbox is not None else "normal"}
    print(json.dumps(out_json, ensure_ascii=False))

    if save_vis:
        os.makedirs(os.path.dirname(vis_path) or ".", exist_ok=True)
        vis = heatmap_overlay(img_rs, heat_up, alpha=float(test_cfg.get("heatmap_alpha", 0.45)))
        vis.save(vis_path)
        print(f"[vis] saved: {vis_path}")


if __name__ == "__main__":
    main()
