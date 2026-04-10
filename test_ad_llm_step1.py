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
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from loss.BinaryDiceLoss import BinaryDiceLoss
from loss.FocalLoss import FocalLoss
from models.visual_proto import DinoClipVisualPrototypeModel, VisualPrototypes, infer_square_hw
from train_ad_llm_step1 import compute_patch_cls_loss
from utils.qwen_common import smart_resize
from utils.qwen_config import load_yaml_config


def _load_state(path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return payload.get("state_dict", payload)


def _bbox_from_map(anom: np.ndarray, thr_quantile: float = 0.98) -> List[int] | None:
    """
    anom: [H,W] float in [0,1]
    返回 [x1,y1,x2,y2] 或 None
    """
    h, w = anom.shape
    thr = float(np.quantile(anom, thr_quantile))
    m = anom >= thr
    if m.sum() < 10:
        return None
    ys, xs = np.where(m)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    # clamp
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    return [x1, y1, x2, y2]


def _heatmap_overlay(image: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    image: PIL RGB
    heat:  [H,W] in [0,1] (same size as image)
    return: overlay image (RGB)
    """
    heat = np.clip(heat, 0.0, 1.0)
    h, w = heat.shape
    # simple "jet-like" colormap without extra deps
    r = np.clip(1.5 * heat - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * heat - 1.0), 0.0, 1.0)
    b = np.clip(0.5 - 1.5 * (heat - 1.0), 0.0, 1.0)
    cm = np.stack([r, g, b], axis=-1)  # [H,W,3]
    cm_u8 = (cm * 255.0).astype(np.uint8)
    hm = Image.fromarray(cm_u8, mode="RGB")

    if hm.size != image.size:
        hm = hm.resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), hm, float(alpha))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("test_ad_llm_step1 single image")
    p.add_argument("--config", type=str, default="configs/ad_llm_step1.yaml")
    # 这些字段默认从 YAML 的 test 段读取；CLI 仅作覆盖
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

    # model
    model = DinoClipVisualPrototypeModel(
        dino_model_path=cfg["dino"]["model_path"],
        clip_model_path=cfg["clip"]["model_path"],
        layer_indices=cfg["dino"]["layer_indices"],
        dino_image_size=int(cfg["dino"]["image_size"]),
        clip_image_size=int(cfg["clip"]["image_size"]),
        local_files_only=bool(cfg.get("model", {}).get("local_files_only", False)),
    ).to(device)
    model.eval()

    # load checkpoint
    prototypes = VisualPrototypes()
    loaded_full = False
    if ckpt_path:
        payload = torch.load(ckpt_path, map_location="cpu")
        if isinstance(payload, dict) and "model_state_dict" in payload:
            # epoch_*.pth from train_ad_llm_step1.py
            model.load_state_dict(payload["model_state_dict"], strict=False)
            p = (payload.get("prototypes") or {})
            if p.get("normal") is not None:
                prototypes.normal = p["normal"]
            if p.get("abnormal") is not None:
                prototypes.abnormal = p["abnormal"]
            loaded_full = True
        else:
            # dino_bridge*.bin: only mapper weights
            sd = _load_state(ckpt_path)
            model.load_state_dict(sd, strict=False)
    else:
        # fallback to ckpt_dir
        if not ckpt_dir:
            raise SystemExit("请提供 --ckpt_path 或 --ckpt_dir（或在 YAML 的 test.ckpt_path/test.ckpt_dir 中填写）。")
        cand = os.path.join(ckpt_dir, "dino_bridge_latest.bin")
        if not os.path.exists(cand):
            cand = os.path.join(ckpt_dir, "dino_bridge.bin")
        sd = _load_state(cand)
        model.load_state_dict(sd, strict=False)

    # image
    img = Image.open(image_path).convert("RGB")
    img_rs, orig_size, _ = smart_resize(img, int(cfg["data"]["max_image_size"]), int(cfg["data"]["factor"]))

    # preprocessing consistent with train_ad_llm_step1.py dataset
    image_size = int(cfg["dino"]["image_size"])
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_t = tfm(img_rs).unsqueeze(0).to(device)

    out = model(img_t, img_t)
    mapped_patches = out["mapped_patches"]  # [B, Pd, C]
    dino_grid = out["dino_grid"]

    # build anomaly heatmap
    if prototypes.ready():
        # prototype-based dense abnormal prob (recommended)
        proto_n = prototypes.normal.to(device)
        proto_a = prototypes.abnormal.to(device)
        loss_patch_cls, patch_logits, patch_prob_abn = compute_patch_cls_loss(
            mapped_patches=mapped_patches,
            dino_grid=dino_grid,
            gt_mask=torch.zeros((1, 1, image_size, image_size), device=device),  # dummy (only for resize shape)
            proto_normal=proto_n,
            proto_abnormal=proto_a,
            focal_loss_fn=FocalLoss(alpha=0.25, gamma=2.0),
            dice_loss_fn=BinaryDiceLoss(),
            temperature=float(cfg.get("step1", {}).get("temperature", 0.07)),
        )
        _ = (loss_patch_cls, patch_logits)  # not used for inference
        heat = patch_prob_abn[0, 0].detach().float().cpu().numpy()  # [Hg,Wg]
    else:
        # fallback heuristic: 1 - cos(mapped_patches, resized clip patch map)
        clip_patches = out["clip_patches"]  # [B,Pc,C] or empty
        clip_grid = out["clip_grid"]
        if clip_patches is None or clip_patches.numel() == 0 or clip_grid == (0, 0):
            raise SystemExit("当前 ckpt 没有 prototypes，且 CLIP patch 不可用，无法生成热力图。请用 epoch_*.pth 测试。")
        b, pc, c = clip_patches.shape
        hc, wc = clip_grid
        hd, wd = dino_grid
        clip_map = clip_patches.view(b, hc, wc, c).permute(0, 3, 1, 2)
        clip_map = F.interpolate(clip_map, size=(hd, wd), mode="bilinear", align_corners=False)
        clip_map = clip_map.permute(0, 2, 3, 1).reshape(b, hd * wd, c)
        mp = F.normalize(mapped_patches, dim=-1)
        cp = F.normalize(clip_map, dim=-1)
        cos = (mp * cp).sum(dim=-1)[0].view(hd, wd)
        heat = (1.0 - cos).detach().float().cpu().numpy()

    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    # upsample to resized image size
    heat_t = torch.from_numpy(heat).unsqueeze(0).unsqueeze(0)
    heat_up = F.interpolate(heat_t, size=(img_rs.height, img_rs.width), mode="bilinear", align_corners=False)
    heat_up = heat_up.squeeze(0).squeeze(0).numpy()

    bbox = _bbox_from_map(heat_up, thr_quantile=thr_q)
    out_json = {"bbox_2d": bbox, "label": "anomaly" if bbox is not None else "normal"}
    print(json.dumps(out_json, ensure_ascii=False))

    if save_vis:
        os.makedirs(os.path.dirname(vis_path) or ".", exist_ok=True)
        vis = _heatmap_overlay(img_rs, heat_up, alpha=float(test_cfg.get("heatmap_alpha", 0.45)))
        vis.save(vis_path)
        print(f"[vis] saved: {vis_path}")


if __name__ == "__main__":
    main()
