#!/usr/bin/env python3
"""
Visualize DINO / CLIP / mapped (DINO->CLIP bridge) patch heatmaps on a single image.

Reads configs/ad_llm_step2.yaml:
- uses inference.image_path
- uses dino.model_path / clip.model_path
- loads model.bridge_ckpt_path (dino_bridge.bin or step1 epoch_*.pth/best_*.pth)
- saves overlays into paths.output_dir
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# allow running from anywhere: add project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch.nn.functional as F

from models.visual_proto import DinoClipVisualPrototypeModel, infer_square_hw
from loss.FocalLoss import FocalLoss
from loss.BinaryDiceLoss import BinaryDiceLoss
from train_ad_llm_step1 import compute_patch_cls_loss, VisualPrototypes
from utils.common import smart_resize
from utils.config import load_yaml_config


def _heatmap_overlay(image: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Overlay heatmap on image with mvtec_proj_1shot.py-like visual style (coolwarm).
    heat should be already normalized to [0,1].
    """
    heat = np.clip(heat, 0.0, 1.0)
    try:
        import matplotlib.cm as cm

        cmap = cm.get_cmap("coolwarm")
        rgba = cmap(heat)  # [H,W,4] float in [0,1]
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    except Exception:
        # fallback: jet-like
        r = np.clip(1.5 * heat - 0.5, 0.0, 1.0)
        g = np.clip(1.5 - np.abs(2.0 * heat - 1.0), 0.0, 1.0)
        b = np.clip(0.5 - 1.5 * (heat - 1.0), 0.0, 1.0)
        rgb = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

    hm = Image.fromarray(rgb, mode="RGB")
    if hm.size != image.size:
        hm = hm.resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), hm, float(alpha))


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx > mn:
        return (x - mn) / (mx - mn)
    return np.zeros_like(x, dtype=np.float32)


def _patch_norm_heatmap(patches: torch.Tensor, grid_hw: Tuple[int, int]) -> np.ndarray:
    """
    patches: [P, C] or [1,P,C]
    return heat: [H,W] in [0,1]
    """
    if patches.dim() == 3:
        patches = patches[0]
    h, w = grid_hw
    norms = patches.float().norm(dim=-1).view(h, w)
    return norms.detach().cpu().numpy()


def _cos_diff_heatmap(
    mapped_patches: torch.Tensor,
    dino_grid: Tuple[int, int],
    clip_patches: torch.Tensor,
    clip_grid: Tuple[int, int],
) -> np.ndarray:
    """
    heat(p) = 1 - cos(mapped_patch(p), clip_patch(p))
    clip patches are resized to dino grid before computing cos.
    mapped_patches: [B, Pd, C]
    clip_patches:   [B, Pc, C]
    return heat [Hd,Wd] in [0,1]
    """
    if mapped_patches.dim() != 3 or clip_patches.dim() != 3:
        raise ValueError("mapped_patches/clip_patches must be [B,P,C]")
    b, pd, c = mapped_patches.shape
    gc_h, gc_w = clip_grid
    gd_h, gd_w = dino_grid
    if gd_h * gd_w != pd:
        raise ValueError("dino_grid does not match mapped patch count")

    clip_map = clip_patches.view(b, gc_h, gc_w, c).permute(0, 3, 1, 2)  # [B,C,Hc,Wc]
    clip_map = F.interpolate(clip_map, size=(gd_h, gd_w), mode="bilinear", align_corners=False)
    clip_map = clip_map.permute(0, 2, 3, 1).reshape(b, pd, c)

    mp = F.normalize(mapped_patches.float(), dim=-1)
    cp = F.normalize(clip_map.float(), dim=-1)
    cos = (mp * cp).sum(dim=-1)[0].view(gd_h, gd_w)
    heat = 1.0 - cos
    return heat.detach().cpu().numpy()


def _load_bridge_weights_into_model(model: DinoClipVisualPrototypeModel, bridge_ckpt_path: str) -> Dict[str, int]:
    """
    Supports:
    - dino_bridge.bin: {"state_dict": {...}}  (preferred)
    - step1 epoch_*.pth / best_*.pth: {"model_state_dict": {...}, ...}
    """
    p = os.path.abspath(os.path.expanduser(str(bridge_ckpt_path)))
    payload = torch.load(p, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        raw = payload["state_dict"]
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        raw = payload["model_state_dict"]
    else:
        raw = payload
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported bridge checkpoint format: {type(raw)}")

    picked: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        k2 = k[7:] if k.startswith("module.") else k
        if k2.startswith("cls_mapper.") or k2.startswith("patch_mapper."):
            picked[k2] = v

    load_ret = model.load_state_dict(picked, strict=False)
    return {
        "picked_keys": int(len(picked)),
        "missing_keys": int(len(getattr(load_ret, "missing_keys", []) or [])),
        "unexpected_keys": int(len(getattr(load_ret, "unexpected_keys", []) or [])),
    }


def _load_prototypes_if_present(bridge_ckpt_path: str) -> VisualPrototypes:
    """
    If bridge_ckpt_path points to step1 epoch_*.pth / best_*.pth, it may contain prototypes.
    Returns VisualPrototypes (may be empty).
    """
    p = os.path.abspath(os.path.expanduser(str(bridge_ckpt_path)))
    payload = torch.load(p, map_location="cpu")
    prot = VisualPrototypes()
    if isinstance(payload, dict):
        p_obj = payload.get("prototypes") or {}
        if isinstance(p_obj, dict):
            n = p_obj.get("normal")
            a = p_obj.get("abnormal")
            if n is not None:
                prot.normal = n
            if a is not None:
                prot.abnormal = a
    return prot


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Visualize DINO/CLIP/mapped patch heatmaps")
    p.add_argument("--config", type=str, default="configs/ad_llm_step2.yaml")
    p.add_argument("--out_dir", type=str, default=None, help="Override save directory (default: paths.output_dir)")
    return p


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)

    img_path = (cfg.get("inference", {}) or {}).get("image_path")
    if not img_path:
        raise SystemExit("configs/ad_llm_step2.yaml: inference.image_path 为空，请先填写要可视化的图片路径。")
    img_path = os.path.abspath(os.path.expanduser(str(img_path)))
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"image not found: {img_path}")

    out_root = args.out_dir or str((cfg.get("paths", {}) or {}).get("output_dir") or "./logs_step2")
    out_root = os.path.abspath(os.path.expanduser(out_root))
    os.makedirs(out_root, exist_ok=True)
    out_dir = os.path.join(out_root, "feature_vis")
    os.makedirs(out_dir, exist_ok=True)

    dino_cfg = cfg.get("dino", {}) or {}
    clip_cfg = cfg.get("clip", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    bridge_ckpt_path = model_cfg.get("bridge_ckpt_path")
    if not bridge_ckpt_path:
        raise SystemExit("configs/ad_llm_step2.yaml: model.bridge_ckpt_path 为空。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load & resize image for visualization
    img = Image.open(img_path).convert("RGB")
    img_rs, orig_size, _ = smart_resize(
        img.copy(),
        max_size=int((cfg.get("data", {}) or {}).get("max_image_size", 512)),
        factor=int((cfg.get("data", {}) or {}).get("factor", 28)),
    )

    local_files_only = bool(model_cfg.get("local_files_only", True))
    # Match step1/test preprocessing (Resize->ToTensor->Normalize)
    dino_image_size = int(dino_cfg.get("image_size", 512))
    clip_image_size = int(clip_cfg.get("image_size", 224))
    tfm_dino = transforms.Compose(
        [
            transforms.Resize((dino_image_size, dino_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tfm_clip = transforms.Compose(
        [
            transforms.Resize((clip_image_size, clip_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dino_pixel_values = tfm_dino(img_rs).unsqueeze(0).to(device)
    clip_pixel_values = tfm_clip(img_rs).unsqueeze(0).to(device)

    # build model (DINO+CLIP + mappers)
    model = DinoClipVisualPrototypeModel(
        dino_model_path=str(dino_cfg["model_path"]),
        clip_model_path=str(clip_cfg["model_path"]),
        layer_indices=[int(x) for x in dino_cfg.get("layer_indices", [12, 16, 20, 24])],
        dino_image_size=dino_image_size,
        clip_image_size=clip_image_size,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    bridge_stats = _load_bridge_weights_into_model(model, str(bridge_ckpt_path))
    prototypes = _load_prototypes_if_present(str(bridge_ckpt_path))

    # forward encoders
    cls_stack, patch_stack, dino_grid = model.encode_dino(dino_pixel_values)
    _, clip_patches, clip_grid = model.encode_clip(clip_pixel_values)
    mapped_patches = model.patch_mapper(patch_stack)  # [B,P,C]

    # choose DINO patches for visualization: last layer raw patches
    dino_last = patch_stack[-1]  # [B,P,D]
    dino_heat_raw = _patch_norm_heatmap(dino_last, dino_grid)
    # mapped heat: prefer step1-style anomaly prob map if prototypes exist; otherwise fallback to cos-diff
    if prototypes.ready():
        dummy_mask = torch.zeros((1, 1, dino_image_size, dino_image_size), device=device)
        _, _, patch_prob_abn = compute_patch_cls_loss(
            mapped_patches=mapped_patches,
            dino_grid=dino_grid,
            gt_mask=dummy_mask,
            proto_normal=prototypes.normal.to(device),
            proto_abnormal=prototypes.abnormal.to(device),
            focal_loss_fn=FocalLoss(alpha=0.25, gamma=2.0),
            dice_loss_fn=BinaryDiceLoss(),
            temperature=float((cfg.get("step1", {}) or {}).get("temperature", 0.07)),
        )
        mapped_heat_raw = patch_prob_abn[0, 0].detach().float().cpu().numpy()
    else:
        mapped_heat_raw = _cos_diff_heatmap(mapped_patches, dino_grid, clip_patches, clip_grid)
    clip_heat_raw = _patch_norm_heatmap(clip_patches, clip_grid)

    # mvtec_proj_1shot.py visualizes per-image normalized maps
    dino_heat = _norm01(dino_heat_raw)
    mapped_heat = _norm01(mapped_heat_raw)
    clip_heat = _norm01(clip_heat_raw)

    # save overlays
    alpha = float((cfg.get("training", {}) or {}).get("eval_heatmap_alpha", 0.45))
    dino_overlay = _heatmap_overlay(img_rs, dino_heat, alpha=alpha)
    mapped_overlay = _heatmap_overlay(img_rs, mapped_heat, alpha=alpha)
    clip_overlay = _heatmap_overlay(img_rs, clip_heat, alpha=alpha)

    dino_path = os.path.join(out_dir, "dino_vis_norm_overlay.png")
    mapped_path = os.path.join(
        out_dir,
        "mapped_patch_prob_abn_overlay.png" if prototypes.ready() else "mapped_clip_cosdiff_overlay.png",
    )
    clip_path = os.path.join(out_dir, "clip_vis_norm_overlay.png")
    img_path_out = os.path.join(out_dir, "image_resized.png")

    # save raw maps as grayscale too (debug)
    dino_raw_path = os.path.join(out_dir, "dino_heat_raw.png")
    mapped_raw_path = os.path.join(out_dir, "mapped_heat_raw.png")
    clip_raw_path = os.path.join(out_dir, "clip_heat_raw.png")

    img_rs.save(img_path_out)
    dino_overlay.save(dino_path)
    mapped_overlay.save(mapped_path)
    clip_overlay.save(clip_path)
    Image.fromarray((_norm01(dino_heat_raw) * 255.0).astype(np.uint8), mode="L").resize(img_rs.size, Image.Resampling.NEAREST).save(dino_raw_path)
    Image.fromarray((_norm01(mapped_heat_raw) * 255.0).astype(np.uint8), mode="L").resize(img_rs.size, Image.Resampling.NEAREST).save(mapped_raw_path)
    Image.fromarray((_norm01(clip_heat_raw) * 255.0).astype(np.uint8), mode="L").resize(img_rs.size, Image.Resampling.NEAREST).save(clip_raw_path)

    meta = {
        "config": os.path.abspath(args.config),
        "image_path": img_path,
        "orig_size": list(orig_size),
        "resized_size": [img_rs.size[0], img_rs.size[1]],
        "device": str(device),
        "bridge_ckpt_path": os.path.abspath(os.path.expanduser(str(bridge_ckpt_path))),
        "bridge_load": bridge_stats,
        "prototypes_ready": bool(prototypes.ready()),
        "dino_grid": list(dino_grid),
        "clip_grid": list(clip_grid),
        "outputs": {
            "image_resized": img_path_out,
            "dino_vis_norm_overlay": dino_path,
            "mapped_overlay": mapped_path,
            "clip_vis_norm_overlay": clip_path,
            "dino_heat_raw": dino_raw_path,
            "mapped_heat_raw": mapped_raw_path,
            "clip_heat_raw": clip_raw_path,
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[ok] saved to:", out_dir)


if __name__ == "__main__":
    main()

