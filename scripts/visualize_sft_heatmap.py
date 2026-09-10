#!/usr/bin/env python3
"""Render H (anomaly heatmap) alongside model boxes for selected eval samples.

H is deterministic (frozen vision encoder + prior), so it is computed once per
(test, ref) pair regardless of checkpoint. Each sample yields a row:

    REF | TEST | H heatmap | TEST+H overlay | boxes(GT/Bc/Bf)

The same samples are used for the cross-checkpoint comparison as
``visualize_sft_boxes.py`` so the two artifacts can be read side by side.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/visualize_sft_heatmap.py \
        --config configs/qwen35_2b_outcome.yaml \
        --eval-dir outputs/train/region_sft/_checkpoint_eval \
        --out outputs/train/region_sft/_checkpoint_eval/_viz_heat
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from data.prior_dataset import PriorCollator
from models.anomaly_prior import heatmap_to_pil, overlay_heatmap_on_image
from outcome.engine import load_model
from outcome.inputs import encode_pair_canonical
from utils.common import set_seed
from utils.config import load_yaml_config

GREEN = (0, 200, 0)
YELLOW = (255, 190, 0)
RED = (255, 45, 45)


def _font(size=16):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def to_pixels(box, wh):
    if box is None:
        return None
    return [box[0] * wh[0] / 1000, box[1] * wh[1] / 1000,
            box[2] * wh[0] / 1000, box[3] * wh[1] / 1000]


def _draw_box(draw, box, color, label, font):
    if box is None:
        return
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
    draw.text((x0 + 3, max(0, y0 - 20)), label, fill=color, font=font)


def draw_boxes(image, row, orig_size):
    im = image.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _font(14)
    w, h = im.size
    ow, oh = orig_size
    gt_px = row.get("gt_box_px")
    if gt_px is not None:
        gt_px = [gt_px[0] / ow * w, gt_px[1] / oh * h, gt_px[2] / ow * w, gt_px[3] / oh * h]
    cand = to_pixels(row.get("candidate_bbox_2d"), (w, h))
    pred = to_pixels(row.get("bbox_2d"), (w, h))
    _draw_box(draw, gt_px, GREEN, "GT", font)
    _draw_box(draw, cand, YELLOW, "Bc", font)
    _draw_box(draw, pred, RED, "Bf", font)
    return im


def caption(text, width, height=28):
    bar = Image.new("RGB", (width, height), (28, 28, 28))
    ImageDraw.Draw(bar).text((6, 5), text, fill=(235, 235, 235), font=_font(15))
    return bar


def hstack(labeled, gap=4):
    """labeled: list of (title, PIL image)."""
    cell_w = max(im.width for _, im in labeled)
    cell_h = max(im.height for _, im in labeled)
    cols = []
    for title, im in labeled:
        im2 = im.resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        cap = caption(title, cell_w)
        col = Image.new("RGB", (cell_w, cell_h + cap.height), (14, 14, 14))
        col.paste(cap, (0, 0))
        col.paste(im2, (0, cap.height))
        cols.append(col)
    total_w = sum(c.width for c in cols) + gap * (len(cols) - 1)
    out = Image.new("RGB", (total_w, cols[0].height), (10, 10, 10))
    x = 0
    for c in cols:
        out.paste(c, (x, 0))
        x += c.width + gap
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--n-anomaly", type=int, default=4)
    p.add_argument("--n-normal", type=int, default=2)
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    out = Path(args.out) if args.out else eval_dir / "_viz_heat"
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml_config(args.config)
    set_seed(int(cfg["training"]["seed"]))
    cfg["outcome"]["sft_adapter"] = str(eval_dir.parent / "checkpoint-2000")

    print("loading model + prior ...", flush=True)
    model, processor, prior = load_model(cfg)
    collator = PriorCollator(processor, prior, cfg)
    device = next(prior.visual.parameters()).device

    # load rows from checkpoint-2000, select samples
    rows = []
    with (eval_dir / "checkpoint-2000.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    anom = [r for r in rows if r["is_anomaly"]]
    norm = [r for r in rows if not r["is_anomaly"]]
    anom_sorted = sorted(anom, key=lambda r: -(r.get("iou_f") or -1))
    n = len(anom_sorted)
    # spread: top-2 and bottom-2
    picks = anom_sorted[:2] + anom_sorted[-2:] if n > 4 else anom_sorted
    picks = picks[:args.n_anomaly] + norm[:args.n_normal]

    panels = []
    for r in picks:
        test = Image.open(r["image_path"]).convert("RGB")
        ref = Image.open(r["ref_path"]).convert("RGB")
        ref_rs, test_rs = collator._align_pair(ref, test)
        vis_in = collator._concat_image_tensors(ref_rs, test_rs)
        with torch.no_grad():
            vis = encode_pair_canonical(
                prior, vis_in["pixel_values"].to(device), vis_in["image_grid_thw"].to(device)
            )
        hmap = vis["patch_map"]
        heat_pil = heatmap_to_pil(hmap, test_rs.size)
        overlay = Image.blend(test_rs.convert("RGB"), heat_pil, 0.45)
        boxes = draw_boxes(test_rs, r, tuple(Image.open(r["image_path"]).size))
        iou = r.get("iou_f")
        title = f"{r['class_name']} gt={'A' if r['is_anomaly'] else 'N'} pred={'A' if r.get('pred') else 'N'} iou={(iou or 0):.2f}"
        panels.append((title, hstack([
            ("REF", ref_rs),
            ("TEST", test_rs),
            ("H", heat_pil),
            ("TEST+H", overlay),
            ("boxes", boxes),
        ])))

    # vertical stack
    w = max(im.width for _, im in panels)
    total_h = sum(im.height for _, im in panels) + 4 * (len(panels) - 1)
    sheet = Image.new("RGB", (w, total_h), (10, 10, 10))
    y = 0
    for title, im in panels:
        sheet.paste(im, (0, y))
        y += im.height + 4
    out_path = out / "heatmap_boxes.png"
    sheet.save(out_path)
    print(f"rendered {len(panels)} samples -> {out_path}")


if __name__ == "__main__":
    main()
