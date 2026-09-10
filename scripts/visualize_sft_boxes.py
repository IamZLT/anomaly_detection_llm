#!/usr/bin/env python3
"""Render eval JSONL rows as images with GT / candidate / final boxes overlaid.

Produces:
  1. Per-checkpoint contact sheet (representative anomaly + normal cases).
  2. A cross-checkpoint comparison for a fixed set of samples (columns = checkpoints).

Usage:
    python scripts/visualize_sft_boxes.py --eval-dir outputs/train/region_sft/_checkpoint_eval \
        --out outputs/train/region_sft/_checkpoint_eval/_viz
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

GREEN = (0, 200, 0)
YELLOW = (255, 190, 0)
RED = (255, 45, 45)


def _font(size=18):
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


def _draw_one(draw, box, color, label, font):
    if box is None:
        return
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
    draw.text((x0 + 3, max(0, y0 - 20)), label, fill=color, font=font)


def render_case(row, max_side=420):
    test = Image.open(row["image_path"]).convert("RGB")
    w, h = test.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        test = test.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
        w, h = test.size

    gt_px = row.get("gt_box_px")
    cand_px = to_pixels(row.get("candidate_bbox_2d"), (w, h))
    pred_px = to_pixels(row.get("bbox_2d"), (w, h))

    # rescale gt (already in original-pixel coords) to the display size
    if gt_px is not None:
        orig_w, orig_h = Image.open(row["image_path"]).size
        gt_px = [gt_px[0] / orig_w * w, gt_px[1] / orig_h * h,
                 gt_px[2] / orig_w * w, gt_px[3] / orig_h * h]

    draw = ImageDraw.Draw(test)
    font = _font(16)
    _draw_one(draw, gt_px, GREEN, "GT", font)
    _draw_one(draw, cand_px, YELLOW, "Bc", font)
    _draw_one(draw, pred_px, RED, "Bf", font)
    return test


def caption(text, width):
    bar = Image.new("RGB", (width, 30), (28, 28, 28))
    ImageDraw.Draw(bar).text((6, 6), text, fill=(235, 235, 235), font=_font(15))
    return bar


def make_sheet(cases, out_path):
    """cases: list of (title, pil_image). Grid 3 per row."""
    cols = 3
    cell_w = max(im.width for _, im in cases)
    cell_h = max(im.height for _, im in cases)
    cell_h += 30  # caption
    rows = [cases[i:i + cols] for i in range(0, len(cases), cols)]
    grid_w = cell_w * cols
    grid_h = cell_h * len(rows)
    sheet = Image.new("RGB", (grid_w, grid_h), (14, 14, 14))
    y = 0
    for r in rows:
        x = 0
        for title, im in r:
            cap = caption(title, cell_w)
            sheet.paste(cap, (x, y))
            sheet.paste(im, (x, y + 30))
            x += cell_w
        y += cell_h
    sheet.save(out_path)
    return out_path


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def title(r):
    iou = r.get("iou_f")
    iou_s = f"{iou:.2f}" if iou is not None else "--"
    gt = "A" if r["is_anomaly"] else "N"
    pred = "A" if r.get("pred") else "N"
    return f"{r['class_name']} gt={gt} pred={pred} iou={iou_s}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--per-ckpt", type=int, default=9)
    p.add_argument("--compare", type=int, default=6)
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    out = Path(args.out) if args.out else eval_dir / "_viz"
    out.mkdir(parents=True, exist_ok=True)

    ckpts = ["checkpoint-1000", "checkpoint-2000", "checkpoint-3000"]
    loaded = {}
    for c in ckpts:
        f = eval_dir / f"{c}.jsonl"
        if f.exists():
            loaded[c] = load_rows(f)

    # 1. per-checkpoint contact sheet (best + worst anomaly + normals)
    for c, rows in loaded.items():
        anom = [r for r in rows if r["is_anomaly"]]
        norm = [r for r in rows if not r["is_anomaly"]]
        anom_sorted = sorted(anom, key=lambda r: -(r.get("iou_f") or -1))
        n_best = max(1, args.per_ckpt // 2)
        pick = anom_sorted[:n_best] + anom_sorted[-n_best:] if len(anom_sorted) > n_best * 2 else anom_sorted
        # dedupe
        seen = set()
        pick = [r for r in pick if not (id(r) in seen or seen.add(id(r)))]
        pick += norm[:2]
        cases = [(title(r), render_case(r)) for r in pick]
        make_sheet(cases, out / f"{c}.png")
        print(f"[{c}] rendered {len(cases)} cases -> {out / (c + '.png')}")

    # 2. cross-checkpoint comparison on a fixed set of anomaly samples
    base = loaded["checkpoint-2000"]
    anom_base = sorted([r for r in base if r["is_anomaly"]],
                       key=lambda r: -(r.get("iou_f") or -1))
    # pick a spread: top-2, mid, bottom-3 by iou
    idxs = []
    if anom_base:
        n = len(anom_base)
        for i in [0, max(1, n // 4), n // 2, n - 1, n - 2, n - 3]:
            if 0 <= i < n and i not in idxs:
                idxs.append(i)
        idxs = idxs[:args.compare]
    samples = [anom_base[i] for i in idxs]

    if samples:
        per_row = len(loaded)
        cell_w = 380
        # render each sample under each checkpoint
        grid_rows = []
        for s in samples:
            key = (s["class_name"], s["image_path"])
            ims = []
            for c in ckpts:
                if c not in loaded:
                    continue
                match = next((r for r in loaded[c]
                              if r["class_name"] == s["class_name"] and r["image_path"] == s["image_path"]), None)
                if match is None:
                    continue
                im = render_case(match, max_side=cell_w)
                cap = caption(f"{c.replace('checkpoint-','')}  iou={ (match.get('iou_f') or 0):.2f}", cell_w)
                col = Image.new("RGB", (cell_w, im.height + 30), (14, 14, 14))
                col.paste(cap, (0, 0))
                col.paste(im, (0, 30))
                ims.append(col)
            if ims:
                row = Image.new("RGB", (cell_w * len(ims), max(i.height for i in ims)), (14, 14, 14))
                x = 0
                for im in ims:
                    row.paste(im, (x, 0))
                    x += cell_w
                grid_rows.append(row)
        if grid_rows:
            w = max(r.width for r in grid_rows)
            total_h = sum(r.height for r in grid_rows)
            comp = Image.new("RGB", (w, total_h), (10, 10, 10))
            y = 0
            for r in grid_rows:
                comp.paste(r, (0, y))
                y += r.height
            comp.save(out / "cross_checkpoint.png")
            print(f"cross-checkpoint comparison ({len(samples)} samples) -> {out / 'cross_checkpoint.png'}")


if __name__ == "__main__":
    main()
