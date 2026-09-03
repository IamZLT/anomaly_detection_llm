#!/usr/bin/env python3
"""Coordinate-alignment diagnostic: print image/mask/H/GT coordinate chains.

For a handful of samples, dump the full coordinate chain used by the pipeline so
we can verify that (a) mask GT → bbox → 0-1000, and (b) H prior points → 0-1000,
live in the *same* 0-1000 space, and (c) the vision-prior fallback is not
silently producing a second, different set of prior points.

Prints per sample:
    orig_size        original image (W, H) pixels
    vision_size      resized image fed to the model (W, H)
    gt_box_px        mask-derived bbox in original pixels
    gt_box_1000      same bbox mapped into the 0-1000 system
    H_points         prior points already in the 0-1000 system
    fallback_triggered / points_before / points_after

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/diagnose_coords.py \
        --config configs/qwen35_08b_grpo.yaml --n 8 [--split dev|test]
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from utils.config import load_yaml_config
from utils.common import set_seed


def _fmt_box(box) -> str:
    if box is None:
        return "None"
    return "[" + ",".join(str(int(round(float(x)))) for x in box) + "]"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/qwen35_08b_grpo.yaml")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    args = p.parse_args()

    cfg = load_yaml_config(args.config)
    cfg.setdefault("runtime", {})["mode"] = "train"
    cfg.setdefault("distributed", {})["num_gpu"] = 1
    set_seed(int(cfg.get("training", {}).get("seed", 42)))

    from data.prior_dataset import PriorCollator, PriorCoTDataset, build_train_ref_pool
    from data.scan import load_prior_split, split_holdout_by_class
    from reasoning.rewards import pixels_to_qwen1000
    from rl.trainer import setup_prior_model

    train_samples, test_samples = load_prior_split(cfg)
    holdout = float((cfg.get("data") or {}).get("holdout_ratio", 0.1) or 0.0)
    train_samples, dev_samples = split_holdout_by_class(
        train_samples, holdout, seed=int(cfg.get("training", {}).get("seed", 42))
    )
    train_ref_pool = build_train_ref_pool(train_samples)
    split_samples = dev_samples if args.split == "dev" else test_samples

    model, processor, prior = setup_prior_model(cfg)
    model = model.cuda()
    model.eval()

    ds = PriorCoTDataset(split_samples, cfg, processor, mode="eval", ref_pool=train_ref_pool)
    collator = PriorCollator(processor, prior, cfg)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)

    print(f"===== coordinate alignment  config={args.config} split={args.split} n={args.n} =====")
    n_fallback = 0
    for i, batch in enumerate(loader):
        if i >= args.n:
            break
        meta = batch["_meta"][0]
        orig_size = tuple(meta["orig_size"])
        vision_size = tuple(meta["vision_size"])
        gt_px = meta.get("gt_box_px")
        gt_1000 = pixels_to_qwen1000(gt_px, orig_size) if gt_px is not None else None
        h_points = meta.get("prior_points")
        fb = bool(meta.get("fallback_triggered"))
        if fb:
            n_fallback += 1

        print(f"\n[{i}] {meta.get('image_path')}  class={meta.get('class_name')} "
              f"is_anomaly={meta.get('is_anomaly')}")
        print(f"    orig_size    = {orig_size}")
        print(f"    vision_size  = {vision_size}")
        print(f"    gt_box_px    = {_fmt_box(gt_px)}")
        print(f"    gt_box_1000  = {_fmt_box(gt_1000)}")
        print(f"    H_points     = {h_points}")
        print(f"    fallback_triggered = {fb}")
        if fb:
            print(f"    points_before = {meta.get('points_before')}")
            print(f"    points_after  = {h_points}")

    print(f"\n===== fallback_triggered count = {n_fallback}/{min(args.n, i + 1)} =====")


if __name__ == "__main__":
    main()
