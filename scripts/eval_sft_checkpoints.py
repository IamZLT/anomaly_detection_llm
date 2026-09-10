#!/usr/bin/env python3
"""Evaluate every region-SFT checkpoint on the SAME dev subset (greedy, single-box).

For each checkpoint under ``--sft-dir`` (and the final directory itself), load the
model with that ``sft_adapter`` and run the single-box ``evaluate`` on a fixed-size
dev slice. Reports the core SFT acceptance metrics so checkpoints can be ranked:

    task_valid / protocol_core / protocol_strict   (5-block format alignment)
    anomaly_recall / normal_correct_rate            (category correctness)
    mean_iou_f / candidate_box_valid / final_box_valid (localization)

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_sft_checkpoints.py \
        --config configs/qwen35_2b_outcome.yaml \
        --sft-dir outputs/train/region_sft --limit 150
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from outcome.engine import datasets, evaluate, load_model
from outcome.inputs import OutcomeDataset
from utils.common import set_seed
from utils.config import load_yaml_config

KEYS = (
    "n", "n_anomaly", "n_normal",
    "task_valid_rate", "protocol_core_rate", "protocol_strict_rate",
    "invalid_decision_rate", "truncation_rate",
    "anomaly_recall", "normal_correct_rate", "normal_fpr", "balanced_accuracy",
    "anomaly_gated_miou", "acc_at_05", "mean_iou_f", "mean_iou_c", "mean_delta_refine",
    "candidate_box_valid_rate", "final_box_valid_rate",
    "mean_h_union_cov", "mean_iou_h_bestk",
)


def subsample_per_class(dataset, per_class, seed):
    """Pick a fixed number of normal + anomaly samples per class.

    ``per_class`` is an int (n normal + n anomaly per class) or a 2-tuple
    (n_normal, n_anomaly). Returns a new OutcomeDataset over the picked samples.
    """
    import random

    rng = random.Random(seed)
    buckets = {}
    for s in dataset.samples:
        meta = s.get("metadata") or {}
        cls = str(meta.get("class") or "object")
        key = "anomaly" if bool(meta.get("anomaly")) else "normal"
        buckets.setdefault(cls, {"anomaly": [], "normal": []})[key].append(s)

    if isinstance(per_class, (list, tuple)):
        n_norm, n_anom = int(per_class[0]), int(per_class[1])
    else:
        n_norm = n_anom = int(per_class)

    picked = []
    for cls in sorted(buckets):
        anoms = list(buckets[cls]["anomaly"]); rng.shuffle(anoms)
        norms = list(buckets[cls]["normal"]); rng.shuffle(norms)
        picked.extend(anoms[:n_anom])
        picked.extend(norms[:n_norm])
    rng.shuffle(picked)
    return OutcomeDataset(picked, dataset.cfg, dataset.processor, "eval")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft-dir", required=True)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--per-class", type=int, default=None,
                        help="subsample n normal + n anomaly per class (overrides --limit)")
    parser.add_argument("--save-viz", action="store_true",
                        help="save per-case heatmap/boxes/text under <out>/<name>_viz/")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="explicit checkpoint names (e.g. 1000 5000 final); default: all")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(int(cfg["training"]["seed"]))

    sft_dir = Path(args.sft_dir).resolve()
    if args.checkpoints:
        names = args.checkpoints
    else:
        cps = sorted(sft_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        names = [p.name for p in cps] + ["final"]

    dirs = [str(sft_dir) if n == "final" else str(sft_dir / n) for n in names]

    # Prime: load once to get processor/prior and build the fixed dev slice.
    cfg["outcome"]["sft_adapter"] = dirs[-1]
    model, processor, prior = load_model(cfg)
    train_set, dev_set, test_set = datasets(cfg, processor)
    eval_set = dev_set if args.split == "dev" else test_set
    if args.per_class is not None:
        eval_set = subsample_per_class(eval_set, args.per_class, int(cfg["training"]["seed"]))
        args.limit = len(eval_set)
        print(f"[per-class] subsampled {len(eval_set)} samples "
              f"({args.per_class} normal + {args.per_class} anomaly per class)", flush=True)

    out_dir = Path(args.out) if args.out else sft_dir / "_checkpoint_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, d in zip(names, dirs):
        cfg["outcome"]["sft_adapter"] = d
        started = time.perf_counter()
        print(f"\n=== {name} ({d}) ===", flush=True)
        model, processor, prior = load_model(cfg)
        stats = evaluate(cfg, model, processor, prior, eval_set,
                         out_dir / f"{name}.json", args.limit, namespace=args.split,
                         save_viz_dir=(out_dir / f"{name}_viz") if args.save_viz else None)
        row = {k: stats.get(k) for k in KEYS}
        row["seconds"] = round(time.perf_counter() - started, 1)
        results[name] = row
        with (out_dir / "_summary.json").open("w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
        # free GPU memory before next load
        del model
        import torch
        torch.cuda.empty_cache()

    print("\n================ SUMMARY ================", flush=True)
    def _fmt(v):
        if v is None:
            return "--"
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    cols = KEYS
    widths = [max(12, len(c)) for c in cols]
    header = "ckpt".ljust(12) + "".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header, flush=True)
    for name, row in results.items():
        cells = [_fmt(row.get(k)) for k in cols]
        line = name.ljust(12) + "".join(c.ljust(w) for c, w in zip(cells, widths))
        print(line, flush=True)


if __name__ == "__main__":
    main()
