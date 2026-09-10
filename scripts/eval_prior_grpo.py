#!/usr/bin/env python3
"""Standalone MVTec test evaluation for a legacy prior-GRPO checkpoint (port 5002 pipeline).

The `train.py` (qwen35_2b_grpo.yaml) pipeline only evals VisA-dev during training and
runs the full MVTec test once at the very end. If training was interrupted before that
final eval, this script reproduces it for an arbitrary `grpo_checkpoint-N`.

Usage:
  CUDA_VISIBLE_DEVICES=7 python scripts/eval_prior_grpo.py \
      --config configs/qwen35_2b_grpo.yaml \
      --ckpt outputs/train/qwen35_2b_prior/qwen35_2b_visa2mvtec_20260905_210255/grpo_checkpoint-2650
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import random

import torch
from torch.utils.data import DataLoader

from utils.config import load_yaml_config
from utils.common import set_seed


def stratified_subset(samples, per_class: int, seed: int = 42) -> list:
    """Balanced normal/anomaly per class, up to `per_class` each."""
    rng = random.Random(seed)
    by_class: dict = {}
    for s in samples:
        cls = str((s.get("metadata") or {}).get("class") or "unknown")
        by_class.setdefault(cls, {"normal": [], "anomaly": []})
        key = "anomaly" if (s.get("metadata") or {}).get("anomaly") else "normal"
        by_class[cls][key].append(s)
    out = []
    for cls in sorted(by_class):
        for key in ("normal", "anomaly"):
            items = by_class[cls][key]
            rng.shuffle(items)
            out.extend(items[: per_class])
    return out


def _disable_hf_datasets_check() -> None:
    try:
        import transformers.trainer as hf_trainer
        hf_trainer.is_datasets_available = lambda: False  # type: ignore[assignment]
    except Exception:
        pass
    try:
        import transformers.utils.import_utils as import_utils
        import_utils._datasets_available = False  # type: ignore[attr-defined]
    except Exception:
        pass


def load_prior_checkpoint(cfg: dict, ckpt_path: str):
    from peft import PeftModel
    from models.qwen35 import setup_model_and_processor, freeze_vision_encoder, force_vision_eval
    from models.anomaly_prior import AnomalyPrior

    model, processor = setup_model_and_processor(cfg, for_inference=True, freeze_vision=False)
    model = PeftModel.from_pretrained(model, ckpt_path, is_trainable=False)
    if bool(cfg.get("model", {}).get("freeze_vit", True)):
        freeze_vision_encoder(model)
        force_vision_eval(model)
    prior = AnomalyPrior.from_qwen(model, cfg)
    return model, processor, prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen35_2b_grpo.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="grpo_checkpoint-N dir")
    parser.add_argument("--max-samples", type=int, default=None, help="None = full MVTec test")
    parser.add_argument("--per-class", type=int, default=None, help="stratified: N normal + N anomaly per class")
    parser.add_argument("--out", type=str, default=None, help="output JSON path for summary")
    args = parser.parse_args()

    _disable_hf_datasets_check()
    cfg = load_yaml_config(args.config)
    set_seed(int(cfg.get("training", {}).get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        print(f"[eval] device={torch.cuda.get_device_name(0)}", flush=True)

    print(f"[eval] loading checkpoint {args.ckpt}", flush=True)
    model, processor, prior = load_prior_checkpoint(cfg, args.ckpt)
    model = model.to(device)
    model.eval()
    print("[eval] model ready", flush=True)

    from data.scan import load_prior_split
    from data.prior_dataset import PriorCoTDataset, PriorCollator
    from evaluation.evaluator import run_simple_eval

    train_samples, test_samples = load_prior_split(cfg)
    n_anom = sum(1 for s in test_samples if bool((s.get("metadata") or {}).get("anomaly")))
    print(f"[eval] MVTec test: {len(test_samples)} samples (anom={n_anom})", flush=True)

    if args.per_class is not None:
        test_samples = stratified_subset(test_samples, args.per_class, seed=int(cfg.get("training", {}).get("seed", 42)))
        print(f"[eval] stratified subset: {len(test_samples)} samples (per-class {args.per_class})", flush=True)
    elif args.max_samples is not None:
        test_samples = test_samples[: args.max_samples]
        print(f"[eval] capped to {len(test_samples)} samples", flush=True)

    collator = PriorCollator(processor, prior, cfg)
    test_set = PriorCoTDataset(test_samples, cfg, processor, mode="eval")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)

    # n_max must be explicit: run_simple_eval falls back to training.eval_num_samples
    # (32) when n_max=None, which would silently truncate a full/subset eval.
    stats = run_simple_eval(
        cfg,
        model,
        processor,
        test_loader,
        writer=None,
        global_step=0,
        n_max=len(test_samples),
    )

    compact = {k: v for k, v in stats.items() if k != "records"}
    print("\n===== MVTec test metrics =====", flush=True)
    for k, v in compact.items():
        if isinstance(v, float):
            print(f"{k:30s} {v:.4f}", flush=True)
        else:
            print(f"{k:30s} {v}", flush=True)

    out_path = args.out or os.path.join(args.ckpt, "mvtec_test_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)
    with open(os.path.splitext(out_path)[0] + "_records.json", "w", encoding="utf-8") as f:
        json.dump(stats.get("records", []), f, ensure_ascii=False, indent=2)
    print(f"\n[eval] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
