#!/usr/bin/env python3
"""Validate whether the frozen-vision spatial prior H is useful — no retraining.

The question is split into an *input-prior validity* problem, not a model-learning
problem:

    1. H -> GT spatial consistency   (no LLM forward, just the frozen ViT + H)
       P_hit = frac(H points inside B_gt),  d(H, B_gt) = mean point->box distance,
       compared against a random-point baseline.

    2. Inference ablation on a fixed checkpoint. Same parameters, only the input
       changes:
           X1 = (I_ref, I_query, H)
           X2 = (I_ref, I_query)
           X3 = (I_ref, I_query, H_random)
           X4 = (I_ref, I_query, H_perturbed)   [mirrored H]
       Compare IoU(B_c, B_gt) and IoU(B_f, B_gt).

    3. Does the candidate box follow H?
       d(B_c, H) for the real-H run vs d(B_c, H_random), plus the center shift of
       B_c when H is perturbed.

Usage:
    # consistency only (fast, needs no checkpoint):
    CUDA_VISIBLE_DEVICES=4 python scripts/validate_prior_h.py \
        --config configs/qwen35_08b_grpo.yaml --experiments consistency

    # full ablation on a fixed LoRA checkpoint:
    CUDA_VISIBLE_DEVICES=4 python scripts/validate_prior_h.py \
        --config configs/qwen35_08b_grpo.yaml \
        --checkpoint outputs/train/qwen35_08b_prior/<run>/grpo_checkpoint-150 \
        --n-ablation 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from utils.config import load_yaml_config
from utils.common import set_seed

# 0-1000 system: max possible L2 distance between two points in the image.
_DIAG_1000 = math.sqrt(1000.0 ** 2 + 1000.0 ** 2)


# --------------------------------------------------------------------------- #
# small geometry helpers (all operate in the Qwen 0-1000 coordinate system)
# --------------------------------------------------------------------------- #
def point_box_dist(px: float, py: float, box: List[float]) -> float:
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.sqrt(dx * dx + dy * dy)


def mean_point_box_dist(points: List[List[int]], box: List[float]) -> float:
    if not points:
        return float("nan")
    return float(sum(point_box_dist(p[0], p[1], box) for p in points) / len(points))


def hit_rate(points: List[List[int]], box: List[float]) -> float:
    if not points:
        return 0.0
    x1, y1, x2, y2 = box
    hits = sum(1 for p in points if x1 <= p[0] <= x2 and y1 <= p[1] <= y2)
    return float(hits / len(points))


def centroid(points: List[List[int]]) -> List[float]:
    if not points:
        return [500.0, 500.0]
    return [
        float(sum(p[0] for p in points) / len(points)),
        float(sum(p[1] for p in points) / len(points)),
    ]


def box_center(box: List[float]) -> List[float]:
    return [0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])]


def l2(a: List[float], b: List[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def random_points(k: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    return [[rng.randint(0, 1000), rng.randint(0, 1000)] for _ in range(k)]


def mirror_points(points: List[List[int]]) -> List[List[int]]:
    return [[1000 - p[0], 1000 - p[1]] for p in points]


def _mean(xs) -> float:
    xs = [float(x) for x in xs]
    return float(sum(xs) / len(xs)) if xs else float("nan")


# --------------------------------------------------------------------------- #
# model / prior / data loading
# --------------------------------------------------------------------------- #
def load_model(cfg: dict, checkpoint: Optional[str], device: torch.device):
    from models.anomaly_prior import AnomalyPrior
    from models.qwen35 import setup_model_and_processor

    model, processor = setup_model_and_processor(
        cfg, for_inference=True, freeze_vision=True
    )
    if checkpoint:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
    model = model.to(device)
    model.eval()
    prior = AnomalyPrior.from_qwen(model, cfg)
    return model, processor, prior


def build_split_samples(cfg: dict, split: str) -> Tuple[List[dict], Dict[str, List[str]]]:
    from data.prior_dataset import build_train_ref_pool
    from data.scan import load_prior_split, split_holdout_by_class

    train_samples, test_samples = load_prior_split(cfg)
    holdout = float((cfg.get("data") or {}).get("holdout_ratio", 0.1) or 0.0)
    seed = int(cfg.get("training", {}).get("seed", 42))
    train_samples, dev_samples = split_holdout_by_class(train_samples, holdout, seed=seed)
    ref_pool = build_train_ref_pool(train_samples)
    split_samples = dev_samples if split == "dev" else test_samples
    return split_samples, ref_pool


# --------------------------------------------------------------------------- #
# generation under a controlled hint
# --------------------------------------------------------------------------- #
def tokenize_variant(
    processor,
    images: List[Any],
    prompt: str,
    hint_points: Optional[List[List[int]]],
    enable_thinking: bool,
    max_length: int,
) -> Tuple[dict, int]:
    from data.prior_dataset import apply_chat_template_safe
    from models.vision_cache import format_prior_hint

    if hint_points is None:
        user_text = prompt.rstrip()
    else:
        user_text = prompt.rstrip() + "\n\n" + format_prior_hint(hint_points)

    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": images[0]},
            {"type": "image", "image": images[1]},
            {"type": "text", "text": user_text},
        ],
    }
    text = apply_chat_template_safe(processor, [user], True, enable_thinking)
    try:
        enc = processor(
            text=[text], images=images, return_tensors="pt",
            truncation=True, max_length=max_length,
        )
    except TypeError:
        enc = processor(text=[text], images=images, return_tensors="pt")
    prompt_len = int(enc["input_ids"].shape[1])
    return enc, prompt_len


def generate_text(
    model,
    tok,
    enc: dict,
    prompt_len: int,
    image_embeds: Optional[torch.Tensor],
    device: torch.device,
    max_new: int,
    stop_criteria,
) -> str:
    from models.vision_cache import bind_cached_image_features
    from rl.grpo import model_inputs

    gen_in = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in model_inputs(enc).items()
    }
    kw = dict(max_new_tokens=max_new, do_sample=False, use_cache=True)
    with torch.no_grad(), bind_cached_image_features(model, image_embeds):
        if stop_criteria is not None:
            try:
                out = model.generate(**gen_in, **kw, stopping_criteria=stop_criteria)
            except TypeError:
                out = model.generate(**gen_in, **kw)
        else:
            try:
                out = model.generate(**gen_in, **kw, stop_strings=["</answer>"], tokenizer=tok)
            except TypeError:
                out = model.generate(**gen_in, **kw)
    if out.dim() == 1:
        out = out.unsqueeze(0)
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #
def run_consistency(samples: List[dict], collator, cfg: dict, base_seed: int) -> dict:
    """Experiment 1: does H itself carry the right spatial information?"""
    from reasoning.rewards import pixels_to_qwen1000

    rows = []
    for idx, item in enumerate(samples):
        if not bool(item.get("is_anomaly")):
            continue
        gt_box_px = item.get("gt_box_px")
        if gt_box_px is None:
            continue
        enc = collator._encode_one(item)
        real_points = enc["_meta"]["prior_points"]
        gt_1000 = pixels_to_qwen1000(gt_box_px, tuple(item["orig_size"]))

        k = len(real_points)
        rand_points = random_points(k, base_seed * 1000003 + idx)

        rows.append(
            {
                "idx": idx,
                "class_name": item.get("class_name"),
                "image_path": item.get("image_path"),
                "p_hit_real": hit_rate(real_points, gt_1000),
                "p_hit_rand": hit_rate(rand_points, gt_1000),
                "d_real": mean_point_box_dist(real_points, gt_1000),
                "d_rand": mean_point_box_dist(rand_points, gt_1000),
                "d_real_norm": mean_point_box_dist(real_points, gt_1000) / _DIAG_1000,
                "d_rand_norm": mean_point_box_dist(rand_points, gt_1000) / _DIAG_1000,
                "center_dist_real": l2(centroid(real_points), box_center(gt_1000)),
                "center_dist_rand": l2(centroid(rand_points), box_center(gt_1000)),
                "k": k,
            }
        )

    if not rows:
        return {"n_anom": 0, "rows": []}

    n = len(rows)
    summary = {
        "n_anom": n,
        "mean_p_hit_real": _mean(r["p_hit_real"] for r in rows),
        "mean_p_hit_rand": _mean(r["p_hit_rand"] for r in rows),
        "frac_hit_real_better": _mean(r["p_hit_real"] > r["p_hit_rand"] for r in rows),
        "mean_d_real": _mean(r["d_real"] for r in rows),
        "mean_d_rand": _mean(r["d_rand"] for r in rows),
        "mean_d_real_norm": _mean(r["d_real_norm"] for r in rows),
        "mean_d_rand_norm": _mean(r["d_rand_norm"] for r in rows),
        "frac_dist_real_better": _mean(r["d_real"] < r["d_rand"] for r in rows),
        "mean_center_dist_real": _mean(r["center_dist_real"] for r in rows),
        "mean_center_dist_rand": _mean(r["center_dist_rand"] for r in rows),
    }
    # per-class breakdown
    by_cls: Dict[str, List[dict]] = {}
    for r in rows:
        by_cls.setdefault(r["class_name"], []).append(r)
    summary["per_class"] = {
        cls: {
            "n": len(rs),
            "mean_p_hit_real": _mean(r["p_hit_real"] for r in rs),
            "mean_d_real_norm": _mean(r["d_real_norm"] for r in rs),
        }
        for cls, rs in sorted(by_cls.items())
    }
    return {"summary": summary, "rows": rows}


def run_ablation(
    samples: List[dict],
    collator,
    model,
    processor,
    cfg: dict,
    conditions: List[str],
    n_max: int,
    base_seed: int,
    stop_criteria,
) -> dict:
    """Experiments 2 & 3: does the model actually *use* H?"""
    from models.vision_cache import format_prior_hint
    from reasoning.parser import parse_cot_output
    from reasoning.rewards import box_iou, qwen1000_to_pixels_strict

    inf = cfg.get("inference") or {}
    gcfg = cfg.get("grpo") or {}
    max_new = int(gcfg.get("max_new_tokens", inf.get("max_new_tokens", 384)))
    max_length = int(cfg.get("training", {}).get("max_length", 2048))
    enable_thinking = bool(cfg.get("prompt", {}).get("enable_thinking", False))
    tok = getattr(processor, "tokenizer", processor)
    device = next(model.parameters()).device

    rows = []
    done = 0
    for idx, item in enumerate(samples):
        if not bool(item.get("is_anomaly")):
            continue
        gt_box_px = item.get("gt_box_px")
        if gt_box_px is None:
            continue
        if done >= n_max:
            break
        done += 1

        # Real-H encoding: one vision pass supplies merged embeddings + H.
        batch = collator([item])
        meta = batch["_meta"][0]
        images = [meta["ref"], meta["test"]]
        real_points = meta["prior_points"]
        image_embeds = batch["image_embeds"]
        real_prompt_len = int(batch["prompt_len"][0].item())

        k = len(real_points)
        rand_points = random_points(k, base_seed * 1000003 + idx)
        perturb_points = mirror_points(real_points)

        hint_map = {
            "real": real_points,
            "none": None,
            "random": rand_points,
            "perturb": perturb_points,
        }

        per_cond: Dict[str, dict] = {}
        for cond in conditions:
            if cond == "real":
                enc, plen = batch, real_prompt_len
            else:
                enc, plen = tokenize_variant(
                    processor, images, item["prompt"], hint_map[cond],
                    enable_thinking, max_length,
                )
            text = generate_text(
                model, tok, enc, plen, image_embeds, device, max_new, stop_criteria
            )
            parsed = parse_cot_output(text)
            cand = parsed.get("candidate_bbox_2d")
            final = parsed.get("bbox_2d")
            iou_c = box_iou(qwen1000_to_pixels_strict(cand, tuple(item["orig_size"])), gt_box_px) if cand else 0.0
            iou_f = box_iou(qwen1000_to_pixels_strict(final, tuple(item["orig_size"])), gt_box_px) if final else 0.0
            per_cond[cond] = {
                "candidate_bbox_2d": cand,
                "bbox_2d": final,
                "pred_is_anomaly": parsed.get("is_anomaly"),
                "trajectory_valid": bool(parsed.get("trajectory_valid")),
                "iou_c": iou_c,
                "iou_f": iou_f,
                "cand_center": box_center(cand) if cand else None,
                "text": text,
            }

        def _d_cand_to_points(cond: str, points: List[List[int]]) -> Optional[float]:
            cand = per_cond[cond]["candidate_bbox_2d"]
            return mean_point_box_dist(points, cand) if cand else None

        row = {
            "idx": idx,
            "class_name": item.get("class_name"),
            "image_path": item.get("image_path"),
            "conditions": per_cond,
            # does B_c (real run) sit closer to the real H than to random points?
            "d_cand_to_realH": _d_cand_to_points("real", real_points),
            "d_cand_to_randH": _d_cand_to_points("real", rand_points),
            "follow_real": _d_cand_to_points("real", real_points),
            "follow_rand": _d_cand_to_points("random", rand_points),
            "follow_perturb": _d_cand_to_points("perturb", perturb_points),
            "center_shift_real_vs_rand": (
                l2(per_cond["real"]["cand_center"], per_cond["random"]["cand_center"])
                if per_cond["real"]["cand_center"] is not None
                and per_cond["random"]["cand_center"] is not None
                else None
            ),
            "center_shift_real_vs_perturb": (
                l2(per_cond["real"]["cand_center"], per_cond["perturb"]["cand_center"])
                if per_cond["real"]["cand_center"] is not None
                and per_cond["perturb"]["cand_center"] is not None
                else None
            ),
        }
        rows.append(row)

    if not rows:
        return {"n": 0, "summary": {}, "rows": []}

    n = len(rows)

    def _cond_mean(cond: str, key: str):
        vals = [r["conditions"][cond][key] for r in rows if cond in r["conditions"]]
        return _mean(vals)

    def _cond_valid_rate(cond: str, key: str):
        vals = [1.0 if r["conditions"][cond][key] else 0.0 for r in rows if cond in r["conditions"]]
        return _mean(vals)

    def _paired_frac(better: str, worse: str):
        cnt = 0
        tot = 0
        for r in rows:
            a = r["conditions"].get(better, {}).get("iou_f", 0.0)
            b = r["conditions"].get(worse, {}).get("iou_f", 0.0)
            tot += 1
            if a > b:
                cnt += 1
        return float(cnt / max(tot, 1))

    summary: Dict[str, Any] = {"n": n}
    for cond in conditions:
        summary[f"iou_c_{cond}"] = _cond_mean(cond, "iou_c")
        summary[f"iou_f_{cond}"] = _cond_mean(cond, "iou_f")
        summary[f"traj_valid_{cond}"] = _cond_valid_rate(cond, "trajectory_valid")
        summary[f"pred_anomaly_{cond}"] = _cond_valid_rate(cond, "pred_is_anomaly")

    summary["frac_iou_f_real_gt_none"] = _paired_frac("real", "none")
    summary["frac_iou_f_real_gt_random"] = _paired_frac("real", "random")
    summary["frac_iou_f_real_gt_perturb"] = _paired_frac("real", "perturb")
    summary["mean_iou_f_delta_real_minus_none"] = (
        _cond_mean("real", "iou_f") - _cond_mean("none", "iou_f")
    )
    summary["mean_iou_f_delta_real_minus_random"] = (
        _cond_mean("real", "iou_f") - _cond_mean("random", "iou_f")
    )

    # candidate-follows-H
    d_real = [r["d_cand_to_realH"] for r in rows if r["d_cand_to_realH"] is not None]
    d_rand = [r["d_cand_to_randH"] for r in rows if r["d_cand_to_randH"] is not None]
    summary["mean_d_cand_to_realH"] = _mean(d_real)
    summary["mean_d_cand_to_randH"] = _mean(d_rand)
    summary["frac_cand_closer_to_realH"] = _mean(
        (r["d_cand_to_realH"] is not None and r["d_cand_to_randH"] is not None)
        and (r["d_cand_to_realH"] < r["d_cand_to_randH"])
        for r in rows
    )
    summary["mean_follow_real"] = _mean(
        r["follow_real"] for r in rows if r["follow_real"] is not None
    )
    summary["mean_follow_rand"] = _mean(
        r["follow_rand"] for r in rows if r["follow_rand"] is not None
    )
    summary["mean_follow_perturb"] = _mean(
        r["follow_perturb"] for r in rows if r["follow_perturb"] is not None
    )
    shift_rand = [r["center_shift_real_vs_rand"] for r in rows if r["center_shift_real_vs_rand"] is not None]
    shift_pert = [r["center_shift_real_vs_perturb"] for r in rows if r["center_shift_real_vs_perturb"] is not None]
    summary["mean_center_shift_real_vs_random"] = _mean(shift_rand)
    summary["mean_center_shift_real_vs_perturb"] = _mean(shift_pert)

    return {"summary": summary, "rows": rows}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate the spatial prior H without retraining")
    p.add_argument("--config", type=str, default="configs/qwen35_08b_grpo.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="LoRA checkpoint dir (adapter_config.json). Required for ablation.")
    p.add_argument("--split", type=str, default="test", choices=["test", "dev"])
    p.add_argument("--experiments", type=str, default="all",
                   choices=["all", "consistency", "ablation"])
    p.add_argument("--conditions", type=str, default="real,none,random,perturb",
                   help="comma-separated subset of real,none,random,perturb")
    p.add_argument("--n-ablation", type=int, default=100)
    p.add_argument("--n-consistency", type=int, default=100000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None,
                   help="output json dir (default outputs/prior_h_validation/<run_tag>)")
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    cfg.setdefault("runtime", {})["mode"] = "train"
    cfg.setdefault("distributed", {})["num_gpu"] = 1
    set_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[validate_H] device={device} config={args.config}", flush=True)

    model, processor, prior = load_model(cfg, args.checkpoint, device)
    tok = getattr(processor, "tokenizer", processor)

    try:
        from transformers.generation.stopping_criteria import StopStringCriteria, StoppingCriteriaList
        stop_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer=tok, stop_strings=["</answer>"])])
    except Exception:
        stop_criteria = None

    from data.prior_dataset import PriorCoTDataset, PriorCollator

    split_samples, ref_pool = build_split_samples(cfg, args.split)
    n_anom = sum(1 for s in split_samples if bool((s.get("metadata") or {}).get("anomaly")))
    print(f"[validate_H] split={args.split} total={len(split_samples)} anomaly={n_anom}", flush=True)

    # VisA-dev reuses the VisA normal ref pool; MVTec-test must scan its own
    # train/good dirs (ref_pool=None), exactly like trainer.train_main does.
    pool_for_ds = ref_pool if args.split == "dev" else None

    # Filter anomaly samples from metadata first (no image I/O), then only load
    # the number of items the requested experiments actually need.
    anom_samples = [
        s for s in split_samples if bool((s.get("metadata") or {}).get("anomaly"))
    ]
    needed = 0
    if args.experiments in ("all", "consistency"):
        needed = max(needed, args.n_consistency)
    if args.experiments in ("all", "ablation"):
        needed = max(needed, args.n_ablation)
    n_load = min(len(anom_samples), needed) if needed > 0 else len(anom_samples)

    ds = PriorCoTDataset(anom_samples[:n_load], cfg, processor, mode="eval", ref_pool=pool_for_ds)
    collator = PriorCollator(processor, prior, cfg)
    anom_items = [ds[i] for i in range(len(ds))]

    out_dir = args.out or os.path.join(
        PROJECT_ROOT, "outputs", "prior_h_validation",
        f"{os.path.basename(os.path.dirname(args.checkpoint)) if args.checkpoint else 'base'}"
        f"_{args.split}",
    )
    os.makedirs(out_dir, exist_ok=True)

    result: Dict[str, Any] = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "device": str(device),
        "conditions": [c for c in args.conditions.split(",") if c],
    }

    if args.experiments in ("all", "consistency"):
        print("[validate_H] Experiment 1: H vs GT consistency ...", flush=True)
        t0 = time.time()
        cons = run_consistency(anom_items[: args.n_consistency], collator, cfg, args.seed)
        result["consistency"] = cons
        with open(os.path.join(out_dir, "consistency.json"), "w", encoding="utf-8") as f:
            json.dump(cons, f, ensure_ascii=False, indent=2)
        print(f"[validate_H] consistency done in {time.time() - t0:.1f}s", flush=True)

    if args.experiments in ("all", "ablation"):
        if args.checkpoint is None:
            print("[validate_H] WARNING: no --checkpoint; ablation uses the base model "
                  "(no LoRA). Set --checkpoint for a trained policy.", flush=True)
        conditions = [c for c in args.conditions.split(",") if c]
        print(f"[validate_H] Experiment 2/3: ablation conditions={conditions} ...", flush=True)
        t0 = time.time()
        abl = run_ablation(
            anom_items, collator, model, processor, cfg, conditions,
            args.n_ablation, args.seed, stop_criteria,
        )
        result["ablation"] = abl
        with open(os.path.join(out_dir, "ablation.json"), "w", encoding="utf-8") as f:
            json.dump(abl, f, ensure_ascii=False, indent=2)
        print(f"[validate_H] ablation done in {time.time() - t0:.1f}s", flush=True)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---- readable report ----
    print("\n" + "=" * 78, flush=True)
    print("PRIOR H VALIDATION SUMMARY", flush=True)
    print("=" * 78, flush=True)
    if "consistency" in result and result["consistency"].get("summary"):
        s = result["consistency"]["summary"]
        print("[Experiment 1] H vs GT spatial consistency", flush=True)
        print(f"  n_anom            = {s['n_anom']}", flush=True)
        print(f"  P_hit  real/rand  = {s['mean_p_hit_real']:.3f} / {s['mean_p_hit_rand']:.3f} "
              f"(frac real better: {s['frac_hit_real_better']:.3f})", flush=True)
        print(f"  d(H,GT) real/rand = {s['mean_d_real_norm']:.4f} / {s['mean_d_rand_norm']:.4f} "
              f"(frac real closer: {s['frac_dist_real_better']:.3f})", flush=True)
        print(f"  center_dist rl/rd = {s['mean_center_dist_real']:.1f} / {s['mean_center_dist_rand']:.1f}", flush=True)
    if "ablation" in result and result["ablation"].get("summary"):
        s = result["ablation"]["summary"]
        print("\n[Experiment 2] Inference ablation (greedy, fixed weights)", flush=True)
        for cond in result["conditions"]:
            print(f"  IoU_c/IoU_f [{cond:<8}] = {s.get(f'iou_c_{cond}', float('nan')):.3f} / "
                  f"{s.get(f'iou_f_{cond}', float('nan')):.3f} "
                  f"(traj_valid {s.get(f'traj_valid_{cond}', 0):.2f})", flush=True)
        print(f"  frac IoU_f(real>none)    = {s['frac_iou_f_real_gt_none']:.3f}", flush=True)
        print(f"  frac IoU_f(real>random)  = {s['frac_iou_f_real_gt_random']:.3f}", flush=True)
        print(f"  frac IoU_f(real>perturb) = {s['frac_iou_f_real_gt_perturb']:.3f}", flush=True)
        print(f"  dIoU_f real-none         = {s['mean_iou_f_delta_real_minus_none']:+.3f}", flush=True)
        print(f"  dIoU_f real-random       = {s['mean_iou_f_delta_real_minus_random']:+.3f}", flush=True)
        print("\n[Experiment 3] Does the candidate follow H?", flush=True)
        print(f"  d(B_c, realH)  = {s['mean_d_cand_to_realH']:.1f}  (0-1000)", flush=True)
        print(f"  d(B_c, randH)  = {s['mean_d_cand_to_randH']:.1f}  "
              f"frac closer-to-realH: {s['frac_cand_closer_to_realH']:.3f}", flush=True)
        print(f"  follow dist real/rand/perturb = {s['mean_follow_real']:.1f} / "
              f"{s['mean_follow_rand']:.1f} / {s['mean_follow_perturb']:.1f}", flush=True)
        print(f"  center shift real->random  = {s['mean_center_shift_real_vs_random']:.1f}", flush=True)
        print(f"  center shift real->perturb = {s['mean_center_shift_real_vs_perturb']:.1f}", flush=True)
    print("=" * 78, flush=True)
    print(f"results saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
