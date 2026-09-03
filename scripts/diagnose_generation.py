#!/usr/bin/env python3
"""Generation-only diagnostic: no optimizer steps, just rollout + parse + reward stats.

Separates "parser bug" from "model never learned the format": for N samples, generate
with the *training* sampling config (do_sample=True, temperature/top_p), then report
protocol / trajectory / raw-IoU breakdown and dump every invalid rollout's repr.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/diagnose_generation.py \
        --config configs/qwen35_08b_grpo.yaml --n 40 [--split dev|test] [--group 1]
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
from utils.common import open_rollout_log, rollout_log, close_rollout_log, set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/qwen35_08b_grpo.yaml")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    p.add_argument("--group", type=int, default=1)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--max-invalid-dump", type=int, default=20)
    args = p.parse_args()

    cfg = load_yaml_config(args.config)
    cfg.setdefault("runtime", {})["mode"] = "train"
    cfg.setdefault("distributed", {})["num_gpu"] = 1
    set_seed(int(cfg.get("training", {}).get("seed", 42)))

    # Import the training pieces lazily so the script stays import-safe without a GPU.
    from data.prior_dataset import PriorCollator, PriorCoTDataset, build_train_ref_pool
    from data.scan import load_prior_split, split_holdout_by_class
    from models.qwen35 import force_vision_eval
    from models.vision_cache import bind_cached_image_features
    from reasoning.parser import parse_cot_output, rollout_protocol_stats
    from reasoning.rewards import compute_rewards
    from rl.grpo import model_inputs, move_batch, unwrap_model
    from rl.trainer import setup_prior_model

    out_dir = cfg["paths"].get("output_dir") or "./outputs/diagnose"
    out_path = args.out or os.path.join(out_dir, "diagnose_generation.txt")
    open_rollout_log(out_path)

    train_samples, test_samples = load_prior_split(cfg)
    holdout = float((cfg.get("data") or {}).get("holdout_ratio", 0.1) or 0.0)
    train_samples, dev_samples = split_holdout_by_class(
        train_samples, holdout, seed=int(cfg.get("training", {}).get("seed", 42))
    )
    train_ref_pool = build_train_ref_pool(train_samples)
    split_samples = dev_samples if args.split == "dev" else test_samples

    n_anom = sum(1 for s in split_samples if bool((s.get("metadata") or {}).get("anomaly")))
    print(f"[diag] split={args.split} total={len(split_samples)} anomaly={n_anom}", flush=True)

    model, processor, prior = setup_prior_model(cfg)
    model = model.cuda()
    model.eval()
    force_vision_eval(model)
    tok = getattr(processor, "tokenizer", processor)

    ds = PriorCoTDataset(split_samples, cfg, processor, mode="eval", ref_pool=train_ref_pool)
    collator = PriorCollator(processor, prior, cfg)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)

    gcfg = cfg.get("grpo") or {}
    max_new = int(gcfg.get("max_new_tokens", cfg.get("inference", {}).get("max_new_tokens", 384)))
    temperature = float(gcfg.get("temperature", 0.9))
    top_p = float(gcfg.get("top_p", 0.95))

    try:
        from transformers.generation.stopping_criteria import StopStringCriteria, StoppingCriteriaList

        stop_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer=tok, stop_strings=["</answer>"])])
    except Exception:
        stop_criteria = None

    parsed_list = []
    details = []
    metas = []
    texts = []
    invalid_dumped = 0

    for batch in loader:
        if len(parsed_list) >= args.n:
            break
        device = next(model.parameters()).device
        batch = move_batch(batch, device)
        gen_in = model_inputs(batch)
        prompt_len = int(batch["prompt_len"][0].item())
        meta = batch["_meta"][0]
        kw = dict(
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=max(int(args.group), 1),
            use_cache=True,
        )
        with torch.no_grad(), bind_cached_image_features(model, batch.get("image_embeds")):
            try:
                if stop_criteria is not None:
                    out = model.generate(**gen_in, **kw, stopping_criteria=stop_criteria)
                else:
                    out = model.generate(**gen_in, **kw, stop_strings=["</answer>"], tokenizer=tok)
            except TypeError:
                out = model.generate(**gen_in, **kw)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        for i in range(int(out.shape[0])):
            if len(parsed_list) >= args.n:
                break
            seq = out[i]
            text = tok.decode(seq[prompt_len:], skip_special_tokens=True)
            parsed = parse_cot_output(text)
            det = compute_rewards(
                parsed,
                meta.get("gt_box_px"),
                tuple(meta["orig_size"]),
                bool(meta["is_anomaly"]),
                cfg,
            )
            parsed_list.append(parsed)
            details.append(det)
            metas.append(meta)
            texts.append(text)
            if not parsed.get("trajectory_valid") and invalid_dumped < args.max_invalid_dump:
                invalid_dumped += 1
                print(f"\n===== PARSER DEBUG (invalid #{invalid_dumped}) =====", flush=True)
                print(f"RAW_REPR: {text!r}", flush=True)
                print(f"tags: {list((parsed.get('tags') or {}).keys())}", flush=True)
                print(
                    f"candidate_state={parsed.get('candidate_bbox_state')} answer_state={parsed.get('answer_state')} "
                    f"final_state={parsed.get('final_bbox_state')} pred={parsed.get('is_anomaly')} "
                    f"bbox={parsed.get('bbox_2d')} desc_ok={parsed.get('description_ok')} "
                    f"prose_ok={parsed.get('prose_ok')}",
                    flush=True,
                )
                print("===============================================\n", flush=True)

    proto = rollout_protocol_stats(parsed_list, texts)
    n = max(len(details), 1)

    def _mean(key):
        return sum(float(d.get(key, 0.0)) for d in details) / n

    anom_idx = [i for i, m in enumerate(metas) if bool(m["is_anomaly"])]
    anom_det = [details[i] for i in anom_idx]
    na = max(len(anom_det), 1)
    mean_raw_iou_f = sum(float(d.get("raw_iou_f", 0.0)) for d in anom_det) / na
    mean_raw_iou_c = sum(float(d.get("raw_iou_c", 0.0)) for d in anom_det) / na

    ans_tag_rate = sum(1 for p in parsed_list if "answer" in (p.get("tags") or {})) / n

    report = [
        f"===== diagnose_generation  config={args.config} split={args.split} n={n} group={args.group} =====",
        f"ans_tag_rate          = {ans_tag_rate:.3f}",
        f"protocol_rate         = {proto.get('protocol_rate', 0.0):.3f}",
        f"trajectory_valid_rate = {proto.get('trajectory_valid_rate', 0.0):.3f}",
        f"candidate_valid_rate  = {proto.get('candidate_valid_rate', 0.0):.3f}",
        f"final_valid_rate      = {proto.get('final_valid_rate', 0.0):.3f}",
        f"box_pair_valid_rate   = {proto.get('box_pair_valid_rate', 0.0):.3f}",
        f"normal_null_rate      = {proto.get('normal_null_consistency_rate', 0.0):.3f}",
        f"unique_response_rate  = {proto.get('unique_response_rate', 0.0):.3f}",
        f"mean_R_ground         = {_mean('R_ground'):.3f}",
        f"mean_R_reason         = {_mean('R_reason'):.3f}",
        f"mean_R_dir            = {_mean('R_dir'):.3f}",
        f"mean_R_final          = {_mean('R_final'):.3f}",
        f"mean_R_fmt            = {_mean('R_fmt'):.3f}",
        f"mean_raw_iou_c (anom) = {mean_raw_iou_c:.3f}",
        f"mean_raw_iou_f (anom) = {mean_raw_iou_f:.3f}",
        f"invalid_rollouts_dumped={invalid_dumped}",
    ]
    print("\n".join(report), flush=True)
    rollout_log("\n".join(report))
    close_rollout_log()
    print(f"[diag] full rollout log -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
