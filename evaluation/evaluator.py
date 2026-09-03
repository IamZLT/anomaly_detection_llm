"""MVTec / VisA CoT evaluation and GRPO probes."""

from __future__ import annotations

import json
import os
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from evaluation.metrics import box_ious_from_parsed, classification_correct, is_truncated
from reasoning.parser import parse_cot_output
from reasoning.rewards import box_iou
from rl.grpo import model_inputs, move_batch, unwrap_model
from utils.common import qwen_norm1000_to_original_pixels, train_log
from visualization.tensorboard import log_heatmap_and_case


def tb_vis_flags(cfg: dict) -> dict:
    tb = cfg.get("tensorboard") or {}
    return {
        "log_heatmap": bool(tb.get("log_heatmap", True)),
        "log_case": bool(tb.get("log_case", True)),
    }


def first_anomaly_item(ds) -> Optional[dict]:
    if ds is None or len(ds) == 0:
        return None
    for i in range(len(ds)):
        cand = ds[i]
        if cand.get("is_anomaly"):
            return cand
    return ds[0]


@torch.no_grad()
def run_simple_eval(
    cfg: dict,
    model,
    processor,
    loader: DataLoader,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
    n_max: Optional[int] = None,
    log_images: bool = True,
) -> dict:
    model.eval()
    inf = cfg.get("inference") or {}
    tb = cfg.get("tensorboard") or {}
    vis_n = int(tb.get("vis_num_samples", 1))
    alpha = float((cfg.get("prior") or {}).get("overlay_alpha", 0.45))
    rec_ok = 0
    ious: List[float] = []
    ious_c: List[float] = []
    n_anom = 0
    n_hit = 0
    parse_ok = 0
    records = []
    if n_max is None:
        n_max = (cfg.get("training") or {}).get("eval_num_samples", 8)
    if n_max in (None, "null", "None", ""):
        n_max = 10**9
    else:
        n_max = int(n_max)
    seen = 0
    tok = getattr(processor, "tokenizer", processor)
    for batch in loader:
        if seen >= n_max:
            break
        device = next(model.parameters()).device
        batch = move_batch(batch, device)
        gen_in = model_inputs(batch)
        outputs = model.generate(
            **gen_in,
            max_new_tokens=int(inf.get("max_new_tokens", 512)),
            temperature=float(inf.get("temperature", 0.0)),
            top_p=float(inf.get("top_p", 0.9)),
            do_sample=bool(inf.get("do_sample", False)),
        )
        prompt_len = int(batch["prompt_len"][0].item()) if "prompt_len" in batch else int(batch["input_ids"].shape[1])
        text = tok.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        meta = batch["_meta"][0]
        parsed = parse_cot_output(text)
        if parsed.get("has_tags"):
            parse_ok += 1
        orig = tuple(meta["orig_size"])
        is_anom = bool(meta["is_anomaly"])
        pred_cls = parsed.get("is_anomaly", None)
        pred_box = parsed.get("bbox_2d")
        ok = classification_correct(pred_cls, is_anom)
        rec_ok += int(ok)
        iou_v = 0.0
        iou_c = 0.0
        pred_px = None
        if is_anom:
            n_anom += 1
            gt = meta.get("gt_box_px")
            if pred_box is not None and gt is not None:
                pred_px = qwen_norm1000_to_original_pixels(pred_box, orig)
                iou_v = box_iou(pred_px, gt)
                ious.append(iou_v)
                if iou_v >= 0.3:
                    n_hit += 1
            else:
                ious.append(0.0)
            cand_box = parsed.get("candidate_bbox")
            if cand_box is not None and gt is not None:
                iou_c = box_iou(qwen_norm1000_to_original_pixels(cand_box, orig), gt)
            ious_c.append(iou_c)
        if writer is not None and log_images and seen < vis_n:
            vis_dir = os.path.join(
                str((cfg.get("paths") or {}).get("output_dir") or "."),
                "eval_steps",
                f"vis_step{int(global_step):08d}",
            )
            log_heatmap_and_case(
                writer,
                step=int(global_step),
                tag_prefix=f"eval_case_{seen}",
                meta=meta,
                response=text,
                parsed=parsed,
                iou=float(iou_v),
                rec_ok=bool(ok),
                iou_c=float(iou_c),
                overlay_alpha=alpha,
                save_dir=vis_dir,
                **tb_vis_flags(cfg),
            )
        records.append(
            {
                "image_path": meta.get("image_path"),
                "is_anomaly": is_anom,
                "pred_anomaly": pred_cls,
                "iou": iou_v,
                "iou_c": iou_c,
                "pred_box": pred_px,
                "response": text[:2000],
                "parsed": parsed.get("raw"),
            }
        )
        seen += 1
    n = max(seen, 1)
    mean_iou = float(sum(ious) / len(ious)) if ious else 0.0
    mean_iou_c = float(sum(ious_c) / len(ious_c)) if ious_c else 0.0
    if writer is not None:
        writer.add_scalar("eval/rec_acc", rec_ok / n, global_step)
        writer.add_scalar("eval/iou_at_03", (n_hit / n_anom) if n_anom else 0.0, global_step)
        writer.add_scalar("eval/mean_iou", mean_iou, global_step)
        writer.add_scalar("eval/mean_iou_c", mean_iou_c, global_step)
        writer.add_scalar("eval/json_parse_rate", parse_ok / n, global_step)
        writer.flush()
    model.train()
    return {
        "n": seen,
        "rec_acc": rec_ok / n,
        "iou_at_03": (n_hit / n_anom) if n_anom else 0.0,
        "mean_iou": mean_iou,
        "mean_iou_c": mean_iou_c,
        "json_parse_rate": parse_ok / n,
        "records": records,
    }


def run_final_mvtec_eval(cfg, model, processor, eval_loader, writer, output_dir, tag: str) -> dict:
    n_max = (cfg.get("training") or {}).get("final_eval_num_samples")
    stats = run_simple_eval(
        cfg,
        unwrap_model(model),
        processor,
        eval_loader,
        writer=writer,
        global_step=0,
        n_max=n_max,
    )
    train_log(
        f"[final-{tag}] MVTec n={stats['n']} rec={stats['rec_acc']:.3f} "
        f"iou03={stats['iou_at_03']:.3f} mean_iou={stats['mean_iou']:.3f}",
        main_only=True,
    )
    os.makedirs(os.path.join(output_dir, "eval_steps"), exist_ok=True)
    compact = {k: v for k, v in stats.items() if k != "records"}
    with open(os.path.join(output_dir, "eval_steps", f"final_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "eval_steps", f"final_{tag}_records.json"), "w", encoding="utf-8") as f:
        json.dump(stats["records"], f, ensure_ascii=False, indent=2)
    return stats


@torch.no_grad()
def log_grpo_probe_cot(
    cfg: dict,
    model,
    processor,
    collator,
    probe_item: dict,
    writer: Optional[SummaryWriter],
    step: int,
    *,
    max_new: int,
    alpha: float,
    tag_prefix: str = "crossdomain_probe",
) -> None:
    if writer is None or probe_item is None:
        return
    device = next(model.parameters()).device
    batch = collator([probe_item])
    batch = move_batch(batch, device)
    gen_in = model_inputs(batch)
    inf = cfg.get("inference") or {}
    was_training = model.training
    model.eval()
    outputs = model.generate(
        **gen_in,
        max_new_tokens=max_new,
        do_sample=False,
        temperature=0.0,
        top_p=float(inf.get("top_p", 0.9)),
    )
    tok = getattr(processor, "tokenizer", processor)
    prompt_len = int(batch["prompt_len"][0].item())
    seq = outputs[0]
    text = tok.decode(seq[prompt_len:], skip_special_tokens=True)
    meta = batch["_meta"][0]
    parsed = parse_cot_output(text)
    iou_c, iou_f, rec_ok = box_ious_from_parsed(parsed, meta)
    tags = parsed.get("tags") or {}
    truncated = is_truncated(seq, prompt_len, max_new, tok.eos_token_id)
    log_heatmap_and_case(
        writer,
        step=step,
        tag_prefix=tag_prefix,
        meta=meta,
        response=text,
        parsed=parsed,
        iou=float(iou_f),
        rec_ok=bool(rec_ok),
        overlay_alpha=alpha,
        iou_c=float(iou_c),
        **tb_vis_flags(cfg),
    )
    writer.add_scalar(f"{tag_prefix}/iou_final", float(iou_f), step)
    writer.add_scalar(f"{tag_prefix}/iou_candidate", float(iou_c), step)
    writer.add_scalar(f"{tag_prefix}/rec_ok", 1.0 if rec_ok else 0.0, step)
    writer.add_scalar(f"{tag_prefix}/answer_tag", 1.0 if "answer" in tags else 0.0, step)
    writer.add_scalar(f"{tag_prefix}/final_bbox", 1.0 if parsed.get("bbox_2d") is not None else 0.0, step)
    writer.add_scalar(f"{tag_prefix}/cls_valid", 1.0 if parsed.get("is_anomaly") is not None else 0.0, step)
    writer.add_scalar(f"{tag_prefix}/truncated", 1.0 if truncated else 0.0, step)
    if was_training:
        model.train()
