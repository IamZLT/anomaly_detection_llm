"""Prior-Guided Process-Aware Spatial GRPO (Qwen3.5 LoRA, no SFT)."""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from data.ad_fs_scan import load_prior_split
from data.mvtec_prior_grounding import MVTecPriorCoTDataset, PriorCollator
from models.anomaly_prior import AnomalyPrior
from models.avNet import setup_model_and_processor
from models.come_qwen35 import apply_lora_to_qwen_llm, _unwrap_qwen_core
from utils.common import prepare_output_dir, qwen_norm1000_to_original_pixels, set_seed
from utils.prior_cot import (
    box_iou,
    completion_segment_ids,
    compute_rewards,
    mix_segment_advantage,
    parse_cot_output,
)
from utils.tb_prior import (
    format_grpo_group_text,
    log_grpo_run_config,
    log_grpo_scalars,
    log_heatmap_and_case,
)
from utils.train import (
    _auto_start_tensorboard,
    _disable_hf_datasets_check,
    _is_main_process,
    _train_log,
    tensorboard_event_dir,
)


def _tb_vis_flags(cfg: dict) -> dict:
    tb = cfg.get("tensorboard") or {}
    return {
        "log_heatmap": bool(tb.get("log_heatmap", True)),
        "log_case": bool(tb.get("log_case", True)),
    }


def _unwrap(m):
    return m.module if hasattr(m, "module") else m


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if k.startswith("_"):
            out[k] = v
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _model_inputs(batch: dict) -> dict:
    skip = {"labels", "_meta", "prompt_len"}
    return {k: v for k, v in batch.items() if k not in skip and torch.is_tensor(v)}


def _forward_with_vision(model, gen_in: dict, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Keep processor vision tensors; pad mm_token_type_ids to generated length."""
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    for k, v in gen_in.items():
        if k in ("input_ids", "attention_mask", "labels") or not torch.is_tensor(v):
            continue
        if k in ("mm_token_type_ids",) and v.shape[-1] != input_ids.shape[-1]:
            seq = int(input_ids.shape[-1])
            cur = int(v.shape[-1])
            if v.ndim == 1:
                v = v.unsqueeze(0)
            if v.shape[0] != input_ids.shape[0]:
                v = v.expand(input_ids.shape[0], *v.shape[1:])
            if cur < seq:
                v = F.pad(v, (0, seq - cur), value=0)
            elif cur > seq:
                v = v[..., :seq]
        kwargs[k] = v
    return model(**kwargs)


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
        batch = _move_batch(batch, device)
        gen_in = _model_inputs(batch)
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
        pred_box = parsed.get("bbox_2d")
        pred_anom = pred_box is not None and parsed.get("is_anomaly", True)
        ok = pred_anom == is_anom
        rec_ok += int(ok)
        iou_v = 0.0
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
                overlay_alpha=alpha,
                save_dir=vis_dir,
                **_tb_vis_flags(cfg),
            )
        records.append(
            {
                "image_path": meta.get("image_path"),
                "is_anomaly": is_anom,
                "pred_anomaly": pred_anom,
                "iou": iou_v,
                "pred_box": pred_px,
                "response": text[:2000],
                "parsed": parsed.get("raw"),
            }
        )
        seen += 1
    n = max(seen, 1)
    mean_iou = float(sum(ious) / len(ious)) if ious else 0.0
    if writer is not None:
        writer.add_scalar("eval/rec_acc", rec_ok / n, global_step)
        writer.add_scalar("eval/iou_at_03", (n_hit / n_anom) if n_anom else 0.0, global_step)
        writer.add_scalar("eval/mean_iou", mean_iou, global_step)
        writer.add_scalar("eval/json_parse_rate", parse_ok / n, global_step)
        writer.flush()
    model.train()
    return {
        "n": seen,
        "rec_acc": rec_ok / n,
        "iou_at_03": (n_hit / n_anom) if n_anom else 0.0,
        "mean_iou": mean_iou,
        "json_parse_rate": parse_ok / n,
        "records": records,
    }


def _run_final_mvtec_eval(cfg, model, processor, eval_loader, writer, output_dir, tag: str) -> None:
    n_max = (cfg.get("training") or {}).get("final_eval_num_samples")
    stats = run_simple_eval(
        cfg,
        _unwrap(model),
        processor,
        eval_loader,
        writer=writer,
        global_step=0,
        n_max=n_max,
    )
    _train_log(
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


@torch.no_grad()
def _log_grpo_probe_cot(
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
) -> None:
    """Fixed eval image: greedy CoT over GRPO steps so TF shows trajectory change."""
    if writer is None:
        return
    device = next(model.parameters()).device
    batch = collator([probe_item])
    batch = _move_batch(batch, device)
    gen_in = _model_inputs(batch)
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
    text = tok.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    meta = batch["_meta"][0]
    parsed = parse_cot_output(text)
    orig = tuple(meta["orig_size"])
    iou_v = 0.0
    gt = meta.get("gt_box_px")
    if parsed.get("bbox_2d") is not None and gt is not None:
        pred_px = qwen_norm1000_to_original_pixels(parsed["bbox_2d"], orig)
        iou_v = box_iou(pred_px, gt)
    rec_ok = (parsed.get("bbox_2d") is not None) == bool(meta.get("is_anomaly"))
    log_heatmap_and_case(
        writer,
        step=step,
        tag_prefix="grpo_probe",
        meta=meta,
        response=text,
        parsed=parsed,
        iou=float(iou_v),
        rec_ok=bool(rec_ok),
        overlay_alpha=alpha,
        **_tb_vis_flags(cfg),
    )
    writer.add_scalar("grpo_probe/iou", float(iou_v), step)
    if was_training:
        model.train()


def _token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits: [B, T, V], labels: [B, T] with -100 on prompt/pad.
    Returns per-token logprob and mask, both [B, T-1].
    """
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = labels[:, 1:]
    gather_idx = tgt.clamp(min=0).unsqueeze(-1)
    token_lp = logp.gather(-1, gather_idx).squeeze(-1)
    mask = tgt.ne(-100)
    return token_lp * mask, mask


def _disable_adapter_ctx(model):
    m = _unwrap(model)
    fn = getattr(m, "disable_adapter", None)
    if fn is None:
        return nullcontext()
    ctx = fn()
    if hasattr(ctx, "__enter__"):
        return ctx
    return nullcontext()


def _token_logprobs_nograd(model, gen_in: dict, outputs: torch.Tensor, attn: torch.Tensor, labels: torch.Tensor):
    lps = []
    masks = []
    m = _unwrap(model)
    with torch.no_grad():
        for i in range(outputs.shape[0]):
            out = _forward_with_vision(m, gen_in, outputs[i : i + 1], attn[i : i + 1])
            lp, mask = _token_logprobs(out.logits, labels[i : i + 1])
            lps.append(lp)
            masks.append(mask)
    return torch.cat(lps, dim=0), torch.cat(masks, dim=0)


def _clipped_pg_kl(
    new_lp: torch.Tensor,
    mask: torch.Tensor,
    old_lp: torch.Tensor,
    ref_lp: torch.Tensor,
    adv: torch.Tensor,
    clip_low: float,
    clip_high: float,
    kl_beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rho = (new_lp - old_lp).exp().clamp(max=1.0e4)
    clipped = rho.clamp(1.0 - float(clip_low), 1.0 + float(clip_high))
    surr = torch.minimum(rho * adv, clipped * adv)
    denom = mask.sum().clamp(min=1)
    L_pg = -(surr * mask).sum() / denom
    L_kl = ((new_lp - ref_lp) * mask).sum() / denom
    clip_frac = (((rho < (1.0 - clip_low)) | (rho > (1.0 + clip_high))).float() * mask).sum() / denom
    rho_mean = (rho * mask).sum() / denom
    return L_pg + float(kl_beta) * L_kl, L_pg, L_kl, rho_mean.detach(), clip_frac.detach()


def _grpo_advantages(rewards: torch.Tensor, group: int, eps: float) -> torch.Tensor:
    r = rewards.view(-1, group)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True).clamp(min=eps)
    adv = (r - mean) / std
    return adv.reshape(-1)


def _avg_across_ranks(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return float(value)
    t = torch.tensor([float(value)], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return float(t.item())


def _grpo_param_map(gcfg: dict, *, lr: float, accum: int, group: int, temperature: float, top_p: float, max_new: int) -> Dict[str, float]:
    rew = gcfg.get("reward") or {}
    return {
        "lr": float(lr),
        "group_size": float(group),
        "policy_epochs": float(gcfg.get("policy_epochs", 3)),
        "accum": float(accum),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": float(max_new),
        "max_grad_norm": float(gcfg.get("max_grad_norm", 1.0)),
        "adv_eps": float(gcfg.get("adv_eps", 1e-6)),
        "clip_low": float(gcfg.get("clip_low", 0.20)),
        "clip_high": float(gcfg.get("clip_high", 0.28)),
        "kl_beta": float(gcfg.get("kl_beta", 1.0e-4)),
        "min_reward_std": float(gcfg.get("min_reward_std", 0.02)),
        "max_resample_attempts": float(gcfg.get("max_resample_attempts", 2)),
        "w_cov": float(rew.get("w_cov", 0.7)),
        "w_compact": float(rew.get("w_compact", 0.3)),
        "w_iou": float(rew.get("w_iou", 0.45)),
        "w_edge": float(rew.get("w_edge", 0.40)),
        "w_center": float(rew.get("w_center", 0.15)),
        "edge_beta": float(rew.get("edge_beta", 8.0)),
        "center_gamma": float(rew.get("center_gamma", 8.0)),
        "format_weight": float(rew.get("format_weight", 0.03)),
        "keep_tol_norm1000": float(rew.get("keep_tol_norm1000", 8.0)),
        "normal_correct": float(rew.get("normal_correct", 1.0)),
        "normal_false_positive": float(rew.get("normal_false_positive", -0.5)),
    }


def _group_std(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    t = torch.tensor(vals, dtype=torch.float32)
    return float(t.std(unbiased=False))


def _build_segment_advantages(
    tokenizer,
    seqs: List[torch.Tensor],
    prompt_len: int,
    max_t: int,
    a_ground: torch.Tensor,
    a_reason: torch.Tensor,
    a_box: torch.Tensor,
    rew_cfg: dict,
    device: torch.device,
) -> torch.Tensor:
    adv = torch.zeros(len(seqs), max(max_t - 1, 1), device=device, dtype=torch.float32)
    for i, s in enumerate(seqs):
        comp = s[prompt_len:]
        segs = completion_segment_ids(tokenizer, comp)
        ag, ar, ab = float(a_ground[i]), float(a_reason[i]), float(a_box[i])
        for k, seg in enumerate(segs):
            j = prompt_len - 1 + k
            if 0 <= j < adv.shape[1]:
                adv[i, j] = mix_segment_advantage(seg, ag, ar, ab, rew_cfg)
    return adv


def train_grpo(cfg: dict, model, processor, prior, train_set, eval_loader, output_dir: str, writer: Optional[SummaryWriter]) -> None:
    gcfg = cfg.get("grpo") or {}
    inf = cfg.get("inference") or {}
    rew_cfg = gcfg.get("reward") or {}
    group = int(gcfg.get("group_size", 8))
    policy_epochs = int(gcfg.get("policy_epochs", 3))
    max_steps = int(gcfg.get("max_steps", 200))
    lr = float(gcfg.get("learning_rate", 5.0e-6))
    accum = int(gcfg.get("gradient_accumulation_steps", 1))
    max_new = int(gcfg.get("max_new_tokens", inf.get("max_new_tokens", 512)))
    temperature = float(gcfg.get("temperature", 0.9))
    top_p = float(gcfg.get("top_p", 0.95))
    clip_low = float(gcfg.get("clip_low", 0.20))
    clip_high = float(gcfg.get("clip_high", 0.28))
    kl_beta = float(gcfg.get("kl_beta", 1.0e-4))
    min_std = float(gcfg.get("min_reward_std", 0.02))
    max_resample = int(gcfg.get("max_resample_attempts", 2))
    eval_every = int(cfg.get("training", {}).get("eval_every_n_steps", 50))
    save_every = int(gcfg.get("save_steps", 100))
    tb = cfg.get("tensorboard") or {}
    vis_every = int(tb.get("vis_every_n_steps", eval_every) or 0)
    log_every = int(gcfg.get("logging_steps", 1))
    alpha = float((cfg.get("prior") or {}).get("overlay_alpha", 0.45))
    adv_eps = float(gcfg.get("adv_eps", 1e-6))

    from torch.utils.data.distributed import DistributedSampler

    world = int(os.environ.get("WORLD_SIZE", "1"))
    sampler = DistributedSampler(train_set, shuffle=True) if world > 1 else None
    collator = PriorCollator(processor, prior, cfg)
    loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    device = next(model.parameters()).device
    tok = getattr(processor, "tokenizer", processor)
    eval_ds = getattr(eval_loader, "dataset", None)
    probe_item = None
    if eval_ds is not None:
        for i in range(len(eval_ds)):
            cand = eval_ds[i]
            if cand.get("is_anomaly"):
                probe_item = cand
                break
        if probe_item is None and len(eval_ds) > 0:
            probe_item = eval_ds[0]
    it = iter(loader)
    model.train()
    t0 = time.time()
    epoch = 0
    opt_step = 0
    last_gnorm: Optional[float] = None
    param_map = _grpo_param_map(
        gcfg, lr=lr, accum=accum, group=group, temperature=temperature, top_p=top_p, max_new=max_new
    )
    if _is_main_process() and writer is not None:
        log_grpo_run_config(writer, cfg)
        _train_log(
            f"[grpo] spatial GRPO group={group} K={policy_epochs} lr={lr} T={temperature} "
            f"clip=({clip_low},{clip_high}) kl_beta={kl_beta} min_std={min_std} "
            f"log_every={log_every} vis_every={vis_every} eval_every={eval_every}",
            main_only=True,
        )

    def _next_batch():
        nonlocal epoch, it
        try:
            return next(it)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            it = iter(loader)
            return next(it)

    def _generate_group(gen_in, prompt_len: int):
        m = _unwrap(model)
        was_training = m.training
        m.eval()
        seqs, texts = [], []
        with torch.no_grad():
            for _ in range(group):
                out_ids = m.generate(
                    **gen_in,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=1,
                )
                seqs.append(out_ids[0].detach())
                texts.append(tok.decode(out_ids[0][prompt_len:], skip_special_tokens=True))
        if was_training:
            m.train()
        return seqs, texts

    def _score_group(texts: List[str], meta: dict):
        details = []
        for text in texts:
            parsed = parse_cot_output(text)
            details.append(
                compute_rewards(
                    parsed,
                    meta.get("gt_box_px"),
                    tuple(meta["orig_size"]),
                    bool(meta["is_anomaly"]),
                    cfg,
                )
            )
        return details

    for step in range(1, max_steps + 1):
        batch = _move_batch(_next_batch(), device)
        gen_in = _model_inputs(batch)
        prompt_len = int(batch["prompt_len"][0].item())
        pad_id = tok.pad_token_id or tok.eos_token_id
        meta = batch["_meta"][0]

        seqs, texts, details = [], [], []
        resample_n = 0
        skipped = False
        for attempt in range(max_resample + 1):
            seqs, texts = _generate_group(gen_in, prompt_len)
            details = _score_group(texts, meta)
            std_box = _group_std([float(d.get("R_box", 0.0)) for d in details])
            std_g = _group_std([float(d.get("R_ground", 0.0)) for d in details])
            if std_box >= min_std or std_g >= min_std:
                resample_n = attempt
                skipped = False
                break
            resample_n = attempt
            skipped = True
        if skipped:
            resample_n = max_resample

        r_box = torch.tensor([float(d.get("R_box", 0.0)) for d in details], device=device, dtype=torch.float32)
        r_ground = torch.tensor([float(d.get("R_ground", 0.0)) for d in details], device=device, dtype=torch.float32)
        r_reason = torch.tensor([float(d.get("R_reason", 0.0)) for d in details], device=device, dtype=torch.float32)
        a_ground = _grpo_advantages(r_ground, group, adv_eps)
        a_reason = _grpo_advantages(r_reason, group, adv_eps)
        a_box = _grpo_advantages(r_box, group, adv_eps)

        max_t = max(int(s.numel()) for s in seqs)
        outputs = torch.full((group, max_t), int(pad_id), device=device, dtype=torch.long)
        for i, s in enumerate(seqs):
            outputs[i, : s.numel()] = s.to(device)
        attn = (outputs != pad_id).long()
        labels = outputs.clone()
        labels[:, :prompt_len] = -100
        labels[outputs == pad_id] = -100

        old_lp, lp_mask = _token_logprobs_nograd(model, gen_in, outputs, attn, labels)
        with _disable_adapter_ctx(model):
            ref_lp, _ = _token_logprobs_nograd(model, gen_in, outputs, attn, labels)
        adv_tok = _build_segment_advantages(
            tok, seqs, prompt_len, max_t, a_ground, a_reason, a_box, rew_cfg, device
        )
        if skipped:
            adv_tok = torch.zeros_like(adv_tok)
        seq_lp = (old_lp * lp_mask).sum(dim=1) / lp_mask.sum(dim=1).clamp(min=1)

        loss_v = 0.0
        pg_v = 0.0
        kl_v = 0.0
        rho_v = 1.0
        clip_v = 0.0
        did_opt = False
        model.train()
        n_pe = max(policy_epochs, 1)
        opt.zero_grad(set_to_none=True)
        for pe in range(n_pe):
            pe_loss = pe_pg = pe_kl = pe_rho = pe_clip = 0.0
            for i in range(group):
                use_ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
                ctx = (
                    model.no_sync()
                    if (use_ddp and i < group - 1 and hasattr(model, "no_sync"))
                    else nullcontext()
                )
                with ctx:
                    out = _forward_with_vision(model, gen_in, outputs[i : i + 1], attn[i : i + 1])
                    if skipped:
                        loss = out.logits.float().sum() * 0.0
                        L_pg = L_kl = loss
                        rho_mean = torch.zeros((), device=device)
                        clip_frac = torch.zeros((), device=device)
                    else:
                        new_lp, mask = _token_logprobs(out.logits, labels[i : i + 1])
                        loss, L_pg, L_kl, rho_mean, clip_frac = _clipped_pg_kl(
                            new_lp,
                            mask,
                            old_lp[i : i + 1],
                            ref_lp[i : i + 1],
                            adv_tok[i : i + 1],
                            clip_low,
                            clip_high,
                            kl_beta,
                        )
                    (loss / float(group) / float(max(accum, 1))).backward()
                pe_loss += float(loss.detach())
                pe_pg += float(L_pg.detach())
                pe_kl += float(L_kl.detach())
                pe_rho += float(rho_mean)
                pe_clip += float(clip_frac)
            if ((pe + 1) % max(accum, 1) == 0) or (pe == n_pe - 1):
                grads = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                if grads:
                    last_gnorm = float(torch.nn.utils.clip_grad_norm_(grads, float(gcfg.get("max_grad_norm", 1.0))))
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                did_opt = True
            g = float(max(group, 1))
            loss_v += pe_loss / g
            pg_v += pe_pg / g
            kl_v += pe_kl / g
            rho_v = pe_rho / g
            clip_v = pe_clip / g
        loss_v /= float(n_pe)
        pg_v /= float(n_pe)
        kl_v /= float(n_pe)

        loss_v = _avg_across_ranks(loss_v, device)
        pg_v = _avg_across_ranks(pg_v, device)
        kl_v = _avg_across_ranks(kl_v, device)
        r_mean = _avg_across_ranks(float(r_box.mean()), device)
        r_std = _avg_across_ranks(float(r_box.std(unbiased=False)), device)
        parse_rate = sum(1 for t in texts if parse_cot_output(t).get("has_tags")) / max(group, 1)
        parse_rate = _avg_across_ranks(parse_rate, device)

        if _is_main_process():
            log_grpo_scalars(
                writer,
                step=step,
                loss=loss_v,
                rewards=r_box,
                details=details,
                advantages=a_box,
                seq_lp=seq_lp,
                texts=texts,
                lr=lr,
                params=param_map,
                grad_norm=last_gnorm if did_opt else None,
                opt_step=opt_step,
                is_anomaly=bool(meta.get("is_anomaly")),
                extra={
                    "loss_pg": pg_v,
                    "loss_kl": kl_v,
                    "rho_mean": rho_v,
                    "clip_frac": clip_v,
                    "A_ground": float(a_ground.mean()),
                    "A_reason": float(a_reason.mean()),
                    "A_box": float(a_box.mean()),
                    "skipped": 1.0 if skipped else 0.0,
                    "resample": float(resample_n),
                    "policy_epochs": float(policy_epochs),
                },
            )
            if log_every > 0 and step % log_every == 0:
                n = max(group, 1)
                skip_s = " skip" if skipped else ""
                _train_log(
                    f"[grpo] step={step}/{max_steps} opt={opt_step} loss={loss_v:.4f} "
                    f"pg={pg_v:.4f} kl={kl_v:.4f} rho={rho_v:.3f} clip={clip_v:.3f} "
                    f"Rb={r_mean:.3f}±{r_std:.3f} "
                    f"Rg={sum(d.get('R_ground', 0.0) for d in details)/n:.3f} "
                    f"Rr={sum(d.get('R_reason', 0.0) for d in details)/n:.3f} "
                    f"iou={sum(d.get('R_iou', 0.0) for d in details)/n:.3f} "
                    f"edge={sum(d.get('R_edge', 0.0) for d in details)/n:.3f} "
                    f"fmt={parse_rate:.2f} rs={resample_n}{skip_s}"
                    + (f" gnorm={last_gnorm:.2f}" if did_opt and last_gnorm is not None else ""),
                    main_only=True,
                )

        if _is_main_process() and vis_every > 0 and step % vis_every == 0:
            best_i = int(torch.argmax(r_box).item())
            log_heatmap_and_case(
                writer,
                step=step,
                tag_prefix="grpo_batch",
                meta=meta,
                response=texts[best_i],
                parsed=parse_cot_output(texts[best_i]),
                iou=float(details[best_i].get("R_iou", 0.0)),
                rec_ok=True,
                overlay_alpha=alpha,
                **_tb_vis_flags(cfg),
            )
            if writer is not None and bool(tb.get("log_grpo_cot", True)):
                writer.add_text(
                    "grpo_batch/4_group_cot",
                    format_grpo_group_text(
                        step=step,
                        meta=meta,
                        texts=texts,
                        details=details,
                        advantages=a_box.detach().cpu().tolist(),
                        logprobs=seq_lp.detach().cpu().tolist(),
                    ),
                    step,
                )
            if probe_item is not None:
                _log_grpo_probe_cot(
                    cfg,
                    _unwrap(model),
                    processor,
                    collator,
                    probe_item,
                    writer,
                    step,
                    max_new=max_new,
                    alpha=alpha,
                )
            if writer is not None:
                writer.flush()

        if _is_main_process() and eval_every > 0 and step % eval_every == 0:
            stats = run_simple_eval(
                cfg, _unwrap(model), processor, eval_loader, writer=writer, global_step=step
            )
            _train_log(
                f"[grpo-eval] step={step} rec={stats['rec_acc']:.3f} mean_iou={stats['mean_iou']:.3f}",
                main_only=True,
            )
            os.makedirs(os.path.join(output_dir, "eval_steps"), exist_ok=True)
            compact = {k: v for k, v in stats.items() if k != "records"}
            with open(os.path.join(output_dir, "eval_steps", f"grpo_{step:08d}.json"), "w", encoding="utf-8") as f:
                json.dump(compact, f, ensure_ascii=False, indent=2)

        if _is_main_process() and save_every > 0 and step % save_every == 0:
            ckpt = os.path.join(output_dir, f"grpo_checkpoint-{step}")
            os.makedirs(ckpt, exist_ok=True)
            _unwrap(model).save_pretrained(ckpt)
            processor.save_pretrained(ckpt)

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

    mins = (time.time() - t0) / 60.0
    if _is_main_process():
        _train_log(f"GRPO 完成，耗时 {mins:.2f} 分钟", main_only=True)


def setup_prior_model(cfg: dict):
    model, processor = setup_model_and_processor(cfg, for_inference=False)
    model = apply_lora_to_qwen_llm(model, cfg)
    core = _unwrap_qwen_core(model)
    visual = getattr(getattr(core, "model", None), "visual", None)
    if visual is not None:
        for p in visual.parameters():
            p.requires_grad = False
        visual.eval()
    if bool(cfg.get("training", {}).get("gradient_checkpointing", True)):
        fn = getattr(model, "gradient_checkpointing_enable", None)
        if fn is not None:
            gckw = cfg.get("training", {}).get("gradient_checkpointing_kwargs") or {"use_reentrant": False}
            try:
                fn(gradient_checkpointing_kwargs=gckw)
            except TypeError:
                fn()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    prior = AnomalyPrior.from_qwen(model, cfg)
    if _is_main_process():
        n = sum(p.numel() for p in model.parameters())
        nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[prior] LoRA 可训练 {nt / 1e6:.2f}M / 总 {n / 1e6:.2f}M；vision 冻结", flush=True)
        print(f"[prior] vision blocks={prior.block_indices} tau={prior.temperature} radius={prior.neighborhood_radius}", flush=True)
    return model, processor, prior


def train_prior_main(cfg: dict) -> None:
    _disable_hf_datasets_check()
    set_seed(int(cfg["training"]["seed"]))
    is_main = _is_main_process()
    output_dir = prepare_output_dir(
        base_dir=cfg["paths"]["output_dir"],
        run_name=cfg["runtime"]["run_name"],
        auto_create=cfg["runtime"]["auto_create_output_dir"],
    )
    cfg["paths"]["output_dir"] = output_dir
    if is_main:
        _train_log(f"输出目录: {output_dir}", main_only=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        with open(os.path.join(output_dir, "config.yaml.txt"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)

    data_cfg = cfg.get("data") or {}
    train_layout = str(data_cfg.get("train_layout", "visa")).lower()
    eval_layout = str(data_cfg.get("eval_layout", "mvtec")).lower()
    if train_layout == "json" or eval_layout == "json":
        raise ValueError("旧 JSON 训练链路已移除，请用 data.train_layout=visa / eval_layout=mvtec 扫盘")
    train_samples, eval_samples = load_prior_split(cfg)
    n_tr_anom = sum(1 for s in train_samples if bool((s.get("metadata") or {}).get("anomaly")))
    n_ev_anom = sum(1 for s in eval_samples if bool((s.get("metadata") or {}).get("anomaly")))
    if is_main:
        ratio = (cfg.get("data") or {}).get("normal_anomaly_ratio")
        _train_log(
            f"数据协议: VisA 训练 / MVTec test 测试；normal:anomaly={ratio}；"
            f"train={len(train_samples)} (anom={n_tr_anom}, normal={len(train_samples)-n_tr_anom}) "
            f"eval={len(eval_samples)} (anom={n_ev_anom}) "
            f"train_layout={train_layout} eval_layout={eval_layout}",
            main_only=True,
        )

    model, processor, prior = setup_prior_model(cfg)
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        model = model.to(local_rank)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=bool(cfg.get("distributed", {}).get("ddp_find_unused_parameters", False)),
        )

    train_set = MVTecPriorCoTDataset(train_samples, cfg, processor, mode="train")
    eval_set = MVTecPriorCoTDataset(eval_samples, cfg, processor, mode="eval")
    eval_collate = PriorCollator(processor, prior, cfg)
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, collate_fn=eval_collate, num_workers=0)

    writer = None
    tb_dir = tensorboard_event_dir(output_dir)
    if is_main:
        writer = SummaryWriter(log_dir=tb_dir)
        _auto_start_tensorboard(cfg, output_dir)
        print("开始 GRPO（无 SFT）…", flush=True)

    train_grpo(cfg, model, processor, prior, train_set, eval_loader, output_dir, writer)
    if is_main:
        gdir = os.path.join(output_dir, "grpo_final")
        _unwrap(model).save_pretrained(gdir)
        processor.save_pretrained(gdir)
        print(f"GRPO 完成: {gdir}", flush=True)
        _run_final_mvtec_eval(cfg, model, processor, eval_loader, writer, output_dir, tag="grpo")

    if writer is not None:
        writer.close()
    if is_main:
        print(f"全部完成: {output_dir}", flush=True)
