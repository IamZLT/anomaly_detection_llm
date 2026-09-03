"""Rollout → reward → process-aware spatial GRPO optimization."""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.prior_dataset import PriorCollator, PriorCoTDataset, build_train_ref_pool
from data.scan import load_prior_split, split_holdout_by_class
from evaluation.evaluator import (
    first_anomaly_item,
    log_grpo_probe_cot,
    run_final_mvtec_eval,
    run_simple_eval,
    tb_vis_flags,
)
from evaluation.metrics import defect_size_bin, gt_relative_area, is_truncated
from models.anomaly_prior import AnomalyPrior
from models.lora import apply_lora
from models.qwen35 import freeze_vision_encoder, force_vision_eval, setup_model_and_processor
from models.vision_cache import bind_cached_image_features
from reasoning.parser import parse_cot_output, rollout_protocol_stats
from reasoning.rewards import compute_rewards
from rl.grpo import (
    avg_across_ranks,
    build_segment_advantages,
    clipped_pg_kl,
    disable_adapter_ctx,
    dropout_eval,
    expand_gen_in_for_group,
    forward_with_vision,
    grpo_advantages,
    grpo_param_map,
    group_std,
    micro_batch_ranges,
    model_inputs,
    move_batch,
    padded_completion_tensors,
    resolve_max_steps,
    token_logprobs,
    token_logprobs_nograd,
    unwrap_model,
)
from utils.common import (
    close_rollout_log,
    is_main_process,
    open_rollout_log,
    prepare_output_dir,
    rollout_log,
    set_seed,
    train_log,
)
from visualization.tensorboard import (
    auto_start_tensorboard,
    format_grpo_group_text,
    log_grpo_run_config,
    log_grpo_scalars,
    log_heatmap_and_case,
    tensorboard_event_dir,
)


def disable_hf_datasets_check() -> None:
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


def _dump_parser_debug(text: str, parsed: dict) -> None:
    block = (
        "\n===== PARSER DEBUG =====\n"
        f"RAW_REPR: {text!r}\n"
        f"tags: {list((parsed.get('tags') or {}).keys())}\n"
        f"has_tags: {parsed.get('has_tags')}\n"
        f"candidate_state: {parsed.get('candidate_bbox_state')}\n"
        f"candidate: {parsed.get('candidate_bbox_2d')}\n"
        f"answer_state: {parsed.get('answer_state')}\n"
        f"final_state: {parsed.get('final_bbox_state')}\n"
        f"pred: {parsed.get('is_anomaly')}\n"
        f"bbox: {parsed.get('bbox_2d')}\n"
        f"description_ok: {parsed.get('description_ok')}\n"
        f"prose_ok: {parsed.get('prose_ok')}\n"
        f"trajectory_valid: {parsed.get('trajectory_valid')}\n"
        "========================\n"
    )
    if is_main_process():
        print(block, flush=True)
    rollout_log(block)


def _log_rollout_txt(tok, input_ids, prompt_len: int, meta: dict, step: int, texts, details, parsed_list) -> None:
    try:
        prompt = tok.decode(input_ids[0][:prompt_len].cpu(), skip_special_tokens=True)
    except Exception:
        prompt = "<prompt decode failed>"
    lines = [
        f"===== step={step} image={meta.get('image_path')} class={meta.get('class_name')} "
        f"is_anomaly={meta.get('is_anomaly')} gt={meta.get('gt_box_px')} =====",
        "[PROMPT]",
        prompt,
        "[/PROMPT]",
    ]
    for i, (text, det, p) in enumerate(zip(texts, details, parsed_list)):
        lines.append(
            f"--- tau[{i}] valid={p.get('trajectory_valid')} pred={p.get('is_anomaly')} "
            f"Rf={det.get('R_final', 0):.3f} Rg={det.get('R_ground', 0):.3f} "
            f"Rr={det.get('R_reason', 0):.3f} raw_iou_f={det.get('raw_iou_f', 0):.3f} "
            f"raw_iou_c={det.get('raw_iou_c', 0):.3f} cand={p.get('candidate_bbox_2d')} "
            f"bbox={p.get('bbox_2d')}"
        )
        lines.append(f"response_repr: {text!r}")
        lines.append("")
    rollout_log("\n".join(lines))


def train_grpo(cfg: dict, model, processor, prior, train_set, eval_loader, output_dir: str, writer: Optional[SummaryWriter]) -> None:
    gcfg = cfg.get("grpo") or {}
    inf = cfg.get("inference") or {}
    if is_main_process():
        open_rollout_log(os.path.join(output_dir, "logs", "rollouts.txt"))
    group = int(gcfg.get("group_size", 8))
    policy_epochs = int(gcfg.get("policy_epochs", 3))
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
    hard_resample = int(gcfg.get("hard_resample_attempts", 4))
    small_area = float(gcfg.get("small_area_thresh", 0.02))

    def _micro(key: str, default: int) -> int:
        v = gcfg.get(key)
        if v in (None, "", "null", "None"):
            v = default
        return max(1, min(int(v), int(group)))

    rollout_micro = _micro("rollout_micro_batch_size", 2)
    logprob_micro = _micro("logprob_micro_batch_size", 2)
    actor_micro = _micro("actor_micro_batch_size", 1)
    log_mvtec_probe = bool((cfg.get("tensorboard") or {}).get("log_mvtec_probe", False))
    eval_every = int(cfg.get("training", {}).get("eval_every_n_steps", 50))
    save_every = int(gcfg.get("save_steps", 100))
    tb = cfg.get("tensorboard") or {}
    vis_every = int(tb.get("vis_every_n_steps", eval_every) or 0)
    log_every = int(gcfg.get("logging_steps", 1))
    alpha = float((cfg.get("prior") or {}).get("overlay_alpha", 0.45))
    adv_eps = float(gcfg.get("adv_eps", 1e-6))
    fmt_mix = float(gcfg.get("format_advantage_weight", 0.20))

    from torch.utils.data.distributed import DistributedSampler

    world = int(os.environ.get("WORLD_SIZE", "1"))
    max_steps = resolve_max_steps(gcfg, len(train_set), world)
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
    eos_id = tok.eos_token_id
    eval_ds = getattr(eval_loader, "dataset", None) if eval_loader is not None else None
    train_probe = first_anomaly_item(train_set) if is_main_process() else None
    crossdomain_probe = None
    if is_main_process() and log_mvtec_probe and eval_ds is not None:
        crossdomain_probe = first_anomaly_item(eval_ds)
    it = iter(loader)
    pending_retry = None
    model.train()
    force_vision_eval(model)
    t0 = time.time()
    epoch = 0
    opt_step = 0
    step = 0
    attempt_step = 0
    last_gnorm: Optional[float] = None
    param_map = grpo_param_map(
        gcfg, lr=lr, accum=accum, group=group, temperature=temperature, top_p=top_p, max_new=max_new
    )
    param_map["max_steps"] = float(max_steps)
    if is_main_process() and writer is not None:
        log_grpo_run_config(writer, cfg)
        train_log(
            f"[grpo] spatial GRPO group={group} K={policy_epochs} lr={lr} T={temperature} "
            f"clip=({clip_low},{clip_high}) kl_beta={kl_beta} min_std={min_std} "
            f"N_train={len(train_set)} world={world} epochs={gcfg.get('epochs', 2)} max_steps={max_steps} "
            f"rollout_micro={rollout_micro} logprob_micro={logprob_micro} actor_micro={actor_micro} "
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

    stop_criteria = None
    try:
        from transformers.generation.stopping_criteria import StopStringCriteria, StoppingCriteriaList

        stop_criteria = StoppingCriteriaList(
            [StopStringCriteria(tokenizer=tok, stop_strings=["</answer>"])]
        )
        if is_main_process():
            train_log("[grpo] cached stop criteria for </answer>", main_only=True)
    except Exception as exc:
        if is_main_process():
            train_log(f"[grpo] stop criteria unavailable ({type(exc).__name__}), generate without it", main_only=True)

    def _generate_group(gen_in, prompt_len: int, image_embeds=None):
        m = unwrap_model(model)
        was_training = m.training
        m.eval()
        seqs, texts = [], []
        remain = int(group)
        micro = max(1, min(int(rollout_micro), int(group)))

        def _one_generate(n: int):
            kw = dict(
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=n,
                use_cache=True,
            )
            if stop_criteria is not None:
                try:
                    return m.generate(**gen_in, **kw, stopping_criteria=stop_criteria)
                except TypeError:
                    pass
            try:
                return m.generate(**gen_in, **kw, stop_strings=["</answer>"], tokenizer=tok)
            except TypeError:
                return m.generate(**gen_in, **kw)

        with torch.no_grad():
            with bind_cached_image_features(m, image_embeds):
                while remain > 0:
                    n = min(micro, remain)
                    try:
                        out_ids = _one_generate(n)
                    except torch.cuda.OutOfMemoryError:
                        if n <= 1:
                            raise
                        torch.cuda.empty_cache()
                        nxt = 2 if n > 2 else 1
                        if is_main_process():
                            train_log(f"[grpo] generate OOM at micro={n}, retry micro={nxt}", main_only=True)
                        micro = nxt
                        continue
                    if out_ids.dim() == 1:
                        out_ids = out_ids.unsqueeze(0)
                    for i in range(int(out_ids.shape[0])):
                        seqs.append(out_ids[i].detach())
                        texts.append(tok.decode(out_ids[i][prompt_len:], skip_special_tokens=True))
                    remain -= int(out_ids.shape[0])
        if was_training:
            m.train()
            force_vision_eval(m)
        return seqs, texts

    def _score_group(texts: List[str], meta: dict):
        details = []
        parsed_list = []
        for text in texts:
            parsed = parse_cot_output(text)
            parsed_list.append(parsed)
            details.append(
                compute_rewards(
                    parsed,
                    meta.get("gt_box_px"),
                    tuple(meta["orig_size"]),
                    bool(meta["is_anomaly"]),
                    cfg,
                )
            )
            if not parsed.get("trajectory_valid"):
                _dump_parser_debug(text, parsed)
        return details, parsed_list

    while step < max_steps:
        attempt_step += 1
        raw_batch = pending_retry
        pending_retry = None
        from_requeue = raw_batch is not None
        if raw_batch is None:
            raw_batch = _next_batch()
        batch = move_batch(raw_batch, device)
        gen_in = model_inputs(batch)
        vision_cache = batch.get("image_embeds")
        prompt_len = int(batch["prompt_len"][0].item())
        pad_id = tok.pad_token_id or tok.eos_token_id
        meta = batch["_meta"][0]
        a_gt = gt_relative_area(meta.get("gt_box_px"), tuple(meta["orig_size"])) if meta.get("is_anomaly") else 0.0
        size_bin = defect_size_bin(a_gt, bool(meta.get("is_anomaly")))
        attempts = max_resample
        if bool(meta.get("is_anomaly")) and a_gt < small_area:
            attempts = max(attempts, hard_resample)

        seqs, texts, details, parsed_list = [], [], [], []
        resample_n = 0
        skipped = False
        t_roll = time.time()
        for attempt in range(attempts + 1):
            seqs, texts = _generate_group(gen_in, prompt_len, vision_cache)
            details, parsed_list = _score_group(texts, meta)
            std_g = group_std([float(d.get("R_ground", 0.0)) for d in details])
            std_r = group_std([float(d.get("R_reason", 0.0)) for d in details])
            std_f = group_std([float(d.get("R_final", 0.0)) for d in details])
            std_fmt = group_std([float(d.get("R_fmt", 0.0)) for d in details])
            if max(std_g, std_r, std_f, std_fmt) >= min_std:
                resample_n = attempt
                skipped = False
                break
            resample_n = attempt
            skipped = True
        if skipped:
            resample_n = attempts
        t_roll = time.time() - t_roll

        r_ground = torch.tensor([float(d.get("R_ground", 0.0)) for d in details], device=device, dtype=torch.float32)
        r_reason = torch.tensor([float(d.get("R_reason", 0.0)) for d in details], device=device, dtype=torch.float32)
        r_final = torch.tensor([float(d.get("R_final", 0.0)) for d in details], device=device, dtype=torch.float32)
        r_fmt = torch.tensor([float(d.get("R_fmt", 0.0)) for d in details], device=device, dtype=torch.float32)
        a_ground = grpo_advantages(r_ground, group, adv_eps)
        a_reason = grpo_advantages(r_reason, group, adv_eps)
        a_final = grpo_advantages(r_final, group, adv_eps)
        a_fmt = grpo_advantages(r_fmt, group, adv_eps)

        use_ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        active = torch.tensor([0 if skipped else 1], device=device, dtype=torch.int64)
        if use_ddp:
            dist.all_reduce(active, op=dist.ReduceOp.SUM)
        all_skipped = int(active.item()) == 0

        outputs, attn, labels = padded_completion_tensors(seqs, prompt_len, int(pad_id), device)
        max_t = int(outputs.shape[1])
        t_upd = time.time()

        if all_skipped:
            t_lp = max(max_t - 1, 1)
            old_lp = torch.zeros(group, t_lp, device=device)
            lp_mask = torch.zeros_like(old_lp)
            ref_lp = old_lp
            ref_gap = 0.0
            adv_tok = torch.zeros_like(old_lp)
            seq_lp = torch.zeros(group, device=device)
        else:
            old_parts, mask_parts, ref_parts = [], [], []
            for s, e in micro_batch_ranges(group, logprob_micro):
                n = e - s
                chunk_in = expand_gen_in_for_group(gen_in, n)
                with bind_cached_image_features(model, vision_cache, repeat_factor=n):
                    lp, mk = token_logprobs_nograd(model, chunk_in, outputs[s:e], attn[s:e], labels[s:e])
                    with disable_adapter_ctx(model):
                        rlp, _ = token_logprobs_nograd(model, chunk_in, outputs[s:e], attn[s:e], labels[s:e])
                old_parts.append(lp)
                mask_parts.append(mk)
                ref_parts.append(rlp)
            old_lp = torch.cat(old_parts, dim=0)
            lp_mask = torch.cat(mask_parts, dim=0)
            ref_lp = torch.cat(ref_parts, dim=0)
            ref_gap = float(((old_lp - ref_lp).abs() * lp_mask).sum() / lp_mask.sum().clamp(min=1))
            adv_tok = build_segment_advantages(
                tok,
                seqs,
                prompt_len,
                max_t,
                a_ground,
                a_reason,
                a_final,
                a_fmt,
                bool(meta.get("is_anomaly")),
                device,
                fmt_mix=fmt_mix,
            )
            if skipped:
                adv_tok = torch.zeros_like(adv_tok)
            seq_lp = (old_lp * lp_mask).sum(dim=1) / lp_mask.sum(dim=1).clamp(min=1)

        n_ans = sum(1 for p in parsed_list if "answer" in (p.get("tags") or {}))
        n_bbox = sum(1 for p in parsed_list if p.get("bbox_2d") is not None)
        n_cls = sum(1 for p in parsed_list if p.get("is_anomaly") is not None)
        n_trunc = sum(1 for s in seqs if is_truncated(s, prompt_len, max_new, eos_id))
        n_g = max(group, 1)
        answer_tag_rate = n_ans / n_g
        final_bbox_rate = n_bbox / n_g
        cls_tag_rate = n_cls / n_g
        truncation_rate = n_trunc / n_g
        proto_stats = rollout_protocol_stats(parsed_list, texts)
        parse_rate_local = proto_stats["protocol_rate"]

        loss_v = 0.0
        pg_v = 0.0
        kl_v = 0.0
        rho_v = 1.0
        clip_v = 0.0
        did_opt = False
        model.train()
        force_vision_eval(model)
        n_pe = max(policy_epochs, 1)
        max_gnorm = float(gcfg.get("max_grad_norm", 1.0))
        if not all_skipped:
            actor_slices = micro_batch_ranges(group, actor_micro)
            n_chunk = max(len(actor_slices), 1)
            with dropout_eval(model):
                for pe in range(n_pe):
                    if pe % max(accum, 1) == 0:
                        opt.zero_grad(set_to_none=True)
                    pg_acc = 0.0
                    kl_acc = 0.0
                    rho_acc = 0.0
                    clip_acc = 0.0
                    for j, (s, e) in enumerate(actor_slices):
                        n = e - s
                        last_chunk = j == n_chunk - 1
                        do_step = ((pe + 1) % max(accum, 1) == 0) or (pe == n_pe - 1)
                        sync = bool(do_step and last_chunk)
                        ctx = (
                            model.no_sync()
                            if (use_ddp and not sync and hasattr(model, "no_sync"))
                            else nullcontext()
                        )
                        chunk_in = expand_gen_in_for_group(gen_in, n)
                        with ctx, bind_cached_image_features(model, vision_cache, repeat_factor=n):
                            out = forward_with_vision(model, chunk_in, outputs[s:e], attn[s:e])
                            if skipped:
                                loss = out.logits.float().sum() * 0.0
                                L_pg = L_kl = loss
                                rho_mean = torch.zeros((), device=device)
                                clip_frac = torch.zeros((), device=device)
                            else:
                                new_lp, mask = token_logprobs(out.logits, labels[s:e])
                                loss, L_pg, L_kl, rho_mean, clip_frac = clipped_pg_kl(
                                    new_lp,
                                    mask,
                                    old_lp[s:e],
                                    ref_lp[s:e],
                                    adv_tok[s:e],
                                    clip_low,
                                    clip_high,
                                    kl_beta,
                                )
                            w = n / float(group)
                            (loss * w / float(max(accum, 1))).backward()
                        pg_acc += float(L_pg.detach()) * w
                        kl_acc += float(L_kl.detach()) * w
                        rho_acc += float(rho_mean) * w
                        clip_acc += float(clip_frac) * w
                        del out
                    if ((pe + 1) % max(accum, 1) == 0) or (pe == n_pe - 1):
                        grads = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                        if grads:
                            last_gnorm = float(torch.nn.utils.clip_grad_norm_(grads, max_gnorm))
                        opt.step()
                        opt_step += 1
                        did_opt = True
                    loss_v += pg_acc + float(kl_beta) * kl_acc
                    pg_v += pg_acc
                    kl_v += kl_acc
                    rho_v = rho_acc
                    clip_v = clip_acc
            loss_v /= float(n_pe)
            pg_v /= float(n_pe)
            kl_v /= float(n_pe)
        t_upd = time.time() - t_upd

        loss_v = avg_across_ranks(loss_v, device)
        pg_v = avg_across_ranks(pg_v, device)
        kl_v = avg_across_ranks(kl_v, device)
        r_mean = avg_across_ranks(float(r_final.mean()), device)
        r_std = avg_across_ranks(float(r_final.std(unbiased=False)), device)
        parse_rate = avg_across_ranks(parse_rate_local, device)
        answer_tag_rate = avg_across_ranks(answer_tag_rate, device)
        final_bbox_rate = avg_across_ranks(final_bbox_rate, device)
        truncation_rate = avg_across_ranks(truncation_rate, device)
        cls_tag_rate = avg_across_ranks(cls_tag_rate, device)
        ref_gap = avg_across_ranks(ref_gap, device)
        proto_avg = {k: avg_across_ranks(float(v), device) for k, v in proto_stats.items()}

        if skipped and bool(meta.get("is_anomaly")) and not from_requeue:
            pending_retry = raw_batch

        # Only all-rank skips omit an official step. Mixed DDP must still advance
        # together so world size stays in the same while-loop / barrier cadence.
        if all_skipped:
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                dist.barrier()
            continue

        step += 1

        if is_main_process():
            _log_rollout_txt(tok, batch["input_ids"], prompt_len, meta, step, texts, details, parsed_list)
            log_grpo_scalars(
                writer,
                step=step,
                loss=loss_v,
                rewards=r_final,
                details=details,
                advantages=a_final,
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
                    "resample": float(resample_n),
                    "skipped": 1.0 if skipped else 0.0,
                },
            )
            if log_every > 0 and step % log_every == 0:
                n = max(group, 1)
                skip_s = " all_skip" if all_skipped else (" skip" if skipped else "")
                train_log(
                    f"[grpo] step={step}/{max_steps} att={attempt_step} opt={opt_step} loss={loss_v:.4f} "
                    f"pg={pg_v:.4f} kl={kl_v:.4f} rho={rho_v:.3f} clip={clip_v:.3f} "
                    f"Rf={r_mean:.3f}±{r_std:.3f} "
                    f"Rg={sum(d.get('R_ground', 0.0) for d in details)/n:.3f} "
                    f"Rr={sum(d.get('R_reason', 0.0) for d in details)/n:.3f} "
                    f"iou_f={sum(d.get('R_iou', 0.0) for d in details)/n:.3f} "
                    f"iou_c={sum(d.get('R_iou_c', 0.0) for d in details)/n:.3f} "
                    f"ans={answer_tag_rate:.2f} bbox={final_bbox_rate:.2f} "
                    f"traj={proto_avg.get('trajectory_valid_rate', 0.0):.2f} "
                    f"cand={proto_avg.get('candidate_valid_rate', 0.0):.2f} "
                    f"uniq={proto_avg.get('unique_response_rate', 0.0):.2f} "
                    f"dIoU={sum(d.get('delta_iou', 0.0) for d in details)/n:.3f} "
                    f"dir={sum(d.get('R_dir', 0.0) for d in details)/n:.2f} "
                    f"Rfmt={sum(d.get('R_fmt', 0.0) for d in details)/n:.2f} "
                    f"sz={size_bin} a={a_gt:.4f} "
                    f"trunc={truncation_rate:.2f} fmt={parse_rate:.2f} rs={resample_n}{skip_s} "
                    f"gen={t_roll:.1f}s upd={t_upd:.1f}s"
                    + (f" gnorm={last_gnorm:.2f}" if did_opt and last_gnorm is not None else ""),
                    main_only=True,
                )

        if is_main_process() and vis_every > 0 and step % vis_every == 0:
            def _vis_key(i):
                p = parsed_list[i]
                d = details[i]
                is_anom = bool(meta.get("is_anomaly"))

                cls_ok = (
                    p.get("is_anomaly") is not None
                    and bool(p.get("is_anomaly")) == is_anom
                )

                if is_anom:
                    spatial_ok = (
                        p.get("candidate_bbox_state") == "box"
                        and p.get("final_bbox_state") == "box"
                    )
                else:
                    spatial_ok = (
                        p.get("is_anomaly") is False
                        and p.get("final_bbox_state") == "null"
                    )

                return (
                    int(bool(p.get("trajectory_valid"))),
                    int(cls_ok),
                    int(spatial_ok),
                    float(d.get("R_final", -1.0)),
                    float(d.get("R_reason", 0.0)),
                    float(d.get("R_ground", 0.0)),
                    float(d.get("R_fmt", 0.0)),
                )

            best_i = max(range(len(texts)), key=_vis_key)
            log_heatmap_and_case(
                writer,
                step=step,
                tag_prefix="grpo_batch",
                meta=meta,
                response=texts[best_i],
                parsed=parsed_list[best_i],
                iou=float(details[best_i].get("R_iou", 0.0)),
                rec_ok=details[best_i].get("pred_cls") is not None
                and bool(details[best_i].get("pred_cls")) == bool(meta.get("is_anomaly")),
                overlay_alpha=alpha,
                iou_c=float(details[best_i].get("R_iou_c", 0.0)),
                **tb_vis_flags(cfg),
            )
            if writer is not None and bool(tb.get("log_grpo_cot", True)):
                writer.add_text(
                    "grpo_batch/4_group_cot",
                    format_grpo_group_text(
                        step=step,
                        meta=meta,
                        texts=texts,
                        details=details,
                        advantages=a_final.detach().cpu().tolist(),
                        logprobs=seq_lp.detach().cpu().tolist(),
                    ),
                    step,
                )
            if train_probe is not None:
                log_grpo_probe_cot(
                    cfg,
                    unwrap_model(model),
                    processor,
                    collator,
                    train_probe,
                    writer,
                    step,
                    max_new=max_new,
                    alpha=alpha,
                    tag_prefix="train_diagnostic_probe",
                )
            if log_mvtec_probe and crossdomain_probe is not None:
                log_grpo_probe_cot(
                    cfg,
                    unwrap_model(model),
                    processor,
                    collator,
                    crossdomain_probe,
                    writer,
                    step,
                    max_new=max_new,
                    alpha=alpha,
                    tag_prefix="crossdomain_probe",
                )
            if writer is not None:
                writer.flush()

        if is_main_process() and eval_every > 0 and eval_loader is not None and step % eval_every == 0:
            stats = run_simple_eval(
                cfg, unwrap_model(model), processor, eval_loader, writer=writer, global_step=step
            )
            train_log(
                f"[grpo-eval visadev] step={step} rec={stats['rec_acc']:.3f} "
                f"gated_mIoU={stats.get('mean_iou_gated', 0.0):.3f} "
                f"mean_iou_f={stats['mean_iou']:.3f} mean_iou_c={stats.get('mean_iou_c', 0.0):.3f} "
                f"acc@0.5={stats.get('acc_at_05', 0.0):.3f}",
                main_only=True,
            )
            os.makedirs(os.path.join(output_dir, "eval_steps"), exist_ok=True)
            compact = {k: v for k, v in stats.items() if k != "records"}
            with open(os.path.join(output_dir, "eval_steps", f"grpo_{step:08d}.json"), "w", encoding="utf-8") as f:
                json.dump(compact, f, ensure_ascii=False, indent=2)

        if is_main_process() and save_every > 0 and step % save_every == 0:
            ckpt = os.path.join(output_dir, f"grpo_checkpoint-{step}")
            os.makedirs(ckpt, exist_ok=True)
            unwrap_model(model).save_pretrained(ckpt)
            processor.save_pretrained(ckpt)

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

    mins = (time.time() - t0) / 60.0
    if is_main_process():
        train_log(f"GRPO 完成，耗时 {mins:.2f} 分钟", main_only=True)
        close_rollout_log()


def setup_prior_model(cfg: dict):
    model, processor = setup_model_and_processor(cfg, for_inference=False, freeze_vision=False)
    model = apply_lora(model, cfg)
    if bool(cfg.get("model", {}).get("freeze_vit", True)):
        freeze_vision_encoder(model)
        force_vision_eval(model)
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
    if is_main_process():
        n = sum(p.numel() for p in model.parameters())
        nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[prior] LoRA 可训练 {nt / 1e6:.2f}M / 总 {n / 1e6:.2f}M；vision 冻结", flush=True)
        print(f"[prior] vision blocks={prior.block_indices} tau={prior.temperature} radius={prior.neighborhood_radius}", flush=True)
    return model, processor, prior


def train_main(cfg: dict) -> None:
    disable_hf_datasets_check()
    set_seed(int(cfg["training"]["seed"]))
    is_main = is_main_process()
    output_dir = prepare_output_dir(
        base_dir=cfg["paths"]["output_dir"],
        run_name=cfg["runtime"]["run_name"],
        auto_create=cfg["runtime"]["auto_create_output_dir"],
    )
    cfg["paths"]["output_dir"] = output_dir
    if is_main:
        train_log(f"输出目录: {output_dir}", main_only=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        with open(os.path.join(output_dir, "config.yaml.txt"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)

    data_cfg = cfg.get("data") or {}
    train_layout = str(data_cfg.get("train_layout", "visa")).lower()
    eval_layout = str(data_cfg.get("eval_layout", "mvtec")).lower()
    if train_layout == "json" or eval_layout == "json":
        raise ValueError("旧 JSON 训练链路已移除，请用 data.train_layout=visa / eval_layout=mvtec 扫盘")
    train_samples, test_samples = load_prior_split(cfg)
    holdout = float((cfg.get("data") or {}).get("holdout_ratio", 0.1) or 0.0)
    train_samples, dev_samples = split_holdout_by_class(
        train_samples, holdout, seed=int(cfg["training"]["seed"])
    )
    train_ref_pool = build_train_ref_pool(train_samples)
    n_tr_anom = sum(1 for s in train_samples if bool((s.get("metadata") or {}).get("anomaly")))
    n_dev_anom = sum(1 for s in dev_samples if bool((s.get("metadata") or {}).get("anomaly")))
    n_ev_anom = sum(1 for s in test_samples if bool((s.get("metadata") or {}).get("anomaly")))
    if is_main:
        ratio = (cfg.get("data") or {}).get("normal_anomaly_ratio")
        train_log(
            f"数据协议: VisA train / VisA-dev holdout={holdout} 调参, MVTec test 仅最终评测；"
            f"normal:anomaly={ratio}；"
            f"train={len(train_samples)} (anom={n_tr_anom}) "
            f"visadev={len(dev_samples)} (anom={n_dev_anom}) "
            f"mvtec={len(test_samples)} (anom={n_ev_anom}) "
            f"train_ref_pool={sum(len(v) for v in train_ref_pool.values())} "
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

    train_set = PriorCoTDataset(train_samples, cfg, processor, mode="train", ref_pool=train_ref_pool)
    eval_collate = PriorCollator(processor, prior, cfg)
    eval_loader = None
    if dev_samples:
        dev_set = PriorCoTDataset(dev_samples, cfg, processor, mode="eval", ref_pool=train_ref_pool)
        eval_loader = DataLoader(dev_set, batch_size=1, shuffle=False, collate_fn=eval_collate, num_workers=0)
    test_set = PriorCoTDataset(test_samples, cfg, processor, mode="eval")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=eval_collate, num_workers=0)

    writer = None
    tb_dir = tensorboard_event_dir(output_dir)
    if is_main:
        writer = SummaryWriter(log_dir=tb_dir)
        auto_start_tensorboard(cfg, output_dir)
        print("开始 GRPO（无 SFT）…", flush=True)

    train_grpo(cfg, model, processor, prior, train_set, eval_loader, output_dir, writer)
    if is_main:
        gdir = os.path.join(output_dir, "grpo_final")
        unwrap_model(model).save_pretrained(gdir)
        processor.save_pretrained(gdir)
        print(f"GRPO 完成: {gdir}", flush=True)
        run_final_mvtec_eval(cfg, model, processor, test_loader, writer, output_dir, tag="grpo")

    if writer is not None:
        writer.close()
    if is_main:
        print(f"全部完成: {output_dir}", flush=True)
