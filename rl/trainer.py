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

from data.prior_dataset import PriorCollator, PriorCoTDataset
from data.scan import load_prior_split
from evaluation.evaluator import (
    first_anomaly_item,
    log_grpo_probe_cot,
    run_final_mvtec_eval,
    run_simple_eval,
    tb_vis_flags,
)
from evaluation.metrics import is_truncated
from models.anomaly_prior import AnomalyPrior
from models.lora import apply_lora
from models.qwen35 import freeze_vision_encoder, setup_model_and_processor
from reasoning.parser import parse_cot_output
from reasoning.rewards import compute_rewards
from rl.grpo import (
    avg_across_ranks,
    build_segment_advantages,
    clipped_pg_kl,
    disable_adapter_ctx,
    dropout_eval,
    forward_with_vision,
    grpo_advantages,
    grpo_param_map,
    group_std,
    model_inputs,
    move_batch,
    resolve_max_steps,
    token_logprobs,
    token_logprobs_nograd,
    unwrap_model,
)
from utils.common import is_main_process, prepare_output_dir, set_seed, train_log
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


def train_grpo(cfg: dict, model, processor, prior, train_set, eval_loader, output_dir: str, writer: Optional[SummaryWriter]) -> None:
    gcfg = cfg.get("grpo") or {}
    inf = cfg.get("inference") or {}
    rew_cfg = gcfg.get("reward") or {}
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
    eval_every = int(cfg.get("training", {}).get("eval_every_n_steps", 50))
    save_every = int(gcfg.get("save_steps", 100))
    tb = cfg.get("tensorboard") or {}
    vis_every = int(tb.get("vis_every_n_steps", eval_every) or 0)
    log_every = int(gcfg.get("logging_steps", 1))
    alpha = float((cfg.get("prior") or {}).get("overlay_alpha", 0.45))
    adv_eps = float(gcfg.get("adv_eps", 1e-6))

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
    eval_ds = getattr(eval_loader, "dataset", None)
    crossdomain_probe = first_anomaly_item(eval_ds) if is_main_process() else None
    train_probe = first_anomaly_item(train_set) if is_main_process() else None
    it = iter(loader)
    model.train()
    t0 = time.time()
    epoch = 0
    opt_step = 0
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
        m = unwrap_model(model)
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
        return details, parsed_list

    for step in range(1, max_steps + 1):
        batch = move_batch(_next_batch(), device)
        gen_in = model_inputs(batch)
        prompt_len = int(batch["prompt_len"][0].item())
        pad_id = tok.pad_token_id or tok.eos_token_id
        meta = batch["_meta"][0]

        seqs, texts, details, parsed_list = [], [], [], []
        resample_n = 0
        skipped = False
        for attempt in range(max_resample + 1):
            seqs, texts = _generate_group(gen_in, prompt_len)
            details, parsed_list = _score_group(texts, meta)
            std_box = group_std([float(d.get("R_box", 0.0)) for d in details])
            std_g = group_std([float(d.get("R_ground", 0.0)) for d in details])
            std_r = group_std([float(d.get("R_reason", 0.0)) for d in details])
            std_c = group_std([float(d.get("R_cls", 0.0)) for d in details])
            if max(std_box, std_g, std_r, std_c) >= min_std:
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
        r_cls = torch.tensor([float(d.get("R_cls", 0.0)) for d in details], device=device, dtype=torch.float32)
        a_ground = grpo_advantages(r_ground, group, adv_eps)
        a_reason = grpo_advantages(r_reason, group, adv_eps)
        a_box = grpo_advantages(r_box, group, adv_eps)
        a_cls = grpo_advantages(r_cls, group, adv_eps)

        use_ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        active = torch.tensor([0 if skipped else 1], device=device, dtype=torch.int64)
        if use_ddp:
            dist.all_reduce(active, op=dist.ReduceOp.SUM)
        all_skipped = int(active.item()) == 0

        max_t = max(int(s.numel()) for s in seqs)
        outputs = torch.full((group, max_t), int(pad_id), device=device, dtype=torch.long)
        for i, s in enumerate(seqs):
            outputs[i, : s.numel()] = s.to(device)
        attn = (outputs != pad_id).long()
        labels = outputs.clone()
        labels[:, :prompt_len] = -100
        labels[outputs == pad_id] = -100

        if all_skipped:
            t_lp = max(max_t - 1, 1)
            old_lp = torch.zeros(group, t_lp, device=device)
            lp_mask = torch.zeros_like(old_lp)
            ref_lp = old_lp
            ref_gap = 0.0
            adv_tok = torch.zeros_like(old_lp)
            seq_lp = torch.zeros(group, device=device)
        else:
            old_lp, lp_mask = token_logprobs_nograd(model, gen_in, outputs, attn, labels)
            with disable_adapter_ctx(model):
                ref_lp, _ = token_logprobs_nograd(model, gen_in, outputs, attn, labels)
            ref_gap = float(((old_lp - ref_lp).abs() * lp_mask).sum() / lp_mask.sum().clamp(min=1))
            adv_tok = build_segment_advantages(
                tok, seqs, prompt_len, max_t, a_ground, a_reason, a_box, a_cls, rew_cfg, device
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
        parse_rate_local = sum(1 for p in parsed_list if p.get("has_tags")) / n_g

        loss_v = 0.0
        pg_v = 0.0
        kl_v = 0.0
        rho_v = 1.0
        clip_v = 0.0
        did_opt = False
        model.train()
        n_pe = max(policy_epochs, 1)
        opt.zero_grad(set_to_none=True)
        if not all_skipped:
            with dropout_eval(model):
                for pe in range(n_pe):
                    pe_loss = pe_pg = pe_kl = pe_rho = pe_clip = 0.0
                    for i in range(group):
                        ctx = (
                            model.no_sync()
                            if (use_ddp and i < group - 1 and hasattr(model, "no_sync"))
                            else nullcontext()
                        )
                        with ctx:
                            out = forward_with_vision(model, gen_in, outputs[i : i + 1], attn[i : i + 1])
                            if skipped:
                                loss = out.logits.float().sum() * 0.0
                                L_pg = L_kl = loss
                                rho_mean = torch.zeros((), device=device)
                                clip_frac = torch.zeros((), device=device)
                            else:
                                new_lp, mask = token_logprobs(out.logits, labels[i : i + 1])
                                loss, L_pg, L_kl, rho_mean, clip_frac = clipped_pg_kl(
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

        loss_v = avg_across_ranks(loss_v, device)
        pg_v = avg_across_ranks(pg_v, device)
        kl_v = avg_across_ranks(kl_v, device)
        r_mean = avg_across_ranks(float(r_box.mean()), device)
        r_std = avg_across_ranks(float(r_box.std(unbiased=False)), device)
        parse_rate = avg_across_ranks(parse_rate_local, device)
        answer_tag_rate = avg_across_ranks(answer_tag_rate, device)
        final_bbox_rate = avg_across_ranks(final_bbox_rate, device)
        truncation_rate = avg_across_ranks(truncation_rate, device)
        cls_tag_rate = avg_across_ranks(cls_tag_rate, device)
        ref_gap = avg_across_ranks(ref_gap, device)

        if is_main_process():
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
                    "A_cls": float(a_cls.mean()),
                    "R_cls": float(r_cls.mean()),
                    "R_iou_c": float(sum(d.get("R_iou_c", 0.0) for d in details) / max(group, 1)),
                    "skipped": 1.0 if skipped else 0.0,
                    "all_skipped": 1.0 if all_skipped else 0.0,
                    "resample": float(resample_n),
                    "policy_epochs": float(policy_epochs),
                    "answer_tag_rate": float(answer_tag_rate),
                    "final_bbox_rate": float(final_bbox_rate),
                    "cls_tag_rate": float(cls_tag_rate),
                    "truncation_rate": float(truncation_rate),
                    "old_ref_logprob_gap": float(ref_gap),
                },
            )
            if log_every > 0 and step % log_every == 0:
                n = max(group, 1)
                skip_s = " all_skip" if all_skipped else (" skip" if skipped else "")
                train_log(
                    f"[grpo] step={step}/{max_steps} opt={opt_step} loss={loss_v:.4f} "
                    f"pg={pg_v:.4f} kl={kl_v:.4f} rho={rho_v:.3f} clip={clip_v:.3f} "
                    f"Rb={r_mean:.3f}±{r_std:.3f} "
                    f"Rg={sum(d.get('R_ground', 0.0) for d in details)/n:.3f} "
                    f"Rr={sum(d.get('R_reason', 0.0) for d in details)/n:.3f} "
                    f"Rc={sum(d.get('R_cls', 0.0) for d in details)/n:.3f} "
                    f"iou_f={sum(d.get('R_iou', 0.0) for d in details)/n:.3f} "
                    f"iou_c={sum(d.get('R_iou_c', 0.0) for d in details)/n:.3f} "
                    f"ans={answer_tag_rate:.2f} bbox={final_bbox_rate:.2f} "
                    f"trunc={truncation_rate:.2f} fmt={parse_rate:.2f} rs={resample_n}{skip_s}"
                    + (f" gnorm={last_gnorm:.2f}" if did_opt and last_gnorm is not None else ""),
                    main_only=True,
                )

        if is_main_process() and vis_every > 0 and step % vis_every == 0:
            best_i = int(torch.argmax(r_cls + r_box).item())
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
                        advantages=a_box.detach().cpu().tolist(),
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
            if crossdomain_probe is not None:
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

        if is_main_process() and eval_every > 0 and step % eval_every == 0:
            stats = run_simple_eval(
                cfg, unwrap_model(model), processor, eval_loader, writer=writer, global_step=step
            )
            train_log(
                f"[grpo-eval] step={step} rec={stats['rec_acc']:.3f} "
                f"mean_iou_f={stats['mean_iou']:.3f} mean_iou_c={stats.get('mean_iou_c', 0.0):.3f}",
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


def setup_prior_model(cfg: dict):
    model, processor = setup_model_and_processor(cfg, for_inference=False, freeze_vision=False)
    model = apply_lora(model, cfg)
    if bool(cfg.get("model", {}).get("freeze_vit", True)):
        freeze_vision_encoder(model)
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
    train_samples, eval_samples = load_prior_split(cfg)
    n_tr_anom = sum(1 for s in train_samples if bool((s.get("metadata") or {}).get("anomaly")))
    n_ev_anom = sum(1 for s in eval_samples if bool((s.get("metadata") or {}).get("anomaly")))
    if is_main:
        ratio = (cfg.get("data") or {}).get("normal_anomaly_ratio")
        train_log(
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

    train_set = PriorCoTDataset(train_samples, cfg, processor, mode="train")
    eval_set = PriorCoTDataset(eval_samples, cfg, processor, mode="eval")
    eval_collate = PriorCollator(processor, prior, cfg)
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, collate_fn=eval_collate, num_workers=0)

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
        run_final_mvtec_eval(cfg, model, processor, eval_loader, writer, output_dir, tag="grpo")

    if writer is not None:
        writer.close()
    if is_main:
        print(f"全部完成: {output_dir}", flush=True)
