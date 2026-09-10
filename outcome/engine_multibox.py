"""outcome-multibox-v1 training loop (bounded, single/multi-GPU).

Same GRPO skeleton as outcome-v1 but with component-level metrics, set-level
reward and localization-collapse resampling (Range(R_set)). Evaluation lives in
``outcome/evaluate_multibox.py``; this module imports only the mid-training
"small eval" entry point.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from data.prior_dataset import build_train_ref_pool
from data.scan import load_prior_split, split_holdout_by_class
from models.anomaly_prior import AnomalyPrior
from models.lora import apply_lora
from models.qwen35 import setup_model_and_processor, freeze_vision_encoder, force_vision_eval, unwrap_model
from models.region_injection import attach_region_adapter, ensure_region_token, save_region_adapter
from outcome.evaluate_multibox import evaluate, make_record
from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
from outcome.policy import generate_group, group_advantages, optimize_group
from outcome.protocol_multibox import VERSION, parse_output, score_output
from outcome.visualize_multibox import log_outcome_single_case
from rl.grpo import avg_across_ranks, move_batch
from utils.common import is_main_process, set_seed


def validate_config(cfg):
    if cfg.get('outcome', {}).get('version') != VERSION:
        raise ValueError(f'expected outcome.version={VERSION}')
    if cfg.get('prompt', {}).get('enable_thinking', False):
        raise ValueError('outcome-multibox-v1 uses short explicit output: enable_thinking must be false')
    if not cfg['model'].get('freeze_vit', True) or not cfg['lora'].get('enabled', False):
        raise ValueError('outcome-multibox-v1 requires frozen ViT and language LoRA')
    gc = cfg['grpo']
    if (float(gc.get('temperature', 1)) != 1 or float(gc.get('top_p', 1)) != 1
            or int(gc.get('top_k', 0)) != 0):
        raise ValueError('raw-policy baseline requires temperature=1, top_p=1, top_k=0')
    if int(gc.get('policy_epochs', 1)) < 1 or int(gc.get('gradient_accumulation_steps', 1)) != 1:
        raise ValueError('outcome-multibox-v1 requires policy_epochs>=1 and gradient_accumulation_steps=1')
    if int(gc['group_size']) < 2 or int(gc['max_new_tokens']) < 1:
        raise ValueError('group_size >= 2 and positive generation budget required')
    if gc.get('reward'):
        raise ValueError('remove legacy grpo.reward: outcome uses only outcome.protocol_weight')
    if not 0 <= float(cfg.get('outcome', {}).get('protocol_weight', .01)) <= .1:
        raise ValueError('protocol_weight must be in [0,.1]')
    if int(cfg.get('outcome', {}).get('max_boxes', 16)) < 1:
        raise ValueError('outcome.max_boxes must be positive')
    if Path(cfg['model']['name'], 'adapter_config.json').exists():
        raise ValueError('model.name must be the base model; use outcome.sft_adapter or --adapter explicitly')


def load_model(cfg, adapter=None):
    from peft import PeftModel
    model, processor = setup_model_and_processor(cfg, for_inference=False, freeze_vision=True)
    ensure_region_token(processor, model)
    sft = cfg.get('outcome', {}).get('sft_adapter')
    if sft:
        model = PeftModel.from_pretrained(model, sft, is_trainable=False).merge_and_unload()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    else:
        model = apply_lora(model, cfg)
    freeze_vision_encoder(model)
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        model = model.to(local_rank)
    else:
        model = model.to('cpu')
    if not adapter and cfg['training'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    force_vision_eval(model)
    prior = AnomalyPrior.from_qwen(model, cfg)
    attach_region_adapter(model, prior, cfg, sft)
    return model, processor, prior


def datasets(cfg, processor):
    train, test = load_prior_split(cfg)
    train, dev = split_holdout_by_class(train, float(cfg['data']['holdout_ratio']), seed=int(cfg['training']['seed']))
    pool = build_train_ref_pool(train)
    return (OutcomeMultiboxDataset(train, cfg, processor, 'train', pool),
            OutcomeMultiboxDataset(dev, cfg, processor, 'eval', pool),
            OutcomeMultiboxDataset(test, cfg, processor, 'eval'))


class _NullWriter:
    """No-op SummaryWriter stand-in for non-main ranks (swallows all logging calls)."""

    def add_scalar(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def add_image(self, *a, **k): pass
    def flush(self): pass
    def close(self): pass


def run_train(cfg, model, processor, prior, train_set, dev_set, test_set, output_dir):
    oc, gc = cfg['outcome'], cfg['grpo']
    collator = OutcomeMultiboxCollator(processor, prior, cfg)

    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', str(local_rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    if world > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=bool(cfg.get('distributed', {}).get('ddp_find_unused_parameters', False)),
        )
    main_proc = is_main_process()
    device = next(model.parameters()).device

    if not len(train_set):
        raise ValueError('empty train set')
    requested = gc.get('max_attempts')
    total_attempts = int(requested) if requested is not None else math.ceil(len(train_set)*float(gc['epochs']))
    if total_attempts <= 0:
        raise ValueError('max_attempts must be positive')
    # `attempts` is per-rank; each rank processes a disjoint shard so the whole job
    # covers `total_attempts` samples with a `world`x wall-clock speedup.
    attempts = max(1, math.ceil(total_attempts / world))
    max_boxes = int(oc.get('max_boxes', 16))
    eval_which = str(cfg['training'].get('eval_split', 'dev')).lower()
    if eval_which == 'test':
        eval_set, eval_name = test_set, 'test'
    elif eval_which == 'dev':
        eval_set, eval_name = dev_set, 'dev'
    else:
        eval_set, eval_name = None, None
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(gc['learning_rate']), weight_decay=0.)
    if main_proc:
        writer = SummaryWriter(str(Path(output_dir)/'tb'))
        writer.add_text('outcome/0_config', json.dumps({
            'outcome': oc, 'grpo': gc, 'lora': cfg.get('lora'),
            'prior': cfg.get('prior'), 'training': cfg.get('training'),
            'tensorboard': cfg.get('tensorboard'),
        }, ensure_ascii=False, indent=2, default=str), 0)
        writer.flush()
    else:
        writer = _NullWriter()
    updates = skipped = 0
    rng = random.Random(int(cfg['training']['seed']))
    order = []
    output_dir = Path(output_dir)
    if main_proc:
        manifest = {name:[s.get('full_img_path') or s.get('image') for s in ds.samples]
                    for name,ds in [('train',train_set),('dev',dev_set),('test',test_set)]}
        (output_dir/'split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    try:
        loc_cfg = {**(oc.get('localization') or {}), **(oc.get('reward') or {})}
        resample_cfg = oc.get('resampling') or {}
        max_group_resamples = int(resample_cfg.get('max_group_resamples', 3))
        min_loc_range = float(resample_cfg.get('min_loc_range', 0.001))
        stream_path = output_dir/'rollouts.jsonl' if main_proc else None
        with (stream_path.open('w') if stream_path is not None else open(os.devnull, 'w')) as stream:
            for attempt in range(1, attempts+1):
                if not order:
                    idx = list(range(len(train_set)))
                    rng.shuffle(idx)
                    order = idx[rank::world]
                batch = move_batch(collator([train_set[order.pop()]]), device)
                started = time.perf_counter()
                meta = batch['_meta'][0]
                is_anomaly = bool(meta.get('is_anomaly'))
                completions = parsed = scores = None
                task_std = loc_std = loc_range = 0.0
                for resamples in range(max_group_resamples + 1):
                    completions = generate_group(model, processor, batch, cfg, group=int(gc['group_size']), sample=True)
                    parsed = [parse_output(c.text, max_boxes=max_boxes) for c in completions]
                    scores = [score_output(p, meta, float(oc['protocol_weight']), loc_cfg, max_boxes=max_boxes) for p in parsed]
                    task_rewards = torch.tensor([s['task'] for s in scores], device=device)
                    loc_rewards = torch.tensor([s['loc_reward'] for s in scores], device=device)
                    task_std = float(task_rewards.std(unbiased=False))
                    loc_std = float(loc_rewards.std(unbiased=False))
                    loc_range = float(loc_rewards.max() - loc_rewards.min())
                    # Set-level localization collapse: resample only when the
                    # anomaly group's R_set has (near) zero range.
                    if not is_anomaly or loc_range >= min_loc_range:
                        break
                rewards = torch.tensor([s['total'] for s in scores], device=device)
                # Advantage uses task reward only; the tiny protocol term (0.01) must
                # not leak into the advantage, or it becomes the sole (std-scaled)
                # gradient whenever the group's task rewards collapse to a constant.
                advantages = group_advantages(task_rewards, bool(gc.get('scale_rewards', False)))
                zero = bool(advantages.abs().max().item() <= 1e-8)
                if world > 1:
                    zero_t = torch.tensor([1 if zero else 0], device=device, dtype=torch.int64)
                    dist.all_reduce(zero_t, op=dist.ReduceOp.SUM)
                    all_zero = int(zero_t.item()) == world
                else:
                    all_zero = zero
                skipped += int(zero)
                loc_collapsed_group = float(is_anomaly and loc_range < min_loc_range)
                collapsed_after_resampling = bool(is_anomaly and loc_range < min_loc_range)
                loc_nonzero_rate = float((loc_rewards > 1e-6).float().mean())
                metrics = dict(attempts=attempt, updates=updates, skipped_total=skipped, zero_advantage_group=float(zero),
                    reward_mean=float(rewards.mean()), reward_std=float(rewards.std(unbiased=False)),
                    task_reward_mean=sum(s['task'] for s in scores)/len(scores),
                    task_reward_std=task_std, resamples_used=resamples, loc_collapsed_group=loc_collapsed_group,
                    collapsed_after_resampling=float(collapsed_after_resampling),
                    loc_reward_mean=float(loc_rewards.mean()), loc_reward_std=loc_std, loc_reward_range=loc_range,
                    loc_nonzero_rate=loc_nonzero_rate,
                    task_valid_rate=sum(p['task_valid'] for p in parsed)/len(parsed),
                    protocol_core_rate=sum(p['protocol_core'] for p in parsed)/len(parsed),
                    protocol_strict_rate=sum(p['protocol_strict'] for p in parsed)/len(parsed),
                    truncation_rate=sum(c.stop_reason == 'length' for c in completions)/len(completions))
                metrics.update(prompt_tokens=meta['prompt_tokens'], visual_tokens=meta['visual_tokens'],
                               prior_hint_tokens=meta['prior_hint_tokens'],
                               mean_new_tokens=sum(len(c.ids)-int(batch['prompt_len'][0]) for c in completions)/len(completions),
                               h_candidate_count=len(meta['prior_candidates']))
                # Average the per-sample metrics across ranks so TB/console show the
                # global value; the counters (attempts/updates/skipped_total) are kept.
                metrics = {k: (avg_across_ranks(float(v), device) if k not in ('attempts', 'updates', 'skipped_total') else v)
                           for k, v in metrics.items()}
                rows = [make_record(p,s,meta,c,int(batch['prompt_len'][0]),0.,max_boxes) for p,s,c in zip(parsed,scores,completions)]
                stream.write(json.dumps(dict(attempt=attempt, update_before=updates, zero_advantage=zero,
                                             task_reward_std=task_std, resamples_used=resamples,
                                             loc_reward_mean=float(loc_rewards.mean()), loc_reward_std=loc_std,
                                             loc_reward_range=loc_range, loc_collapsed_group=loc_collapsed_group,
                                             advantages=advantages.cpu().tolist(), trajectories=rows), ensure_ascii=False, default=str)+'\n')
                stream.flush()
                for name,value in metrics.items():
                    writer.add_scalar(f'train/{name}', value, attempt)
                split_prefix = 'train_anomaly' if is_anomaly else 'train_normal'
                for name,value in metrics.items():
                    writer.add_scalar(f'{split_prefix}/{name}', value, attempt)
                writer.flush()
                loss_stats = None
                if not all_zero:
                    loss_stats = optimize_group(model, processor, batch, completions, advantages, opt, cfg, skip=zero)
                    updates += 1
                    for name,value in loss_stats.items():
                        writer.add_scalar(f'optimizer/{name}', value, updates)
                metrics['collapsed_but_updated'] = float(collapsed_after_resampling and not zero)
                writer.add_scalar('train/collapsed_but_updated', metrics['collapsed_but_updated'], attempt)
                writer.add_scalar('train/updates_after', updates, attempt)
                writer.add_scalar('train/seconds_per_attempt', time.perf_counter()-started, attempt)

                def _mean(key, rows):
                    vals = [r[key] for r in rows if r.get(key) is not None]
                    return sum(vals) / len(vals) if vals else None
                anom_rows = [r for r in rows if r.get('is_anomaly')]
                mean_maskiou = _mean('mask_iou', anom_rows)
                mean_delta = _mean('delta_refine', anom_rows)
                mean_iou_h = _mean('iou_h_bestk', anom_rows)
                mean_loc = sum(s['loc_reward'] for s in scores) / len(scores)

                if loss_stats is not None:
                    ls = loss_stats
                    loss_part = (f"loss={ls.get('loss', float('nan')):.4f} "
                                 f"pg={ls.get('pg', float('nan')):.4f} "
                                 f"kl={ls.get('kl', float('nan')):.3f} "
                                 f"ratio={ls.get('ratio', float('nan')):.3f} "
                                 f"clip={ls.get('clip_fraction', float('nan')):.2f} "
                                 f"gnorm={ls.get('grad_norm', float('nan')):.2f}")
                else:
                    loss_part = 'loss=-- pg=-- kl=-- ratio=-- clip=-- gnorm=--'

                def _f(v):
                    return f'{v:.3f}' if v is not None else '--'

                if main_proc:
                    print(f'[multibox] a={attempt}/{attempts} up={updates} sk={skipped} '
                          f'rw={metrics["reward_mean"]:.3f} loc={_f(mean_loc)} lrng={loc_range:.4f} '
                          f'lz={loc_nonzero_rate:.2f} lc={loc_collapsed_group:.0f} '
                          f'maskiou={_f(mean_maskiou)} delta={_f(mean_delta)} iou_h={_f(mean_iou_h)} '
                          f'rs={resamples} tv={metrics["task_valid_rate"]:.2f} '
                          f'pc={metrics["protocol_core_rate"]:.2f} ps={metrics["protocol_strict_rate"]:.2f} '
                          f'tok={metrics["mean_new_tokens"]:.0f} '
                          f'{loss_part} '
                          f'({time.perf_counter()-started:.1f}s)', flush=True)
                every = int(cfg['training'].get('eval_every_n_steps', 0))
                vis_every = int((cfg.get('tensorboard') or {}).get('vis_every_n_steps', 0) or 0)
                if main_proc and vis_every > 0 and attempt % vis_every == 0:
                    best = max(range(len(parsed)), key=lambda i: (
                        scores[i]['correct'],
                        scores[i]['union_iou'],
                        scores[i]['loc_reward'],
                    ))
                    log_outcome_single_case(
                        writer, step=attempt, meta=meta,
                        response=completions[best].text, parsed=parsed[best],
                        union_iou=scores[best]['union_iou'], loc_reward=scores[best]['loc_reward'],
                        correct=scores[best]['correct'], tag_prefix='train_case',
                    )
                if every > 0 and attempt % every == 0 and eval_set is not None and len(eval_set):
                    if main_proc:
                        evaluate(cfg, unwrap_model(model), processor, prior, eval_set, output_dir/f'{eval_name}_{attempt:06d}.json',
                                 cfg['training'].get('eval_num_samples'), writer, attempt, eval_name)
                    if world > 1 and dist.is_initialized():
                        dist.barrier()
                save = int(gc.get('save_steps', 0))
                if save > 0 and attempt % save == 0:
                    if main_proc:
                        unwrap_model(model).save_pretrained(output_dir/f'checkpoint-{attempt}')
                        processor.save_pretrained(output_dir/f'checkpoint-{attempt}')
                        save_region_adapter(getattr(unwrap_model(model), 'region_adapter', None),
                                            output_dir/f'checkpoint-{attempt}'/'region_adapter.pt')
                    if world > 1 and dist.is_initialized():
                        dist.barrier()
        if main_proc:
            unwrap_model(model).save_pretrained(output_dir/'adapter_final')
            processor.save_pretrained(output_dir/'adapter_final')
            save_region_adapter(getattr(unwrap_model(model), 'region_adapter', None),
                                output_dir/'adapter_final'/'region_adapter.pt')
        if main_proc:
            (output_dir/'training_summary.json').write_text(json.dumps(dict(attempts=attempts,updates=updates,skipped=skipped), indent=2))
    finally:
        writer.close()
        if world > 1 and dist.is_initialized():
            dist.barrier()
