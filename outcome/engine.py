"""Shared outcome evaluation and a bounded single-GPU training baseline."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
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
from outcome.inputs import OutcomeCollator, OutcomeDataset
from outcome.policy import generate_group, group_advantages, optimize_group
from outcome.protocol import VERSION, iou, parse_output, score_output, to_pixels, box_union_coverage
from outcome.visualize import log_outcome_eval_grid, log_outcome_single_case, render_outcome_case
from rl.grpo import avg_across_ranks, move_batch
from utils.common import is_main_process, set_seed


def validate_config(cfg):
    import os
    if cfg.get('outcome', {}).get('version') != VERSION:
        raise ValueError(f'expected outcome.version={VERSION}')
    if cfg.get('prompt', {}).get('enable_thinking', False):
        raise ValueError('outcome-v1 uses short explicit output: enable_thinking must be false')
    if not cfg['model'].get('freeze_vit', True) or not cfg['lora'].get('enabled', False):
        raise ValueError('outcome-v1 requires frozen ViT and language LoRA')
    gc = cfg['grpo']
    if (float(gc.get('temperature', 1)) != 1 or float(gc.get('top_p', 1)) != 1
            or int(gc.get('top_k', 0)) != 0):
        raise ValueError('raw-policy baseline requires temperature=1, top_p=1, top_k=0')
    if int(gc.get('policy_epochs', 1)) != 1 or int(gc.get('gradient_accumulation_steps', 1)) != 1:
        raise ValueError('outcome-v1 requires policy_epochs=1 and gradient_accumulation_steps=1')
    # outcome-v1 intentionally keeps a single update per group; opt into the
    # single-epoch path explicitly so optimize_group does not raise.
    gc['allow_single_epoch'] = True
    if int(gc['group_size']) < 2 or int(gc['max_new_tokens']) < 1:
        raise ValueError('group_size >= 2 and positive generation budget required')
    if gc.get('reward'):
        raise ValueError('remove legacy grpo.reward: outcome uses only outcome.protocol_weight')
    if not 0 <= float(cfg.get('outcome', {}).get('protocol_weight', .05)) <= .1:
        raise ValueError('protocol_weight must be in [0,.1]')
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
        # Evaluation only. Training resumes are intentionally not implied by --adapter.
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
    return (OutcomeDataset(train, cfg, processor, 'train', pool),
            OutcomeDataset(dev, cfg, processor, 'eval', pool),
            OutcomeDataset(test, cfg, processor, 'eval'))


def make_record(parsed, score, meta, completion, prompt_len, elapsed):
    anomaly = bool(meta['is_anomaly'])
    gt = meta.get('gt_box_px')
    area = ((gt[2]-gt[0])*(gt[3]-gt[1])/(meta['orig_size'][0]*meta['orig_size'][1])) if anomaly else 0
    candidates = meta.get('prior_candidates') or []
    if anomaly and gt is not None:
        h_ious = [iou(to_pixels(c['bbox_2d'], meta['orig_size']), gt) for c in candidates]
        iou_h_top1 = h_ious[0] if h_ious else None
        iou_h_bestk = max(h_ious) if h_ious else None
        h_union_cov = box_union_coverage(gt, [to_pixels(c['bbox_2d'], meta['orig_size']) for c in candidates])
    else:
        iou_h_top1 = iou_h_bestk = h_union_cov = None
    iou_c = (iou(to_pixels(parsed['candidate_bbox_2d'], meta['orig_size']), gt)
             if anomaly and parsed.get('candidate_bbox_2d') is not None and gt is not None else None)
    iou_f = (iou(to_pixels(parsed['bbox_2d'], meta['orig_size']), gt)
             if anomaly and gt is not None and parsed.get('bbox_2d') is not None else None)
    delta_refine = (iou_f - iou_c) if iou_f is not None and iou_c is not None else None
    return dict(image_path=meta['image_path'], ref_path=meta['ref_path'], class_name=meta['class_name'],
        is_anomaly=anomaly, pred=parsed['is_anomaly'], task_valid=parsed['task_valid'],
        protocol_core=parsed['protocol_core'], protocol_strict=parsed['protocol_strict'],
        candidate_state=parsed['candidate_state'], verify_action=parsed['verify_action'],
        iou=score['iou'], reward=score['total'], loc_reward=score['loc_reward'],
        gt_box_px=gt, bbox_2d=parsed['bbox_2d'], candidate_bbox_2d=parsed['candidate_bbox_2d'],
        iou_h_top1=iou_h_top1, iou_h_bestk=iou_h_bestk, iou_c=iou_c, iou_f=iou_f,
        delta_refine=delta_refine, h_union_cov=h_union_cov,
        s_center=score['s_center'], s_w=score['s_w'], s_h=score['s_h'], s_geo=score['s_geo'],
        size_bin='normal' if not anomaly else 'small' if area < .02 else 'medium' if area < .1 else 'large',
        prior_candidates=candidates, prior_condition=meta.get('prior_condition'),
        image_count=meta.get('image_count'), prompt_tokens=meta.get('prompt_tokens'),
        visual_tokens=meta.get('visual_tokens'), prior_hint_tokens=meta.get('prior_hint_tokens'),
        stop_reason=completion.stop_reason, new_tokens=len(completion.ids)-prompt_len,
        seconds=elapsed, text=completion.text)


def summarize(rows):
    def mean(values):
        values = list(values)
        return sum(values)/len(values) if values else None
    normal = [r for r in rows if not r['is_anomaly']]
    abnormal = [r for r in rows if r['is_anomaly']]
    recall = mean(r['pred'] is True for r in abnormal)
    tnr = mean(r['pred'] is False for r in normal)
    def zfill_iou(key):
        return mean((r[key] if r.get(key) is not None else 0.0) for r in abnormal)
    out = dict(n=len(rows), n_anomaly=len(abnormal), n_normal=len(normal),
        task_valid_rate=mean(r['task_valid'] for r in rows),
        protocol_core_rate=mean(r['protocol_core'] for r in rows),
        protocol_strict_rate=mean(r['protocol_strict'] for r in rows),
        anomaly_recall=recall, normal_fpr=mean(r['pred'] is True for r in normal),
        normal_correct_rate=tnr,
        invalid_decision_rate=mean(r['pred'] is None for r in rows),
        balanced_accuracy=(recall+tnr)/2 if recall is not None and tnr is not None else None,
        anomaly_gated_miou=mean(r['iou'] for r in abnormal),
        acc_at_05=mean(r['iou'] >= .5 for r in abnormal),
        truncation_rate=mean(r['stop_reason'] == 'length' for r in rows),
        mean_new_tokens=mean(r['new_tokens'] for r in rows), mean_seconds=mean(r['seconds'] for r in rows),
        mean_iou_h_top1=mean(r['iou_h_top1'] for r in abnormal if r['iou_h_top1'] is not None),
        mean_iou_h_bestk=mean(r['iou_h_bestk'] for r in abnormal if r['iou_h_bestk'] is not None),
        mean_h_union_cov=mean(r['h_union_cov'] for r in abnormal if r['h_union_cov'] is not None),
        prior_recall_at_01=mean((r['iou_h_bestk'] or 0.0) >= .1 for r in abnormal),
        prior_recall_at_03=mean((r['iou_h_bestk'] or 0.0) >= .3 for r in abnormal),
        candidate_box_valid_rate=mean(r['candidate_bbox_2d'] is not None for r in abnormal),
        final_box_valid_rate=mean(r['bbox_2d'] is not None for r in abnormal),
        mean_iou_c=mean(r['iou_c'] for r in abnormal if r['iou_c'] is not None),
        mean_iou_f=mean(r['iou_f'] for r in abnormal if r['iou_f'] is not None),
        iou_c_all=zfill_iou('iou_c'),
        iou_f_all=zfill_iou('iou_f'),
        mean_delta_refine=mean(r['delta_refine'] for r in rows if r['delta_refine'] is not None))
    for size in ('small','medium','large'):
        subset = [r for r in abnormal if r['size_bin'] == size]
        out[f'n_{size}'] = len(subset)
        out[f'miou_{size}'] = mean(r['iou'] for r in subset)
    for key in ('image_count','prompt_tokens','visual_tokens','prior_hint_tokens'):
        out[f'mean_{key}'] = mean(r[key] for r in rows if r.get(key) is not None)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r['class_name']].append(r)
    out['per_class'] = {c: {'n':len(rs), 'n_anomaly':sum(r['is_anomaly'] for r in rs),
        'miou':mean(r['iou'] for r in rs if r['is_anomaly']),
        'normal_fpr':mean(r['pred'] is True for r in rs if not r['is_anomaly'])} for c,rs in by_class.items()}
    out['macro_miou'] = mean(v['miou'] for v in out['per_class'].values() if v['miou'] is not None)
    return out


def stratified_eval_indices(dataset, count: int, seed: int = 42):
    """Class-balanced indices so a limited eval spans every class (not just the
    alphabetically-first class, which is what a plain ``[:count]`` slice yields for
    the class-sorted MVTec scan). Within each class, normal/anomaly are balanced as
    evenly as the quota allows.
    """
    samples = getattr(dataset, 'samples', None)
    if samples is None:
        return list(range(max(1, min(int(count), len(dataset)))))
    total = len(samples)
    count = max(1, min(int(count), total))
    if count >= total:
        return list(range(total))
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for i, s in enumerate(samples):
        cls = str((s.get('metadata') or {}).get('class') or 'object')
        by_cls[cls].append(i)
    classes = sorted(by_cls)
    per = count // len(classes)
    rem = count % len(classes)
    picked: list = []
    picked_set: set = set()
    for j, cls in enumerate(classes):
        quota = per + (1 if j < rem else 0)
        if quota <= 0:
            continue
        idxs = list(by_cls[cls])
        if len(idxs) <= quota:
            picked.extend(idxs)
            picked_set.update(idxs)
            continue
        norm = [i for i in idxs if not (samples[i].get('metadata') or {}).get('anomaly', False)]
        anom = [i for i in idxs if (samples[i].get('metadata') or {}).get('anomaly', False)]
        rng.shuffle(norm)
        rng.shuffle(anom)
        half = quota // 2
        sel = anom[:quota - half] + norm[:half]
        if len(sel) < quota:
            remaining = [i for i in idxs if i not in set(sel)]
            rng.shuffle(remaining)
            sel.extend(remaining[:quota - len(sel)])
        picked.extend(sel)
        picked_set.update(sel)
    if len(picked) < count:
        remaining = [i for i in range(total) if i not in picked_set]
        rng.shuffle(remaining)
        picked.extend(remaining[:count - len(picked)])
    return picked[:count]


def evaluate(cfg, model, processor, prior, dataset, output_path, limit=None, writer=None, step=0, namespace='dev', save_viz_dir=None, partial_every=10):
    """None means the entire supplied split. Each case is saved, not just averages.

    When ``save_viz_dir`` is set, each case also saves its heatmap panel, bbox
    overlay, and CoT text as ``NNNN_heatmap.png`` / ``NNNN_boxes.png`` /
    ``NNNN_text.txt`` under that directory.

    ``partial_every`` controls how often the running summary is refreshed to
    ``output_path`` (the ``.json`` stats file) so progress can be inspected
    before the checkpoint finishes; 0 disables partial writes (final only).
    """
    count = len(dataset) if limit is None else min(int(limit), len(dataset))
    if count <= 0:
        raise ValueError('evaluation split/limit must be nonempty')
    collator = OutcomeCollator(processor, prior, cfg)
    device = next(model.parameters()).device
    rows = []
    cases = []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    viz_dir = Path(save_viz_dir) if save_viz_dir else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
    partial_every = int(partial_every)
    indices = stratified_eval_indices(dataset, count, seed=int((cfg.get('training') or {}).get('seed', 42)))
    with output_path.with_suffix('.jsonl').open('w') as stream:
        for pos, index in enumerate(indices):
            started = time.perf_counter()
            batch = move_batch(collator([dataset[index]]), device)
            completion = generate_group(model, processor, batch, cfg)[0]
            parsed = parse_output(completion.text)
            meta = batch['_meta'][0]
            reward = score_output(parsed, meta, float(cfg['outcome']['protocol_weight']),
                                  cfg['outcome'].get('localization'))
            row = make_record(parsed, reward, meta, completion, int(batch['prompt_len'][0]), time.perf_counter()-started)
            rows.append(row)
            cases.append(dict(meta=meta, parsed=parsed, response=completion.text,
                              iou=reward['iou'], loc_reward=reward['loc_reward'],
                              correct=reward['correct']))
            stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
            if viz_dir is not None:
                panel, vis, cot = render_outcome_case(
                    meta=meta, response=completion.text, parsed=parsed,
                    iou=reward['iou'], loc_reward=reward['loc_reward'],
                    correct=reward['correct'], step=step)
                if panel is not None:
                    panel.save(viz_dir / f'{index:04d}_heatmap.png')
                if vis is not None:
                    vis.save(viz_dir / f'{index:04d}_boxes.png')
                (viz_dir / f'{index:04d}_text.txt').write_text(cot, encoding='utf-8')
            if pos % 10 == 0:
                print(f'[{namespace}] {pos+1}/{len(indices)}', flush=True)
            if partial_every > 0 and (pos + 1) % partial_every == 0:
                partial = summarize(rows)
                partial['_partial'] = True
                partial['_done'] = pos + 1
                partial['_total'] = len(indices)
                output_path.write_text(json.dumps(partial, ensure_ascii=False, indent=2))
    stats = summarize(rows)
    stats['_done'] = len(indices)
    stats['_total'] = len(indices)
    output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    if writer:
        for name, value in stats.items():
            if isinstance(value, (float,int)):
                writer.add_scalar(f'{namespace}/{name}', value, step)
        log_outcome_eval_grid(writer, step=step, cases=cases)
        writer.flush()
    return stats


class _NullWriter:
    """No-op SummaryWriter stand-in for non-main ranks (swallows all logging calls)."""

    def add_scalar(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def add_image(self, *a, **k): pass
    def flush(self): pass
    def close(self): pass


def run_train(cfg, model, processor, prior, train_set, dev_set, test_set, output_dir):
    oc, gc = cfg['outcome'], cfg['grpo']
    collator = OutcomeCollator(processor, prior, cfg)

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
        # Dump hyperparams into TEXT so they sit next to the curves.
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
        if oc.get('eval_before_train', True) and len(dev_set) and main_proc:
            evaluate(cfg, unwrap_model(model), processor, prior, dev_set, output_dir/'dev_initial.json',
                     cfg['training'].get('eval_num_samples'), writer, 0, 'dev')
        loc_cfg = oc.get('localization') or {}
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
                    parsed = [parse_output(c.text) for c in completions]
                    scores = [score_output(p, meta, float(oc['protocol_weight']), loc_cfg) for p in parsed]
                    task_rewards = torch.tensor([s['task'] for s in scores], device=device)
                    loc_rewards = torch.tensor([s['loc_reward'] for s in scores], device=device)
                    task_std = float(task_rewards.std(unbiased=False))
                    loc_std = float(loc_rewards.std(unbiased=False))
                    loc_range = float(loc_rewards.max() - loc_rewards.min())
                    # DifferAD-R1 style: resample only on localization collapse,
                    # not on classification-driven total-reward spread.
                    if not is_anomaly or loc_range >= min_loc_range:
                        break
                rewards = torch.tensor([s['total'] for s in scores], device=device)
                advantages = group_advantages(rewards, bool(gc.get('scale_rewards', False)))
                zero = bool(advantages.abs().max().item() <= 1e-8)
                if world > 1:
                    zero_t = torch.tensor([1 if zero else 0], device=device, dtype=torch.int64)
                    dist.all_reduce(zero_t, op=dist.ReduceOp.SUM)
                    all_zero = int(zero_t.item()) == world
                else:
                    all_zero = zero
                skipped += int(zero)
                loc_collapsed_group = float(is_anomaly and loc_range < min_loc_range)
                loc_nonzero_rate = float((loc_rewards > 1e-6).float().mean())
                metrics = dict(attempts=attempt, updates=updates, skipped_total=skipped, zero_advantage_group=float(zero),
                    reward_mean=float(rewards.mean()), reward_std=float(rewards.std(unbiased=False)),
                    task_reward_mean=sum(s['task'] for s in scores)/len(scores),
                    task_reward_std=task_std, resamples_used=resamples, loc_collapsed_group=loc_collapsed_group,
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
                # Log every attempted group BEFORE any skip or optimizer failure.
                rows = [make_record(p,s,meta,c,int(batch['prompt_len'][0]),0.) for p,s,c in zip(parsed,scores,completions)]
                stream.write(json.dumps(dict(attempt=attempt, update_before=updates, zero_advantage=zero,
                                             task_reward_std=task_std, resamples_used=resamples,
                                             loc_reward_mean=float(loc_rewards.mean()), loc_reward_std=loc_std,
                                             loc_reward_range=loc_range, loc_collapsed_group=loc_collapsed_group,
                                             advantages=advantages.cpu().tolist(), trajectories=rows), ensure_ascii=False)+'\n')
                stream.flush()
                for name,value in metrics.items():
                    writer.add_scalar(f'train/{name}', value, attempt)
                # Split curves by GT class so anomaly/normal trends are separable.
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
                writer.add_scalar('train/updates_after', updates, attempt)
                writer.add_scalar('train/seconds_per_attempt', time.perf_counter()-started, attempt)

                # --- richer per-attempt console line (loss + localization + H) ---
                def _mean(key, rows):
                    vals = [r[key] for r in rows if r.get(key) is not None]
                    return sum(vals) / len(vals) if vals else None

                anom_rows = [r for r in rows if r.get('is_anomaly')]
                mean_iou_f = _mean('iou_f', anom_rows)
                mean_iou_c = _mean('iou_c', anom_rows)
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
                    print(f'[outcome] a={attempt}/{attempts} up={updates} sk={skipped} '
                          f'rw={metrics["reward_mean"]:.3f} loc={_f(mean_loc)} lrng={loc_range:.4f} '
                          f'lz={loc_nonzero_rate:.2f} lc={loc_collapsed_group:.0f} '
                          f'iou_f={_f(mean_iou_f)} iou_c={_f(mean_iou_c)} '
                          f'delta={_f(mean_delta)} iou_h={_f(mean_iou_h)} '
                          f'rs={resamples} tv={metrics["task_valid_rate"]:.2f} '
                          f'pc={metrics["protocol_core_rate"]:.2f} ps={metrics["protocol_strict_rate"]:.2f} '
                          f'tok={metrics["mean_new_tokens"]:.0f} '
                          f'{loss_part} '
                          f'({time.perf_counter()-started:.1f}s)', flush=True)
                every = int(cfg['training'].get('eval_every_n_steps', 0))
                vis_every = int((cfg.get('tensorboard') or {}).get('vis_every_n_steps', 0) or 0)
                if main_proc and vis_every > 0 and attempt % vis_every == 0:
                    # Pick the best rollout of this group for a visual panel.
                    best = max(range(len(parsed)), key=lambda i: (
                        scores[i]['correct'],
                        scores[i]['iou'],
                        scores[i]['loc_reward'],
                    ))
                    log_outcome_single_case(
                        writer, step=attempt, meta=meta,
                        response=completions[best].text, parsed=parsed[best],
                        iou=scores[best]['iou'], loc_reward=scores[best]['loc_reward'],
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
        if main_proc and len(dev_set):
            evaluate(cfg, unwrap_model(model), processor, prior, dev_set, output_dir/'dev_final.json',
                     cfg['training'].get('eval_num_samples'), writer, attempts, 'dev_final')
        if main_proc and oc.get('final_test', False):
            evaluate(cfg, unwrap_model(model), processor, prior, test_set, output_dir/'test_final.json',
                     cfg['training'].get('final_eval_num_samples'), writer, attempts, 'test_final')
        if main_proc:
            (output_dir/'training_summary.json').write_text(json.dumps(dict(attempts=attempts,updates=updates,skipped=skipped), indent=2))
    finally:
        writer.close()
        if world > 1 and dist.is_initialized():
            dist.barrier()
