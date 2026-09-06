"""outcome-multibox-v1 evaluation + bounded single-GPU training loop.

Same GRPO skeleton as outcome-v1 but with component-level metrics, set-level
reward and localization-collapse resampling (Range(R_set)).
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from data.prior_dataset import build_train_ref_pool
from data.scan import load_prior_split, split_holdout_by_class
from models.anomaly_prior import AnomalyPrior
from models.lora import apply_lora
from models.qwen35 import setup_model_and_processor, freeze_vision_encoder, force_vision_eval
from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
from outcome.metrics import component_metrics, union_iou
from outcome.policy import generate_group, group_advantages, optimize_group
from outcome.protocol import iou, to_pixels
from outcome.protocol_multibox import VERSION, parse_output, score_output
from outcome.visualize_multibox import log_outcome_eval_grid, log_outcome_single_case
from rl.grpo import move_batch
from utils.common import set_seed


def validate_config(cfg):
    import os
    if cfg.get('outcome', {}).get('version') != VERSION:
        raise ValueError(f'expected outcome.version={VERSION}')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or int(cfg.get('distributed', {}).get('num_gpu', 1)) != 1:
        raise ValueError('outcome-multibox-v1 currently supports one GPU; use --num-gpu 1.')
    if cfg.get('prompt', {}).get('enable_thinking', False):
        raise ValueError('outcome-multibox-v1 uses short explicit output: enable_thinking must be false')
    if not cfg['model'].get('freeze_vit', True) or not cfg['lora'].get('enabled', False):
        raise ValueError('outcome-multibox-v1 requires frozen ViT and language LoRA')
    gc = cfg['grpo']
    if (float(gc.get('temperature', 1)) != 1 or float(gc.get('top_p', 1)) != 1
            or int(gc.get('top_k', 0)) != 0):
        raise ValueError('raw-policy baseline requires temperature=1, top_p=1, top_k=0')
    if int(gc.get('policy_epochs', 1)) != 1 or int(gc.get('gradient_accumulation_steps', 1)) != 1:
        raise ValueError('outcome-multibox-v1 requires policy_epochs=1 and gradient_accumulation_steps=1')
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
    sft = cfg.get('outcome', {}).get('sft_adapter')
    if sft:
        model = PeftModel.from_pretrained(model, sft, is_trainable=False).merge_and_unload()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    else:
        model = apply_lora(model, cfg)
    freeze_vision_encoder(model)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    if not adapter and cfg['training'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    force_vision_eval(model)
    return model, processor, AnomalyPrior.from_qwen(model, cfg)


def datasets(cfg, processor):
    train, test = load_prior_split(cfg)
    train, dev = split_holdout_by_class(train, float(cfg['data']['holdout_ratio']), seed=int(cfg['training']['seed']))
    pool = build_train_ref_pool(train)
    return (OutcomeMultiboxDataset(train, cfg, processor, 'train', pool),
            OutcomeMultiboxDataset(dev, cfg, processor, 'eval', pool),
            OutcomeMultiboxDataset(test, cfg, processor, 'eval'))


def _size_bin(meta, anomaly):
    if not anomaly:
        return 'normal'
    frac = meta.get('mask_area_fraction')
    if frac is None:
        gt = meta.get('gt_box_px')
        frac = (gt[2]-gt[0])*(gt[3]-gt[1])/(meta['orig_size'][0]*meta['orig_size'][1]) if gt else 0.0
    return 'small' if frac < .02 else 'medium' if frac < .1 else 'large'


def _component_bins(meta, anomaly):
    """'normal' | 'single' (1 component) | 'multi' (>=2 components)."""
    if not anomaly:
        return 'normal'
    return 'multi' if int(meta.get('num_components') or 1) >= 2 else 'single'


def make_record(parsed, score, meta, completion, prompt_len, elapsed):
    anomaly = bool(meta['is_anomaly'])
    gt = meta.get('gt_box_px')
    comps = list(meta.get('component_bboxes') or [])
    if anomaly and not comps and gt is not None:
        comps = [gt]
    orig = meta['orig_size']
    pred_px = [to_pixels(b, orig) for b in parsed['bboxes_2d']]
    cand_px = [to_pixels(b, orig) for b in parsed['candidate_bboxes_2d']]

    candidates = meta.get('prior_candidates') or []
    if anomaly and comps:
        h_ious = []
        for c in candidates:
            cb = to_pixels(c['bbox_2d'], orig)
            h_ious.append(max((iou(cb, g) for g in comps), default=0.0))
        iou_h_top1 = h_ious[0] if h_ious else None
        iou_h_bestk = max(h_ious) if h_ious else None
    else:
        iou_h_top1 = iou_h_bestk = None

    cm = component_metrics(pred_px, comps) if anomaly else None
    rec = dict(image_path=meta['image_path'], ref_path=meta['ref_path'], class_name=meta['class_name'],
        is_anomaly=anomaly, pred=parsed['is_anomaly'], task_valid=parsed['task_valid'],
        protocol_core=parsed['protocol_core'], protocol_strict=parsed['protocol_strict'],
        candidate_state=parsed['candidate_state'], verify_action=parsed['verify_action'],
        reward=score['total'], loc_reward=score['loc_reward'], set_c_reward=score['set_c_reward'],
        set_f_reward=score['set_f_reward'], delta_refine=score['delta_refine'],
        union_iou=score['union_iou'], gt_box_px=gt, bboxes_2d=parsed['bboxes_2d'],
        candidate_bboxes_2d=parsed['candidate_bboxes_2d'], num_boxes=parsed['num_boxes'],
        num_components=int(meta.get('num_components') or (len(comps) if anomaly else 0)),
        iou_h_top1=iou_h_top1, iou_h_bestk=iou_h_bestk,
        size_bin=_size_bin(meta, anomaly), component_bin=_component_bins(meta, anomaly),
        prior_candidates=candidates, roi=meta.get('roi'), prior_condition=meta.get('prior_condition'),
        image_count=meta.get('image_count'), prompt_tokens=meta.get('prompt_tokens'),
        visual_tokens=meta.get('visual_tokens'), prior_hint_tokens=meta.get('prior_hint_tokens'),
        stop_reason=completion.stop_reason, new_tokens=len(completion.ids)-prompt_len,
        seconds=elapsed, text=completion.text)
    if cm is not None:
        rec.update(matched_miou=cm['matched_miou'], count_error=cm['count_error'],
                   recall_at_01=cm['recall_at_01'], recall_at_03=cm['recall_at_03'], recall_at_05=cm['recall_at_05'],
                   precision_at_01=cm['precision_at_01'], precision_at_03=cm['precision_at_03'],
                   precision_at_05=cm['precision_at_05'])
    return rec


def summarize(rows):
    def mean(values):
        values = list(values)
        return sum(values)/len(values) if values else None
    normal = [r for r in rows if not r['is_anomaly']]
    abnormal = [r for r in rows if r['is_anomaly']]
    recall = mean(r['pred'] is True for r in abnormal)
    tnr = mean(r['pred'] is False for r in normal)
    out = dict(n=len(rows), n_anomaly=len(abnormal), n_normal=len(normal),
        task_valid_rate=mean(r['task_valid'] for r in rows),
        protocol_core_rate=mean(r['protocol_core'] for r in rows),
        protocol_strict_rate=mean(r['protocol_strict'] for r in rows),
        anomaly_recall=recall, normal_fpr=mean(r['pred'] is True for r in normal),
        normal_correct_rate=tnr,
        invalid_decision_rate=mean(r['pred'] is None for r in rows),
        balanced_accuracy=(recall+tnr)/2 if recall is not None and tnr is not None else None,
        union_miou=mean(r['union_iou'] for r in abnormal),
        union_acc_at_05=mean(r['union_iou'] >= .5 for r in abnormal),
        truncation_rate=mean(r['stop_reason'] == 'length' for r in rows),
        mean_new_tokens=mean(r['new_tokens'] for r in rows), mean_seconds=mean(r['seconds'] for r in rows),
        matched_miou=mean(r['matched_miou'] for r in abnormal if r.get('matched_miou') is not None),
        recall_at_01=mean(r['recall_at_01'] for r in abnormal if r.get('recall_at_01') is not None),
        recall_at_03=mean(r['recall_at_03'] for r in abnormal if r.get('recall_at_03') is not None),
        recall_at_05=mean(r['recall_at_05'] for r in abnormal if r.get('recall_at_05') is not None),
        precision_at_01=mean(r['precision_at_01'] for r in abnormal if r.get('precision_at_01') is not None),
        precision_at_03=mean(r['precision_at_03'] for r in abnormal if r.get('precision_at_03') is not None),
        precision_at_05=mean(r['precision_at_05'] for r in abnormal if r.get('precision_at_05') is not None),
        count_error=mean(r['count_error'] for r in abnormal if r.get('count_error') is not None),
        mean_set_f_reward=mean(r['set_f_reward'] for r in abnormal),
        mean_set_c_reward=mean(r['set_c_reward'] for r in abnormal),
        mean_delta_refine=mean(r['delta_refine'] for r in rows if r['delta_refine'] is not None),
        mean_iou_h_top1=mean(r['iou_h_top1'] for r in abnormal if r['iou_h_top1'] is not None),
        mean_iou_h_bestk=mean(r['iou_h_bestk'] for r in abnormal if r['iou_h_bestk'] is not None),
        prior_recall_at_01=mean((r['iou_h_bestk'] or 0.0) >= .1 for r in abnormal),
        prior_recall_at_03=mean((r['iou_h_bestk'] or 0.0) >= .3 for r in abnormal),
        mean_num_boxes=mean(r['num_boxes'] for r in abnormal),
        mean_num_components=mean(r['num_components'] for r in abnormal))
    for size in ('small','medium','large'):
        subset = [r for r in abnormal if r['size_bin'] == size]
        out[f'n_{size}'] = len(subset)
        out[f'matched_miou_{size}'] = mean(r['matched_miou'] for r in subset if r.get('matched_miou') is not None)
        out[f'union_miou_{size}'] = mean(r['union_iou'] for r in subset)
    for cb in ('single','multi'):
        subset = [r for r in abnormal if r['component_bin'] == cb]
        out[f'n_{cb}'] = len(subset)
        out[f'matched_miou_{cb}'] = mean(r['matched_miou'] for r in subset if r.get('matched_miou') is not None)
        out[f'union_miou_{cb}'] = mean(r['union_iou'] for r in subset)
        out[f'recall_at_05_{cb}'] = mean(r['recall_at_05'] for r in subset if r.get('recall_at_05') is not None)
    for key in ('image_count','prompt_tokens','visual_tokens','prior_hint_tokens'):
        out[f'mean_{key}'] = mean(r[key] for r in rows if r.get(key) is not None)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r['class_name']].append(r)
    out['per_class'] = {c: {'n':len(rs), 'n_anomaly':sum(r['is_anomaly'] for r in rs),
        'matched_miou':mean(r['matched_miou'] for r in rs if r['is_anomaly'] and r.get('matched_miou') is not None),
        'union_miou':mean(r['union_iou'] for r in rs if r['is_anomaly']),
        'normal_fpr':mean(r['pred'] is True for r in rs if not r['is_anomaly'])} for c,rs in by_class.items()}
    out['macro_matched_miou'] = mean(v['matched_miou'] for v in out['per_class'].values() if v['matched_miou'] is not None)
    out['macro_union_miou'] = mean(v['union_miou'] for v in out['per_class'].values() if v['union_miou'] is not None)
    return out


def evaluate(cfg, model, processor, prior, dataset, output_path, limit=None, writer=None, step=0, namespace='dev'):
    count = len(dataset) if limit is None else min(int(limit), len(dataset))
    if count <= 0:
        raise ValueError('evaluation split/limit must be nonempty')
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    device = next(model.parameters()).device
    rows = []
    cases = []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_boxes = int(cfg['outcome'].get('max_boxes', 16))
    with output_path.with_suffix('.jsonl').open('w') as stream:
        for index in range(count):
            started = time.perf_counter()
            batch = move_batch(collator([dataset[index]]), device)
            completion = generate_group(model, processor, batch, cfg)[0]
            parsed = parse_output(completion.text, max_boxes=max_boxes)
            meta = batch['_meta'][0]
            reward = score_output(parsed, meta, float(cfg['outcome']['protocol_weight']),
                                  cfg['outcome'].get('localization'), max_boxes=max_boxes)
            row = make_record(parsed, reward, meta, completion, int(batch['prompt_len'][0]), time.perf_counter()-started)
            rows.append(row)
            cases.append(dict(meta=meta, parsed=parsed, response=completion.text,
                              union_iou=reward['union_iou'], loc_reward=reward['loc_reward'],
                              correct=reward['correct']))
            stream.write(json.dumps(row, ensure_ascii=False, default=str)+'\n'); stream.flush()
            if index % 10 == 0:
                print(f'[{namespace}] {index+1}/{count}', flush=True)
    stats = summarize(rows)
    output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    if writer:
        for name, value in stats.items():
            if isinstance(value, (float,int)):
                writer.add_scalar(f'{namespace}/{name}', value, step)
        log_outcome_eval_grid(writer, step=step, cases=cases)
        writer.flush()
    return stats


def run_train(cfg, model, processor, prior, train_set, dev_set, test_set, output_dir):
    oc, gc = cfg['outcome'], cfg['grpo']
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    device = next(model.parameters()).device
    if not len(train_set):
        raise ValueError('empty train set')
    requested = gc.get('max_attempts')
    attempts = int(requested) if requested is not None else math.ceil(len(train_set)*float(gc['epochs']))
    if attempts <= 0:
        raise ValueError('max_attempts must be positive')
    max_boxes = int(oc.get('max_boxes', 16))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(gc['learning_rate']), weight_decay=0.)
    writer = SummaryWriter(str(Path(output_dir)/'tb'))
    writer.add_text('outcome/0_config', json.dumps({
        'outcome': oc, 'grpo': gc, 'lora': cfg.get('lora'),
        'prior': cfg.get('prior'), 'training': cfg.get('training'),
        'tensorboard': cfg.get('tensorboard'),
    }, ensure_ascii=False, indent=2, default=str), 0)
    writer.flush()
    updates = skipped = 0
    rng = random.Random(int(cfg['training']['seed']))
    order = []
    output_dir = Path(output_dir)
    manifest = {name:[s.get('full_img_path') or s.get('image') for s in ds.samples]
                for name,ds in [('train',train_set),('dev',dev_set),('test',test_set)]}
    (output_dir/'split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    try:
        if oc.get('eval_before_train', True) and len(dev_set):
            evaluate(cfg, model, processor, prior, dev_set, output_dir/'dev_initial.json',
                     cfg['training'].get('eval_num_samples'), writer, 0, 'dev')
        loc_cfg = oc.get('localization') or {}
        resample_cfg = oc.get('resampling') or {}
        max_group_resamples = int(resample_cfg.get('max_group_resamples', 3))
        min_loc_range = float(resample_cfg.get('min_loc_range', 0.001))
        with (output_dir/'rollouts.jsonl').open('w') as stream:
            for attempt in range(1, attempts+1):
                if not order:
                    order = list(range(len(train_set))); rng.shuffle(order)
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
                advantages = group_advantages(rewards, bool(gc.get('scale_rewards', False)))
                zero = bool(advantages.abs().max().item() <= 1e-8)
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
                rows = [make_record(p,s,meta,c,int(batch['prompt_len'][0]),0.) for p,s,c in zip(parsed,scores,completions)]
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
                if not zero:
                    loss_stats = optimize_group(model, processor, batch, completions, advantages, opt, cfg)
                    updates += 1
                    for name,value in loss_stats.items():
                        writer.add_scalar(f'optimizer/{name}', value, updates)
                writer.add_scalar('train/updates_after', updates, attempt)
                writer.add_scalar('train/seconds_per_attempt', time.perf_counter()-started, attempt)

                def _mean(key, rows):
                    vals = [r[key] for r in rows if r.get(key) is not None]
                    return sum(vals) / len(vals) if vals else None
                anom_rows = [r for r in rows if r.get('is_anomaly')]
                mean_union = _mean('union_iou', anom_rows)
                mean_mmiou = _mean('matched_miou', anom_rows)
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

                print(f'[multibox] a={attempt}/{attempts} up={updates} sk={skipped} '
                      f'rw={metrics["reward_mean"]:.3f} loc={_f(mean_loc)} lrng={loc_range:.4f} '
                      f'lz={loc_nonzero_rate:.2f} lc={loc_collapsed_group:.0f} '
                      f'union={_f(mean_union)} mmiou={_f(mean_mmiou)} delta={_f(mean_delta)} iou_h={_f(mean_iou_h)} '
                      f'rs={resamples} tv={metrics["task_valid_rate"]:.2f} '
                      f'pc={metrics["protocol_core_rate"]:.2f} ps={metrics["protocol_strict_rate"]:.2f} '
                      f'tok={metrics["mean_new_tokens"]:.0f} '
                      f'{loss_part} '
                      f'({time.perf_counter()-started:.1f}s)', flush=True)
                every = int(cfg['training'].get('eval_every_n_steps', 0))
                vis_every = int((cfg.get('tensorboard') or {}).get('vis_every_n_steps', 0) or 0)
                if vis_every > 0 and attempt % vis_every == 0:
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
                if every > 0 and attempt % every == 0 and len(dev_set):
                    evaluate(cfg, model, processor, prior, dev_set, output_dir/f'dev_{attempt:06d}.json',
                             cfg['training'].get('eval_num_samples'), writer, attempt, 'dev')
                save = int(gc.get('save_steps', 0))
                if save > 0 and attempt % save == 0:
                    model.save_pretrained(output_dir/f'checkpoint-{attempt}')
                    processor.save_pretrained(output_dir/f'checkpoint-{attempt}')
        model.save_pretrained(output_dir/'adapter_final')
        processor.save_pretrained(output_dir/'adapter_final')
        if len(dev_set):
            evaluate(cfg, model, processor, prior, dev_set, output_dir/'dev_final.json',
                     cfg['training'].get('eval_num_samples'), writer, attempts, 'dev_final')
        if oc.get('final_test', False):
            evaluate(cfg, model, processor, prior, test_set, output_dir/'test_final.json',
                     cfg['training'].get('final_eval_num_samples'), writer, attempts, 'test_final')
        (output_dir/'training_summary.json').write_text(json.dumps(dict(attempts=attempts,updates=updates,skipped=skipped), indent=2))
    finally:
        writer.close()
