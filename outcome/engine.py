"""Shared outcome evaluation and a bounded single-GPU training baseline."""
from __future__ import annotations

import hashlib
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
from outcome.inputs import OutcomeCollator, OutcomeDataset
from outcome.policy import generate_group, group_advantages, optimize_group
from outcome.protocol import VERSION, iou, parse_output, score_output, to_pixels
from rl.grpo import move_batch
from utils.common import set_seed


def validate_config(cfg):
    import os
    if cfg.get('outcome', {}).get('version') != VERSION:
        raise ValueError(f'expected outcome.version={VERSION}')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or int(cfg.get('distributed', {}).get('num_gpu', 1)) != 1:
        raise ValueError('outcome-v1 currently supports one GPU; use --num-gpu 1. Legacy DDP is unchanged.')
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
    if int(gc['group_size']) < 2 or int(gc['max_new_tokens']) < 1:
        raise ValueError('group_size >= 2 and positive generation budget required')
    if gc.get('reward'):
        raise ValueError('remove legacy grpo.reward: outcome uses only outcome.protocol_weight')
    if not 0 <= float(cfg.get('outcome', {}).get('protocol_weight', .05)) <= .1:
        raise ValueError('protocol_weight must be in [0,.1]')
    if float(cfg.get('outcome', {}).get('roi', {}).get('margin', .25)) < 0:
        raise ValueError('ROI margin must be nonnegative')
    if Path(cfg['model']['name'], 'adapter_config.json').exists():
        raise ValueError('model.name must be the base model; use outcome.sft_adapter or --adapter explicitly')


def load_model(cfg, adapter=None):
    from peft import PeftModel
    model, processor = setup_model_and_processor(cfg, for_inference=False, freeze_vision=True)
    sft = cfg.get('outcome', {}).get('sft_adapter')
    if sft:
        model = PeftModel.from_pretrained(model, sft, is_trainable=False).merge_and_unload()
    if adapter:
        # Evaluation only. Training resumes are intentionally not implied by --adapter.
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
    return (OutcomeDataset(train, cfg, processor, 'train', pool),
            OutcomeDataset(dev, cfg, processor, 'eval', pool),
            OutcomeDataset(test, cfg, processor, 'eval'))


def make_record(parsed, score, meta, completion, prompt_len, elapsed):
    anomaly = bool(meta['is_anomaly'])
    gt = meta.get('gt_box_px')
    area = ((gt[2]-gt[0])*(gt[3]-gt[1])/(meta['orig_size'][0]*meta['orig_size'][1])) if anomaly else 0
    candidates = meta.get('prior_candidates') or []
    candidate_iou = iou(to_pixels(candidates[0]['bbox_2d'], meta['orig_size']), gt) if candidates and anomaly else None
    return dict(image_path=meta['image_path'], ref_path=meta['ref_path'], class_name=meta['class_name'],
        is_anomaly=anomaly, pred=parsed['is_anomaly'], task_valid=parsed['task_valid'],
        protocol_valid=parsed['protocol_valid'], iou=score['iou'], reward=score['total'],
        gt_box_px=gt, bbox_2d=parsed['bbox_2d'], candidate_iou=candidate_iou,
        candidate_to_final_delta=(score['iou']-candidate_iou) if candidate_iou is not None else None,
        size_bin='normal' if not anomaly else 'small' if area < .02 else 'medium' if area < .1 else 'large',
        prior_candidates=candidates, roi=meta.get('roi'), prior_condition=meta.get('prior_condition'),
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
    out = dict(n=len(rows), n_anomaly=len(abnormal), n_normal=len(normal),
        task_valid_rate=mean(r['task_valid'] for r in rows),
        protocol_valid_rate=mean(r['protocol_valid'] for r in rows),
        anomaly_recall=recall, normal_fpr=mean(r['pred'] is True for r in normal),
        normal_correct_rate=tnr,
        invalid_decision_rate=mean(r['pred'] is None for r in rows),
        balanced_accuracy=(recall+tnr)/2 if recall is not None and tnr is not None else None,
        anomaly_gated_miou=mean(r['iou'] for r in abnormal),
        acc_at_05=mean(r['iou'] >= .5 for r in abnormal),
        truncation_rate=mean(r['stop_reason'] == 'length' for r in rows),
        mean_new_tokens=mean(r['new_tokens'] for r in rows), mean_seconds=mean(r['seconds'] for r in rows),
        mean_candidate_to_final_delta=mean(r['candidate_to_final_delta'] for r in rows if r['candidate_to_final_delta'] is not None))
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


def evaluate(cfg, model, processor, prior, dataset, output_path, limit=None, writer=None, step=0, namespace='dev'):
    """None means the entire supplied split. Each case is saved, not just averages."""
    count = len(dataset) if limit is None else min(int(limit), len(dataset))
    if count <= 0:
        raise ValueError('evaluation split/limit must be nonempty')
    collator = OutcomeCollator(processor, prior, cfg)
    device = next(model.parameters()).device
    rows = []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.with_suffix('.jsonl').open('w') as stream:
        for index in range(count):
            started = time.perf_counter()
            batch = move_batch(collator([dataset[index]]), device)
            completion = generate_group(model, processor, batch, cfg)[0]
            parsed = parse_output(completion.text)
            meta = batch['_meta'][0]
            reward = score_output(parsed, meta, float(cfg['outcome']['protocol_weight']))
            row = make_record(parsed, reward, meta, completion, int(batch['prompt_len'][0]), time.perf_counter()-started)
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
            if index % 10 == 0:
                print(f'[{namespace}] {index+1}/{count}', flush=True)
    stats = summarize(rows)
    output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    if writer:
        for name, value in stats.items():
            if isinstance(value, (float,int)):
                writer.add_scalar(f'{namespace}/{name}', value, step)
        writer.flush()
    return stats


def run_train(cfg, model, processor, prior, train_set, dev_set, test_set, output_dir):
    oc, gc = cfg['outcome'], cfg['grpo']
    collator = OutcomeCollator(processor, prior, cfg)
    device = next(model.parameters()).device
    if not len(train_set):
        raise ValueError('empty train set')
    requested = gc.get('max_attempts')
    attempts = int(requested) if requested is not None else math.ceil(len(train_set)*float(gc['epochs']))
    if attempts <= 0:
        raise ValueError('max_attempts must be positive')
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(gc['learning_rate']), weight_decay=0.)
    writer = SummaryWriter(str(Path(output_dir)/'tb'))
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
        with (output_dir/'rollouts.jsonl').open('w') as stream:
            for attempt in range(1, attempts+1):
                if not order:
                    order = list(range(len(train_set))); rng.shuffle(order)
                batch = move_batch(collator([train_set[order.pop()]]), device)
                started = time.perf_counter()
                completions = generate_group(model, processor, batch, cfg, group=int(gc['group_size']), sample=True)
                meta = batch['_meta'][0]
                parsed = [parse_output(c.text) for c in completions]
                scores = [score_output(p, meta, float(oc['protocol_weight'])) for p in parsed]
                rewards = torch.tensor([s['total'] for s in scores], device=device)
                advantages = group_advantages(rewards, bool(gc.get('scale_rewards', False)))
                zero = bool(advantages.abs().max().item() <= 1e-8)
                skipped += int(zero)
                metrics = dict(attempts=attempt, updates=updates, skipped_total=skipped, zero_advantage_group=float(zero),
                    reward_mean=float(rewards.mean()), reward_std=float(rewards.std(unbiased=False)),
                    task_reward_mean=sum(s['task'] for s in scores)/len(scores),
                    task_valid_rate=sum(p['task_valid'] for p in parsed)/len(parsed),
                    protocol_valid_rate=sum(p['protocol_valid'] for p in parsed)/len(parsed),
                    truncation_rate=sum(c.stop_reason == 'length' for c in completions)/len(completions))
                metrics.update(prompt_tokens=meta['prompt_tokens'], visual_tokens=meta['visual_tokens'],
                               prior_hint_tokens=meta['prior_hint_tokens'],
                               mean_new_tokens=sum(len(c.ids)-int(batch['prompt_len'][0]) for c in completions)/len(completions),
                               h_candidate_count=len(meta['prior_candidates']))
                # Log every attempted group BEFORE any skip or optimizer failure.
                rows = [make_record(p,s,meta,c,int(batch['prompt_len'][0]),0.) for p,s,c in zip(parsed,scores,completions)]
                stream.write(json.dumps(dict(attempt=attempt, update_before=updates, zero_advantage=zero,
                                             advantages=advantages.cpu().tolist(), trajectories=rows), ensure_ascii=False)+'\n')
                stream.flush()
                for name,value in metrics.items():
                    writer.add_scalar(f'train/{name}', value, attempt)
                writer.flush()
                if not zero:
                    loss_stats = optimize_group(model, processor, batch, completions, advantages, opt, cfg)
                    updates += 1
                    for name,value in loss_stats.items():
                        writer.add_scalar(f'optimizer/{name}', value, updates)
                writer.add_scalar('train/updates_after', updates, attempt)
                writer.add_scalar('train/seconds_per_attempt', time.perf_counter()-started, attempt)
                print(f'[outcome] attempt={attempt}/{attempts} updates={updates} skipped={skipped} '
                      f"reward={metrics['reward_mean']:.3f} task_valid={metrics['task_valid_rate']:.3f}", flush=True)
                every = int(cfg['training'].get('eval_every_n_steps', 0))
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
