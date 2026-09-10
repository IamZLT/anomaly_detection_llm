"""outcome-multibox-v1 evaluation: component-level metrics, set-level summary, viz.

Kept separate from the training loop so a standalone eval script and the
mid-training "small eval" in ``engine_multibox.run_train`` share one
implementation. ``evaluate`` writes per-sample ``.jsonl`` + a summary ``.json`` and
optionally logs a TensorBoard grid.
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from outcome.inputs_multibox import OutcomeMultiboxCollator
from outcome.metrics import component_metrics, detection_metrics, union_iou
from outcome.policy import generate_group
from outcome.protocol import iou, to_pixels
from outcome.protocol_multibox import parse_output, score_output
from outcome.visualize_multibox import log_outcome_eval_grid
from rl.grpo import move_batch


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


def make_record(parsed, score, meta, completion, prompt_len, elapsed, max_boxes):
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
        prior_any_top1_iou = iou_h_top1
        prior_any_best_iou = iou_h_bestk
        h_boxes_px = [to_pixels(c['bbox_2d'], orig) for c in candidates]
        hcm = component_metrics(h_boxes_px, comps)
    else:
        iou_h_top1 = iou_h_bestk = None
        prior_any_top1_iou = prior_any_best_iou = None
        hcm = None

    cm = component_metrics(pred_px, comps) if anomaly else None
    dm = detection_metrics(pred_px, comps) if anomaly else None
    rec = dict(image_path=meta['image_path'], ref_path=meta['ref_path'], class_name=meta['class_name'],
        is_anomaly=anomaly, pred=parsed['is_anomaly'], task_valid=parsed['task_valid'],
        protocol_core=parsed['protocol_core'], protocol_strict=parsed['protocol_strict'],
        candidate_state=parsed['candidate_state'], verify_action=parsed['verify_action'],
        reward=score['total'], loc_reward=score['loc_reward'], set_c_reward=score['set_c_reward'],
        set_f_reward=score['set_f_reward'], delta_refine=score['delta_refine'],
        set_giou=score['set_giou'], count_reward=score['count_reward'], focus_reward=score['focus_reward'],
        task_reward=score['task'],
        mask_iou=score['mask_iou'], union_iou=score['union_iou'], gt_box_px=gt, bboxes_2d=parsed['bboxes_2d'],
        candidate_bboxes_2d=parsed['candidate_bboxes_2d'], num_boxes=parsed['num_boxes'],
        num_components=int(meta.get('num_components') or (len(comps) if anomaly else 0)),
        iou_h_top1=iou_h_top1, iou_h_bestk=iou_h_bestk,
        prior_any_top1_iou=prior_any_top1_iou, prior_any_best_iou=prior_any_best_iou,
        prior_component_recall_at_01=(hcm['recall_at_01'] if hcm is not None else None),
        prior_component_recall_at_03=(hcm['recall_at_03'] if hcm is not None else None),
        prior_component_recall_at_05=(hcm['recall_at_05'] if hcm is not None else None),
        gt_over_max_boxes=bool(anomaly and len(comps) > max_boxes),
        reward_ceiling_from_cap=(min(max_boxes, len(comps)) / len(comps) if anomaly and comps else 1.0),
        size_bin=_size_bin(meta, anomaly), component_bin=_component_bins(meta, anomaly),
        prior_candidates=candidates, prior_condition=meta.get('prior_condition'),
        image_count=meta.get('image_count'), prompt_tokens=meta.get('prompt_tokens'),
        visual_tokens=meta.get('visual_tokens'), prior_hint_tokens=meta.get('prior_hint_tokens'),
        stop_reason=completion.stop_reason, new_tokens=len(completion.ids)-prompt_len,
        seconds=elapsed, text=completion.text)
    if cm is not None:
        rec.update(matched_miou=cm['matched_miou'], set_iou=cm['set_iou'], count_error=cm['count_error'],
                   recall_at_01=cm['recall_at_01'], recall_at_03=cm['recall_at_03'], recall_at_05=cm['recall_at_05'],
                   precision_at_01=cm['precision_at_01'], precision_at_03=cm['precision_at_03'],
                   precision_at_05=cm['precision_at_05'])
    if dm is not None:
        rec.update({f'det_{k}': v for k, v in dm.items()})
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
        mask_miou=mean(r['mask_iou'] for r in abnormal),
        mask_acc_at_05=mean(r['mask_iou'] >= .5 for r in abnormal),
        mask_acc_at_03=mean(r['mask_iou'] >= .3 for r in abnormal),
        union_miou=mean(r['union_iou'] for r in abnormal),
        union_acc_at_05=mean(r['union_iou'] >= .5 for r in abnormal),
        truncation_rate=mean(r['stop_reason'] == 'length' for r in rows),
        mean_new_tokens=mean(r['new_tokens'] for r in rows), mean_seconds=mean(r['seconds'] for r in rows),
        matched_miou=mean(r['matched_miou'] for r in abnormal if r.get('matched_miou') is not None),
        set_iou=mean(r['set_iou'] for r in abnormal if r.get('set_iou') is not None),
        recall_at_01=mean(r['recall_at_01'] for r in abnormal if r.get('recall_at_01') is not None),
        recall_at_03=mean(r['recall_at_03'] for r in abnormal if r.get('recall_at_03') is not None),
        recall_at_05=mean(r['recall_at_05'] for r in abnormal if r.get('recall_at_05') is not None),
        precision_at_01=mean(r['precision_at_01'] for r in abnormal if r.get('precision_at_01') is not None),
        precision_at_03=mean(r['precision_at_03'] for r in abnormal if r.get('precision_at_03') is not None),
        precision_at_05=mean(r['precision_at_05'] for r in abnormal if r.get('precision_at_05') is not None),
        count_error=mean(r['count_error'] for r in abnormal if r.get('count_error') is not None),
        mean_set_f_reward=mean(r['set_f_reward'] for r in abnormal),
        mean_set_c_reward=mean(r['set_c_reward'] for r in abnormal),
        mean_set_giou=mean(r['set_giou'] for r in abnormal if r.get('set_giou') is not None),
        mean_count_reward=mean(r['count_reward'] for r in abnormal if r.get('count_reward') is not None),
        mean_focus_reward=mean(r['focus_reward'] for r in normal if r.get('focus_reward') is not None),
        mean_delta_refine=mean(r['delta_refine'] for r in rows if r['delta_refine'] is not None),
        # Multi-threshold greedy-IoU detection metrics (mAP@50 / mAP@75 conventions).
        det_precision_at_50=mean(r['det_precision_at_50'] for r in abnormal if r.get('det_precision_at_50') is not None),
        det_recall_at_50=mean(r['det_recall_at_50'] for r in abnormal if r.get('det_recall_at_50') is not None),
        det_f1_at_50=mean(r['det_f1_at_50'] for r in abnormal if r.get('det_f1_at_50') is not None),
        det_precision_at_75=mean(r['det_precision_at_75'] for r in abnormal if r.get('det_precision_at_75') is not None),
        det_recall_at_75=mean(r['det_recall_at_75'] for r in abnormal if r.get('det_recall_at_75') is not None),
        det_f1_at_75=mean(r['det_f1_at_75'] for r in abnormal if r.get('det_f1_at_75') is not None),
        mean_iou_h_top1=mean(r['iou_h_top1'] for r in abnormal if r['iou_h_top1'] is not None),
        mean_iou_h_bestk=mean(r['iou_h_bestk'] for r in abnormal if r['iou_h_bestk'] is not None),
        prior_any_best_iou=mean(r['prior_any_best_iou'] for r in abnormal if r['prior_any_best_iou'] is not None),
        prior_recall_at_01=mean((r['iou_h_bestk'] or 0.0) >= .1 for r in abnormal),
        prior_recall_at_03=mean((r['iou_h_bestk'] or 0.0) >= .3 for r in abnormal),
        prior_component_recall_at_01=mean(r['prior_component_recall_at_01'] for r in abnormal if r.get('prior_component_recall_at_01') is not None),
        prior_component_recall_at_03=mean(r['prior_component_recall_at_03'] for r in abnormal if r.get('prior_component_recall_at_03') is not None),
        prior_component_recall_at_05=mean(r['prior_component_recall_at_05'] for r in abnormal if r.get('prior_component_recall_at_05') is not None),
        gt_over_max_boxes_rate=mean(r['gt_over_max_boxes'] for r in abnormal),
        mean_cap_reward_ceiling=mean(r['reward_ceiling_from_cap'] for r in abnormal),
        mean_num_boxes=mean(r['num_boxes'] for r in abnormal),
        mean_num_components=mean(r['num_components'] for r in abnormal))
    for size in ('small','medium','large'):
        subset = [r for r in abnormal if r['size_bin'] == size]
        out[f'n_{size}'] = len(subset)
        out[f'mask_miou_{size}'] = mean(r['mask_iou'] for r in subset)
        out[f'matched_miou_{size}'] = mean(r['matched_miou'] for r in subset if r.get('matched_miou') is not None)
        out[f'set_iou_{size}'] = mean(r['set_iou'] for r in subset if r.get('set_iou') is not None)
        out[f'union_miou_{size}'] = mean(r['union_iou'] for r in subset)
        out[f'det_f1_at_50_{size}'] = mean(r['det_f1_at_50'] for r in subset if r.get('det_f1_at_50') is not None)
        out[f'det_f1_at_75_{size}'] = mean(r['det_f1_at_75'] for r in subset if r.get('det_f1_at_75') is not None)
    for cb in ('single','multi'):
        subset = [r for r in abnormal if r['component_bin'] == cb]
        out[f'n_{cb}'] = len(subset)
        out[f'mask_miou_{cb}'] = mean(r['mask_iou'] for r in subset)
        out[f'matched_miou_{cb}'] = mean(r['matched_miou'] for r in subset if r.get('matched_miou') is not None)
        out[f'set_iou_{cb}'] = mean(r['set_iou'] for r in subset if r.get('set_iou') is not None)
        out[f'union_miou_{cb}'] = mean(r['union_iou'] for r in subset)
        out[f'recall_at_05_{cb}'] = mean(r['recall_at_05'] for r in subset if r.get('recall_at_05') is not None)
        out[f'det_f1_at_50_{cb}'] = mean(r['det_f1_at_50'] for r in subset if r.get('det_f1_at_50') is not None)
        out[f'det_f1_at_75_{cb}'] = mean(r['det_f1_at_75'] for r in subset if r.get('det_f1_at_75') is not None)
    for key in ('image_count','prompt_tokens','visual_tokens','prior_hint_tokens'):
        out[f'mean_{key}'] = mean(r[key] for r in rows if r.get(key) is not None)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r['class_name']].append(r)
    out['per_class'] = {c: {'n':len(rs), 'n_anomaly':sum(r['is_anomaly'] for r in rs),
        'mask_miou':mean(r['mask_iou'] for r in rs if r['is_anomaly']),
        'matched_miou':mean(r['matched_miou'] for r in rs if r['is_anomaly'] and r.get('matched_miou') is not None),
        'union_miou':mean(r['union_iou'] for r in rs if r['is_anomaly']),
        'det_f1_at_50':mean(r['det_f1_at_50'] for r in rs if r['is_anomaly'] and r.get('det_f1_at_50') is not None),
        'det_f1_at_75':mean(r['det_f1_at_75'] for r in rs if r['is_anomaly'] and r.get('det_f1_at_75') is not None),
        'normal_fpr':mean(r['pred'] is True for r in rs if not r['is_anomaly'])} for c,rs in by_class.items()}
    out['macro_mask_miou'] = mean(v['mask_miou'] for v in out['per_class'].values() if v['mask_miou'] is not None)
    out['macro_matched_miou'] = mean(v['matched_miou'] for v in out['per_class'].values() if v['matched_miou'] is not None)
    out['macro_union_miou'] = mean(v['union_miou'] for v in out['per_class'].values() if v['union_miou'] is not None)
    out['macro_det_f1_at_50'] = mean(v['det_f1_at_50'] for v in out['per_class'].values() if v['det_f1_at_50'] is not None)
    out['macro_det_f1_at_75'] = mean(v['det_f1_at_75'] for v in out['per_class'].values() if v['det_f1_at_75'] is not None)
    return out


def _eval_running_metrics(rows):
    """Compact running summary of the key eval metrics (cheap, computed incrementally)."""
    abnormal = [r for r in rows if r['is_anomaly']]
    normal = [r for r in rows if not r['is_anomaly']]

    def mean(vals):
        vals = list(vals)
        return sum(vals) / len(vals) if vals else None

    return {
        'rec': mean(r['pred'] is True for r in abnormal),
        'tnr': mean(r['pred'] is False for r in normal),
        'mask_miou': mean(r['mask_iou'] for r in abnormal),
        'set_iou': mean(r['set_iou'] for r in abnormal if r.get('set_iou') is not None),
        'task_valid': mean(r['task_valid'] for r in rows),
        'protocol_core': mean(r['protocol_core'] for r in rows),
    }


def stratified_eval_indices(dataset, count: int, seed: int = 42):
    """Class-balanced indices so a limited eval spans every class (not just the
    alphabetically-first class, which is what a plain ``[:count]`` slice yields
    for the class-sorted MVTec scan). Within each class, normal/anomaly are
    balanced as evenly as the quota allows.
    """
    samples = dataset.samples
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
    indices = stratified_eval_indices(dataset, count, seed=int(cfg['training']['seed']))
    t_start = time.perf_counter()
    with output_path.with_suffix('.jsonl').open('w') as stream:
        for pos, index in enumerate(indices):
            started = time.perf_counter()
            batch = move_batch(collator([dataset[index]]), device)
            completion = generate_group(model, processor, batch, cfg)[0]
            parsed = parse_output(completion.text, max_boxes=max_boxes)
            meta = batch['_meta'][0]
            reward = score_output(parsed, meta, float(cfg['outcome']['protocol_weight']),
                                  {**cfg['outcome'].get('localization', {}),
                                   **cfg['outcome'].get('reward', {})}, max_boxes=max_boxes)
            row = make_record(parsed, reward, meta, completion, int(batch['prompt_len'][0]), time.perf_counter()-started, max_boxes)
            rows.append(row)
            cases.append(dict(meta=meta, parsed=parsed, response=completion.text,
                              union_iou=reward['union_iou'], loc_reward=reward['loc_reward'],
                              correct=reward['correct']))
            stream.write(json.dumps(row, ensure_ascii=False, default=str)+'\n'); stream.flush()
            done = pos + 1
            if done % 10 == 0 or done == len(indices):
                m = _eval_running_metrics(rows)

                def _f(v):
                    return '--' if v is None else f'{v:.3f}'

                el = time.perf_counter() - t_start
                print(f'[{namespace}] {done}/{len(indices)} '
                      f'rec={_f(m["rec"])} tnr={_f(m["tnr"])} mIoU={_f(m["mask_miou"])} '
                      f'sIoU={_f(m["set_iou"])} tv={_f(m["task_valid"])} pc={_f(m["protocol_core"])} '
                      f'| {el/done:.1f}s/sample', flush=True)
    stats = summarize(rows)
    output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    if writer:
        for name, value in stats.items():
            if isinstance(value, (float,int)):
                writer.add_scalar(f'{namespace}/{name}', value, step)
        log_outcome_eval_grid(writer, step=step, cases=cases)
        writer.flush()
    return stats
