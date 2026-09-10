#!/usr/bin/env python3
"""Region-module SFT: train the region adapter + language LoRA with real category/bbox.

Freezes the vision encoder and H/matching computation. Supervises the machine-readable
blocks that GT actually supports (<ground> candidate box and <answer> category + bbox +
description), and masks the free-form process blocks (<understand>/<compare>/<verify>)
that cannot be reliably generated from a bbox alone.

After this run, the saved directory is pointed to by ``outcome.sft_adapter`` in the RL
config: ``load_model`` merges the SFT LoRA into the base weights, mounts the frozen
region adapter, and starts a fresh RL LoRA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data.prior_dataset import build_train_ref_pool
from data.scan import load_prior_split, split_holdout_by_class
from models.anomaly_prior import AnomalyPrior
from models.lora import apply_lora
from models.qwen35 import setup_model_and_processor, freeze_vision_encoder, force_vision_eval, unwrap_model
from models.region_adapter import build_region_adapter
from models.region_injection import bind_region_injection, ensure_region_token, region_raw_from_batch, save_region_adapter
from models.vision_cache import bind_cached_image_features
from outcome.inputs import OutcomeCollator, OutcomeDataset
from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
from rl.grpo import forward_with_vision, model_inputs, move_batch
from utils.common import set_seed
from utils.config import load_yaml_config


def _bbox_to_1000(gt_px, orig_size):
    w, h = float(orig_size[0]), float(orig_size[1])
    return [round(gt_px[0] * 1000.0 / w, 3), round(gt_px[1] * 1000.0 / h, 3),
            round(gt_px[2] * 1000.0 / w, 3), round(gt_px[3] * 1000.0 / h, 3)]


def _boxes_to_1000(boxes_px, orig_size):
    return [_bbox_to_1000(b, orig_size) for b in boxes_px]


def build_sft_target(meta: dict, multibox: bool = False) -> str:
    """Five-block target; category/boxes come from GT, process blocks are neutral filler.

    ``multibox=True`` emits the outcome-multibox-v1 answer schema (``bboxes_2d`` as a
    list of one box per disconnected GT component, empty list for normal) instead of
    the single union box (``bbox_2d``). Use it for the SFT that seeds multi-box RL so
    the language LoRA is already aligned with the multi-box output format.
    """
    is_anom = bool(meta['is_anomaly'])
    defect = str(meta.get('defect_type') or 'defect')
    if multibox:
        if is_anom:
            comps = list(meta.get('component_bboxes') or [])
            if not comps and meta.get('gt_box_px') is not None:
                comps = [meta['gt_box_px']]
            boxes = _boxes_to_1000(comps, meta['orig_size'])
            answer = json.dumps({'is_anomaly': True, 'bboxes_2d': boxes, 'description': f'{defect} defect'})
            ground = f'candidate_bboxes_2d={boxes}'
        else:
            answer = json.dumps({'is_anomaly': False, 'bboxes_2d': [], 'description': 'no anomaly'})
            ground = 'candidate_bboxes_2d=[]'
        return (
            '<understand>\ninspect the object and the provided region evidence\n</understand>\n'
            '<compare>\ncompare the inspection image against the defect-free reference\n</compare>\n'
            f'<ground>\n{ground}\n</ground>\n'
            '<verify>\nkeep; the evidence matches the inspection\n</verify>\n'
            f'<answer>\n{answer}\n</answer>'
        )
    if is_anom:
        box = _bbox_to_1000(meta['gt_box_px'], meta['orig_size'])
        answer = json.dumps({'is_anomaly': True, 'bbox_2d': box, 'description': f'{defect} defect'})
        ground = f'candidate_bbox_2d={box}'
    else:
        answer = json.dumps({'is_anomaly': False, 'bbox_2d': None, 'description': 'no anomaly'})
        ground = 'candidate_bbox_2d=null'
    return (
        '<understand>\ninspect the object and the provided region evidence\n</understand>\n'
        '<compare>\ncompare the inspection image against the defect-free reference\n</compare>\n'
        f'<ground>\n{ground}\n</ground>\n'
        '<verify>\nkeep; the evidence matches the inspection\n</verify>\n'
        f'<answer>\n{answer}\n</answer>'
    )


def _supervised_spans(target: str):
    spans = []
    for open_tag, close_tag in (('<ground>', '</ground>'), ('<answer>', '</answer>')):
        s = target.find(open_tag)
        e = target.find(close_tag)
        if s >= 0 and e >= 0:
            spans.append((s, e + len(close_tag)))
    return spans


def build_labels(prompt_ids, target: str, tokenizer):
    """Teacher-forced labels that supervise only <ground> and <answer>.

    The process blocks are still seen by the model (so it learns the format) but are
    masked (``-100``) because a bbox cannot justify their prose.
    """
    spans = _supervised_spans(target)
    try:
        enc = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc['offset_mapping']
    except Exception:
        offsets = None
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    labels = [-100] * (len(prompt_ids) + len(target_ids))
    for i, tid in enumerate(target_ids):
        if offsets is None:
            labels[len(prompt_ids) + i] = tid
            continue
        a, b = offsets[i]
        if any(a >= s and b <= e for s, e in spans):
            labels[len(prompt_ids) + i] = tid
    return labels


def load_sft_model(cfg):
    model, processor = setup_model_and_processor(cfg, for_inference=False, freeze_vision=True)
    ensure_region_token(processor, model)
    model = apply_lora(model, cfg)
    freeze_vision_encoder(model)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg['training'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    force_vision_eval(model)
    prior = AnomalyPrior.from_qwen(model, cfg)
    feature_dim = int(prior.visual.config.hidden_size)
    hidden_size = int(model.config.text_config.hidden_size)
    model.region_adapter = build_region_adapter(cfg, feature_dim, hidden_size)
    # The adapter is created after model.to(cuda): sync it explicitly to the model's
    # device and dtype or the SFT forward will hit a CPU/FP32 vs GPU/BF16 mismatch.
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.region_adapter.to(device=device, dtype=dtype)
    return model, processor, prior


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--accum', type=int, default=1)
    parser.add_argument('--save-steps', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(int(cfg['training']['seed']))

    multibox = (cfg.get('outcome', {}).get('version') == 'outcome-multibox-v1')
    dataset_cls = OutcomeMultiboxDataset if multibox else OutcomeDataset
    collator_cls = OutcomeMultiboxCollator if multibox else OutcomeCollator

    model, processor, prior = load_sft_model(cfg)
    tokenizer = getattr(processor, 'tokenizer', processor)
    device = next(model.parameters()).device

    train, test = load_prior_split(cfg)
    train, _dev = split_holdout_by_class(train, float(cfg['data']['holdout_ratio']),
                                         seed=int(cfg['training']['seed']))
    pool = build_train_ref_pool(train)
    dataset = dataset_cls(train, cfg, processor, 'train', pool)
    if args.max_samples is not None:
        dataset.samples = dataset.samples[:max(1, int(args.max_samples))]

    collator = collator_cls(processor, prior, cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=0.0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    total_steps = max(1, int(args.epochs))
    seen = 0
    step = 0
    running_loss = 0.0
    running_supervised = 0
    window_samples = 0
    for epoch in range(1, total_steps + 1):
        model.train()
        force_vision_eval(model)
        for sample in dataset:
            batch = move_batch(collator([sample]), device)
            meta = batch['_meta'][0]
            prompt_ids = batch['input_ids'][0].tolist()
            target = build_sft_target(meta, multibox=multibox)
            labels = build_labels(prompt_ids, target, tokenizer)
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            input_ids = torch.tensor([prompt_ids + target_ids], device=device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            labels_t = torch.tensor([labels], device=device, dtype=torch.long)
            gen_in = model_inputs(batch)
            cache = batch['image_embeds']
            adapter = getattr(unwrap_model(model), 'region_adapter', None)
            region_raw = region_raw_from_batch(batch)
            region_token_id = int(batch['region_token_id'])
            with bind_cached_image_features(model, cache), \
                    bind_region_injection(model, adapter, region_raw, region_token_id):
                out = forward_with_vision(model, gen_in, input_ids, attention_mask)
                logits = out.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels_t[:, 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                       shift_labels.view(-1), ignore_index=-100)
                (loss / max(1, int(args.accum))).backward()
            n_sup = int((labels_t != -100).sum())
            running_loss += float(loss.detach())
            running_supervised += n_sup
            window_samples += 1
            seen += 1
            if seen % max(1, int(args.accum)) == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(cfg.get('grpo', {}).get('max_grad_norm', 1.0)))
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f'[sft] step={step} loss={running_loss / max(1, window_samples):.4f} '
                          f'supervised_tok={running_supervised} seen={seen}', flush=True)
                    running_loss = 0.0
                    running_supervised = 0
                    window_samples = 0
                if args.save_steps > 0 and step % int(args.save_steps) == 0:
                    ckpt = output_dir / f'checkpoint-{step}'
                    model.save_pretrained(ckpt)
                    processor.save_pretrained(ckpt)
                    save_region_adapter(getattr(unwrap_model(model), 'region_adapter', None),
                                        ckpt / 'region_adapter.pt')
    # Tail flush for a final partial accumulation batch.
    if seen % max(1, int(args.accum)) != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float(cfg.get('grpo', {}).get('max_grad_norm', 1.0)))
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    save_region_adapter(getattr(unwrap_model(model), 'region_adapter', None),
                        output_dir / 'region_adapter.pt')
    print(f'[sft] saved SFT LoRA + region adapter to {output_dir} (steps={step})', flush=True)


if __name__ == '__main__':
    main()
