#!/usr/bin/env python3
"""Short explicit detection: train, evaluate or predict through the SAME input path."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from outcome.engine import datasets, evaluate, load_model, run_train, validate_config
from outcome.inputs import OutcomeCollator
from outcome.policy import generate_group
from outcome.protocol import VERSION, parse_output, to_pixels
from rl.grpo import move_batch
from utils.common import set_seed
from utils.config import load_yaml_config


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='configs/qwen35_2b_outcome.yaml')
    parser.add_argument('--mode', choices=['train','eval','predict'], default='train')
    parser.add_argument('--split', choices=['dev','test'], default='dev')
    parser.add_argument('--adapter', help='Evaluation/prediction LoRA; not an optimizer resume')
    parser.add_argument('--num-gpu', type=int, default=1)
    parser.add_argument('--max-attempts', type=int)
    parser.add_argument('--eval-limit', type=int, help='Diagnostic subset; omitted means complete split')
    parser.add_argument('--output-dir')
    parser.add_argument('--image')
    parser.add_argument('--reference')
    parser.add_argument('--class-name', default='object')
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    cfg.setdefault('distributed', {})['num_gpu'] = args.num_gpu
    if args.adapter and args.mode == 'train':
        parser.error('--adapter is evaluation only; use outcome.sft_adapter for an SFT reference')
    if args.eval_limit is not None and args.eval_limit <= 0:
        parser.error('--eval-limit must be positive')
    if args.max_attempts is not None:
        cfg['grpo']['max_attempts'] = args.max_attempts
    if args.eval_limit is not None:
        cfg['training']['eval_num_samples'] = args.eval_limit
        cfg['training']['final_eval_num_samples'] = args.eval_limit
    validate_config(cfg)
    set_seed(int(cfg['training']['seed']))
    name = f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output = Path(args.output_dir) if args.output_dir else Path(cfg['paths']['output_dir'])/name
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    root = Path(__file__).resolve().parent
    sources = list((root/'outcome').glob('*.py')) + [root/'train_outcome.py']
    sources += [root/p for p in ['rl/grpo.py','models/qwen35.py','models/vision_cache.py','models/anomaly_prior.py','data/scan.py','data/prior_dataset.py']]
    (output/'provenance.json').write_text(json.dumps(dict(protocol_version=VERSION,
        source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        reference_policy='merged_sft' if cfg['outcome'].get('sft_adapter') else 'frozen_base',
        adapter=args.adapter, mode=args.mode, split=args.split, eval_limit=args.eval_limit), indent=2))
    model, processor, prior = load_model(cfg, args.adapter)
    if args.mode == 'predict':
        if not args.image or not args.reference:
            parser.error('predict requires --image and --reference')
        if Path(args.image).resolve() == Path(args.reference).resolve():
            parser.error('reference and inspection image must differ')
        test = Image.open(args.image).convert('RGB')
        item = dict(test=test, ref=Image.open(args.reference).convert('RGB'), image_path=args.image,
                    ref_path=args.reference, orig_size=test.size, class_name=args.class_name,
                    gt_box_px=None, is_anomaly=False, defect_type=None)
        batch = move_batch(OutcomeCollator(processor, prior, cfg)([item]), next(model.parameters()).device)
        completion = generate_group(model, processor, batch, cfg)[0]
        result = parse_output(completion.text)
        result.update(bbox_original_px=to_pixels(result['bbox_2d'], test.size),
                      text=completion.text, stop_reason=completion.stop_reason, input=batch['_meta'][0])
        # No GT is consulted by input construction or prediction; metadata false is a placeholder only.
        result['input'].pop('is_anomaly', None); result['input'].pop('gt_box_px', None)
        (output/'prediction.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        train_set, dev_set, test_set = datasets(cfg, processor)
        if args.mode == 'eval':
            selected = dev_set if args.split == 'dev' else test_set
            stats = evaluate(cfg, model, processor, prior, selected, output/f'{args.split}.json', args.eval_limit, namespace=args.split)
            print(json.dumps(stats, ensure_ascii=False, indent=2))
        else:
            run_train(cfg, model, processor, prior, train_set, dev_set, test_set, output)
    print(f'Output: {output}', flush=True)


if __name__ == '__main__':
    main()
