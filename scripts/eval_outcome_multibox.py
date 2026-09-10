#!/usr/bin/env python3
"""Standalone evaluation for outcome-multibox-v1.

Evaluates a trained checkpoint (RL adapter, SFT adapter, or both) on the
MVTec/VisA dev or test split, writing per-sample ``.jsonl`` + summary ``.json``.

Examples:
  # RL adapter (adapter_final) trained on top of the SFT reference
  python scripts/eval_outcome_multibox.py \
      --config configs/qwen35_2b_outcome_multibox.yaml \
      --adapter outputs/train/.../adapter_final --split test --limit 200

  # SFT checkpoint only (no RL adapter)
  python scripts/eval_outcome_multibox.py \
      --config configs/qwen35_2b_outcome_multibox.yaml \
      --split test --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from outcome.engine_multibox import datasets, load_model, validate_config
from outcome.evaluate_multibox import evaluate
from utils.common import set_seed
from utils.config import load_yaml_config


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='test')
    parser.add_argument('--adapter', default=None,
                        help='RL adapter dir (adapter_final / checkpoint-N); '
                             'omit to evaluate the SFT reference in outcome.sft_adapter')
    parser.add_argument('--limit', type=int, default=None, help='sample count; omit = full split')
    parser.add_argument('--output', default=None,
                        help='summary .json path (default: ./outputs/eval_<split>.json)')
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    validate_config(cfg)
    set_seed(int(cfg['training']['seed']))

    model, processor, prior = load_model(cfg, args.adapter)
    _, dev_set, test_set = datasets(cfg, processor)
    selected = test_set if args.split == 'test' else dev_set

    output = Path(args.output) if args.output else Path('outputs') / f'eval_{args.split}.json'
    stats = evaluate(cfg, model, processor, prior, selected, output, args.limit,
                     namespace=args.split)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f'Summary: {output}', flush=True)


if __name__ == '__main__':
    main()
