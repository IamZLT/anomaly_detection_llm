#!/usr/bin/env python3
"""Offline diagnostics for the region-token run (four dev-set questions).

Reads JSONL produced by ``outcome/engine.py``:

- ``dev_*.jsonl`` / ``test_final.jsonl``: one flat ``make_record`` row per line.
- ``rollouts.jsonl``: one dict per group with a ``trajectories`` list + advantage stats.

Reports:
  1. Does H cover the defect?      (h_union_cov, iou_h_bestk, prior recall)
  2. Does the model fix candidates? (iou_c -> iou_f, delta_refine)
  3. Does it rely on real visual contrast? (cross-run hint: prior.condition real/none/shuffled)
  4. Does RL still have signal?     (zero-advantage groups, anomaly all-zero-IoU groups)
"""

from __future__ import annotations

import argparse
import json
import sys


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(v):
    return '--' if v is None else f'{v:.4f}'


def _load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if 'trajectories' in obj:
                rows.extend(obj['trajectories'])
            else:
                rows.append(obj)
    return rows


def report(rows):
    abnormal = [r for r in rows if r.get('is_anomaly')]
    normal = [r for r in rows if not r.get('is_anomaly')]

    print('== 1. Does H cover the defect? ==')
    print(f"  mean h_union_cov        : {_fmt(_mean([r.get('h_union_cov') for r in abnormal]))}")
    print(f"  mean iou_h_bestk        : {_fmt(_mean([r.get('iou_h_bestk') for r in abnormal]))}")
    print(f"  prior_recall@0.1 (bestk): {_fmt(_mean([(r.get('iou_h_bestk') or 0.0) >= 0.1 for r in abnormal]))}")
    print(f"  prior_recall@0.3 (bestk): {_fmt(_mean([(r.get('iou_h_bestk') or 0.0) >= 0.3 for r in abnormal]))}")

    print('== 2. Does the model fix candidates? ==')
    print(f"  mean iou_c (candidate)  : {_fmt(_mean([r.get('iou_c') for r in abnormal]))}")
    print(f"  mean iou_f (final)      : {_fmt(_mean([r.get('iou_f') for r in abnormal]))}")
    print(f"  mean delta_refine       : {_fmt(_mean([r.get('delta_refine') for r in abnormal]))}")

    print('== 3. Real visual contrast reliance (cross-run) ==')
    print('  Re-run eval with outcome.prior.condition = real | none | shuffled and compare iou_f /')
    print('  normal_fpr. A large drop under none/shuffled => the model relies on the real H channel.')

    print('== 4. Does RL still have a signal? ==')
    print(f"  anomaly_recall          : {_fmt(_mean([r.get('pred') is True for r in abnormal]))}")
    print(f"  normal_fpr              : {_fmt(_mean([r.get('pred') is True for r in normal]))}")
    print(f"  anomaly all-zero IoU    : {_fmt(_mean([(r.get('iou') or 0.0) <= 1e-9 for r in abnormal]))}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    rows = []
    for path in args.files:
        rows.extend(_load_rows(path))
    if not rows:
        print('no rows found', file=sys.stderr)
        return 1
    report(rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
