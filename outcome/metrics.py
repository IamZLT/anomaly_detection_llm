"""Set-level localization metrics shared by eval and reward.

All functions operate on pixel-space boxes (already converted via ``to_pixels``)
so a single geometry definition is reused everywhere. ``iou`` is imported from
``outcome.protocol`` to avoid duplicating the intersection formula.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from outcome.protocol import iou

# Component recall/precision thresholds (reported at 0.1 / 0.3 / 0.5).
THRESHOLDS = (0.1, 0.3, 0.5)


def hungarian_matching(scores) -> List[Tuple[int, int]]:
    """One-to-one assignment maximizing total score.

    ``scores`` has shape (N_pred, M_gt). Returns matched (pred_idx, gt_idx) pairs
    (exactly ``min(N, M)`` pairs). Empty inputs return an empty list.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0 or scores.shape[0] == 0 or scores.shape[1] == 0:
        return []
    rows, cols = linear_sum_assignment(-scores)
    return [(int(i), int(j)) for i, j in zip(rows, cols)]


def union_box(boxes: Sequence[Sequence[float]]) -> Optional[List[float]]:
    """Tight bounding box spanning all boxes, or None when empty."""
    boxes = list(boxes)
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return [x0, y0, x1, y1]


def union_iou(pred_boxes: Sequence[Sequence[float]], gt_union: Optional[Sequence[float]]) -> float:
    """Benchmark-compatible union-box IoU: IoU(union(pred), union(gt))."""
    up = union_box(pred_boxes)
    if up is None or gt_union is None:
        return 0.0
    return iou(up, gt_union)


def component_metrics(
    pred_boxes: Sequence[Sequence[float]],
    gt_components: Sequence[Sequence[float]],
    thresholds: Sequence[float] = THRESHOLDS,
) -> dict:
    """Hungarian-matched component-level metrics.

    Returns a dict with ``matched_miou``, ``recall_at_*``, ``precision_at_*`` and
    ``count_error`` (|N_pred - M_gt|). Missing/duplicate boxes are penalized via
    count_error and the one-to-one matching, not a separate heuristic.
    """
    N, M = len(pred_boxes), len(gt_components)
    base = dict(n_pred=N, n_gt=M, count_error=abs(N - M), matched_pairs=[], matched_ious=[])
    for t in thresholds:
        base[f'recall_at_{int(round(t * 10)):02d}'] = 0.0
        base[f'precision_at_{int(round(t * 10)):02d}'] = 0.0
    if N == 0 or M == 0:
        base['matched_miou'] = 0.0
        return base
    C = np.zeros((N, M))
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_components):
            C[i, j] = iou(p, g)
    pairs = hungarian_matching(C)
    matched_ious = [float(C[i, j]) for i, j in pairs]
    base['matched_pairs'] = pairs
    base['matched_ious'] = matched_ious
    base['matched_miou'] = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    for t in thresholds:
        base[f'recall_at_{int(round(t * 10)):02d}'] = sum(1 for v in matched_ious if v >= t) / M
        base[f'precision_at_{int(round(t * 10)):02d}'] = sum(1 for v in matched_ious if v >= t) / N
    return base
