"""Set-level localization metrics shared by eval and reward.

All functions operate on pixel-space boxes (already converted via ``to_pixels``)
so a single geometry definition is reused everywhere. ``iou`` is imported from
``outcome.protocol`` to avoid duplicating the intersection formula.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from outcome.protocol import giou, iou

# Component recall/precision thresholds (reported at 0.1 / 0.3 / 0.5).
THRESHOLDS = (0.1, 0.3, 0.5)

# Detection-match thresholds for MVTec/VisA bbox detection. mAP@50 (IoU>=0.5) is
# the primary metric and mAP@75 (IoU>=0.75) the strict counterpart, per
# "Adapting Vision-Language Models for Few-Shot Industrial Defect Detection"
# (MDPI Algorithms 2026) and standard COCO/YOLO detection conventions.
DET_IOU_THRESHOLD = 0.5
DET_IOU_THRESHOLDS = (0.5, 0.75)


def _greedy_match(pred_boxes, gt_boxes, iou_threshold):
    """Greedy one-to-one IoU matching; returns (tp, fp, fn, matched_pairs)."""
    N, M = len(pred_boxes), len(gt_boxes)
    pairs = []
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            pairs.append((iou(p, g), i, j))
    pairs.sort(key=lambda t: t[0], reverse=True)
    used_pred, used_gt = set(), set()
    matched = []
    for score, i, j in pairs:
        if score < iou_threshold:
            break
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matched.append((i, j))
    tp = len(matched)
    fp = N - len(used_pred)
    fn = M - len(used_gt)
    return tp, fp, fn, matched


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def detection_metrics(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    thresholds: Sequence[float] = DET_IOU_THRESHOLDS,
) -> dict:
    """Greedy IoU matching detection metrics at multiple IoU thresholds.

    VLM text-box outputs carry no confidence score, so instead of the
    confidence-sorted mAP PR curve this reports the single-point tp/fp/fn and
    precision/recall/f1 at each threshold (``at_50`` ~ mAP@50, ``at_75`` ~
    mAP@75). Empty-pred-vs-empty-GT is a perfect match (p=r=f1=1.0).

    Returns a flat dict keyed ``{tp,fp,fn,precision,recall,f1}_at_{int(t*100)}``
    plus ``n_pred`` and ``n_gt``.
    """
    pred_boxes = list(pred_boxes)
    gt_boxes = list(gt_boxes)
    N, M = len(pred_boxes), len(gt_boxes)
    out = dict(n_pred=N, n_gt=M)
    if N == 0 and M == 0:
        for t in thresholds:
            key = f'{int(round(t * 100)):02d}'
            out.update({f'tp_at_{key}': 0, f'fp_at_{key}': 0, f'fn_at_{key}': 0,
                        f'precision_at_{key}': 1.0, f'recall_at_{key}': 1.0,
                        f'f1_at_{key}': 1.0})
        return out
    if N == 0 or M == 0:
        for t in thresholds:
            key = f'{int(round(t * 100)):02d}'
            out.update({f'tp_at_{key}': 0, f'fp_at_{key}': N, f'fn_at_{key}': M,
                        f'precision_at_{key}': 0.0, f'recall_at_{key}': 0.0,
                        f'f1_at_{key}': 0.0})
        return out
    for t in thresholds:
        tp, fp, fn, _ = _greedy_match(pred_boxes, gt_boxes, float(t))
        precision, recall, f1 = _prf(tp, fp, fn)
        key = f'{int(round(t * 100)):02d}'
        out.update({f'tp_at_{key}': tp, f'fp_at_{key}': fp, f'fn_at_{key}': fn,
                    f'precision_at_{key}': precision, f'recall_at_{key}': recall,
                    f'f1_at_{key}': f1})
    return out


def greedy_detection_metrics(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    iou_threshold: float = DET_IOU_THRESHOLD,
) -> dict:
    """Single-threshold greedy IoU matching (kept for backward compatibility).

    Returns ``tp/fp/fn/precision/recall/f1`` at ``iou_threshold`` (default 0.5).
    Prefer :func:`detection_metrics` for multi-threshold reporting.
    """
    pred_boxes = list(pred_boxes)
    gt_boxes = list(gt_boxes)
    N, M = len(pred_boxes), len(gt_boxes)
    if N == 0 and M == 0:
        return dict(tp=0, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0,
                    n_pred=0, n_gt=0, matched_pairs=[])
    if N == 0 or M == 0:
        return dict(tp=0, fp=N, fn=M,
                    precision=0.0, recall=0.0, f1=0.0,
                    n_pred=N, n_gt=M, matched_pairs=[])
    tp, fp, fn, matched = _greedy_match(pred_boxes, gt_boxes, float(iou_threshold))
    precision, recall, f1 = _prf(tp, fp, fn)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1,
                n_pred=N, n_gt=M, matched_pairs=matched)


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


def _rasterize_boxes(boxes: Sequence[Sequence[float]], size: Sequence[float]) -> np.ndarray:
    """Rasterize axis-aligned boxes onto a boolean grid of shape (H, W).

    ``size`` is ``(W, H)`` pixel dimensions. Boxes are clipped to the grid and
    OR-ed together, so disconnected or overlapping fragments collapse into a
    single mask (AD-Copilot's BBox-Mask representation).
    """
    W = max(1, int(round(size[0])))
    H = max(1, int(round(size[1])))
    grid = np.zeros((H, W), dtype=bool)
    for b in boxes:
        x0 = int(np.clip(round(min(b[0], b[2])), 0, W))
        x1 = int(np.clip(round(max(b[0], b[2])), 0, W))
        y0 = int(np.clip(round(min(b[1], b[3])), 0, H))
        y1 = int(np.clip(round(max(b[1], b[3])), 0, H))
        if x1 > x0 and y1 > y0:
            grid[y0:y1, x0:x1] = True
    return grid


def mask_iou(pred_boxes: Sequence[Sequence[float]],
             gt_boxes: Sequence[Sequence[float]],
             size: Sequence[float]) -> float:
    """BBox-Mask IoU (AD-Copilot): mask-space IoU of the box unions.

    Rasterizes the predicted and GT boxes onto the same pixel grid and returns
    ``|P & G| / |P | G|``. Unlike Hungarian set-IoU this does not pair boxes nor
    penalize the box count directly, so it is robust to irregular/disconnected
    defects and to differing output granularity (one box vs many). Empty input on
    either side returns 0.0. Range [0, 1].
    """
    if not pred_boxes or not gt_boxes:
        return 0.0
    P = _rasterize_boxes(pred_boxes, size)
    G = _rasterize_boxes(gt_boxes, size)
    inter = int(np.logical_and(P, G).sum())
    union = int(np.logical_or(P, G).sum())
    return inter / union if union > 0 else 0.0


def component_metrics(
    pred_boxes: Sequence[Sequence[float]],
    gt_components: Sequence[Sequence[float]],
    thresholds: Sequence[float] = THRESHOLDS,
) -> dict:
    """Hungarian-matched component-level metrics.

    Returns a dict with ``matched_miou``, ``set_iou``, ``recall_at_*``,
    ``precision_at_*`` and ``count_error`` (|N_pred - M_gt|).
    ``matched_miou`` measures precision of matched pairs; ``set_iou`` divides
    the matched IoU sum by max(N,M) so it also penalizes missed/duplicate boxes.
    """
    N, M = len(pred_boxes), len(gt_components)
    base = dict(n_pred=N, n_gt=M, count_error=abs(N - M), matched_pairs=[], matched_ious=[])
    for t in thresholds:
        base[f'recall_at_{int(round(t * 10)):02d}'] = 0.0
        base[f'precision_at_{int(round(t * 10)):02d}'] = 0.0
    if N == 0 or M == 0:
        base['matched_miou'] = 0.0
        base['set_iou'] = 0.0
        return base
    C = np.zeros((N, M))
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_components):
            C[i, j] = iou(p, g)
    pairs = hungarian_matching(C)
    matched_ious = [float(C[i, j]) for i, j in pairs]
    base['matched_pairs'] = pairs
    base['matched_ious'] = matched_ious
    iou_sum = sum(matched_ious)
    base['matched_miou'] = iou_sum / len(matched_ious) if matched_ious else 0.0
    base['set_iou'] = iou_sum / max(N, M) if max(N, M) > 0 else 0.0
    for t in thresholds:
        base[f'recall_at_{int(round(t * 10)):02d}'] = sum(1 for v in matched_ious if v >= t) / M
        base[f'precision_at_{int(round(t * 10)):02d}'] = sum(1 for v in matched_ious if v >= t) / N
    return base


def set_giou(pred_boxes: Sequence[Sequence[float]], gt_components: Sequence[Sequence[float]]) -> float:
    """Hungarian-matched GIoU divided by max(N, M).

    Mirrors ``component_metrics(...)['set_iou']`` but uses GIoU so non-overlapping
    boxes still carry a directional (negative) gradient instead of a flat 0, which
    is the reward term used to suppress "spurious correctness" (AD-FM). Range is
    [-1, 1] because GIoU is bounded below by -1.
    """
    N, M = len(pred_boxes), len(gt_components)
    if N == 0 or M == 0:
        return 0.0
    C = np.zeros((N, M))
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_components):
            C[i, j] = giou(p, g)
    pairs = hungarian_matching(C)
    s_sum = float(sum(C[i, j] for i, j in pairs))
    return s_sum / max(N, M) if max(N, M) > 0 else 0.0
