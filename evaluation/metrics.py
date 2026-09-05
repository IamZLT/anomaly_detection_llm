"""Classification + bbox metrics for VisA→MVTec evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from reasoning.rewards import box_iou, qwen1000_to_pixels_strict, valid_bbox_1000


def classification_correct(pred_cls, is_anomaly: bool) -> bool:
    return pred_cls is not None and bool(pred_cls) == bool(is_anomaly)


def gt_relative_area(gt_box, orig_wh) -> float:
    if gt_box is None or orig_wh is None:
        return 0.0
    w, h = float(max(orig_wh[0], 1)), float(max(orig_wh[1], 1))
    x1, y1, x2, y2 = map(float, gt_box)
    return max(0.0, (x2 - x1) * (y2 - y1)) / max(w * h, 1.0)


def defect_size_bin(a_gt: float, is_anomaly: bool) -> str:
    if not is_anomaly:
        return "normal"
    if a_gt < 0.02:
        return "small"
    if a_gt < 0.10:
        return "medium"
    return "large"


def box_ious_from_parsed(parsed: dict, meta: dict):
    orig = tuple(meta.get("orig_size") or (1, 1))
    gt = meta.get("gt_box_px")
    iou_f = 0.0
    iou_c = 0.0
    if gt is not None:
        pred = parsed.get("bbox_2d")
        cand = parsed.get("candidate_bbox_2d")
        if valid_bbox_1000(pred):
            iou_f = box_iou(qwen1000_to_pixels_strict(pred, orig), gt)
        if valid_bbox_1000(cand):
            iou_c = box_iou(qwen1000_to_pixels_strict(cand, orig), gt)
    rec_ok = classification_correct(parsed.get("is_anomaly", None), bool(meta.get("is_anomaly")))
    return float(iou_c), float(iou_f), rec_ok


def is_truncated(seq, prompt_len: int, max_new: int, eos_id: Optional[int]) -> bool:
    n_new = int(seq.numel()) - int(prompt_len)
    if n_new < int(max_new):
        return False
    if eos_id is None:
        return True
    return int(seq[-1].item()) != int(eos_id)


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def summarize_detection_metrics(rows: List[dict]) -> Dict[str, float]:
    """rows: is_anomaly, pred_is_anomaly, class_name, iou_f, iou_c, a_gt, rec_ok, trajectory_valid."""
    n = max(len(rows), 1)
    rec = sum(1 for r in rows if r.get("rec_ok"))
    anom = [r for r in rows if r.get("is_anomaly")]
    raw_ious = [float(r.get("iou_f") or 0.0) for r in anom]
    gated_ious = [
        float(r.get("iou_f") or 0.0) if r.get("pred_is_anomaly") is True else 0.0
        for r in anom
    ]
    strict_gated_ious = [
        float(r.get("iou_f") or 0.0)
        if (r.get("pred_is_anomaly") is True and r.get("trajectory_valid"))
        else 0.0
        for r in anom
    ]
    cand_ious = [float(r.get("iou_c") or 0.0) for r in anom]

    def acc_at(th: float, vals: Sequence[float]) -> float:
        if not vals:
            return 0.0
        return float(sum(1 for v in vals if v >= th) / len(vals))

    by_cls: Dict[str, List[float]] = defaultdict(list)
    by_bin: Dict[str, List[float]] = defaultdict(list)
    by_bin_gated: Dict[str, List[float]] = defaultdict(list)
    for r in anom:
        cls = str(r.get("class_name") or "_")
        b = str(r.get("size_bin") or defect_size_bin(float(r.get("a_gt") or 0.0), True))
        raw = float(r.get("iou_f") or 0.0)
        gated = raw if r.get("pred_is_anomaly") is True else 0.0
        by_cls[cls].append(gated)
        by_bin[b].append(raw)
        by_bin_gated[b].append(gated)

    out = {
        "n": float(len(rows)),
        "rec_acc": rec / n,
        "mean_iou": _mean(raw_ious),
        "mean_iou_gated": _mean(gated_ious),
        "mean_iou_strict_gated": _mean(strict_gated_ious),
        "mean_iou_c": _mean(cand_ious),
        "acc_at_01": acc_at(0.1, gated_ious),
        "acc_at_03": acc_at(0.3, gated_ious),
        "acc_at_05": acc_at(0.5, gated_ious),
        "strict_acc_at_01": acc_at(0.1, strict_gated_ious),
        "strict_acc_at_03": acc_at(0.3, strict_gated_ious),
        "strict_acc_at_05": acc_at(0.5, strict_gated_ious),
        "iou_at_03": acc_at(0.3, raw_ious),
        "trajectory_valid_rate": (
            sum(bool(r.get("trajectory_valid")) for r in rows)
            / n
        ),
        "macro_miou": _mean([_mean(v) for v in by_cls.values()]),
        "n_anom": float(len(anom)),
    }
    for b in ("small", "medium", "large"):
        out[f"mean_iou_{b}"] = _mean(by_bin.get(b, []))
        out[f"mean_iou_gated_{b}"] = _mean(by_bin_gated.get(b, []))
        out[f"n_{b}"] = float(len(by_bin.get(b, [])))
    return out
