"""Verifiable spatial rewards: coverage, direction, classification, IoU."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from reasoning.parser import EDGE_KEYS, TAG_NAMES
from utils.common import qwen_norm1000_to_original_pixels


def box_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = area_a + area_b - inter
    return float(inter / den) if den > 0 else 0.0


def box_coverage(pred: List[float], gt: List[float]) -> float:
    ax1, ay1, ax2, ay2 = pred
    bx1, by1, bx2, by2 = gt
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_gt = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / area_gt) if area_gt > 0 else 0.0


def box_area(box: List[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def pixels_to_qwen1000(box: List[float], orig_wh: Tuple[int, int]) -> List[int]:
    w, h = int(orig_wh[0]), int(orig_wh[1])
    x1, y1, x2, y2 = box
    return [
        int(round(max(0.0, min(1000.0, x1 / max(w, 1) * 1000.0)))),
        int(round(max(0.0, min(1000.0, y1 / max(h, 1) * 1000.0)))),
        int(round(max(0.0, min(1000.0, x2 / max(w, 1) * 1000.0)))),
        int(round(max(0.0, min(1000.0, y2 / max(h, 1) * 1000.0)))),
    ]


def boundary_targets(candidate: List[float], gt: List[float], keep_tol: float) -> Dict[str, str]:
    cx1, cy1, cx2, cy2 = candidate
    gx1, gy1, gx2, gy2 = gt
    l = "keep" if abs(cx1 - gx1) <= keep_tol else ("inward" if cx1 < gx1 else "outward")
    r = "keep" if abs(cx2 - gx2) <= keep_tol else ("inward" if cx2 > gx2 else "outward")
    t = "keep" if abs(cy1 - gy1) <= keep_tol else ("inward" if cy1 < gy1 else "outward")
    b = "keep" if abs(cy2 - gy2) <= keep_tol else ("inward" if cy2 > gy2 else "outward")
    return {"L": l, "R": r, "T": t, "B": b}


def edge_precision_reward(pred: List[float], gt: List[float], orig_wh: Tuple[int, int], beta: float) -> float:
    w, h = float(max(orig_wh[0], 1)), float(max(orig_wh[1], 1))
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    dists = [
        abs(px1 - gx1) / w,
        abs(px2 - gx2) / w,
        abs(py1 - gy1) / h,
        abs(py2 - gy2) / h,
    ]
    return float(sum(math.exp(-beta * d) for d in dists) / 4.0)


def center_reward(pred: List[float], gt: List[float], orig_wh: Tuple[int, int], gamma: float) -> float:
    w, h = float(max(orig_wh[0], 1)), float(max(orig_wh[1], 1))
    pc = ((pred[0] + pred[2]) * 0.5, (pred[1] + pred[3]) * 0.5)
    gc = ((gt[0] + gt[2]) * 0.5, (gt[1] + gt[3]) * 0.5)
    dist = math.hypot(pc[0] - gc[0], pc[1] - gc[1])
    diag = math.sqrt(w * w + h * h)
    return float(math.exp(-float(gamma) * dist / max(diag, 1e-6)))


def _format_ok(parsed: Dict[str, Any], is_anomaly: bool) -> bool:
    tags = parsed.get("tags") or {}
    has_tags = all(n in tags for n in TAG_NAMES)
    pred_cls = parsed.get("is_anomaly", None)
    if not has_tags or pred_cls is None:
        return False
    if not is_anomaly:
        return (pred_cls is False) and parsed.get("bbox_2d") is None
    bound = parsed.get("boundary") or {}
    return (
        pred_cls is True
        and parsed.get("candidate_bbox") is not None
        and parsed.get("bbox_2d") is not None
        and all(k in bound for k in EDGE_KEYS)
    )


def compute_rewards(
    parsed: Dict[str, Any],
    gt_box_px: Optional[List[float]],
    orig_wh: Tuple[int, int],
    is_anomaly: bool,
    cfg: dict,
) -> Dict[str, float]:
    rew_cfg = cfg.get("grpo", {}).get("reward", {}) or {}
    w_cov = float(rew_cfg.get("w_cov", 0.7))
    w_compact = float(rew_cfg.get("w_compact", 0.3))
    w_iou = float(rew_cfg.get("w_iou", 0.45))
    w_edge = float(rew_cfg.get("w_edge", 0.40))
    w_center = float(rew_cfg.get("w_center", 0.15))
    beta = float(rew_cfg.get("edge_beta", 8.0))
    gamma = float(rew_cfg.get("center_gamma", 8.0))
    keep_tol = float(rew_cfg.get("keep_tol_norm1000", 8.0))
    fmt_w = float(rew_cfg.get("format_weight", 0.03))
    normal_fp_pen = float(rew_cfg.get("normal_false_positive", -0.5))
    cls_correct = float(rew_cfg.get("cls_correct", 1.0))
    cls_wrong = float(rew_cfg.get("cls_wrong", -1.0))
    cls_invalid = float(rew_cfg.get("cls_invalid", -0.5))

    pred_f = parsed.get("bbox_2d")
    cand = parsed.get("candidate_bbox")
    pred_cls = parsed.get("is_anomaly", None)
    if pred_cls is None:
        r_cls = cls_invalid
    elif bool(pred_cls) == bool(is_anomaly):
        r_cls = cls_correct
    else:
        r_cls = cls_wrong

    pred_px = None
    cand_px = None
    if pred_f is not None:
        pred_px = [float(x) for x in qwen_norm1000_to_original_pixels(pred_f, orig_wh)]
    if cand is not None:
        cand_px = [float(x) for x in qwen_norm1000_to_original_pixels(cand, orig_wh)]

    r_fmt = 1.0 if _format_ok(parsed, is_anomaly) else 0.0
    zeros = {
        "R_cov": 0.0,
        "R_compact": 0.0,
        "R_dir": 0.0,
        "R_iou": 0.0,
        "R_iou_c": 0.0,
        "R_edge": 0.0,
        "R_center": 0.0,
        "R_format": float(r_fmt),
        "R_cls": float(r_cls),
        "pred_box_px": pred_px,
        "cand_box_px": cand_px,
        "pred_cls": pred_cls,
        "d_star": {},
    }

    if not is_anomaly:
        r_ground = fmt_w * r_fmt
        r_reason = fmt_w * r_fmt
        r_box = (normal_fp_pen if pred_px is not None else 0.0) + fmt_w * r_fmt
        zeros.update(
            {
                "R_ground": float(r_ground),
                "R_reason": float(r_reason),
                "R_box": float(r_box),
                "R": float(r_cls),
            }
        )
        return zeros

    if gt_box_px is None:
        zeros.update(
            {
                "R_ground": fmt_w * r_fmt,
                "R_reason": fmt_w * r_fmt,
                "R_box": fmt_w * r_fmt,
                "R": float(r_cls),
            }
        )
        return zeros

    w_img, h_img = float(max(orig_wh[0], 1)), float(max(orig_wh[1], 1))
    img_area = w_img * h_img
    r_cov = box_coverage(cand_px, gt_box_px) if cand_px is not None else 0.0
    r_compact = 0.0
    if cand_px is not None:
        r_compact = float(max(0.0, min(1.0, 1.0 - box_area(cand_px) / max(img_area, 1.0))))
    r_iou = box_iou(pred_px, gt_box_px) if pred_px is not None else 0.0
    r_iou_c = box_iou(cand_px, gt_box_px) if cand_px is not None else 0.0
    r_edge = edge_precision_reward(pred_px, gt_box_px, orig_wh, beta) if pred_px is not None else 0.0
    r_center = center_reward(pred_px, gt_box_px, orig_wh, gamma) if pred_px is not None else 0.0

    d_star: Dict[str, str] = {}
    if cand_px is not None:
        d_star = boundary_targets(
            pixels_to_qwen1000(cand_px, orig_wh),
            pixels_to_qwen1000(gt_box_px, orig_wh),
            keep_tol,
        )
    pred_d = parsed.get("boundary") or {}
    r_dir = (sum(1 for k in EDGE_KEYS if pred_d.get(k) == d_star.get(k)) / 4.0) if d_star else 0.0

    r_ground = w_cov * r_cov + w_compact * r_compact + fmt_w * r_fmt
    r_reason = r_dir + fmt_w * r_fmt
    r_box = w_iou * r_iou + w_edge * r_edge + w_center * r_center + fmt_w * r_fmt
    return {
        "R_cov": float(r_cov),
        "R_compact": float(r_compact),
        "R_dir": float(r_dir),
        "R_iou": float(r_iou),
        "R_iou_c": float(r_iou_c),
        "R_edge": float(r_edge),
        "R_center": float(r_center),
        "R_format": float(r_fmt),
        "R_cls": float(r_cls),
        "R_ground": float(r_ground),
        "R_reason": float(r_reason),
        "R_box": float(r_box),
        "R": float(r_box),
        "pred_box_px": pred_px,
        "cand_box_px": cand_px,
        "pred_cls": pred_cls,
        "d_star": d_star,
    }
