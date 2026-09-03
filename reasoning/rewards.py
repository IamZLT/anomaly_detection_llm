"""Three-level verifiable rewards: candidate grounding, boundary reasoning, final gate."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from reasoning.parser import EDGE_KEYS


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


def valid_bbox_1000(box) -> bool:
    if box is None or not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = map(float, box)
    except (TypeError, ValueError):
        return False
    return 0.0 <= x1 < x2 <= 1000.0 and 0.0 <= y1 < y2 <= 1000.0


def qwen1000_to_pixels_strict(box, orig_wh: Tuple[int, int]) -> List[float]:
    w, h = float(max(orig_wh[0], 1)), float(max(orig_wh[1], 1))
    x1, y1, x2, y2 = map(float, box)
    return [x1 / 1000.0 * w, y1 / 1000.0 * h, x2 / 1000.0 * w, y2 / 1000.0 * h]


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


def _to_px(box, orig_wh: Tuple[int, int]) -> Optional[List[float]]:
    if not valid_bbox_1000(box):
        return None
    return qwen1000_to_pixels_strict(box, orig_wh)


def compute_rewards(
    parsed: Dict[str, Any],
    gt_box_px: Optional[List[float]],
    orig_wh: Tuple[int, int],
    is_anomaly: bool,
    cfg: dict,
) -> Dict[str, float]:
    rew_cfg = (cfg.get("grpo") or {}).get("reward") or {}
    w_cov = float(rew_cfg.get("w_cov", 0.6))
    w_cand_iou = float(rew_cfg.get("w_cand_iou", 0.4))
    w_iou = float(rew_cfg.get("w_iou", 0.6))
    w_edge = float(rew_cfg.get("w_edge", 0.4))
    beta = float(rew_cfg.get("edge_beta", 8.0))
    keep_tol = float(rew_cfg.get("keep_tol_norm1000", 8.0))
    r_ok = float(rew_cfg.get("normal_correct", 1.0))
    r_wrong = float(rew_cfg.get("wrong_decision", -1.0))
    r_invalid = float(rew_cfg.get("invalid_output", -0.5))

    pred_cls = parsed.get("is_anomaly")
    protocol_ok = bool(parsed.get("has_tags", False))
    traj_ok = bool(parsed.get("trajectory_valid", False))
    cand_state = parsed.get("candidate_bbox_state")
    bound_state = parsed.get("boundary_state")
    cand = parsed.get("candidate_bbox_2d")
    if cand is None:
        cand = parsed.get("candidate_bbox")
    final_box = parsed.get("bbox_2d")
    cand_px = _to_px(cand, orig_wh)
    final_px = _to_px(final_box, orig_wh)

    def _pack(
        *,
        r_ground: float,
        r_reason: float,
        r_final: float,
        r_cov: float = 0.0,
        r_iou_c: float = 0.0,
        r_dir: float = 0.0,
        r_iou: float = 0.0,
        r_edge: float = 0.0,
        d_star: Optional[dict] = None,
    ) -> Dict[str, Any]:
        return {
            "R_ground": float(r_ground),
            "R_reason": float(r_reason),
            "R_final": float(r_final),
            "R": float(r_final),
            "R_cov": float(r_cov),
            "R_iou_c": float(r_iou_c),
            "R_dir": float(r_dir),
            "R_iou": float(r_iou),
            "R_edge": float(r_edge),
            "pred_box_px": final_px,
            "cand_box_px": cand_px,
            "pred_cls": pred_cls,
            "protocol_ok": bool(protocol_ok),
            "trajectory_valid": bool(traj_ok),
            "d_star": d_star or {},
        }

    if not is_anomaly:
        if pred_cls is True:
            r_final = r_wrong
        elif pred_cls is False and traj_ok:
            r_final = r_ok
        else:
            r_final = r_invalid
        return _pack(r_ground=0.0, r_reason=0.0, r_final=r_final)

    # Classification error beats protocol: explicit false on an anomaly is -1, not -0.5.
    if pred_cls is False:
        r_final_gate = r_wrong
    elif pred_cls is None:
        r_final_gate = r_invalid
    elif not traj_ok:
        r_final_gate = r_invalid
    else:
        r_final_gate = None

    if gt_box_px is None:
        return _pack(
            r_ground=0.0,
            r_reason=0.0,
            r_final=r_invalid if r_final_gate is None else r_final_gate,
        )

    cand_ok = cand_state == "box" and cand_px is not None
    r_cov = box_coverage(cand_px, gt_box_px) if cand_ok else 0.0
    r_iou_c = box_iou(cand_px, gt_box_px) if cand_ok else 0.0
    r_ground = (w_cov * r_cov + w_cand_iou * r_iou_c) if cand_ok else 0.0

    d_star: Dict[str, str] = {}
    r_dir = 0.0
    if cand_ok:
        d_star = boundary_targets(
            pixels_to_qwen1000(cand_px, orig_wh),
            pixels_to_qwen1000(gt_box_px, orig_wh),
            keep_tol,
        )
        pred_d = parsed.get("boundary") or {}
        r_dir = sum(1 for k in EDGE_KEYS if pred_d.get(k) == d_star.get(k)) / 4.0
    r_reason = r_dir if cand_ok and bound_state == "complete" else 0.0

    r_iou = 0.0
    r_edge = 0.0
    if r_final_gate is not None:
        r_final = r_final_gate
    else:
        r_iou = box_iou(final_px, gt_box_px)
        r_edge = edge_precision_reward(final_px, gt_box_px, orig_wh, beta)
        r_final = w_iou * r_iou + w_edge * r_edge

    return _pack(
        r_ground=r_ground,
        r_reason=r_reason,
        r_final=r_final,
        r_cov=r_cov,
        r_iou_c=r_iou_c,
        r_dir=r_dir,
        r_iou=r_iou,
        r_edge=r_edge,
        d_star=d_star,
    )
