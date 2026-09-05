"""Three-level verifiable rewards: candidate grounding, refinement direction, final gate."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

EDGE_KEYS = ("L", "R", "T", "B")

DEFAULT_FORMAT_WEIGHTS = {
    "understand": 0.10,
    "compare": 0.15,
    "ground": 0.30,
    "verify": 0.15,
    "answer": 0.30,
}


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
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, box)
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def h_anchor_reward(
    cand_box: Optional[List[float]],
    prior_box: Optional[List[float]],
) -> float:
    """IoU between the candidate proposal and the H-derived prior box.

    Both boxes are in the Qwen 0-1000 system. The prior box is a *coarse hint*,
    not a defect label: this term only shapes the candidate (R_ground) and is
    damped by (1 - R_dense) so it can never outrank GT truth. It exists purely to
    break the zero-IoU dead zone — giving near-identical rollouts a group-relative
    ordering that pulls the candidate toward the H region.
    """
    if cand_box is None or prior_box is None:
        return 0.0
    return float(box_iou(cand_box, prior_box))


def dense_geometry_reward(
    pred: List[float],
    gt: List[float],
    orig_wh: Tuple[int, int],
    *,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> float:
    """Dense localization reward.

    Key property: even when IoU == 0, a box closer to GT receives a larger
    reward than a far-away box, removing the sparse-reward dead zone.

    Center distance is normalized by the image diagonal. Width and height use
    symmetric min/max ratios raised to `gamma`, and geometry is multiplicative
    (S_center * S_w * S_h) so neither a huge box nor a same-area wrong-aspect
    box can earn a large reward from center alone.

    `gamma > 1` sharpens S_w/S_h: a box with the right center but wrong size
    is pushed down faster, so raw IoU dominates as soon as boxes start to
    overlap. This kills the "right-center-wrong-scale" reward plateau where
    imprecise boxes all sit near the same ~0.4 reward and produce ~0 advantage.
    """
    if pred is None or gt is None:
        return 0.0

    iou = box_iou(pred, gt)

    cx_p, cy_p = box_center(pred)
    cx_g, cy_g = box_center(gt)

    dist = math.sqrt(
        (cx_p - cx_g) ** 2
        + (cy_p - cy_g) ** 2
    )

    w = float(max(orig_wh[0], 1))
    h = float(max(orig_wh[1], 1))
    image_diag = math.sqrt(w * w + h * h)

    s_center = 1.0 - min(
        1.0,
        dist / (image_diag + eps),
    )

    px1, py1, px2, py2 = map(float, pred)
    gx1, gy1, gx2, gy2 = map(float, gt)
    wp, hp = max(0.0, px2 - px1), max(0.0, py2 - py1)
    wg, hg = max(0.0, gx2 - gx1), max(0.0, gy2 - gy1)

    gamma = max(float(gamma), 1e-6)

    if wp <= eps or wg <= eps:
        s_w = 0.0
    else:
        s_w = (min(wp, wg) / max(wp, wg)) ** gamma

    if hp <= eps or hg <= eps:
        s_h = 0.0
    else:
        s_h = (min(hp, hg) / max(hp, hg)) ** gamma

    # Location and both side lengths must be plausible.
    s_geo = s_center * s_w * s_h
    s_geo = max(0.0, min(1.0, s_geo))

    reward = iou + (1.0 - iou) * s_geo

    return float(max(0.0, min(1.0, reward)))


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


def refinement_directions(source_box, target_box, keep_tol: float) -> Dict[str, str]:
    """φ(B_source, B_target): how each edge of the source box moved to the target."""
    sx1, sy1, sx2, sy2 = map(float, source_box)
    tx1, ty1, tx2, ty2 = map(float, target_box)
    return {
        "L": "keep" if abs(tx1 - sx1) <= keep_tol else ("inward" if tx1 > sx1 else "outward"),
        "R": "keep" if abs(tx2 - sx2) <= keep_tol else ("inward" if tx2 < sx2 else "outward"),
        "T": "keep" if abs(ty1 - sy1) <= keep_tol else ("inward" if ty1 > sy1 else "outward"),
        "B": "keep" if abs(ty2 - sy2) <= keep_tol else ("inward" if ty2 < sy2 else "outward"),
    }


def edge_precision_reward(
    pred: List[float],
    gt: List[float],
    orig_wh: Tuple[int, int],
    beta: float,
    min_frac: float = 0.05,
) -> float:
    w, h = float(max(orig_wh[0], 1)), float(max(orig_wh[1], 1))
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    gw = max(gx2 - gx1, float(min_frac) * w)
    gh = max(gy2 - gy1, float(min_frac) * h)
    dists = [
        abs(px1 - gx1) / gw,
        abs(px2 - gx2) / gw,
        abs(py1 - gy1) / gh,
        abs(py2 - gy2) / gh,
    ]
    return float(sum(math.exp(-beta * d) for d in dists) / 4.0)


def _to_px(box, orig_wh: Tuple[int, int]) -> Optional[List[float]]:
    if not valid_bbox_1000(box):
        return None
    return qwen1000_to_pixels_strict(box, orig_wh)


def format_reward(parsed: Dict[str, Any], fmt_weights: Dict[str, float]) -> float:
    tags = parsed.get("tags") or {}

    understand_ok = (
        "understand" in tags
        and bool(str(tags.get("understand", "")).strip())
    )

    compare_ok = (
        "compare" in tags
        and bool(str(tags.get("compare", "")).strip())
    )

    cand_state = parsed.get("candidate_bbox_state")
    cand = parsed.get("candidate_bbox_2d")

    ground_ok = (
        "ground" in tags
        and (
            cand_state == "null"
            or (
                cand_state == "box"
                and valid_bbox_1000(cand)
            )
        )
    )

    verify_ok = (
        "verify" in tags
        and bool(str(tags.get("verify", "")).strip())
    )

    pred = parsed.get("is_anomaly")
    final_state = parsed.get("final_bbox_state")
    final_box = parsed.get("bbox_2d")

    answer_schema_ok = (
        parsed.get("answer_state") == "ok"
        and isinstance(pred, bool)
    )

    if pred is True:
        answer_box_ok = (
            final_state == "box"
            and valid_bbox_1000(final_box)
        )
    elif pred is False:
        answer_box_ok = final_state == "null"
    else:
        answer_box_ok = False

    answer_ok = (
        "answer" in tags
        and answer_schema_ok
        and answer_box_ok
    )

    components = {
        "understand": understand_ok,
        "compare": compare_ok,
        "ground": ground_ok,
        "verify": verify_ok,
        "answer": answer_ok,
    }

    return float(
        sum(
            float(fmt_weights.get(name, 0.0))
            for name, ok in components.items()
            if ok
        )
    )


def compute_rewards(
    parsed: Dict[str, Any],
    gt_box_px: Optional[List[float]],
    orig_wh: Tuple[int, int],
    is_anomaly: bool,
    cfg: dict,
    prior_box: Optional[List[float]] = None,
) -> Dict[str, float]:
    rew_cfg = (cfg.get("grpo") or {}).get("reward") or {}
    # Paired convex weights: only one free parameter per pair.
    w_dir = float(rew_cfg.get("reason_dir_weight", 0.70))
    w_progress = 1.0 - w_dir
    # R_final = final_iou_weight * IoU(B_f) + final_dense_weight * dense(B_f)
    #           + final_edge_weight * edge.
    # Raw IoU is the only plateau-free term; including it forces B_f to be
    # genuinely refined instead of copied from B_c. The three weights are
    # normalized to a convex combination so R_final stays in [0, 1].
    w_iou_f = float(rew_cfg.get("final_iou_weight", 0.40))
    w_dense_f = float(rew_cfg.get("final_dense_weight", 0.50))
    w_edge = float(rew_cfg.get("final_edge_weight", 0.10))
    _wf = max(w_iou_f, 0.0) + max(w_dense_f, 0.0) + max(w_edge, 0.0)
    if _wf <= 0.0:
        w_iou_f, w_dense_f, w_edge = 0.4, 0.5, 0.1
    else:
        w_iou_f = max(w_iou_f, 0.0) / _wf
        w_dense_f = max(w_dense_f, 0.0) / _wf
        w_edge = max(w_edge, 0.0) / _wf
    dense_scale_gamma = float(rew_cfg.get("dense_scale_gamma", 2.0))
    beta = float(rew_cfg.get("edge_beta", 8.0))
    keep_tol = float(rew_cfg.get("keep_tol_norm1000", 8.0))
    r_ok = float(rew_cfg.get("normal_correct", 1.0))
    r_wrong = float(rew_cfg.get("wrong_decision", -1.0))
    r_invalid = float(rew_cfg.get("invalid_output", -1.0))
    edge_min_frac = float(rew_cfg.get("edge_min_frac", 0.05))
    large_box_area_thresh = float(rew_cfg.get("large_box_area_thresh", 0.80))
    h_anchor_weight = float(rew_cfg.get("h_anchor_weight", 0.15))
    fmt_weights = rew_cfg.get("format_weights") or DEFAULT_FORMAT_WEIGHTS

    pred_cls = parsed.get("is_anomaly")
    protocol_ok = bool(parsed.get("has_tags", False))
    traj_ok = bool(parsed.get("trajectory_valid", False))
    cand_state = parsed.get("candidate_bbox_state")
    cand = parsed.get("candidate_bbox_2d")
    final_box = parsed.get("bbox_2d")
    cand_px = _to_px(cand, orig_wh)
    final_px = _to_px(final_box, orig_wh)

    r_fmt = format_reward(parsed, fmt_weights)

    img_area = float(max(orig_wh[0], 1) * max(orig_wh[1], 1))

    def _area_ratio(box_px: Optional[List[float]]) -> float:
        if box_px is None:
            return 0.0
        return float(box_area(box_px) / img_area)

    cand_area_ratio = _area_ratio(cand_px)
    final_area_ratio = _area_ratio(final_px)
    gt_area = box_area(gt_box_px) if gt_box_px is not None else 0.0
    if final_px is not None and gt_area > 1e-6:
        pred_gt_area_ratio = float(box_area(final_px) / gt_area)
    else:
        pred_gt_area_ratio = 0.0
    full_image_cand = float(cand_area_ratio > large_box_area_thresh)
    full_image_final = float(final_area_ratio > large_box_area_thresh)

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
        delta_iou: float = 0.0,
        raw_iou_f: float = 0.0,
        raw_iou_c: float = 0.0,
        r_dense_c: float = 0.0,
        r_dense_f: float = 0.0,
        delta_dense: float = 0.0,
        r_fmt: float = 0.0,
        h_anchor: float = 0.0,
        h_follow: float = 0.0,
        h_override: float = 0.0,
        d_star: Optional[dict] = None,
    ) -> Dict[str, Any]:
        return {
            "R_ground": float(r_ground),
            "R_reason": float(r_reason),
            "R_final": float(r_final),
            "R_cov": float(r_cov),
            "R_iou_c": float(r_iou_c),
            "R_dir": float(r_dir),
            "R_iou": float(r_iou),
            "R_edge": float(r_edge),
            "delta_iou": float(delta_iou),
            "raw_iou_f": float(raw_iou_f),
            "raw_iou_c": float(raw_iou_c),
            "R_dense_c": float(r_dense_c),
            "R_dense_f": float(r_dense_f),
            "delta_dense": float(delta_dense),
            "R_fmt": float(r_fmt),
            "h_anchor": float(h_anchor),
            "h_follow": float(h_follow),
            "h_override": float(h_override),
            "candidate_area_ratio": float(cand_area_ratio),
            "final_area_ratio": float(final_area_ratio),
            "pred_gt_area_ratio": float(pred_gt_area_ratio),
            "full_image_cand": float(full_image_cand),
            "full_image_final": float(full_image_final),
            "pred_box_px": final_px,
            "cand_box_px": cand_px,
            "pred_is_anomaly": pred_cls,
            "protocol_ok": bool(protocol_ok),
            "trajectory_valid": bool(traj_ok),
            "d_star": d_star or {},
        }

    if not is_anomaly:
        # Normal samples have no GT localization target; the supervision is
        # asymmetric: classify correctly AND reject any localized proposal.
        # A degenerate full-image candidate is still a false-positive proposal
        # and must be penalized, otherwise the model can earn the full normal
        # reward by just answering false while proposing the whole image.
        if pred_cls is True:
            r_final = r_wrong
        elif pred_cls is False and traj_ok:
            if cand_state == "null":
                # Correct and directly decided: no suspicious proposal at all.
                r_final = r_ok
            elif full_image_cand > 0.5:
                # Degenerate full-image candidate on a normal image.
                r_final = r_invalid
            else:
                # A reasonable localized candidate was proposed then rejected
                # by verification: this is the desired rejection reasoning.
                r_final = r_ok
        else:
            r_final = r_invalid

        return _pack(
            r_ground=0.0,
            r_reason=0.0,
            r_final=r_final,
            r_fmt=r_fmt,
        )

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
            r_fmt=r_fmt,
        )

    cand_ok = cand_state == "box" and cand_px is not None
    # Coverage is diagnostic only; it must not enter R_ground (oversized-box shortcut).
    r_cov = box_coverage(cand_px, gt_box_px) if cand_ok else 0.0
    r_iou_c = box_iou(cand_px, gt_box_px) if cand_ok else 0.0
    r_dense_c = (
        dense_geometry_reward(cand_px, gt_box_px, orig_wh, gamma=dense_scale_gamma)
        if cand_ok
        else 0.0
    )
    r_ground = float(r_dense_c) if cand_ok else 0.0
    # H anchoring: only shapes the candidate in the zero-IoU dead zone. Damped by
    # (1 - R_dense_c) so it fades out as soon as the candidate overlaps GT — GT
    # truth always dominates, so a wrong H cannot drag a correct box away.
    h_anchor = h_anchor_reward(cand, prior_box) if cand_ok else 0.0
    if cand_ok:
        r_ground = r_ground + h_anchor_weight * (1.0 - r_dense_c) * h_anchor

    # Monitoring: did the candidate follow H, and did verify refine beyond it?
    h_follow = 0.0
    if cand is not None and prior_box is not None:
        cx, cy = box_center(cand)
        px1, py1, px2, py2 = prior_box
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            h_follow = 1.0
    h_override = 0.0
    if final_px is not None and cand_px is not None:
        iou_f_now = box_iou(final_px, gt_box_px)
        if iou_f_now > r_iou_c + 1e-6:
            h_override = 1.0

    d_star: Dict[str, str] = {}
    r_dir = 0.0
    if cand_ok and final_px is not None:
        # Bc/Bf are already in the 0-1000 system; only GT needs the pixel→1000 map.
        gt_1000 = pixels_to_qwen1000(gt_box_px, orig_wh)
        d_star = refinement_directions(cand, gt_1000, keep_tol)
        d_pred = refinement_directions(cand, final_box, keep_tol)
        r_dir = sum(1 for k in EDGE_KEYS if d_pred.get(k) == d_star.get(k)) / 4.0

    r_dense_f = (
        dense_geometry_reward(final_px, gt_box_px, orig_wh, gamma=dense_scale_gamma)
        if final_px is not None
        else 0.0
    )

    delta_dense = 0.0
    if cand_ok and final_px is not None:
        delta_dense = r_dense_f - r_dense_c

    r_reason = w_dir * r_dir + w_progress * delta_dense

    r_iou = 0.0
    r_edge = 0.0
    iou_f_diag = box_iou(final_px, gt_box_px) if final_px is not None else 0.0
    if r_final_gate is not None:
        r_final = r_final_gate
    elif final_px is None:
        r_final = r_invalid
    else:
        r_iou = iou_f_diag
        r_edge = edge_precision_reward(final_px, gt_box_px, orig_wh, beta, min_frac=edge_min_frac)
        r_final = w_iou_f * r_iou + w_dense_f * r_dense_f + w_edge * r_edge

    delta_iou = float(iou_f_diag - r_iou_c)

    return _pack(
        r_ground=r_ground,
        r_reason=r_reason,
        r_final=r_final,
        r_cov=r_cov,
        r_iou_c=r_iou_c,
        r_dir=r_dir,
        r_iou=r_iou,
        r_edge=r_edge,
        delta_iou=delta_iou,
        raw_iou_f=float(iou_f_diag),
        raw_iou_c=float(r_iou_c),
        r_dense_c=float(r_dense_c),
        r_dense_f=float(r_dense_f),
        delta_dense=float(delta_dense),
        r_fmt=r_fmt,
        h_anchor=h_anchor,
        h_follow=h_follow,
        h_override=h_override,
        d_star=d_star,
    )
