"""Parse Grounded Comparative CoT (XML tags) and compute process/outcome GRPO rewards."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.common import qwen_norm1000_to_original_pixels

_EDGE_KEYS = ("L", "R", "T", "B")
_TAG_NAMES = ("compare", "ground", "verify", "boundary", "answer")
_TAG_BLOCK_RE = re.compile(
    r"<(compare|ground|verify|boundary|answer)\s*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_BOX_ASSIGN_RE = re.compile(
    r"(?:candidate_bbox|bbox_2d|final_bbox|bbox)\s*=\s*(null|none|n/a|\[.*?\])",
    re.IGNORECASE | re.DOTALL,
)
_BARE_BOX_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?\s*,\s*){3}-?\d+(?:\.\d+)?\s*\]")
_EDGE_LINE_RE = {
    "L": re.compile(r"(?:^|\n)\s*(?:left|l)\s*[:\-]\s*([^\n<]+)", re.IGNORECASE),
    "R": re.compile(r"(?:^|\n)\s*(?:right|r)\s*[:\-]\s*([^\n<]+)", re.IGNORECASE),
    "T": re.compile(r"(?:^|\n)\s*(?:top|t)\s*[:\-]\s*([^\n<]+)", re.IGNORECASE),
    "B": re.compile(r"(?:^|\n)\s*(?:bottom|b)\s*[:\-]\s*([^\n<]+)", re.IGNORECASE),
}


def strip_think(text: str) -> str:
    s = text.strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    return s


def extract_tag_blocks(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in _TAG_BLOCK_RE.finditer(text):
        out[m.group(1).lower()] = m.group(2).strip()
    return out


def _as_box(v) -> Optional[List[float]]:
    if v is None or v is False:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("null", "none", "normal", "n/a", "nil", ""):
            return None
        m = _BARE_BOX_RE.search(v)
        if m:
            try:
                arr = json.loads(m.group(0))
                return _as_box(arr)
            except Exception:
                return None
        return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            return [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
        except (TypeError, ValueError):
            return None
    return None


def _box_from_text(chunk: str, *, prefer: Sequence[str] = ()) -> Optional[List[float]]:
    if not chunk:
        return None
    for key in prefer:
        m = re.search(rf"(?<![A-Za-z_]){re.escape(key)}\s*=\s*(null|none|n/a|\[.*?\])", chunk, re.I | re.S)
        if m:
            return _as_box(m.group(1))
    if prefer:
        return None
    m = _BOX_ASSIGN_RE.search(chunk)
    if m:
        return _as_box(m.group(1))
    m = _BARE_BOX_RE.search(chunk)
    if m:
        try:
            return _as_box(json.loads(m.group(0)))
        except Exception:
            return None
    return None


def _expand_to_move(edge: str) -> str:
    return {"L": "move_left", "R": "move_right", "T": "move_up", "B": "move_down"}[edge]


def _shrink_to_move(edge: str) -> str:
    return {"L": "move_right", "R": "move_left", "T": "move_down", "B": "move_up"}[edge]


def _parse_dir(raw: str, edge: str) -> Optional[str]:
    s = str(raw).strip().lower()
    if not s:
        return None
    if any(k in s for k in ("keep", "hold", "same", "unchanged")):
        return "keep"
    if "move left" in s or "shift left" in s:
        return "move_left"
    if "move right" in s or "shift right" in s:
        return "move_right"
    if "move up" in s or "shift up" in s:
        return "move_up"
    if "move down" in s or "shift down" in s:
        return "move_down"
    if "expand" in s or "enlarge" in s or "outward" in s:
        return _expand_to_move(edge)
    if "shrink" in s or "contract" in s or "inward" in s:
        return _shrink_to_move(edge)
    if s in ("left", "l"):
        return "move_left"
    if s in ("right", "r"):
        return "move_right"
    if s in ("up", "t"):
        return "move_up"
    if s in ("down", "b"):
        return "move_down"
    return None


def _norm_boundary(text_or_dict: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if isinstance(text_or_dict, dict):
        alias = {"left": "L", "right": "R", "top": "T", "bottom": "B", "l": "L", "r": "R", "t": "T", "b": "B"}
        for k, v in text_or_dict.items():
            edge = alias.get(str(k).lower(), str(k).upper() if str(k).upper() in _EDGE_KEYS else None)
            if edge is None:
                continue
            d = _parse_dir(v, edge)
            if d:
                out[edge] = d
        return out
    blob = f"\n{text_or_dict or ''}"
    for edge, pat in _EDGE_LINE_RE.items():
        m = pat.search(blob)
        if not m:
            continue
        d = _parse_dir(m.group(1), edge)
        if d:
            out[edge] = d
    return out


def parse_cot_output(text: str) -> Dict[str, Any]:
    """Parse XML Grounded CoT; JSON is only a fallback."""
    raw = strip_think(text)
    tags = extract_tag_blocks(raw)
    ground_txt = tags.get("ground", "")
    answer_txt = tags.get("answer", "")
    bound_txt = tags.get("boundary", "")
    json_obj: Dict[str, Any] = {}
    if not tags:
        try:
            obj = json.loads(raw)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                obj = obj[0]
            if isinstance(obj, dict):
                json_obj = obj
        except Exception:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    if isinstance(obj, dict):
                        json_obj = obj
                except Exception:
                    json_obj = {}

    cand = _box_from_text(ground_txt, prefer=("candidate_bbox", "bbox_c")) if ground_txt else None
    bbox = _box_from_text(answer_txt, prefer=("bbox_2d", "final_bbox", "bbox")) if answer_txt else None
    if cand is None:
        cand = _box_from_text(raw, prefer=("candidate_bbox", "bbox_c"))
    if cand is None:
        cand = _as_box(json_obj.get("candidate_bbox") or json_obj.get("bbox_c"))
    if bbox is None:
        bbox = _box_from_text(raw, prefer=("bbox_2d", "final_bbox", "bbox"))
    if bbox is None:
        bbox = _as_box(json_obj.get("bbox_2d") or json_obj.get("final_bbox") or json_obj.get("bbox"))

    boundary = _norm_boundary(bound_txt or json_obj.get("boundary") or json_obj.get("D") or raw)
    hypo_src = " ".join(
        [
            tags.get("compare", ""),
            tags.get("ground", ""),
            tags.get("answer", ""),
            str(json_obj.get("hypothesize") or json_obj.get("label") or ""),
        ]
    ).lower()
    is_anom = None
    if any(k in hypo_src for k in ("normal", "no defect", "defect-free", "false positive")) and bbox is None:
        is_anom = False
    elif bbox is not None:
        is_anom = True
    elif any(k in hypo_src for k in ("anomaly", "abnormal", "defect")):
        is_anom = True

    has_tags = all(n in tags for n in _TAG_NAMES)
    format_ok = bool(has_tags)
    return {
        "raw": tags if tags else (json_obj or raw),
        "tags": tags,
        "has_tags": has_tags,
        "format_ok": format_ok,
        "bbox_2d": bbox,
        "candidate_bbox": cand,
        "boundary": boundary,
        "is_anomaly": is_anom if is_anom is not None else (bbox is not None),
        "label": json_obj.get("label") if json_obj else None,
        "text": raw,
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
    l = "keep" if abs(cx1 - gx1) <= keep_tol else ("move_right" if cx1 < gx1 else "move_left")
    r = "keep" if abs(cx2 - gx2) <= keep_tol else ("move_left" if cx2 > gx2 else "move_right")
    t = "keep" if abs(cy1 - gy1) <= keep_tol else ("move_down" if cy1 < gy1 else "move_up")
    b = "keep" if abs(cy2 - gy2) <= keep_tol else ("move_up" if cy2 > gy2 else "move_down")
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
    has_tags = all(n in tags for n in _TAG_NAMES)
    if not has_tags:
        return False
    if not is_anomaly:
        return parsed.get("bbox_2d") is None
    bound = parsed.get("boundary") or {}
    return (
        parsed.get("candidate_bbox") is not None
        and parsed.get("bbox_2d") is not None
        and all(k in bound for k in _EDGE_KEYS)
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
    normal_bonus = float(rew_cfg.get("normal_correct", 1.0))
    normal_fp_pen = float(rew_cfg.get("normal_false_positive", -0.5))

    pred_f = parsed.get("bbox_2d")
    cand = parsed.get("candidate_bbox") or pred_f
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
        "R_edge": 0.0,
        "R_center": 0.0,
        "R_format": float(r_fmt),
        "pred_box_px": pred_px,
        "d_star": {},
    }

    if not is_anomaly:
        r = normal_fp_pen if pred_px is not None else normal_bonus
        r = float(r) + fmt_w * r_fmt
        zeros.update(
            {
                "R_ground": float(r),
                "R_reason": float(r),
                "R_box": float(r),
                "R": float(r),
            }
        )
        return zeros

    if gt_box_px is None:
        zeros.update({"R_ground": fmt_w * r_fmt, "R_reason": fmt_w * r_fmt, "R_box": fmt_w * r_fmt, "R": fmt_w * r_fmt})
        return zeros

    w_img, h_img = float(max(orig_wh[0], 1)), float(max(orig_wh[1], 1))
    img_area = w_img * h_img
    r_cov = box_coverage(cand_px, gt_box_px) if cand_px is not None else 0.0
    r_compact = 0.0
    if cand_px is not None:
        r_compact = float(max(0.0, min(1.0, 1.0 - box_area(cand_px) / max(img_area, 1.0))))
    r_iou = box_iou(pred_px, gt_box_px) if pred_px is not None else 0.0
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
    r_dir = (sum(1 for k in _EDGE_KEYS if pred_d.get(k) == d_star.get(k)) / 4.0) if d_star else 0.0

    r_ground = w_cov * r_cov + w_compact * r_compact + fmt_w * r_fmt
    r_reason = r_dir + fmt_w * r_fmt
    r_box = w_iou * r_iou + w_edge * r_edge + w_center * r_center + fmt_w * r_fmt
    return {
        "R_cov": float(r_cov),
        "R_compact": float(r_compact),
        "R_dir": float(r_dir),
        "R_iou": float(r_iou),
        "R_edge": float(r_edge),
        "R_center": float(r_center),
        "R_format": float(r_fmt),
        "R_ground": float(r_ground),
        "R_reason": float(r_reason),
        "R_box": float(r_box),
        "R": float(r_box),
        "pred_box_px": pred_px,
        "d_star": d_star,
    }


def _seg_role(tag: str) -> str:
    t = tag.lower()
    if t in ("compare", "ground"):
        return "ground"
    if t in ("verify", "boundary"):
        return "reason"
    return "box"


def completion_segment_ids(tokenizer, completion_ids: Sequence[int]) -> List[str]:
    """Map each completion token to ground / reason / box."""
    ids = completion_ids.tolist() if hasattr(completion_ids, "tolist") else list(completion_ids)
    if not ids:
        return []
    full = tokenizer.decode(ids, skip_special_tokens=True)
    spans: List[Tuple[int, int, str]] = []
    for m in _TAG_BLOCK_RE.finditer(full):
        spans.append((m.start(), m.end(), _seg_role(m.group(1))))
    if not spans:
        return ["box"] * len(ids)
    segs: List[str] = []
    for i in range(len(ids)):
        cur = tokenizer.decode(ids[: i + 1], skip_special_tokens=True)
        pos = max(len(cur) - 1, 0)
        role = None
        for a, b, name in spans:
            if a <= pos < b:
                role = name
                break
        if role is None:
            role = "ground" if spans and pos < spans[0][0] else "box"
        if not cur:
            role = segs[-1] if segs else "ground"
        segs.append(role)
    return segs


def mix_segment_advantage(seg: str, a_ground: float, a_reason: float, a_box: float, rew_cfg: Optional[dict] = None) -> float:
    cfg = rew_cfg or {}
    if seg == "ground":
        return float(cfg.get("a_ground_on_ground", 0.7)) * a_ground + float(cfg.get("a_box_on_ground", 0.3)) * a_box
    if seg == "reason":
        return float(cfg.get("a_reason_on_reason", 0.5)) * a_reason + float(cfg.get("a_box_on_reason", 0.5)) * a_box
    return float(a_box)
