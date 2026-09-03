"""Parse Grounded Comparative CoT: tags, bbox, boundary, final decision."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

EDGE_KEYS = ("L", "R", "T", "B")
EDGE_CHOICES = ("inward", "outward", "keep")
TAG_NAMES = ("compare", "ground", "verify", "boundary", "answer")
TAG_BLOCK_RE = re.compile(
    r"<(compare|ground|verify|boundary|answer)\s*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_BOX_ASSIGN_RE = re.compile(
    r"(?:candidate_bbox|bbox_2d|final_bbox|bbox)\s*=\s*(null|none|n/a|\[.*?\])",
    re.IGNORECASE | re.DOTALL,
)
_BARE_BOX_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?\s*,\s*){3}-?\d+(?:\.\d+)?\s*\]")
_FINAL_CLS_RE = re.compile(r"is_anomaly\s*[:=]\s*(true|false)", re.IGNORECASE)
_EDGE_LINE_RE = {
    "L": re.compile(r"(?:^|\n)\s*(?:left|l)\s*=\s*(inward|outward|keep)\s*(?:\n|$)", re.IGNORECASE),
    "R": re.compile(r"(?:^|\n)\s*(?:right|r)\s*=\s*(inward|outward|keep)\s*(?:\n|$)", re.IGNORECASE),
    "T": re.compile(r"(?:^|\n)\s*(?:top|t)\s*=\s*(inward|outward|keep)\s*(?:\n|$)", re.IGNORECASE),
    "B": re.compile(r"(?:^|\n)\s*(?:bottom|b)\s*=\s*(inward|outward|keep)\s*(?:\n|$)", re.IGNORECASE),
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
    for m in TAG_BLOCK_RE.finditer(text):
        out[m.group(1).lower()] = m.group(2).strip()
    return out


def parse_bbox(v) -> Optional[List[float]]:
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
                return parse_bbox(arr)
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
            return parse_bbox(m.group(1))
    if prefer:
        return None
    m = _BOX_ASSIGN_RE.search(chunk)
    if m:
        return parse_bbox(m.group(1))
    m = _BARE_BOX_RE.search(chunk)
    if m:
        try:
            return parse_bbox(json.loads(m.group(0)))
        except Exception:
            return None
    return None


def parse_final_decision(answer: str) -> Optional[bool]:
    m = _FINAL_CLS_RE.search(answer or "")
    if not m:
        return None
    return m.group(1).lower() == "true"


def _parse_edge_choice(raw: str) -> Optional[str]:
    s = str(raw).strip().lower()
    if s in EDGE_CHOICES:
        return s
    return None


def parse_boundary(text_or_dict: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if isinstance(text_or_dict, dict):
        alias = {"left": "L", "right": "R", "top": "T", "bottom": "B", "l": "L", "r": "R", "t": "T", "b": "B"}
        for k, v in text_or_dict.items():
            edge = alias.get(str(k).lower(), str(k).upper() if str(k).upper() in EDGE_KEYS else None)
            if edge is None:
                continue
            d = _parse_edge_choice(v)
            if d:
                out[edge] = d
        return out
    blob = f"\n{text_or_dict or ''}"
    for edge, pat in _EDGE_LINE_RE.items():
        m = pat.search(blob)
        if not m:
            continue
        d = _parse_edge_choice(m.group(1))
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
        cand = parse_bbox(json_obj.get("candidate_bbox") or json_obj.get("bbox_c"))
    if bbox is None:
        bbox = _box_from_text(raw, prefer=("bbox_2d", "final_bbox", "bbox"))
    if bbox is None:
        bbox = parse_bbox(json_obj.get("bbox_2d") or json_obj.get("final_bbox") or json_obj.get("bbox"))

    boundary = parse_boundary(bound_txt or json_obj.get("boundary") or json_obj.get("D") or "")
    is_anom = parse_final_decision(answer_txt)

    has_tags = all(n in tags for n in TAG_NAMES)
    return {
        "raw": tags if tags else (json_obj or raw),
        "tags": tags,
        "has_tags": has_tags,
        "format_ok": bool(has_tags),
        "bbox_2d": bbox,
        "candidate_bbox": cand,
        "boundary": boundary,
        "is_anomaly": is_anom,
        "label": json_obj.get("label") if json_obj else None,
        "text": raw,
    }
