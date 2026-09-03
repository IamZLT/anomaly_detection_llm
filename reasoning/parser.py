"""Parse Grounded Comparative CoT: tags, bbox, boundary, final decision."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

BOX_STATE_BOX = "box"
BOX_STATE_NULL = "null"
BOX_STATE_MISSING = "missing"
BOX_STATE_INVALID = "invalid"
EDGE_KEYS = ("L", "R", "T", "B")
EDGE_CHOICES = ("inward", "outward", "keep")
TAG_NAMES = ("compare", "ground", "verify", "boundary", "answer")
_COPY_MARKERS = (
    "return exactly five",
    "do not copy",
    "xml blocks",
    "output exactly one json",
    "candidate_bbox_2d=[x1,y1,x2,y2]",
    "only spatial hints",
    "not defect labels",
    "high_response_points_2d",
)
TAG_BLOCK_RE = re.compile(
    r"<(compare|ground|verify|boundary|answer)\s*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_BOX_ASSIGN_RE = re.compile(
    r"(?:candidate_bbox_2d|candidate_bbox|bbox_2d|final_bbox|bbox)\s*=\s*(null|none|n/a|\[.*?\])",
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


def _valid_bbox_1000(box) -> bool:
    if box is None or not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = map(float, box)
    except (TypeError, ValueError):
        return False
    return 0.0 <= x1 < x2 <= 1000.0 and 0.0 <= y1 < y2 <= 1000.0


def parse_bbox_field(text: str, field_name: str) -> tuple:
    if not text or not str(text).strip():
        return BOX_STATE_MISSING, None
    pat = re.search(
        rf"(?<![A-Za-z_]){re.escape(field_name)}\s*[:=]\s*(null|none|\[[^\]]*\])",
        text,
        re.I | re.S,
    )
    if pat is None:
        return BOX_STATE_MISSING, None
    raw = pat.group(1).strip()
    if raw.lower() in ("null", "none"):
        return BOX_STATE_NULL, None
    try:
        value = json.loads(raw)
    except Exception:
        return BOX_STATE_INVALID, None
    if isinstance(value, list) and len(value) == 4 and all(isinstance(x, (int, float)) for x in value):
        return BOX_STATE_BOX, [float(x) for x in value]
    return BOX_STATE_INVALID, None


def parse_answer_block(answer_txt: str) -> Dict[str, Any]:
    blob = (answer_txt or "").strip()
    if not blob:
        return {
            "answer_state": BOX_STATE_MISSING,
            "is_anomaly": None,
            "final_bbox_state": BOX_STATE_MISSING,
            "bbox_2d": None,
        }
    obj = None
    try:
        obj = json.loads(blob)
    except Exception:
        obj = None
    if not isinstance(obj, dict):
        return {
            "answer_state": BOX_STATE_INVALID,
            "is_anomaly": None,
            "final_bbox_state": BOX_STATE_INVALID,
            "bbox_2d": None,
        }
    pred_cls = obj.get("is_anomaly", None)
    if not isinstance(pred_cls, bool):
        pred_cls = None
    if "bbox_2d" not in obj:
        bbox_state, bbox = BOX_STATE_MISSING, None
    elif obj["bbox_2d"] is None:
        bbox_state, bbox = BOX_STATE_NULL, None
    else:
        bbox = parse_bbox(obj["bbox_2d"])
        bbox_state = BOX_STATE_BOX if bbox is not None else BOX_STATE_INVALID
    return {
        "answer_state": "ok",
        "is_anomaly": pred_cls,
        "final_bbox_state": bbox_state,
        "bbox_2d": bbox,
    }


def parse_boundary_block(bound_txt: str) -> tuple:
    t = (bound_txt or "").strip()
    if not t:
        return BOX_STATE_INVALID, {}
    if t.lower() == "not_applicable":
        return "not_applicable", {}
    boundary = parse_boundary(t)
    return ("complete" if len(boundary) == 4 else BOX_STATE_INVALID), boundary


def structural_prose_ok(chunk: str, min_chars: int = 12) -> bool:
    s = re.sub(
        r"candidate_bbox_2d\s*[:=]\s*(null|none|\[[^\]]*\])",
        "",
        chunk or "",
        flags=re.I,
    )
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) < min_chars:
        return False
    low = s.lower()
    if any(m in low for m in _COPY_MARKERS):
        return False
    tokens = re.findall(r"[A-Za-z]+", s)
    if tokens and len(set(t.lower() for t in tokens)) <= 1:
        return False
    return True


def _trajectory_valid(
    *,
    has_tags: bool,
    tags: Dict[str, str],
    candidate_state: str,
    candidate_bbox,
    boundary_state: str,
    answer: Dict[str, Any],
) -> bool:
    pred_cls = answer.get("is_anomaly")
    compare_ok = structural_prose_ok(tags.get("compare", ""))
    ground_ok = structural_prose_ok(tags.get("ground", ""))
    verify_ok = structural_prose_ok(tags.get("verify", ""))
    prose_ok = compare_ok and ground_ok and verify_ok
    if pred_cls is False:
        return (
            has_tags
            and prose_ok
            and candidate_state == BOX_STATE_NULL
            and boundary_state == "not_applicable"
            and answer.get("answer_state") == "ok"
            and answer.get("final_bbox_state") == BOX_STATE_NULL
        )
    if pred_cls is True:
        return (
            has_tags
            and prose_ok
            and candidate_state == BOX_STATE_BOX
            and _valid_bbox_1000(candidate_bbox)
            and boundary_state == "complete"
            and answer.get("answer_state") == "ok"
            and answer.get("final_bbox_state") == BOX_STATE_BOX
            and _valid_bbox_1000(answer.get("bbox_2d"))
        )
    return False


def _pack_parsed(
    *,
    raw: str,
    tags: Dict[str, str],
    candidate_state: str,
    candidate_bbox,
    boundary_state: str,
    boundary: dict,
    answer: Dict[str, Any],
    strict: bool,
    extra: Optional[dict] = None,
) -> Dict[str, Any]:
    has_tags = all(n in tags for n in TAG_NAMES)
    cand = candidate_bbox if candidate_state == BOX_STATE_BOX else None
    out = {
        "raw": tags if tags else raw,
        "tags": tags,
        "has_tags": has_tags,
        "format_ok": bool(has_tags),
        "candidate_bbox_2d": cand,
        "candidate_bbox": cand,
        "candidate_bbox_state": candidate_state,
        "boundary": boundary,
        "boundary_state": boundary_state,
        "is_anomaly": answer.get("is_anomaly"),
        "bbox_2d": answer.get("bbox_2d"),
        "final_bbox_state": answer.get("final_bbox_state"),
        "answer_state": answer.get("answer_state"),
        "prose_ok": structural_prose_ok(tags.get("compare", ""))
        and structural_prose_ok(tags.get("ground", ""))
        and structural_prose_ok(tags.get("verify", "")),
        "trajectory_valid": _trajectory_valid(
            has_tags=has_tags,
            tags=tags,
            candidate_state=candidate_state,
            candidate_bbox=cand,
            boundary_state=boundary_state,
            answer=answer,
        ),
        "text": raw,
        "strict": strict,
        "label": None,
    }
    if extra:
        out.update(extra)
    return out


def rollout_protocol_stats(parsed_list: Sequence[dict], texts: Optional[Sequence[str]] = None) -> Dict[str, float]:
    n = max(len(parsed_list), 1)

    def rate(pred) -> float:
        return float(sum(1 for p in parsed_list if pred(p))) / n

    stats = {
        "protocol_rate": rate(lambda p: bool(p.get("has_tags"))),
        "trajectory_valid_rate": rate(lambda p: bool(p.get("trajectory_valid"))),
        "candidate_box_rate": rate(lambda p: p.get("candidate_bbox_state") == BOX_STATE_BOX),
        "candidate_valid_rate": rate(
            lambda p: p.get("candidate_bbox_state") == BOX_STATE_BOX
            and _valid_bbox_1000(p.get("candidate_bbox_2d") or p.get("candidate_bbox"))
        ),
        "final_box_rate": rate(lambda p: p.get("final_bbox_state") == BOX_STATE_BOX),
        "final_valid_rate": rate(
            lambda p: p.get("final_bbox_state") == BOX_STATE_BOX and _valid_bbox_1000(p.get("bbox_2d"))
        ),
        "boundary_complete_rate": rate(lambda p: p.get("boundary_state") == "complete"),
        "normal_null_consistency_rate": rate(
            lambda p: p.get("is_anomaly") is False
            and p.get("candidate_bbox_state") == BOX_STATE_NULL
            and p.get("boundary_state") == "not_applicable"
            and p.get("final_bbox_state") == BOX_STATE_NULL
        ),
    }
    if texts is not None:
        stats["unique_response_rate"] = float(len(set(texts))) / max(len(texts), 1)
    return stats


def parse_cot_output(text: str) -> Dict[str, Any]:
    """Strict process-aware parse: Bc from <ground>, D from <boundary>, Bf/ŷ from <answer>."""
    raw = strip_think(text)
    tags = extract_tag_blocks(raw)
    candidate_state, candidate_bbox = parse_bbox_field(tags.get("ground", ""), "candidate_bbox_2d")
    boundary_state, boundary = parse_boundary_block(tags.get("boundary", ""))
    answer = parse_answer_block(tags.get("answer", ""))
    return _pack_parsed(
        raw=raw,
        tags=tags,
        candidate_state=candidate_state,
        candidate_bbox=candidate_bbox,
        boundary_state=boundary_state,
        boundary=boundary,
        answer=answer,
        strict=True,
    )


def parse_cot_output_tolerant(text: str) -> Dict[str, Any]:
    """Demo / legacy fallback: also search whole text and JSON if tags are missing."""
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

    candidate_state, candidate_bbox = parse_bbox_field(ground_txt, "candidate_bbox_2d")
    if candidate_state == BOX_STATE_MISSING:
        candidate_state, candidate_bbox = parse_bbox_field(ground_txt, "candidate_bbox")
    if candidate_state == BOX_STATE_MISSING and json_obj:
        if "candidate_bbox_2d" in json_obj or "candidate_bbox" in json_obj or "bbox_c" in json_obj:
            raw_c = json_obj.get("candidate_bbox_2d", json_obj.get("candidate_bbox", json_obj.get("bbox_c")))
            if raw_c is None:
                candidate_state, candidate_bbox = BOX_STATE_NULL, None
            else:
                box = parse_bbox(raw_c)
                candidate_state = BOX_STATE_BOX if box is not None else BOX_STATE_INVALID
                candidate_bbox = box

    boundary_state, boundary = parse_boundary_block(bound_txt)
    if not bound_txt and json_obj:
        bsrc = json_obj.get("boundary") or json_obj.get("D") or ""
        if isinstance(bsrc, dict):
            boundary = parse_boundary(bsrc)
            boundary_state = "complete" if len(boundary) == 4 else BOX_STATE_INVALID
        elif bsrc:
            boundary_state, boundary = parse_boundary_block(str(bsrc))

    answer = parse_answer_block(answer_txt)
    if answer["answer_state"] != "ok":
        pred = parse_final_decision(answer_txt)
        st, box = parse_bbox_field(answer_txt, "bbox_2d")
        if st == BOX_STATE_MISSING:
            st, box = parse_bbox_field(answer_txt, "bbox")
        if pred is not None or st != BOX_STATE_MISSING:
            answer = {
                "answer_state": "ok" if pred is not None else BOX_STATE_INVALID,
                "is_anomaly": pred,
                "final_bbox_state": st,
                "bbox_2d": box if st == BOX_STATE_BOX else None,
            }
    if answer["answer_state"] != "ok" and json_obj:
        pred = json_obj.get("is_anomaly")
        if not isinstance(pred, bool):
            pred = parse_final_decision(str(pred)) if pred is not None else None
            if pred is None and str(json_obj.get("label", "")).lower() == "normal":
                pred = False
        if "bbox_2d" in json_obj:
            if json_obj["bbox_2d"] is None:
                st, box = BOX_STATE_NULL, None
            else:
                box = parse_bbox(json_obj["bbox_2d"])
                st = BOX_STATE_BOX if box is not None else BOX_STATE_INVALID
        elif "bbox" in json_obj:
            if json_obj["bbox"] is None:
                st, box = BOX_STATE_NULL, None
            else:
                box = parse_bbox(json_obj["bbox"])
                st = BOX_STATE_BOX if box is not None else BOX_STATE_INVALID
        else:
            st, box = BOX_STATE_MISSING, None
        if pred is not None or st != BOX_STATE_MISSING:
            answer = {
                "answer_state": "ok" if pred is not None else BOX_STATE_INVALID,
                "is_anomaly": pred,
                "final_bbox_state": st,
                "bbox_2d": box if st == BOX_STATE_BOX else None,
            }

    packed = _pack_parsed(
        raw=raw,
        tags=tags,
        candidate_state=candidate_state,
        candidate_bbox=candidate_bbox,
        boundary_state=boundary_state,
        boundary=boundary,
        answer=answer,
        strict=False,
        extra={"label": json_obj.get("label") if json_obj else None},
    )
    if json_obj and not tags:
        packed["raw"] = json_obj
    return packed

