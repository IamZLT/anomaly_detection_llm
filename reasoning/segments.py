"""Process-aware credit assignment: map CoT tokens to ground / reason / answer."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from reasoning.parser import TAG_BLOCK_RE


def _seg_role(tag: str) -> str:
    t = tag.lower()
    if t in ("compare", "ground"):
        return "ground"
    if t in ("verify", "boundary"):
        return "reason"
    return "answer"


def completion_segment_ids(tokenizer, completion_ids: Sequence[int]) -> List[str]:
    """Map each completion token to ground / reason / answer."""
    ids = completion_ids.tolist() if hasattr(completion_ids, "tolist") else list(completion_ids)
    if not ids:
        return []
    full = tokenizer.decode(ids, skip_special_tokens=True)
    spans: List[Tuple[int, int, str]] = []
    for m in TAG_BLOCK_RE.finditer(full):
        spans.append((m.start(), m.end(), _seg_role(m.group(1))))
    if not spans:
        return ["answer"] * len(ids)
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
            role = "ground" if spans and pos < spans[0][0] else "answer"
        if not cur:
            role = segs[-1] if segs else "ground"
        segs.append(role)
    return segs


def mix_segment_advantage(
    seg: str,
    a_ground: float,
    a_reason: float,
    a_box: float,
    a_cls: float,
    rew_cfg: Optional[dict] = None,
) -> float:
    cfg = rew_cfg or {}
    if seg == "ground":
        return (
            float(cfg.get("a_ground_on_ground", 0.6)) * a_ground
            + float(cfg.get("a_cls_on_ground", 0.2)) * a_cls
            + float(cfg.get("a_box_on_ground", 0.2)) * a_box
        )
    if seg == "reason":
        return (
            float(cfg.get("a_reason_on_reason", 0.3)) * a_reason
            + float(cfg.get("a_cls_on_reason", 0.3)) * a_cls
            + float(cfg.get("a_box_on_reason", 0.4)) * a_box
        )
    return float(cfg.get("a_cls_on_answer", 0.7)) * a_cls + float(cfg.get("a_box_on_answer", 0.3)) * a_box
