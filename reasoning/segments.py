"""Process-aware credit assignment: map CoT tokens to ground / reason / answer."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from reasoning.parser import TAG_BLOCK_RE

# Method-defined mix (not yaml hyperparameters).
_GROUND_AG, _GROUND_AF = 0.8, 0.2
_REASON_AR, _REASON_AF = 0.7, 0.3
_ANSWER_AR, _ANSWER_AF = 0.3, 0.7


def _seg_role(tag: str) -> str:
    t = tag.lower()
    # <understand> is unsupervised native reading: format credit only.
    if t == "understand":
        return "understand"
    if t in ("compare", "ground"):
        return "ground"
    if t == "verify":
        return "reason"
    return "answer"


def completion_segment_ids(tokenizer, completion_ids: Sequence[int]) -> List[str]:
    """Map each completion token to understand / ground / reason / answer."""
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
    a_final: float,
    a_fmt: float,
    is_anomaly: bool,
    fmt_mix: float = 0.20,
) -> float:
    # Unsupervised native understanding: only format advantage.
    if seg == "understand":
        return float(a_fmt)

    if not is_anomaly:
        base = a_final
    elif seg == "ground":
        base = _GROUND_AG * a_ground + _GROUND_AF * a_final
    elif seg == "reason":
        base = _REASON_AR * a_reason + _REASON_AF * a_final
    else:
        base = _ANSWER_AR * a_reason + _ANSWER_AF * a_final

    lam = max(0.0, min(float(fmt_mix), 1.0))

    return (1.0 - lam) * base + lam * a_fmt
