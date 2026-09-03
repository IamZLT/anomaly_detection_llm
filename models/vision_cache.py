"""Frozen-vision embedding cache + H → soft spatial hints.

Qwen3.5 re-runs `visual()` whenever `pixel_values` is present. Vision is frozen, so
one shared encode can supply both DualBrain layers and merger tokens, then every
GRPO generate / old-logprob / ref-logprob / policy epoch reuses the merger output.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterator, List, Optional, Sequence

import torch
import torch.nn.functional as F

from models.qwen35 import unwrap_qwen_core


def _unwrap_model(m):
    return m.module if hasattr(m, "module") else m


def topk_spatial_points(
    hmap: torch.Tensor,
    k: int = 5,
    nms_radius: int = 2,
) -> List[List[int]]:
    """Local-max NMS on a patch heatmap → K points in the Qwen 0-1000 system (x, y)."""
    if hmap.ndim == 3:
        hmap = hmap.squeeze(0)
    x = hmap.detach().float()
    ht, wt = int(x.shape[0]), int(x.shape[1])
    k = max(int(k), 1)
    radius = max(int(nms_radius), 0)
    if ht <= 0 or wt <= 0:
        return [[500, 500] for _ in range(k)]

    if radius <= 0:
        peak_mask = torch.ones((ht, wt), dtype=torch.bool, device=x.device)
    else:
        win = 2 * radius + 1
        pooled = F.max_pool2d(x.unsqueeze(0).unsqueeze(0), kernel_size=win, stride=1, padding=radius)
        if pooled.shape[-2:] != (ht, wt):
            pooled = F.interpolate(pooled, size=(ht, wt), mode="nearest")
        peak_mask = x >= (pooled[0, 0] - 1e-6)

    ys, xs = torch.nonzero(peak_mask, as_tuple=True)
    if int(ys.numel()) == 0:
        ys, xs = torch.nonzero(torch.ones((ht, wt), dtype=torch.bool, device=x.device), as_tuple=True)
    scores = x[ys, xs]
    order = torch.argsort(scores, descending=True)

    def _to_1000(px: int, py: int) -> List[int]:
        qx = int(round((float(px) + 0.5) / float(wt) * 1000.0))
        qy = int(round((float(py) + 0.5) / float(ht) * 1000.0))
        return [max(0, min(1000, qx)), max(0, min(1000, qy))]

    picked: List[List[int]] = []
    used: List[tuple] = []
    for idx in order.tolist():
        py, px = int(ys[idx]), int(xs[idx])
        if radius > 0 and any(max(abs(px - ux), abs(py - uy)) <= radius for ux, uy in used):
            continue
        used.append((px, py))
        picked.append(_to_1000(px, py))
        if len(picked) >= k:
            break
    if len(picked) < k:
        for idx in order.tolist():
            py, px = int(ys[idx]), int(xs[idx])
            pt = _to_1000(px, py)
            if pt in picked:
                continue
            picked.append(pt)
            if len(picked) >= k:
                break
    while len(picked) < k:
        picked.append(picked[-1] if picked else [500, 500])
    return picked[:k]


def format_prior_hint(points: Sequence[Sequence[int]]) -> str:
    rows = ",\n  ".join(f"[{int(p[0])}, {int(p[1])}]" for p in points)
    return (
        "<prior_hint>\n"
        "high_response_points_2d=[\n"
        f"  {rows}\n"
        "]\n"
        "</prior_hint>"
    )


def _qwen_vl(model):
    m = _unwrap_model(model)
    m = unwrap_qwen_core(m)
    return m


def bind_cached_image_features(model, image_embeds: Optional[torch.Tensor]):
    """Replace `get_image_features` with a frozen merger cache (no visual.forward)."""
    if image_embeds is None:
        return nullcontext()
    return _bind_cached_image_features(model, image_embeds)


@contextmanager
def _bind_cached_image_features(model, image_embeds: torch.Tensor) -> Iterator[None]:
    qwen = _qwen_vl(model)
    inner = getattr(qwen, "model", qwen)
    if not hasattr(inner, "get_image_features"):
        yield
        return
    orig = inner.get_image_features
    visual = getattr(inner, "visual", None)
    merge = int(getattr(visual, "spatial_merge_size", 2) or 2) if visual is not None else 2
    cache = image_embeds.detach()

    def _cached(pixel_values, image_grid_thw=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        if image_grid_thw is None:
            feats = cache
            parts = (feats,)
        else:
            split_sizes = (image_grid_thw.prod(-1) // (merge * merge)).tolist()
            feats = cache.to(device=image_grid_thw.device)
            if pixel_values is not None:
                feats = feats.to(dtype=pixel_values.dtype)
            n = int(sum(int(s) for s in split_sizes))
            if int(feats.shape[0]) != n:
                raise RuntimeError(
                    f"cached image tokens {int(feats.shape[0])} != grid tokens {n} "
                    f"(grid={image_grid_thw.tolist()} merge={merge})"
                )
            parts = torch.split(feats, [int(s) for s in split_sizes])
        return BaseModelOutputWithPooling(pooler_output=parts, last_hidden_state=None)

    inner.get_image_features = _cached
    try:
        yield
    finally:
        inner.get_image_features = orig
