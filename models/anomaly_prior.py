"""
Training-free reference-conditioned anomaly spatial prior.

Aligns with CloudEdge-DualBrain (edge/methods/encoders.py + patch_gallery_ad.py):
  Qwen vision blocks keep **pre-merger** tokens
  unpack merge-packed order → spatial row-major
  per-layer L2 + NN vs normal patches
  softmax-weighted fusion of **distance maps** (not channel-softmax on features)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


from models.qwen35 import unwrap_qwen_core
from models.vision_cache import topk_spatial_points


def get_qwen_visual(qwen: nn.Module) -> nn.Module:
    core = unwrap_qwen_core(qwen)
    visual = getattr(getattr(core, "model", None), "visual", None)
    if visual is None:
        visual = getattr(core, "visual", None)
    if visual is None:
        raise RuntimeError("Cannot find Qwen visual encoder (model.visual).")
    return visual


def to_block_indices(layer_indices: Sequence[int], index_base: int, depth: int) -> List[int]:
    """Paper uses 1-based layer ids (12,16,20,24). Blocks are 0-based."""
    base = int(index_base)
    out = []
    for x in layer_indices:
        idx = int(x) - base
        if idx < 0 or idx >= depth:
            raise ValueError(f"vision layer {x} (base={base}) → block {idx} out of range 0..{depth - 1}")
        out.append(idx)
    return out


def unpack_merge_order(tokens: torch.Tensor, h: int, w: int, merge: int) -> torch.Tensor:
    """Qwen merge-packed [H*W, D] → spatial row-major [H*W, D].

    Packing matches pos_embed: (h/m, w/m, m, m) groups. DualBrain permutes to
    (h/m, m, w/m, m) so neighboring pixels are adjacent on the heatmap.
    """
    m = int(merge)
    if m <= 1 or tokens.ndim != 2:
        return tokens
    if h % m != 0 or w % m != 0 or int(tokens.shape[0]) != h * w:
        return tokens
    d = int(tokens.shape[-1])
    x = tokens.reshape(h // m, w // m, m, m, d)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.reshape(h * w, d)


def softmax_fuse_maps(maps: torch.Tensor, temperature: float) -> torch.Tensor:
    """maps [L,H,W] → fused [H,W]. Larger per-layer distance → larger weight."""
    tau = max(float(temperature), 1e-6)
    x = maps / tau
    x = x - x.amax(dim=0, keepdim=True)
    w = torch.exp(x)
    w = w / (w.sum(dim=0, keepdim=True) + 1e-8)
    return (w * maps).sum(dim=0)


def jet_colormap(values: np.ndarray) -> np.ndarray:
    """values in [0,1] → uint8 RGB, matplotlib-like jet."""
    x = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).round().astype(np.uint8)


def heatmap_to_pil(heatmap: torch.Tensor, size: Tuple[int, int]) -> Image.Image:
    """
    heatmap: [H, W] or [1,H,W]
    size: (W, H) PIL size
    """
    hmap = heatmap.detach().float().cpu()
    if hmap.ndim == 3:
        hmap = hmap.squeeze(0)
    arr = hmap.numpy()
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = (arr - lo) / (hi - lo)
    rgb = jet_colormap(norm)
    img = Image.fromarray(rgb, mode="RGB")
    if img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return img


def overlay_heatmap_on_image(image: Image.Image, heatmap: torch.Tensor, alpha: float = 0.45) -> Image.Image:
    heat = heatmap_to_pil(heatmap, image.size)
    return Image.blend(image.convert("RGB"), heat, alpha=float(np.clip(alpha, 0.0, 1.0)))


def prior_box_from_heatmap(
    anomaly: torch.Tensor,
    thresh_frac: float = 0.5,
    min_area_frac: float = 0.0,
) -> Optional[List[float]]:
    """Fused anomaly patch map [H,W] → largest connected component bbox in 0-1000.

    The fused map `anomaly` is a per-patch distance map (higher = more anomalous).
    We threshold it at a fraction of its own range, keep the largest connected
    component, and return its bounding box mapped into the Qwen 0-1000 system as
    [x1, y1, x2, y2]. Returns None when the map is flat or the component is too small.

    This box is a *coarse hint* used for candidate anchoring (R_ground) only; it is
    never treated as a defect label.
    """
    from scipy import ndimage as ndi

    if anomaly is None:
        return None
    if anomaly.ndim == 3:
        anomaly = anomaly.squeeze(0)
    x = anomaly.detach().float()
    ht, wt = int(x.shape[0]), int(x.shape[1])
    if ht <= 0 or wt <= 0:
        return None
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return None
    thresh = lo + (hi - lo) * float(max(0.0, min(1.0, thresh_frac)))
    mask = (x >= thresh).cpu().numpy()
    labeled, n = ndi.label(mask)
    if n <= 0:
        return None
    sizes = ndi.sum(mask, labeled, range(1, n + 1))
    best = int(sizes.argmax()) + 1
    ys, xs = np.nonzero(labeled == best)
    if int(xs.size) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    frac = float((x1 - x0 + 1) * (y1 - y0 + 1)) / float(ht * wt)
    if frac < float(max(0.0, min_area_frac)):
        return None

    qx0 = int(round((x0 + 0.5) / float(wt) * 1000.0))
    qy0 = int(round((y0 + 0.5) / float(ht) * 1000.0))
    qx1 = int(round((x1 + 0.5) / float(wt) * 1000.0))
    qy1 = int(round((y1 + 0.5) / float(ht) * 1000.0))
    qx0, qx1 = min(qx0, qx1), max(qx0, qx1)
    qy0, qy1 = min(qy0, qy1), max(qy0, qy1)
    qx0 = max(0, min(1000, qx0))
    qy0 = max(0, min(1000, qy0))
    qx1 = max(0, min(1000, qx1))
    qy1 = max(0, min(1000, qy1))
    if qx1 <= qx0 or qy1 <= qy0:
        return None
    return [float(qx0), float(qy0), float(qx1), float(qy1)]


class AnomalyPrior(nn.Module):
    """Frozen vision encoder → H from layers 12/16/20/24, plus merger tokens V_r, V_t."""

    def __init__(self, visual: nn.Module, cfg: dict):
        super().__init__()
        self.visual = visual
        prior_cfg = cfg.get("prior", {}) or {}
        depth = int(getattr(visual.config, "depth", len(visual.blocks)))
        index_base = int(prior_cfg.get("layer_index_base", 1))
        layers = prior_cfg.get("layer_indices") or [12, 16, 20, 24]
        self.block_indices = to_block_indices(layers, index_base, depth)
        self.temperature = float(prior_cfg.get("softmax_temperature", 0.5))
        self.neighborhood_radius = int(prior_cfg.get("neighborhood_radius", 0))
        self.spatial_merge_size = int(
            getattr(visual, "spatial_merge_size", getattr(visual.config, "spatial_merge_size", 2))
        )
        self.hint_topk = int(prior_cfg.get("hint_topk", 5))
        self.hint_nms_radius = int(prior_cfg.get("hint_nms_radius", 2))
        self.prior_box_thresh = float(prior_cfg.get("prior_box_thresh", 0.5))
        self.prior_box_min_area = float(prior_cfg.get("prior_box_min_area", 0.0))

        for p in self.visual.parameters():
            p.requires_grad = False
        self.visual.eval()

    @classmethod
    def from_qwen(cls, qwen: nn.Module, cfg: dict) -> "AnomalyPrior":
        return cls(get_qwen_visual(qwen), cfg)

    def _encode_one(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], Tuple[int, int], torch.Tensor]:
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                get_vision_bilinear_indices_and_weights,
                get_vision_cu_seqlens,
                get_vision_position_ids,
            )
        except ImportError as e:
            raise ImportError(
                "AnomalyPrior needs transformers Qwen3.5 vision helpers "
                "(pin requirements.txt: transformers>=5.14,<5.15; tested 5.14.1). "
                f"Original error: {e}"
            ) from e

        visual = self.visual
        pv = pixel_values.type(visual.dtype)
        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            image_grid_thw,
            num_grid_per_side=visual.num_grid_per_side,
            spatial_merge_size=visual.config.spatial_merge_size,
        )
        position_ids = get_vision_position_ids(image_grid_thw, visual.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(image_grid_thw)

        hs = visual.patch_embed(pv)
        pos_embeds = (visual.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]).sum(0)
        hs = hs + pos_embeds.to(hs.dtype)
        rotary_pos_emb = visual.rotary_pos_emb(position_ids)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        want = set(self.block_indices)
        collected: Dict[int, torch.Tensor] = {}
        for i, blk in enumerate(visual.blocks):
            hs = blk(hs, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            if i in want:
                collected[i] = hs

        missing = [i for i in self.block_indices if i not in collected]
        if missing:
            raise RuntimeError(f"vision blocks missing {missing}; valid 0..{len(visual.blocks) - 1}")

        merged = visual.merger(hs)
        t = int(image_grid_thw[0, 0])
        th, tw = int(image_grid_thw[0, 1]), int(image_grid_thw[0, 2])
        m = self.spatial_merge_size
        feats: List[torch.Tensor] = []
        for i in self.block_indices:
            tok = collected[i]
            if t == 1 and tok.shape[0] == th * tw and th % m == 0 and tw % m == 0:
                tok = unpack_merge_order(tok, th, tw, m)
            feats.append(tok)
        return feats, (th, tw), merged

    def _nn_map(
        self,
        f_t: torch.Tensor,
        f_r: torch.Tensor,
        hw_t: Tuple[int, int],
        hw_r: Tuple[int, int],
        radius: int,
    ) -> torch.Tensor:
        ft = F.normalize(f_t.float(), dim=-1)
        fr = F.normalize(f_r.float(), dim=-1)
        ht, wt = int(hw_t[0]), int(hw_t[1])
        if ft.shape[0] != ht * wt:
            side = int(round(ft.shape[0] ** 0.5))
            ht, wt = side, side

        if int(radius) <= 0:
            sim = ft @ fr.T
            dist = (1.0 - sim.max(dim=1).values.clamp(-1.0, 1.0)).view(ht, wt)
            return dist

        hr, wr = int(hw_r[0]), int(hw_r[1])
        c = int(fr.shape[-1])
        if fr.shape[0] != hr * wr:
            side = int(round(fr.shape[0] ** 0.5))
            hr, wr = side, side
        if (hr, wr) != (ht, wt):
            fr_map = fr.view(1, hr, wr, c).permute(0, 3, 1, 2)
            fr_map = F.interpolate(fr_map, size=(ht, wt), mode="bilinear", align_corners=False)
            fr = F.normalize(fr_map.permute(0, 2, 3, 1).reshape(-1, c), dim=-1)
            hr, wr = ht, wt

        win = 2 * int(radius) + 1
        fr_map = fr.view(1, hr, wr, c).permute(0, 3, 1, 2)
        padded = F.pad(fr_map, (radius, radius, radius, radius), mode="replicate")
        patches = F.unfold(padded, kernel_size=win)
        k = win * win
        patches = patches.view(c, k, ht * wt).permute(2, 1, 0)
        sim = (ft.unsqueeze(1) * patches).sum(dim=-1)
        return (1.0 - sim.max(dim=1).values.clamp(-1.0, 1.0)).view(ht, wt)

    @torch.no_grad()
    def forward(
        self,
        ref_pixel_values: torch.Tensor,
        ref_grid_thw: torch.Tensor,
        test_pixel_values: torch.Tensor,
        test_grid_thw: torch.Tensor,
        upsample_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        self.visual.eval()
        f_r_layers, hw_r, v_r = self._encode_one(ref_pixel_values, ref_grid_thw)
        f_t_layers, hw_t, v_t = self._encode_one(test_pixel_values, test_grid_thw)
        maps = [
            self._nn_map(ft, fr, hw_t, hw_r, self.neighborhood_radius)
            for ft, fr in zip(f_t_layers, f_r_layers)
        ]
        stack = torch.stack(maps, dim=0)
        anomaly = softmax_fuse_maps(stack, self.temperature) if stack.shape[0] > 1 else stack[0]
        heatmap = anomaly.unsqueeze(0).unsqueeze(0)
        if upsample_size is not None:
            heatmap = F.interpolate(heatmap, size=upsample_size, mode="bilinear", align_corners=False)
        prior_points = topk_spatial_points(anomaly, k=self.hint_topk, nms_radius=self.hint_nms_radius)
        prior_box = prior_box_from_heatmap(
            anomaly,
            thresh_frac=self.prior_box_thresh,
            min_area_frac=self.prior_box_min_area,
        )
        return {
            "heatmap": heatmap.squeeze(0).squeeze(0),
            "patch_map": anomaly,
            "grid_hw": hw_t,
            "ref_grid_hw": hw_r,
            "ref_merged": v_r,
            "test_merged": v_t,
            "merged_embeddings": torch.cat([v_r, v_t], dim=0),
            "prior_points": prior_points,
            "prior_box": prior_box,
        }

    @torch.no_grad()
    def encode_pair(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        upsample_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """One frozen ViT pass per image → layers for H and merger tokens for the LLM."""
        if image_grid_thw.ndim == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)
        if int(image_grid_thw.shape[0]) != 2:
            raise ValueError(f"encode_pair expects 2 images, got grid {tuple(image_grid_thw.shape)}")
        counts = [int(x) for x in image_grid_thw.prod(dim=-1).tolist()]
        chunks = torch.split(pixel_values, counts, dim=0)
        if len(chunks) != 2:
            raise ValueError(f"pixel_values split into {len(chunks)} images, expected 2")
        return self.forward(
            chunks[0],
            image_grid_thw[0:1],
            chunks[1],
            image_grid_thw[1:2],
            upsample_size=upsample_size,
        )
