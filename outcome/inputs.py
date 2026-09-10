"""Shared full-image + optional original-resolution ROI input construction."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.prior_dataset import PriorCollator, PriorCoTDataset, apply_chat_template_safe
from outcome.protocol import render_prompt, validate_gt
from models.anomaly_prior import heatmap_to_pil, softmax_fuse_maps, unpack_merge_order
from models.region_injection import REGION_TOKEN, region_token_id_of


@torch.no_grad()
def encode_pair_canonical(prior, pixels, grid):
    """Get H features and merger cache from ONE official joint vision forward.

    Hooks observe selected pre-merger block outputs without changing computation.
    This avoids model-/shape-dependent BF16 differences from manual per-image ViT.
    """
    if grid.shape != (2, 3) or not bool((grid[:, 0] == 1).all()):
        raise ValueError('canonical pair encoder expects two still images')
    captured = {}
    handles = []
    def hook(index):
        def capture(module, args, output):
            captured[index] = (output[0] if isinstance(output, tuple) else output).detach()
        return capture
    prior.visual.eval()
    try:
        for index in prior.block_indices:
            handles.append(prior.visual.blocks[index].register_forward_hook(hook(index)))
        merged = prior.visual(pixels.to(dtype=prior.visual.dtype), grid_thw=grid).pooler_output
    finally:
        for handle in handles:
            handle.remove()
    counts = [int(v) for v in grid.prod(-1)]
    hw_r, hw_t = tuple(int(v) for v in grid[0, 1:]), tuple(int(v) for v in grid[1, 1:])
    region_idx = int(getattr(prior, 'region_feature_index', prior.block_indices[-1]))
    match_fn = getattr(prior, '_nn_match', None)
    test_features = matched_ref_features = None
    maps = []
    for index in prior.block_indices:
        ref, test = torch.split(captured[index], counts, dim=0)
        ref = unpack_merge_order(ref, *hw_r, prior.spatial_merge_size)
        test = unpack_merge_order(test, *hw_t, prior.spatial_merge_size)
        if match_fn is not None:
            match = match_fn(test, ref, hw_t, hw_r, prior.neighborhood_radius)
            maps.append(match['distance'])
            if index == region_idx:
                test_features = test.detach()
                matched_ref_features = match['matched_ref_features'].detach()
        else:
            maps.append(prior._nn_map(test, ref, hw_t, hw_r, prior.neighborhood_radius))
    stack = torch.stack(maps)
    hmap = softmax_fuse_maps(stack, prior.temperature) if len(maps) > 1 else stack[0]
    return dict(patch_map=hmap, merged_embeddings=merged.detach(),
                test_features=test_features, matched_ref_features=matched_ref_features,
                test_grid_hw=hw_t)


def region_proposals(hmap, cfg):
    """Four-connected patch regions; bbox uses cell EDGES (including singleton).

    Returns ``(proposal_meta, proposal_masks, mode)``:
      - ``proposal_meta`` : list of dicts (id/bbox_2d/peak_2d/raw_peak/raw_mean/area_fraction)
      - ``proposal_masks`` : bool ndarray ``[R, Ht, Wt]`` (one mask per candidate region)
      - ``mode``          : threshold mode string for logging

    Masks are kept so region feature extraction does not treat every patch inside a
    region's bbox as defect — only the connected cells that formed the region.
    """
    arr = hmap.detach().float().cpu().numpy() if torch.is_tensor(hmap) else np.asarray(hmap, dtype=float)
    if arr.ndim != 2 or not arr.size or not np.isfinite(arr).all():
        raise ValueError('H must be a finite nonempty 2D patch map')
    h, w = arr.shape
    threshold = cfg.get('raw_threshold')
    mode = 'absolute_raw_uncalibrated' if threshold is not None else 'relative_uncalibrated'
    if threshold is None:
        if float(arr.max()-arr.min()) < 1e-8:
            return [], np.zeros((0, h, w), dtype=bool), mode
        fraction = float(cfg.get('relative_threshold', 0.7))
        if not 0 < fraction <= 1:
            raise ValueError('relative_threshold must be in (0,1]')
        threshold = float(arr.min()+fraction*(arr.max()-arr.min()))
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError('nonfinite H threshold')
    mask = arr >= threshold
    seen = np.zeros_like(mask)
    regions = []
    region_masks = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        queue = [(int(y), int(x))]
        seen[y, x] = True
        cells = []
        while queue:
            cy, cx = queue.pop()
            cells.append((cy, cx))
            for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx] = True
                    queue.append((ny,nx))
        if len(cells) < int(cfg.get('min_cells', 1)):
            continue
        yy, xx = np.array(cells).T
        scores = arr[yy,xx]
        peak_idx = int(scores.argmax())
        regions.append(dict(bbox_2d=[round(float(xx.min())/w*1000, 3), round(float(yy.min())/h*1000, 3),
                                    round(float(xx.max()+1)/w*1000, 3), round(float(yy.max()+1)/h*1000, 3)],
                            peak_2d=[round((float(xx[peak_idx])+.5)/w*1000, 3),
                                     round((float(yy[peak_idx])+.5)/h*1000, 3)],
                            raw_peak=round(float(scores.max()), 6), raw_mean=round(float(scores.mean()), 6),
                            area_fraction=round(len(cells)/(h*w), 6)))
        region_masks.append(_cells_to_mask(h, w, cells))
    order = sorted(range(len(regions)),
                   key=lambda i: (-regions[i]['raw_peak'], -regions[i]['raw_mean'], regions[i]['bbox_2d']))
    order = order[:max(0, int(cfg.get('max_candidates', 3)))]
    regions = [regions[i] for i in order]
    region_masks = [region_masks[i] for i in order]
    for i, region in enumerate(regions):
        region['id'] = f'h{i+1}'
    masks_arr = np.stack(region_masks, axis=0) if region_masks else np.zeros((0, h, w), dtype=bool)
    return regions, masks_arr, mode


def _cells_to_mask(h, w, cells):
    """Build a dense [H, W] bool mask from a list of (y, x) cell indices."""
    out = np.zeros((h, w), dtype=bool)
    if not cells:
        return out
    yy, xx = np.array(cells).T
    out[yy, xx] = True
    return out


def extract_region_cells(proposal_masks, test_features, matched_ref_features, hmap, cfg):
    """Turn candidate region masks into per-cell contrast features for the region adapter.

    Each region is split into (up to) 2x2 spatial sub-cells along its bbox; each
    non-empty sub-cell contributes one token. Within a sub-cell, test and matched
    reference features are aggregated with the SAME H-strength weights, so the token
    expresses "local test content vs its normal counterpart".

    Returns dict(test=[1,n,D], ref=[1,n,D], geom=[1,n,G], hstat=[1,n,H], valid=[1,n]).
    Always returns at least one cell: a single invalid "empty" cell when there are no
    candidate regions (an explicit missing representation).
    """
    dev = test_features.device
    Ht, Wt = int(hmap.shape[0]), int(hmap.shape[1])
    D = int(test_features.shape[-1])
    test_f = test_features.float().reshape(Ht, Wt, D)
    ref_f = matched_ref_features.float().reshape(Ht, Wt, D)
    h = hmap.detach().float().reshape(Ht, Wt)
    R = int(proposal_masks.shape[0])
    max_cells = max(1, int(cfg.get('max_cells', 12)))
    pmask = torch.from_numpy(np.asarray(proposal_masks, dtype=bool)).to(dev)
    out_test, out_ref, out_geom, out_hstat = [], [], [], []
    for r in range(R):
        ys, xs = torch.nonzero(pmask[r], as_tuple=True)
        if ys.numel() == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        height, width = y1 - y0 + 1, x1 - x0 + 1
        y_edges = [y0, y0 + height // 2, y1 + 1] if height >= 2 else [y0, y1 + 1]
        x_edges = [x0, x0 + width // 2, x1 + 1] if width >= 2 else [x0, x1 + 1]
        for i in range(len(y_edges) - 1):
            for j in range(len(x_edges) - 1):
                sy0, sy1, sx0, sx1 = y_edges[i], y_edges[i + 1], x_edges[j], x_edges[j + 1]
                sub_mask = pmask[r, sy0:sy1, sx0:sx1].reshape(-1)
                if not sub_mask.any():
                    continue
                sub_test = test_f[sy0:sy1, sx0:sx1].reshape(-1, D)[sub_mask]
                sub_ref = ref_f[sy0:sy1, sx0:sx1].reshape(-1, D)[sub_mask]
                sub_h = h[sy0:sy1, sx0:sx1].reshape(-1)[sub_mask]
                w = sub_h - sub_h.min() + 1e-6
                w = w / w.sum()
                out_test.append((sub_test * w[:, None]).sum(0))
                out_ref.append((sub_ref * w[:, None]).sum(0))
                gcx = ((sx0 + sx1 - 1) / 2.0 + 0.5) / Wt
                gcy = ((sy0 + sy1 - 1) / 2.0 + 0.5) / Ht
                gw = (sx1 - sx0) / Wt
                gh = (sy1 - sy0) / Ht
                out_geom.append([gcx, gcy, gw, gh, r / max(R, 1)])
                out_hstat.append([float(sub_h.max()), float(sub_h.mean())])
    if not out_test:
        return dict(test=torch.zeros(1, 1, D, device=dev),
                    ref=torch.zeros(1, 1, D, device=dev),
                    geom=torch.zeros(1, 1, int(cfg.get('geometry_dim', 5)), device=dev),
                    hstat=torch.zeros(1, 1, int(cfg.get('hstat_dim', 2)), device=dev),
                    valid=torch.zeros(1, 1, dtype=torch.bool, device=dev))
    n = min(len(out_test), max_cells)
    return dict(test=torch.stack(out_test[:n]).unsqueeze(0),
                ref=torch.stack(out_ref[:n]).unsqueeze(0),
                geom=torch.tensor(out_geom[:n], device=dev, dtype=torch.float32).unsqueeze(0),
                hstat=torch.tensor(out_hstat[:n], device=dev, dtype=torch.float32).unsqueeze(0),
                valid=torch.ones(1, n, dtype=torch.bool, device=dev))


class OutcomeDataset(PriorCoTDataset):
    def __getitem__(self, index):
        item = self._load_pair(self.samples[index])
        if Path(item['image_path']).resolve() == Path(item['ref_path']).resolve():
            raise ValueError('normal reference must not be the inspection image')
        getattr(self, 'validate_gt_fn', validate_gt)(item)
        return item


class OutcomeCollator(PriorCollator):
    def __call__(self, batch):
        if len(batch) != 1:
            raise ValueError('OutcomeCollator expects one sample; group sampling happens downstream')
        item = batch[0]
        device = self._device()
        ref, test = self._align_pair(item['ref'], item['test'])
        initial = self._concat_image_tensors(ref, test)
        vis = encode_pair_canonical(self.prior, initial['pixel_values'].to(device), initial['image_grid_thw'].to(device))
        pcfg = self.cfg.get('outcome', {}).get('prior', {})
        hmap = vis['patch_map']
        condition = pcfg.get('condition', 'real')
        if condition == 'shuffled':
            # Fixed per sample and across all members of its group; no GT involved.
            key = f"{self.cfg['training']['seed']}:{item['image_path']}:{item['ref_path']}"
            seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
            perm = torch.randperm(hmap.numel(), generator=torch.Generator().manual_seed(seed))
            hmap = hmap.flatten()[perm.to(hmap.device)].reshape_as(hmap)
        if condition not in ('real', 'none', 'shuffled'):
            raise ValueError(f'unknown H condition: {condition}')
        proposals, proposal_masks, threshold_mode = region_proposals(hmap, pcfg)
        if condition == 'none':
            proposals = []
            proposal_masks = np.zeros((0, *proposal_masks.shape[1:]), dtype=bool)
        region_cfg = self.cfg.get('outcome', {}).get('region', {}) or {}
        images = [ref, test]
        caches = [vis['merged_embeddings'].detach()]
        grids = [initial['image_grid_thw']]
        if vis.get('test_features') is None or vis.get('matched_ref_features') is None:
            raise ValueError('region tokens require matched features; check prior.region_feature_index')
        region_raw = extract_region_cells(proposal_masks, vis['test_features'],
                                          vis['matched_ref_features'], hmap, region_cfg)
        region_token_id = region_token_id_of(self.processor)
        n_region = int(region_raw['valid'].shape[1])
        region_tokens = ' '.join([REGION_TOKEN] * n_region)
        text = render_prompt(self.cfg, item['class_name'], region_tokens=region_tokens)
        user = dict(role='user', content=[dict(type='image', image=im) for im in images]+[dict(type='text', text=text)])
        rendered = apply_chat_template_safe(self.processor, [user], True, False)
        full = self.processor(text=[rendered], images=images, return_tensors='pt', truncation=False)
        length = full['input_ids'].shape[-1]
        maximum = int(self.cfg['training']['max_length'])
        if length > maximum:
            raise ValueError(f'prompt has {length} tokens > {maximum}; increase max_length or reduce image budget; refusing silent truncation')
        n_region = int(region_raw['valid'].shape[1])
        actual = int((full['input_ids'][0] == region_token_id).sum())
        if actual != n_region:
            raise RuntimeError(f'region tokenization mismatch: expected {n_region} <|region|> tokens, got {actual}')
        expected_grid = torch.cat(grids).cpu()
        if not torch.equal(full['image_grid_thw'].reshape(-1, 3).cpu(), expected_grid):
            raise RuntimeError('processor geometry changed between vision cache and prompt construction')
        full['image_embeds'] = torch.cat(caches)
        full['prompt_len'] = torch.tensor([length])
        full['region_test'] = region_raw['test'].cpu()
        full['region_ref'] = region_raw['ref'].cpu()
        full['region_geom'] = region_raw['geom'].cpu()
        full['region_hstat'] = region_raw['hstat'].cpu()
        full['region_valid'] = region_raw['valid'].cpu()
        full['region_token_id'] = int(region_token_id)
        prior_hint_tokens = int(region_raw['valid'].shape[1])
        full['_meta'] = [{key: item.get(key) for key in ('orig_size','gt_box_px','is_anomaly','image_path','ref_path','class_name','defect_type','component_bboxes','num_components','mask_area_fraction','union_area_fraction','full_mask_path')}]
        full['_meta'][0].update(prior_candidates=proposals, prior_condition=condition,
                                prior_threshold_mode=threshold_mode, h_min=float(hmap.min()), h_max=float(hmap.max()),
                                image_count=len(images), prompt_tokens=int(length),
                                visual_tokens=int((full['image_grid_thw'].prod(-1)//(self.prior.spatial_merge_size**2)).sum()),
                                prior_hint_tokens=prior_hint_tokens)
        # Visualization payload (heatmap + original images + H peaks in 0-1000).
        full['_meta'][0].update(
            ref=item['ref'],
            test=item['test'],
            heatmap=heatmap_to_pil(hmap, item['test'].size),
            prior_points=[p['peak_2d'] for p in proposals],
        )
        return full
