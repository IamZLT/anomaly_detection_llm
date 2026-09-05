"""Shared full-image + optional original-resolution ROI input construction."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.prior_dataset import PriorCollator, PriorCoTDataset, apply_chat_template_safe
from outcome.protocol import prompt, validate_gt
from models.anomaly_prior import softmax_fuse_maps, unpack_merge_order


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
    maps = []
    for index in prior.block_indices:
        ref, test = torch.split(captured[index], counts, dim=0)
        ref = unpack_merge_order(ref, *hw_r, prior.spatial_merge_size)
        test = unpack_merge_order(test, *hw_t, prior.spatial_merge_size)
        maps.append(prior._nn_map(test, ref, hw_t, hw_r, prior.neighborhood_radius))
    stack = torch.stack(maps)
    hmap = softmax_fuse_maps(stack, prior.temperature) if len(maps) > 1 else stack[0]
    return dict(patch_map=hmap, merged_embeddings=merged.detach())


def region_proposals(hmap, cfg):
    """Four-connected patch regions, bbox uses cell EDGES (including singleton).

    Relative threshold is explicitly uncalibrated. With raw_threshold configured,
    low or flat maps may produce zero regions; a uniformly high map may cover all.
    """
    arr = hmap.detach().float().cpu().numpy() if torch.is_tensor(hmap) else np.asarray(hmap, dtype=float)
    if arr.ndim != 2 or not arr.size or not np.isfinite(arr).all():
        raise ValueError('H must be a finite nonempty 2D patch map')
    h, w = arr.shape
    threshold = cfg.get('raw_threshold')
    mode = 'absolute_raw_uncalibrated' if threshold is not None else 'relative_uncalibrated'
    if threshold is None:
        if float(arr.max()-arr.min()) < 1e-8:
            return [], mode
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
    regions.sort(key=lambda p: (-p['raw_peak'], -p['raw_mean'], p['bbox_2d']))
    regions = regions[:max(0, int(cfg.get('max_candidates', 3)))]
    for i, region in enumerate(regions):
        region['id'] = f'h{i+1}'
    return regions, mode


def crop_original(image, bbox, margin=0.25):
    """Expand each SIDE by margin * box width/height. Return exact integer bounds."""
    w, h = image.size
    x0, y0, x1, y1 = bbox[0]*w/1000, bbox[1]*h/1000, bbox[2]*w/1000, bbox[3]*h/1000
    dx, dy = (x1-x0)*margin, (y1-y0)*margin
    bounds = [max(0, math.floor(x0-dx)), max(0, math.floor(y0-dy)),
              min(w, math.ceil(x1+dx)), min(h, math.ceil(y1+dy))]
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError('empty crop')
    return image.crop(tuple(bounds)), bounds


def local_box_to_full(local_box, bounds_px, orig_wh):
    x0,y0,x1,y1 = bounds_px
    w,h = orig_wh
    return [(x0+local_box[0]*(x1-x0)/1000)/w*1000,
            (y0+local_box[1]*(y1-y0)/1000)/h*1000,
            (x0+local_box[2]*(x1-x0)/1000)/w*1000,
            (y0+local_box[3]*(y1-y0)/1000)/h*1000]


class OutcomeDataset(PriorCoTDataset):
    def __getitem__(self, index):
        item = self._load_pair(self.samples[index])
        if Path(item['image_path']).resolve() == Path(item['ref_path']).resolve():
            raise ValueError('normal reference must not be the inspection image')
        validate_gt(item)
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
        proposals, threshold_mode = region_proposals(hmap, pcfg)
        if condition == 'none':
            proposals = []
        roi_cfg = self.cfg.get('outcome', {}).get('roi', {})
        images = [ref, test]
        caches = [vis['merged_embeddings'].detach()]
        grids = [initial['image_grid_thw']]
        roi_info = None
        if roi_cfg.get('enabled', False) and proposals:
            roi, bounds = crop_original(item['test'], proposals[0]['bbox_2d'], float(roi_cfg.get('margin', .25)))
            _, roi = self._align_pair(roi, roi)
            images.append(roi)
            enc = self.processor.image_processor(images=roi, return_tensors='pt')
            grid = enc['image_grid_thw'].reshape(-1, 3)
            grids.append(grid)
            roi_info = dict(image_index=3, source='original_inspection', candidate_id=proposals[0]['id'],
                            crop_bounds_px=bounds, original_wh=list(item['orig_size']),
                            crop_bounds_2d=local_box_to_full([0,0,1000,1000], bounds, item['orig_size']),
                            output_coordinates='full_inspection_0_1000', reference_registered=False)
        hint = dict(coordinates='full_inspection_0_1000', candidates=proposals,
                    score_meaning='raw feature discrepancy, not probability', threshold_mode=threshold_mode,
                    roi=roi_info)
        text = prompt(item['class_name'], bool(roi_cfg.get('enabled', False)))+'\n<prior_hint>'+json.dumps(hint, separators=(',', ':'))+'</prior_hint>'
        user = dict(role='user', content=[dict(type='image', image=im) for im in images]+[dict(type='text', text=text)])
        rendered = apply_chat_template_safe(self.processor, [user], True, False)
        full = self.processor(text=[rendered], images=images, return_tensors='pt', truncation=False)
        length = full['input_ids'].shape[-1]
        maximum = int(self.cfg['training']['max_length'])
        if length > maximum:
            raise ValueError(f'prompt has {length} tokens > {maximum}; increase max_length or reduce image budget; refusing silent truncation')
        expected_grid = torch.cat(grids).cpu()
        if not torch.equal(full['image_grid_thw'].reshape(-1, 3).cpu(), expected_grid):
            raise RuntimeError('processor geometry changed between vision cache and prompt construction')
        if roi_info is not None:
            # Canonical joint three-image encode. Separately concatenating ROI
            # merger features is not numerically equivalent on the real BF16
            # model. Pay this once per sample, then share across the whole group.
            with torch.no_grad():
                canonical = self.prior.visual(
                    full['pixel_values'].to(device=device, dtype=self.prior.visual.dtype),
                    grid_thw=full['image_grid_thw'].to(device)).pooler_output
            full['image_embeds'] = canonical.detach()
        else:
            full['image_embeds'] = torch.cat(caches)
        full['prompt_len'] = torch.tensor([length])
        full['_meta'] = [{key: item.get(key) for key in ('orig_size','gt_box_px','is_anomaly','image_path','ref_path','class_name','defect_type')}]
        full['_meta'][0].update(prior_candidates=proposals, roi=roi_info, prior_condition=condition,
                                prior_threshold_mode=threshold_mode, h_min=float(hmap.min()), h_max=float(hmap.max()),
                                image_count=len(images), prompt_tokens=int(length),
                                visual_tokens=int((full['image_grid_thw'].prod(-1)//(self.prior.spatial_merge_size**2)).sum()),
                                prior_hint_tokens=len(getattr(self.processor, 'tokenizer', self.processor).encode(
                                    json.dumps(hint, separators=(',', ':')), add_special_tokens=False)))
        return full
