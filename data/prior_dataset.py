"""VisA/MVTec samples: normal ref + test + prior spatial hint for GRPO."""

from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from models.qwen35 import qwen_vision_factor
from models.anomaly_prior import heatmap_to_pil, overlay_heatmap_on_image
from models.vision_cache import format_prior_hint
from utils.common import smart_resize


def list_normal_refs(
    cls: str,
    dataset_root: str,
    query_path: str,
    source: str = "mvtec_anomaly_detection",
    ref_dir: Optional[str] = None,
    layout: str = "mvtec",
) -> List[str]:
    query_abs = os.path.abspath(query_path)
    roots: List[str] = []
    if ref_dir:
        roots.append(ref_dir)
    if str(layout).lower() == "visa":
        roots.extend(
            [
                os.path.join(dataset_root, "VisA", cls, "Data", "Images", "Normal"),
                os.path.join(dataset_root, cls, "Data", "Images", "Normal"),
            ]
        )
    roots.extend(
        [
            os.path.join(dataset_root, source, cls, "train", "good"),
            os.path.join(dataset_root, cls, "train", "good"),
        ]
    )
    cands: List[str] = []
    seen = set()
    for d in roots:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            p = os.path.join(d, name)
            ap = os.path.abspath(p)
            if ap in seen or not os.path.isfile(p):
                continue
            seen.add(ap)
            cands.append(p)
    valid = [p for p in cands if os.path.abspath(p) != query_abs]
    return valid or cands


def pick_ref_image(
    cls: str,
    dataset_root: str,
    query_path: str,
    source: str = "mvtec_anomaly_detection",
    ref_dir: Optional[str] = None,
    layout: str = "mvtec",
    randomize: bool = False,
    query_size: Optional[tuple] = None,
    topk: int = 5,
    cands: Optional[List[str]] = None,
) -> str:
    if cands is None:
        cands = list_normal_refs(cls, dataset_root, query_path, source=source, ref_dir=ref_dir, layout=layout)
    else:
        query_abs = os.path.abspath(query_path)
        filtered = [p for p in cands if os.path.abspath(p) != query_abs]
        cands = filtered or list(cands)
    if not cands:
        raise FileNotFoundError(f"no normal reference for class={cls} query={query_path}")
    if not randomize or len(cands) == 1:
        return cands[0]
    topk = max(int(topk or 1), 1)
    if query_size is None or topk >= len(cands):
        return random.choice(cands)
    qw, qh = float(query_size[0]), float(max(query_size[1], 1))
    qasp = qw / qh
    qarea = max(qw * qh, 1.0)
    scored = []
    for p in cands:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            continue
        asp = float(w) / float(max(h, 1))
        area = max(float(w * h), 1.0)
        score = abs(math.log(max(asp, 1e-6) / max(qasp, 1e-6))) + abs(math.log(area / qarea))
        scored.append((score, p))
    scored.sort(key=lambda x: x[0])
    pool = [p for _, p in scored[:topk]] or cands
    return random.choice(pool)


def apply_chat_template_safe(processor, messages, add_generation_prompt: bool, enable_thinking: bool):
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return processor.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def build_user_prompt(cfg: dict, class_name: str) -> str:
    tmpl = str((cfg.get("prompt") or {}).get("user") or "").strip()
    if not tmpl:
        tmpl = (
            "Image 1 is a defect-free normal reference of a {class_name}. "
            "Image 2 is the inspection sample. "
            "The <prior_hint> lists high-response points from a normal-test feature discrepancy map. "
            "These points are only spatial hints and are not defect labels. "
            "Verify them by comparing Image 1 and Image 2. "
            "Return exactly five XML blocks in order: <compare>, <ground>, <verify>, <boundary>, <answer>. "
            "Do not copy instruction text. Always output all five blocks. "
            "In <compare> write 1-2 sentences of concrete visual differences. "
            "In <ground> write one sentence then candidate_bbox_2d=[x1,y1,x2,y2] or candidate_bbox_2d=null. "
            "In <verify> decide true defect vs normal variation. "
            "In <boundary>, if candidate exists write four lines left/right/top/bottom=inward|outward|keep; "
            "if candidate_bbox_2d=null write not_applicable. "
            "In <answer> output JSON "
            "{\"is_anomaly\": true, \"bbox_2d\": [x1,y1,x2,y2], "
            "\"description\": \"One concise sentence describing the defect and its location.\"} "
            "or {\"is_anomaly\": false, \"bbox_2d\": null, "
            "\"description\": \"One concise sentence stating that no clear defect is observed.\"}. "
            "Do not add a defect_type field. "
            "All bbox coordinates are integers in the 0-1000 system of Image 2."
        )
    return tmpl.replace("{class_name}", class_name)


def build_train_ref_pool(train_samples: List[dict]) -> Dict[str, List[str]]:
    """Normal images from VisA-train only; VisA-dev must not appear as I_r."""
    pool: Dict[str, List[str]] = {}
    seen: Dict[str, set] = {}
    for s in train_samples:
        meta = s.get("metadata") or {}
        if bool(meta.get("anomaly")):
            continue
        cls = str(meta.get("class") or "object")
        path = s.get("full_img_path") or s.get("image")
        if not path:
            continue
        ap = os.path.abspath(str(path))
        if ap in seen.setdefault(cls, set()):
            continue
        seen[cls].add(ap)
        pool.setdefault(cls, []).append(str(path))
    return pool


def _is_test_path(path: str) -> bool:
    parts = str(path).replace("\\", "/").split("/")
    return "test" in parts


class PriorCoTDataset(Dataset):
    def __init__(
        self,
        samples: List[dict],
        cfg: dict,
        processor,
        mode: str = "train",
        ref_pool: Optional[Dict[str, List[str]]] = None,
    ):
        self.cfg = cfg
        self.processor = processor
        self.mode = mode
        self.ref_pool = ref_pool
        self.max_length = int(cfg.get("training", {}).get("max_length", 2048))
        self.dataset_root = str(cfg.get("paths", {}).get("dataset_root", ""))
        self.source = str(cfg.get("data", {}).get("source_dirname", "mvtec_anomaly_detection"))
        self.random_train_ref = bool(cfg.get("data", {}).get("random_train_ref", True))
        self.ref_topk = int(cfg.get("data", {}).get("ref_topk", 5))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _load_pair(self, sample: dict) -> dict:
        img_path = sample.get("full_img_path") or sample.get("image")
        meta = sample.get("metadata") or {}
        cls = str(meta.get("class") or "object")
        defect = str(meta.get("defect_type") or ("defect" if meta.get("anomaly") else "good"))
        is_anom = bool(meta.get("anomaly", False))
        gt_box = meta.get("bbox")
        test = Image.open(str(img_path)).convert("RGB")
        orig_size = test.size
        pool_cands = None
        if self.ref_pool is not None:
            pool_cands = list(self.ref_pool.get(cls) or [])
        ref_path = pick_ref_image(
            cls,
            self.dataset_root,
            str(img_path),
            source=self.source,
            ref_dir=meta.get("ref_dir"),
            layout=str(meta.get("layout") or "mvtec"),
            randomize=(self.mode == "train" and self.random_train_ref),
            query_size=orig_size,
            topk=self.ref_topk,
            cands=pool_cands,
        )
        ref = Image.open(ref_path).convert("RGB")
        gt_px = None
        if gt_box is not None and len(gt_box) == 4:
            gt_px = [float(gt_box[0]), float(gt_box[1]), float(gt_box[2]), float(gt_box[3])]
        return {
            "ref": ref,
            "test": test,
            "orig_size": orig_size,
            "gt_box_px": gt_px,
            "is_anomaly": is_anom,
            "class_name": cls,
            "defect_type": defect,
            "image_path": str(img_path),
            "ref_path": ref_path,
            "id": sample.get("id"),
        }

    def __getitem__(self, idx: int) -> dict:
        item = self._load_pair(self.samples[idx])
        item["prompt"] = build_user_prompt(self.cfg, item["class_name"])
        return item


def filter_prior_samples(raw: List[dict], cfg: dict, split: str) -> List[dict]:
    data_cfg = cfg.get("data") or {}
    train_anomaly_only = bool(data_cfg.get("train_anomaly_only", False))
    holdout_ratio = float(data_cfg.get("holdout_ratio", 0.1))
    max_samples = data_cfg.get("max_samples")
    seed = int(cfg.get("training", {}).get("seed", 42))

    samples = []
    for s in raw:
        p = str(s.get("full_img_path") or s.get("image") or "")
        if not _is_test_path(p):
            continue
        meta = s.get("metadata") or {}
        if train_anomaly_only and split == "train" and not bool(meta.get("anomaly", False)):
            continue
        if not os.path.isfile(str(s.get("full_img_path") or "")):
            continue
        samples.append(s)

    rng = random.Random(seed)
    indexed = list(enumerate(samples))
    rng.shuffle(indexed)
    n_hold = int(round(len(indexed) * holdout_ratio))
    hold_ids = set(i for i, _ in indexed[:n_hold]) if n_hold > 0 else set()
    if split == "train":
        out = [s for i, s in enumerate(samples) if i not in hold_ids]
    else:
        out = [s for i, s in enumerate(samples) if i in hold_ids] or samples[: min(8, len(samples))]

    if max_samples not in (None, "null", "None", "") and split == "train":
        out = out[: int(max_samples)]
    eval_max = data_cfg.get("max_eval_samples")
    if eval_max not in (None, "null", "None", "") and split != "train":
        out = out[: int(eval_max)]
    return out


class PriorCollator:
    def __init__(self, processor, prior, cfg: dict):
        self.processor = processor
        self.prior = prior
        self.cfg = cfg
        self.enable_thinking = bool(cfg.get("prompt", {}).get("enable_thinking", False))
        self.render = str((cfg.get("prior") or {}).get("render", "colormap"))
        self.overlay_alpha = float((cfg.get("prior") or {}).get("overlay_alpha", 0.45))
        tok = getattr(processor, "tokenizer", processor)
        self.pad_token_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0)
        self.last_metas: List[dict] = []

    def _device(self) -> torch.device:
        try:
            return next(self.prior.visual.parameters()).device
        except Exception:
            return torch.device("cpu")

    def _align_pair(self, ref: Image.Image, test: Image.Image):
        visual = getattr(self.prior, "visual", None)
        factor = qwen_vision_factor(self.processor, visual)
        img_proc = getattr(self.processor, "image_processor", None)
        data = self.cfg.get("data") or {}
        max_size = int(data.get("max_image_size", 448))
        min_pixels = int(getattr(img_proc, "min_pixels", 256 * 256) or (256 * 256))
        max_pixels = int(getattr(img_proc, "max_pixels", None) or (max_size * max_size))
        test_rs, _, _ = smart_resize(test, max_size=max_size, factor=factor, min_pixels=int(min_pixels), max_pixels=int(max_pixels))
        ref_rs = ref.resize(test_rs.size, Image.Resampling.BICUBIC)
        return ref_rs, test_rs

    @staticmethod
    def _clip_text_tensors(enc: dict, max_length: int) -> dict:
        ids = enc.get("input_ids")
        if not torch.is_tensor(ids) or ids.shape[-1] <= max_length:
            return enc
        seq = int(ids.shape[-1])
        for k, v in list(enc.items()):
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[-1] == seq:
                enc[k] = v[..., :max_length]
        return enc

    def _concat_image_tensors(self, ref_im: Image.Image, test_im: Image.Image) -> dict:
        img_proc = getattr(self.processor, "image_processor", None)
        if img_proc is None:
            raise RuntimeError("processor.image_processor is required for shared vision encode")

        def _pixels(t: torch.Tensor) -> torch.Tensor:
            if t.ndim == 3:
                return t.reshape(-1, t.shape[-1])
            return t

        def _grid(t: torch.Tensor) -> torch.Tensor:
            if t.ndim == 1:
                t = t.unsqueeze(0)
            if t.ndim == 3:
                t = t.reshape(-1, int(t.shape[-1]))
            return t

        ref_enc = img_proc(images=ref_im, return_tensors="pt")
        test_enc = img_proc(images=test_im, return_tensors="pt")
        if "pixel_values" not in ref_enc or "image_grid_thw" not in ref_enc:
            raise KeyError(f"image_grid_thw missing; keys={list(ref_enc.keys())}")
        return {
            "pixel_values": torch.cat([_pixels(ref_enc["pixel_values"]), _pixels(test_enc["pixel_values"])], dim=0),
            "image_grid_thw": torch.cat([_grid(ref_enc["image_grid_thw"]), _grid(test_enc["image_grid_thw"])], dim=0),
        }

    def _encode_one(self, item: dict) -> dict:
        device = self._device()
        ref_rs, test_rs = self._align_pair(item["ref"], item["test"])
        vis_in = self._concat_image_tensors(ref_rs, test_rs)
        pv = vis_in["pixel_values"].to(device)
        grid = vis_in["image_grid_thw"].to(device)
        th, tw = test_rs.size[1], test_rs.size[0]
        vis = self.prior.encode_pair(pv, grid, upsample_size=(th, tw))
        hmap = vis["heatmap"]
        points = vis["prior_points"]
        merged = vis["merged_embeddings"].detach()
        if self.render == "overlay":
            heat = overlay_heatmap_on_image(test_rs, hmap, alpha=self.overlay_alpha)
        else:
            heat = heatmap_to_pil(hmap, test_rs.size)
        user_text = item["prompt"].rstrip() + "\n\n" + format_prior_hint(points)
        images = [ref_rs, test_rs]
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0]},
                {"type": "image", "image": images[1]},
                {"type": "text", "text": user_text},
            ],
        }
        messages = [user]
        text = apply_chat_template_safe(self.processor, messages, True, self.enable_thinking)
        max_length = int(self.cfg.get("training", {}).get("max_length", 2048))
        try:
            full = self.processor(
                text=[text],
                images=images,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
        except TypeError:
            full = self.processor(text=[text], images=images, return_tensors="pt")
        full = self._clip_text_tensors(full, max_length)
        proc_grid = full.get("image_grid_thw")
        if proc_grid is not None:
            g = proc_grid.detach()
            if g.ndim == 3:
                g = g.squeeze(0)
            merge = max(int(getattr(self.prior, "spatial_merge_size", 2) or 2), 1)
            n_tok = int((g.prod(dim=-1) // (merge * merge)).sum().item())
            if n_tok != int(merged.shape[0]):
                pv2 = full["pixel_values"]
                if pv2.ndim == 3:
                    pv2 = pv2.reshape(-1, pv2.shape[-1])
                vis = self.prior.encode_pair(pv2.to(device), g.to(device), upsample_size=(th, tw))
                merged = vis["merged_embeddings"].detach()
                hmap = vis["heatmap"]
                points = vis["prior_points"]
                if self.render == "overlay":
                    heat = overlay_heatmap_on_image(test_rs, hmap, alpha=self.overlay_alpha)
                else:
                    heat = heatmap_to_pil(hmap, test_rs.size)
        out = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.shape[0] == 1 else v for k, v in full.items()}
        prompt_ids = out["input_ids"]
        out["prompt_len"] = torch.tensor(int(prompt_ids.numel()), dtype=torch.long)
        out["image_embeds"] = merged.cpu()
        out["_meta"] = {
            "orig_size": item["orig_size"],
            "gt_box_px": item["gt_box_px"],
            "is_anomaly": item["is_anomaly"],
            "image_path": item["image_path"],
            "class_name": item["class_name"],
            "defect_type": item.get("defect_type"),
            "ref": ref_rs,
            "test": test_rs,
            "heatmap": heat,
            "hmap_tensor": hmap.detach().cpu(),
            "prior_points": points,
            "vision_size": test_rs.size,
        }
        return out

    def __call__(self, batch: List[dict]) -> dict:
        encs = [self._encode_one(x) for x in batch]
        keys = [k for k in encs[0].keys() if k not in ("_meta",) and torch.is_tensor(encs[0][k])]
        out: Dict[str, Any] = {}
        for k in keys:
            vals = [e[k] for e in encs]
            if k in ("input_ids", "attention_mask", "labels"):
                pad_val = -100 if k == "labels" else (0 if k == "attention_mask" else int(self.pad_token_id))
                out[k] = pad_sequence(vals, batch_first=True, padding_value=pad_val)
            elif k == "pixel_values":
                out[k] = torch.cat([v if v.ndim == 2 else v.reshape(-1, v.shape[-1]) for v in vals], dim=0)
            elif k == "image_embeds":
                out[k] = torch.cat([v if v.ndim == 2 else v.reshape(-1, v.shape[-1]) for v in vals], dim=0)
            elif k == "image_grid_thw":
                stacked = []
                for v in vals:
                    stacked.append(v if v.ndim == 2 else v.unsqueeze(0))
                out[k] = torch.cat(stacked, dim=0)
            elif vals[0].ndim == 0:
                out[k] = torch.stack(vals)
            else:
                try:
                    out[k] = torch.stack(vals)
                except Exception:
                    out[k] = torch.cat(vals, dim=0)
        out["_meta"] = [e["_meta"] for e in encs]
        self.last_metas = out["_meta"]
        return out
