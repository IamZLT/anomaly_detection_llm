"""VisA/MVTec samples: normal ref + test + prior heatmap prompt for GRPO."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.common import smart_resize


def pick_ref_image(
    cls: str,
    dataset_root: str,
    query_path: str,
    source: str = "mvtec_anomaly_detection",
    ref_dir: Optional[str] = None,
    layout: str = "mvtec",
) -> str:
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
    for d in roots:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                cands.append(os.path.join(d, name))
    for p in cands:
        if os.path.isfile(p) and os.path.abspath(p) != query_abs:
            return p
    if cands:
        return cands[0]
    raise FileNotFoundError(f"no normal reference for class={cls} query={query_path}")


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
            "Image 3 is a normal-test patch discrepancy prior H (not a ground-truth label). "
            "Write Grounded Comparative CoT with tags compare, ground, verify, boundary, answer. "
            "Put candidate_bbox=[x1,y1,x2,y2] inside <ground> and bbox=[x1,y1,x2,y2] inside <answer>. "
            "Boundary lines are left/right/top/bottom with keep, move left/right, or move up/down. "
            "Coordinates are Qwen 0-1000 on Image 2. If normal, set both boxes to null."
        )
    return tmpl.replace("{class_name}", class_name)


def _is_test_path(path: str) -> bool:
    parts = str(path).replace("\\", "/").split("/")
    return "test" in parts


class MVTecPriorCoTDataset(Dataset):
    def __init__(
        self,
        samples: List[dict],
        cfg: dict,
        processor,
        mode: str = "train",
    ):
        self.cfg = cfg
        self.processor = processor
        self.mode = mode
        self.max_length = int(cfg.get("training", {}).get("max_length", 2048))
        self.max_image_size = int(cfg.get("data", {}).get("max_image_size", 448))
        self.factor = int(cfg.get("data", {}).get("factor", 28))
        self.dataset_root = str(cfg.get("paths", {}).get("dataset_root", ""))
        self.source = str(cfg.get("data", {}).get("source_dirname", "mvtec_anomaly_detection"))
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
        ref_path = pick_ref_image(
            cls,
            self.dataset_root,
            str(img_path),
            source=self.source,
            ref_dir=meta.get("ref_dir"),
            layout=str(meta.get("layout") or "mvtec"),
        )
        test = Image.open(str(img_path)).convert("RGB")
        ref = Image.open(ref_path).convert("RGB")
        test_rs, orig_size, _ = smart_resize(test, self.max_image_size, self.factor)
        ref_rs, _, _ = smart_resize(ref, self.max_image_size, self.factor)
        if ref_rs.size != test_rs.size:
            ref_rs = ref_rs.resize(test_rs.size, Image.Resampling.LANCZOS)
        gt_px = None
        if gt_box is not None and len(gt_box) == 4:
            # bbox is on original pixels; scale to orig_size which is original, keep original for 0-1000
            gt_px = [float(gt_box[0]), float(gt_box[1]), float(gt_box[2]), float(gt_box[3])]
        return {
            "ref": ref_rs,
            "test": test_rs,
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

    def _encode_one(self, item: dict) -> dict:
        device = self._device()
        heat, hmap = self.prior.heatmap_image(
            self.processor,
            item["ref"],
            item["test"],
            device=device,
            render=self.render,
            overlay_alpha=self.overlay_alpha,
        )
        images = [item["ref"], item["test"], heat]
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0]},
                {"type": "image", "image": images[1]},
                {"type": "image", "image": images[2]},
                {"type": "text", "text": item["prompt"]},
            ],
        }
        messages = [user]
        text = apply_chat_template_safe(self.processor, messages, True, self.enable_thinking)
        prompt_text = text

        max_length = int(self.cfg.get("training", {}).get("max_length", 2048))

        def _proc(txt: str):
            try:
                return self.processor(
                    text=[txt],
                    images=images,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
            except TypeError:
                return self.processor(text=[txt], images=images, return_tensors="pt")

        full = _proc(text)
        prompt_enc = _proc(prompt_text)
        for enc in (full, prompt_enc):
            if "input_ids" in enc and enc["input_ids"].shape[-1] > max_length:
                enc["input_ids"] = enc["input_ids"][..., :max_length]
                if "attention_mask" in enc:
                    enc["attention_mask"] = enc["attention_mask"][..., :max_length]
        out = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.shape[0] == 1 else v for k, v in full.items()}
        prompt_ids = prompt_enc["input_ids"].squeeze(0)
        out["prompt_len"] = torch.tensor(int(prompt_ids.numel()), dtype=torch.long)
        # meta for GRPO / eval (not fed to model.forward)
        out["_meta"] = {
            "orig_size": item["orig_size"],
            "gt_box_px": item["gt_box_px"],
            "is_anomaly": item["is_anomaly"],
            "image_path": item["image_path"],
            "class_name": item["class_name"],
            "defect_type": item.get("defect_type"),
            "ref": item["ref"],
            "test": item["test"],
            "heatmap": heat,
            "hmap_tensor": hmap.detach().cpu(),
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
