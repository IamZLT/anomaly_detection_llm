"""Scan VisA / MVTec folders into prior-CoT sample dicts (image + mask bbox + ref dir)."""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def extract_bbox_from_mask(mask_path: str) -> Optional[List[int]]:
    if mask_path is None or not os.path.exists(mask_path):
        return None
    try:
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask) > 0
        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)
        if not (rows.any() and cols.any()):
            return None
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        y1, y2 = y_indices[0], y_indices[-1]
        x1, x2 = x_indices[0], x_indices[-1]
        return [int(x1), int(y1), int(x2) + 1, int(y2) + 1]
    except Exception:
        return None


def _optional_int(v) -> Optional[int]:
    if v in (None, "null", "None", ""):
        return None
    return int(v)


def _is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in _IMG_EXT


def _list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(n for n in os.listdir(folder) if _is_image(n))


def _mask_for_stem(mask_dir: str, stem: str) -> Optional[str]:
    if not mask_dir or not os.path.isdir(mask_dir):
        return None
    for suf in (".png", ".jpg", ".jpeg", "_mask.png"):
        p = os.path.join(mask_dir, stem + suf)
        if os.path.isfile(p):
            return p
        p2 = os.path.join(mask_dir, f"{stem}_mask.png")
        if os.path.isfile(p2):
            return p2
    return None


def scan_visa(root: str, *, max_normal_per_class: Optional[int] = None, seed: int = 42) -> List[dict]:
    """
    VisA 全量：{root}/{cls}/Data/Images/{Normal,Anomaly}，Masks/Anomaly/{stem}.png。
    默认使用全部异常 + 全部正常 query；max_normal_per_class 仅在需要子采样时设置。
    """
    rng = random.Random(seed)
    samples: List[dict] = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"VisA root not found: {root}")
    classes = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "Data", "Images"))
    )
    for cls in classes:
        img_anom = os.path.join(root, cls, "Data", "Images", "Anomaly")
        img_norm = os.path.join(root, cls, "Data", "Images", "Normal")
        mask_dir = os.path.join(root, cls, "Data", "Masks", "Anomaly")
        for name in _list_images(img_anom):
            stem = os.path.splitext(name)[0]
            mask = _mask_for_stem(mask_dir, stem)
            bbox = extract_bbox_from_mask(mask) if mask else None
            samples.append(
                {
                    "id": f"visa_{cls}_anom_{stem}",
                    "full_img_path": os.path.join(img_anom, name),
                    "image": os.path.join(img_anom, name),
                    "metadata": {
                        "class": cls,
                        "anomaly": True,
                        "defect_type": "anomaly",
                        "bbox": bbox,
                        "full_mask_path": mask,
                        "layout": "visa",
                        "ref_dir": img_norm,
                    },
                }
            )
        normals = _list_images(img_norm)
        if max_normal_per_class is not None and len(normals) > int(max_normal_per_class):
            normals = rng.sample(normals, int(max_normal_per_class))
        for name in normals:
            stem = os.path.splitext(name)[0]
            samples.append(
                {
                    "id": f"visa_{cls}_good_{stem}",
                    "full_img_path": os.path.join(img_norm, name),
                    "image": os.path.join(img_norm, name),
                    "metadata": {
                        "class": cls,
                        "anomaly": False,
                        "defect_type": "good",
                        "bbox": None,
                        "full_mask_path": None,
                        "layout": "visa",
                        "ref_dir": img_norm,
                    },
                }
            )
    return samples


def scan_mvtec(root: str, split: str = "test") -> List[dict]:
    """MVTec-AD: {root}/{cls}/{train|test}/{defect}/xxx.png + ground_truth/{defect}/{stem}_mask.png"""
    samples: List[dict] = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"MVTec root not found: {root}")
    classes = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "test"))
    )
    for cls in classes:
        ref_dir = os.path.join(root, cls, "train", "good")
        split_dir = os.path.join(root, cls, split)
        if not os.path.isdir(split_dir):
            continue
        for defect in sorted(os.listdir(split_dir)):
            img_dir = os.path.join(split_dir, defect)
            if not os.path.isdir(img_dir):
                continue
            is_anom = defect.lower() not in ("good", "ok", "normal")
            gt_dir = os.path.join(root, cls, "ground_truth", defect)
            for name in _list_images(img_dir):
                stem = os.path.splitext(name)[0]
                mask = _mask_for_stem(gt_dir, stem) if is_anom else None
                bbox = extract_bbox_from_mask(mask) if mask else None
                samples.append(
                    {
                        "id": f"mvtec_{cls}_{defect}_{stem}",
                        "full_img_path": os.path.join(img_dir, name),
                        "image": os.path.join(img_dir, name),
                        "metadata": {
                            "class": cls,
                            "anomaly": is_anom,
                            "defect_type": defect,
                            "bbox": bbox,
                            "full_mask_path": mask,
                            "layout": "mvtec",
                            "ref_dir": ref_dir,
                        },
                    }
                )
    return samples


def load_prior_split(cfg: dict) -> tuple[List[dict], List[dict]]:
    """Train / eval from data.train_layout and data.eval_layout (visa | mvtec)."""
    data_cfg = cfg.get("data") or {}
    paths = cfg.get("paths") or {}
    dataset_root = str(paths.get("dataset_root", ""))
    seed = int(cfg.get("training", {}).get("seed", 42))

    def _scan(layout: str, root: Optional[str]) -> List[dict]:
        layout = (layout or "").lower()
        if layout == "json":
            raise ValueError("旧 JSON 扫盘已移除，请用 data.train_layout=visa / eval_layout=mvtec")
        root = os.path.expanduser(str(root or ""))
        if not root:
            raise ValueError(f"need data.*_root for layout={layout}")
        if layout == "visa":
            max_n = _optional_int(data_cfg.get("visa_max_normal_per_class"))
            return scan_visa(root, max_normal_per_class=max_n, seed=seed)
        if layout == "mvtec":
            split = str(data_cfg.get("mvtec_split", "test"))
            return scan_mvtec(root, split=split)
        raise ValueError(f"unknown layout {layout}")

    train_layout = str(data_cfg.get("train_layout", "visa")).lower()
    eval_layout = str(data_cfg.get("eval_layout", "mvtec")).lower()
    train_root = data_cfg.get("train_root") or (
        os.path.join(dataset_root, "VisA") if train_layout == "visa" else None
    )
    eval_root = data_cfg.get("eval_root") or (
        os.path.join(dataset_root, "mvtec_anomaly_detection") if eval_layout == "mvtec" else None
    )

    train = _scan(train_layout, train_root)
    evals = _scan(eval_layout, eval_root)

    max_train = _optional_int(data_cfg.get("max_samples"))
    if max_train is not None:
        train = train[: max_train]
    train = balance_normal_anomaly(train, data_cfg, seed=seed)
    max_eval = _optional_int(data_cfg.get("max_eval_samples"))
    if max_eval is not None:
        rng = random.Random(seed)
        evals = list(evals)
        rng.shuffle(evals)
        evals = evals[: max_eval]
    return train, evals


def _is_anomaly_sample(sample: dict) -> bool:
    return bool((sample.get("metadata") or {}).get("anomaly", False))


def _sample_class(sample: dict) -> str:
    return str((sample.get("metadata") or {}).get("class") or "_")


def balance_normal_anomaly(samples: List[dict], data_cfg: dict, seed: int = 42) -> List[dict]:
    """Downsample normals so train ≈ (normal_anomaly_ratio) normals per anomaly, per class.

    normal_anomaly_ratio=1.0 → 每类 1 正常 : 1 异常（保留全部异常，下采样正常）。
    null → 不限制，用扫盘全量。
    """
    ratio = data_cfg.get("normal_anomaly_ratio")
    if ratio in (None, "null", "None", ""):
        return samples
    ratio = float(ratio)
    if ratio < 0:
        raise ValueError(f"data.normal_anomaly_ratio must be >= 0, got {ratio}")

    from collections import defaultdict

    by_cls_anom: Dict[str, List[dict]] = defaultdict(list)
    by_cls_norm: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        cls = _sample_class(s)
        if _is_anomaly_sample(s):
            by_cls_anom[cls].append(s)
        else:
            by_cls_norm[cls].append(s)

    rng = random.Random(seed)
    out: List[dict] = []
    classes = sorted(set(by_cls_anom) | set(by_cls_norm))
    for cls in classes:
        anoms = list(by_cls_anom.get(cls) or [])
        norms = list(by_cls_norm.get(cls) or [])
        rng.shuffle(norms)
        n_keep = min(len(norms), int(round(len(anoms) * ratio)))
        out.extend(anoms)
        out.extend(norms[:n_keep])
    rng.shuffle(out)
    return out


def split_holdout_by_class(samples: List[dict], ratio: float, seed: int = 42) -> tuple[List[dict], List[dict]]:
    """Per-class VisA holdout so mid-training eval does not use MVTec test."""
    from collections import defaultdict

    ratio = float(ratio or 0.0)
    if ratio <= 0 or not samples:
        return list(samples), []
    by_cls: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        by_cls[_sample_class(s)].append(s)
    rng = random.Random(seed)
    train: List[dict] = []
    dev: List[dict] = []
    for cls in sorted(by_cls):
        items = list(by_cls[cls])
        rng.shuffle(items)
        n_hold = int(round(len(items) * ratio))
        if len(items) <= 1:
            n_hold = 0
        else:
            n_hold = min(max(n_hold, 0), len(items) - 1)
        dev.extend(items[:n_hold])
        train.extend(items[n_hold:])
    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev
