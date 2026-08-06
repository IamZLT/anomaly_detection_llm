import os
import time
import random
import argparse
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from transformers import AutoImageProcessor
from loss.FocalLoss import FocalLoss
from loss.BinaryDiceLoss import BinaryDiceLoss
from models.visual_proto import (
    DinoClipVisualPrototypeModel,
    VisualPrototypes,
    normalize_feature,
    update_visual_prototypes,
)
from data.visual_proto_data import build_samples_from_specs
from utils.metrics import pixel_level_metrics
from utils.common import prepare_output_dir, smart_resize
from utils.config import apply_runtime_overrides, load_yaml_config


# =========================
# Utils
# =========================
def _gt_mask_overlay_on_rgb(img_rgb: Image.Image, mask_l: Image.Image, blend: float = 0.45) -> Image.Image:
    """在 RGB 图上用半透明红色标出 GT mask（便于看缺陷位置）。"""
    w, h = img_rgb.size
    m = np.array(mask_l.convert("L").resize((w, h), Image.NEAREST)) > 127
    arr = np.array(img_rgb.convert("RGB")).astype(np.float32)
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    b = float(blend)
    for c in range(3):
        arr[..., c] = np.where(m, (1.0 - b) * arr[..., c] + b * red[c], arr[..., c])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _eval_vis_caption_bar(total_w: int, caption: str, bar_h: int = 32) -> Image.Image:
    bar = Image.new("RGB", (max(total_w, 1), bar_h), (245, 245, 245))
    dr = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 15)
        except OSError:
            font = ImageFont.load_default()
    dr.text((6, 6), caption, fill=(20, 20, 20), font=font)
    return bar


def _hstack_panels(panels: List[Image.Image]) -> Image.Image:
    if not panels:
        raise ValueError("panels empty")
    w0, h0 = panels[0].size
    out = Image.new("RGB", (w0 * len(panels), h0))
    for i, p in enumerate(panels):
        if p.size != (w0, h0):
            p = p.resize((w0, h0), Image.Resampling.BILINEAR)
        out.paste(p, (i * w0, 0))
    return out


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Dataset
# =========================
class VisualPrototypeImageDataset(Dataset):
    """JSON 样本列表上的图像级 Dataset（任意 benchmark，不限 MVTec）。"""

    def __init__(
        self,
        samples: List[Dict],
        image_size: int,
    ):
        self.samples = samples
        self.image_size = image_size
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        img = Image.open(item["img_path"]).convert("RGB")
        img_tensor = self.image_transform(img)

        if item["mask_path"] is not None and os.path.exists(item["mask_path"]):
            mask_img = Image.open(item["mask_path"]).convert("L")
            mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_np = (np.array(mask_img) > 0).astype(np.float32)
        else:
            mask_np = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        return {
            "img": img_tensor,
            "mask": torch.from_numpy(mask_np).unsqueeze(0),
            "anomaly": torch.tensor(item["anomaly"], dtype=torch.long),
            "img_path": item["img_path"],
            "dataset_name": item.get("dataset_name", "unknown"),
            "cls_name": item["cls_name"],
            "defect_type": item["defect_type"],
        }


# =========================
# Loss helpers
# =========================
def compute_global_alignment_loss(mapped_cls: torch.Tensor, clip_global: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mapped_cls = normalize_feature(mapped_cls)
    clip_global = normalize_feature(clip_global)
    logits = mapped_cls @ clip_global.t()
    logits = logits / max(temperature, 1e-8)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_i2t + loss_t2i)
    return loss, {"loss_i2t": loss_i2t.detach(), "loss_t2i": loss_t2i.detach()}


def compute_dense_patch_alignment_loss(
    mapped_patches: torch.Tensor,
    dino_grid: Tuple[int, int],
    clip_patches: torch.Tensor,
    clip_grid: Tuple[int, int],
    gt_mask: torch.Tensor,
    patch_pos_weight: float,
) -> torch.Tensor:
    if clip_patches is None or clip_patches.numel() == 0:
        return mapped_patches.new_zeros(())

    b, pd, c = mapped_patches.shape
    gd_h, gd_w = dino_grid
    gc_h, gc_w = clip_grid
    if gd_h * gd_w != pd:
        raise ValueError(f"DINO patch count mismatch: {gd_h}x{gd_w} != {pd}")

    clip_map = clip_patches.view(b, gc_h, gc_w, c).permute(0, 3, 1, 2)  # [B, C, Hc, Wc]
    clip_map = F.interpolate(clip_map, size=(gd_h, gd_w), mode="bilinear", align_corners=False)
    clip_map = clip_map.permute(0, 2, 3, 1).reshape(b, pd, c)

    mapped_patches = normalize_feature(mapped_patches)
    clip_map = normalize_feature(clip_map)

    mask_small = F.interpolate(gt_mask.float(), size=(gd_h, gd_w), mode="nearest").reshape(b, pd)
    weights = 1.0 + patch_pos_weight * mask_small

    cos = (mapped_patches * clip_map).sum(dim=-1)
    loss = ((1.0 - cos) * weights).sum() / weights.sum().clamp_min(1e-6)
    return loss


def compute_patch_cls_loss(
    mapped_patches: torch.Tensor,
    dino_grid: Tuple[int, int],
    gt_mask: torch.Tensor,
    proto_normal: torch.Tensor,
    proto_abnormal: torch.Tensor,
    focal_loss_fn: nn.Module,
    dice_loss_fn: nn.Module,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, p, c = mapped_patches.shape
    gh, gw = dino_grid
    if gh * gw != p:
        raise ValueError(f"DINO grid mismatch: {gh}x{gw} != {p}")

    patch_map = normalize_feature(mapped_patches).view(b, gh, gw, c)
    proto_normal = normalize_feature(proto_normal.view(1, c)).view(c)
    proto_abnormal = normalize_feature(proto_abnormal.view(1, c)).view(c)

    logit_n = torch.einsum("bhwc,c->bhw", patch_map, proto_normal)
    logit_a = torch.einsum("bhwc,c->bhw", patch_map, proto_abnormal)
    patch_logits = torch.stack([logit_n, logit_a], dim=1) / max(temperature, 1e-8)  # [B, 2, H, W]

    gt_small = F.interpolate(gt_mask.float(), size=(gh, gw), mode="nearest")
    gt_small_long = gt_small.long()

    focal = focal_loss_fn(patch_logits, gt_small_long)
    abnormal_prob = F.softmax(patch_logits, dim=1)[:, 1:2]
    dice = dice_loss_fn(abnormal_prob, gt_small)
    return focal + dice, patch_logits, abnormal_prob


def compute_img_cls_loss(
    mapped_cls: torch.Tensor,
    anomaly_label: torch.Tensor,
    proto_normal: torch.Tensor,
    proto_abnormal: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mapped_cls = normalize_feature(mapped_cls)
    proto_normal = normalize_feature(proto_normal.view(1, -1))
    proto_abnormal = normalize_feature(proto_abnormal.view(1, -1))
    protos = torch.cat([proto_normal, proto_abnormal], dim=0)  # [2, C]
    logits = mapped_cls @ protos.t()
    logits = logits / max(temperature, 1e-8)
    loss = F.cross_entropy(logits, anomaly_label.long())
    return loss, logits


def compute_consistency_loss(img_logits: torch.Tensor, patch_prob_abnormal: torch.Tensor) -> torch.Tensor:
    img_abn = F.softmax(img_logits, dim=-1)[:, 1]
    patch_abn = patch_prob_abnormal.flatten(1).amax(dim=1)
    return F.mse_loss(img_abn, patch_abn.detach())


# =========================
# Validation
# =========================
@torch.no_grad()
def _save_step1_eval_vis_for_leaf(
    model: DinoClipVisualPrototypeModel,
    device: torch.device,
    prototypes: VisualPrototypes,
    val_samples: List[Dict],
    args: argparse.Namespace,
    epoch_idx: int,
    ds_name: str,
    cls_name: str,
) -> None:
    """每类验证指标完成后保存若干张热力图（与 epoch 目录结构一致）。"""
    from utils.visualization import (
        compute_step1_heat_up,
        heatmap_overlay,
        prepare_step1_image_tensor,
    )

    k = int(getattr(args, "eval_vis_num_samples", 0) or 0)
    if k <= 0 or not prototypes.ready():
        return
    pool = [
        s
        for s in val_samples
        if str(s.get("dataset_name", "")) == str(ds_name)
        and str(s.get("cls_name", "")) == str(cls_name)
        and s.get("img_path")
        and os.path.isfile(str(s.get("img_path")))
    ]
    if not pool:
        tqdm.write(f"[eval_vis] skip {ds_name}/{cls_name}: no local img_path in val set")
        return
    rng = random.Random(
        (int(args.seed) + int(epoch_idx)) * 1_000_003 + abs(hash((ds_name, cls_name))) % 1_000_000_007
    )
    picks = rng.sample(pool, k=min(k, len(pool)))
    vis_root = os.path.join(args.output_dir, f"epoch_{epoch_idx + 1:03d}", "eval_vis")
    ds_safe = str(ds_name).replace("/", "_")
    cls_safe = str(cls_name).replace("/", "_")
    vis_dir = os.path.join(vis_root, ds_safe, cls_safe)
    os.makedirs(vis_dir, exist_ok=True)
    alpha = float(getattr(args, "eval_heatmap_alpha", 0.45))
    cfg_vis = {
        "dino": {"image_size": int(args.image_size)},
        "step1": {"temperature": float(args.temperature)},
        "training": {"step1_visual_debug": bool(getattr(args, "step1_visual_debug", False))},
        "test": {"step1_visual_debug": bool(getattr(args, "step1_visual_debug", False))},
    }
    n_ok = 0
    for j, s in enumerate(picks):
        img_path = str(s["img_path"])
        try:
            img = Image.open(img_path).convert("RGB")
            img_rs, _, _ = smart_resize(
                img.copy(),
                max_size=int(getattr(args, "max_image_size", 512)),
                factor=int(getattr(args, "resize_factor", 28)),
            )
            img_t = prepare_step1_image_tensor(img_rs, int(args.image_size), device)
            heat_up, mode = compute_step1_heat_up(
                model=model,
                img_t=img_t,
                img_rs=img_rs,
                cfg=cfg_vis,
                prototypes=prototypes,
                device=device,
            )
            anomaly = int(s.get("anomaly", 0) or 0)
            defect = str(s.get("defect_type") or "unknown")
            tag = "ABNORMAL" if anomaly else "NORMAL"
            cap = (
                f"{tag} (y={anomaly}) | defect={defect} | heat={mode} | "
                f"左:原图 中:GT(红) 右:预测 | {os.path.basename(img_path)}"
            )
            mp = s.get("mask_path")
            if mp and os.path.isfile(str(mp)):
                mask_l = Image.open(str(mp)).convert("L")
            else:
                mask_l = Image.new("L", img_rs.size, 0)
            gt_vis = _gt_mask_overlay_on_rgb(img_rs, mask_l, blend=min(0.55, alpha + 0.1))
            pred_vis = heatmap_overlay(img_rs, heat_up, alpha=alpha)
            row = _hstack_panels([img_rs.convert("RGB"), gt_vis, pred_vis])
            cap_bar = _eval_vis_caption_bar(row.width, cap)
            vis = Image.new("RGB", (row.width, cap_bar.height + row.height))
            vis.paste(cap_bar, (0, 0))
            vis.paste(row, (0, cap_bar.height))
            stem = "abn" if anomaly else "ok"
            out_path = os.path.join(vis_dir, f"{j:02d}_{stem}_{mode}.png")
            vis.save(out_path)
            n_ok += 1
        except Exception as e:
            tqdm.write(f"[eval_vis][{ds_name}/{cls_name}] skip {img_path}: {e}")
    tqdm.write(f"[eval_vis] {ds_name}/{cls_name}: saved {n_ok}/{len(picks)} -> {vis_dir}")


@torch.no_grad()
def validate(
    model: DinoClipVisualPrototypeModel,
    dataloader: DataLoader,
    prototypes: VisualPrototypes,
    device: torch.device,
    temperature: float,
    on_leaf_eval_visuals: Optional[Callable[[str, str, Dict[str, float]], None]] = None,
) -> Dict[str, float]:
    model.eval()

    if not prototypes.ready():
        return {
            "image_auroc": 0.0,
            "image_ap": 0.0,
            "pixel_aupro": 0.0,
            "pixel_auroc": 0.0,
            "per_dataset": {},
            "per_dataset_class": {},
        }

    # Single accumulation: (dataset, class). Pooled metrics per leaf, then dataset / overall
    # as arithmetic mean over leaves — same convention as main_mvtec.py (per-object metrics,
    # then `mean` row over objects).
    per_dataset_class: Dict[str, Dict[str, Dict[str, list]]] = {}

    def _mean_leaf_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
        keys = ("image_auroc", "image_ap", "pixel_aupro", "pixel_auroc")
        if not rows:
            return {k: 0.0 for k in keys}
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    def _compute_metrics(
        image_scores_list: list,
        image_labels_list: list,
        pixel_scores_list: list,
        pixel_labels_list: list,
    ) -> Dict[str, float]:
        m = {"image_auroc": 0.0, "image_ap": 0.0, "pixel_aupro": 0.0, "pixel_auroc": 0.0}
        if len(image_scores_list) == 0:
            return m
        image_scores_np_local = np.asarray(image_scores_list)
        image_labels_np_local = np.asarray(image_labels_list)
        if len(np.unique(image_labels_np_local)) > 1:
            m["image_auroc"] = float(roc_auc_score(image_labels_np_local, image_scores_np_local))
            m["image_ap"] = float(average_precision_score(image_labels_np_local, image_scores_np_local))
        pixel_scores_np_local = np.concatenate(pixel_scores_list, axis=0).reshape(-1)
        pixel_labels_np_local = np.concatenate(pixel_labels_list, axis=0).reshape(-1)
        if len(np.unique(pixel_labels_np_local)) > 1:
            results_local = {
                "step1_eval": {
                    "gt_sp": image_labels_np_local,
                    "pr_sp": image_scores_np_local,
                    "imgs_masks": np.concatenate(pixel_labels_list, axis=0),
                    "anomaly_maps": np.concatenate(pixel_scores_list, axis=0),
                }
            }
            m["pixel_auroc"] = float(pixel_level_metrics(results_local, "step1_eval", "pixel-auroc"))
            m["pixel_aupro"] = float(pixel_level_metrics(results_local, "step1_eval", "pixel-aupro"))
        return m

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        image = batch["img"].to(device)
        mask = batch["mask"].to(device)
        anomaly = batch["anomaly"].to(device)

        out = model(image, image)
        _, patch_logits, patch_prob_abn = compute_patch_cls_loss(
            mapped_patches=out["mapped_patches"],
            dino_grid=out["dino_grid"],
            gt_mask=mask,
            proto_normal=prototypes.normal.to(device),
            proto_abnormal=prototypes.abnormal.to(device),
            focal_loss_fn=FocalLoss(alpha=0.25, gamma=2.0),
            dice_loss_fn=BinaryDiceLoss(),
            temperature=temperature,
        )
        _ = patch_logits  # only for symmetry/readability

        patch_map = patch_prob_abn
        img_score = patch_map.flatten(1).amax(dim=1)

        patch_map_up = F.interpolate(patch_map, size=mask.shape[-2:], mode="bilinear", align_corners=False)

        patch_map_np = patch_map_up.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        ds_names = batch.get("dataset_name", ["unknown"] * len(img_score))
        cls_names = batch.get("cls_name", ["unknown"] * len(img_score))
        img_score_np = img_score.detach().cpu().numpy()
        anomaly_np = anomaly.detach().cpu().numpy()
        for bi, ds in enumerate(ds_names):
            ds_key = str(ds)
            cls_key = str(cls_names[bi]) if bi < len(cls_names) else "unknown"
            ds_rec = per_dataset_class.setdefault(ds_key, {})
            cls_rec = ds_rec.setdefault(
                cls_key,
                {"image_scores": [], "image_labels": [], "pixel_scores": [], "pixel_labels": []},
            )
            cls_rec["image_scores"].append(float(img_score_np[bi]))
            cls_rec["image_labels"].append(int(anomaly_np[bi]))
            cls_rec["pixel_scores"].append(patch_map_np[bi : bi + 1])
            cls_rec["pixel_labels"].append(mask_np[bi : bi + 1])

    leaf_items: List[Tuple[str, str, Dict[str, list]]] = [
        (ds, cls, v)
        for ds, cls_dict in sorted(per_dataset_class.items())
        for cls, v in sorted(cls_dict.items(), key=lambda x: x[0])
    ]
    metrics_pdc: Dict[str, Dict[str, Dict[str, float]]] = {}
    if leaf_items:
        print(f"[validate] pixel/image metrics: {len(leaf_items)} (dataset, class) group(s)", flush=True)
    pbar_m = tqdm(
        leaf_items,
        desc="Val metrics (per class)",
        leave=True,
        dynamic_ncols=True,
        unit="cls",
    )
    for ds, cls, v in pbar_m:
        pbar_m.set_postfix_str(f"{ds}/{cls}", refresh=False)
        mrow = _compute_metrics(v["image_scores"], v["image_labels"], v["pixel_scores"], v["pixel_labels"])
        metrics_pdc.setdefault(ds, {})[cls] = mrow
        n_img = len(v["image_scores"])
        tqdm.write(
            f"[validate] {ds} / {cls} (n={n_img}) | "
            f"image_auroc={mrow['image_auroc']:.4f} image_ap={mrow['image_ap']:.4f} | "
            f"pixel_auroc={mrow['pixel_auroc']:.4f} pixel_aupro={mrow['pixel_aupro']:.4f}",
        )
        if on_leaf_eval_visuals is not None:
            on_leaf_eval_visuals(ds, cls, mrow)
    leaf_rows = [m for cls_dict in metrics_pdc.values() for m in cls_dict.values()]
    metrics = _mean_leaf_metrics(leaf_rows)
    metrics["per_dataset_class"] = metrics_pdc
    metrics["per_dataset"] = {
        ds: _mean_leaf_metrics(list(cls_dict.values())) for ds, cls_dict in metrics_pdc.items()
    }

    model.train()
    return metrics


# =========================
# Training
# =========================
def train(args: argparse.Namespace) -> None:
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("DINO -> CLIP Visual Prototype Training (Image-only)")
    print("=" * 80)
    print(f"device: {device}")
    dataset_root = getattr(args, "dataset_root", None)
    print(f"dataset_root: {dataset_root}")
    print(f"output_dir: {args.output_dir}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"features_list: {args.features_list}")
    print(f"temperature: {args.temperature}")
    print("=" * 80)

    train_specs = list(getattr(args, "train_datasets", []) or [])
    test_specs = list(getattr(args, "test_datasets", []) or [])
    train_samples = build_samples_from_specs(str(dataset_root), train_specs)
    val_samples = build_samples_from_specs(str(dataset_root), test_specs)
    print(f"train samples: {len(train_samples)}")
    print(f"test samples:  {len(val_samples)}")

    train_dataset = VisualPrototypeImageDataset(train_samples, image_size=args.image_size)
    val_dataset = VisualPrototypeImageDataset(val_samples, image_size=args.image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    _ = AutoImageProcessor.from_pretrained(args.dino_model_path, local_files_only=args.local_files_only)

    model = DinoClipVisualPrototypeModel(
        dino_model_path=args.dino_model_path,
        clip_model_path=args.clip_model_path,
        layer_indices=args.features_list,
        dino_image_size=args.image_size,
        clip_image_size=args.clip_image_size,
        local_files_only=args.local_files_only,
    ).to(device)
    model.set_train_mode(train_all=args.train_all)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(train_loader)))

    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    dice_loss_fn = BinaryDiceLoss()

    prototypes = VisualPrototypes()
    best_image_auroc = 0.0
    best_pixel_auroc = 0.0

    for epoch in range(args.epochs):
        ep = epoch

        def _on_leaf_eval_vis(ds: str, cls: str, _mrow: Dict[str, float], _ep: int = ep) -> None:
            _save_step1_eval_vis_for_leaf(
                model=model,
                device=device,
                prototypes=prototypes,
                val_samples=val_samples,
                args=args,
                epoch_idx=_ep,
                ds_name=ds,
                cls_name=cls,
            )

        t0 = time.time()
        meter_total = []
        meter_global = []
        meter_patch_align = []
        meter_patch_cls = []
        meter_img_cls = []
        meter_consistency = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for batch in pbar:
            image = batch["img"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            anomaly = batch["anomaly"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            out = model(image, image)
            mapped_cls = out["mapped_cls"]
            mapped_patches = out["mapped_patches"]
            clip_global = out["clip_global"]
            clip_patches = out["clip_patches"].detach()
            dino_grid = out["dino_grid"]
            clip_grid = out["clip_grid"]

            with torch.no_grad():
                prototypes = update_visual_prototypes(
                    prototypes=prototypes,
                    clip_patches=clip_patches,
                    clip_grid=clip_grid,
                    gt_mask=mask,
                    momentum=args.proto_momentum,
                )

            loss_global, loss_global_parts = compute_global_alignment_loss(
                mapped_cls=mapped_cls,
                clip_global=clip_global.detach(),
                temperature=args.temperature,
            )
            loss_patch_align = compute_dense_patch_alignment_loss(
                mapped_patches=mapped_patches,
                dino_grid=dino_grid,
                clip_patches=clip_patches,
                clip_grid=clip_grid,
                gt_mask=mask,
                patch_pos_weight=args.patch_pos_weight,
            )

            loss_patch_cls = mapped_cls.new_zeros(())
            loss_img_cls = mapped_cls.new_zeros(())
            loss_consistency = mapped_cls.new_zeros(())
            patch_prob_abn = None
            img_logits = None

            if prototypes.ready():
                loss_patch_cls, _, patch_prob_abn = compute_patch_cls_loss(
                    mapped_patches=mapped_patches,
                    dino_grid=dino_grid,
                    gt_mask=mask,
                    proto_normal=prototypes.normal.to(device),
                    proto_abnormal=prototypes.abnormal.to(device),
                    focal_loss_fn=focal_loss_fn,
                    dice_loss_fn=dice_loss_fn,
                    temperature=args.temperature,
                )
                loss_img_cls, img_logits = compute_img_cls_loss(
                    mapped_cls=mapped_cls,
                    anomaly_label=anomaly,
                    proto_normal=prototypes.normal.to(device),
                    proto_abnormal=prototypes.abnormal.to(device),
                    temperature=args.temperature,
                )
                loss_consistency = compute_consistency_loss(img_logits, patch_prob_abn)

            total_loss = (
                args.lambda_global * loss_global
                + args.lambda_patch_align * loss_patch_align
                + args.lambda_patch_cls * loss_patch_cls
                + args.lambda_img_cls * loss_img_cls
                + args.lambda_consistency * loss_consistency
            )

            total_loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()

            meter_total.append(float(total_loss.detach().item()))
            meter_global.append(float(loss_global.detach().item()))
            meter_patch_align.append(float(loss_patch_align.detach().item()))
            meter_patch_cls.append(float(loss_patch_cls.detach().item()))
            meter_img_cls.append(float(loss_img_cls.detach().item()))
            meter_consistency.append(float(loss_consistency.detach().item()))

            pbar.set_postfix(
                total=f"{np.mean(meter_total):.4f}",
                global_=f"{np.mean(meter_global):.4f}",
                palign=f"{np.mean(meter_patch_align):.4f}",
                pcls=f"{np.mean(meter_patch_cls):.4f}",
                icls=f"{np.mean(meter_img_cls):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                i2t=f"{float(loss_global_parts['loss_i2t']):.3f}",
                t2i=f"{float(loss_global_parts['loss_t2i']):.3f}",
            )

        _vis_cb = (
            _on_leaf_eval_vis
            if int(getattr(args, "eval_vis_num_samples", 0) or 0) > 0
            else None
        )
        metrics = validate(
            model=model,
            dataloader=val_loader,
            prototypes=prototypes,
            device=device,
            temperature=args.temperature,
            on_leaf_eval_visuals=_vis_cb,
        )

        epoch_time = time.time() - t0
        print("-" * 80)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"time={epoch_time:.1f}s | "
            f"loss={np.mean(meter_total):.4f} | "
            f"global={np.mean(meter_global):.4f} | "
            f"patch_align={np.mean(meter_patch_align):.4f} | "
            f"patch_cls={np.mean(meter_patch_cls):.4f} | "
            f"img_cls={np.mean(meter_img_cls):.4f} | "
            f"consistency={np.mean(meter_consistency):.4f}"
        )
        print(
            f"Val | image_auroc={metrics['image_auroc']:.4f} | "
            f"image_ap={metrics['image_ap']:.4f} | "
            f"pixel_aupro={metrics['pixel_aupro']:.4f} | "
            f"pixel_auroc={metrics['pixel_auroc']:.4f}"
        )
        per_ds = metrics.get("per_dataset", {}) or {}
        if per_ds:
            print("Val per-dataset:")
            print(f"{'dataset':>24} | {'image_auroc':>11} | {'image_ap':>9} | {'pixel_aupro':>12} | {'pixel_auroc':>12}")
            for ds_name in sorted(per_ds.keys()):
                mds = per_ds[ds_name]
                print(
                    f"{ds_name:>24} | "
                    f"{mds['image_auroc']:11.4f} | "
                    f"{mds['image_ap']:9.4f} | "
                    f"{mds['pixel_aupro']:12.4f} | "
                    f"{mds['pixel_auroc']:12.4f}"
                )
        per_ds_cls = metrics.get("per_dataset_class", {}) or {}
        if per_ds_cls:
            print("Val per-dataset per-class:")
            print(
                f"{'dataset':>20} | {'class':>16} | {'image_auroc':>11} | {'image_ap':>9} | {'pixel_aupro':>12} | {'pixel_auroc':>12}"
            )
            for ds_name in sorted(per_ds_cls.keys()):
                cls_dict = per_ds_cls[ds_name]
                for cls_name in sorted(cls_dict.keys()):
                    mdc = cls_dict[cls_name]
                    print(
                        f"{ds_name:>20} | "
                        f"{cls_name:>16} | "
                        f"{mdc['image_auroc']:11.4f} | "
                        f"{mdc['image_ap']:9.4f} | "
                        f"{mdc['pixel_aupro']:12.4f} | "
                        f"{mdc['pixel_auroc']:12.4f}"
                    )
        print("-" * 80)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "prototypes": {
                "normal": prototypes.normal.detach().cpu() if prototypes.normal is not None else None,
                "abnormal": prototypes.abnormal.detach().cpu() if prototypes.abnormal is not None else None,
            },
            "metrics": metrics,
            "args": vars(args),
        }

        epoch_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}.pth")
        torch.save(checkpoint, epoch_path)

        if metrics["image_auroc"] >= best_image_auroc:
            best_image_auroc = metrics["image_auroc"]
            torch.save(checkpoint, os.path.join(args.output_dir, "best_image_auroc.pth"))

        if metrics["pixel_auroc"] >= best_pixel_auroc:
            best_pixel_auroc = metrics["pixel_auroc"]
            torch.save(checkpoint, os.path.join(args.output_dir, "best_pixel_auroc.pth"))

    # 单独保存桥接权重，便于后续加载 mapper
    bridge_state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if k.startswith("cls_mapper") or k.startswith("patch_mapper")
    }
    torch.save({"state_dict": bridge_state}, os.path.join(args.output_dir, "dino_bridge.bin"))

    print("=" * 80)
    print("Training complete")
    print(f"Best image AUROC: {best_image_auroc:.4f}")
    print(f"Best pixel AUROC: {best_pixel_auroc:.4f}")
    print(f"Saved to: {args.output_dir}")
    print("=" * 80)


# =========================
# CLI
# =========================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("DINO->CLIP visual prototype training", add_help=True)

    # yaml config (recommended)
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径（configs/ad_llm_step1.yaml）")
    parser.add_argument("--output_dir", type=str, default=None, help="覆盖 paths.output_dir（仅当 --config 提供时）")
    parser.add_argument("--run_name", type=str, default=None, help="覆盖 runtime.run_name（仅当 --config 提供时）")
    parser.add_argument("--num-gpu", type=int, default=None, help="覆盖 distributed.num_gpu（仅当 --config 提供时）")

    # paths
    parser.add_argument("--dataset_root", type=str, default=None, help="dataset root path")
    parser.add_argument("--dino_model_path", type=str, default=None, help="DINO 模型路径（无 --config 时必填）")
    parser.add_argument("--clip_model_path", type=str, default=None, help="CLIP 模型路径（无 --config 时必填）")
    parser.add_argument("--local_files_only", action="store_true")

    # model
    parser.add_argument("--features_list", type=int, nargs="+", default=[12, 16, 20, 24])
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--clip_image_size", type=int, default=224)
    parser.add_argument("--train_all", action="store_true", help="train DINO/CLIP too; default only trains mappers")

    # optimization
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # loss / prototype
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--patch_pos_weight", type=float, default=4.0)
    parser.add_argument("--proto_momentum", type=float, default=0.9)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--lambda_global", type=float, default=1.0)
    parser.add_argument("--lambda_patch_align", type=float, default=1.0)
    parser.add_argument("--lambda_patch_cls", type=float, default=2.0)
    parser.add_argument("--lambda_img_cls", type=float, default=0.5)
    parser.add_argument("--lambda_consistency", type=float, default=0.25)

    return parser


if __name__ == "__main__":
    cli = build_parser().parse_args()

    if cli.config:
        cfg = load_yaml_config(cli.config)
        cfg = apply_runtime_overrides(cfg, cli)

        out_dir = prepare_output_dir(
            base_dir=str(cfg["paths"]["output_dir"]),
            run_name=str(cfg["runtime"]["run_name"]),
            auto_create=bool(cfg["runtime"].get("auto_create_output_dir", True)),
        )

        step1 = cfg.get("step1", {}) or {}
        dino = cfg.get("dino", {}) or {}
        clip = cfg.get("clip", {}) or {}
        tr = cfg.get("training", {}) or {}
        data_cfg = cfg.get("data", {}) or {}
        model_cfg = cfg.get("model", {}) or {}

        args = argparse.Namespace(
            # paths
            dataset_root=str(cfg["paths"]["dataset_root"]),
            output_dir=out_dir,
            dino_model_path=str(dino["model_path"]),
            clip_model_path=str(clip["model_path"]),
            local_files_only=bool(model_cfg.get("local_files_only", False)),
            train_datasets=[],
            test_datasets=[],
            # model
            features_list=[int(x) for x in dino.get("layer_indices", [12, 16, 20, 24])],
            image_size=int(dino.get("image_size", 512)),
            clip_image_size=int(clip.get("image_size", 224)),
            train_all=bool(step1.get("train_all", False)),
            # optimization
            epochs=int(tr.get("num_epochs", 20)),
            batch_size=int(step1.get("batch_size", 8)),
            num_workers=int(step1.get("num_workers", 0)),
            lr=float(step1.get("lr", 1e-4)),
            weight_decay=float(step1.get("weight_decay", 1e-2)),
            grad_clip=float(step1.get("grad_clip", 1.0)),
            seed=int(tr.get("seed", 42)),
            max_image_size=int(data_cfg.get("max_image_size", 512)),
            resize_factor=int(data_cfg.get("factor", 28)),
            eval_vis_num_samples=int(step1.get("eval_vis_num_samples", 0)),
            eval_heatmap_alpha=float(step1.get("eval_heatmap_alpha", 0.45)),
            step1_visual_debug=bool(tr.get("step1_visual_debug", False)),
            # loss / prototype
            temperature=float(step1.get("temperature", 0.07)),
            patch_pos_weight=float(step1.get("patch_pos_weight", 4.0)),
            proto_momentum=float(step1.get("proto_momentum", 0.9)),
            focal_alpha=float(step1.get("focal_alpha", 0.25)),
            focal_gamma=float(step1.get("focal_gamma", 2.0)),
            lambda_global=float(step1.get("lambda_global", 1.0)),
            lambda_patch_align=float(step1.get("lambda_patch_align", 1.0)),
            lambda_patch_cls=float(step1.get("lambda_patch_cls", 2.0)),
            lambda_img_cls=float(step1.get("lambda_img_cls", 0.5)),
            lambda_consistency=float(step1.get("lambda_consistency", 0.25)),
        )
        ds_cfg = cfg.get("datasets", []) or []
        if isinstance(ds_cfg, dict):
            args.train_datasets = list(ds_cfg.get("train", []) or [])
            args.test_datasets = list(ds_cfg.get("test", []) or [])
        else:
            # backward compatibility: old list format => use same list for train/test
            args.train_datasets = list(ds_cfg)
            args.test_datasets = list(ds_cfg)
        if not args.train_datasets:
            raise SystemExit("step1 config requires datasets.train list")
        if not args.test_datasets:
            raise SystemExit("step1 config requires datasets.test list")
        train(args)
    else:
        raise SystemExit("step1 现在仅支持 --config 模式，请在配置中提供 datasets.train / datasets.test。")
