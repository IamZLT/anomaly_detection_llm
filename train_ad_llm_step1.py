import os
import glob
import math
import time
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

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
from utils.qwen_common import prepare_output_dir
from utils.qwen_config import apply_runtime_overrides, load_yaml_config


# =========================
# Utils
# =========================
def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_mask_paths(mask_dir: str, img_path: str) -> List[str]:
    img_name = os.path.basename(img_path)
    stem, ext = os.path.splitext(img_name)
    return [
        os.path.join(mask_dir, f"{stem}_mask{ext}"),
        os.path.join(mask_dir, img_name),
        os.path.join(mask_dir, f"{stem}_mask.png"),
        os.path.join(mask_dir, f"{stem}.png"),
    ]


# =========================
# Dataset
# =========================
class MVTecVisualPrototypeDataset(Dataset):
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
            "cls_name": item["cls_name"],
            "defect_type": item["defect_type"],
        }


def build_mvtec_splits(data_path: str, seed: int = 42, val_ratio: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    mvtec_classes = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
    ]

    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    rng = random.Random(seed)

    for cls_name in mvtec_classes:
        cls_train: List[Dict] = []
        cls_val: List[Dict] = []

        # normal: train/good
        normal_dir = os.path.join(data_path, cls_name, "train", "good")
        if os.path.exists(normal_dir):
            normal_files = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
            rng.shuffle(normal_files)
            split_idx = max(1, int(len(normal_files) * (1 - val_ratio))) if len(normal_files) > 1 else len(normal_files)
            for i, img_path in enumerate(normal_files):
                sample = {
                    "img_path": img_path,
                    "mask_path": None,
                    "anomaly": 0,
                    "cls_name": cls_name,
                    "defect_type": "good",
                }
                (cls_train if i < split_idx else cls_val).append(sample)

        # abnormal: test/<defect>
        test_root = os.path.join(data_path, cls_name, "test")
        if os.path.exists(test_root):
            for defect_type in sorted(os.listdir(test_root)):
                defect_dir = os.path.join(test_root, defect_type)
                if defect_type == "good" or not os.path.isdir(defect_dir):
                    continue
                defect_files = sorted(glob.glob(os.path.join(defect_dir, "*.png")))
                rng.shuffle(defect_files)
                split_idx = max(1, int(len(defect_files) * (1 - val_ratio))) if len(defect_files) > 1 else len(defect_files)
                mask_dir = os.path.join(data_path, cls_name, "ground_truth", defect_type)
                alt_mask_dir = os.path.join(data_path, cls_name, "groundtruth", defect_type)
                for i, img_path in enumerate(defect_files):
                    mask_path = None
                    for cand in build_mask_paths(mask_dir, img_path) + build_mask_paths(alt_mask_dir, img_path):
                        if os.path.exists(cand):
                            mask_path = cand
                            break
                    sample = {
                        "img_path": img_path,
                        "mask_path": mask_path,
                        "anomaly": 1,
                        "cls_name": cls_name,
                        "defect_type": defect_type,
                    }
                    (cls_train if i < split_idx else cls_val).append(sample)

        train_samples.extend(cls_train)
        val_samples.extend(cls_val)

    return train_samples, val_samples


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
def validate(
    model: DinoClipVisualPrototypeModel,
    dataloader: DataLoader,
    prototypes: VisualPrototypes,
    device: torch.device,
    temperature: float,
) -> Dict[str, float]:
    model.eval()

    if not prototypes.ready():
        return {
            "image_auroc": 0.0,
            "pixel_auroc": 0.0,
            "pixel_ap": 0.0,
        }

    image_scores = []
    image_labels = []
    pixel_scores = []
    pixel_labels = []

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

        image_scores.extend(img_score.detach().cpu().numpy().tolist())
        image_labels.extend(anomaly.detach().cpu().numpy().tolist())
        pixel_scores.append(patch_map_up.detach().cpu().numpy())
        pixel_labels.append(mask.detach().cpu().numpy())

    metrics = {"image_auroc": 0.0, "pixel_auroc": 0.0, "pixel_ap": 0.0}

    image_scores_np = np.asarray(image_scores)
    image_labels_np = np.asarray(image_labels)
    if len(np.unique(image_labels_np)) > 1:
        metrics["image_auroc"] = float(roc_auc_score(image_labels_np, image_scores_np))

    pixel_scores_np = np.concatenate(pixel_scores, axis=0).reshape(-1)
    pixel_labels_np = np.concatenate(pixel_labels, axis=0).reshape(-1)
    if len(np.unique(pixel_labels_np)) > 1:
        metrics["pixel_auroc"] = float(roc_auc_score(pixel_labels_np, pixel_scores_np))
        metrics["pixel_ap"] = float(average_precision_score(pixel_labels_np, pixel_scores_np))

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
    print(f"data_path: {args.data_path}")
    print(f"output_dir: {args.output_dir}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"features_list: {args.features_list}")
    print(f"temperature: {args.temperature}")
    print("=" * 80)

    train_samples, val_samples = build_mvtec_splits(args.data_path, seed=args.seed, val_ratio=args.val_ratio)
    print(f"train samples: {len(train_samples)}")
    print(f"val samples:   {len(val_samples)}")

    train_dataset = MVTecVisualPrototypeDataset(train_samples, image_size=args.image_size)
    val_dataset = MVTecVisualPrototypeDataset(val_samples, image_size=args.image_size)
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

        metrics = validate(
            model=model,
            dataloader=val_loader,
            prototypes=prototypes,
            device=device,
            temperature=args.temperature,
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
            f"pixel_auroc={metrics['pixel_auroc']:.4f} | "
            f"pixel_ap={metrics['pixel_ap']:.4f}"
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
    parser.add_argument("--data_path", type=str, default=None, help="MVTec root path（无 --config 时必填）")
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
    parser.add_argument("--val_ratio", type=float, default=0.2)

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
        model_cfg = cfg.get("model", {}) or {}

        args = argparse.Namespace(
            # paths
            data_path=str(cfg["paths"]["dataset_root"]),
            output_dir=out_dir,
            dino_model_path=str(dino["model_path"]),
            clip_model_path=str(clip["model_path"]),
            local_files_only=bool(model_cfg.get("local_files_only", False)),
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
            val_ratio=float(step1.get("val_ratio", 0.2)),
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
        train(args)
    else:
        # legacy CLI mode
        if not cli.data_path or not cli.dino_model_path or not cli.clip_model_path:
            raise SystemExit("无 --config 时，必须提供 --data_path --dino_model_path --clip_model_path")

        # keep old behavior: output_dir default if not provided
        if cli.output_dir is None:
            cli.output_dir = "./outputs/dino_clip_visual_proto"
        train(cli)
