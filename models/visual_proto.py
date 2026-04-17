import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from utils.dinov3_utils import dinov3_encode_image
from utils import prompt_generator


def normalize_feature(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def mapped_patch_abnormal_prob_hw(
    mapped_patches: torch.Tensor,
    dino_grid: Tuple[int, int],
    proto_normal: torch.Tensor,
    proto_abnormal: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    与 Step1 / ``test_ad_llm_step1`` 一致：CLIP 空间 mapped patch 相对 (normal, abnormal) 原型的 softmax 异常概率。
    ``mapped_patches``: [B, P, C]；``dino_grid``：DINO patch 网格 (H, W)，H*W=P。

    Returns:
        ``[B, H, W]``，每格为属于 abnormal 原型的概率。
    """
    b, p, c = mapped_patches.shape
    gh, gw = dino_grid
    if gh * gw != p:
        raise ValueError(f"DINO grid mismatch: {gh}x{gw} != patch count {p}")
    patch_map = normalize_feature(mapped_patches).view(b, gh, gw, c)
    proto_n = normalize_feature(proto_normal.reshape(1, c)).view(c)
    proto_a = normalize_feature(proto_abnormal.reshape(1, c)).view(c)
    logit_n = torch.einsum("bhwc,c->bhw", patch_map, proto_n)
    logit_a = torch.einsum("bhwc,c->bhw", patch_map, proto_a)
    logits = torch.stack([logit_n, logit_a], dim=1) / max(float(temperature), 1e-8)
    return torch.softmax(logits, dim=1)[:, 1]


def infer_square_hw(num_patches: int) -> Tuple[int, int]:
    side = int(round(math.sqrt(max(1, num_patches))))
    if side * side != num_patches:
        raise ValueError(f"Cannot infer square grid from num_patches={num_patches}")
    return side, side


def masked_average(feats: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    feats:   [B, P, C]
    weights: [B, P]
    return:  [B, C]
    """
    w = weights.unsqueeze(-1)
    pooled = (feats * w).sum(dim=1) / (w.sum(dim=1) + eps)
    return pooled


class ResidualVisualProjection(nn.Module):
    def __init__(self, vis_dim: int, output_dim: int):
        super().__init__()
        self.base = prompt_generator.VisualProjection(vis_dim=vis_dim, output_dim=output_dim)
        self.skip = nn.Linear(vis_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.skip(x)


class MultiLayerCLSProjection(nn.Module):
    def __init__(self, vis_dim: int, output_dim: int, num_layers: int, init_layer_indices: Optional[List[int]] = None):
        super().__init__()
        self.num_layers = num_layers
        self.layer_projs = nn.ModuleList([ResidualVisualProjection(vis_dim, output_dim) for _ in range(num_layers)])
        self.layer_logits = nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))
        if init_layer_indices is not None and len(init_layer_indices) == num_layers:
            vals = torch.tensor(init_layer_indices, dtype=torch.float32)
            vals = vals - vals.mean()
            with torch.no_grad():
                self.layer_logits.copy_(vals)

    def forward(self, cls_stack: torch.Tensor) -> torch.Tensor:
        # cls_stack: [L, B, D]
        l, b, _ = cls_stack.shape
        if l != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {l}")
        outs = []
        for li in range(l):
            y = self.layer_projs[li](cls_stack[li])
            y = normalize_feature(y)
            outs.append(y)
        outs = torch.stack(outs, dim=0)
        weights = F.softmax(self.layer_logits[:l], dim=0).view(l, 1, 1)
        return (outs * weights).sum(dim=0)


class MultiLayerPatchProjection(nn.Module):
    def __init__(self, vis_dim: int, output_dim: int, num_layers: int, init_layer_indices: Optional[List[int]] = None):
        super().__init__()
        self.num_layers = num_layers
        self.layer_projs = nn.ModuleList([ResidualVisualProjection(vis_dim, output_dim) for _ in range(num_layers)])
        self.layer_logits = nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))
        if init_layer_indices is not None and len(init_layer_indices) == num_layers:
            vals = torch.tensor(init_layer_indices, dtype=torch.float32)
            vals = vals - vals.mean()
            with torch.no_grad():
                self.layer_logits.copy_(vals)

    def forward(self, layer_patches: torch.Tensor) -> torch.Tensor:
        # layer_patches: [L, B, P, D]
        l, b, p, d = layer_patches.shape
        if l != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {l}")
        outs = []
        for li in range(l):
            x = layer_patches[li].reshape(b * p, d)
            y = self.layer_projs[li](x)
            y = normalize_feature(y)
            outs.append(y.view(b, p, -1))
        outs = torch.stack(outs, dim=0)
        weights = F.softmax(self.layer_logits[:l], dim=0).view(l, 1, 1, 1)
        return (outs * weights).sum(dim=0)


class DinoClipVisualPrototypeModel(nn.Module):
    def __init__(
        self,
        dino_model_path: str,
        clip_model_path: str,
        layer_indices: List[int],
        dino_image_size: int = 512,
        clip_image_size: int = 224,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.layer_indices = [int(x) for x in layer_indices]
        self.dino_image_size = int(dino_image_size)
        self.clip_image_size = int(clip_image_size)

        self.dino_model = AutoModel.from_pretrained(
            dino_model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.clip_model = AutoModel.from_pretrained(
            clip_model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        self.dino_hidden = int(getattr(self.dino_model.config, "hidden_size", 1024))
        self.clip_hidden = int(
            getattr(self.clip_model.config, "projection_dim", getattr(self.clip_model.config, "hidden_size", 768))
        )

        self.cls_mapper = MultiLayerCLSProjection(
            vis_dim=self.dino_hidden,
            output_dim=self.clip_hidden,
            num_layers=len(self.layer_indices),
            init_layer_indices=self.layer_indices,
        )
        self.patch_mapper = MultiLayerPatchProjection(
            vis_dim=self.dino_hidden,
            output_dim=self.clip_hidden,
            num_layers=len(self.layer_indices),
            init_layer_indices=self.layer_indices,
        )

    def set_train_mode(self, train_all: bool = False) -> None:
        for p in self.parameters():
            p.requires_grad = False
        if train_all:
            for p in self.parameters():
                p.requires_grad = True
        else:
            for p in self.cls_mapper.parameters():
                p.requires_grad = True
            for p in self.patch_mapper.parameters():
                p.requires_grad = True

    def encode_dino(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        x = pixel_values.float()
        if x.shape[-2:] != (self.dino_image_size, self.dino_image_size):
            x = F.interpolate(x, size=(self.dino_image_size, self.dino_image_size), mode="bilinear", align_corners=False)
        out = dinov3_encode_image(
            x,
            processor=None,
            model=self.dino_model,
            device=x.device,
            layer_indices=self.layer_indices,
        )
        if "multi_layer_features" not in out:
            raise RuntimeError("dinov3_encode_image must return multi_layer_features")
        mlf = out["multi_layer_features"]
        cls_stack = torch.stack([feat[:, 0, :] for feat in mlf], dim=0)
        patch_stack = torch.stack([feat[:, 1:, :] for feat in mlf], dim=0)
        gh, gw = out["grid_size"].tolist()
        return cls_stack, patch_stack, (int(gh), int(gw))

    def encode_clip(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        x = pixel_values.float()
        if x.shape[-2:] != (self.clip_image_size, self.clip_image_size):
            x = F.interpolate(x, size=(self.clip_image_size, self.clip_image_size), mode="bilinear", align_corners=False)

        if hasattr(self.clip_model, "vision_model"):
            vision_out = self.clip_model.vision_model(pixel_values=x)
            if getattr(vision_out, "pooler_output", None) is not None:
                feat = vision_out.pooler_output
            else:
                feat = vision_out.last_hidden_state[:, 0, :]
            patch = vision_out.last_hidden_state[:, 1:, :]
            if hasattr(self.clip_model, "visual_projection"):
                feat = self.clip_model.visual_projection(feat)
                patch = self.clip_model.visual_projection(patch)
            ph, pw = infer_square_hw(patch.shape[1])
            return feat, patch, (ph, pw)

        if hasattr(self.clip_model, "get_image_features"):
            feat = self.clip_model.get_image_features(pixel_values=x)
            empty = torch.empty((feat.shape[0], 0, feat.shape[-1]), device=feat.device, dtype=feat.dtype)
            return feat, empty, (0, 0)

        raise RuntimeError("Current CLIP model does not support image feature extraction")

    def forward(self, dino_pixel_values: torch.Tensor, clip_pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        dino_cls_layers, dino_patch_layers, dino_grid = self.encode_dino(dino_pixel_values)
        clip_global, clip_patches, clip_grid = self.encode_clip(clip_pixel_values)
        mapped_cls = self.cls_mapper(dino_cls_layers)
        mapped_patches = self.patch_mapper(dino_patch_layers)
        return {
            "mapped_cls": mapped_cls,
            "mapped_patches": mapped_patches,
            "clip_global": clip_global,
            "clip_patches": clip_patches,
            "dino_grid": dino_grid,
            "clip_grid": clip_grid,
        }


@dataclass
class VisualPrototypes:
    normal: Optional[torch.Tensor] = None
    abnormal: Optional[torch.Tensor] = None

    def ready(self) -> bool:
        return self.normal is not None and self.abnormal is not None


@torch.no_grad()
def update_visual_prototypes(
    prototypes: VisualPrototypes,
    clip_patches: torch.Tensor,
    clip_grid: Tuple[int, int],
    gt_mask: torch.Tensor,
    momentum: float = 0.9,
    min_abnormal_pixels: int = 1,
) -> VisualPrototypes:
    """
    用 CLIP patch 特征和 mask 更新视觉原型。

    clip_patches: [B, Pc, C]
    gt_mask:      [B, 1, H, W]
    """
    if clip_patches.numel() == 0:
        return prototypes

    b, p, c = clip_patches.shape
    gh, gw = clip_grid
    if gh * gw != p:
        raise ValueError(f"clip_grid={clip_grid} does not match patch count={p}")

    mask_small = F.interpolate(gt_mask.float(), size=(gh, gw), mode="nearest").reshape(b, -1)
    clip_patches = normalize_feature(clip_patches)

    normal_weights = 1.0 - mask_small
    normal_valid = normal_weights.sum(dim=1) > 0
    normal_proto_batch = []
    if normal_valid.any():
        proto = masked_average(clip_patches[normal_valid], normal_weights[normal_valid])
        proto = normalize_feature(proto)
        normal_proto_batch.append(proto)

    abnormal_valid = mask_small.sum(dim=1) >= float(min_abnormal_pixels)
    abnormal_proto_batch = []
    if abnormal_valid.any():
        proto = masked_average(clip_patches[abnormal_valid], mask_small[abnormal_valid])
        proto = normalize_feature(proto)
        abnormal_proto_batch.append(proto)

    if normal_proto_batch:
        current = normalize_feature(torch.cat(normal_proto_batch, dim=0).mean(dim=0, keepdim=True))
        if prototypes.normal is None:
            prototypes.normal = current.squeeze(0)
        else:
            prototypes.normal = normalize_feature(momentum * prototypes.normal + (1.0 - momentum) * current.squeeze(0))

    if abnormal_proto_batch:
        current = normalize_feature(torch.cat(abnormal_proto_batch, dim=0).mean(dim=0, keepdim=True))
        if prototypes.abnormal is None:
            prototypes.abnormal = current.squeeze(0)
        else:
            prototypes.abnormal = normalize_feature(momentum * prototypes.abnormal + (1.0 - momentum) * current.squeeze(0))

    return prototypes

