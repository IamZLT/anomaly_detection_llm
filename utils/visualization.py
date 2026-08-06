from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from loss.BinaryDiceLoss import BinaryDiceLoss
from loss.FocalLoss import FocalLoss
from models.visual_proto import DinoClipVisualPrototypeModel, VisualPrototypes
from train_ad_llm_step1 import compute_patch_cls_loss


def _tensor_stats_line(name: str, t: torch.Tensor) -> str:
    x = t.detach().float().cpu().reshape(-1)
    return (
        f"{name} shape={tuple(t.shape)} mean={float(x.mean()):.6f} std={float(x.std()):.6f} "
        f"min={float(x.min()):.6f} max={float(x.max()):.6f}"
    )


def bbox_from_map(anom: np.ndarray, thr_quantile: float = 0.98) -> List[int] | None:
    """
    anom: [H,W] float in [0,1]
    返回 [x1,y1,x2,y2] 或 None
    """
    h, w = anom.shape
    thr = float(np.quantile(anom, thr_quantile))
    m = anom >= thr
    if m.sum() < 10:
        return None
    ys, xs = np.where(m)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    return [x1, y1, x2, y2]


def heatmap_overlay(image: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    image: PIL RGB
    heat:  [H,W] in [0,1] (same size as image)
    return: overlay image (RGB)
    """
    heat = np.clip(heat, 0.0, 1.0)
    r = np.clip(1.5 * heat - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * heat - 1.0), 0.0, 1.0)
    b = np.clip(0.5 - 1.5 * (heat - 1.0), 0.0, 1.0)
    cm = np.stack([r, g, b], axis=-1)
    cm_u8 = (cm * 255.0).astype(np.uint8)
    hm = Image.fromarray(cm_u8, mode="RGB")
    if hm.size != image.size:
        hm = hm.resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), hm, float(alpha))


def prepare_step1_image_tensor(
    img_rs: Image.Image,
    dino_image_size: int,
    device: torch.device,
) -> torch.Tensor:
    """与 ``test_ad_llm_step1``：smart_resize 后的 PIL → [1,3,H,W] ImageNet。"""
    image_size = int(dino_image_size)
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tfm(img_rs).unsqueeze(0).to(device)


def compute_step1_heat_up(
    *,
    model: DinoClipVisualPrototypeModel,
    img_t: torch.Tensor,
    img_rs: Image.Image,
    cfg: dict,
    prototypes: VisualPrototypes,
    device: torch.device,
) -> Tuple[np.ndarray, str]:
    """
    与 ``test_ad_llm_step1.main`` 中 ``out = model(img_t, img_t)`` 之后到 ``heat_up`` 完全一致。
    Returns:
        heat_up: [H_img, W_img] float
        mode: ``\"prototype\"`` | ``\"cosdiff\"``
    """
    image_size = int(cfg["dino"]["image_size"])
    # 训练：training.step1_visual_debug；单图脚本 test_ad_llm_step1：可在 test.step1_visual_debug 打开
    _dbg = bool((cfg.get("training", {}) or {}).get("step1_visual_debug", False)) or bool(
        (cfg.get("test", {}) or {}).get("step1_visual_debug", False)
    )

    out = model(img_t, img_t)
    mapped_patches = out["mapped_patches"]
    dino_grid = out["dino_grid"]

    if _dbg:
        print(
            "[compute_step1_heat_up] 路径说明: 使用 DinoClipVisualPrototypeModel.forward "
            "（patch_mapper 直接输出；不经 QwenDinoBridgeModel._build_visual_tokens，无 AVNet proto 幅度调制）",
            flush=True,
        )
        print(f"[compute_step1_heat_up] {_tensor_stats_line('mapped_patches', mapped_patches)}", flush=True)
        print(f"[compute_step1_heat_up] dino_grid={dino_grid} img_t {_tensor_stats_line('img_t', img_t)}", flush=True)

    if prototypes.ready():
        proto_n = prototypes.normal.to(device)
        proto_a = prototypes.abnormal.to(device)
        loss_patch_cls, patch_logits, patch_prob_abn = compute_patch_cls_loss(
            mapped_patches=mapped_patches,
            dino_grid=dino_grid,
            gt_mask=torch.zeros((1, 1, image_size, image_size), device=device),
            proto_normal=proto_n,
            proto_abnormal=proto_a,
            focal_loss_fn=FocalLoss(alpha=0.25, gamma=2.0),
            dice_loss_fn=BinaryDiceLoss(),
            temperature=float(cfg.get("step1", {}).get("temperature", 0.07)),
        )
        _ = (loss_patch_cls, patch_logits)
        heat = patch_prob_abn[0, 0].detach().float().cpu().numpy()
        mode = "prototype"
        if _dbg:
            ht = torch.from_numpy(heat)
            print(
                "[compute_step1_heat_up] prototype 模式 "
                f"temperature={float(cfg.get('step1', {}).get('temperature', 0.07))} "
                f"{_tensor_stats_line('patch_prob_abn[0,0]', ht)}",
                flush=True,
            )
    else:
        clip_patches = out["clip_patches"]
        clip_grid = out["clip_grid"]
        if clip_patches is None or clip_patches.numel() == 0 or clip_grid == (0, 0):
            raise RuntimeError(
                "当前无 prototypes 且 CLIP patch 不可用，无法生成与 test_ad_llm_step1 一致的热力图。"
            )
        b, pc, c = clip_patches.shape
        hc, wc = clip_grid
        hd, wd = dino_grid
        clip_map = clip_patches.view(b, hc, wc, c).permute(0, 3, 1, 2)
        clip_map = F.interpolate(clip_map, size=(hd, wd), mode="bilinear", align_corners=False)
        clip_map = clip_map.permute(0, 2, 3, 1).reshape(b, hd * wd, c)
        mp = F.normalize(mapped_patches, dim=-1)
        cp = F.normalize(clip_map, dim=-1)
        cos = (mp * cp).sum(dim=-1)[0].view(hd, wd)
        heat = (1.0 - cos).detach().float().cpu().numpy()
        mode = "cosdiff"
        if _dbg:
            print("[compute_step1_heat_up] cosdiff 模式（无 prototypes）", flush=True)

    if _dbg:
        ht0 = torch.from_numpy(heat.astype("float64"))
        print(f"[compute_step1_heat_up] heat 归一化前 {_tensor_stats_line('heat', ht0)}", flush=True)

    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_t = torch.from_numpy(heat).unsqueeze(0).unsqueeze(0)
    heat_up = F.interpolate(
        heat_t,
        size=(int(img_rs.height), int(img_rs.width)),
        mode="bilinear",
        align_corners=False,
    )
    heat_up = heat_up.squeeze(0).squeeze(0).numpy()
    if _dbg:
        hu = torch.from_numpy(heat_up.astype("float64"))
        print(f"[compute_step1_heat_up] heat_up(overlay 前) {_tensor_stats_line('heat_up', hu)}", flush=True)
    return heat_up, mode


def visual_prototypes_from_avnet(av) -> VisualPrototypes:
    """从 ``QwenDinoBridgeModel`` 已加载的 buffer 构造 ``VisualPrototypes``（与单测一致）。"""
    p = VisualPrototypes()
    if av.visual_prototypes_ready():
        p.normal = av.proto_normal.detach().cpu()
        p.abnormal = av.proto_abnormal.detach().cpu()
    return p


def attach_avnet_to_step1_shell(av, sm: DinoClipVisualPrototypeModel) -> None:
    """
    用 ``QwenDinoBridgeModel`` 的视觉骨干与 mapper 替换 ``DinoClipVisualPrototypeModel`` 内对应模块。

    注意：PyTorch 子模块只能有一个父模块；赋值后 DINO/CLIP/mapper 会从 ``av`` 上被摘掉。
    热力图算完后**必须**调用 ``restore_avnet_bridge_from_step1_shell``，否则 ``av`` 的 ``generate``/训练 forward 用的是残缺图编码。
    """
    sm.dino_model = av.dino_model
    sm.clip_model = av.clip_model
    sm.cls_mapper = av.cls_mapper
    sm.patch_mapper = av.patch_mapper
    sm.dino_image_size = int(av.dino_image_size)
    sm.clip_image_size = int(av.clip_image_size)
    sm.layer_indices = list(int(x) for x in av.dino_layer_indices)


def restore_avnet_bridge_from_step1_shell(av, sm: DinoClipVisualPrototypeModel) -> None:
    """与 ``attach_avnet_to_step1_shell`` 成对：用 ``add_module`` 把子模块挂回 ``QwenDinoBridgeModel``。"""
    av.add_module("dino_model", sm.dino_model)
    av.add_module("clip_model", sm.clip_model)
    av.add_module("cls_mapper", sm.cls_mapper)
    av.add_module("patch_mapper", sm.patch_mapper)


def build_step1_shell(cfg: dict) -> DinoClipVisualPrototypeModel:
    """
    在 CPU 上构建壳子（会短暂加载两份权重到内存）；随后 ``attach_avnet_to_step1_shell``
    用训练中的 ``QwenDinoBridgeModel`` 子模块替换，避免长期双份 GPU 占用。
    """
    return DinoClipVisualPrototypeModel(
        dino_model_path=cfg["dino"]["model_path"],
        clip_model_path=cfg["clip"]["model_path"],
        layer_indices=cfg["dino"]["layer_indices"],
        dino_image_size=int(cfg["dino"]["image_size"]),
        clip_image_size=int(cfg["clip"]["image_size"]),
        local_files_only=bool(cfg.get("model", {}).get("local_files_only", False)),
    )
