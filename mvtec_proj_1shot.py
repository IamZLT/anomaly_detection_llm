import os
import glob
import argparse
import time
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModel
from utils.dinov3_utils import dinov3_encode_image
from utils.metrics import image_level_metrics, pixel_level_metrics
from utils import prompt_generator


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sync_if_cuda(device: str):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _sum_profiler_flops(prof) -> float:
    # torch.profiler may provide flops per op when with_flops=True
    total = 0.0
    for evt in prof.key_averages():
        fl = getattr(evt, "flops", None)
        if fl is not None:
            total += float(fl)
    return total


def profile_inference(
    args,
    img: torch.Tensor,
    anomaly_head: nn.Module,
    memory_bank: List[torch.Tensor],
    dino_processor,
    dino_model,
    device: str,
):
    """
    打印推理耗时与 FLOPs（若 torch.profiler 可用）。

    - DINO: dinov3_encode_image(...)
    - Head(映射层): anomaly_head(...)
    - Combined: DINO + Head

    注：这里不统计 few-shot memory bank 的相似度检索 FLOPs（只统计 DINO + 映射层）。
    """
    iters = int(getattr(args, "profile_iters", 20))
    warmup = int(getattr(args, "profile_warmup", 5))
    iters = max(1, iters)
    warmup = max(0, warmup)

    # 只测单张图像（B=1）
    img_1 = img[:1]

    print("\n" + "=" * 80)
    print("[Profile] Inference profiling (single image, B=1)")
    print(f"  device={device}, img={tuple(img_1.shape)}, warmup={warmup}, iters={iters}")

    # --- DINO time ---
    for _ in range(warmup):
        _ = dinov3_encode_image(img_1, dino_processor, dino_model, device=device, layer_indices=args.features_list)
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        dino_out = dinov3_encode_image(img_1, dino_processor, dino_model, device=device, layer_indices=args.features_list)
    _sync_if_cuda(device)
    dino_ms = (time.perf_counter() - t0) * 1000.0 / iters

    mlf = dino_out["multi_layer_features"]

    # --- Head time ---
    for _ in range(warmup):
        _ = anomaly_head(mlf, args.temperature, return_per_layer=False)
    _sync_if_cuda(device)
    t1 = time.perf_counter()
    for _ in range(iters):
        _ = anomaly_head(mlf, args.temperature, return_per_layer=False)
    _sync_if_cuda(device)
    head_ms = (time.perf_counter() - t1) * 1000.0 / iters

    # --- Combined time (DINO + Head) ---
    for _ in range(warmup):
        dino_out2 = dinov3_encode_image(img_1, dino_processor, dino_model, device=device, layer_indices=args.features_list)
        _ = anomaly_head(dino_out2["multi_layer_features"], args.temperature, return_per_layer=False)
    _sync_if_cuda(device)
    t2 = time.perf_counter()
    for _ in range(iters):
        dino_out2 = dinov3_encode_image(img_1, dino_processor, dino_model, device=device, layer_indices=args.features_list)
        _ = anomaly_head(dino_out2["multi_layer_features"], args.temperature, return_per_layer=False)
    _sync_if_cuda(device)
    combined_ms = (time.perf_counter() - t2) * 1000.0 / iters

    print(f"  DINO time:     {dino_ms:.2f} ms/batch")
    print(f"  Head time:     {head_ms:.2f} ms/batch")
    print(f"  Combined time: {combined_ms:.2f} ms/batch")

    # --- FLOPs (best-effort, optional) ---
    try:
        import torch.profiler as prof

        activities = [prof.ProfilerActivity.CPU]
        if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
            activities.append(prof.ProfilerActivity.CUDA)

        with prof.profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
            profile_memory=False,
        ) as p_dino:
            _ = dinov3_encode_image(img_1, dino_processor, dino_model, device=device, layer_indices=args.features_list)
            _sync_if_cuda(device)
        dino_flops = _sum_profiler_flops(p_dino)

        with prof.profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
            profile_memory=False,
        ) as p_head:
            _ = anomaly_head(mlf, args.temperature, return_per_layer=False)
            _sync_if_cuda(device)
        head_flops = _sum_profiler_flops(p_head)

        with prof.profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
            profile_memory=False,
        ) as p_comb:
            dino_out3 = dinov3_encode_image(img_1, dino_processor, dino_model, device=device, layer_indices=args.features_list)
            _ = anomaly_head(dino_out3["multi_layer_features"], args.temperature, return_per_layer=False)
            _sync_if_cuda(device)
        comb_flops = _sum_profiler_flops(p_comb)

        def _fmt_flops(x: float) -> str:
            if x <= 0:
                return "n/a"
            if x >= 1e12:
                return f"{x/1e12:.3f} TFLOPs"
            if x >= 1e9:
                return f"{x/1e9:.3f} GFLOPs"
            if x >= 1e6:
                return f"{x/1e6:.3f} MFLOPs"
            return f"{x:.0f} FLOPs"

        print(f"  DINO FLOPs:     {_fmt_flops(dino_flops)} / batch")
        print(f"  Head FLOPs:     {_fmt_flops(head_flops)} / batch")
        print(f"  Combined FLOPs: {_fmt_flops(comb_flops)} / batch")
    except Exception as e:
        print(f"  [Profile] FLOPs not available: {e}")

    print("=" * 80 + "\n")


class OneShotNormalDataset(Dataset):
    """只包含若干正常图像（默认1张），用于一/少样本拟合投影层。"""

    def __init__(self, img_paths: List[str], image_size: int):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return {"img": img, "img_path": path}


class BottleTestDataset(Dataset):
    """`bottle` 类别的完整测试集（normal + abnormal），用于验证异常检测。"""

    def __init__(self, data_path: str, obj_name: str, image_size: int):
        self.data = []
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # normal: test/good
        normal_dir = os.path.join(data_path, obj_name, "test", "good")
        if os.path.exists(normal_dir):
            normal_files = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
            for p in normal_files:
                self.data.append(
                    {
                        "img_path": p,
                        "mask_path": None,
                        "anomaly": 0,
                        "defect_type": "good",
                    }
                )

        # abnormal: test/* except good
        test_root = os.path.join(data_path, obj_name, "test")
        if os.path.exists(test_root):
            for defect in os.listdir(test_root):
                defect_dir = os.path.join(test_root, defect)
                if defect == "good" or not os.path.isdir(defect_dir):
                    continue
                defect_files = sorted(glob.glob(os.path.join(defect_dir, "*.png")))
                for p in defect_files:
                    img_filename = os.path.basename(p)
                    name_wo_ext = os.path.splitext(img_filename)[0]
                    mask_dir = os.path.join(data_path, obj_name, "ground_truth", defect)

                    # MVTec 常见命名：xxx_mask.png 或同名 png
                    candidates = [
                        os.path.join(mask_dir, f"{name_wo_ext}_mask.png"),
                        os.path.join(mask_dir, img_filename),
                    ]
                    mask_path = None
                    for c in candidates:
                        if os.path.exists(c):
                            mask_path = c
                            break

                    # 额外尝试 groundtruth 目录（有的代码习惯不同）
                    if mask_path is None:
                        alt_dir = os.path.join(data_path, obj_name, "groundtruth", defect)
                        candidates_alt = [
                            os.path.join(alt_dir, f"{name_wo_ext}_mask.png"),
                            os.path.join(alt_dir, img_filename),
                        ]
                        for c in candidates_alt:
                            if os.path.exists(c):
                                mask_path = c
                                break

                    self.data.append(
                        {
                            "img_path": p,
                            "mask_path": mask_path,
                            "anomaly": 1,
                            "defect_type": defect,
                        }
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["img_path"]).convert("RGB")
        img = self.transform(img)

        # 加载/构造像素级掩码（0/1，大小为 image_size）
        if item.get("mask_path") is not None and os.path.exists(item["mask_path"]):
            mask_img = Image.open(item["mask_path"]).convert("L")
            mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_np = np.array(mask_img)
            mask_np = (mask_np > 0).astype(np.float32)  # [H,W] in {0,1}
        else:
            mask_np = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        mask = torch.from_numpy(mask_np).unsqueeze(0)  # [1,H,W]
        return {
            "img": img,
            "img_path": item["img_path"],
            "anomaly": torch.tensor(item["anomaly"], dtype=torch.long),
            "defect_type": item["defect_type"],
            "mask": mask,
        }


def build_models(args, device):
    # DINOv3
    dino_processor = AutoImageProcessor.from_pretrained(args.dinov3_model_path)
    dino_model = AutoModel.from_pretrained(args.dinov3_model_path)
    dino_model.eval()
    dino_model.to(device)

    # 多尺度层级异常分类器：每层独立 W_cls^l 和 W_seg^l，⟨W,F⟩/τ + softmax
    num_layers = len(args.features_list)
    anomaly_head = PerLayerAnomalyClassifier(
        vis_dim=args.vis_dim,
        num_classes=2,  # normal=0, abnormal=1
        num_layers=num_layers,
        init_layer_indices=args.features_list,
    ).to(device)

    return dino_processor, dino_model, anomaly_head


class PerLayerAnomalyClassifier(nn.Module):
    """
    多尺度层级异常分类器，对应公式：
      y_zl = softmax(⟨W_cls^l, x_q^l⟩ / τ)   # 图像级
      Y_zl = softmax(⟨W_seg^l, F_q^l⟩ / τ)   # 像素级
    每层独立的 W_cls^l 和 W_seg^l，不同层嵌入不同流形。
    """

    def __init__(self, vis_dim: int, num_classes: int, num_layers: int, init_layer_indices=None):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 每层独立的分类器：W_cls^l, W_seg^l，⟨W,x⟩ 即 Linear 的 weight @ x（bias=False）
        self.W_cls = nn.ModuleList([
            nn.Linear(vis_dim, num_classes, bias=False) for _ in range(num_layers)
        ])
        self.W_seg = nn.ModuleList([
            nn.Linear(vis_dim, num_classes, bias=False) for _ in range(num_layers)
        ])

        # 层间融合权重（softmax）
        self.layer_logits = nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))
        if init_layer_indices is not None and len(init_layer_indices) == num_layers:
            init_vals = torch.tensor(init_layer_indices, dtype=torch.float32)
            init_vals = init_vals - init_vals.mean()
            with torch.no_grad():
                self.layer_logits.copy_(init_vals)

    def forward(self, multi_layer_features, temperature: float, return_per_layer: bool = False):
        """
        Args:
            multi_layer_features: list of [B, 1+P, D]，每层的 (CLS + patch) 特征
            temperature: τ
            return_per_layer: 是否返回每层的 anomaly 图（用于可视化）

        Returns:
            cls_anomaly: [B] 图像级异常分数
            seg_anomaly: [B, P] 像素级异常分数
            per_layer_dict: (optional) 每层的 cls/seg anomaly
        """
        L = len(multi_layer_features)
        layer_logits_cls = []
        layer_logits_seg = []

        for li in range(L):
            feat = multi_layer_features[li]  # [B, 1+P, D]
            x_cls = feat[:, 0, :]            # [B, D]
            F_patch = feat[:, 1:, :]         # [B, P, D]

            # ⟨W_cls^l, x_q^l⟩ / τ
            logits_cls = self.W_cls[li](x_cls) / temperature   # [B, num_classes]
            # ⟨W_seg^l, F_q^l⟩ / τ
            B, P, D = F_patch.shape
            logits_seg = self.W_seg[li](F_patch.reshape(B * P, D)).reshape(B, P, -1) / temperature  # [B, P, num_classes]

            layer_logits_cls.append(logits_cls)
            layer_logits_seg.append(logits_seg)

        # softmax → 取 abnormal 类 (index 1)
        probs_cls = [F.softmax(lc, dim=-1)[:, 1] for lc in layer_logits_cls]   # list of [B]
        probs_seg = [F.softmax(ls, dim=-1)[:, :, 1] for ls in layer_logits_seg]  # list of [B, P]

        # 层权重
        weights = F.softmax(self.layer_logits[:L], dim=0)  # [L]

        # 加权融合
        cls_anomaly = sum(w * p for w, p in zip(weights, probs_cls))   # [B]
        seg_stack = torch.stack(probs_seg, dim=0) * weights.view(L, 1, 1)  # [L,B,P]
        seg_anomaly = seg_stack.sum(dim=0)  # [B, P]

        if return_per_layer:
            return cls_anomaly, seg_anomaly, {"cls": probs_cls, "seg": probs_seg}
        return cls_anomaly, seg_anomaly


class ResidualVisualProjection(nn.Module):
    """
    在原有 MLP 投影层外加一条线性跳跃连接：
        y = MLP(x) + W_skip x
    其中 MLP 使用 utils.prompt_generator.VisualProjection，
    W_skip 是一个 vis_dim -> output_dim 的线性层。
    """

    def __init__(self, vis_dim: int, output_dim: int):
        super().__init__()
        # 原始 MLP
        self.base = prompt_generator.VisualProjection(
            vis_dim=vis_dim, output_dim=output_dim
        )
        # 跳跃连接：1x1 线性映射，保证维度一致后再相加
        self.skip = nn.Linear(vis_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, vis_dim]
        return: [B, output_dim]
        """
        return self.base(x) + self.skip(x)


class MultiLayerPatchProjection(nn.Module):
    """
    针对 DINO 多层 patch 特征的投影模块。

    - 每一层都有一套独立的 ResidualVisualProjection 参数；
    - 前向时对每一层分别投影到 CLIP 维度并归一化；
    - 然后通过一组可学习的层权重（softmax）对各层特征做加权求和。
    """

    def __init__(self, vis_dim: int, output_dim: int, num_layers: int, init_layer_indices=None):
        super().__init__()
        self.num_layers = num_layers
        # 为每一层创建独立的投影子网络
        self.layer_projs = nn.ModuleList(
            [ResidualVisualProjection(vis_dim, output_dim) for _ in range(num_layers)]
        )

        # 可学习的层权重 logits
        self.layer_logits = nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))

        # 用 layer_indices 初始化权重（深层初始权重点更大）
        if init_layer_indices is not None and len(init_layer_indices) == num_layers:
            init_vals = torch.tensor(init_layer_indices, dtype=torch.float32)
            # 标准化到零均值，避免初始 softmax 过度偏置
            init_vals = init_vals - init_vals.mean()
            with torch.no_grad():
                self.layer_logits.copy_(init_vals)

    def forward(self, layer_patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer_patches: [L, B, P, D]，来自多层的 patch token（已去掉 CLS）

        Returns:
            fused_patches: [B, P, output_dim]，多层加权融合后的 patch 特征
        """
        L, B, P, D = layer_patches.shape
        assert L == self.num_layers, f"Expected {self.num_layers} layers, but got {L}"

        layer_outputs = []
        for li in range(L):
            x = layer_patches[li].reshape(B * P, D)          # [B*P, D]
            y = self.layer_projs[li](x)                      # [B*P, output_dim]
            y = F.normalize(y, dim=-1)                       # 逐 token 归一化
            y = y.view(B, P, -1)                             # [B, P, output_dim]
            layer_outputs.append(y)

        # [L, B, P, output_dim]
        layer_outputs = torch.stack(layer_outputs, dim=0)

        # softmax 层权重：[L] -> [L,1,1,1]
        weights = F.softmax(self.layer_logits, dim=0).view(L, 1, 1, 1)

        # 加权求和，得到 [B,P,output_dim]
        fused = (layer_outputs * weights).sum(dim=0)
        return fused


def build_normal_memory_bank(
    shot_paths: List[str],
    dino_processor,
    dino_model,
    device,
    image_size: int,
    features_list: List[int],
) -> List[torch.Tensor]:
    """
    构建多尺度正常内存库 M^l：存储 K 张正常图像的各层 patch 特征。
    返回: memory_bank[l] = [N, D]，N = K*P，D 为特征维，已 L2 归一化。
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    all_patches_per_layer = [[] for _ in features_list]

    with torch.no_grad():
        for path in shot_paths:
            img = Image.open(path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)
            dino_out = dinov3_encode_image(
                img_t, dino_processor, dino_model,
                device=device, layer_indices=features_list
            )
            if "multi_layer_features" not in dino_out:
                raise RuntimeError("multi_layer_features required.")
            mlf = dino_out["multi_layer_features"]
            for li, feat in enumerate(mlf):
                # feat: [1, 1+P, D]，取 patch 去掉 CLS
                F_patch = feat[:, 1:, :]  # [1, P, D]
                F_flat = F_patch.reshape(-1, feat.shape[-1])  # [P, D]
                all_patches_per_layer[li].append(F_flat)

    memory_bank = []
    for li in range(len(features_list)):
        M_l = torch.cat(all_patches_per_layer[li], dim=0)  # [K*P, D]
        M_l = F.normalize(M_l, dim=-1)
        memory_bank.append(M_l)
    return memory_bank


def train_one_shot(
    args,
    anomaly_head: nn.Module,
    dino_processor,
    dino_model,
    device,
    cls_name: str,
):
    """用若干正常图做 one-shot 拟合，训练每层独立的 W_cls^l 和 W_seg^l，使 normal 的 p_abnormal 最小。"""
    # 1. 准备 one-shot 正常样本
    normal_dir = os.path.join(args.data_path, cls_name, "train", "good")
    normal_files = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
    if len(normal_files) == 0:
        raise RuntimeError(f"No normal images found in {normal_dir}")

    n_shot = max(1, args.n_shot)
    shot_paths = normal_files[:n_shot]
    print(f"Using {len(shot_paths)} normal image(s) for one-shot training:")
    for p in shot_paths:
        print(f"  - {p}")

    train_dataset = OneShotNormalDataset(shot_paths, args.image_size)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0
    )

    # 2. 只训练 anomaly_head，冻结 DINO
    optimizer = torch.optim.AdamW(anomaly_head.parameters(), lr=args.lr, weight_decay=1e-4)

    anomaly_head.train()
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in train_loader:
            img = batch["img"].to(device)

            with torch.no_grad():
                dino_out = dinov3_encode_image(
                    img, dino_processor, dino_model,
                    device=device, layer_indices=args.features_list
                )
            if "multi_layer_features" not in dino_out:
                raise RuntimeError("multi_layer_features required. Set layer_indices in dinov3_encode_image.")

            mlf = [f.clone().detach() for f in dino_out["multi_layer_features"]]

            # 前向：y_zl = softmax(⟨W_cls^l, x_q^l⟩/τ), Y_zl = softmax(⟨W_seg^l, F_q^l⟩/τ)
            cls_anomaly, seg_anomaly = anomaly_head(mlf, args.temperature, return_per_layer=False)

            # 损失：normal 样本的 p_abnormal 应最小化
            cls_loss = cls_anomaly.mean()
            patch_loss = seg_anomaly.mean()
            loss = cls_loss + patch_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        print(
            f"[One-shot] Epoch {epoch+1}/{args.epochs} - "
            f"Loss: {np.mean(epoch_losses):.6f}"
        )

    anomaly_head.eval()

    # 构建多尺度正常内存库：存储 K 张正常图各层 patch 特征
    print("[One-shot] Building normal memory bank...")
    memory_bank = build_normal_memory_bank(
        shot_paths, dino_processor, dino_model, device,
        args.image_size, args.features_list,
    )
    for li, M in enumerate(memory_bank):
        print(f"  Layer {args.features_list[li]}: M^l shape {M.shape}")

    return anomaly_head, memory_bank


def compute_few_shot_anomaly(
    mlf: List[torch.Tensor],
    memory_bank: List[torch.Tensor],
    device,
) -> torch.Tensor:
    """
    计算 Few-Shot 异常分数：Y^f_l(i,j) = min_{m in M^l} (1 - ⟨F^l_q(i,j), m⟩)
    多尺度聚合：Y^f = (1/|L|) * sum_l Y^f_l
    返回: [B, P]
    """
    L = len(mlf)
    Y_f_l_list = []
    for li in range(L):
        F_patch = mlf[li][:, 1:, :]  # [B, P, D]，去掉 CLS
        M_l = memory_bank[li].to(device)  # [N, D]
        B, P, D = F_patch.shape
        # ⟨F, M⟩: [B,P,D] @ [D,N] -> [B,P,N]，余弦相似度（均已归一化）
        F_flat = F_patch.reshape(B * P, D)
        sim = F_flat @ M_l.T  # [B*P, N]
        max_sim = sim.max(dim=-1)[0].reshape(B, P)  # 最近邻相似度
        Y_f_l = 1.0 - max_sim  # 余弦距离 = 1 - cos_sim
        Y_f_l_list.append(Y_f_l)
    Y_f = torch.stack(Y_f_l_list, dim=0).mean(dim=0)  # [B, P]
    return Y_f


def evaluate_bottle(
    args,
    anomaly_head: nn.Module,
    memory_bank: List[torch.Tensor],
    dino_processor,
    dino_model,
    device,
    obj_name: str,
):
    """在指定类别的 test 集上评估：Zero-Shot (W_cls/W_seg) + Few-Shot (memory) 融合。"""
    dataset = BottleTestDataset(args.data_path, obj_name, args.image_size)
    if len(dataset) == 0:
        raise RuntimeError(f"No test images found for {obj_name}.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_scores = []
    all_labels = []
    pixel_maps = []   # list of [B,H,W] anomaly maps
    gt_maps = []      # list of [B,H,W] gt masks
    all_anom_values = []  # 收集所有异常分数，用于计算可视化阈值

    with torch.no_grad():
        img_counter = 0  # 全局图片计数，用于可视化命名避免覆盖
        profiled = False

        for batch in tqdm(loader, desc="Evaluating bottle"):
            img = batch["img"].to(device)  # [B,3,H,W]
            label = batch["anomaly"].to(device)  # [B]
            gt_mask = batch["mask"].to(device)   # [B,1,H,W] in {0,1}
            img_paths = batch["img_path"]        # list of str

            # 可选：只在评估阶段的第一个 batch 做一次 profiling
            if (not profiled) and getattr(args, "profile_infer", False):
                profile_inference(args, img, anomaly_head, memory_bank, dino_processor, dino_model, device)
                profiled = True

            dino_out = dinov3_encode_image(
                img, dino_processor, dino_model,
                device=device, layer_indices=args.features_list
            )
            if "multi_layer_features" not in dino_out:
                raise RuntimeError("multi_layer_features required for evaluate_bottle.")

            mlf = dino_out["multi_layer_features"]  # list of [B,1+P,D]
            grid_h, grid_w = dino_out["grid_size"].tolist()

            # Zero-Shot: y^z (cls), Y^z (patch) 来自 W_cls/W_seg
            cls_anomaly_z, patch_anomaly_z, per_layer_dict = anomaly_head(
                mlf, args.temperature, return_per_layer=True
            )

            # Few-Shot: Y^f = (1/|L|)*sum_l Y^f_l，Y^f_l(i,j)=min_m(1-⟨F^l,m⟩)
            Y_f = compute_few_shot_anomaly(mlf, memory_bank, device)  # [B, P]

            # 融合 Zero-Shot 与 Few-Shot
            # Y^fused = (1-λ_f)*Y^z + λ_f*Y^f
            patch_anomaly = (1.0 - args.lambda_f) * patch_anomaly_z + args.lambda_f * Y_f
            # y^f = (1-λ_p)*y^z + λ_p*max(Y^fused)
            y_f = (1.0 - args.lambda_p) * cls_anomaly_z + args.lambda_p * patch_anomaly.max(dim=1)[0]

            # 图像级分数、像素级异常图
            img_score = y_f  # [B]

            # 构建每层异常图用于可视化 [L,B,H,W]
            per_layer_anom_maps = []
            for seg_l in per_layer_dict["seg"]:
                # seg_l: [B, P]
                b = seg_l.shape[0]
                map_l = seg_l.view(b, grid_h, grid_w)
                up_l = F.interpolate(
                    map_l.unsqueeze(1),
                    size=(args.image_size, args.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                per_layer_anom_maps.append(up_l)
            per_layer_anom_maps = torch.stack(per_layer_anom_maps, dim=0)  # [L,B,H,W]

            # 记录 image-level
            all_scores.append(img_score.cpu().numpy())
            all_labels.append(label.cpu().numpy())

            # 记录 pixel-level：将 patch 异常度恢复成 [B, H_g, W_g] 再上采样到图像大小
            b = patch_anomaly.shape[0]
            patch_anomaly_map = patch_anomaly.view(b, grid_h, grid_w)  # [B,Hg,Wg]
            anomaly_up = F.interpolate(
                patch_anomaly_map.unsqueeze(1),  # [B,1,Hg,Wg]
                size=(args.image_size, args.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # [B,H,W]

            pixel_maps.append(anomaly_up.cpu().numpy())
            gt_maps.append(gt_mask.squeeze(1).cpu().numpy())  # [B,H,W]

            # === 单图可视化：原图 + GT + 预测异常图 + Overlay（可通过 --no_vis 关闭）===
            if not getattr(args, "no_vis", False):
                Y_f_map = Y_f.view(b, grid_h, grid_w)
                Y_f_up = F.interpolate(
                    Y_f_map.unsqueeze(1),
                    size=(args.image_size, args.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)  # [B,H,W]

                vis_root = getattr(args, "save_dir", "./results/one_shot_bottle")
                vis_dir = os.path.join(vis_root, obj_name, "per_image_vis")
                os.makedirs(vis_dir, exist_ok=True)

                img_np = img.cpu().numpy()  # [B,3,H,W]
                gt_np = gt_mask.squeeze(1).cpu().numpy()  # [B,H,W]
                anom_np = anomaly_up.cpu().numpy()        # [B,H,W]
                Y_f_np = Y_f_up.cpu().numpy()             # [B,H,W] Few-Shot only
                per_layer_np = per_layer_anom_maps.cpu().numpy() if per_layer_anom_maps is not None else None

                for i in range(b):
                    # 还原到 [H,W,3]，反标准化到 [0,1]
                    img_i = img_np[i].transpose(1, 2, 0)
                    img_i = np.clip(
                        img_i * np.array([0.229, 0.224, 0.225])[None, None, :]
                        + np.array([0.485, 0.456, 0.406])[None, None, :],
                        0, 1
                    )
                    gt_i = gt_np[i]          # [H,W]
                    anom_i = anom_np[i]      # [H,W]

                    # 仅用于可视化的 per-image 归一化
                    anom_vis = anom_i.copy()
                    min_v, max_v = float(anom_vis.min()), float(anom_vis.max())
                    if max_v > min_v:
                        anom_vis = (anom_vis - min_v) / (max_v - min_v)
                    else:
                        anom_vis = np.zeros_like(anom_vis)

                    base = os.path.splitext(os.path.basename(img_paths[i]))[0]
                    out_name_base = f"{img_counter:05d}_{base}"
                    img_counter += 1

                    # ① 归一化后的可视化（对比更明显）
                    out_path_norm = os.path.join(vis_dir, f"{out_name_base}_vis_norm.png")
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 4, 1)
                    plt.imshow(img_i)
                    plt.title("Image\n" + ("Abnormal" if int(label[i].item()) == 1 else "Normal"))
                    plt.axis("off")
                    plt.subplot(1, 4, 2)
                    plt.imshow(gt_i, cmap="gray", vmin=0, vmax=1)
                    plt.title("GT mask")
                    plt.axis("off")
                    plt.subplot(1, 4, 3)
                    im = plt.imshow(anom_vis, cmap="coolwarm", vmin=0, vmax=1)
                    plt.title("Anomaly map (norm)")
                    plt.axis("off")
                    plt.colorbar(im, fraction=0.046, pad=0.04)
                    plt.subplot(1, 4, 4)
                    plt.imshow(img_i)
                    plt.imshow(anom_vis, cmap="coolwarm", vmin=0, vmax=1, alpha=0.5)
                    plt.title("Overlay (norm)")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(out_path_norm, dpi=150, bbox_inches="tight")
                    plt.close()

                    # ② 不做 per-image 归一化的可视化（直接使用原始 anomaly 值）
                    out_path_raw = os.path.join(vis_dir, f"{out_name_base}_vis_raw.png")
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 4, 1)
                    plt.imshow(img_i)
                    plt.title("Image\n" + ("Abnormal" if int(label[i].item()) == 1 else "Normal"))
                    plt.axis("off")
                    plt.subplot(1, 4, 2)
                    plt.imshow(gt_i, cmap="gray", vmin=0, vmax=1)
                    plt.title("GT mask")
                    plt.axis("off")
                    plt.subplot(1, 4, 3)
                    im2 = plt.imshow(anom_i, cmap="coolwarm", vmin=0, vmax=float(anom_i.max()))
                    plt.title("Anomaly map (raw)")
                    plt.axis("off")
                    plt.colorbar(im2, fraction=0.046, pad=0.04)
                    plt.subplot(1, 4, 4)
                    plt.imshow(img_i)
                    plt.imshow(anom_i, cmap="coolwarm", vmin=0, vmax=1, alpha=0.5)
                    plt.title("Overlay (raw)")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(out_path_raw, dpi=150, bbox_inches="tight")
                    plt.close()

                    # ③ Few-Shot (memory bank) 单独可视化
                    Y_f_i = Y_f_np[i]  # [H,W]
                    Y_f_vis = Y_f_i.copy()
                    yf_min, yf_max = float(Y_f_vis.min()), float(Y_f_vis.max())
                    if yf_max > yf_min:
                        Y_f_vis = (Y_f_vis - yf_min) / (yf_max - yf_min)
                    else:
                        Y_f_vis = np.zeros_like(Y_f_vis)

                    out_path_fewshot = os.path.join(vis_dir, f"{out_name_base}_fewshot.png")
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 4, 1)
                    plt.imshow(img_i)
                    plt.title("Image\n" + ("Abnormal" if int(label[i].item()) == 1 else "Normal"))
                    plt.axis("off")
                    plt.subplot(1, 4, 2)
                    plt.imshow(gt_i, cmap="gray", vmin=0, vmax=1)
                    plt.title("GT mask")
                    plt.axis("off")
                    plt.subplot(1, 4, 3)
                    im_f = plt.imshow(Y_f_vis, cmap="coolwarm", vmin=0, vmax=1)
                    plt.title("Few-Shot (memory bank)")
                    plt.axis("off")
                    plt.colorbar(im_f, fraction=0.046, pad=0.04)
                    plt.subplot(1, 4, 4)
                    plt.imshow(img_i)
                    plt.imshow(Y_f_vis, cmap="coolwarm", vmin=0, vmax=1, alpha=0.5)
                    plt.title("Overlay (Few-Shot)")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(out_path_fewshot, dpi=150, bbox_inches="tight")
                    plt.close()

                    # ④ 每一层的异常图可视化（仅当有 multi-layer 特征时）
                    if per_layer_np is not None:
                        L_layers = per_layer_np.shape[0]
                        out_path_layers = os.path.join(vis_dir, f"{out_name_base}_layers.png")
                        n_cols = L_layers + 1
                        plt.figure(figsize=(4 * n_cols, 4))
                        plt.subplot(1, n_cols, 1)
                        plt.imshow(img_i)
                        plt.title("Image")
                        plt.axis("off")
                        for li in range(L_layers):
                            layer_map = per_layer_np[li, i]  # [H,W]
                            lm = layer_map.copy()
                            lm_min, lm_max = float(lm.min()), float(lm.max())
                            if lm_max > lm_min:
                                lm = (lm - lm_min) / (lm_max - lm_min)
                            else:
                                lm = np.zeros_like(lm)
                            plt.subplot(1, n_cols, li + 2)
                            plt.imshow(img_i)
                            plt.imshow(lm, cmap="coolwarm", vmin=0, vmax=1, alpha=0.5)
                            plt.title(f"Layer {args.features_list[li]}")
                            plt.axis("off")
                        plt.tight_layout()
                        plt.savefig(out_path_layers, dpi=150, bbox_inches="tight")
                        plt.close()

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    n_images = len(all_labels)

    if len(np.unique(all_labels)) < 2:
        print("Only one class in labels, AUROC undefined.")
        return

    t0 = time.perf_counter()
    pixel_maps_np = np.concatenate(pixel_maps, axis=0)  # [N,H,W]
    gt_maps_np = np.concatenate(gt_maps, axis=0)        # [N,H,W]

    # 与 main_mvtec 一致的 results 格式
    results = {
        obj_name: {
            "gt_sp": all_labels,
            "pr_sp": all_scores,
            "imgs_masks": gt_maps_np,
            "anomaly_maps": pixel_maps_np,
        }
    }

    # image-level metrics（与 main_mvtec 一致）
    print(f"[Metrics] Computing on {n_images} images...")
    print("  [1/5] Image AUROC...", end=" ", flush=True)
    t1 = time.perf_counter()
    image_auroc = image_level_metrics(results, obj_name, "image-auroc")
    print(f"{image_auroc:.4f} ({time.perf_counter() - t1:.1f}s)")

    print("  [2/5] Image AP...", end=" ", flush=True)
    t2 = time.perf_counter()
    image_ap = image_level_metrics(results, obj_name, "image-ap")
    print(f"{image_ap:.4f} ({time.perf_counter() - t2:.1f}s)")

    print("  [3/5] Pixel AUROC...", end=" ", flush=True)
    t3 = time.perf_counter()
    pixel_auroc = pixel_level_metrics(results, obj_name, "pixel-auroc")
    print(f"{pixel_auroc:.4f} ({time.perf_counter() - t3:.1f}s)")

    print("  [4/5] Pixel AUPRO...", end=" ", flush=True)
    t4 = time.perf_counter()
    pixel_aupro = pixel_level_metrics(results, obj_name, "pixel-aupro")
    print(f"{pixel_aupro:.4f} ({time.perf_counter() - t4:.1f}s)")

    pixel_scores_flat = pixel_maps_np.reshape(-1)
    pixel_labels_flat = gt_maps_np.reshape(-1)
    print(f"[Metrics] Total: {time.perf_counter() - t0:.1f}s")

    # 算完立即打印结果
    print("\n" + "=" * 80)
    print(f"One-shot evaluation ({obj_name})")
    print(f"{'objects':>10} | {'image_auroc':>11} | {'image_ap':>9} | {'pixel_aupro':>12} | {'pixel_auroc':>12}")
    print("-" * 80)
    print(
        f"{obj_name:>10} | "
        f"{image_auroc:11.4f} | "
        f"{image_ap:9.4f} | "
        f"{pixel_aupro:12.4f} | "
        f"{pixel_auroc:12.4f}"
    )
    print("-" * 80)
    print(f"Normal scores:  mean={all_scores[all_labels==0].mean():.4f}, "
          f"std={all_scores[all_labels==0].std():.4f}, n={len(all_scores[all_labels==0])}")
    print(f"Abnormal scores: mean={all_scores[all_labels==1].mean():.4f}, "
          f"std={all_scores[all_labels==1].std():.4f}, n={len(all_scores[all_labels==1])}")
    print("=" * 80 + "\n")

    # 创建保存目录
    save_root = getattr(args, "save_dir", "./results/one_shot_bottle")
    obj_dir = os.path.join(save_root, obj_name)
    os.makedirs(obj_dir, exist_ok=True)

    print("  [5/5] Saving results + aggregate plots...", end=" ", flush=True)
    t4 = time.perf_counter()
    # 保存指标到 CSV
    metrics_csv = os.path.join(obj_dir, "metrics.csv")
    with open(metrics_csv, "w") as f:
        f.write("objects,image_auroc,image_ap,pixel_aupro,pixel_auroc\n")
        f.write(f"{obj_name},{image_auroc:.6f},{image_ap:.6f},{pixel_aupro:.6f},{pixel_auroc:.6f}\n")

    # aggregate 可视化（始终保存）
    # 1) 图像级分数分布
    normal_scores = all_scores[all_labels == 0]
    abnormal_scores = all_scores[all_labels == 1]
    plt.figure(figsize=(8, 5))
    if len(normal_scores) > 0:
        plt.hist(normal_scores, bins=30, alpha=0.5, label="Normal", color="green", density=True)
    if len(abnormal_scores) > 0:
        plt.hist(abnormal_scores, bins=30, alpha=0.5, label="Abnormal", color="red", density=True)
    plt.xlabel("Image anomaly score")
    plt.ylabel("Density")
    plt.title(f"{obj_name} - image scores\nAUROC={image_auroc:.4f}, AP={image_ap:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(obj_dir, "image_score_distribution.png"), dpi=150)
    plt.close()

    # 2) 图像级 ROC & PR 曲线
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    prec, rec, _ = precision_recall_curve(all_labels, all_scores)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC={image_auroc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"Image-level ROC ({obj_name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(rec, prec, label=f"AP={image_ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Image-level PR ({obj_name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(obj_dir, "image_roc_pr.png"), dpi=150)
    plt.close()

    # 3) 像素级 ROC 曲线
    if len(np.unique(pixel_labels_flat)) >= 2:
        fpr_p, tpr_p, _ = roc_curve(pixel_labels_flat, pixel_scores_flat)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr_p, tpr_p, label=f"AUC={pixel_auroc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"Pixel-level ROC ({obj_name})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(obj_dir, "pixel_roc.png"), dpi=150)
        plt.close()
    print(f"done ({time.perf_counter() - t4:.1f}s)")


def main(args):
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 指定要实验的 MVTec 类别
    obj_name = args.class_name

    # 构建模型
    dino_processor, dino_model, anomaly_head = build_models(args, device)

    # 一/少样本拟合：只看 normal 图（默认一张），训练每层 W_cls^l 和 W_seg^l，并构建正常内存库
    anomaly_head, memory_bank = train_one_shot(
        args, anomaly_head, dino_processor, dino_model, device, obj_name
    )

    # 在该类别的 test 集上做异常检测验证（Zero-Shot + Few-Shot 融合）
    evaluate_bottle(args, anomaly_head, memory_bank, dino_processor, dino_model, device, obj_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("One-shot projection on MVTec", add_help=True)

    parser.add_argument(
        "--data_path",
        type=str,
        default="/data2/zlt/code/abnormal_dataset/mvtec",
        help="path to MVTec dataset",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="bottle",
        help="MVTec object class name, e.g. bottle, transistor, capsule, ...",
    )
    parser.add_argument(
        "--dinov3_model_path",
        type=str,
        default="./model_card/dinov3-vitl16-pretrain-lvd1689m",
        help="path to DINOv3 model",
    )

    # model / feature config
    parser.add_argument("--vis_dim", type=int, default=1024, help="DINOv3 feature dim")
    parser.add_argument(
        "--features_list",
        type=int,
        nargs="+",
        default=[12, 15, 18, 21, 24],
        help="DINOv3 block indices for multi-scale features (per-layer W_cls and W_seg)",
    )
    parser.add_argument("--image_size", type=int, default=512)

    # one-shot training hyper-params
    parser.add_argument("--n_shot", type=int, default=1, help="number of normal images to use")
    parser.add_argument("--epochs", type=int, default=100, help="epochs for one-shot fitting")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for proj")

    # eval & visualization
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="temperature for similarity -> probability when computing anomaly scores")
    parser.add_argument("--lambda_f", type=float, default=0.5,
                        help="fusion weight for few-shot pixel map: Y=(1-λ_f)*Y_z+λ_f*Y_f")
    parser.add_argument("--lambda_p", type=float, default=0.5,
                        help="fusion weight for image score: y=(1-λ_p)*y_z+λ_p*max(Y_fused)")
    parser.add_argument("--save_dir", type=str, default="./results/one_shot_bottle",
                        help="directory to save evaluation results and visualizations")
    parser.add_argument("--no_vis", action="store_true",
                        help="disable per-image visualization (vis_norm, vis_raw, fewshot, layers); aggregate plots always saved")
    parser.add_argument("--profile_infer", action="store_true",
                        help="profile inference FLOPs/time on the first eval batch (DINO vs head vs combined)")
    parser.add_argument("--profile_warmup", type=int, default=5,
                        help="warmup iterations for inference profiling")
    parser.add_argument("--profile_iters", type=int, default=20,
                        help="timed iterations for inference profiling (average per batch)")

    args = parser.parse_args()
    print(args)
    main(args)

