"""
VVCLIP 可视化工具
用于排查和调试 VVCLIP 模型的异常检测结果
"""

# import cv2  # 已替换为PIL
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image
import seaborn as sns
from typing import List, Dict, Tuple, Optional

class Visualizer:
    """VVCLIP 异常检测可视化器"""
    
    def __init__(self, save_path: str, img_size: int = 224):
        self.save_path = save_path
        self.img_size = img_size
        self.colormap = cm.get_cmap('jet')
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, 'anomaly_maps'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'feature_maps'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'feature_overlays'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'attention_maps'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'comparisons'), exist_ok=True)
    
    def normalize(self, tensor: torch.Tensor) -> np.ndarray:
        """归一化张量到 [0, 1] 范围"""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        # 确保是 numpy 数组
        if isinstance(tensor, np.ndarray):
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            if tensor_max - tensor_min > 0:
                return (tensor - tensor_min) / (tensor_max - tensor_min)
            else:
                return np.zeros_like(tensor)
        return tensor
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """将张量转换为 numpy 数组"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def save_anomaly_map_overlay(self, 
                                original_image: torch.Tensor, 
                                anomaly_map: torch.Tensor, 
                                gt_mask: Optional[torch.Tensor] = None,
                                filename: str = "anomaly_map",
                                alpha: float = 0.6) -> str:
        """保存异常图叠加原图的可视化"""
        
        # 转换张量为 numpy
        img_np = self.tensor_to_numpy(original_image)
        anomaly_np = self.tensor_to_numpy(anomaly_map)
        
        # 处理批次维度
        if img_np.ndim == 4:
            img_np = img_np[0]
        if anomaly_np.ndim == 4:
            anomaly_np = anomaly_np[0]
        
        # 处理通道维度 - 假设是 [C, H, W] 格式
        if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))
        
        # 处理异常图的维度
        if anomaly_np.ndim == 3:
            if anomaly_np.shape[0] == 1:  # [1, H, W] -> [H, W]
                anomaly_np = anomaly_np[0]
            elif anomaly_np.shape[-1] == 1:  # [H, W, 1] -> [H, W]
                anomaly_np = anomaly_np[:, :, 0]
        
        # 确保异常图是2D的
        if anomaly_np.ndim != 2:
            print(f"Warning: anomaly_map shape {anomaly_np.shape} is not 2D, taking first channel")
            anomaly_np = anomaly_np.reshape(-1)[:self.img_size*self.img_size].reshape(self.img_size, self.img_size)
        
        # 归一化异常图到 [0, 1]
        anomaly_np = self.normalize(anomaly_np)
        
        # 调整图像尺寸（使用PIL替代cv2）
        if img_np.shape[:2] != (self.img_size, self.img_size):
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil = img_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
            img_np = np.array(img_pil) / 255.0
        
        if anomaly_np.shape != (self.img_size, self.img_size):
            anomaly_pil = Image.fromarray((anomaly_np * 255).astype(np.uint8))
            anomaly_pil = anomaly_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
            anomaly_np = np.array(anomaly_pil) / 255.0
        
        # 修复图像归一化 - 使用正确的CLIP归一化方法
        # CLIP图像通常在 [-1, 1] 范围内，需要转换到 [0, 1]
        if img_np.min() < 0:
            # 如果是 [-1, 1] 范围，转换到 [0, 1]
            img_np = (img_np + 1.0) / 2.0
        elif img_np.max() > 1.0:
            # 如果是 [0, 255] 范围，转换到 [0, 1]
            img_np = img_np / 255.0
        
        # 确保图像数据在 [0, 1] 范围内
        img_np = np.clip(img_np, 0.0, 1.0)
        
        print(f"    - 图像范围: [{img_np.min():.3f}, {img_np.max():.3f}]")
        
        # 如果图像是单通道，转换为三通道
        if len(img_np.shape) == 2 or img_np.shape[-1] == 1:
            if len(img_np.shape) == 2:
                img_np = np.stack([img_np, img_np, img_np], axis=-1)
            else:
                img_np = np.repeat(img_np, 3, axis=-1)
        
        # 创建热图
        heatmap = self.colormap(anomaly_np)[:, :, :3]  # 去掉 alpha 通道
        
        # 叠加原图和热图
        overlay = alpha * img_np + (1 - alpha) * heatmap
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 异常热图
        im1 = axes[1].imshow(anomaly_np, cmap='jet')
        axes[1].set_title('Anomaly Heatmap')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 叠加图
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (α=0.6)')
        axes[2].axis('off')
        
        # 如果有真实标签，添加第四个子图
        if gt_mask is not None:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # 原图
            axes[0].imshow(img_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # 真实标签
            gt_np = self.tensor_to_numpy(gt_mask)
            # 处理GT mask的维度
            if gt_np.ndim == 4:
                gt_np = gt_np[0]
            if gt_np.ndim == 3:
                if gt_np.shape[0] == 1:
                    gt_np = gt_np[0]
                elif gt_np.shape[-1] == 1:
                    gt_np = gt_np[:, :, 0]
            
            # 确保GT mask是2D的
            if gt_np.ndim != 2:
                print(f"Warning: gt_mask shape {gt_np.shape} is not 2D")
                gt_np = gt_np.reshape(-1)[:self.img_size*self.img_size].reshape(self.img_size, self.img_size)
            
            # 调整GT mask尺寸（使用PIL）
            if gt_np.shape != (self.img_size, self.img_size):
                gt_pil = Image.fromarray((gt_np * 255).astype(np.uint8))
                gt_pil = gt_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
                gt_np = np.array(gt_pil) / 255.0
            
            axes[1].imshow(gt_np, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # 异常热图
            im2 = axes[2].imshow(anomaly_np, cmap='jet')
            axes[2].set_title('Anomaly Heatmap')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            # 叠加图
            axes[3].imshow(overlay)
            axes[3].set_title('Overlay')
            axes[3].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        save_file = os.path.join(self.save_path, 'anomaly_maps', f"{filename}.png")
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_file
    
    def visualize_feature_overlay(self, 
                                 original_image: torch.Tensor,
                                 patch_features: List[torch.Tensor], 
                                 layer_names: List[str],
                                 filename: str = "feature_overlay") -> str:
        """可视化feature map与原图的叠加热力图（参考用户提供的代码）"""
        
        if not patch_features:
            print("Warning: No patch features provided for overlay visualization")
            return ""
        
        print(f"  🎨 创建feature map叠加可视化...")
        
        # 处理原图
        img_np = self.tensor_to_numpy(original_image)
        if img_np.ndim == 4:
            img_np = img_np[0]
        if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))
        
        # 修复图像归一化
        if img_np.min() < 0:
            img_np = (img_np + 1.0) / 2.0
        elif img_np.max() > 1.0:
            img_np = img_np / 255.0
        img_np = np.clip(img_np, 0.0, 1.0)
        
        # 确保是三通道
        if len(img_np.shape) == 2 or img_np.shape[-1] == 1:
            if len(img_np.shape) == 2:
                img_np = np.stack([img_np, img_np, img_np], axis=-1)
            else:
                img_np = np.repeat(img_np, 3, axis=-1)
        
        # 调整图像尺寸（使用PIL）
        if img_np.shape[:2] != (self.img_size, self.img_size):
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil = img_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
            img_np = np.array(img_pil) / 255.0
        
        print(f"    - 原图形状: {img_np.shape}, 范围: [{img_np.min():.3f}, {img_np.max():.3f}]")
        
        num_layers = len(patch_features)
        cols = min(4, num_layers)
        rows = (num_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, (patch_feature, layer_name) in enumerate(zip(patch_features, layer_names)):
            try:
                # 转换为 numpy
                feature_np = self.tensor_to_numpy(patch_feature)
                print(f"    - Layer {i}: {feature_np.shape}")
                
                # 处理维度
                if feature_np.ndim == 3:
                    feature_np = feature_np[0]  # [seq_len, hidden_dim]
                
                # 转换为 tensor 进行处理（参考用户代码）
                if isinstance(patch_feature, torch.Tensor):
                    feature_tensor = patch_feature
                else:
                    feature_tensor = torch.tensor(patch_feature)
                
                # 处理维度
                if feature_tensor.ndim == 3:
                    feature_tensor = feature_tensor[0]  # [seq_len, hidden_dim]
                
                # 计算特征的平均激活（类似用户代码中的注意力计算）
                feature_mean = torch.mean(feature_tensor, dim=-1)  # 沿特征维度平均
                
                # 使用用户代码的方法：尝试不同的配置
                seq_len = feature_mean.shape[0]
                
                # 根据实际图像尺寸计算正确的patch配置
                # 从DINOv3配置中获取patch_size，通常是16
                patch_size = 16  # DINOv3-ViT-L/16的默认patch_size
                
                # 计算不同图像尺寸下的patch数量
                patch_configs = []
                for img_size in [224, 336, 512]:
                    num_patches = img_size // patch_size
                    total_patches = num_patches * num_patches
                    expected_seq_len = total_patches + 1  # +1 for CLS token
                    patch_configs.append((patch_size, num_patches, expected_seq_len))
                
                # 特别处理512x512图像的情况
                # 512 // 16 = 32, 所以是32x32=1024个patches + 1个CLS token = 1025
                patch_configs.append((16, 32, 1025))  # 512x512图像
                
                # 添加一些常见的配置
                patch_configs.extend([
                    (14, 16, 16*16+1),  # patch_size=14, image_size=224 -> 16x16=256 patches
                    (14, 24, 24*24+1),  # patch_size=14, image_size=336 -> 24x24=576 patches
                    (32, 7, 7*7+1),     # patch_size=32, image_size=224 -> 7x7=49 patches
                ])
                
                patch_map = None
                patch_info = ""
                
                for patch_size, grid_size, expected_seq_len in patch_configs:
                    
                    if seq_len == expected_seq_len:
                        # 跳过CLS token (第一个元素)，取patch tokens
                        expected_patches = grid_size * grid_size  # 计算patch数量
                        patch_features_1d = feature_mean[1:expected_patches+1]  # 去掉CLS token
                        
                        # 参考用户代码：reshape为2D并准备插值
                        patch_map_2d = patch_features_1d.reshape(grid_size, grid_size)
                        patch_map = patch_map_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                        
                        patch_info = f"{grid_size}x{grid_size}"
                        print(f"    - 叠加匹配配置: patch_size={patch_size}, grid_size={grid_size}x{grid_size}")
                        break
                
                if patch_map is not None:
                    # 参考用户代码：使用F.interpolate进行上采样
                    # 上采样到原图尺寸（类似用户代码中的F.interpolate）
                    feature_map_upsampled = F.interpolate(
                        patch_map,
                        scale_factor=patch_size,  # 使用匹配的patch_size
                        mode="bilinear",
                        align_corners=False
                    ).squeeze()
                    
                    # 转换为numpy并归一化
                    feature_map_np = feature_map_upsampled.detach().cpu().numpy()
                    feature_map_np = (feature_map_np - feature_map_np.min()) / (feature_map_np.max() - feature_map_np.min() + 1e-8)
                    
                    # 叠加显示（类似用户代码）
                    axes[i].imshow(img_np)
                    axes[i].imshow(feature_map_np, cmap='jet', alpha=0.5)  # 使用用户代码中的alpha=0.5
                    axes[i].set_title(f'{layer_name} ({patch_info})')
                    axes[i].axis('off')
                    
                    print(f"    - ✅ Layer {i} 叠加成功 ({patch_info})")
                    
                else:
                    print(f"    - Warning: Cannot reshape seq_len {seq_len} to square for layer {i}")
                    # 尝试其他重塑方式
                    if seq_len > 0:
                        # 找到最接近正方形的因子分解
                        factors = []
                        for j in range(1, int(np.sqrt(seq_len)) + 1):
                            if seq_len % j == 0:
                                factors.append((j, seq_len // j))
                        
                        if factors:
                            # 选择最接近正方形的因子对
                            best_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
                            h, w = best_factor
                            
                            # 跳过CLS token（如果存在）
                            if seq_len > h * w:
                                # 有CLS token，跳过第一个元素
                                patch_features_1d = feature_mean[1:h*w+1]
                            else:
                                # 没有CLS token，直接使用
                                patch_features_1d = feature_mean[:h*w]
                            
                            # 使用tensor进行处理
                            patch_map_2d = patch_features_1d.reshape(h, w)
                            patch_map = patch_map_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                            
                            # 使用F.interpolate进行上采样
                            # 计算scale_factor
                            scale_factor = self.img_size // max(h, w)
                            if scale_factor < 1:
                                scale_factor = 1
                            
                            feature_map_upsampled = F.interpolate(
                                patch_map,
                                scale_factor=scale_factor,
                                mode="bilinear",
                                align_corners=False
                            ).squeeze()
                            
                            # 转换为numpy并归一化
                            feature_map_np = feature_map_upsampled.detach().cpu().numpy()
                            feature_map_np = (feature_map_np - feature_map_np.min()) / (feature_map_np.max() - feature_map_np.min() + 1e-8)
                            
                            # 叠加显示
                            axes[i].imshow(img_np)
                            axes[i].imshow(feature_map_np, cmap='jet', alpha=0.5)
                            axes[i].set_title(f'{layer_name} ({h}x{w})')
                            axes[i].axis('off')
                            print(f"    - ✅ Layer {i} 使用 {h}x{w} 重塑成功")
                        else:
                            # 显示原图
                            axes[i].imshow(img_np)
                            axes[i].set_title(f'{layer_name} (No Reshape)')
                            axes[i].axis('off')
                    else:
                        # 显示原图
                        axes[i].imshow(img_np)
                        axes[i].set_title(f'{layer_name} (Empty)')
                        axes[i].axis('off')
                    
            except Exception as e:
                print(f"    - Error processing layer {i}: {e}")
                # 显示原图
                axes[i].imshow(img_np)
                axes[i].set_title(f'{layer_name} (Error)')
                axes[i].axis('off')
        
        # 隐藏多余的子图
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        save_file = os.path.join(self.save_path, 'feature_overlays', f"{filename}.png")
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Feature叠加图已保存: {save_file}")
        return save_file
    
    def visualize_feature_maps(self, 
                              patch_features: List[torch.Tensor], 
                              layer_names: List[str],
                              filename: str = "feature_maps") -> str:
        """可视化多层特征图（参考用户提供的代码实现）"""
        
        if not patch_features:
            print("Warning: No patch features provided for visualization")
            return ""
        
        num_layers = len(patch_features)
        print(f"  📊 可视化 {num_layers} 层特征图...")
        
        fig, axes = plt.subplots(2, num_layers, figsize=(4*num_layers, 8))
        
        if num_layers == 1:
            axes = axes.reshape(2, 1)
        
        for i, (patch_feature, layer_name) in enumerate(zip(patch_features, layer_names)):
            try:
                # 转换为 tensor 进行处理（参考用户代码）
                if isinstance(patch_feature, torch.Tensor):
                    feature_tensor = patch_feature
                else:
                    feature_tensor = torch.tensor(patch_feature)
                
                print(f"    - Layer {i}: {feature_tensor.shape}")
                
                # 处理维度
                if feature_tensor.ndim == 3:
                    feature_tensor = feature_tensor[0]  # [seq_len, hidden_dim]
                
                # 计算特征的平均激活（类似用户代码中的注意力计算）
                feature_mean = torch.mean(feature_tensor, dim=-1)  # 沿特征维度平均
                
                # 使用用户代码的方法：尝试不同的配置
                seq_len = feature_mean.shape[0]
                print(f"    - Seq len: {seq_len}")
                
                # 根据实际图像尺寸计算正确的patch配置
                # 从DINOv3配置中获取patch_size，通常是16
                patch_size = 16  # DINOv3-ViT-L/16的默认patch_size
                
                # 计算不同图像尺寸下的patch数量
                patch_configs = []
                for img_size in [224, 336, 512]:
                    num_patches = img_size // patch_size
                    total_patches = num_patches * num_patches
                    expected_seq_len = total_patches + 1  # +1 for CLS token
                    patch_configs.append((patch_size, num_patches, expected_seq_len))
                
                # 特别处理512x512图像的情况
                # 512 // 16 = 32, 所以是32x32=1024个patches + 1个CLS token = 1025
                patch_configs.append((16, 32, 1025))  # 512x512图像
                
                # 添加一些常见的配置
                patch_configs.extend([
                    (14, 16, 16*16+1),  # patch_size=14, image_size=224 -> 16x16=256 patches
                    (14, 24, 24*24+1),  # patch_size=14, image_size=336 -> 24x24=576 patches
                    (32, 7, 7*7+1),     # patch_size=32, image_size=224 -> 7x7=49 patches
                ])
                
                patch_map = None
                patch_info = ""
                
                for patch_size, grid_size, expected_seq_len in patch_configs:
                    
                    if seq_len == expected_seq_len:
                        # 跳过CLS token (第一个元素)，取patch tokens
                        expected_patches = grid_size * grid_size  # 计算patch数量
                        patch_features_1d = feature_mean[1:expected_patches+1]  # 去掉CLS token
                        
                        # 参考用户代码：reshape为2D并准备插值
                        patch_map_2d = patch_features_1d.reshape(grid_size, grid_size)
                        patch_map = patch_map_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                        
                        patch_info = f"patch_{patch_size}_grid_{grid_size}x{grid_size}"
                        print(f"    - 匹配配置: patch_size={patch_size}, grid_size={grid_size}x{grid_size}")
                        break
                
                if patch_map is not None:
                    # 参考用户代码：使用F.interpolate进行上采样
                    # 上采样到原图尺寸（类似用户代码中的F.interpolate）
                    feature_map_upsampled = F.interpolate(
                        patch_map,
                        scale_factor=patch_size,  # 使用匹配的patch_size
                        mode="bilinear",
                        align_corners=False
                    ).squeeze()
                    
                    # 转换为numpy并归一化
                    feature_map_np = feature_map_upsampled.detach().cpu().numpy()
                    feature_map_np = (feature_map_np - feature_map_np.min()) / (feature_map_np.max() - feature_map_np.min() + 1e-8)
                    
                    # 显示特征图
                    im1 = axes[0, i].imshow(feature_map_np, cmap='viridis')
                    axes[0, i].set_title(f'{layer_name}\n({patch_info})')
                    axes[0, i].axis('off')
                    plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
                    
                    # 显示特征的标准差
                    feature_std = torch.std(feature_tensor, dim=-1)
                    # 同样处理标准差：跳过CLS token
                    patch_std_1d = feature_std[1:len(patch_features_1d)+1]
                    patch_std_2d = patch_std_1d.reshape(grid_size, grid_size)
                    patch_std_map = patch_std_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    
                    # 上采样标准差图
                    std_map_upsampled = F.interpolate(
                        patch_std_map,
                        scale_factor=patch_size,
                        mode="bilinear",
                        align_corners=False
                    ).squeeze()
                    
                    std_map_np = std_map_upsampled.detach().cpu().numpy()
                    std_map_np = (std_map_np - std_map_np.min()) / (std_map_np.max() - std_map_np.min() + 1e-8)
                    
                    im2 = axes[1, i].imshow(std_map_np, cmap='plasma')
                    axes[1, i].set_title(f'{layer_name}\n(Std {patch_info})')
                    axes[1, i].axis('off')
                    plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
                else:
                    print(f"    - Warning: Cannot reshape seq_len {seq_len} to square")
                    # 如果无法重塑为正方形，尝试其他方法
                    # 方法1：尝试不同的重塑方式
                    if seq_len > 0:
                        # 找到最接近正方形的因子分解
                        factors = []
                        for j in range(1, int(np.sqrt(seq_len)) + 1):
                            if seq_len % j == 0:
                                factors.append((j, seq_len // j))
                        
                        if factors:
                            # 选择最接近正方形的因子对
                            best_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
                            h, w = best_factor
                            
                            # 确保feature_mean是numpy数组
                            if isinstance(feature_mean, torch.Tensor):
                                feature_mean_np = feature_mean.detach().cpu().numpy()
                            else:
                                feature_mean_np = feature_mean
                            
                            feature_map = feature_mean_np[:h*w].reshape(h, w)
                            
                            # 使用PIL上采样到标准尺寸
                            feature_pil = Image.fromarray((feature_map * 255).astype(np.uint8))
                            feature_pil = feature_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
                            feature_map = np.array(feature_pil) / 255.0
                            feature_map = self.normalize(feature_map)
                            
                            axes[0, i].imshow(feature_map, cmap='viridis')
                            axes[0, i].set_title(f'{layer_name}\n(Reshaped {h}x{w})')
                            axes[0, i].axis('off')
                            
                            # 标准差
                            # 确保feature_np是numpy数组
                            if isinstance(feature_np, torch.Tensor):
                                feature_np_tensor = feature_np.detach().cpu().numpy()
                            else:
                                feature_np_tensor = feature_np
                            
                            feature_std = np.std(feature_np_tensor, axis=-1)[:h*w].reshape(h, w)
                            std_pil = Image.fromarray((feature_std * 255).astype(np.uint8))
                            std_pil = std_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
                            feature_std_map = np.array(std_pil) / 255.0
                            feature_std_map = self.normalize(feature_std_map)
                            
                            axes[1, i].imshow(feature_std_map, cmap='plasma')
                            axes[1, i].set_title(f'{layer_name}\n(Std {h}x{w})')
                            axes[1, i].axis('off')
                        else:
                            # 如果无法重塑，显示原始数据的统计信息
                            feature_flat = feature_mean.reshape(1, -1)
                            axes[0, i].text(0.5, 0.5, f'Seq Len: {seq_len}\nMean: {np.mean(feature_mean):.3f}\nStd: {np.std(feature_mean):.3f}', 
                                          ha='center', va='center', transform=axes[0, i].transAxes)
                            axes[0, i].set_title(f'{layer_name}\n(Stats)')
                            axes[0, i].axis('off')
                            
                            axes[1, i].text(0.5, 0.5, f'Min: {np.min(feature_mean):.3f}\nMax: {np.max(feature_mean):.3f}', 
                                          ha='center', va='center', transform=axes[1, i].transAxes)
                            axes[1, i].set_title(f'{layer_name}\n(Range)')
                            axes[1, i].axis('off')
                    else:
                        # 空数据
                        axes[0, i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[0, i].transAxes)
                        axes[0, i].set_title(f'{layer_name}\n(Empty)')
                        axes[0, i].axis('off')
                        
                        axes[1, i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[1, i].transAxes)
                        axes[1, i].set_title(f'{layer_name}\n(Empty)')
                        axes[1, i].axis('off')
                    
            except Exception as e:
                print(f"    - Error processing layer {i}: {e}")
                # 创建空白图像
                blank = np.zeros((self.img_size, self.img_size))
                axes[0, i].imshow(blank, cmap='viridis')
                axes[0, i].set_title(f'{layer_name}\n(Error)')
                axes[0, i].axis('off')
                
                axes[1, i].imshow(blank, cmap='plasma')
                axes[1, i].set_title(f'{layer_name}\n(Error)')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        save_file = os.path.join(self.save_path, 'feature_maps', f"{filename}.png")
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 特征图已保存: {save_file}")
        return save_file
    
    def visualize_attention_patterns(self, 
                                   attention_weights: torch.Tensor,
                                   layer_name: str,
                                   filename: str = "attention") -> str:
        """可视化注意力模式"""
        
        attn_np = self.tensor_to_numpy(attention_weights)
        
        # 处理维度
        if attn_np.ndim == 4:  # [batch, heads, seq, seq]
            attn_np = attn_np[0]  # 取第一个样本
            # 平均所有注意力头
            attn_np = np.mean(attn_np, axis=0)
        
        # 创建注意力热图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 注意力矩阵
        im1 = axes[0].imshow(attn_np, cmap='Blues')
        axes[0].set_title(f'{layer_name}\nAttention Matrix')
        axes[0].set_xlabel('Key Position')
        axes[0].set_ylabel('Query Position')
        plt.colorbar(im1, ax=axes[0])
        
        # CLS token 的注意力 (第一行)
        cls_attention = attn_np[0, 1:]  # 去掉 CLS token 自身
        spatial_size = int(np.sqrt(len(cls_attention)))
        
        if spatial_size * spatial_size == len(cls_attention):
            cls_attn_map = cls_attention.reshape(spatial_size, spatial_size)
            # 使用PIL替代cv2
            cls_attn_pil = Image.fromarray((cls_attn_map * 255).astype(np.uint8))
            cls_attn_pil = cls_attn_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
            cls_attn_map = np.array(cls_attn_pil) / 255.0
            
            im2 = axes[1].imshow(cls_attn_map, cmap='hot')
            axes[1].set_title(f'{layer_name}\nCLS Token Attention')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 保存图像
        save_file = os.path.join(self.save_path, 'attention_maps', f"{filename}.png")
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_file
    
    def visualize_text_prompt_similarity(self, 
                                       patch_features: torch.Tensor,
                                       text_features: torch.Tensor,
                                       filename: str = "text_similarity") -> str:
        """可视化文本提示与图像补丁的相似度"""
        
        print(f"  🔗 可视化文本-图像相似度...")
        
        patch_np = self.tensor_to_numpy(patch_features)
        text_np = self.tensor_to_numpy(text_features)
        
        print(f"    - patch_features: {patch_np.shape}")
        print(f"    - text_features: {text_np.shape}")
        
        try:
            # 计算相似度
            # patch_np: [batch, seq_len, hidden_dim] 或 [seq_len, hidden_dim]
            # text_np: [batch, num_classes, hidden_dim] 或 [num_classes, hidden_dim]
            
            if patch_np.ndim == 3:
                patch_np = patch_np[0]  # [seq_len, hidden_dim]
            
            if text_np.ndim == 3:
                text_np = text_np[0]  # [num_classes, hidden_dim]
            
            # 确保维度正确
            if patch_np.ndim != 2 or text_np.ndim != 2:
                print(f"    - Warning: Unexpected dimensions - patch: {patch_np.shape}, text: {text_np.shape}")
                return ""
            
            # 计算余弦相似度
            patch_norm = patch_np / (np.linalg.norm(patch_np, axis=-1, keepdims=True) + 1e-8)
            text_norm = text_np / (np.linalg.norm(text_np, axis=-1, keepdims=True) + 1e-8)
            
            similarity = np.dot(patch_norm, text_norm.T)  # [seq_len, num_classes]
            
            print(f"    - similarity shape: {similarity.shape}")
            print(f"    - similarity range: [{similarity.min():.4f}, {similarity.max():.4f}]")
            print(f"    - similarity mean: {similarity.mean():.4f}, std: {similarity.std():.4f}")
            
            # 创建可视化
            num_classes = similarity.shape[1]
            fig, axes = plt.subplots(1, num_classes + 1, figsize=(5*(num_classes + 1), 5))
            
            if num_classes == 1:
                axes = [axes]
            
            # 为每个类别创建相似度图
            for i in range(num_classes):
                sim_map = similarity[:, i]
                seq_len = len(sim_map)
                
                print(f"    - Class {i}: seq_len={seq_len}, range=[{sim_map.min():.4f}, {sim_map.max():.4f}], mean={sim_map.mean():.4f}")
                
                # 使用与feature_maps相同的patch配置方法
                # 根据实际图像尺寸计算正确的patch配置
                # 从DINOv3配置中获取patch_size，通常是16
                patch_size = 16  # DINOv3-ViT-L/16的默认patch_size
                
                # 计算不同图像尺寸下的patch数量
                patch_configs = []
                for img_size in [224, 336, 512]:
                    num_patches = img_size // patch_size
                    total_patches = num_patches * num_patches
                    expected_seq_len = total_patches + 1  # +1 for CLS token
                    patch_configs.append((patch_size, num_patches, expected_seq_len))
                
                # 特别处理512x512图像的情况
                # 512 // 16 = 32, 所以是32x32=1024个patches + 1个CLS token = 1025
                patch_configs.append((16, 32, 1025))  # 512x512图像
                
                # 添加一些常见的配置
                patch_configs.extend([
                    (14, 16, 16*16+1),  # patch_size=14, image_size=224 -> 16x16=256 patches
                    (14, 24, 24*24+1),  # patch_size=14, image_size=336 -> 24x24=576 patches
                    (32, 7, 7*7+1),     # patch_size=32, image_size=224 -> 7x7=49 patches
                ])
                
                sim_map_2d = None
                patch_info = ""
                
                for patch_size, grid_size, expected_seq_len in patch_configs:
                    
                    if seq_len == expected_seq_len:
                        # 跳过CLS token (第一个元素)，取patch tokens
                        expected_patches = grid_size * grid_size  # 计算patch数量
                        patch_sim_1d = sim_map[1:expected_patches+1]  # 去掉CLS token
                        
                        # reshape为2D
                        sim_map_2d = patch_sim_1d.reshape(grid_size, grid_size)
                        patch_info = f"patch_{patch_size}_grid_{grid_size}x{grid_size}"
                        print(f"    - 匹配配置: patch_size={patch_size}, grid_size={grid_size}x{grid_size}")
                        break
                
                if sim_map_2d is not None:
                    # 使用F.interpolate进行上采样（与feature_maps一致）
                    sim_tensor = torch.tensor(sim_map_2d).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    
                    sim_map_upsampled = F.interpolate(
                        sim_tensor,
                        scale_factor=patch_size,  # 使用匹配的patch_size
                        mode="bilinear",
                        align_corners=False
                    ).squeeze()
                    
                    sim_map_np = sim_map_upsampled.detach().cpu().numpy()
                    
                    # 使用动态范围而不是固定的-1到1
                    vmin, vmax = sim_map_np.min(), sim_map_np.max()
                    im = axes[i].imshow(sim_map_np, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                    axes[i].set_title(f'Text Class {i}\n({patch_info})\nRange: [{vmin:.3f}, {vmax:.3f}]')
                    axes[i].axis('off')
                    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                else:
                    print(f"    - Warning: Cannot reshape seq_len {seq_len} to square for class {i}")
                    # 尝试其他重塑方式
                    if seq_len > 0:
                        # 找到最接近正方形的因子分解
                        factors = []
                        for j in range(1, int(np.sqrt(seq_len)) + 1):
                            if seq_len % j == 0:
                                factors.append((j, seq_len // j))
                        
                        if factors:
                            # 选择最接近正方形的因子对
                            best_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
                            h, w = best_factor
                            
                            # 跳过CLS token（如果存在）
                            if seq_len > h * w:
                                # 有CLS token，跳过第一个元素
                                patch_sim_1d = sim_map[1:h*w+1]
                            else:
                                # 没有CLS token，直接使用
                                patch_sim_1d = sim_map[:h*w]
                            
                            sim_map_2d = patch_sim_1d.reshape(h, w)
                            
                            # 使用PIL上采样到标准尺寸
                            sim_pil = Image.fromarray((sim_map_2d * 255).astype(np.uint8))
                            sim_pil = sim_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
                            sim_map_resized = np.array(sim_pil) / 255.0
                            
                            # 使用动态范围
                            vmin, vmax = sim_map_resized.min(), sim_map_resized.max()
                            im = axes[i].imshow(sim_map_resized, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                            axes[i].set_title(f'Text Class {i}\n(Reshaped {h}x{w})\nRange: [{vmin:.3f}, {vmax:.3f}]')
                            axes[i].axis('off')
                            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                        else:
                            # 创建线性插值图作为fallback
                            sim_map_linear = np.linspace(-1, 1, self.img_size * self.img_size).reshape(self.img_size, self.img_size)
                            axes[i].imshow(sim_map_linear, cmap='RdBu_r', vmin=-1, vmax=1)
                            axes[i].set_title(f'Text Class {i}\n(Linear Fallback)')
                            axes[i].axis('off')
            
            # 显示平均相似度（使用与上面相同的patch配置方法）
            avg_sim = np.mean(similarity, axis=1)
            seq_len = len(avg_sim)
            
            print(f"    - Average similarity: seq_len={seq_len}")
            
            # 使用相同的patch配置方法
            # 根据实际图像尺寸计算正确的patch配置
            # 从DINOv3配置中获取patch_size，通常是16
            patch_size = 16  # DINOv3-ViT-L/16的默认patch_size
            
            # 计算不同图像尺寸下的patch数量
            patch_configs = []
            for img_size in [224, 336, 512]:
                num_patches = img_size // patch_size
                total_patches = num_patches * num_patches
                expected_seq_len = total_patches + 1  # +1 for CLS token
                patch_configs.append((patch_size, num_patches, expected_seq_len))
            
            # 特别处理512x512图像的情况
            # 512 // 16 = 32, 所以是32x32=1024个patches + 1个CLS token = 1025
            patch_configs.append((16, 32, 1025))  # 512x512图像
            
            # 添加一些常见的配置
            patch_configs.extend([
                (14, 16, 16*16+1),  # patch_size=14, image_size=224 -> 16x16=256 patches
                (14, 24, 24*24+1),  # patch_size=14, image_size=336 -> 24x24=576 patches
                (32, 7, 7*7+1),     # patch_size=32, image_size=224 -> 7x7=49 patches
            ])
            
            avg_sim_2d = None
            patch_info = ""
            
            for patch_size, grid_size, expected_seq_len in patch_configs:
                if seq_len == expected_seq_len:
                    # 跳过CLS token (第一个元素)，取patch tokens
                    expected_patches = grid_size * grid_size  # 计算patch数量
                    patch_avg_1d = avg_sim[1:expected_patches+1]  # 去掉CLS token
                    
                    # reshape为2D
                    avg_sim_2d = patch_avg_1d.reshape(grid_size, grid_size)
                    patch_info = f"patch_{patch_size}_grid_{grid_size}x{grid_size}"
                    print(f"    - 平均相似度匹配配置: patch_size={patch_size}, grid_size={grid_size}x{grid_size}")
                    break
            
            if avg_sim_2d is not None:
                # 使用F.interpolate进行上采样（与feature_maps一致）
                avg_tensor = torch.tensor(avg_sim_2d).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                avg_sim_upsampled = F.interpolate(
                    avg_tensor,
                    scale_factor=patch_size,  # 使用匹配的patch_size
                    mode="bilinear",
                    align_corners=False
                ).squeeze()
                
                avg_sim_np = avg_sim_upsampled.detach().cpu().numpy()
                
                im = axes[-1].imshow(avg_sim_np, cmap='viridis')
                axes[-1].set_title(f'Average Similarity\n({patch_info})')
                axes[-1].axis('off')
                plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
            else:
                print(f"    - Warning: Cannot reshape avg similarity seq_len {seq_len} to square")
                # 尝试其他重塑方式
                if seq_len > 0:
                    # 找到最接近正方形的因子分解
                    factors = []
                    for j in range(1, int(np.sqrt(seq_len)) + 1):
                        if seq_len % j == 0:
                            factors.append((j, seq_len // j))
                    
                    if factors:
                        # 选择最接近正方形的因子对
                        best_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
                        h, w = best_factor
                        
                        # 跳过CLS token（如果存在）
                        if seq_len > h * w:
                            # 有CLS token，跳过第一个元素
                            patch_avg_1d = avg_sim[1:h*w+1]
                        else:
                            # 没有CLS token，直接使用
                            patch_avg_1d = avg_sim[:h*w]
                        
                        avg_sim_2d = patch_avg_1d.reshape(h, w)
                        
                        # 使用PIL上采样到标准尺寸
                        avg_pil = Image.fromarray((avg_sim_2d * 255).astype(np.uint8))
                        avg_pil = avg_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
                        avg_sim_resized = np.array(avg_pil) / 255.0
                        
                        im = axes[-1].imshow(avg_sim_resized, cmap='viridis')
                        axes[-1].set_title(f'Average Similarity\n(Reshaped {h}x{w})')
                        axes[-1].axis('off')
                        plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
                    else:
                        # 创建线性插值图作为fallback
                        avg_sim_linear = np.linspace(0, 1, self.img_size * self.img_size).reshape(self.img_size, self.img_size)
                        axes[-1].imshow(avg_sim_linear, cmap='viridis')
                        axes[-1].set_title('Average Similarity\n(Linear Fallback)')
                        axes[-1].axis('off')
            
            plt.tight_layout()
            
            # 保存图像
            save_file = os.path.join(self.save_path, 'comparisons', f"{filename}.png")
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ 相似度图已保存: {save_file}")
            return save_file
            
        except Exception as e:
            print(f"  ❌ 相似度可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def create_debug_report(self, 
                          original_image: torch.Tensor,
                          anomaly_map: torch.Tensor,
                          patch_features: List[torch.Tensor],
                          text_features: torch.Tensor,
                          patch_token_memory: Optional[List[torch.Tensor]] = None,
                          gt_mask: Optional[torch.Tensor] = None,
                          scores: Optional[Dict] = None,
                          attention_weights: Optional[torch.Tensor] = None,
                          filename: str = "debug_report") -> str:
        """创建完整的调试报告"""
        
        print(f"🔍 创建调试报告: {filename}")
        
        # 保存各种可视化
        anomaly_file = self.save_anomaly_map_overlay(
            original_image, anomaly_map, gt_mask, f"{filename}_anomaly"
        )
        
        layer_names = [f"Layer_{i}" for i in range(len(patch_features))]
        feature_file = self.visualize_feature_maps(
            patch_features, layer_names, f"{filename}_features"
        )
        
        # 添加patch_token_memory的可视化
        memory_file = ""
        if patch_token_memory is not None and len(patch_token_memory) > 0:
            try:
                memory_layer_names = [f"Memory_Layer_{i}" for i in range(len(patch_token_memory))]
                memory_file = self.visualize_feature_maps(
                    patch_token_memory, memory_layer_names, f"{filename}_memory_features"
                )
                print(f"  ✅ Patch Token Memory特征图已保存: {memory_file}")
            except Exception as e:
                print(f"  ⚠️ Patch Token Memory可视化失败: {e}")
                memory_file = ""
        else:
            print(f"  ℹ️ 无Patch Token Memory数据")
        
        # 添加feature map与原图的叠加可视化
        overlay_file = self.visualize_feature_overlay(
            original_image, patch_features, layer_names, f"{filename}_overlay"
        )
        
        similarity_file = self.visualize_text_prompt_similarity(
            patch_features[-1], text_features, f"{filename}_similarity"
        )
        
        # 可视化注意力模式 (如果可用)
        attention_file = ""
        if attention_weights is not None:
            try:
                attention_file = self.visualize_attention_patterns(
                    attention_weights, "attention_layer", f"{filename}_attention"
                )
                print(f"  ✅ 注意力图已保存: {attention_file}")
            except Exception as e:
                print(f"  ⚠️ 注意力可视化失败: {e}")
                attention_file = ""
        else:
            print(f"  ℹ️ 无注意力权重数据")
        
        # 创建统计信息
        anomaly_np = self.tensor_to_numpy(anomaly_map)
        # 处理异常图的维度
        if anomaly_np.ndim == 4:
            anomaly_np = anomaly_np[0]
        if anomaly_np.ndim == 3:
            if anomaly_np.shape[0] == 1:
                anomaly_np = anomaly_np[0]
            elif anomaly_np.shape[-1] == 1:
                anomaly_np = anomaly_np[:, :, 0]
        
        # 确保是2D数组
        if anomaly_np.ndim != 2:
            anomaly_np = anomaly_np.reshape(-1)[:self.img_size*self.img_size].reshape(self.img_size, self.img_size)
        
        stats = {
            'anomaly_mean': float(np.mean(anomaly_np)),
            'anomaly_std': float(np.std(anomaly_np)),
            'anomaly_min': float(np.min(anomaly_np)),
            'anomaly_max': float(np.max(anomaly_np)),
        }
        
        if scores:
            stats.update(scores)
        
        # 保存统计信息
        stats_file = os.path.join(self.save_path, f"{filename}_stats.txt")
        with open(stats_file, 'w') as f:
            f.write("VVCLIP Debug Report\n")
            f.write("=" * 50 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value:.4f}\n")
        
        print(f"✅ 调试报告已保存:")
        print(f"   - 异常图: {anomaly_file}")
        print(f"   - 特征图: {feature_file}")
        if memory_file:
            print(f"   - Patch Token Memory特征图: {memory_file}")
        print(f"   - 特征叠加图: {overlay_file}")
        print(f"   - 相似度图: {similarity_file}")
        if attention_file:
            print(f"   - 注意力图: {attention_file}")
        print(f"   - 统计信息: {stats_file}")
        
        return stats_file


def create_visualization_for_batch(visualizer: Visualizer,
                                 batch_data: Dict,
                                 model_outputs: Dict,
                                 batch_idx: int = 0) -> None:
    """为批次数据创建可视化"""
    
    # 提取数据
    image = batch_data['img'][batch_idx:batch_idx+1]  # 保持批次维度
    gt_mask = batch_data.get('img_mask', None)
    if gt_mask is not None:
        gt_mask = gt_mask[batch_idx:batch_idx+1]
    
    # 提取模型输出
    anomaly_map = model_outputs['anomaly_map']
    patch_features = model_outputs.get('patch_features', [])
    text_features = model_outputs.get('text_features', None)
    
    # 创建调试报告
    visualizer.create_debug_report(
        original_image=image,
        anomaly_map=anomaly_map,
        patch_features=patch_features,
        text_features=text_features,
        gt_mask=gt_mask,
        scores=model_outputs.get('scores', {}),
        filename=f"batch_{batch_idx}"
    )
