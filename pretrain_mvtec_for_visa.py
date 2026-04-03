import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn.init as init

import torch.nn as nn
import open_clip
from torch import optim

import os
import random
import numpy as np

# dinov3
from transformers import AutoImageProcessor, AutoModel
from utils.dinov3_utils import dinov3_encode_image
from utils.logger import get_logger
from utils import prompt_generator

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    """
    Focal Loss from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss


def validate(image_proj, patch_proj, val_dataloader, text_features, dino_processor, dino_model, 
            clip_model, device, features_list, temperature=0.07):
    """验证函数：计算异常检测性能"""
    from sklearn.metrics import roc_auc_score
    
    image_proj.eval()
    patch_proj.eval()
    
    all_anomaly_scores = []
    all_labels = []
    
    with torch.no_grad():
        for items in tqdm(val_dataloader, desc="Validating", leave=False):
            image = items['img'].to(device)
            gt_mask = items['img_mask'].to(device)
            anomaly_label = items['anomaly'].to(device)  # 0: normal, 1: abnormal
            
            batch_size = image.shape[0]
            
            # DINOv3 特征提取（验证时不需要梯度）
            dino_out = dinov3_encode_image(image, dino_processor, dino_model, 
                                          device=device, layer_indices=features_list)
            image_embedding = dino_out["cls"].clone()
            
            if "multi_layer_features" in dino_out:
                patch_token_memory = [feat.clone() for feat in dino_out["multi_layer_features"]]
            else:
                combined = torch.cat([image_embedding.unsqueeze(1), dino_out["patch_flat"].clone()], dim=1)
                patch_token_memory = [combined] * len(features_list)
            
            grid_h, grid_w = dino_out["grid_size"].tolist()
            
            # 投影到 CLIP 空间
            image_features_768 = image_proj(image_embedding)
            image_features_768 = F.normalize(image_features_768, dim=-1)
            
            # 投影 patch features
            patch_features_list = []
            for layer_feat in patch_token_memory:
                batch_size_l, num_tokens, dim_1024 = layer_feat.shape
                layer_768 = patch_proj(layer_feat.view(-1, dim_1024)).view(batch_size_l, num_tokens, 768)
                layer_768 = F.normalize(layer_768, dim=-1)
                patch_features_list.append(layer_768)
            
            patch_features = patch_features_list[-1][:, 1:, :]
            patch_features = patch_features.reshape(batch_size, grid_h, grid_w, 768)
            
            # 计算跨模态异常图
            similarity = torch.einsum('bhwc,nc->bhwn', patch_features, text_features)
            similarity = similarity / temperature
            anomaly_map = F.softmax(similarity, dim=-1)[..., 1]  # [B, H, W]
            
            # 图像级异常分数：异常图的最大值
            image_anomaly_score = anomaly_map.view(batch_size, -1).max(dim=1)[0]
            
            all_anomaly_scores.extend(image_anomaly_score.cpu().numpy())
            all_labels.extend(anomaly_label.cpu().numpy())
    
    # 计算 Image-level AUROC
    all_labels = np.array(all_labels)
    all_anomaly_scores = np.array(all_anomaly_scores)
    
    if len(np.unique(all_labels)) > 1:  # 确保有正负样本
        auroc = roc_auc_score(all_labels, all_anomaly_scores)
    else:
        auroc = 0.0
    
    image_proj.train()
    patch_proj.train()
    
    return auroc


def main(args):
    img_size = args.image_size
    features_list = args.features_list
    epochs = args.epochs

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 记录实验配置
    logger.info("="*80)
    logger.info("SUPERVISED PRE-TRAINING (Projection Layers with Anomaly Masks)")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset.upper()}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Image Size: {args.image_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Save Path: {args.save_path}")
    logger.info(f"Checkpoint Path: {args.checkpoint_path}")
    logger.info("="*80)
    logger.info("HYPERPARAMETERS")
    logger.info("="*80)
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Weight Decay: {args.weight_decay}")
    logger.info(f"Lambda CM (Segmentation): {args.lambda_cm}")
    logger.info(f"Lambda AACM (Awareness): {args.lambda_aacm}")
    logger.info(f"Lambda Global (Classification): {args.lambda_global}")
    logger.info(f"Lambda Image Alignment (Contrastive+Consistency): {args.lambda_image_alignment}")
    logger.info(f"  ├─ Lambda Consistency: {args.lambda_consistency}")
    logger.info(f"  └─ Contrastive Margin: {args.margin}")
    logger.info(f"Focal Alpha: {args.focal_alpha}")
    logger.info(f"Focal Gamma: {args.focal_gamma}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Features List: {args.features_list}")
    logger.info("="*80)

    # load clip model
    clip_model, _, _ = open_clip.create_model_and_transforms(args.clip_name, img_size, pretrained="openai")
    clip_model.eval()
    clip_model.to(device)

    # load dino model
    dino_processor = AutoImageProcessor.from_pretrained(args.dinov3_model_path)
    dino_model = AutoModel.from_pretrained(args.dinov3_model_path)
    dino_model.eval()
    dino_model.to(device)
    
    tokenizer = open_clip.get_tokenizer(args.clip_name)

    # 自定义数据加载
    logger.info(f"Loading data from {args.data_path}")
    import glob
    from PIL import Image
    from torchvision import transforms
    
    # 数据预处理
    preprocess = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # MVTec数据集类别列表
    mvtec_classes = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    
    # 构建训练/验证数据（包括正常和异常样本）
    train_data = []
    val_data = []
    
    for obj in mvtec_classes:
        logger.info(f"Processing MVTec object: {obj}")
        
        # 正常样本 - 从train/good目录
        normal_path = os.path.join(args.data_path, obj, 'train', 'good')
        if os.path.exists(normal_path):
            normal_files = sorted(glob.glob(os.path.join(normal_path, '*.png')))
            
            # 80% 训练，20% 验证
            split_idx = int(len(normal_files) * 0.8)
            for i, img_path in enumerate(normal_files):
                data_item = {
                    'img_path': img_path,
                    'img_mask': None,  # 正常样本没有掩码
                    'anomaly': 0,  # 0: normal
                    'cls_name': obj
                }
                if i < split_idx:
                    train_data.append(data_item)
                else:
                    val_data.append(data_item)
        
        # 异常样本 - 从test目录的各个缺陷类型
        test_path = os.path.join(args.data_path, obj, 'test')
        if os.path.exists(test_path):
            for defect_type in os.listdir(test_path):
                defect_path = os.path.join(test_path, defect_type)
                if os.path.isdir(defect_path) and defect_type != 'good':
                    defect_files = sorted(glob.glob(os.path.join(defect_path, '*.png')))
                    
                    # 80% 训练，20% 验证
                    split_idx = int(len(defect_files) * 0.8)
                    for i, img_path in enumerate(defect_files):
                        # 查找对应的掩码文件 - MVTec掩码在ground_truth目录
                        mask_dir = os.path.join(args.data_path, obj, 'ground_truth', defect_type)
                        mask_filename = os.path.basename(img_path)
                        mask_path = os.path.join(mask_dir, mask_filename)
                        
                        data_item = {
                            'img_path': img_path,
                            'img_mask': mask_path if os.path.exists(mask_path) else None,
                            'anomaly': 1,  # 1: abnormal
                            'cls_name': obj
                        }
                        
                        if i < split_idx:
                            train_data.append(data_item)
                        else:
                            val_data.append(data_item)
    
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    
    # 自定义数据集类
    class PreTrainDataset(torch.utils.data.Dataset):
        def __init__(self, data_list, preprocess):
            self.data_list = data_list
            self.preprocess = preprocess
        
        def __len__(self):
            return len(self.data_list)
        
        def __getitem__(self, idx):
            item = self.data_list[idx]
            
            # 加载图像
            image = Image.open(item['img_path']).convert('RGB')
            image = self.preprocess(image)
            
            # 加载掩码（如果有）
            if item['img_mask'] and os.path.exists(item['img_mask']):
                mask = Image.open(item['img_mask']).convert('L')
                mask = transforms.Resize((args.image_size, args.image_size))(mask)
                mask = transforms.ToTensor()(mask)
            else:
                # 正常样本：全零掩码
                mask = torch.zeros(1, args.image_size, args.image_size)
            
            return {
                'img': image,
                'img_mask': mask,
                'anomaly': torch.tensor(item['anomaly'], dtype=torch.long),
                'cls_name': item['cls_name'],
                'img_path': item['img_path']
            }
    
    # 创建数据加载器
    train_dataset = PreTrainDataset(train_data, preprocess)
    val_dataset = PreTrainDataset(val_data, preprocess)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # 初始化两个独立的投影层
    image_proj = prompt_generator.VisualProjection(vis_dim=args.vis_dim, output_dim=args.output_dim)
    image_proj = image_proj.to(device)
    
    patch_proj = prompt_generator.VisualProjection(vis_dim=args.vis_dim, output_dim=args.output_dim)
    patch_proj = patch_proj.to(device)

    # 【监督预训练】只训练投影层
    logger.info("🔧 Supervised Pre-training: Training image_proj and patch_proj")
    optimizer = optim.AdamW(
        list(image_proj.parameters()) + list(patch_proj.parameters()), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    # 使用 warmup + cosine annealing scheduler（参考代码风格）
    total_steps = args.epochs * len(train_dataloader)
    warmup_steps = int(0.03 * total_steps)  # 3% warmup
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine annealing after warmup
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(args.eta_min / args.learning_rate, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps} (3%)")
    logger.info(f"Scheduler: Warmup + Cosine Annealing")
    
    # 损失函数
    focal_loss = FocalLoss(apply_nonlin=nn.Softmax(dim=1), alpha=args.focal_alpha, gamma=args.focal_gamma)
    dice_loss = BinaryDiceLoss()
    
    # 获取 CLIP 文本特征 - 使用丰富的提示模板（参考 prompt_ensemble.py）
    logger.info("Generating text prompts for MVTec dataset...")
    
    # 从 prompt_ensemble.py 导入相关模板
    from utils.prompt_ensemble import state_normal, state_anomaly, class_state_abnormal, img_temp, inds_temp, text_temp, surf_temp, texture_list
    
    # 工业图像模板（从 prompt_ensemble.py 复制）
    img_templates = img_temp
    
    with torch.no_grad():
        # 生成正常样本的文本特征
        normal_prompts = []
        for state in state_normal:
            for template in img_templates:
                normal_prompts.append(template.format(state.format("object")))
        
        logger.info(f"Generated {len(normal_prompts)} normal prompts")
        
        # 生成异常样本的文本特征 - 使用MVTec特定的异常状态
        abnormal_prompts = []
        # 使用MVTec的通用异常状态和特定异常状态
        mvtec_abnormal_states = class_state_abnormal.get('object', [])
        combined_abnormal = state_anomaly + mvtec_abnormal_states
        
        for state in combined_abnormal:
            for template in img_templates:
                abnormal_prompts.append(template.format(state.format("object")))
        
        logger.info(f"Generated {len(abnormal_prompts)} abnormal prompts")
        
        # 编码文本
        normal_text = tokenizer(normal_prompts).to(device)
        abnormal_text = tokenizer(abnormal_prompts).to(device)
        
        normal_embeddings = clip_model.encode_text(normal_text)
        abnormal_embeddings = clip_model.encode_text(abnormal_text)
        
        # 平均所有提示的特征
        normal_features = normal_embeddings.mean(dim=0, keepdim=True)
        abnormal_features = abnormal_embeddings.mean(dim=0, keepdim=True)
        
        # 归一化
        normal_features = F.normalize(normal_features, dim=-1)
        abnormal_features = F.normalize(abnormal_features, dim=-1)
        
        # 组合成 [2, 768]
        text_features = torch.cat([normal_features, abnormal_features], dim=0)  # [2, 768]
    
    logger.info(f"Text features shape: {text_features.shape}")
    logger.info(f"Normal feature norm: {normal_features.norm().item():.4f}")
    logger.info(f"Abnormal feature norm: {abnormal_features.norm().item():.4f}")
    logger.info(f"Feature similarity: {(normal_features @ abnormal_features.T).item():.4f}")
    
    # 创建checkpoint目录
    os.makedirs(args.checkpoint_path, exist_ok=True)
    best_loss = float('inf')
    best_auroc = 0.0

    # 训练循环
    logger.info("Starting Supervised Pre-training...")
    for epoch in range(epochs):
        image_proj.train()
        patch_proj.train()
        
        epoch_losses = []
        epoch_cm_losses = []
        epoch_aacm_losses = []
        epoch_global_losses = []
        epoch_image_alignment_losses = []
        epoch_contrastive_losses = []
        epoch_consistency_losses = []
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, items in enumerate(pbar):
            image = items['img'].to(device)  # [B, 3, H, W]
            gt_mask = items['img_mask'].to(device)  # [B, 1, H, W]
            gt_mask = (gt_mask > 0.5).float()  # 二值化
            anomaly_label = items['anomaly'].to(device)  # [B]
            
            batch_size = image.shape[0]
            
            with torch.no_grad():
                # DINOv3 特征提取
                dino_out = dinov3_encode_image(image, dino_processor, dino_model, 
                                              device=device, layer_indices=features_list)
                image_embedding_nograd = dino_out["cls"]  # [B, 1024]
                
                if "multi_layer_features" in dino_out:
                    patch_token_memory_nograd = dino_out["multi_layer_features"]
                else:
                    combined = torch.cat([image_embedding_nograd.unsqueeze(1), dino_out["patch_flat"]], dim=1)
                    patch_token_memory_nograd = [combined] * len(features_list)
                
                grid_h, grid_w = dino_out["grid_size"].tolist()
            
            # 克隆张量以支持梯度计算
            image_embedding = image_embedding_nograd.clone().detach().requires_grad_(False)
            patch_token_memory = [feat.clone().detach().requires_grad_(False) for feat in patch_token_memory_nograd]
            
            # 投影到 CLIP 空间
            image_features_768 = image_proj(image_embedding)  # [B, 768]
            image_features_768 = F.normalize(image_features_768, dim=-1)
            
            # 投影 patch features
            patch_features_list = []
            for layer_feat in patch_token_memory:
                # layer_feat: [B, 1+P, 1024]
                batch_size_l, num_tokens, dim_1024 = layer_feat.shape
                layer_768 = patch_proj(layer_feat.view(-1, dim_1024)).view(batch_size_l, num_tokens, 768)
                layer_768 = F.normalize(layer_768, dim=-1)
                patch_features_list.append(layer_768)
            
            # 使用最后一层的 patch features
            patch_features = patch_features_list[-1][:, 1:, :]  # [B, P, 768], 去掉 CLS token
            patch_features = patch_features.reshape(batch_size, grid_h, grid_w, 768)
            
            # ============ 1. 跨模态分割损失 (Cross-Modal Segmentation Loss) ============
            # 计算跨模态异常图：patch features 与 text features 的相似度
            similarity = torch.einsum('bhwc,nc->bhwn', patch_features, text_features)  # [B, H, W, 2]
            similarity = similarity / args.temperature
            similarity_logits = similarity.permute(0, 3, 1, 2)  # [B, 2, H, W] for FocalLoss
            anomaly_map_prob = F.softmax(similarity, dim=-1)  # [B, H, W, 2]
            anomaly_map = anomaly_map_prob[..., 1]  # [B, H, W], 取 abnormal 的概率
            
            # 调整 gt_mask 大小到 anomaly_map
            gt_mask_resized = F.interpolate(gt_mask, size=(grid_h, grid_w), 
                                           mode='bilinear', align_corners=False)  # [B, 1, H, W]
            gt_mask_resized_long = (gt_mask_resized > 0.5).long()  # [B, 1, H, W], 转为类别索引
            
            # 计算分割损失 (类似参考代码的 seg_loss)
            focal_seg = focal_loss(similarity_logits, gt_mask_resized_long)
            dice_seg = dice_loss(anomaly_map, gt_mask_resized.squeeze(1))
            seg_loss = focal_seg + dice_seg
            
            # ============ 2. AACM 异常感知损失 (Anomaly Awareness Loss) ============
            # CLS-patch 相似度分布
            cls_patch_sim_raw = torch.einsum('bc,bhwc->bhw', image_features_768, patch_features)  # [B, H, W]
            
            # 转换为二分类 logits [B, 2, H, W]
            # 负类（normal）: -cls_patch_sim, 正类（abnormal）: cls_patch_sim
            cls_patch_logits = torch.stack([-cls_patch_sim_raw, cls_patch_sim_raw], dim=1)  # [B, 2, H, W]
            cls_patch_sim = torch.sigmoid(cls_patch_sim_raw)  # [B, H, W], 用于 Dice Loss
            
            # AACM 损失 (类似参考代码的 anomaly_awareness_loss)
            focal_aware = focal_loss(cls_patch_logits, gt_mask_resized_long)
            dice_aware = dice_loss(cls_patch_sim, gt_mask_resized.squeeze(1))
            awareness_loss = focal_aware + dice_aware
            
            # ============ 3. 全局异常分类损失 (Global Anomaly Classification) ============
            # 使用异常图的最大值作为全局异常分数
            global_anomaly_score = anomaly_map.view(batch_size, -1).max(dim=1)[0]  # [B]
            
            # 构建二分类 logits [B, 2]
            normal_score = 1 - global_anomaly_score
            global_logits = torch.stack([normal_score, global_anomaly_score], dim=1)  # [B, 2]
            
            # 图像级分类损失
            global_loss = F.cross_entropy(global_logits, anomaly_label.long())
            
            # ============ 4. 图像CLS对比学习+一致性损失 (优化image_proj) ============
            # 策略：让image_proj学习CLS token与文本特征的相对关系，而非绝对分类
            
            # 4.1 对比学习损失 (Contrastive Loss)
            # 计算投影后的CLS特征与文本特征的相似度
            image_text_sim = torch.matmul(image_features_768, text_features.T) / args.temperature  # [B, 2]
            normal_sim = image_text_sim[:, 0]    # 与"normal"文本的相似度 [B]
            abnormal_sim = image_text_sim[:, 1]  # 与"abnormal"文本的相似度 [B]
            
            # Margin-based对比学习：学习相对关系而非绝对分类
            normal_mask = (anomaly_label == 0).float()    # 正常样本mask [B]
            abnormal_mask = (anomaly_label == 1).float()  # 异常样本mask [B]
            
            # 正常样本：确保 normal_sim > abnormal_sim + margin
            normal_contrastive = normal_mask * torch.clamp(args.margin - (normal_sim - abnormal_sim), min=0)
            # 异常样本：确保 abnormal_sim > normal_sim + margin
            abnormal_contrastive = abnormal_mask * torch.clamp(args.margin - (abnormal_sim - normal_sim), min=0)
            
            contrastive_loss = (normal_contrastive + abnormal_contrastive).mean()
            
            # 4.2 一致性损失 (Consistency Loss)
            # 让图像级CLS特征与patch级异常图保持一致
            patch_based_score = anomaly_map.view(batch_size, -1).mean(dim=1)  # [B] patch级异常分数
            cls_based_score = torch.sigmoid(abnormal_sim)  # [B] CLS级异常分数
            consistency_loss = F.mse_loss(cls_based_score, patch_based_score.detach())
            
            # 组合两个损失
            image_alignment_loss = contrastive_loss + args.lambda_consistency * consistency_loss
            
            # ============ 总损失 (原有损失 + 新增的对比学习损失) ============
            loss = (args.lambda_aacm * awareness_loss + 
                   args.lambda_cm * seg_loss + 
                   args.lambda_global * global_loss + 
                   args.lambda_image_alignment * image_alignment_loss)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(image_proj.parameters()) + list(patch_proj.parameters()), 
                max_norm=args.grad_clip_norm
            )
            optimizer.step()
            scheduler.step()  # 每个 batch 后更新学习率（参考代码风格）
            
            # 记录
            epoch_losses.append(loss.item())
            epoch_cm_losses.append(seg_loss.item())
            epoch_aacm_losses.append(awareness_loss.item())
            epoch_global_losses.append(global_loss.item())
            epoch_image_alignment_losses.append(image_alignment_loss.item())
            epoch_contrastive_losses.append(contrastive_loss.item())
            epoch_consistency_losses.append(consistency_loss.item())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'seg': f'{seg_loss.item():.4f}',
                'contra': f'{contrastive_loss.item():.4f}',
                'consis': f'{consistency_loss.item():.4f}'
            })
        
        # Epoch 统计
        avg_loss = np.mean(epoch_losses)
        avg_cm_loss = np.mean(epoch_cm_losses)
        avg_aacm_loss = np.mean(epoch_aacm_losses)
        avg_global_loss = np.mean(epoch_global_losses)
        avg_image_alignment_loss = np.mean(epoch_image_alignment_losses)
        avg_contrastive_loss = np.mean(epoch_contrastive_losses)
        avg_consistency_loss = np.mean(epoch_consistency_losses)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        logger.info(f"  Total Loss: {avg_loss:.4f}")
        logger.info(f"  Segmentation Loss: {avg_cm_loss:.4f}")
        logger.info(f"  Awareness Loss: {avg_aacm_loss:.4f}")
        logger.info(f"  Global Loss: {avg_global_loss:.4f}")
        logger.info(f"  Image Alignment Loss: {avg_image_alignment_loss:.4f} (Contrastive: {avg_contrastive_loss:.4f}, Consistency: {avg_consistency_loss:.4f})")
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Segmentation Loss: {avg_cm_loss:.4f}")
        print(f"  Awareness Loss: {avg_aacm_loss:.4f}")
        print(f"  Global Loss: {avg_global_loss:.4f}")
        print(f"  Image Alignment Loss: {avg_image_alignment_loss:.4f}")
        print(f"    ├─ Contrastive Loss: {avg_contrastive_loss:.4f}")
        print(f"    └─ Consistency Loss: {avg_consistency_loss:.4f}")
        print(f"{'='*80}\n")
        
        # 验证
        logger.info("Running validation...")
        val_auroc = validate(image_proj, patch_proj, val_dataloader, text_features, 
                            dino_processor, dino_model, clip_model, device, 
                            features_list, args.temperature)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Validation AUROC: {val_auroc:.4f}")
        print(f"\n{'='*80}")
        print(f"Validation AUROC: {val_auroc:.4f}")
        print(f"{'='*80}\n")
        
        # 每个 epoch 都保存模型
        checkpoint = {
            'epoch': epoch + 1,
            'image_proj_state_dict': image_proj.state_dict(),
            'patch_proj_state_dict': patch_proj.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_loss,
            'val_auroc': val_auroc,
            'args': vars(args)
        }
        
        # 保存当前 epoch 的模型
        epoch_checkpoint_path = os.path.join(args.checkpoint_path, f'projection_layers_epoch_{epoch+1}.pth')
        torch.save(checkpoint, epoch_checkpoint_path)
        logger.info(f"💾 Saved epoch {epoch+1} checkpoint: AUROC={val_auroc:.4f}, Loss={avg_loss:.4f}")
        print(f"💾 Saved epoch {epoch+1} checkpoint")
        
        # 如果是最佳模型，额外保存为 best 模型
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            checkpoint['best_auroc'] = best_auroc
            best_checkpoint_path = os.path.join(args.checkpoint_path, 'best_projection_layers.pth')
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"🌟 New best model! AUROC: {best_auroc:.4f}")
            print(f"🌟 New best model! AUROC: {best_auroc:.4f}")
    
    logger.info("="*80)
    logger.info("Supervised Pre-training Complete!")
    logger.info(f"Best Validation AUROC: {best_auroc:.4f}")
    logger.info(f"Final Training Loss: {avg_loss:.4f}")
    logger.info("="*80)
    
    print(f"\n{'='*80}")
    print("Pre-training Complete!")
    print(f"Best Validation AUROC: {best_auroc:.4f}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Supervised Pre-training for Projection Layers", add_help=True)
    
    # paths
    parser.add_argument("--data_path", type=str, 
                       default="/data2/zlt/code/abnormal_dataset/mvtec", 
                       help="path to MVTec dataset (with train/test splits)")
    parser.add_argument("--save_path", type=str, 
                       default='./results/pretrain_mvtec_for_visa', 
                       help='path to save logs')
    parser.add_argument("--checkpoint_path", type=str, 
                       default='./model_card/pretrain_mvtec_for_visa/', 
                       help='path to save checkpoints')
    parser.add_argument("--dinov3_model_path", type=str, 
                       default='./model_card/dinov3-vitl16-pretrain-lvd1689m')
    
    # training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--T_max", type=int, default=50, help="T_max for scheduler")
    parser.add_argument("--eta_min", type=float, default=1e-7, help="min learning rate")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="gradient clipping")
    
    # loss weights (参考代码权重: 0.25, 0.5, 0.25)
    parser.add_argument("--lambda_cm", type=float, default=0.5, help="weight for cross-modal segmentation loss")
    parser.add_argument("--lambda_aacm", type=float, default=0.25, help="weight for AACM awareness loss")
    parser.add_argument("--lambda_global", type=float, default=0.25, help="weight for global anomaly classification loss")
    parser.add_argument("--lambda_image_alignment", type=float, default=0.5, help="weight for image alignment loss (contrastive+consistency)")
    parser.add_argument("--lambda_consistency", type=float, default=0.3, help="weight for CLS-patch consistency within image alignment loss")
    parser.add_argument("--margin", type=float, default=0.5, help="margin for contrastive loss")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="focal loss alpha")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="focal loss gamma")
    parser.add_argument("--temperature", type=float, default=0.07, help="temperature for similarity")
    
    # model architecture
    parser.add_argument("--vis_dim", type=int, default=1024, help="DINOv3 dimension")
    parser.add_argument("--output_dim", type=int, default=768, help="CLIP dimension")
    parser.add_argument("--features_list", type=int, nargs="+", default=[12, 16, 20, 24])
    parser.add_argument("--image_size", type=int, default=512)
    
    # dataset
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--clip_name", type=str, default='ViT-L-14-336')
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    main(args)

