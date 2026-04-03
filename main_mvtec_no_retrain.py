# import VVCLIP_lib
import torch
import argparse
import torch.nn.functional as F

from tqdm import tqdm
import torch.nn.init as init
from torch.optim import lr_scheduler
import torch.nn as nn
from diffusers.pipelines import BlipDiffusionPipeline
from diffusers.utils import load_image
from torch import optim
from transformers import AutoImageProcessor, AutoModel


from utils.dataset import Dataset
from utils.logger import get_logger
import utils.prompt_generator as prompt_generator
from utils.prompt_ensemble import encode_text_with_prompt_ensemble
from utils.dinov3_utils import dinov3_encode_image
from utils.metrics import image_level_metrics, pixel_level_metrics


from utils.utils import get_transform
from utils.utils import aug

# from visualization import visualizer

from tqdm import tqdm
from scipy.ndimage import gaussian_filter

import open_clip
import os
import random
import numpy as np
from tabulate import tabulate




# ============= 修复 transformers 和 diffusers 兼容性问题 =============
# 问题：transformers 4.56.1 的 Blip2QFormerAttention 不接受 past_key_value 参数
# 但 diffusers 的 BlipDiffusion 仍在传递这个参数
def fix_blip_diffusion_compatibility():
    """修复 BlipDiffusion 与新版本 transformers 的兼容性"""
    try:
        from diffusers.pipelines.blip_diffusion.modeling_blip2 import Blip2QFormerAttention
        
        # 保存原始的 forward 方法
        original_forward = Blip2QFormerAttention.forward
        
        # 创建新的 forward 方法，接受但忽略 past_key_value 参数
        def patched_forward(self, hidden_states, attention_mask=None, head_mask=None, 
                          encoder_hidden_states=None, encoder_attention_mask=None, 
                          past_key_value=None, output_attentions=False, **kwargs):
            # 调用原始方法，但不传递 past_key_value（因为新版本不支持）
            return original_forward(
                self, 
                hidden_states, 
                attention_mask=attention_mask, 
                head_mask=head_mask, 
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask, 
                output_attentions=output_attentions
            )
        
        # 替换 forward 方法
        Blip2QFormerAttention.forward = patched_forward
        print("✓ 已成功修复 BlipDiffusion 兼容性问题")
        
    except Exception as e:
        print(f"⚠ 警告：无法应用兼容性补丁: {e}")
        print("如果出现 'past_key_value' 错误，请考虑降级 transformers 版本")

# 在导入 BlipDiffusionPipeline 之前应用补丁
fix_blip_diffusion_compatibility()
# ====================================================================

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_similarity(image_features, text_features, temperature=0.07):
    prob_1 = image_features[:, :1, :] @ text_features.t()
    b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
    # print(b, n_t, n_i, c)
    feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
    similarity = feats.sum(-1)
    return (similarity/temperature).softmax(-1), prob_1
    # return similarity, prob_1




def main(args):
    img_size = args.image_size
    features_list = args.features_list  # DINOv3 多层特征索引，例如 [6, 12, 18, 24]
    #few-shot learning parameter
    shot = args.shot
    epochs = args.epochs


    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建带时间戳的checkpoints目录
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.save_path, f"checkpoints_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"📁 Checkpoint directory: {checkpoint_dir}")
    print(f"📁 Checkpoint directory: {checkpoint_dir}")
    
    # 记录实验配置
    logger.info("="*80)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Shot: {args.shot}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Test Interval: {args.test_interval}")
    logger.info(f"Image Size: {args.image_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Save Path: {args.save_path}")
    logger.info("="*80)
    logger.info("HYPERPARAMETERS")
    logger.info("="*80)
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Weight Decay: {args.weight_decay}")
    logger.info(f"T_max: {args.T_max}")
    logger.info(f"Eta Min: {args.eta_min}")
    logger.info(f"Patch Loss Weight: {args.patch_loss_weight}")
    logger.info(f"Noise Scale: {args.noise_scale}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Grad Clip Norm: {args.grad_clip_norm}")
    logger.info(f"Random Seed: {args.seed}")
    logger.info(f"Features List: {args.features_list}")
    logger.info("="*80)

    # load clip model
    clip_model, _, _ = open_clip.create_model_and_transforms(args.clip_name, img_size, pretrained="openai")
    clip_model.eval()


    # load dino model from local path
    dinov3_model_path = args.dinov3_model_path
    
    # 使用transformers加载本地模型
    dino_processor = AutoImageProcessor.from_pretrained(dinov3_model_path)
    processor = AutoImageProcessor.from_pretrained(dinov3_model_path)
    dino_model = AutoModel.from_pretrained(
        dinov3_model_path,
        # device_map="auto",
        # torch_dtype=torch.float16  # 可选：使用半精度
    )
    dino_model.eval()

    # #this parameter are not used in our model, they just use to make VVCLIP to be built successfully.
    # VVCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    # #introduing VV-attention mechanism ONLY use its visual encoder
    # model, _ = VVCLIP_lib.load("ViT-L-14", device=device, design_details = VVCLIP_parameters)
    
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    # model.eval()


    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    #introduing Q-former from BLIP-diffusion to extract cls tokens
    blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
        args.blip_model_path, torch_dtype=torch.float32
    ).to(device)


    results = {}
    metrics = {}

    
    dino_model.to(device)
    clip_model.to(device)

    # model.visual.DAPM_replace(DPAM_layer = 20)


    padding = tokenizer("").to(device)
    repersent_vec = {}
    visual_feature_bank_1 = {}
    visual_feature_bank_2 = {}
    soft_prompt_list = {}
    optimizer_list = {}
    cos_loss = nn.CosineSimilarity(dim=2)
    criterion = nn.CrossEntropyLoss().to(device)
    best_pixel_auroc = 0
    best_result = None

    #obtain embedding of manual prompt
    with torch.no_grad():
        text_prompts, text_prompts_list = encode_text_with_prompt_ensemble(clip_model, ['object'], tokenizer, device, dataset = args.dataset)

    # 初始化两个独立的视觉投影模块（DINOv3 1024 -> CLIP 768）
    # image_proj: 用于 image_embedding (CLS token) -> 生成 soft prompt
    image_proj = prompt_generator.VisualProjection(vis_dim=args.vis_dim, output_dim=args.output_dim)
    image_proj = image_proj.to(device)
    
    # patch_proj: 用于 patch_embedding (patch tokens) -> patch-level 对齐和异常检测
    patch_proj = prompt_generator.VisualProjection(vis_dim=args.vis_dim, output_dim=args.output_dim)
    patch_proj = patch_proj.to(device)
    
    # 加载预训练的投影层权重并设置微调模式
    if args.pretrained_proj_path and os.path.exists(args.pretrained_proj_path):
        logger.info(f"🔄 Loading pretrained projection layers from: {args.pretrained_proj_path}")
        checkpoint = torch.load(args.pretrained_proj_path, map_location=device, weights_only=False)
        image_proj.load_state_dict(checkpoint['image_proj_state_dict'])
        patch_proj.load_state_dict(checkpoint['patch_proj_state_dict'])
        
        epoch_info = checkpoint.get('epoch', 'unknown')
        loss_value = checkpoint.get('best_loss', checkpoint.get('train_loss', checkpoint.get('loss', None)))
        auroc_value = checkpoint.get('best_auroc', checkpoint.get('val_auroc', None))
        
        logger.info(f"✅ Successfully loaded pretrained projection layers from epoch {epoch_info}")
        if loss_value is not None:
            logger.info(f"   Pretrained loss: {loss_value:.4f}")
        if auroc_value is not None:
            logger.info(f"   Pretrained AUROC: {auroc_value:.4f}")
        
        print(f"✅ Loaded pretrained projection layers from epoch {epoch_info}")
        if auroc_value is not None:
            print(f"   AUROC: {auroc_value:.4f}")
    else:
        logger.error(f"❌ Pretrained projection path not found: {args.pretrained_proj_path}")
        logger.error("   Projection layers must be pretrained and provided!")
        print(f"❌ Error: Pretrained projection path not found: {args.pretrained_proj_path}")
        print("   Please provide a valid pretrained projection path!")
        return
    
    # 🔧 冻结投影层（不训练）
    logger.info("🧊 Freezing projection layers (image_proj, patch_proj); only training soft_prompt")
    image_proj.eval()
    patch_proj.eval()
    for param in image_proj.parameters():
        param.requires_grad = False
    for param in patch_proj.parameters():
        param.requires_grad = False
    print("🧊 Projection layers frozen - only training soft_prompt")
    
    # 初始化 SoftPrompt（不再包含视觉投影层）
    soft_prompt = prompt_generator.SoftPrompt(query_dim=args.query_dim, output_dim=args.output_dim)
    soft_prompt = soft_prompt.to(device)

    # 优化器：为不同模块设置不同学习率
    logger.info("📊 Optimizer: Training soft_prompt only (projection layers frozen)")
    
    optimizer = optim.Adam([
        {'params': soft_prompt.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay}
    ])
    
    print(f"📊 Learning rates:")
    print(f"   - Soft prompt: {args.learning_rate:.2e}")
    # 修复：eta_min应该远小于初始学习率，避免学习率增大导致训练崩溃
    scheduler=lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
    
    # 统计可训练参数数量
    total_params = sum(p.numel() for p in image_proj.parameters()) + \
                   sum(p.numel() for p in patch_proj.parameters()) + \
                   sum(p.numel() for p in soft_prompt.parameters())
    trainable_params = sum(p.numel() for p in image_proj.parameters() if p.requires_grad) + \
                       sum(p.numel() for p in patch_proj.parameters() if p.requires_grad) + \
                       sum(p.numel() for p in soft_prompt.parameters() if p.requires_grad)
    
    logger.info("="*80)
    logger.info("MODEL PARAMETERS")
    logger.info("="*80)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"Frozen parameters: {total_params - trainable_params:,} ({(total_params-trainable_params)/total_params*100:.2f}%)")
    logger.info("="*80)
    print(f"\n{'='*80}")
    print(f"Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  Frozen: {total_params - trainable_params:,} ({(total_params-trainable_params)/total_params*100:.2f}%)")
    print(f"{'='*80}\n")

    #training
    logger.info("Starting Training...")
    for epoch in tqdm(range(epochs), desc="Training Epochs", position=0):
        # 确保投影层保持在训练模式（微调）
        image_proj.train()
        patch_proj.train()
        
        # 获取当前学习率用于日志记录
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*50}\nEpoch {epoch+1}/{epochs} | LR: {current_lr:.2e}\n{'='*50}")
        logger.info(f"Starting Epoch {epoch+1}/{epochs} | Learning Rate: {current_lr:.2e}")
        results = {}
        metrics = {}

        random.shuffle(obj_list)
        
        # 用于跟踪每个epoch的平均损失
        epoch_losses = []
        epoch_text_losses = []
        epoch_patch_losses = []
        epoch_fg_losses = []
        
        for obj in tqdm(obj_list, desc="Training Objects", position=1, leave=False):
            results[obj] = {}
            results[obj]['gt_sp'] = []
            results[obj]['pr_sp'] = []
            results[obj]['imgs_masks'] = []
            results[obj]['anomaly_maps'] = []
            metrics[obj] = {}
            metrics[obj]['pixel-auroc'] = 0
            metrics[obj]['pixel-aupro'] = 0
            metrics[obj]['image-auroc'] = 0
            metrics[obj]['image-ap'] = 0

            if args.dataset == 'mvtec':
                data_path = args.mvtec_data_root + obj + '/train'
            else:
                data_path = args.visa_data_root + obj + '/train/good/'
            
            #fetch few-shot data
            for i in tqdm(range(shot), desc=f"  Training Shots ({obj})", position=2, leave=False):
                if args.dataset == 'mvtec':
                    # cond_image = load_image(data_path + '/good/00' + str(i * 3) + '.png')
                    # print(f"{data_path}/good/{i*3:03d}.png")
                    cond_image = load_image(f"{data_path}/good/{i*args.sample_stride:03d}.png")
                else:
                    # cond_image = load_image(data_path + '000' + str(i * 3) + '.JPG')  
                    # print(f"{data_path}/{i*3:04d}.JPG")
                    cond_image = load_image(f"{data_path}/{i*args.sample_stride:03d}.JPG")            
                reference_image = blip_diffusion_pipe.image_processor.preprocess(
                            cond_image, do_resize=True, image_mean=blip_diffusion_pipe.config.mean, image_std=blip_diffusion_pipe.config.std, return_tensors="pt"
                        )["pixel_values"]
                reference_image = reference_image.to(device)


                with torch.no_grad():
                    query = blip_diffusion_pipe.get_query_embeddings(reference_image, ['object']*10)
                    query = query.mean(dim=0).unsqueeze(dim=0)

                    # Use DINOv3 on original PIL image (cond_image) for correct preprocessing
                    dino_out = dinov3_encode_image(reference_image, dino_processor, dino_model, device=device, layer_indices=features_list)
                    image_embedding = dino_out["cls"]  # [1, D], already normalized
                    
                    # Get multi-layer features
                    if "multi_layer_features" in dino_out:
                        patch_embedding = dino_out["multi_layer_features"]  # List of [1, 1+P, D=1024]
                    else:
                        # Fallback: use last layer features repeated
                        combined = torch.cat([image_embedding.unsqueeze(1), dino_out["patch_flat"]], dim=1)  # [1, 1+P, D=1024]
                        patch_embedding = [combined] * len(features_list)
                    
                    grid_h, grid_w = dino_out["grid_size"].tolist()

                # Clone tensors to make them trainable (remove inference mode restriction)
                query_trainable = query.clone()
                image_embedding_trainable = image_embedding.clone()
                
                # 使用 image_proj 投影 CLS token (1024 -> 768) 用于生成 soft prompt
                image_embedding_768 = image_proj(image_embedding_trainable)
                
                # 使用投影后的特征生成 soft prompt
                pos_query, neg_query = soft_prompt(query_trainable, image_embedding_768)
                
                # 使用 patch_proj 投影 Patch embeddings (1024 -> 768) 用于 patch-level 对齐
                patch_embedding_768 = []
                for layer_feat in patch_embedding:
                    layer_feat_trainable = layer_feat.clone()  # [1, 1+P, 1024]
                    batch_size, num_tokens, dim_1024 = layer_feat_trainable.shape
                    # 使用 patch_proj 投影层
                    layer_768 = patch_proj(layer_feat_trainable.view(-1, dim_1024)).view(batch_size, num_tokens, 768)
                    patch_embedding_768.append(layer_768)
                patch_embedding = patch_embedding_768  # 用投影后的替换

                
                pos_query_embedding, pos_token = clip_model.encode_text_prompt(pos_query, padding, device)
                neg_query_embedding, neg_token = clip_model.encode_text_prompt(neg_query, padding, device)
                pos_token = pos_token / pos_token.norm(dim = -1, keepdim = True)
                neg_token = neg_token / neg_token.norm(dim = -1, keepdim = True)
                pos_query_embedding = pos_query_embedding / pos_query_embedding.norm(dim = -1, keepdim = True)
                neg_query_embedding = neg_query_embedding / neg_query_embedding.norm(dim = -1, keepdim = True)

                # text-prompt alignment
                p_ptext_sim = torch.dot(pos_query_embedding[0], text_prompts['object'][:,0]) / (pos_query_embedding[0].norm() * text_prompts['object'][:,0].norm())
                n_ptext_sim = torch.dot(neg_query_embedding[0], text_prompts['object'][:,0]) / (neg_query_embedding[0].norm() * text_prompts['object'][:,0].norm())
                p_ntext_sim = torch.dot(pos_query_embedding[0], text_prompts['object'][:,1]) / (pos_query_embedding[0].norm() * text_prompts['object'][:,1].norm())
                n_ntext_sim = torch.dot(neg_query_embedding[0], text_prompts['object'][:,1]) / (neg_query_embedding[0].norm() * text_prompts['object'][:,1].norm())
                text_loss = ((1 - p_ptext_sim) + (1 - n_ntext_sim) + n_ptext_sim + p_ntext_sim) / 4
                
                patch_loss = 0
                fg_pos_it_patch = 0
                fg_pos_ti_patch = 0
                fg_neg_it_patch = 0
                fg_neg_ti_patch = 0
                thhold = args.threshold


                for i in range(4):
                    # 修复：降低噪声系数，避免后期训练不稳定
                    noise = torch.randn_like(patch_embedding[i]) * args.noise_scale
                    patch_embedding[i] = patch_embedding[i] / patch_embedding[i].norm(dim = -1, keepdim = True)

                    neg_patch_embedding = patch_embedding[i - 1] + patch_embedding[i] + noise
                    neg_patch_embedding = neg_patch_embedding / neg_patch_embedding.norm(dim = -1, keepdim = True)
                    
                    # patch-prompt alignment
                    p_ppatch_sim = cos_loss(patch_embedding[i][:, 1:, :], pos_query_embedding[0]).mean().mean()
                    n_ppatch_sim = cos_loss(patch_embedding[i][:, 1:, :], neg_query_embedding[0]).mean().mean()
                    n_npatch_sim = cos_loss(neg_patch_embedding[:, 1:, :], neg_query_embedding[0]).mean().mean()
                    p_npatch_sim = cos_loss(neg_patch_embedding[:, 1:, :], pos_query_embedding[0]).mean().mean()
                    patch_loss += ((1 - p_ppatch_sim) + (1 - n_npatch_sim) + n_ppatch_sim + p_npatch_sim) / 4

                    # patch-token alignment
                    pos_similarity = torch.einsum('btd,bpd->btp', pos_token, patch_embedding[i][:, 1:, :])
                    pos_similarity = (pos_similarity - torch.min(pos_similarity, dim = -1, keepdim = True)[0]) / (torch.max(pos_similarity, dim = -1, keepdim = True)[0] - torch.min(pos_similarity, dim = -1, keepdim = True)[0])
                    pos_similarity = torch.where(pos_similarity < thhold, 0.0, pos_similarity)
                    pos_weights = pos_similarity / torch.sum(pos_similarity, dim=-1).T
                    pos_group_embed = torch.einsum('btp,bpd->btd', pos_weights, patch_embedding[i][:, 1:, :])
                    pos_group_embed = pos_group_embed / pos_group_embed.norm(dim = -1, keepdim = True)

                    pos_it_logits = torch.einsum('btd,bpd->btp', pos_group_embed, pos_token).squeeze(dim=0)
                    pos_it_labels = torch.eye(pos_it_logits.shape[1]).to(device)
                    pos_ti_logits = torch.einsum('btd,bpd->btp', pos_token, pos_group_embed).squeeze(dim=0)
                    pos_ti_labels = torch.eye(pos_ti_logits.shape[1]).to(device)
                    fg_pos_it_patch += criterion(pos_it_logits, pos_it_labels)
                    fg_pos_ti_patch += criterion(pos_ti_logits, pos_ti_labels)

                    neg_similarity = torch.einsum('btd,bpd->btp', neg_token, neg_patch_embedding[:, 1:, :])
                    neg_similarity = (neg_similarity - torch.min(neg_similarity, dim = -1, keepdim = True)[0]) / (torch.max(neg_similarity, dim = -1, keepdim = True)[0] - torch.min(neg_similarity, dim = -1, keepdim = True)[0])
                    neg_similarity = torch.where(neg_similarity < thhold, 0.0, neg_similarity)
                    neg_weights = neg_similarity / torch.sum(neg_similarity, dim=-1).T

                    neg_group_embed = torch.einsum('btp,bpd->btd', neg_weights, neg_patch_embedding[:, 1:, :])
                    neg_group_embed = neg_group_embed / neg_group_embed.norm(dim = -1, keepdim = True)

                    neg_it_logits = torch.einsum('btd,bpd->btp', neg_group_embed, neg_token).squeeze(dim=0)
                    neg_it_labels = torch.eye(neg_it_logits.shape[1]).to(device)
                    neg_ti_logits = torch.einsum('btd,bpd->btp', neg_token, neg_group_embed).squeeze(dim=0)

                    neg_ti_labels = torch.eye(neg_ti_logits.shape[1]).to(device)
                    fg_neg_it_patch += criterion(neg_it_logits, neg_it_labels)
                    fg_neg_ti_patch += criterion(neg_ti_logits, neg_ti_labels)
                    
                patch_loss /= 4.0
                fg_pos_it_patch /= 4.0
                fg_pos_ti_patch /= 4.0
                fg_neg_it_patch /= 4.0
                fg_neg_ti_patch /= 4.0
                fg_loss = (fg_pos_it_patch + fg_pos_ti_patch + fg_neg_it_patch + fg_neg_ti_patch) / 4.0
                
                loss = text_loss + args.patch_loss_weight * patch_loss + fg_loss 

                optimizer.zero_grad()
                loss.backward()
                # 添加梯度裁剪，防止梯度爆炸（包含所有可训练参数）
                torch.nn.utils.clip_grad_norm_(
                    list(soft_prompt.parameters()), 
                    max_norm=args.grad_clip_norm
                )
                optimizer.step()
                
                # 记录损失值
                epoch_losses.append(loss.item())
                epoch_text_losses.append(text_loss.item())
                epoch_patch_losses.append(patch_loss.item())
                epoch_fg_losses.append(fg_loss.item())
        
        # 记录epoch平均损失
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        avg_text_loss = sum(epoch_text_losses) / len(epoch_text_losses) if epoch_text_losses else 0
        avg_patch_loss = sum(epoch_patch_losses) / len(epoch_patch_losses) if epoch_patch_losses else 0
        avg_fg_loss = sum(epoch_fg_losses) / len(epoch_fg_losses) if epoch_fg_losses else 0
        
        # 打印详细的损失信息
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs} Loss Summary:")
        print(f"  Total Loss:  {avg_loss:.4f}")
        print(f"  Text Loss:   {avg_text_loss:.4f} (weight: 1.0)")
        print(f"  Patch Loss:  {avg_patch_loss:.4f} (weight: {args.patch_loss_weight}, weighted: {avg_patch_loss*args.patch_loss_weight:.4f})")
        print(f"  FG Loss:     {avg_fg_loss:.4f} (weight: 1.0)")
        print(f"{'='*80}\n")
        
        # 记录到日志
        logger.info(f"Epoch {epoch+1}/{epochs} | Training Loss:")
        logger.info(f"  Total Loss:  {avg_loss:.4f}")
        logger.info(f"  Text Loss:   {avg_text_loss:.4f}")
        logger.info(f"  Patch Loss:  {avg_patch_loss:.4f} (weighted: {avg_patch_loss*args.patch_loss_weight:.4f})")
        logger.info(f"  FG Loss:     {avg_fg_loss:.4f}")
        
        scheduler.step()
        
        # 保存当前epoch的模型（不包含bank数据，因为bank数据在测试后才生成）
        epoch_checkpoint = {
            'epoch': epoch + 1,
            'image_proj_state_dict': image_proj.state_dict(),
            'patch_proj_state_dict': patch_proj.state_dict(),
            'soft_prompt_state_dict': soft_prompt.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_loss,
            'text_loss': avg_text_loss,
            'patch_loss': avg_patch_loss,
            'fg_loss': avg_fg_loss,
            'learning_rate': current_lr,
            'args': vars(args)
        }
        
        # 保存当前epoch的checkpoint（仅模型权重）
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
        torch.save(epoch_checkpoint, epoch_checkpoint_path)
        logger.info(f"💾 Saved epoch {epoch+1} checkpoint: {epoch_checkpoint_path}")
        print(f"💾 Saved epoch {epoch+1} checkpoint")
        

        #building memory bank
        print("\nBuilding Memory Bank...")
        with torch.no_grad():
            shot_represents_bank = []
            pos_shot_represents = []
            neg_shot_represents = []
            visual_feature_bank_1['object'] = []
            visual_feature_bank_2['object'] = []
            query_bank = []
            for obj in tqdm(obj_list, desc="Memory Bank Objects", position=1, leave=False):
                if args.dataset == 'mvtec':
                    data_path = args.mvtec_data_root + obj + '/train'
                else:
                    data_path = args.visa_data_root + obj + '/train/good/'
                shot_represents = []
                for i in tqdm(range(shot), desc=f"  Processing Shots ({obj})", position=2, leave=False):
                    if args.dataset == 'mvtec':
                        # cond_image = load_image(data_path + '/good/00' + str(i * 3) + '.png')
                        # print(f"{data_path}/good/{i*3:03d}.png")
                        cond_image = load_image(f"{data_path}/good/{i*args.sample_stride:03d}.png")
                    else:
                        # cond_image = load_image(data_path + '000' + str(i * 3) + '.JPG')  
                        # print(f"{data_path}/{i*3:04d}.JPG")
                        cond_image = load_image(f"{data_path}/{i*args.sample_stride:04d}.JPG")
                    reference_image = blip_diffusion_pipe.image_processor.preprocess(
                                cond_image, image_mean=blip_diffusion_pipe.config.mean, image_std=blip_diffusion_pipe.config.std, return_tensors="pt"
                            )["pixel_values"]
                    reference_image = reference_image.to(device)
                    query = blip_diffusion_pipe.get_query_embeddings(reference_image, ['object']*10)
                    query = query.mean(dim=0).unsqueeze(dim=0)

                    # DINOv3 features for memory bank
                    dino_out = dinov3_encode_image(reference_image, dino_processor, dino_model, device=device, layer_indices=features_list)
                    image_embedding = dino_out["cls"]  # [1, D=1024]
                    
                    # Get multi-layer features
                    if "multi_layer_features" in dino_out:
                        patch_token_memory = dino_out["multi_layer_features"]  # List of [1, 1+P, D=1024]
                    else:
                        # Fallback: use last layer features repeated
                        combined = torch.cat([image_embedding.unsqueeze(1), dino_out["patch_flat"]], dim=1)
                        patch_token_memory = [combined] * len(features_list)
                    
                    # 使用 image_proj 投影 CLS token (1024 -> 768) 用于生成 soft prompt
                    image_embedding_768 = image_proj(image_embedding)
                    
                    # 使用投影后的特征生成 soft prompt
                    pos_prompt_query, neg_prompt_query = soft_prompt(query, image_embedding_768)
                    
                    # 使用 patch_proj 投影 Patch embeddings (1024 -> 768) 用于后续处理
                    patch_embedding = []
                    for layer_feat in patch_token_memory:
                        batch_size, num_tokens, dim_1024 = layer_feat.shape
                        layer_768 = patch_proj(layer_feat.view(-1, dim_1024)).view(batch_size, num_tokens, 768)
                        patch_embedding.append(layer_768)
                    pos_query_embedding, _ = clip_model.encode_text_prompt(pos_prompt_query, padding, device)
                    neg_query_embedding, _ = clip_model.encode_text_prompt(neg_prompt_query, padding, device)

                    visual_feature_bank_1['object'].append(patch_token_memory[0][0][1:])
                    visual_feature_bank_2['object'].append(patch_token_memory[2][0][1:])
  
                    pos_shot_represents.append(pos_query_embedding)
                    neg_shot_represents.append(neg_query_embedding)
                    query /= query.norm(dim=-1, keepdim=True)
                    query_bank.append(query)

                    shot_represents_bank.append(pos_query_embedding)
                    shot_represents_bank.append(neg_query_embedding)

            visual_feature_bank_1['object'] = torch.stack(visual_feature_bank_1['object'], dim=0)
            visual_feature_bank_2['object'] = torch.stack(visual_feature_bank_2['object'], dim=0)
            visual_feature_bank_1['object'] = F.normalize(visual_feature_bank_1['object'], dim=-1)
            visual_feature_bank_2['object'] = F.normalize(visual_feature_bank_2['object'], dim=-1)

            shot_represents_bank = torch.stack(shot_represents_bank,dim=0).view(-1, 2, 768)
            query_bank = torch.vstack(query_bank)
            shot_represents_bank /= shot_represents_bank.norm(dim = -1, keepdim = True)

            pos_shot_represents = torch.vstack(pos_shot_represents)
            neg_shot_represents = torch.vstack(neg_shot_represents)

            pos_shot_represents = pos_shot_represents.mean(dim = 0)
            neg_shot_represents = neg_shot_represents.mean(dim = 0)
            pos_shot_represents /= pos_shot_represents.norm(dim = -1, keepdim = True)
            neg_shot_represents /= neg_shot_represents.norm(dim = -1, keepdim = True)
            shot_represents = text_prompts['object'].clone().T
            shot_represents[0] = pos_shot_represents
            shot_represents[1] = neg_shot_represents
            shot_represents = shot_represents.T


        # 检查是否需要测试（每test_interval个epoch测试一次，或最后一个epoch）
        should_test = (epoch + 1) % args.test_interval == 0 or (epoch + 1) == epochs
        
        if not should_test:
            print(f"\n⏭️  Skipping test for epoch {epoch+1}/{epochs} (test every {args.test_interval} epochs)")
            logger.info(f"Skipping test for epoch {epoch+1}/{epochs}")
            continue
        
        #testing 
        print("\nTesting...")
        dino_model.to(device)
        
        # 计算动态温度：随epoch增加而增大
        current_temperature = args.temperature + epoch * args.temperature_scale
        print(f"📊 Current epoch: {epoch+1}, Dynamic temperature: {current_temperature:.3f}")
        for idx, items in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing Images", position=1, leave=False):
            image = items['img'].to(device)
            cls_name = items['cls_name']
            cls_id = items['cls_id']
            gt_mask = items['img_mask']
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
            results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())
            with torch.no_grad():
                # Use DINOv3 on batch tensor with multi-layer features
                dino_out = dinov3_encode_image(image, dino_processor, dino_model, device=device, layer_indices=features_list)
                image_features = dino_out["cls"]  # [B, D=1024], already normalized
                
                # Get multi-layer features
                if "multi_layer_features" in dino_out:
                    patch_token_memory = dino_out["multi_layer_features"]  # List of [B, 1+P, D=1024]
                else:
                    # Fallback: use last layer features repeated
                    combined = torch.cat([image_features.unsqueeze(1), dino_out["patch_flat"]], dim=1)
                    patch_token_memory = [combined] * len(features_list)
                
                # 使用 patch_proj 投影 Patch features 到 768-dim CLIP 空间
                patch_features = []
                for layer_feat in patch_token_memory:
                    batch_size, num_tokens, dim_1024 = layer_feat.shape
                    layer_768 = patch_proj(layer_feat.view(-1, dim_1024)).view(batch_size, num_tokens, 768)
                    patch_features.append(layer_768)
                
                grid_h, grid_w = dino_out["grid_size"].tolist()

                # Load original image for BLIP preprocessing (BLIP requires specific input size)
                img_path = items['img_path'][0]  # Get the first (and only) image path from batch
                pil_image = load_image(img_path)
                blip_image = blip_diffusion_pipe.image_processor.preprocess(
                    pil_image, do_resize=True, 
                    image_mean=blip_diffusion_pipe.config.mean, 
                    image_std=blip_diffusion_pipe.config.std, 
                    return_tensors="pt"
                )["pixel_values"].to(device)
                
                query = blip_diffusion_pipe.get_query_embeddings(blip_image, ['object']*10)
                query = query.mean(dim=0).unsqueeze(dim=0)

                # 使用 image_proj 投影 CLS token (1024 -> 768) 用于生成 soft prompt
                image_features_768 = image_proj(image_features)
                
                # 使用投影后的特征生成 soft prompt
                pos_prompt_query, neg_prompt_query = soft_prompt(query, image_features_768)

                pos_query_embedding, _ = clip_model.encode_text_prompt(pos_prompt_query, padding, device)
                neg_query_embedding, _ = clip_model.encode_text_prompt(neg_prompt_query, padding, device)

                pos_token = pos_token / pos_token.norm(dim = -1, keepdim = True)
                neg_token = neg_token / neg_token.norm(dim = -1, keepdim = True)
                pos_query_embedding = pos_query_embedding / pos_query_embedding.norm()
                neg_query_embedding = neg_query_embedding / neg_query_embedding.norm()

                query /= query.norm(dim=-1, keepdim=True)
                query = query.expand(len(obj_list) * shot, -1, -1)
                query_sim = torch.mean(torch.sum(torch.mul(query_bank, query), dim=-1), dim = -1)
                
                obj_idx = torch.topk(query_sim, k=shot)[1].cpu().numpy().tolist()
                cur_visual_feature_bank_1 = visual_feature_bank_1['object'][obj_idx].view(-1, 1024)
                cur_visual_feature_bank_2 = visual_feature_bank_2['object'][obj_idx].view(-1, 1024)


                text_features = shot_represents_bank.clone()

                # Use projected 768-dim features for text-image comparison
                text_probs = torch.matmul(text_features, image_features_768.T).permute(0,2,1)
                text_probs = (text_probs).softmax(-1).view(len(obj_list) * shot, -1)
                text_probs = text_probs[obj_idx] + query_sim[obj_idx].view(len(obj_idx), -1)

                text_probs = text_probs.softmax(0)

                cur_text_features = text_features[obj_idx].clone().mean(dim=0)

                cur_text_features[0] = text_features[obj_idx][:, 0, :].clone().mean(dim=0)
                cur_text_features[1] = text_features[obj_idx][:, 1, :].clone().mean(dim=0)


                # Use projected 768-dim features for text-image comparison
                text_probs = torch.matmul(text_features, image_features_768.T).permute(0,2,1)
                text_probs = (text_probs/current_temperature).softmax(-1).view(len(obj_list) * shot, -1)
                text_probs = text_probs[obj_idx].mean(dim=0)

                cur_text_features[0] = (cur_text_features[0].unsqueeze(dim=0) + pos_query_embedding) / 2
                cur_text_features[1] = (cur_text_features[1].unsqueeze(dim=0) + neg_query_embedding) / 2


                cur_text_features[0] = cur_text_features[0] / cur_text_features[0].norm()
                cur_text_features[1] = cur_text_features[1] / cur_text_features[1].norm()
                cur_text_features = cur_text_features.T
                
                anomaly_map_list = []

                for idx, patch_feature in enumerate(patch_features):
                    if idx >= args.feature_map_layer[0]:
                        patch_feature = patch_feature / patch_feature.norm(dim = -1, keepdim = True)
                        similarity, _ = compute_similarity(patch_feature, cur_text_features.T, current_temperature)
                        similarity_map = similarity[:, 1:, :]
                        similarity_map = similarity_map.reshape(similarity_map.shape[0], grid_h, grid_w, -1).permute(0, 3, 1, 2)
                        similarity_map = similarity_map.permute(0, 2, 3, 1)
                        anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
                        anomaly_map_list.append(anomaly_map)

                        
                vis_feature_1 = patch_token_memory[0][0][1:]
                vis_feature_1 = vis_feature_1 / vis_feature_1.norm(dim=-1, keepdim=True)
                vis_feature_2 = patch_token_memory[2][0][1:]
                vis_feature_2 = vis_feature_2 / vis_feature_2.norm(dim=-1, keepdim=True)

                score1, _ = (1.0 - vis_feature_1 @ cur_visual_feature_bank_1.t()).min(dim=-1)
                score1 /= 2.0

                score2, _ = (1.0 - vis_feature_2 @ cur_visual_feature_bank_2.t()).min(dim=-1)
                score2 /= 2.0
                score = score1 + score2
                vis_score = score.reshape(1,1,grid_h,grid_w)
                
                anomaly_map = torch.stack(anomaly_map_list)
                textual_anomaly_map = anomaly_map.sum(dim = 0) / 4.0
                textual_anomaly_map = textual_anomaly_map.reshape(1,1,grid_h,grid_w)

                anomaly_map = textual_anomaly_map + vis_score 

                # 添加标签信息（在计算之前获取）
                gt_label = items['anomaly'][0].item()  # 0是正常，1是异常
                label_str = "NORMAL" if gt_label == 0 else "ABNORMAL"
                
                # 保存原始text_probs用于打印
                original_text_probs = text_probs[0].item()
                
                # 策略一 mean + max - 分步计算便于调试
                term1 = -text_probs[0].unsqueeze(dim=0)
                term2 = -query_sim[obj_idx].mean(dim=0)
                term3 = torch.max(textual_anomaly_map)
                term4 = torch.max(vis_score)
                text_probs =  term1 + term2 + term3 + term4
                
                print(f"\nepoch: {epoch+1}")
                print("cls_name: ", cls_name[0])
                print(f"[{label_str}] Image Components:")
                print(f"  term1 (-text_probs[0]):         {term1.item():.4f}")
                print(f"  term2 (-query_sim.mean()):      {term2.item():.4f}")
                print(f"  term3 (max textual_anomaly):    {term3.item():.4f}")
                print(f"  term4 (max vis_score):          {term4.item():.4f}")
                print(f"  Sum (should be final):          {(term1 + term2 + term3 + term4).item():.4f}")
                print(f"  Final text_probs:               {text_probs.item():.4f}")
                # 策略二 mean + mean (修正版)
                # text_probs = -text_probs[0].unsqueeze(dim=0) - query_sim[obj_idx].mean(dim=0) + textual_anomaly_map.mean() + vis_score.mean()
                            
               
                text_probs = text_probs.view(1)

                anomaly_map = F.interpolate(anomaly_map, size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(0)
                results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
                anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )
                results[cls_name[0]]['anomaly_maps'].append(anomaly_map)

        print("\nCalculating Metrics...")
        table_ls = []
        image_auroc_list = []
        image_ap_list = []
        pixel_auroc_list = []
        pixel_aupro_list = []
        for obj in tqdm(obj_list, desc="Calculating Metrics", position=1, leave=False):
            table = []
            table.append(obj)
            results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
            results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
            if args.metrics == 'image-level':
                image_auroc = image_level_metrics(results, obj, "image-auroc")
                image_ap = image_level_metrics(results, obj, "image-ap")
                table.append(str(np.round(image_auroc * 100, decimals=1)))
                table.append(str(np.round(image_ap * 100, decimals=1)))
                image_auroc_list.append(image_auroc)
                image_ap_list.append(image_ap) 
            elif args.metrics == 'pixel-level':
                pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                table.append(str(np.round(pixel_auroc * 100, decimals=1)))
                table.append(str(np.round(pixel_aupro * 100, decimals=1)))
                pixel_auroc_list.append(pixel_auroc)
                pixel_aupro_list.append(pixel_aupro)
            elif args.metrics == 'image-pixel-level':
                image_auroc = image_level_metrics(results, obj, "image-auroc")
                image_ap = image_level_metrics(results, obj, "image-ap")
                pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                table.append(str(np.round(pixel_auroc * 100, decimals=1)))
                table.append(str(np.round(pixel_aupro * 100, decimals=1)))
                table.append(str(np.round(image_auroc * 100, decimals=1)))
                table.append(str(np.round(image_ap * 100, decimals=1)))
                image_auroc_list.append(image_auroc)
                image_ap_list.append(image_ap) 
                pixel_auroc_list.append(pixel_auroc)
                pixel_aupro_list.append(pixel_aupro)
            table_ls.append(table)

        if args.metrics == 'image-level':
            # logger
            table_ls.append(['mean', 
                            str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                            str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
            results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
        elif args.metrics == 'pixel-level':
            # logger
            table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                            str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                        ])
            results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
        elif args.metrics == 'image-pixel-level':
            # logger
            table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                            str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)), 
                            str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                            str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
            results = tabulate(table_ls, headers=['objects','image_auroc', 'image_ap','pixel_aupro', 'pixel_auroc'], tablefmt="pipe")
        current_pixel_auroc = np.mean(pixel_auroc_list) * 100
        if current_pixel_auroc > best_pixel_auroc:
            best_pixel_auroc = current_pixel_auroc
            best_result = results
            
            # 保存最佳模型
            best_checkpoint = {
                'epoch': epoch + 1,
                'image_proj_state_dict': image_proj.state_dict(),
                'patch_proj_state_dict': patch_proj.state_dict(),
                'soft_prompt_state_dict': soft_prompt.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_pixel_auroc': best_pixel_auroc,
                'current_pixel_auroc': current_pixel_auroc,
                'train_loss': avg_loss,
                'text_loss': avg_text_loss,
                'patch_loss': avg_patch_loss,
                'fg_loss': avg_fg_loss,
                'learning_rate': current_lr,
                'args': vars(args)
            }
            
            # 保存最佳模型的bank数据
            best_bank_data = {
                'visual_feature_bank_1': visual_feature_bank_1,
                'visual_feature_bank_2': visual_feature_bank_2,
                'shot_represents_bank': shot_represents_bank,
                'query_bank': query_bank,
                'pos_shot_represents': pos_shot_represents,
                'neg_shot_represents': neg_shot_represents,
                'shot_represents': shot_represents
            }
            best_checkpoint['bank_data'] = best_bank_data
            
            # 保存最佳模型
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(best_checkpoint, best_checkpoint_path)
            logger.info(f"🌟 New best model! AUROC: {best_pixel_auroc:.2f}% - Saved to: {best_checkpoint_path}")
            print(f"🌟 New best model! AUROC: {best_pixel_auroc:.2f}% - Saved to: {best_checkpoint_path}")
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs} Results:")
        print(f"Current Pixel AUROC: {current_pixel_auroc:.2f}%")
        print(f"Best Pixel AUROC: {best_pixel_auroc:.2f}%")
        print(f"{'='*80}\n")
        print(results)
        
        # 立即保存当前epoch的结果到日志
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch+1}/{epochs} Results:")
        logger.info(f"Current Pixel AUROC: {current_pixel_auroc:.2f}%")
        logger.info(f"Best Pixel AUROC: {best_pixel_auroc:.2f}%")
        logger.info(f"{'='*80}")
        logger.info("\n%s", results)
        
        # 重新保存包含bank数据的完整checkpoint
        complete_checkpoint = {
            'epoch': epoch + 1,
            'image_proj_state_dict': image_proj.state_dict(),
            'patch_proj_state_dict': patch_proj.state_dict(),
            'soft_prompt_state_dict': soft_prompt.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_loss,
            'text_loss': avg_text_loss,
            'patch_loss': avg_patch_loss,
            'fg_loss': avg_fg_loss,
            'learning_rate': current_lr,
            'current_pixel_auroc': current_pixel_auroc,
            'best_pixel_auroc': best_pixel_auroc,
            'args': vars(args)
        }
        
        # 保存bank数据
        bank_data = {
            'visual_feature_bank_1': visual_feature_bank_1,
            'visual_feature_bank_2': visual_feature_bank_2,
            'shot_represents_bank': shot_represents_bank,
            'query_bank': query_bank,
            'pos_shot_represents': pos_shot_represents,
            'neg_shot_represents': neg_shot_represents,
            'shot_represents': shot_represents
        }
        complete_checkpoint['bank_data'] = bank_data
        
        # 保存完整的checkpoint（包含bank数据）
        complete_checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_complete.pth')
        torch.save(complete_checkpoint, complete_checkpoint_path)
        logger.info(f"💾 Saved complete epoch {epoch+1} checkpoint with bank data: {complete_checkpoint_path}")
        print(f"💾 Saved complete epoch {epoch+1} checkpoint with bank data")
    
    # 保存最终模型（最后一个epoch）
    final_checkpoint = {
        'epoch': epochs,
        'image_proj_state_dict': image_proj.state_dict(),
        'patch_proj_state_dict': patch_proj.state_dict(),
        'soft_prompt_state_dict': soft_prompt.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_pixel_auroc': current_pixel_auroc,
        'best_pixel_auroc': best_pixel_auroc,
        'args': vars(args)
    }
    
    # 保存最终模型的bank数据
    final_bank_data = {
        'visual_feature_bank_1': visual_feature_bank_1,
        'visual_feature_bank_2': visual_feature_bank_2,
        'shot_represents_bank': shot_represents_bank,
        'query_bank': query_bank,
        'pos_shot_represents': pos_shot_represents,
        'neg_shot_represents': neg_shot_represents,
        'shot_represents': shot_represents
    }
    final_checkpoint['bank_data'] = final_bank_data
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save(final_checkpoint, final_checkpoint_path)
    logger.info(f"💾 Saved final model: {final_checkpoint_path}")
    print(f"💾 Saved final model: {final_checkpoint_path}")
    
    # 保存最终最佳结果
    logger.info(f"\n{'='*80}")
    logger.info("FINAL BEST RESULTS:")
    logger.info(f"Best Pixel AUROC: {best_pixel_auroc:.2f}%")
    logger.info(f"Final Pixel AUROC: {current_pixel_auroc:.2f}%")
    logger.info(f"{'='*80}")
    logger.info("\n%s", best_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VVCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="/data2/zlt/code/abnormal_dataset/mvtec", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/main_mvtec_shot2_re', help='path to save results and checkpoints')
    parser.add_argument("--dinov3_model_path", type=str, default='./model_card/dinov3-vitl16-pretrain-lvd1689m', help='path to DINOv3 model')
    parser.add_argument("--blip_model_path", type=str, default='./model_card', help='path to BLIP-Diffusion model')
    parser.add_argument("--mvtec_data_root", type=str, default='/data2/zlt/code/abnormal_dataset/mvtec/', help='root path for MVTec dataset')
    parser.add_argument("--visa_data_root", type=str, default='/data2/zlt/code/abnormal_dataset/visa_save/1cls/', help='root path for ViSA dataset')
    
    # pretrained projection layers
    parser.add_argument("--pretrained_proj_path",default='model_card/pretrain_visa_for_mvtec/projection_layers_epoch_7.pth', type=str, help='path to pretrained projection layers checkpoint (e.g., ./checkpoints/projection_layers/best_projection_layers.pth)')
    
    # training hyperparameters
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--test_interval", type=int, default=1, help="test every N epochs (default: 5)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--T_max", type=int, default=20, help="T_max for CosineAnnealingLR")
    parser.add_argument("--eta_min", type=float, default=1e-7, help="minimum learning rate for scheduler")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="gradient clipping max norm")
    parser.add_argument("--patch_loss_weight", type=float, default=0.125, help="weight for patch loss")
    parser.add_argument("--noise_scale", type=float, default=2.0, help="scale factor for noise in negative samples")
    parser.add_argument("--threshold", type=float, default=1/256, help="threshold for patch-token alignment")
    # best: 0.07
    parser.add_argument("--temperature", type=float, default=0.2, help="base temperature for softmax in text-image similarity")
    parser.add_argument("--temperature_scale", type=float, default=0.1, help="temperature scaling factor per epoch")
    parser.add_argument("--proj_finetune_ratio", type=float, default=0.01, help="learning rate ratio for projection layers fine-tuning (e.g., 0.01 means 1/100 of main LR)")
    parser.add_argument("--sample_stride", type=int, default=3, help="stride for sampling training images (i*stride)")
    
    # model architecture
    parser.add_argument("--vis_dim", type=int, default=1024, help="visual feature dimension (DINOv3 output)")
    parser.add_argument("--query_dim", type=int, default=768, help="query embedding dimension")
    parser.add_argument("--output_dim", type=int, default=768, help="output embedding dimension (CLIP space)")
    # parser.add_argument("--patch_size", type=int, default=14, help="patch size")
    
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help='dataset name (mvtec or visa)')
    # best: 12, 16, 20, 24
    parser.add_argument("--features_list", type=int, nargs="+", default=[12, 16, 20, 24], help="DINOv3 layer indices for multi-layer features")
    parser.add_argument("--image_size", type=int, default=512, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="depth parameter")
    parser.add_argument("--n_ctx", type=int, default=12, help="context length")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="text context length")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="feature map layers to use")
    parser.add_argument("--metrics", type=str, default='image-pixel-level', help='metrics to compute')
    parser.add_argument("--seed", type=int, default=4, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="sigma for gaussian filter")
    
    parser.add_argument("--shot", type=int, default=2, choices=[1, 2, 4], help="number of shots (1, 2, or 4)")
    
    parser.add_argument("--clip", type=str, default='CLIP', help="CLIP model type")
    parser.add_argument("--clip_name", type=str, default='ViT-L-14-336',  help="CLIP model name")

   
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    main(args)
