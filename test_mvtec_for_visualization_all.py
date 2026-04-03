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

from utils.visualizer import Visualizer, create_visualization_for_batch

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


def compute_similarity(image_features, text_features, t=2):
    prob_1 = image_features[:, :1, :] @ text_features.t()
    b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
    # print(b, n_t, n_i, c)
    feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
    similarity = feats.sum(-1)
    return (similarity/0.07).softmax(-1), prob_1
    # return similarity, prob_1




def main(args):
    img_size = args.image_size
    features_list = args.features_list  # DINOv3 多层特征索引，例如 [6, 12, 18, 24]
    #few-shot learning parameter
    shot = args.shot

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 记录实验配置
    logger.info("="*80)
    logger.info("TESTING CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Shot: {args.shot}")
    logger.info(f"Image Size: {args.image_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Save Path: {args.save_path}")
    logger.info(f"Model Path: {args.model_path}")
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
    
    # 初始化 SoftPrompt
    soft_prompt = prompt_generator.SoftPrompt(query_dim=args.query_dim, output_dim=args.output_dim)
    soft_prompt = soft_prompt.to(device)
    
    # 加载训练好的最佳模型
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"🔄 Loading trained model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        
        # 加载模型权重
        image_proj.load_state_dict(checkpoint['image_proj_state_dict'])
        patch_proj.load_state_dict(checkpoint['patch_proj_state_dict'])
        soft_prompt.load_state_dict(checkpoint['soft_prompt_state_dict'])
        
        # 加载bank数据
        if 'bank_data' in checkpoint:
            visual_feature_bank_1 = checkpoint['bank_data']['visual_feature_bank_1']
            visual_feature_bank_2 = checkpoint['bank_data']['visual_feature_bank_2']
            shot_represents_bank = checkpoint['bank_data']['shot_represents_bank']
            query_bank = checkpoint['bank_data']['query_bank']
            pos_shot_represents = checkpoint['bank_data']['pos_shot_represents']
            neg_shot_represents = checkpoint['bank_data']['neg_shot_represents']
            shot_represents = checkpoint['bank_data']['shot_represents']
        else:
            logger.warning("⚠️  No bank data found in checkpoint, will build from scratch")
            visual_feature_bank_1 = {}
            visual_feature_bank_2 = {}
            shot_represents_bank = None
            query_bank = None
            pos_shot_represents = None
            neg_shot_represents = None
            shot_represents = None
        
        epoch_info = checkpoint.get('epoch', 'unknown')
        best_auroc = checkpoint.get('best_pixel_auroc', checkpoint.get('best_auroc', 'unknown'))
        
        logger.info(f"✅ Successfully loaded trained model from epoch {epoch_info}")
        if best_auroc != 'unknown':
            logger.info(f"   Best AUROC: {best_auroc:.2f}%")
        
        print(f"✅ Loaded trained model from epoch {epoch_info}")
        if best_auroc != 'unknown':
            print(f"   Best AUROC: {best_auroc:.2f}%")
    else:
        logger.error(f"❌ Model path not found: {args.model_path}")
        print(f"❌ Error: Model path not found: {args.model_path}")
        return
    
    # 设置模型为评估模式
    image_proj.eval()
    patch_proj.eval()
    soft_prompt.eval()
    print("🔒 All models set to evaluation mode")

    # 初始化可视化器
    visualizer = None
    visualization_save_path = None
    if args.enable_visualization:
        visualization_save_path = os.path.join(args.save_path, 'visualizations')
        visualizer = Visualizer(visualization_save_path, img_size=img_size)
        print(f"🎨 可视化器已初始化，保存路径: {visualization_save_path}")
        logger.info(f"Visualizer initialized at: {visualization_save_path}")
    else:
        print("🚫 可视化功能已禁用")
        logger.info("Visualization disabled")

    # 如果没有bank数据，需要重新构建
    if shot_represents_bank is None:
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
                        cond_image = load_image(f"{data_path}/good/{i*args.sample_stride:03d}.png")
                    else:
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

    # 初始化结果字典
    results = {}
    metrics = {}
    for obj in obj_list:
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

    # 开始测试
    print("\nTesting...")
    dino_model.to(device)
    
    # 可视化（不再限制每类数量，保存所有样本）
    visualization_count_per_class = {}
    
    for idx, items in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing Images", position=1, leave=False):
        # 🔍 提前检查是否需要跳过可视化（避免不必要的计算节省内存）
        skip_computation = False
        if args.enable_visualization and visualizer is not None and visualization_save_path is not None:
            cls_name = items['cls_name']
            img_path = items['img_path'][0]
            gt_label = items['anomaly'][0].item()  # 0是正常，1是异常
            label_str = "NORMAL" if gt_label == 0 else "ABNORMAL"
            
            # 获取缺陷类型（specie_name），由于batch_size=1，取第一个元素
            defect_type = items.get('specie_name', ['unknown'])[0] if 'specie_name' in items else 'unknown'
            if defect_type == 'good':
                defect_type = 'NORMAL'
            
            # 获取源文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # 构建文件名：类别_缺陷类型_源文件名_标签
            filename = f"{cls_name[0]}_{defect_type}_{base_name}_{label_str}"
            
            # 检查文件是否已存在
            anomaly_map_file = os.path.join(visualization_save_path, 'anomaly_maps', f"{filename}_anomaly.png")
            if os.path.exists(anomaly_map_file):
                print(f"⏭️  跳过已可视化的图像: {filename} (文件已存在，跳过所有计算)")
                skip_computation = True
        
        # 如果文件已存在，跳过所有计算直接进入下一个循环
        if skip_computation:
            continue
        
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
            
            # CLIP 编码也应在 no_grad 上下文中执行（避免内存累积）
            pos_query_embedding, pos_token = clip_model.encode_text_prompt(pos_prompt_query, padding, device)
            neg_query_embedding, neg_token = clip_model.encode_text_prompt(neg_prompt_query, padding, device)

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
        text_probs = (text_probs/args.temperature).softmax(-1).view(len(obj_list) * shot, -1)
        text_probs = text_probs[obj_idx].mean(dim=0)

        cur_text_features = text_features[obj_idx].clone().mean(dim=0)
        cur_text_features[0] = text_features[obj_idx][:, 0, :].clone().mean(dim=0)
        cur_text_features[1] = text_features[obj_idx][:, 1, :].clone().mean(dim=0)

        cur_text_features[0] = (cur_text_features[0].unsqueeze(dim=0) + pos_query_embedding) / 2
        cur_text_features[1] = (cur_text_features[1].unsqueeze(dim=0) + neg_query_embedding) / 2

        cur_text_features[0] = cur_text_features[0] / cur_text_features[0].norm()
        cur_text_features[1] = cur_text_features[1] / cur_text_features[1].norm()
        cur_text_features = cur_text_features.T
        
        anomaly_map_list = []

        for idx, patch_feature in enumerate(patch_features):
            if idx >= args.feature_map_layer[0]:
                patch_feature = patch_feature / patch_feature.norm(dim = -1, keepdim = True)
                similarity, _ = compute_similarity(patch_feature, cur_text_features.T)
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
        
        print(f"\nTesting image: {idx+1}")
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
        
        # 🔍 添加可视化功能
        # 初始化当前类别的计数器
        current_class = cls_name[0]
        if current_class not in visualization_count_per_class:
            visualization_count_per_class[current_class] = 0
        
        if args.enable_visualization and visualizer is not None:
            # 生成文件名（由于已经通过提前检查，这里直接创建可视化）
            # 获取缺陷类型（specie_name），由于batch_size=1，取第一个元素
            defect_type = items.get('specie_name', ['unknown'])[0] if 'specie_name' in items else 'unknown'
            if defect_type == 'good':
                defect_type = 'NORMAL'
            
            # 获取源文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # 构建文件名：类别_缺陷类型_源文件名_标签
            filename = f"{cls_name[0]}_{defect_type}_{base_name}_{label_str}"
            
            print(f"📊 为类别 {current_class} 的样本 {idx+1} 创建可视化...")
            
            # 收集可视化数据 - 修复数据格式
            # 确保异常图是正确的格式 [H, W]
            anomaly_vis = anomaly_map
            if anomaly_vis.ndim == 3 and anomaly_vis.shape[0] == 1:
                anomaly_vis = anomaly_vis[0]  # [1, H, W] -> [H, W]
            
            vis_data = {
                'original_image': image,
                'anomaly_map': anomaly_vis,  # 使用正确的格式
                'patch_features': patch_features,
                'patch_token_memory': patch_token_memory,  # 添加patch_token_memory
                'text_features': cur_text_features.T,  # 转置以匹配期望格式
                'gt_mask': gt_mask,
                'scores': {
                    'textual_anomaly_max': float(torch.max(textual_anomaly_map)),
                    'visual_score_max': float(torch.max(vis_score)),
                    'final_score': float(text_probs),
                    'query_similarity': float(query_sim[obj_idx].mean(dim=0))
                }
            }
            
            # 创建可视化
            try:
                print(f"  📐 数据形状检查:")
                print(f"    - original_image: {vis_data['original_image'].shape}")
                print(f"    - anomaly_map: {vis_data['anomaly_map'].shape}")
                print(f"    - patch_features: {len(vis_data['patch_features'])} layers")
                print(f"    - patch_token_memory: {len(vis_data['patch_token_memory'])} layers")
                print(f"    - text_features: {vis_data['text_features'].shape}")
                if vis_data['gt_mask'] is not None:
                    print(f"    - gt_mask: {vis_data['gt_mask'].shape}")
                
                visualizer.create_debug_report(
                    original_image=vis_data['original_image'],
                    anomaly_map=vis_data['anomaly_map'],
                    patch_features=vis_data['patch_features'],
                    text_features=vis_data['text_features'],
                    patch_token_memory=vis_data['patch_token_memory'],  # 添加patch_token_memory参数
                    gt_mask=vis_data['gt_mask'],
                    scores=vis_data['scores'],
                    filename=filename
                )
                print(f"  ✅ 可视化创建成功: {filename}")
                visualization_count_per_class[current_class] = visualization_count_per_class.get(current_class, 0) + 1
            except Exception as e:
                print(f"  ⚠️ 可视化创建失败: {e}")
                import traceback
                traceback.print_exc()
                visualization_count_per_class[current_class] = visualization_count_per_class.get(current_class, 0) + 1  # 即使失败也计数，避免无限循环
        
        results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
        anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )
        results[cls_name[0]]['anomaly_maps'].append(anomaly_map)

    # 打印可视化总结
    if args.enable_visualization and visualizer is not None:
        print(f"\n🎨 可视化总结:")
        for class_name, count in visualization_count_per_class.items():
            print(f"  - {class_name}: {count} 个样本已可视化")

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
    
    # 计算最终结果
    final_pixel_auroc = np.mean(pixel_auroc_list) * 100 if pixel_auroc_list else 0
    final_image_auroc = np.mean(image_auroc_list) * 100 if image_auroc_list else 0
    final_pixel_aupro = np.mean(pixel_aupro_list) * 100 if pixel_aupro_list else 0
    final_image_ap = np.mean(image_ap_list) * 100 if image_ap_list else 0
    
    print(f"\n{'='*80}")
    print(f"Test Results:")
    print(f"Pixel AUROC: {final_pixel_auroc:.2f}%")
    print(f"Image AUROC: {final_image_auroc:.2f}%")
    print(f"Pixel AUPRO: {final_pixel_aupro:.2f}%")
    print(f"Image AP: {final_image_ap:.2f}%")
    print(f"{'='*80}\n")
    print(results)
    
    # 保存测试结果到日志
    logger.info(f"\n{'='*80}")
    logger.info("TEST RESULTS:")
    logger.info(f"Pixel AUROC: {final_pixel_auroc:.2f}%")
    logger.info(f"Image AUROC: {final_image_auroc:.2f}%")
    logger.info(f"Pixel AUPRO: {final_pixel_aupro:.2f}%")
    logger.info(f"Image AP: {final_image_ap:.2f}%")
    logger.info(f"{'='*80}")
    logger.info("\n%s", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VVCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="/data2/zlt/code/abnormal_dataset/mvtec", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/main_mvtec_visualization_shot1_best', help='path to save results')
    parser.add_argument("--dinov3_model_path", type=str, default='./model_card/dinov3-vitl16-pretrain-lvd1689m', help='path to DINOv3 model')
    parser.add_argument("--blip_model_path", type=str, default='./model_card', help='path to BLIP-Diffusion model')
    parser.add_argument("--mvtec_data_root", type=str, default='/data2/zlt/code/abnormal_dataset/mvtec/', help='root path for MVTec dataset')
    parser.add_argument("--visa_data_root", type=str, default='/data2/zlt/code/abnormal_dataset/visa_save/1cls/', help='root path for ViSA dataset')
    
    # trained model path
    parser.add_argument("--model_path", type=str, default='results/main_mvtec_shot1_retrain/checkpoints_20251027_171117/best_model.pth', help='path to trained model')
    
    # testing hyperparameters
    parser.add_argument("--temperature", type=float, default=0.07, help="temperature for softmax in text-image similarity")
    parser.add_argument("--sample_stride", type=int, default=3, help="stride for sampling training images (i*stride)")
    parser.add_argument("--noise_scale", type=float, default=2.0, help="scale factor for noise in negative samples")
    parser.add_argument("--threshold", type=float, default=1/256, help="threshold for patch-token alignment")
    
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
    
    parser.add_argument("--shot", type=int, default=1, choices=[1, 2, 4], help="number of shots (1, 2, or 4)")
    
    parser.add_argument("--clip", type=str, default='CLIP', help="CLIP model type")
    parser.add_argument("--clip_name", type=str, default='ViT-L-14-336',  help="CLIP model name")
    
    # 可视化相关参数
    parser.add_argument("--max_visualizations_per_epoch", type=int, default=10, help="每个类别最大可视化样本数量")
    parser.add_argument("--enable_visualization", action='store_true', default=True, help="是否启用可视化功能")
    parser.add_argument("--visualize", action='store_true', default=True, help="是否启用可视化功能（兼容参数）")
    parser.add_argument("--vis_samples", type=int, default=10, help="每个类别可视化样本数量（兼容参数）")

   
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    main(args)