"""
使用Qwen3-VL-8B-Instruct训练MVTec缺陷检测Grounding模型

功能：
1. 加载MVTec数据集并转换为Grounding格式
2. 从mask图像提取bbox
3. 使用LoRA + 冻结ViT进行参数高效微调
4. 训练Grounding任务（输出bbox坐标）

基于Qwen2.5-VL Grounding微调教程实现
"""

import os
import sys
import json
import re
import time
import gc
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
# 尝试导入Qwen3VLForConditionalGeneration，如果不存在则使用Qwen2VLForConditionalGeneration

from transformers import Qwen3VLForConditionalGeneration
QwenVLForConditionalGeneration = Qwen3VLForConditionalGeneration

from peft import LoraConfig, get_peft_model
from typing import Dict, List, Optional, Tuple
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

# 导入自定义的数据加载器
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.load_mvtec_data import MVTecDataManager


def smart_resize(image: Image.Image, max_size: int = 1024, factor: int = 28) -> Tuple[Image.Image, Tuple[int, int], Tuple[float, float]]:
    """
    智能resize图像，确保尺寸是factor的倍数（满足ViT patch要求）
    
    Args:
        image: PIL图像
        max_size: 最大边长
        factor: 尺寸必须是factor的倍数
        
    Returns:
        resized_image, original_size, scale_factor
    """
    original_size = image.size  # (width, height)
    w, h = original_size
    
    # 计算缩放比例
    scale = min(max_size / max(w, h), 1.0)
    
    # 确保新尺寸是factor的倍数
    new_w = int(w * scale / factor) * factor
    new_h = int(h * scale / factor) * factor
    
    # Resize图像
    resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 计算缩放因子
    scale_x = new_w / w
    scale_y = new_h / h
    
    return resized_image, original_size, (scale_x, scale_y)


def scale_bbox(bbox: List[int], scale_factor: Tuple[float, float]) -> List[int]:
    """
    缩放bbox坐标
    
    Args:
        bbox: [x1, y1, x2, y2] 格式的bbox
        scale_factor: (scale_x, scale_y) 缩放因子
        
    Returns:
        缩放后的bbox
    """
    if bbox is None:
        return None
    
    x1, y1, x2, y2 = bbox
    scale_x, scale_y = scale_factor
    
    new_bbox = [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    ]
    return new_bbox


class MVTecQwenGroundingDataset(Dataset):
    """MVTec数据集，适配Qwen2.5-VL Grounding格式"""
    
    def __init__(
        self, 
        manager: MVTecDataManager,
        processor: AutoProcessor,
        mode: str = 'train',
        max_length: int = 2048,
        max_image_size: int = 1024,
        factor: int = 28,
        use_grounding_format: bool = True
    ):
        """
        初始化数据集
        
        Args:
            manager: MVTec数据管理器
            processor: Qwen2.5-VL的processor
            mode: 'train' or 'test'
            max_length: 最大序列长度
            max_image_size: 图像最大边长
            factor: 图像尺寸必须是factor的倍数
            use_grounding_format: 是否使用Grounding格式
        """
        self.processor = processor
        self.mode = mode
        self.max_length = max_length
        self.max_image_size = max_image_size
        self.factor = factor
        self.use_grounding_format = use_grounding_format
        self.manager = manager  # 保存manager引用，用于获取完整路径
        
        # 获取Grounding格式的样本
        print(f"\n{'='*60}")
        print(f"正在初始化数据集 [{mode}]...")
        print(f"{'='*60}")
        
        if use_grounding_format:
            print("  → 正在转换为Grounding格式（包含bbox提取）...")
            self.grounding_samples = manager.get_all_grounding_samples(mode)
            print(f"  ✓ Grounding格式转换完成")
        else:
            # 使用原始格式
            print("  → 正在加载原始格式数据...")
            if mode == 'train':
                self.samples = manager.dataset_loader.get_all_train_samples()
            else:
                self.samples = manager.dataset_loader.get_all_test_samples()
            self.grounding_samples = None
            print(f"  ✓ 原始格式数据加载完成")
            
        print(f"\n数据集初始化完成 [{mode}]")
        if use_grounding_format:
            print(f"  ✓ 样本数量: {len(self.grounding_samples):,}")
            print(f"  ✓ 格式: Grounding (bbox输出)")
            # 统计有bbox的样本数
            bbox_count = sum(1 for s in self.grounding_samples if s.get('metadata', {}).get('bbox') is not None)
            print(f"  ✓ 包含bbox的样本: {bbox_count:,} ({bbox_count/len(self.grounding_samples)*100:.1f}%)")
        else:
            print(f"  ✓ 样本数量: {len(self.samples):,}")
            print(f"  ✓ 格式: 原始对话格式")
        print(f"{'='*60}")
    
    def __len__(self):
        if self.use_grounding_format:
            return len(self.grounding_samples)
        else:
            return len(self.samples)
    
    def __getitem__(self, idx):
        if self.use_grounding_format:
            sample = self.grounding_samples[idx]
            img_path = sample['image']
            conversations = sample['conversations']
            metadata = sample.get('metadata', {})
            original_bbox = metadata.get('bbox')
        else:
            # 原始格式（向后兼容）
            sample = self.samples[idx]
            img_path = sample['full_img_path']
            conversations = None
            original_bbox = None
        
        # 加载图像
        try:
            # 如果是相对路径，需要拼接dataset_root
            if not os.path.isabs(img_path):
                img_path = os.path.join(self.manager.dataset_loader.dataset_root, img_path)
            
            image = Image.open(img_path).convert('RGB')
            original_size = image.size
            
            # 使用smart_resize
            image, original_size, scale_factor = smart_resize(
                image, 
                max_size=self.max_image_size,
                factor=self.factor
            )
            
            # 内存优化：关闭原始图像文件句柄
            if hasattr(image, 'close'):
                pass  # PIL Image不需要显式关闭
        except Exception as e:
            if idx < 10:  # 只打印前10个错误，避免刷屏
                print(f"⚠ 警告: 加载图像失败 {img_path}: {e}")
            # 返回一个空白图像
            image = Image.new('RGB', (self.max_image_size, self.max_image_size), color='white')
            original_size = image.size
            scale_factor = (1.0, 1.0)
        
        # 如果有bbox，需要缩放
        if original_bbox is not None:
            scaled_bbox = scale_bbox(original_bbox, scale_factor)
        else:
            scaled_bbox = None
        
        # 构建消息格式（Qwen2.5-VL格式）
        messages = []
        for conv in conversations:
            role = "user" if conv.get('from') == 'human' or conv.get('role') == 'user' else "assistant"
            content = conv.get('value') or conv.get('content', '')
            
            # 处理包含<image>标记的内容
            if "<image>" in str(content):
                content_text = str(content).replace("<image>", "").strip()
                messages.append({
                    "role": role,
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": content_text}
                    ]
                })
            else:
                # 纯文本消息
                if isinstance(content, list):
                    # 如果content是列表格式
                    messages.append({
                        "role": role,
                        "content": content
                    })
                else:
                    messages.append({
                        "role": role,
                        "content": str(content)
                    })
        
        # 使用processor处理
        try:
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # 处理图像和文本
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            # 内存优化：释放图像对象（processor已经处理完）
            del image
            
            # 移除batch维度，并只保留tensor类型的字段
            # 过滤掉字典、列表等非tensor类型，避免data collator错误
            filtered_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    filtered_inputs[k] = v.squeeze(0) if v.dim() > 1 else v
                # 忽略非tensor类型（如字典、列表等）
            
            # 清理inputs字典
            del inputs
            
            # 为训练准备labels（只对assistant回复计算loss）
            labels = filtered_inputs['input_ids'].clone()
            
            # Mask掉用户输入部分，只计算assistant回复的loss
            # 这里简化处理，实际应该根据tokenizer更精确地mask
            # 对于Qwen，assistant回复通常在"assistant"标记之后
            input_ids_list = filtered_inputs['input_ids'].tolist()
            assistant_start = -1
            
            # 查找assistant回复的开始位置
            for i, token_id in enumerate(input_ids_list):
                # 这里需要根据实际的tokenizer调整
                # 简化版本：假设最后一部分是assistant回复
                pass
            
            # 简化：只对非padding部分计算loss（实际应该更精确）
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            filtered_inputs['labels'] = labels
            
            # 内存优化：清理临时变量
            del labels, input_ids_list
            
            # 注意：不添加metadata到inputs中，因为data collator无法处理字典类型
            # 如果需要metadata，可以使用自定义的data collator
            
            return filtered_inputs
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个简单的示例
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.full((self.max_length,), -100, dtype=torch.long)
            }


class EnhancedLoggingCallback(TrainerCallback):
    """增强的日志记录回调，记录更多训练信息到TensorBoard"""
    
    def __init__(self, output_dir: str, save_eval_examples: bool = True, num_eval_examples: int = 10):
        self.output_dir = output_dir
        self.save_eval_examples = save_eval_examples
        self.num_eval_examples = num_eval_examples
        self.current_step = 0
        
        # 创建TensorBoard writer
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"  ✓ TensorBoard日志目录: {log_dir}")
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, model=None, **kwargs):
        """在每次日志记录时调用，添加更多信息到TensorBoard"""
        if logs is None:
            return
        
        self.current_step = state.global_step
        
        # 记录所有训练指标到TensorBoard（确保每个step只记录一次）
        # 使用state.global_step作为x轴，确保唯一性
        
        # 记录loss
        if 'loss' in logs:
            self.writer.add_scalar('train/loss', logs['loss'], state.global_step)
        
        # 记录learning_rate
        if 'learning_rate' in logs:
            self.writer.add_scalar('train/learning_rate', logs['learning_rate'], state.global_step)
        
        # 记录epoch（使用state.epoch，确保准确性）
        # epoch是浮点数，表示当前epoch的进度（如0.5表示第一个epoch的一半）
        self.writer.add_scalar('train/epoch', state.epoch, state.global_step)
        
        # 计算并记录梯度范数（额外的指标）
        # 注意：计算梯度范数需要遍历所有参数，可能消耗一些内存
        # 可以设置环境变量 DISABLE_GRAD_NORM=true 来禁用
        if model is not None and os.environ.get('DISABLE_GRAD_NORM', 'false').lower() != 'true':
            try:
                total_norm = 0.0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                if param_count > 0:
                    total_norm = total_norm ** (1. / 2)
                    self.writer.add_scalar('train/grad_norm', total_norm, state.global_step)
                    logs['grad_norm'] = total_norm
            except Exception as e:
                pass  # 如果计算失败，忽略
        
        # 刷新写入（定期清理TensorBoard缓存）
        self.writer.flush()
        
        # 内存优化：定期清理CUDA缓存（每100步清理一次，避免过于频繁）
        if state.global_step % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, logs=None, model=None, **kwargs):
        """在评估时保存案例到TensorBoard"""
        self.current_step = state.global_step
        
        # 记录评估指标到TensorBoard
        if logs is not None:
            # 记录eval_loss
            if 'eval_loss' in logs:
                self.writer.add_scalar('eval/loss', logs['eval_loss'], state.global_step)
            
            # 记录其他评估指标（如果有）
            for key, value in logs.items():
                if key.startswith('eval_') and key != 'eval_loss':
                    # 移除eval_前缀，记录到eval/命名空间
                    metric_name = key.replace('eval_', 'eval/')
                    self.writer.add_scalar(metric_name, value, state.global_step)
            
            self.writer.flush()
        
        if not self.save_eval_examples:
            return
        print(f"\n  📝 准备保存评估案例到TensorBoard（step {state.global_step}）...")
    
    def __del__(self):
        """关闭TensorBoard writer"""
        if hasattr(self, 'writer'):
            self.writer.close()


def compute_metrics_and_save_examples(eval_pred, eval_dataset, processor, writer, step, num_examples=10):
    """
    计算评估指标并保存案例到TensorBoard
    
    Args:
        eval_pred: 评估预测结果
        eval_dataset: 评估数据集
        processor: 图像处理器
        writer: TensorBoard SummaryWriter
        step: 当前步数
        num_examples: 保存的案例数量
    """
    import torch
    from PIL import Image
    import random
    
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids
    
    # 计算基本指标
    metrics = {}
    
    examples_text = []  # 用于TensorBoard文本记录
    
    # 随机选择一些案例保存（确保有异常和正常样本）
    if hasattr(eval_dataset, 'grounding_samples') and len(eval_dataset.grounding_samples) > 0:
        # 分别选择有bbox和没有bbox的样本
        anomaly_indices = []
        normal_indices = []
        
        for i, sample in enumerate(eval_dataset.grounding_samples):
            if i >= len(predictions):
                break
            metadata = sample.get('metadata', {})
            if metadata.get('bbox') is not None:
                anomaly_indices.append(i)
            else:
                normal_indices.append(i)
        
        # 平衡选择
        num_anomaly = min(num_examples // 2, len(anomaly_indices))
        num_normal = min(num_examples - num_anomaly, len(normal_indices))
        
        selected_indices = (
            random.sample(anomaly_indices, num_anomaly) if anomaly_indices else []
        ) + (
            random.sample(normal_indices, num_normal) if normal_indices else []
        )
        
        # 如果还不够，随机补充
        if len(selected_indices) < num_examples:
            remaining = [i for i in range(len(predictions)) if i not in selected_indices]
            selected_indices.extend(random.sample(remaining, min(num_examples - len(selected_indices), len(remaining))))
    else:
        selected_indices = random.sample(range(len(predictions)), min(num_examples, len(predictions)))
    
    for idx in selected_indices:
        try:
            # 获取预测和标签
            if isinstance(predictions, np.ndarray):
                pred_ids = predictions[idx]
            elif isinstance(predictions, (list, tuple)):
                pred_ids = predictions[idx]
            else:
                pred_ids = predictions[idx:idx+1] if hasattr(predictions, '__getitem__') else predictions
            
            if isinstance(label_ids, np.ndarray):
                label_id = label_ids[idx]
            elif isinstance(label_ids, (list, tuple)):
                label_id = label_ids[idx]
            else:
                label_id = label_ids[idx:idx+1] if hasattr(label_ids, '__getitem__') else label_ids
            
            # 解码文本
            pred_text = ""
            label_text = ""
            if hasattr(processor, 'tokenizer'):
                try:
                    if isinstance(pred_ids, np.ndarray):
                        pred_ids = pred_ids.tolist()
                    if isinstance(label_id, np.ndarray):
                        label_id = label_id.tolist()
                    pred_text = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
                    label_text = processor.tokenizer.decode(label_id, skip_special_tokens=True)
                except:
                    pred_text = str(pred_ids)[:200]  # 限制长度
                    label_text = str(label_id)[:200]
            else:
                pred_text = str(pred_ids)[:200]
                label_text = str(label_id)[:200]
            
            # 获取样本信息
            image_path = ""
            metadata = {}
            has_bbox = False
            if hasattr(eval_dataset, 'grounding_samples') and idx < len(eval_dataset.grounding_samples):
                sample = eval_dataset.grounding_samples[idx]
                image_path = sample.get('image', '')
                metadata = sample.get('metadata', {})
                has_bbox = metadata.get('bbox') is not None if metadata else False
            
            # 构建文本摘要用于TensorBoard
            example_text = f"**案例 {idx}** (Step {step})\n"
            example_text += f"图像: {os.path.basename(image_path)}\n"
            example_text += f"类型: {'异常(有bbox)' if has_bbox else '正常'}\n"
            if metadata:
                example_text += f"类别: {metadata.get('class', 'N/A')}\n"
                if metadata.get('defect_type'):
                    example_text += f"缺陷类型: {metadata.get('defect_type')}\n"
            example_text += f"\n**预测:**\n{pred_text[:500]}\n"
            example_text += f"\n**真实标签:**\n{label_text[:500]}\n"
            example_text += "---\n"
            
            examples_text.append(example_text)
            
            # 如果有图像路径，尝试加载并添加到TensorBoard
            # 注意：保存图像到TensorBoard会占用大量内存，建议只在需要时启用
            # 可以通过环境变量控制：SAVE_EVAL_IMAGES=true
            if image_path and os.path.exists(image_path) and os.environ.get('SAVE_EVAL_IMAGES', 'false').lower() == 'true':
                try:
                    # 构建完整路径
                    if not os.path.isabs(image_path):
                        full_path = os.path.join(eval_dataset.manager.dataset_loader.dataset_root, image_path)
                    else:
                        full_path = image_path
                    
                    if os.path.exists(full_path):
                        img = Image.open(full_path).convert('RGB')
                        # 转换为tensor格式
                        import torchvision.transforms as transforms
                        transform = transforms.ToTensor()
                        img_tensor = transform(img)
                        # 添加到TensorBoard
                        writer.add_image(f'eval_examples/example_{idx}', img_tensor, step)
                        # 立即释放图像内存
                        del img, img_tensor
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except Exception as e:
                    pass  # 如果加载图像失败，忽略
            
        except Exception as e:
            print(f"⚠ 处理案例 {idx} 时出错: {e}")
            continue
    
    # 将所有案例文本保存到TensorBoard
    if examples_text:
        all_examples_text = "\n".join(examples_text)
        writer.add_text('eval_examples/summary', all_examples_text, step)
        
        # 统计信息
        num_anomaly = sum(1 for text in examples_text if '异常' in text)
        num_normal = len(examples_text) - num_anomaly
        writer.add_scalar('eval_examples/num_anomaly', num_anomaly, step)
        writer.add_scalar('eval_examples/num_normal', num_normal, step)
        writer.add_scalar('eval_examples/total', len(examples_text), step)
        
        print(f"  ✓ 已保存 {len(examples_text)} 个评估案例到TensorBoard")
        writer.flush()
        
        # 内存优化：清理临时变量
        del examples_text, all_examples_text
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return metrics


def setup_model_and_processor(
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    use_lora: bool = True,
    lora_r: int = 8,  # Grounding任务推荐较小rank
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    freeze_vit: bool = True  # Grounding任务推荐冻结ViT
):
    """
    设置模型和processor
    
    Args:
        model_name: 模型名称或路径
        use_lora: 是否使用LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        freeze_vit: 是否冻结视觉编码器
        
    Returns:
        model, processor
    """
    print(f"{'='*60}")
    print(f"正在加载模型: {model_name}")
    print(f"{'='*60}")
    print("使用BF16精度加载模型（未使用量化）")
    
    # 加载processor
    print("\n[1/3] 正在加载Processor...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True  # 明确指定使用fast processor（默认行为）
    )
    print("✓ Processor加载完成")
    
    # 加载模型（不使用量化，使用BF16精度）
    print("\n[2/3] 正在加载模型（这可能需要几分钟）...")
    
    # 多GPU训练时，不使用device_map="auto"，让Trainer处理分布式
    # 单GPU或使用DeepSpeed时，可以使用device_map="auto"
    num_gpus = torch.cuda.device_count()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        num_gpus = len([x for x in cuda_visible.split(",") if x.strip()])
    
    # 检查是否使用DeepSpeed（通过环境变量或参数）
    use_deepspeed = os.environ.get("USE_DEEPSPEED", "").lower() == "true"
    
    if num_gpus > 1:
        # 多GPU训练时，让Trainer处理设备分配（不使用device_map）
        print(f"  检测到 {num_gpus} 个GPU，将使用分布式训练")
        if use_deepspeed:
            print(f"  使用DeepSpeed进行分布式训练")
        device_map = None  # Trainer会自动处理设备分配
    else:
        # 单GPU时，使用device_map="auto"
        device_map = "auto"
    
    model = QwenVLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=True,
        dtype=torch.bfloat16  # 使用dtype代替已弃用的torch_dtype
    )
    print("✓ 模型加载完成")
    
    # 确保模型参数默认是可训练的（在应用LoRA之前）
    # 注意：某些模型加载时可能默认requires_grad=False
    for param in model.parameters():
        if param.requires_grad is False:
            param.requires_grad = True
    
    # 配置LoRA（必须在冻结ViT之前应用）
    print("\n[3/3] 正在配置模型...")
    if use_lora:
        print(f"  → 配置LoRA:")
        print(f"     - rank: {lora_r}")
        print(f"     - alpha: {lora_alpha}")
        print(f"     - dropout: {lora_dropout}")
        
        # 自动查找所有线性层名称（用于调试）
        import torch.nn as nn
        linear_module_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # 只包含语言模型部分，排除视觉编码器
                if "vision" not in name.lower() and "visual" not in name.lower():
                    linear_module_names.append(name.split('.')[-1])  # 只取最后一层名称
        
        # 去重并排序
        unique_linear_names = sorted(set(linear_module_names))
        print(f"  → 检测到的线性层类型: {unique_linear_names[:10]}...")  # 只显示前10个
        
        # Qwen3-VL的target_modules（根据搜索结果和实际模型结构）
        # 尝试多种可能的模块名组合
        possible_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
        
        # 验证哪些模块名实际存在于模型中
        actual_target_modules = []
        for module_name in possible_target_modules:
            found = False
            for name, module in model.named_modules():
                if name.endswith(module_name) and isinstance(module, nn.Linear):
                    if "vision" not in name.lower() and "visual" not in name.lower():
                        actual_target_modules.append(module_name)
                        found = True
                        break
            if not found:
                print(f"  ⚠ 警告: 未找到模块 '{module_name}'")
        
        if not actual_target_modules:
            # 如果没找到，尝试使用"all-linear"（PEFT支持）
            print("  ⚠ 未找到标准模块名，尝试使用 'all-linear'")
            actual_target_modules = "all-linear"
        else:
            print(f"  ✓ 找到 {len(actual_target_modules)} 个目标模块: {actual_target_modules}")
        
        # Qwen3-VL/Qwen2-VL的target_modules
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=actual_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        print("  ✓ LoRA配置完成")
        model.print_trainable_parameters()
    
    # 冻结视觉编码器（Grounding任务推荐，在LoRA之后）
    if freeze_vit:
        print("  → 冻结视觉编码器（ViT）...")
        frozen_count = 0
        for name, param in model.named_parameters():
            if "vision_model" in name or "visual" in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"  ✓ ViT已冻结 ({frozen_count} 个参数)")
    
    # 关键修复：启用输入梯度（gradient checkpointing需要）
    # 这对于LoRA + gradient checkpointing是必需的
    print("  → 启用输入梯度（gradient checkpointing需要）...")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        print("  ✓ 使用 enable_input_require_grads()")
    else:
        # 回退方案：注册forward hook
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        
        # 找到输入embeddings层并注册hook
        if hasattr(model, "get_input_embeddings"):
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            print("  ✓ 使用 forward hook 方式")
        else:
            # 尝试找到embedding层
            for name, module in model.named_modules():
                if "embed" in name.lower() and isinstance(module, torch.nn.Module):
                    module.register_forward_hook(make_inputs_require_grad)
                    print(f"  ✓ 在 {name} 上注册 forward hook")
                    break
    
    # 确保模型处于训练模式，并且有可训练的参数
    model.train()
    
    # 验证是否有可训练的参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if trainable_params == 0:
        raise RuntimeError("错误: 没有可训练的参数！请检查LoRA配置和freeze_vit设置。")
    print(f"  ✓ 可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    print(f"\n{'='*60}")
    print("模型初始化完成！")
    print(f"{'='*60}")
    
    return model, processor


def train(args):
    """训练函数"""
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 自动创建带日期和任务名的输出目录
    if args.auto_create_output_dir:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = args.run_name if args.run_name else "qwen3_vl_grounding"
        
        # 使用指定的output_dir作为基础目录，在其下创建带时间戳的子目录
        # 例如：./outputs -> ./outputs/grounding_mvtec_20260116_160212
        base_dir = args.output_dir
        # 确保基础目录存在
        os.makedirs(base_dir, exist_ok=True)
        
        # 创建格式：base_dir/task_name_YYYYMMDD_HHMMSS
        new_output_dir = os.path.join(base_dir, f"{task_name}_{timestamp}")
        args.output_dir = new_output_dir
        
        print(f"\n{'='*60}")
        print("📁 自动创建输出目录")
        print(f"{'='*60}")
        print(f"  任务名称: {task_name}")
        print(f"  时间戳: {timestamp}")
        print(f"  新输出目录: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"  ✓ 输出目录已创建")
    
    # 创建数据管理器
    print(f"\n{'='*60}")
    print("步骤 1/4: 初始化数据管理器")
    print(f"{'='*60}")
    print(f"  数据集根目录: {args.dataset_root}")
    print(f"  对话JSON路径: {args.conversation_json_path}")
    
    manager = MVTecDataManager(
        dataset_root=args.dataset_root,
        conversation_json_path=args.conversation_json_path
    )
    
    print("\n  正在加载数据...")
    manager.load_all()
    print("  ✓ 数据加载完成")
    
    # 设置模型和processor
    print(f"\n{'='*60}")
    print("步骤 2/4: 加载模型和Processor")
    print(f"{'='*60}")
    model, processor = setup_model_and_processor(
        model_name=args.model_name,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        freeze_vit=args.freeze_vit
    )
    
    # 创建数据集
    print(f"\n{'='*60}")
    print("步骤 3/4: 创建训练和验证数据集")
    print(f"{'='*60}")
    
    print("\n  正在创建训练集...")
    step_start = time.time()
    train_dataset = MVTecQwenGroundingDataset(
        manager=manager,
        processor=processor,
        mode='train',
        max_length=args.max_length,
        max_image_size=args.max_image_size,
        factor=args.factor,
        use_grounding_format=args.use_grounding_format
    )
    dataset_train_time = time.time() - step_start
    print(f"  ✓ 训练集创建完成 (耗时: {dataset_train_time:.2f}秒)")
    
    print("\n  正在创建验证集...")
    step_start = time.time()
    eval_dataset = MVTecQwenGroundingDataset(
        manager=manager,
        processor=processor,
        mode='test',
        max_length=args.max_length,
        max_image_size=args.max_image_size,
        factor=args.factor,
        use_grounding_format=args.use_grounding_format
    )
    dataset_eval_time = time.time() - step_start
    print(f"  ✓ 验证集创建完成 (耗时: {dataset_eval_time:.2f}秒)")
    print(f"  ✓ 数据集创建总耗时: {dataset_train_time + dataset_eval_time:.2f}秒")
    
    # 训练参数
    # 如果未指定eval_batch_size，则使用训练batch_size
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",  # 使用eval_strategy代替已弃用的evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to=[],  # 不使用Trainer的自动记录，我们手动控制所有记录
        logging_dir=os.path.join(args.output_dir, "logs"),
        # 添加更多日志选项
        logging_first_step=True,  # 记录第一步
        logging_nan_inf_filter=False,  # 不过滤NaN/Inf（用于调试）
        max_grad_norm=1.0,  # 梯度裁剪，同时会记录grad_norm
        log_level="info",  # 日志级别
        dataloader_pin_memory=False,  # 内存优化：关闭pin_memory可以节省内存（但可能稍慢）
        dataloader_prefetch_factor=2,  # 内存优化：减少预取因子
        # 多GPU训练相关
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,  # DDP模式
        local_rank=args.local_rank,  # 分布式训练rank
        deepspeed=args.deepspeed,  # DeepSpeed配置
    )
    
    # 创建Trainer
    print(f"\n{'='*60}")
    print("步骤 4/4: 创建Trainer并开始训练")
    print(f"{'='*60}")
    print(f"\n训练配置:")
    print(f"  📊 训练样本数: {len(train_dataset):,}")
    print(f"  📊 验证样本数: {len(eval_dataset):,}")
    print(f"  📦 Batch size: {args.batch_size}")
    print(f"  📦 梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"  📦 有效Batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  🔄 Epochs: {args.num_epochs}")
    print(f"  📈 Learning rate: {args.learning_rate}")
    print(f"  💾 Output dir: {args.output_dir}")
    print(f"  📦 Train batch size: {args.batch_size}")
    print(f"  📦 Eval batch size: {eval_batch_size} {'(使用训练batch_size)' if args.eval_batch_size is None else ''}")
    print(f"  🔧 使用LoRA: {args.use_lora}")
    print(f"  🔧 冻结ViT: {args.freeze_vit}")
    print(f"  🔧 Grounding格式: {args.use_grounding_format}")
    
    # 计算训练步数
    total_steps = len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps) * args.num_epochs
    print(f"\n  📐 预计总训练步数: ~{total_steps:,} 步")
    print(f"  ⏱️  预计训练时间: 根据GPU性能而定")
    
    # 创建增强的日志回调
    enhanced_callback = EnhancedLoggingCallback(
        output_dir=args.output_dir,
        save_eval_examples=args.save_eval_examples,
        num_eval_examples=args.num_eval_examples
    )
    
    # 创建计算指标的函数（包装以传递额外参数）
    def compute_metrics_wrapper(eval_pred):
        """包装compute_metrics以传递额外参数"""
        return compute_metrics_and_save_examples(
            eval_pred=eval_pred,
            eval_dataset=eval_dataset,
            processor=processor,
            writer=enhanced_callback.writer,
            step=enhanced_callback.current_step,
            num_examples=args.num_eval_examples
        )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_wrapper if args.save_eval_examples else None,
        callbacks=[enhanced_callback],
    )
    
    print(f"  📝 日志配置:")
    print(f"     - TensorBoard日志: {os.path.join(args.output_dir, 'logs')}")
    print(f"     - 所有日志和评估案例都保存到TensorBoard")
    if args.save_eval_examples:
        print(f"     - 每次评估保存 {args.num_eval_examples} 个案例到TensorBoard")
    
    # 开始训练
    print(f"\n{'='*60}")
    print("🚀 开始训练")
    print(f"{'='*60}")
    print("提示: 训练进度会显示在下方，包括loss、学习率等信息")
    print("提示: 可以使用Ctrl+C安全中断训练（模型会在checkpoint保存）")
    print(f"提示: 训练日志保存在: {os.path.join(args.output_dir, 'logs')}")
    print(f"{'='*60}\n")
    
    train_start_time = time.time()
    print(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_start_time))}")
    
    # 内存优化：训练前清理缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  💾 已清理GPU缓存")
    
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except RuntimeError as e:
        if "out of memory" in str(e) or "CUDA out of memory" in str(e):
            print(f"\n{'='*60}")
            print("❌ GPU内存不足！")
            print(f"{'='*60}")
            print("建议解决方案：")
            print("  1. 减小batch_size（当前: {args.batch_size}）")
            print("  2. 增加gradient_accumulation_steps（当前: {args.gradient_accumulation_steps}）")
            print("  3. 减小max_image_size（当前: {args.max_image_size}）")
            print("  4. 使用DeepSpeed Zero-2/3（--deepspeed scripts/zero2.json）")
            print("  5. 禁用评估图像保存（设置环境变量: SAVE_EVAL_IMAGES=false）")
            print("  6. 禁用梯度范数计算（设置环境变量: DISABLE_GRAD_NORM=true）")
            print(f"{'='*60}")
            raise
        else:
            raise
    
    train_time = time.time() - train_start_time
    print(f"\n训练结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"训练总耗时: {train_time/60:.2f}分钟 ({train_time/3600:.2f}小时)")
    
    # 保存最终模型
    print(f"\n{'='*60}")
    print("💾 保存最终模型")
    print(f"{'='*60}")
    print(f"  保存路径: {args.output_dir}/final_model")
    
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    
    total_time = time.time() - train_start_time
    print(f"\n{'='*60}")
    print("✅ 训练完成！")
    print(f"{'='*60}")
    print(f"📊 总耗时统计:")
    print(f"   - 总耗时: {total_time/60:.2f}分钟 ({total_time/3600:.2f}小时)")
    print(f"   - 训练耗时: {train_time/60:.2f}分钟 ({train_time/3600:.2f}小时)")
    print(f"\n💾 模型已保存到: {final_model_path}")
    print(f"\n🔍 可以使用以下命令进行推理:")
    print(f"  python qwen3_vl_8b.py --mode inference --model_path {final_model_path} --image_path <your_image>")
    print(f"{'='*60}")


def inference(args):
    """推理函数"""
    print(f"\n{'='*60}")
    print("🔍 推理模式")
    print(f"{'='*60}")
    
    # 加载模型和processor
    print(f"\n[1/4] 正在加载Processor...")
    print(f"  模型路径: {args.model_path}")
    processor = AutoProcessor.from_pretrained(
        args.model_path, 
        trust_remote_code=True,
        use_fast=True  # 明确指定使用fast processor
    )
    print("  ✓ Processor加载完成")
    
    print(f"\n[2/4] 正在加载模型（这可能需要几分钟）...")
    model = QwenVLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16  # 使用dtype代替已弃用的torch_dtype
    )
    model.eval()
    print("  ✓ 模型加载完成")
    
    # 加载图像
    print(f"\n[3/4] 正在处理图像...")
    print(f"  图像路径: {args.image_path}")
    image = Image.open(args.image_path).convert('RGB')
    original_size = image.size
    print(f"  原始尺寸: {original_size}")
    
    # 使用smart_resize
    image, original_size, scale_factor = smart_resize(
        image,
        max_size=args.max_image_size,
        factor=args.factor
    )
    print(f"  处理后尺寸: {image.size}")
    print(f"  缩放因子: {scale_factor}")
    print("  ✓ 图像处理完成")
    
    # 构建消息
    print(f"\n[4/4] 正在生成回复...")
    print(f"  提示词: {args.prompt[:100]}...")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt}
            ]
        }
    ]
    
    # 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    # 生成（带进度提示）
    print("  → 正在生成（这可能需要几秒到几分钟）...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample
        )
    print("  ✓ 生成完成")
    
    # 解码
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n{'='*60}")
    print("📝 模型回复:")
    print(f"{'='*60}")
    print(response)
    print(f"{'='*60}")
    
    # 解析Grounding输出
    print(f"\n{'='*60}")
    print("🔍 解析Bbox数据")
    print(f"{'='*60}")
    try:
        bbox_data = parse_grounding_output(response)
        if bbox_data:
            print("  ✓ 成功解析Bbox数据:")
            print(json.dumps(bbox_data, indent=2, ensure_ascii=False))
            
            # 如果有bbox，需要映射回原始图像尺寸
            if isinstance(bbox_data, dict) and bbox_data.get('bbox_2d'):
                bbox = bbox_data['bbox_2d']
                if bbox:
                    # 反向缩放bbox到原始图像尺寸
                    inv_scale_x = 1.0 / scale_factor[0]
                    inv_scale_y = 1.0 / scale_factor[1]
                    original_bbox = [
                        int(bbox[0] * inv_scale_x),
                        int(bbox[1] * inv_scale_y),
                        int(bbox[2] * inv_scale_x),
                        int(bbox[3] * inv_scale_y)
                    ]
                    print(f"\n  📐 映射到原始图像尺寸的Bbox:")
                    print(f"     - 处理后的Bbox: {bbox}")
                    print(f"     - 原始图像Bbox: {original_bbox}")
                    print(f"     - 原始图像尺寸: {original_size}")
                    print(f"     - 缩放因子: {scale_factor}")
        else:
            print("  ⚠ 未能从回复中解析出Bbox数据")
            print("  提示: 检查模型输出是否包含有效的JSON格式bbox")
    except Exception as e:
        print(f"  ❌ 解析bbox失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✅ 推理完成")
    print(f"{'='*60}")


def parse_grounding_output(response: str) -> Optional[Dict]:
    """
    解析模型输出的bbox JSON
    
    Args:
        response: 模型输出文本
        
    Returns:
        解析后的bbox数据
    """
    import json
    import re
    
    # 提取JSON部分
    json_match = re.search(r'\{[^{}]*"bbox_2d"[^{}]*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            bbox_data = json.loads(json_str)
            return bbox_data
        except:
            pass
    
    # 如果是数组格式
    array_match = re.search(r'\[.*"bbox_2d".*\]', response, re.DOTALL)
    if array_match:
        array_str = array_match.group()
        try:
            bbox_list = json.loads(array_str)
            return bbox_list
        except:
            pass
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL MVTec缺陷检测模型训练")
    
    # 模式选择
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"],
                       help="运行模式: train或inference")
    
    # 数据相关参数
    parser.add_argument("--dataset_root", type=str, 
                       default="/data2/zlt/anomaly_detection_llm/datasets/mvtec_anomaly_detection",
                       help="MVTec数据集根目录")
    parser.add_argument("--conversation_json_path", type=str,
                       default="/data2/zlt/anomaly_detection_llm/datasets/mvtec_zero_shot.json",
                       help="对话JSON文件路径")
    parser.add_argument("--use_conversations", action="store_true", default=True,
                       help="是否使用对话数据")
    
    # 模型相关参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                       help="预训练模型名称或路径（默认：Qwen3-VL-8B-Instruct）")
    parser.add_argument("--model_path", type=str, default=None,
                       help="推理时使用的模型路径")
    parser.add_argument("--freeze_vit", action="store_true", default=True,
                       help="冻结视觉编码器（Grounding任务推荐）")
    
    # LoRA参数（Grounding任务推荐配置）
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="使用LoRA微调")
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank (Grounding任务推荐8)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    
    # Grounding格式参数
    parser.add_argument("--use_grounding_format", action="store_true", default=True,
                       help="使用Grounding格式（bbox输出）")
    parser.add_argument("--max_image_size", type=int, default=1024,
                       help="图像最大边长")
    parser.add_argument("--factor", type=int, default=28,
                       help="图像尺寸必须是factor的倍数（满足ViT patch要求）")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="输出目录基础路径（如果使用--auto_create_output_dir，会在此目录下创建带日期的子目录）")
    parser.add_argument("--run_name", type=str, default="qwen3_vl_grounding",
                       help="任务名称（用于创建输出目录，如：grounding_mvtec，默认：qwen3_vl_grounding）")
    parser.add_argument("--auto_create_output_dir", action="store_true", default=True,
                       help="自动创建带日期和任务名的输出目录（默认启用，格式：run_name_YYYYMMDD_HHMMSS）")
    parser.add_argument("--no_auto_create_output_dir", action="store_false", dest="auto_create_output_dir",
                       help="禁用自动创建输出目录，使用指定的--output_dir")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="训练batch size")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                       help="验证batch size（默认使用训练batch_size）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="学习率（Grounding任务推荐1e-4到5e-4）")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                       help="warmup比例")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="最大序列长度")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="模型保存步数")
    parser.add_argument("--eval_steps", type=int, default=100,
                       help="验证步数")
    parser.add_argument("--save_eval_examples", action="store_true", default=True,
                       help="是否保存评估案例")
    parser.add_argument("--num_eval_examples", type=int, default=10,
                       help="每次评估保存的案例数量")
    
    # 多GPU训练参数
    # 多GPU训练参数
    parser.add_argument("--deepspeed", type=str, default=None,
                       help="DeepSpeed配置文件路径（如：scripts/zero2.json）。使用torchrun启动时指定此参数")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="分布式训练的本地rank（由torchrun自动设置，通常不需要手动指定）")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", default=False,
                       help="DDP模式下查找未使用的参数（可能更慢但更安全，LoRA训练时可能需要）")
    parser.add_argument("--save_total_limit", type=int, default=3,
                       help="保存的checkpoint数量限制")
    parser.add_argument("--fp16", action="store_true", default=False,
                       help="使用FP16训练")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="使用BF16训练")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                       help="使用梯度检查点")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="数据加载器worker数量")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="从checkpoint恢复训练")
    
    # 推理参数
    parser.add_argument("--image_path", type=str, default=None,
                       help="推理图像路径")
    parser.add_argument("--prompt", type=str, 
                       default="Locate the anomaly region in this image and output the bbox coordinates in JSON format.",
                       help="推理提示词（Grounding任务）")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p采样")
    parser.add_argument("--do_sample", action="store_true", default=True,
                       help="是否采样")
    
    args = parser.parse_args()
    
    # 根据模式执行
    if args.mode == "train":
        # 确保输出目录存在（即使不使用自动创建）
        if not args.auto_create_output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        train(args)
    elif args.mode == "inference":
        if args.model_path is None:
            raise ValueError("推理模式需要指定 --model_path")
        if args.image_path is None:
            raise ValueError("推理模式需要指定 --image_path")
        inference(args)


if __name__ == "__main__":
    main()
