#!/usr/bin/env python3
"""
独立的Qwen3-VL微调模型测试脚本
随机从MVTec数据集中选择一张图像进行测试
"""

import os
import random
import json
import re
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# ============================================================================
# 配置参数
# ============================================================================
MODEL_PATH = "./outputs/grounding_mvtec_20260116_204905/final_model"
DATASET_ROOT = "/data2/zlt/anomaly_detection_llm/datasets/mvtec_anomaly_detection"
MAX_IMAGE_SIZE = 512
FACTOR = 28
PROMPT = "Locate the anomaly region in this image and output the bbox coordinates in JSON format."
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# ============================================================================
# 辅助函数
# ============================================================================

def smart_resize(image: Image.Image, max_size: int = 1024, factor: int = 28):
    """
    智能resize图像，确保尺寸是factor的倍数（满足ViT patch要求）
    
    Args:
        image: PIL图像
        max_size: 最大边长
        factor: 尺寸必须是factor的倍数
        
    Returns:
        resized_image, original_size, scale_factor
    """
    original_size = image.size
    width, height = original_size
    
    # 计算缩放比例
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width = width
        new_height = height
    
    # 确保尺寸是factor的倍数
    new_width = (new_width // factor) * factor
    new_height = (new_height // factor) * factor
    
    # 至少保持factor大小
    new_width = max(new_width, factor)
    new_height = max(new_height, factor)
    
    # Resize
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 计算缩放因子
    scale_factor = (new_width / width, new_height / height)
    
    return resized_image, original_size, scale_factor


def parse_grounding_output(response: str):
    """
    解析模型输出的bbox JSON
    
    Args:
        response: 模型输出文本
        
    Returns:
        bbox数据字典或None
    """
    # 尝试提取JSON格式的bbox
    # 匹配 {...} 格式的JSON
    json_pattern = r'\{[^{}]*"bbox[^}]*\}'
    matches = re.findall(json_pattern, response, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        try:
            data = json.loads(match)
            if 'bbox' in data or 'bbox_2d' in data:
                return data
        except:
            continue
    
    # 尝试提取数组格式的bbox [x1, y1, x2, y2]
    array_pattern = r'\[[\d\s,\.]+\]'
    array_match = re.search(array_pattern, response)
    if array_match:
        array_str = array_match.group()
        try:
            bbox_list = json.loads(array_str)
            if isinstance(bbox_list, list) and len(bbox_list) == 4:
                return {"bbox_2d": bbox_list}
        except:
            pass
    
    return None


def find_random_test_image():
    """随机找一张测试图像"""
    print("🔍 正在随机选择测试图像...")
    
    test_images = []
    for category in os.listdir(DATASET_ROOT):
        category_path = os.path.join(DATASET_ROOT, category)
        if not os.path.isdir(category_path):
            continue
        
        test_dir = os.path.join(category_path, "test")
        if os.path.exists(test_dir):
            for root, dirs, files in os.walk(test_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        test_images.append(os.path.join(root, f))
    
    if not test_images:
        print("❌ 未找到测试图像！")
        return None
    
    selected_image = random.choice(test_images)
    print(f"  ✓ 已选择图像: {selected_image}")
    print(f"  📁 类别: {os.path.basename(os.path.dirname(os.path.dirname(selected_image)))}")
    print(f"  📁 子类型: {os.path.basename(os.path.dirname(selected_image))}")
    
    return selected_image


# ============================================================================
# 主推理函数
# ============================================================================

def run_inference(image_path: str):
    """运行推理"""
    print(f"\n{'='*60}")
    print("🚀 开始推理")
    print(f"{'='*60}")
    
    # 1. 加载Processor
    print(f"\n[1/4] 正在加载Processor...")
    print(f"  模型路径: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True
    )
    print("  ✓ Processor加载完成")
    
    # 2. 加载模型
    print(f"\n[2/4] 正在加载模型（这可能需要几分钟）...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    model.eval()
    print("  ✓ 模型加载完成")
    
    # 3. 处理图像
    print(f"\n[3/4] 正在处理图像...")
    print(f"  图像路径: {image_path}")
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    print(f"  原始尺寸: {original_size}")
    
    image, original_size, scale_factor = smart_resize(
        image,
        max_size=MAX_IMAGE_SIZE,
        factor=FACTOR
    )
    print(f"  处理后尺寸: {image.size}")
    print(f"  缩放因子: {scale_factor}")
    print("  ✓ 图像处理完成")
    
    # 4. 构建消息并生成
    print(f"\n[4/4] 正在生成回复...")
    print(f"  提示词: {PROMPT[:100]}...")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT}
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
    
    # 生成
    print("  → 正在生成（这可能需要几秒到几分钟）...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=DO_SAMPLE
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


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print(f"{'='*60}")
    print("🧪 Qwen3-VL 微调模型测试脚本（独立版本）")
    print(f"{'='*60}")
    
    # 检查模型路径
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型路径不存在: {MODEL_PATH}")
        print(f"\n请修改脚本中的 MODEL_PATH 变量为你的模型路径")
        print(f"例如: ./outputs/grounding_mvtec_YYYYMMDD_HHMMSS/final_model")
        return
    
    print(f"✓ 模型路径: {MODEL_PATH}")
    
    # 随机选择测试图像
    image_path = find_random_test_image()
    if not image_path:
        return
    
    # 运行推理
    try:
        run_inference(image_path)
    except KeyboardInterrupt:
        print(f"\n⚠️  推理被用户中断")
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n💡 提示:")
    print(f"  - 可以修改脚本中的 MODEL_PATH 来测试不同的模型")
    print(f"  - 可以修改 DATASET_ROOT 来指定不同的数据集路径")
    print(f"  - 可以修改 PROMPT 参数来使用不同的提示词")
    print(f"  - 可以修改 MAX_IMAGE_SIZE, TEMPERATURE 等参数")


if __name__ == "__main__":
    main()
