#!/usr/bin/env python3
"""
Flask Web应用 - Qwen3-VL微调模型推理界面
支持上传图像、输入提示词、显示结果，并在图像上标注bbox
"""

import os
import json
import re
import io
import base64
import torch
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, jsonify, send_file
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

# 获取app目录的绝对路径
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# 确保上传和输出目录存在（相对于app目录）
UPLOAD_DIR = os.path.join(APP_DIR, app.config['UPLOAD_FOLDER'])
OUTPUT_DIR = os.path.join(APP_DIR, app.config['OUTPUT_FOLDER'])
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 更新配置为绝对路径
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['OUTPUT_FOLDER'] = OUTPUT_DIR

# ============================================================================
# 配置参数
# ============================================================================
# 模型路径（相对于项目根目录）
MODEL_PATH = os.path.join(os.path.dirname(APP_DIR), "outputs/grounding_mvtec_20260116_204905/final_model")
MAX_IMAGE_SIZE = 512
FACTOR = 28
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# 全局变量存储模型和processor
model = None
processor = None
device = None

# ============================================================================
# 辅助函数
# ============================================================================

def smart_resize(image: Image.Image, max_size: int = 1024, factor: int = 28):
    """智能resize图像，确保尺寸是factor的倍数"""
    original_size = image.size
    w, h = original_size
    
    scale = min(max_size / max(w, h), 1.0)
    new_w = int(w * scale / factor) * factor
    new_h = int(h * scale / factor) * factor
    
    resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    scale_x = new_w / w
    scale_y = new_h / h
    
    return resized_image, original_size, (scale_x, scale_y)


def parse_grounding_output(response: str):
    """解析模型输出的bbox JSON"""
    # 尝试提取JSON格式的bbox
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


def draw_bbox_on_image(image: Image.Image, bbox: list, label: str = "Anomaly"):
    """在图像上绘制边界框"""
    draw = ImageDraw.Draw(image)
    
    # 确保bbox格式正确 [x1, y1, x2, y2]
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        
        # 绘制矩形框（红色，宽度3）
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # 绘制标签背景
        try:
            # 尝试使用默认字体
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except:
                font = ImageFont.load_default()
        
        # 计算文本尺寸
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # 绘制标签背景
        label_y = max(0, y1 - text_height - 4)
        draw.rectangle(
            [x1, label_y, x1 + text_width + 8, label_y + text_height + 4],
            fill="red",
            outline="red"
        )
        
        # 绘制标签文本
        draw.text(
            (x1 + 4, label_y + 2),
            label,
            fill="white",
            font=font
        )
    
    return image


def load_model():
    """加载模型和processor（懒加载）"""
    global model, processor, device
    
    if model is None or processor is None:
        print("正在加载模型和processor...")
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=True
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        model.eval()
        print(f"模型已加载到 {device}")
    
    return model, processor


# ============================================================================
# Flask路由
# ============================================================================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/inference', methods=['POST'])
def inference():
    """推理接口"""
    try:
        # 检查文件上传
        if 'image' not in request.files:
            return jsonify({'error': '未上传图像'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
        
        # 获取提示词
        prompt = request.form.get('prompt', 'Locate the anomaly region in this image and output the bbox coordinates in JSON format.')
        
        # 保存上传的图像
        image_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(image_path)
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # 加载模型
        model, processor = load_model()
        
        # 处理图像
        processed_image, _, scale_factor = smart_resize(
            image.copy(),
            max_size=MAX_IMAGE_SIZE,
            factor=FACTOR
        )
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 处理输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[processed_image],
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=DO_SAMPLE
            )
        
        # 解码
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        # 解析bbox
        bbox_data = parse_grounding_output(response)
        
        # 如果有bbox，在图像上标注
        annotated_image = image.copy()
        bbox_info = None
        
        if bbox_data:
            bbox = bbox_data.get('bbox_2d') or bbox_data.get('bbox')
            if bbox and len(bbox) == 4:
                # 将bbox从处理后的尺寸映射回原始尺寸
                inv_scale_x = 1.0 / scale_factor[0]
                inv_scale_y = 1.0 / scale_factor[1]
                original_bbox = [
                    int(bbox[0] * inv_scale_x),
                    int(bbox[1] * inv_scale_y),
                    int(bbox[2] * inv_scale_x),
                    int(bbox[3] * inv_scale_y)
                ]
                
                # 确保bbox在图像范围内
                original_bbox[0] = max(0, min(original_bbox[0], original_size[0]))
                original_bbox[1] = max(0, min(original_bbox[1], original_size[1]))
                original_bbox[2] = max(0, min(original_bbox[2], original_size[0]))
                original_bbox[3] = max(0, min(original_bbox[3], original_size[1]))
                
                # 绘制bbox
                label = bbox_data.get('label', 'Anomaly')
                annotated_image = draw_bbox_on_image(annotated_image, original_bbox, label)
                
                bbox_info = {
                    'bbox': original_bbox,
                    'label': label,
                    'processed_bbox': bbox
                }
        
        # 保存标注后的图像
        output_filename = f"annotated_{os.path.basename(file.filename)}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        annotated_image.save(output_path)
        
        # 返回结果（使用相对路径）
        return jsonify({
            'success': True,
            'response': response,
            'bbox_info': bbox_info,
            'annotated_image': f'/static/outputs/{output_filename}',
            'original_image': f'/uploads/{file.filename}'
        })
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': f'推理失败: {error_msg}'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """返回上传的文件"""
    upload_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(upload_path):
        return send_file(upload_path)
    else:
        return "File not found", 404


@app.route('/static/outputs/<filename>')
def output_file(filename):
    """返回标注后的图像"""
    output_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(output_path):
        return send_file(output_path)
    else:
        return "File not found", 404


# 如果直接运行app.py，也可以启动（但推荐使用根目录的run_app.py）
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Qwen3-VL 微调模型 Web 应用")
    print("=" * 60)
    print(f"应用目录: {APP_DIR}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"上传目录: {UPLOAD_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    print("\n💡 提示: 推荐在项目根目录使用 'python run_app.py' 启动")
    print("访问 http://localhost:5000 使用Web界面")
    print("按 Ctrl+C 停止服务器\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
