#!/usr/bin/env python3
"""
Flask Web 应用 - Qwen-VL 推理界面。
与 CLI 一致：models.avNet.setup_model_and_processor + utils.infer.build_generation_inputs。
"""

import argparse
import os
import sys
import json
import time

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_APP_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from models.avNet import setup_model_and_processor
from utils.infer import build_generation_inputs, decode_generation_output
from utils.common import (
    draw_bbox_on_image,
    infer_model_compute_device,
    parse_grounding_output,
    qwen_norm1000_to_original_pixels,
    smart_resize,
)
from utils.config import load_yaml_config, apply_runtime_overrides

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

APP_DIR = _APP_DIR

# 确保上传和输出目录存在（相对于app目录）
UPLOAD_DIR = os.path.join(APP_DIR, app.config['UPLOAD_FOLDER'])
OUTPUT_DIR = os.path.join(APP_DIR, app.config['OUTPUT_FOLDER'])
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 更新配置为绝对路径
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['OUTPUT_FOLDER'] = OUTPUT_DIR

# ============================================================================
# 全局状态（懒加载）
# ============================================================================
_web_cfg = None
model = None
processor = None
device = None


def get_web_cfg():
    """读取与训练相同的 YAML；可通过环境变量覆盖（由 run_app.py 设置）。"""
    global _web_cfg
    if _web_cfg is not None:
        return _web_cfg
    cfg_path = os.environ.get(
        "QWEN_WEB_CONFIG",
        os.path.join(_PROJECT_ROOT, "configs", "ad_llm_qwen35_9b_zeroshot.yaml"),
    )
    cfg = load_yaml_config(cfg_path)
    mp = os.environ.get("QWEN_WEB_MODEL_PATH")
    if mp:
        cfg = apply_runtime_overrides(
            cfg,
            argparse.Namespace(
                model_path=mp,
                mode=None,
                image_path=None,
                prompt=None,
                output_dir=None,
                run_name=None,
            ),
        )
    if not cfg.get("inference", {}).get("model_path"):
        raise ValueError(
            "未设置 inference.model_path。请在 configs/ad_llm_qwen35_9b_zeroshot.yaml 中配置，或使用 "
            "python run_app.py --model-path /path/to/final_model"
        )
    _web_cfg = cfg
    return _web_cfg


def _infer_model_device(mod: torch.nn.Module) -> str:
    return str(infer_model_compute_device(mod))


# ============================================================================
# 辅助函数
# ============================================================================

def load_model():
    """与 CLI inference_main 相同：setup_model_and_processor。"""
    global model, processor, device

    if model is None or processor is None:
        cfg = get_web_cfg()

        def _web_load_log(msg: str) -> None:
            print(f"[Web加载 {time.strftime('%H:%M:%S')}] {msg}", flush=True)

        _web_load_log("开始 setup_model_and_processor …")
        t0 = time.perf_counter()
        model, processor = setup_model_and_processor(
            cfg=cfg,
            for_inference=True,
            model_name_override=cfg["inference"]["model_path"],
        )
        _web_load_log(f"setup_model_and_processor 返回 (+{time.perf_counter() - t0:.1f}s)")
        device = _infer_model_device(model)
        _web_load_log(f"全部就绪，主模型设备: {device}")

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
        
        cfg = get_web_cfg()
        default_prompt = cfg["inference"].get(
            "prompt",
            "Locate the anomaly region in this image and output the bbox coordinates in JSON format.",
        )
        prompt = request.form.get("prompt", default_prompt)

        # 保存上传的图像
        image_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(image_path)

        # 加载图像
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        # 加载模型
        model, processor = load_model()

        processed_image, _, scale_factor = smart_resize(
            image.copy(),
            max_size=cfg["data"]["max_image_size"],
            factor=cfg["data"]["factor"],
        )

        inputs = build_generation_inputs(cfg, processor, processed_image, prompt)
        model_device = infer_model_compute_device(model)
        inputs = {
            k: v.to(model_device) if torch.is_tensor(v) else v for k, v in inputs.items()
        }

        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg["inference"]["max_new_tokens"],
                temperature=cfg["inference"]["temperature"],
                top_p=cfg["inference"]["top_p"],
                do_sample=cfg["inference"]["do_sample"],
            )
        
        # 解码新生成 token
        response = decode_generation_output(processor, outputs, inputs, cfg)
        
        # 解析bbox
        bbox_data = parse_grounding_output(response)
        
        # 如果有bbox，在图像上标注
        annotated_image = image.copy()
        bbox_info = None
        
        if bbox_data:
            bbox = bbox_data.get("bbox_2d") or bbox_data.get("bbox")
            if bbox and len(bbox) == 4:
                original_bbox = qwen_norm1000_to_original_pixels(
                    list(map(float, bbox)), original_size
                ) 
                # 绘制bbox
                label = bbox_data.get('label', 'Anomaly')
                annotated_image = draw_bbox_on_image(annotated_image, original_bbox, label)
                
                bbox_info = {
                    "bbox": original_bbox,
                    "label": label,
                    "processed_bbox": px,
                    "model_bbox_raw": bbox,
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
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Qwen3-VL 微调模型 Web 应用")
    print("=" * 60)
    print(f"应用目录: {APP_DIR}")
    try:
        _c = get_web_cfg()
        print(f"配置文件: {os.environ.get('QWEN_WEB_CONFIG', os.path.join(_PROJECT_ROOT, 'configs', 'ad_llm_qwen35_9b_zeroshot.yaml'))}")
        print(f"模型路径: {_c['inference']['model_path']}")
    except ValueError as e:
        print(f"配置错误: {e}")
    print(f"上传目录: {UPLOAD_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    print("\n💡 推荐在项目根目录使用: python run_app.py --model-path ...")
    print("访问 http://localhost:5000 使用Web界面")
    print("按 Ctrl+C 停止服务器\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
