import json
import os
import time
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor

from models.avNet import setup_model_and_processor
from utils.qwen_common import (
    bbox_to_processed_pixels,
    draw_bbox_on_image,
    infer_model_compute_device,
    parse_grounding_output,
    smart_resize,
)

_INFER_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_INFER_PKG_DIR, ".."))


def build_generation_inputs(
    cfg: dict,
    processor,
    image: Image.Image,
    prompt: str,
    *,
    dino_processor: Optional[Any] = None,
    clip_processor: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    与训练/CLI 推理一致：DINO 桥开启时走 dino_pixel_values + clip_pixel_values，否则走 Qwen 原生图文输入。
    image 需已按 data.max_image_size / factor 做过 smart_resize。
    若传入已加载的 dino_processor / clip_processor（如 Web 懒加载缓存），可避免每次请求重复 from_pretrained。
    """
    use_dino_bridge = bool(cfg.get("dino", {}).get("enabled", True))
    local_files_only = cfg.get("model", {}).get("local_files_only", True)
    if use_dino_bridge:
        if dino_processor is None:
            dino_processor = AutoImageProcessor.from_pretrained(
                cfg["dino"]["model_path"],
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
        if clip_processor is None:
            clip_processor = AutoImageProcessor.from_pretrained(
                cfg["clip"]["model_path"],
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        dino_inputs = dino_processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={
                "height": int(cfg.get("dino", {}).get("image_size", 512)),
                "width": int(cfg.get("dino", {}).get("image_size", 512)),
            },
        )
        inputs["dino_pixel_values"] = dino_inputs["pixel_values"]
        clip_inputs = clip_processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={
                "height": int(cfg.get("clip", {}).get("image_size", 224)),
                "width": int(cfg.get("clip", {}).get("image_size", 224)),
            },
        )
        inputs["clip_pixel_values"] = clip_inputs["pixel_values"]
    else:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    return inputs


def decode_generation_output(
    processor,
    outputs: torch.Tensor,
    inputs: Dict[str, Any],
    cfg: dict,
) -> str:
    """
    只解码「模型新生成」的文本，避免把 prompt 混进 answer。

    - **DINO 桥（`dino_pixel_values` 存在）**：`QwenDinoBridgeModel.generate` 内部只用 `inputs_embeds`，
      HF 侧 `input_ids` 从空序列增长，**返回的 `sequences` 一般就是纯新生成 token**（不含文本 prompt 的
      重复）。此时**禁止**再按 `inputs["input_ids"]` 长度去切，否则会误切掉回答开头（表现为从
      `and` / `,` / `which` 等半句起）。
    - **非桥、标准 `input_ids` 生成**：`sequences` 为 ``[prompt || new]``，应按 ``input_ids.shape[1]`` 切片。
    """
    tok = getattr(processor, "tokenizer", processor)
    gen = outputs[0]
    if gen.dim() > 1:
        gen = gen[0]

    use_bridge = bool(cfg.get("dino", {}).get("enabled", True)) and inputs.get("dino_pixel_values") is not None
    if use_bridge:
        return tok.decode(gen, skip_special_tokens=True)

    if inputs.get("input_ids") is not None:
        plen = int(inputs["input_ids"].shape[1])
        if gen.numel() > plen:
            gen = gen[plen:]
    return tok.decode(gen, skip_special_tokens=True)


def _decode_generation_only(
    processor,
    outputs: torch.Tensor,
    inputs: Dict[str, Any],
    cfg: dict,
) -> str:
    return decode_generation_output(processor, outputs, inputs, cfg)


def inference_main(cfg: dict) -> None:
    model_path = cfg["inference"]["model_path"]
    image_path = cfg["inference"]["image_path"]
    prompt = cfg["inference"]["prompt"]
    if not model_path:
        raise ValueError("推理模式需要配置 inference.model_path")
    if not image_path:
        raise ValueError("推理模式需要配置 inference.image_path")

    model_path = os.path.abspath(os.path.expanduser(str(model_path)))
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")
    cfg["inference"]["model_path"] = model_path
    bridge_bin = os.path.join(model_path, "dino_bridge.bin")
    if os.path.isfile(bridge_bin):
        cfg.setdefault("model", {})["bridge_ckpt_path"] = bridge_bin
    else:
        print(
            "[inference] 警告: 未找到 dino_bridge.bin，桥接层为随机初始化；"
            "Qwen 主干仍从该目录 from_pretrained。"
        )

    print(
        f"[inference] Processor + Qwen 主干从同一目录加载（微调权重应在此）: {model_path}"
    )
    model, processor = setup_model_and_processor(
        cfg=cfg,
        for_inference=True,
        model_name_override=model_path,
    )
    # 默认把模型放到 CUDA（若可用）；否则 1.7B 在 CPU 上 generate 会像“卡住”
    if torch.cuda.is_available():
        try:
            model = model.to("cuda")
        except Exception as e:
            print(f"[inference][warn] move model to cuda failed: {e}")

    image_original = Image.open(image_path).convert("RGB")
    image, original_size, scale_factor = smart_resize(
        image_original.copy(),
        max_size=cfg["data"]["max_image_size"],
        factor=cfg["data"]["factor"],
    )
    inputs = build_generation_inputs(cfg, processor, image, prompt)

    model_device = infer_model_compute_device(model)
    inputs = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg["inference"]["max_new_tokens"],
            temperature=cfg["inference"]["temperature"],
            top_p=cfg["inference"]["top_p"],
            do_sample=cfg["inference"]["do_sample"],
        )
    response = _decode_generation_only(processor, outputs, inputs, cfg)
    print("\n模型回复（仅新生成）:\n")
    print(response)

    bbox_data = parse_grounding_output(response)
    bbox = None
    if isinstance(bbox_data, dict):
        bbox = bbox_data.get("bbox_2d") or bbox_data.get("bbox")
    if bbox_data and isinstance(bbox_data, dict) and bbox and len(bbox) == 4:
        norm01 = bool(cfg.get("data", {}).get("bbox_normalize_01", False))
        px = bbox_to_processed_pixels(
            list(map(float, bbox)),
            image.size,
            normalized_01=norm01,
        )
        inv_scale_x = 1.0 / scale_factor[0]
        inv_scale_y = 1.0 / scale_factor[1]
        original_bbox = [
            int(px[0] * inv_scale_x),
            int(px[1] * inv_scale_y),
            int(px[2] * inv_scale_x),
            int(px[3] * inv_scale_y),
        ]
        original_bbox[0] = max(0, min(original_bbox[0], original_size[0]))
        original_bbox[1] = max(0, min(original_bbox[1], original_size[1]))
        original_bbox[2] = max(0, min(original_bbox[2], original_size[0]))
        original_bbox[3] = max(0, min(original_bbox[3], original_size[1]))
        print("\n解析到 bbox:")
        print(json.dumps(bbox_data, ensure_ascii=False, indent=2))
        print(f"映射到原图尺寸 {original_size}: {original_bbox}")

        inf_vis = cfg.get("inference") or {}
        if bool(inf_vis.get("save_visualization", True)):
            out_dir = inf_vis.get("visual_output_dir")
            if not out_dir:
                out_dir = os.path.join(_PROJECT_ROOT, "outputs", "qwen_infer_vis")
            else:
                out_dir = os.path.abspath(os.path.expanduser(str(out_dir)))
            os.makedirs(out_dir, exist_ok=True)
            label = str(bbox_data.get("label", "Anomaly"))
            annotated = draw_bbox_on_image(image_original.copy(), original_bbox, label)
            stem, ext = os.path.splitext(os.path.basename(image_path))
            if ext.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                ext = ".png"
            out_name = f"annotated_{stem}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
            out_path = os.path.join(out_dir, out_name)
            annotated.save(out_path)
            print(f"\n可视化已保存: {out_path}")
    else:
        print("\n未解析到有效 bbox。")

