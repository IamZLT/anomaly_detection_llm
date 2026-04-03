import json

import torch
from PIL import Image
from transformers import AutoImageProcessor

from models.qwen3_modeling import setup_model_and_processor
from utils.qwen_common import parse_grounding_output, smart_resize


def inference_main(cfg: dict) -> None:
    model_path = cfg["inference"]["model_path"]
    image_path = cfg["inference"]["image_path"]
    prompt = cfg["inference"]["prompt"]
    use_dino_bridge = bool(cfg.get("dino", {}).get("enabled", True))
    use_clip_bridge = bool(cfg.get("clip", {}).get("enabled", True))
    local_files_only = cfg.get("model", {}).get("local_files_only", True)
    if not model_path:
        raise ValueError("推理模式需要配置 inference.model_path")
    if not image_path:
        raise ValueError("推理模式需要配置 inference.image_path")

    model, processor = setup_model_and_processor(
        cfg=cfg,
        for_inference=True,
        model_name_override=model_path,
    )

    image = Image.open(image_path).convert("RGB")
    image, original_size, scale_factor = smart_resize(
        image,
        max_size=cfg["data"]["max_image_size"],
        factor=cfg["data"]["factor"],
    )
    if use_dino_bridge:
        dino_processor = AutoImageProcessor.from_pretrained(
            cfg["dino"]["model_path"],
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        clip_processor = None
        if use_clip_bridge:
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
        if clip_processor is not None:
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

    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg["inference"]["max_new_tokens"],
            temperature=cfg["inference"]["temperature"],
            top_p=cfg["inference"]["top_p"],
            do_sample=cfg["inference"]["do_sample"],
        )
    response = processor.decode(outputs[0], skip_special_tokens=True)
    print("\n模型回复:\n")
    print(response)

    bbox_data = parse_grounding_output(response)
    if bbox_data and isinstance(bbox_data, dict) and bbox_data.get("bbox_2d"):
        bbox = bbox_data["bbox_2d"]
        inv_scale_x = 1.0 / scale_factor[0]
        inv_scale_y = 1.0 / scale_factor[1]
        original_bbox = [
            int(bbox[0] * inv_scale_x),
            int(bbox[1] * inv_scale_y),
            int(bbox[2] * inv_scale_x),
            int(bbox[3] * inv_scale_y),
        ]
        print("\n解析到 bbox:")
        print(json.dumps(bbox_data, ensure_ascii=False, indent=2))
        print(f"映射到原图尺寸 {original_size}: {original_bbox}")
    else:
        print("\n未解析到有效 bbox。")

