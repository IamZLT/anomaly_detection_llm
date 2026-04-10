import json
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


def infer_model_compute_device(mod: nn.Module) -> torch.device:
    """与词嵌入同一设备，供输入张量 .to(device)；避免 device_map 下 next(parameters) 先遍历到 CPU 分片。"""
    try:
        if hasattr(mod, "get_input_embeddings"):
            emb = mod.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                return emb.weight.device
    except Exception:
        pass
    return next(mod.parameters()).device


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def smart_resize(
    image: Image.Image, max_size: int = 1024, factor: int = 28
) -> Tuple[Image.Image, Tuple[int, int], Tuple[float, float]]:
    original_size = image.size
    w, h = original_size
    scale = min(max_size / max(w, h), 1.0)

    new_w = int(w * scale / factor) * factor
    new_h = int(h * scale / factor) * factor
    new_w = max(new_w, factor)
    new_h = max(new_h, factor)

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized, original_size, (new_w / w, new_h / h)


def scale_bbox(bbox: Optional[List[int]], scale_factor: Tuple[float, float]) -> Optional[List[int]]:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    sx, sy = scale_factor
    return [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]


def _strip_markdown_json_fence(text: str) -> str:
    s = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def parse_grounding_output(response: str) -> Optional[Dict]:
    """解析模型输出中的 bbox；支持 ```json 代码块、单对象、数组包对象、[x1,y1,x2,y2]。"""
    stripped = _strip_markdown_json_fence(response)
    try:
        j = json.loads(stripped)
        if isinstance(j, list):
            for item in j:
                if isinstance(item, dict) and (item.get("bbox_2d") is not None or item.get("bbox") is not None):
                    return item
        if isinstance(j, dict) and (j.get("bbox_2d") is not None or j.get("bbox") is not None):
            return j
    except Exception:
        pass

    json_match = re.search(r'\{[^{}]*"bbox_2d"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except Exception:
            pass

    array_match = re.search(r"\[[^\]]+\]", response, re.DOTALL)
    if array_match:
        try:
            arr = json.loads(array_match.group())
            if isinstance(arr, list) and len(arr) == 4 and all(isinstance(x, (int, float)) for x in arr):
                return {"bbox_2d": arr}
        except Exception:
            pass
    return None


def draw_bbox_on_image(image: Image.Image, bbox: List[int], label: str = "Anomaly") -> Image.Image:
    """在图像上绘制边界框（与 app/app.py 一致）。"""
    draw = ImageDraw.Draw(image)
    if len(bbox) != 4:
        return image
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except OSError:
            font = ImageFont.load_default()
    bbox_text = draw.textbbox((0, 0), label, font=font)
    tw = bbox_text[2] - bbox_text[0]
    th = bbox_text[3] - bbox_text[1]
    label_y = max(0, y1 - th - 4)
    draw.rectangle([x1, label_y, x1 + tw + 8, label_y + th + 4], fill="red", outline="red")
    draw.text((x1 + 4, label_y + 2), label, fill="white", font=font)
    return image


def prepare_output_dir(base_dir: str, run_name: str, auto_create: bool) -> str:
    if not auto_create:
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{run_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

