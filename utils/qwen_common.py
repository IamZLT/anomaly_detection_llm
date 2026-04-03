import json
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


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


def parse_grounding_output(response: str) -> Optional[Dict]:
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
            if isinstance(arr, list) and len(arr) == 4:
                return {"bbox_2d": arr}
        except Exception:
            pass
    return None


def prepare_output_dir(base_dir: str, run_name: str, auto_create: bool) -> str:
    if not auto_create:
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{run_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

