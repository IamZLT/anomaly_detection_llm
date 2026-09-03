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


def is_main_process() -> bool:
    r = os.environ.get("RANK")
    if r is not None:
        return int(r) == 0
    lr = os.environ.get("LOCAL_RANK")
    if lr is not None:
        return int(lr) == 0
    return True


def train_log(msg: str, main_only: bool = False) -> None:
    if main_only and not is_main_process():
        return
    rank = os.environ.get("RANK", "?")
    lr = os.environ.get("LOCAL_RANK", "?")
    print(f"[train rank={rank} local={lr}] {msg}", flush=True)


_ROLLOUT_LOG_FILE = None


def open_rollout_log(path: str) -> None:
    """Open (append) the plain-text rollout log; only the main process should call this."""
    global _ROLLOUT_LOG_FILE
    if _ROLLOUT_LOG_FILE is not None:
        try:
            _ROLLOUT_LOG_FILE.close()
        except Exception:
            pass
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    _ROLLOUT_LOG_FILE = open(path, "a", encoding="utf-8")


def rollout_log(msg: str) -> None:
    """Append a line to the rollout log (no-op if never opened or on a non-main rank)."""
    global _ROLLOUT_LOG_FILE
    if _ROLLOUT_LOG_FILE is None:
        return
    try:
        _ROLLOUT_LOG_FILE.write(msg + "\n")
        _ROLLOUT_LOG_FILE.flush()
    except Exception:
        pass


def close_rollout_log() -> None:
    global _ROLLOUT_LOG_FILE
    if _ROLLOUT_LOG_FILE is not None:
        try:
            _ROLLOUT_LOG_FILE.close()
        except Exception:
            pass
        _ROLLOUT_LOG_FILE = None


def qwen_smart_hw(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 256 * 256,
    max_pixels: int = 448 * 448,
) -> Tuple[int, int]:
    """Official Qwen-VL smart_resize: both sides divisible by factor, pixels in [min, max].

    Returns (height, width). Qwen3.5 uses factor = patch_size * spatial_merge_size = 32.
    """
    import math

    factor = max(int(factor), 1)
    min_pixels = max(int(min_pixels), factor * factor)
    max_pixels = max(int(max_pixels), min_pixels)
    h = max(int(height), 1)
    w = max(int(width), 1)
    if max(h, w) / min(h, w) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(h, w) / min(h, w)}")
    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return int(h_bar), int(w_bar)


def smart_resize(
    image: Image.Image,
    max_size: int = 1024,
    factor: int = 32,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[Image.Image, Tuple[int, int], Tuple[float, float]]:
    original_size = image.size
    w, h = original_size
    factor = max(int(factor), 1)
    if max_pixels is None:
        max_pixels = int(max_size) * int(max_size)
    if min_pixels is None:
        min_pixels = 256 * 256
    new_h, new_w = qwen_smart_hw(h, w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    new_w = max(new_w, factor)
    new_h = max(new_h, factor)
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return resized, original_size, (new_w / w, new_h / h)


def scale_bbox(bbox: Optional[List[int]], scale_factor: Tuple[float, float]) -> Optional[List[int]]:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    sx, sy = scale_factor
    return [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]


_BBOX_TAG_RE = re.compile(r"<bbox>\s*([^<]+?)\s*</bbox>", re.IGNORECASE)


def normalize_bbox_pixels_to_01(
    bbox: List[float], width: int, height: int
) -> List[float]:
    """将像素框 [x1,y1,x2,y2] 按当前图像宽高归一化到 [0,1]（相对 width/height 各除一次）。"""
    if width <= 0 or height <= 0:
        raise ValueError(f"normalize_bbox_pixels_to_01: invalid size ({width}, {height})")
    x1, y1, x2, y2 = bbox
    def _c(v: float, denom: int) -> float:
        return min(1.0, max(0.0, float(v) / float(denom)))

    return [_c(x1, width), _c(y1, height), _c(x2, width), _c(y2, height)]


def rewrite_bbox_tags_to_normalized_01(text: str, norm_bbox: List[float], decimals: int = 6) -> str:
    """将文本中所有 <bbox>...</bbox> 替换为归一化坐标（与训练用 resize 后图像一致）。"""
    if len(norm_bbox) != 4:
        return text
    fmt = ",".join(f"{float(v):.{decimals}f}" for v in norm_bbox)
    return _BBOX_TAG_RE.sub(f"<bbox>{fmt}</bbox>", text)


def rewrite_bbox_tags_original_pixels_to_normalized_01(
    text: str,
    scale_factor: Tuple[float, float],
    resized_width: int,
    resized_height: int,
    decimals: int = 6,
) -> str:
    """
    无 metadata.bbox 时：逐个解析 <bbox>...</bbox>，转为相对 smart_resize 后图像的 0–1。

    - 若四个数均在 [0,1] 且最大值 ≤1：视为已相对「当前训练图（resize 后）」的归一化坐标，仅裁剪到 [0,1] 并统一格式。
    - 否则视为「原图像素」：先按 scale_factor 映射到 resize 后像素，再除以 resized_wh。
    """
    if resized_width <= 0 or resized_height <= 0:
        return text
    sx, sy = scale_factor

    def repl(m: re.Match) -> str:
        raw = m.group(1).strip()
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 4:
            return m.group(0)
        try:
            x1, y1, x2, y2 = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            return m.group(0)
        vals = [x1, y1, x2, y2]
        if max(vals) <= 1.0 + 1e-6 and min(vals) >= -1e-6:
            norm = [min(1.0, max(0.0, v)) for v in vals]
        else:
            scaled = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            norm = normalize_bbox_pixels_to_01(scaled, resized_width, resized_height)
        fmt = ",".join(f"{float(v):.{decimals}f}" for v in norm)
        return f"<bbox>{fmt}</bbox>"

    return _BBOX_TAG_RE.sub(repl, text)


def bbox_to_processed_pixels(
    bbox: List[float],
    processed_size: Tuple[int, int],
    *,
    normalized_01: bool,
) -> List[float]:
    """
    将模型解析出的框转为「与 smart_resize 后图像一致」的像素坐标。
    normalized_01=True：bbox 为相对处理后图像的 0–1；否则视为已在处理后图像上的像素坐标。
    """
    pw, ph = int(processed_size[0]), int(processed_size[1])
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if normalized_01:
        return [x1 * pw, y1 * ph, x2 * pw, y2 * ph]
    return [x1, y1, x2, y2]


def qwen_norm1000_to_original_pixels(
    bbox: List[float], original_size: Tuple[int, int]
) -> List[int]:
    """
    Qwen3-VL / Qwen3.5 官方 2D grounding：
    bbox_2d 为相对原图的 0–1000（千分比），与内部 resize 无关。
    像素 = coord / 1000 * (W 或 H)。
    若四个数均 ≤1.5，视为误用 0–1，按原图比例还原。
    """
    ow, oh = int(original_size[0]), int(original_size[1])
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    mx = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if mx <= 1.5:
        xs = [x1 * ow, y1 * oh, x2 * ow, y2 * oh]
    else:
        xs = [x1 / 1000.0 * ow, y1 / 1000.0 * oh, x2 / 1000.0 * ow, y2 / 1000.0 * oh]
    x1i = int(max(0, min(ow - 1, round(xs[0]))))
    y1i = int(max(0, min(oh - 1, round(xs[1]))))
    x2i = int(max(0, min(ow, round(xs[2]))))
    y2i = int(max(0, min(oh, round(xs[3]))))
    if x2i < x1i:
        x1i, x2i = x2i, x1i
    if y2i < y1i:
        y1i, y2i = y2i, y1i
    return [x1i, y1i, x2i, y2i]


def _strip_markdown_json_fence(text: str) -> str:
    s = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def parse_grounding_output(response: str) -> Optional[Dict]:
    """解析模型输出中的 bbox：JSON（bbox_2d / bbox）、裸四维数组、以及训练常用的 <bbox>x1,y1,x2,y2</bbox> 标签。"""
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

    # 与 generate_data2json / 训练监督一致：<bbox>x1,y1,x2,y2</bbox>（可为小数 0–1）
    tag_m = _BBOX_TAG_RE.search(response)
    if tag_m:
        inner = tag_m.group(1).strip().replace(" ", "")
        parts = inner.split(",")
        if len(parts) == 4:
            try:
                arr = [float(parts[i]) for i in range(4)]
                return {"bbox_2d": arr}
            except ValueError:
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

