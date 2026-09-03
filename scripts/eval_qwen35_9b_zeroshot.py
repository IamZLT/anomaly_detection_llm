#!/usr/bin/env python3
"""Zero-shot / 1-shot-ref defect recognition + localization with vanilla Qwen3.5-9B."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.qwen35 import qwen_vision_factor, setup_model_and_processor
from utils.common import parse_grounding_output, qwen_norm1000_to_original_pixels, smart_resize
from utils.config import load_yaml_config
from evaluation.infer import build_generation_inputs, decode_generation_output


ROOT = "/data2/zlt/anomaly_detection_llm/datasets/mvtec_anomaly_detection"

CASES = [
    # obvious structural
    ("bottle", "broken_large", "000.png", True),
    ("zipper", "broken_teeth", "006.png", True),
    # small / local
    ("capsule", "crack", "000.png", True),
    ("pill", "scratch", "014.png", True),
    ("transistor", "bent_lead", "000.png", True),
    # texture / geometry
    ("leather", "color", "000.png", True),
    ("hazelnut", "crack", "000.png", True),
    ("metal_nut", "bent", "000.png", True),
    # negatives
    ("bottle", "good", "000.png", False),
    ("leather", "good", "000.png", False),
]

REF_PROMPT = """The first image is a defect-free normal reference image, and the second image is the inspection sample image. Compare the two images: if any manufacturing defect is present in the inspection image, locate that defect, and **finally output valid JSON**:

```json
[
  {"bbox_2d": [x1, y1, x2, y2], "reason": "analysis"}
]
```
"""


def _font(size: int = 16):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def mask_bbox(mask_path: str) -> Optional[List[int]]:
    if not mask_path or not os.path.isfile(mask_path):
        return None
    arr = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(arr > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = area_a + area_b - inter
    return float(inter / den) if den > 0 else 0.0


def parse_all_items(response: str) -> List[Dict[str, Any]]:
    """Parse one or many bbox JSON objects from model text."""
    items: List[Dict[str, Any]] = []
    stripped = response.strip()
    if "</think>" in stripped:
        stripped = stripped.split("</think>", 1)[1].strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    try:
        j = json.loads(stripped)
        if isinstance(j, list):
            items.extend([x for x in j if isinstance(x, dict)])
        elif isinstance(j, dict):
            items.append(j)
    except Exception:
        pass
    if not items:
        one = parse_grounding_output(response)
        if one:
            items.append(one)
    return items


def draw_boxes(
    image: Image.Image,
    pred_boxes: List[Tuple[List[int], str]],
    gt_box: Optional[List[int]],
) -> Image.Image:
    im = image.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _font(18)
    if gt_box is not None:
        draw.rectangle(gt_box, outline=(0, 220, 0), width=4)
        draw.text((gt_box[0] + 4, max(0, gt_box[1] - 22)), "GT", fill=(0, 220, 0), font=font)
    for box, label in pred_boxes:
        draw.rectangle(box, outline=(255, 40, 40), width=4)
        draw.text((box[0] + 4, max(0, box[1] - 22)), f"PRED:{label[:40]}", fill=(255, 40, 40), font=font)
    return im


def case_paths(cls: str, defect: str, fname: str, is_anom: bool) -> Tuple[str, Optional[str]]:
    img = os.path.join(ROOT, cls, "test", defect, fname)
    mask = None
    if is_anom:
        stem = os.path.splitext(fname)[0]
        mask = os.path.join(ROOT, cls, "ground_truth", defect, f"{stem}_mask.png")
        if not os.path.isfile(mask):
            mask = None
    return img, mask


def pick_ref_image(cls: str, query_path: str) -> str:
    """Same-class normal sample; never reuse the query image."""
    query_abs = os.path.abspath(query_path)
    candidates = [
        os.path.join(ROOT, cls, "train", "good", "000.png"),
        os.path.join(ROOT, cls, "test", "good", "001.png"),
        os.path.join(ROOT, cls, "test", "good", "000.png"),
    ]
    good_dirs = [
        os.path.join(ROOT, cls, "train", "good"),
        os.path.join(ROOT, cls, "test", "good"),
    ]
    for d in good_dirs:
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.lower().endswith((".png", ".jpg", ".jpeg")):
                    candidates.append(os.path.join(d, name))
    for p in candidates:
        if os.path.isfile(p) and os.path.abspath(p) != query_abs:
            return p
    raise FileNotFoundError(f"no normal reference for class={cls}")


def build_ref_inputs(processor, ref_image: Image.Image, query_image: Image.Image, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ref_image},
                {"type": "image", "image": query_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[ref_image, query_image], return_tensors="pt", padding=True)


def concat_ref_query(ref: Image.Image, query_vis: Image.Image) -> Image.Image:
    h = max(ref.height, query_vis.height)
    rw = int(ref.width * h / ref.height)
    r = ref.resize((rw, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (rw + query_vis.width + 8, h), (30, 30, 30))
    canvas.paste(r, (0, 0))
    canvas.paste(query_vis, (rw + 8, 0))
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    draw.text((8, 8), "REF (normal)", fill=(180, 255, 180), font=font)
    draw.text((rw + 16, 8), "QUERY", fill=(255, 180, 180), font=font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-ref", action="store_true", help="给一张同类别正常参考图做对比")
    args = parser.parse_args()

    cfg_path = os.path.join(PROJECT_ROOT, "configs", "qwen35_9b_zeroshot.yaml")
    cfg = load_yaml_config(cfg_path)
    cfg.setdefault("lora", {})["enabled"] = False

    # Ice Lake CPU has no native BF16; float32 is safer/faster here.
    if not torch.cuda.is_available():
        cfg.setdefault("model", {})["torch_dtype"] = "float32"
        print("[eval] CUDA unavailable → CPU float32", flush=True)
    else:
        print(f"[eval] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    nthreads = int(os.environ.get("OMP_NUM_THREADS", "16"))
    torch.set_num_threads(nthreads)
    print(f"[eval] torch threads={nthreads}", flush=True)

    out_name = "qwen35_9b_with_ref" if args.with_ref else "qwen35_9b_zeroshot"
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "eval", out_name)
    os.makedirs(out_dir, exist_ok=True)

    model_path = cfg["inference"]["model_path"]
    prompt = REF_PROMPT if args.with_ref else cfg["inference"]["prompt"]
    print(f"[eval] load {model_path}", flush=True)
    print(f"[eval] with_ref={args.with_ref} out={out_dir}", flush=True)
    t0 = time.time()
    model, processor = setup_model_and_processor(
        cfg=cfg, for_inference=True, model_name_override=model_path
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"[eval] model ready on {device} in {time.time() - t0:.1f}s", flush=True)

    max_image_size = int(cfg.get("data", {}).get("max_image_size", 512))
    raw_factor = cfg.get("data", {}).get("factor")
    factor = int(raw_factor) if raw_factor not in (None, "", "null", "None") else qwen_vision_factor(processor)
    print(f"[eval] input resize: max_image_size={max_image_size} factor={factor} bbox=qwen_norm1000", flush=True)

    results: List[Dict[str, Any]] = []
    for cls, defect, fname, is_anom in CASES:
        img_path, mask_path = case_paths(cls, defect, fname, is_anom)
        tag = f"{cls}/{defect}/{fname}"
        if not os.path.isfile(img_path):
            print(f"[skip] missing {img_path}", flush=True)
            continue
        image_original = Image.open(img_path).convert("RGB")
        image, orig_size, scale_factor = smart_resize(
            image_original.copy(), max_size=max_image_size, factor=factor
        )
        gt = mask_bbox(mask_path) if mask_path else None

        ref_path = None
        ref_image = None
        if args.with_ref:
            ref_path = pick_ref_image(cls, img_path)
            ref_original = Image.open(ref_path).convert("RGB")
            # 参考图拉到与待检图相同的输入尺寸，便于视觉对齐；bbox 仍按官方 0–1000 映射到待检原图
            ref_image = ref_original.resize(image.size, Image.Resampling.LANCZOS)
            print(
                f"[eval] ref={ref_path} query_in={image.size} orig={orig_size}",
                flush=True,
            )
            inputs = build_ref_inputs(processor, ref_image, image, prompt)
        else:
            print(f"[eval] query_in={image.size} orig={orig_size}", flush=True)
            inputs = build_generation_inputs(cfg, processor, image, prompt)
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        t1 = time.time()
        gen_kw = dict(
            max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
            do_sample=bool(cfg["inference"]["do_sample"]),
        )
        if args.with_ref:
            gen_kw["max_new_tokens"] = max(gen_kw["max_new_tokens"], 1536)
        if gen_kw["do_sample"]:
            gen_kw["temperature"] = float(cfg["inference"]["temperature"])
            gen_kw["top_p"] = float(cfg["inference"]["top_p"])
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kw)
        elapsed = time.time() - t1
        text = decode_generation_output(processor, outputs, inputs, cfg)
        items = parse_all_items(text)

        pred_boxes: List[Tuple[List[int], str]] = []
        pred_is_anom = False
        for it in items:
            lab = str(it.get("reason") or it.get("label") or "anomaly")
            bb = it.get("bbox_2d") if it.get("bbox_2d") is not None else it.get("bbox")
            if lab.lower() in ("normal", "none", "no anomaly", "good") and not bb:
                continue
            if bb and len(bb) == 4:
                pred_is_anom = True
                pred_boxes.append(
                    (
                        qwen_norm1000_to_original_pixels(list(bb), orig_size),
                        lab,
                    )
                )
            elif lab.lower() not in ("normal", "none", "good"):
                pred_is_anom = True

        ious = [iou(pb[0], gt) for pb in pred_boxes] if (gt and pred_boxes) else []
        best_iou = max(ious) if ious else 0.0
        rec_ok = (pred_is_anom == is_anom)
        loc_ok = bool(is_anom and best_iou >= 0.3)

        rec = {
            "case": tag,
            "is_anomaly": is_anom,
            "pred_anomaly": pred_is_anom,
            "recognition_correct": rec_ok,
            "gt_bbox": gt,
            "ref_path": ref_path,
            "orig_size": list(orig_size),
            "input_size": list(image.size),
            "pred_boxes": [
                {"bbox": b, "reason": lab} for b, lab in pred_boxes
            ],
            "best_iou": round(best_iou, 4),
            "localization_iou03": loc_ok,
            "seconds": round(elapsed, 1),
            "response": text,
        }
        results.append(rec)

        vis = draw_boxes(image_original, pred_boxes, gt)
        vis_name = f"{cls}_{defect}_{os.path.splitext(fname)[0]}.png"
        if args.with_ref:
            vis = concat_ref_query(ref_original, vis)
        vis.save(os.path.join(out_dir, vis_name))
        print(
            f"[{tag}] rec={'OK' if rec_ok else 'FAIL'} "
            f"pred_anom={pred_is_anom} iou={best_iou:.3f} {elapsed:.1f}s",
            flush=True,
        )
        print(text[:600], flush=True)
        print("-" * 60, flush=True)

        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    n = len(results)
    rec_acc = sum(1 for r in results if r["recognition_correct"]) / max(n, 1)
    anom = [r for r in results if r["is_anomaly"]]
    loc_hit = sum(1 for r in anom if r["localization_iou03"]) / max(len(anom), 1)
    mean_iou = float(np.mean([r["best_iou"] for r in anom])) if anom else 0.0
    summary = {
        "n": n,
        "recognition_acc": round(rec_acc, 4),
        "anomaly_loc_iou03": round(loc_hit, 4),
        "anomaly_mean_best_iou": round(mean_iou, 4),
        "with_ref": bool(args.with_ref),
        "device": str(device),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print("[summary]", json.dumps(summary), flush=True)
    print(f"[eval] saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
