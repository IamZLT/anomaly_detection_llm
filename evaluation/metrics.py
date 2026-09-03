"""Classification + bbox metrics for VisA→MVTec evaluation."""

from __future__ import annotations

from typing import Optional

from reasoning.rewards import box_iou
from utils.common import qwen_norm1000_to_original_pixels


def classification_correct(pred_cls, is_anomaly: bool) -> bool:
    return pred_cls is not None and bool(pred_cls) == bool(is_anomaly)


def box_ious_from_parsed(parsed: dict, meta: dict):
    orig = tuple(meta.get("orig_size") or (1, 1))
    gt = meta.get("gt_box_px")
    iou_f = 0.0
    iou_c = 0.0
    if gt is not None:
        if parsed.get("bbox_2d") is not None:
            iou_f = box_iou(qwen_norm1000_to_original_pixels(parsed["bbox_2d"], orig), gt)
        if parsed.get("candidate_bbox") is not None:
            iou_c = box_iou(qwen_norm1000_to_original_pixels(parsed["candidate_bbox"], orig), gt)
    rec_ok = classification_correct(parsed.get("is_anomaly", None), bool(meta.get("is_anomaly")))
    return float(iou_c), float(iou_f), rec_ok


def is_truncated(seq, prompt_len: int, max_new: int, eos_id: Optional[int]) -> bool:
    n_new = int(seq.numel()) - int(prompt_len)
    if n_new < int(max_new):
        return False
    if eos_id is None:
        return True
    return int(seq[-1].item()) != int(eos_id)
