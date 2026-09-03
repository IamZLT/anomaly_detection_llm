"""Classification + bbox metrics for VisA→MVTec evaluation."""

from __future__ import annotations

from typing import Optional

from reasoning.rewards import box_iou, qwen1000_to_pixels_strict, valid_bbox_1000


def classification_correct(pred_cls, is_anomaly: bool) -> bool:
    return pred_cls is not None and bool(pred_cls) == bool(is_anomaly)


def box_ious_from_parsed(parsed: dict, meta: dict):
    orig = tuple(meta.get("orig_size") or (1, 1))
    gt = meta.get("gt_box_px")
    iou_f = 0.0
    iou_c = 0.0
    if gt is not None:
        pred = parsed.get("bbox_2d")
        cand = parsed.get("candidate_bbox")
        if valid_bbox_1000(pred):
            iou_f = box_iou(qwen1000_to_pixels_strict(pred, orig), gt)
        if valid_bbox_1000(cand):
            iou_c = box_iou(qwen1000_to_pixels_strict(cand, orig), gt)
    rec_ok = classification_correct(parsed.get("is_anomaly", None), bool(meta.get("is_anomaly")))
    return float(iou_c), float(iou_f), rec_ok


def is_truncated(seq, prompt_len: int, max_new: int, eos_id: Optional[int]) -> bool:
    n_new = int(seq.numel()) - int(prompt_len)
    if n_new < int(max_new):
        return False
    if eos_id is None:
        return True
    return int(seq[-1].item()) != int(eos_id)
