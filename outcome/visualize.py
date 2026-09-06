"""TensorBoard visualization for outcome-v1: heatmaps + GT/Bc/Bf boxes + CoT text.

Reuses the low-level PIL helpers from visualization.tensorboard so the panels
look identical to the legacy GRPO runs, but adapts to the outcome parse_output
schema (0-1000 coordinates, JSON blocks, task_valid/protocol_valid).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PIL import Image

from outcome.protocol import to_pixels
from visualization.tensorboard import (
    draw_case_boxes,
    hstack_labeled,
    make_heatmap_panel,
    pil_to_tb,
    vstack_labeled,
)


def _fmt_box(box) -> str:
    if box is None:
        return 'null'
    return '[' + ','.join(str(int(round(v))) for v in box) + ']'


def format_outcome_case_text(
    *,
    step: int,
    meta: dict,
    response: str,
    parsed: dict,
    iou: float,
    loc_reward: float,
    correct: bool,
) -> str:
    lines = [
        f"step={step}",
        f"image={meta.get('image_path')}",
        f"class={meta.get('class_name')} anomaly_gt={meta.get('is_anomaly')} correct={correct} "
        f"iou_f={iou:.3f} loc_reward={loc_reward:.3f}",
        f"pred={parsed.get('is_anomaly')} bbox_2d={_fmt_box(parsed.get('bbox_2d'))} "
        f"candidate_bbox_2d={_fmt_box(parsed.get('candidate_bbox_2d'))}",
        f"task_valid={parsed.get('task_valid')} protocol_valid={parsed.get('protocol_valid')} "
        f"action={parsed.get('action')}",
        f"description={parsed.get('description') or ''}",
        "",
        response or "",
    ]
    return "\n".join(lines)


def render_outcome_case(
    *,
    meta: dict,
    response: str,
    parsed: dict,
    iou: float,
    loc_reward: float,
    correct: bool,
    step: int = 0,
    overlay_alpha: float = 0.45,
) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    """Build (heatmap_panel, bbox_vis, cot_text) for a single outcome case."""
    ref = meta.get('ref')
    test = meta.get('test')
    heat = meta.get('heatmap')
    orig = tuple(meta.get('orig_size') or (test.size if test is not None else (1, 1)))
    prior_points = meta.get('prior_points')

    panel = None
    if ref is not None and test is not None and heat is not None:
        panel = make_heatmap_panel(ref, test, heat, alpha=overlay_alpha, prior_points=prior_points)

    vis = None
    if test is not None:
        pred_px = to_pixels(parsed.get('bbox_2d'), orig) if parsed.get('bbox_2d') is not None else None
        cand_px = to_pixels(parsed.get('candidate_bbox_2d'), orig) if parsed.get('candidate_bbox_2d') is not None else None
        vis = draw_case_boxes(
            test,
            gt_orig=meta.get('gt_box_px'),
            pred_orig=pred_px,
            cand_orig=cand_px,
            orig_wh=orig,
        )

    cot = format_outcome_case_text(
        step=step,
        meta=meta,
        response=response,
        parsed=parsed,
        iou=iou,
        loc_reward=loc_reward,
        correct=correct,
    )
    return panel, vis, cot


def log_outcome_eval_grid(
    writer,
    *,
    step: int,
    cases: List[dict],
    overlay_alpha: float = 0.45,
    max_cases: int = 16,
) -> None:
    """Consolidate multiple eval samples into ONE heatmap panel + ONE text panel."""
    if writer is None:
        return
    rows: List[Tuple[str, Image.Image]] = []
    cot_parts: List[str] = []
    for ci, c in enumerate(cases[:max_cases]):
        panel, vis, cot = render_outcome_case(
            meta=c['meta'],
            response=c.get('response', ''),
            parsed=c['parsed'],
            iou=float(c.get('iou', 0.0)),
            loc_reward=float(c.get('loc_reward', 0.0)),
            correct=bool(c.get('correct', False)),
            step=step,
            overlay_alpha=overlay_alpha,
        )
        parts = []
        if panel is not None:
            parts.append(('H', panel))
        if vis is not None:
            parts.append(('bbox', vis))
        if parts:
            m = c['meta']
            title = (
                f"#{ci} {m.get('class_name')} gt_anom={m.get('is_anomaly')} "
                f"pred={c['parsed'].get('is_anomaly')} iou={float(c.get('iou', 0.0)):.2f} "
                f"proto={c['parsed'].get('protocol_valid')}"
            )
            rows.append((title, hstack_labeled(parts)))
        cot_parts.append(cot)
    if rows:
        writer.add_image('eval/cases_grid', pil_to_tb(vstack_labeled(rows)), step)
    writer.add_text('eval/cases_cot', '\n\n'.join(cot_parts), step)
    writer.flush()


def log_outcome_single_case(
    writer,
    *,
    step: int,
    meta: dict,
    response: str,
    parsed: dict,
    iou: float,
    loc_reward: float,
    correct: bool,
    tag_prefix: str = 'train',
    overlay_alpha: float = 0.45,
) -> None:
    """One heatmap + bbox panel + CoT text for a single rollout."""
    if writer is None:
        return
    panel, vis, cot = render_outcome_case(
        meta=meta, response=response, parsed=parsed,
        iou=iou, loc_reward=loc_reward, correct=correct,
        step=step, overlay_alpha=overlay_alpha,
    )
    if panel is not None:
        writer.add_image(f'{tag_prefix}/1_heatmap', pil_to_tb(panel), step)
    if vis is not None:
        writer.add_image(f'{tag_prefix}/2_bbox', pil_to_tb(vis), step)
    writer.add_text(f'{tag_prefix}/3_cot', cot, step)
    writer.flush()
