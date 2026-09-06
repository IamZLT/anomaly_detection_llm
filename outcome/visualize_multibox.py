"""TensorBoard visualization for outcome-multibox-v1: heatmap + component/multi-box overlay."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from outcome.protocol import to_pixels
from visualization.tensorboard import (
    _font,
    hstack_labeled,
    make_heatmap_panel,
    orig_box_to_resized,
    pil_to_tb,
    vstack_labeled,
)


def _draw_box(draw, box, color, label):
    if box is None:
        return
    xy = orig_box_to_resized(box, draw._orig_wh, draw.im.size)
    draw.rectangle(xy, outline=color, width=3)
    draw.text((xy[0] + 3, max(0, xy[1] - 18)), label, fill=color, font=draw._font)


def _draw_boxes_multibox(
    image: Image.Image,
    *,
    gt_components: Optional[Sequence[Sequence[float]]],
    gt_union: Optional[Sequence[float]],
    pred_boxes: Sequence[Sequence[float]],
    cand_boxes: Sequence[Sequence[float]],
    orig_wh: Tuple[int, int],
) -> Image.Image:
    im = image.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    draw._orig_wh = orig_wh
    draw._font = _font(14)
    for g in (gt_components or []):
        _draw_box(draw, g, (0, 220, 0), "GT")
    if gt_union is not None:
        xy = orig_box_to_resized(gt_union, orig_wh, im.size)
        draw.rectangle(xy, outline=(120, 200, 120), width=2)
        draw.text((xy[0] + 3, max(0, xy[3] - 18)), "union", fill=(120, 200, 120), font=draw._font)
    for b in cand_boxes:
        _draw_box(draw, b, (255, 165, 0), "Bc")
    for b in pred_boxes:
        _draw_box(draw, b, (255, 0, 0), "Bf")
    return im


def _fmt_boxes(boxes) -> str:
    if not boxes:
        return '[]'
    return '[' + ' '.join('[' + ','.join(str(int(round(v))) for v in b) + ']' for b in boxes) + ']'


def format_outcome_case_text(step, meta, response, parsed, union_iou, loc_reward, correct) -> str:
    lines = [
        f"step={step}",
        f"image={meta.get('image_path')}",
        f"class={meta.get('class_name')} anomaly_gt={meta.get('is_anomaly')} correct={correct} "
        f"union_iou={union_iou:.3f} loc_reward={loc_reward:.3f}",
        f"pred={parsed.get('is_anomaly')} bboxes_2d={_fmt_boxes(parsed.get('bboxes_2d'))} "
        f"candidate_bboxes_2d={_fmt_boxes(parsed.get('candidate_bboxes_2d'))}",
        f"task_valid={parsed.get('task_valid')} core={parsed.get('protocol_core')} "
        f"strict={parsed.get('protocol_strict')} action={parsed.get('action')}",
        f"num_boxes={parsed.get('num_boxes')} num_components={meta.get('num_components')}",
        f"description={parsed.get('description') or ''}",
        "",
        response or "",
    ]
    return "\n".join(lines)


def render_outcome_case(meta, response, parsed, union_iou, loc_reward, correct, step=0, overlay_alpha=0.45):
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
        pred_px = [to_pixels(b, orig) for b in parsed.get('bboxes_2d') or []]
        cand_px = [to_pixels(b, orig) for b in parsed.get('candidate_bboxes_2d') or []]
        vis = _draw_boxes_multibox(
            test,
            gt_components=meta.get('component_bboxes'),
            gt_union=meta.get('gt_box_px'),
            pred_boxes=pred_px,
            cand_boxes=cand_px,
            orig_wh=orig,
        )

    cot = format_outcome_case_text(step, meta, response, parsed, union_iou, loc_reward, correct)
    return panel, vis, cot


def log_outcome_eval_grid(writer, *, step, cases, overlay_alpha=0.45, max_cases=16):
    if writer is None:
        return
    rows = []
    cot_parts = []
    for ci, c in enumerate(cases[:max_cases]):
        panel, vis, cot = render_outcome_case(
            c['meta'], c.get('response', ''), c['parsed'],
            float(c.get('union_iou', 0.0)), float(c.get('loc_reward', 0.0)),
            bool(c.get('correct', False)), step=step, overlay_alpha=overlay_alpha,
        )
        parts = []
        if panel is not None:
            parts.append(('H', panel))
        if vis is not None:
            parts.append(('bbox', vis))
        if parts:
            m = c['meta']
            title = (f"#{ci} {m.get('class_name')} gt_anom={m.get('is_anomaly')} "
                     f"pred={c['parsed'].get('is_anomaly')} union_iou={float(c.get('union_iou', 0.0)):.2f} "
                     f"core={c['parsed'].get('protocol_core')} strict={c['parsed'].get('protocol_strict')}")
            rows.append((title, hstack_labeled(parts)))
        cot_parts.append(cot)
    if rows:
        writer.add_image('eval/cases_grid', pil_to_tb(vstack_labeled(rows)), step)
    writer.add_text('eval/cases_cot', '\n\n'.join(cot_parts), step)
    writer.flush()


def log_outcome_single_case(writer, *, step, meta, response, parsed, union_iou, loc_reward, correct,
                            tag_prefix='train', overlay_alpha=0.45):
    if writer is None:
        return
    panel, vis, cot = render_outcome_case(
        meta, response, parsed, union_iou, loc_reward, correct,
        step=step, overlay_alpha=overlay_alpha,
    )
    if panel is not None:
        writer.add_image(f'{tag_prefix}/1_heatmap', pil_to_tb(panel), step)
    if vis is not None:
        writer.add_image(f'{tag_prefix}/2_bbox', pil_to_tb(vis), step)
    writer.add_text(f'{tag_prefix}/3_cot', cot, step)
    writer.flush()
