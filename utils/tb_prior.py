"""TensorBoard panels for prior-guided CoT: heatmaps, case vis, GRPO trajectories."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter

from utils.common import qwen_norm1000_to_original_pixels
from utils.prior_cot import parse_cot_output


def _font(size: int = 16):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pil_to_tb(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1)


def orig_box_to_resized(
    box: Sequence[float], orig_wh: Tuple[int, int], resized_wh: Tuple[int, int]
) -> List[int]:
    ow, oh = float(max(orig_wh[0], 1)), float(max(orig_wh[1], 1))
    rw, rh = float(resized_wh[0]), float(resized_wh[1])
    return [
        int(round(box[0] / ow * rw)),
        int(round(box[1] / oh * rh)),
        int(round(box[2] / ow * rw)),
        int(round(box[3] / oh * rh)),
    ]


def draw_case_boxes(
    image: Image.Image,
    *,
    gt_orig: Optional[Sequence[float]],
    pred_orig: Optional[Sequence[float]],
    cand_orig: Optional[Sequence[float]],
    orig_wh: Tuple[int, int],
) -> Image.Image:
    im = image.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _font(16)
    rw, rh = im.size

    def _one(box, color, label):
        if box is None:
            return
        xy = orig_box_to_resized(box, orig_wh, (rw, rh))
        draw.rectangle(xy, outline=color, width=3)
        draw.text((xy[0] + 3, max(0, xy[1] - 18)), label, fill=color, font=font)

    _one(gt_orig, (0, 220, 0), "GT")
    _one(cand_orig, (255, 200, 0), "Bc")
    _one(pred_orig, (255, 40, 40), "Bf")
    return im


def _caption_bar(width: int, text: str, fill=(30, 30, 30)) -> Image.Image:
    bar = Image.new("RGB", (width, 28), fill)
    draw = ImageDraw.Draw(bar)
    draw.text((8, 5), text, fill=(240, 240, 240), font=_font(16))
    return bar


def hstack_labeled(pairs: List[Tuple[str, Image.Image]], gap: int = 6) -> Image.Image:
    if not pairs:
        return Image.new("RGB", (16, 16), (0, 0, 0))
    h = max(im.height for _, im in pairs)
    imgs = []
    for title, im in pairs:
        if im.height != h:
            w = max(1, int(im.width * h / im.height))
            im = im.resize((w, h), Image.Resampling.BILINEAR)
        cap = _caption_bar(im.width, title)
        canvas = Image.new("RGB", (im.width, h + cap.height), (20, 20, 20))
        canvas.paste(cap, (0, 0))
        canvas.paste(im.convert("RGB"), (0, cap.height))
        imgs.append(canvas)
    total_w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (total_w, imgs[0].height), (12, 12, 12))
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


def make_heatmap_panel(ref: Image.Image, test: Image.Image, heat: Image.Image, alpha: float = 0.45) -> Image.Image:
    heat_rs = heat.convert("RGB").resize(test.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(test.convert("RGB"), heat_rs, float(np.clip(alpha, 0.0, 1.0)))
    return hstack_labeled(
        [
            ("REF (normal)", ref),
            ("TEST", test),
            ("PRIOR H", heat_rs),
            ("TEST + H", overlay),
        ]
    )


def pretty_json(obj: Any, limit: int = 4000) -> str:
    if isinstance(obj, str):
        s = obj
        try:
            s = json.dumps(json.loads(obj), ensure_ascii=False, indent=2)
        except Exception:
            s = obj
    else:
        try:
            s = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        except Exception:
            s = str(obj)
    if len(s) > limit:
        s = s[:limit] + "\n…(truncated)"
    return s


def format_case_text(
    *,
    step: int,
    stage: str,
    meta: dict,
    response: str,
    parsed: dict,
    iou: float,
    rec_ok: bool,
) -> str:
    lines = [
        f"stage={stage} step={step}",
        f"image={meta.get('image_path')}",
        f"class={meta.get('class_name')} anomaly_gt={meta.get('is_anomaly')} rec_ok={rec_ok} iou={iou:.3f}",
        "",
        "=== model CoT ===",
        (response or "")[:4000],
        "",
        "=== parsed ===",
        f"has_tags={parsed.get('has_tags')} is_anomaly={parsed.get('is_anomaly')} label={parsed.get('label')}",
        f"candidate_bbox={parsed.get('candidate_bbox')}",
        f"boundary={parsed.get('boundary')}",
        f"bbox_2d={parsed.get('bbox_2d')}",
    ]
    return "\n".join(lines)


def format_grpo_group_text(
    *,
    step: int,
    meta: dict,
    texts: List[str],
    details: List[dict],
    advantages: Optional[Sequence[float]] = None,
    logprobs: Optional[Sequence[float]] = None,
) -> str:
    lines = [
        f"GRPO CoT group  step={step}  image={meta.get('image_path')}  class={meta.get('class_name')}",
        "",
    ]
    for i, (text, det) in enumerate(zip(texts, details)):
        adv = f" adv={float(advantages[i]):+.3f}" if advantages is not None else ""
        lp = f" logp={float(logprobs[i]):.3f}" if logprobs is not None else ""
        lines.append(
            f"--- τ[{i}]  Rb={det.get('R_box', 0):.3f} Rg={det.get('R_ground', 0):.3f} "
            f"Rr={det.get('R_reason', 0):.3f} cov={det.get('R_cov', 0):.3f} "
            f"dir={det.get('R_dir', 0):.3f} iou={det.get('R_iou', 0):.3f} "
            f"edge={det.get('R_edge', 0):.3f} ctr={det.get('R_center', 0):.3f}{adv}{lp}"
        )
        lines.append((text or "")[:1800])
        lines.append("")
    return "\n".join(lines)


def log_heatmap_and_case(
    writer: Optional[SummaryWriter],
    *,
    step: int,
    tag_prefix: str,
    meta: dict,
    response: str,
    parsed: dict,
    iou: float,
    rec_ok: bool,
    overlay_alpha: float = 0.45,
    save_dir: Optional[str] = None,
    log_heatmap: bool = True,
    log_case: bool = True,
) -> None:
    if writer is None and not save_dir:
        return
    ref = meta.get("ref")
    test = meta.get("test")
    heat = meta.get("heatmap")
    orig = tuple(meta.get("orig_size") or (test.size if test is not None else (1, 1)))
    pred_px = None
    cand_px = None
    if parsed.get("bbox_2d") is not None:
        pred_px = qwen_norm1000_to_original_pixels(parsed["bbox_2d"], orig)
    if parsed.get("candidate_bbox") is not None:
        cand_px = qwen_norm1000_to_original_pixels(parsed["candidate_bbox"], orig)
    gt = meta.get("gt_box_px")

    panel = None
    vis = None
    if ref is not None and test is not None and heat is not None:
        panel = make_heatmap_panel(ref, test, heat, alpha=overlay_alpha)
        if writer is not None and log_heatmap:
            writer.add_image(f"{tag_prefix}/1_heatmap_compare", pil_to_tb(panel), step)

    if test is not None:
        vis = draw_case_boxes(
            test,
            gt_orig=gt,
            pred_orig=pred_px,
            cand_orig=cand_px,
            orig_wh=orig,
        )
        if writer is not None and log_case:
            writer.add_image(f"{tag_prefix}/2_bbox_vis", pil_to_tb(vis), step)

    cot = format_case_text(
        step=step,
        stage=tag_prefix,
        meta=meta,
        response=response,
        parsed=parsed,
        iou=iou,
        rec_ok=rec_ok,
    )
    if writer is not None and log_case:
        writer.add_text(f"{tag_prefix}/3_cot", cot, step)
        writer.flush()
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        if panel is not None:
            panel.save(os.path.join(save_dir, f"{tag_prefix}_heatmap.png"))
        if vis is not None:
            vis.save(os.path.join(save_dir, f"{tag_prefix}_bbox.png"))
        with open(os.path.join(save_dir, f"{tag_prefix}_cot.txt"), "w", encoding="utf-8") as f:
            f.write(cot)


def log_grpo_run_config(writer: Optional[SummaryWriter], cfg: dict) -> None:
    """Dump the GRPO hyperparams into TEXT so they sit next to the curves."""
    if writer is None:
        return
    gcfg = cfg.get("grpo") or {}
    payload = {
        "grpo": gcfg,
        "lora": cfg.get("lora") or {},
        "prior": cfg.get("prior") or {},
        "training": {
            "eval_every_n_steps": (cfg.get("training") or {}).get("eval_every_n_steps"),
            "eval_num_samples": (cfg.get("training") or {}).get("eval_num_samples"),
            "seed": (cfg.get("training") or {}).get("seed"),
        },
        "tensorboard": cfg.get("tensorboard") or {},
    }
    writer.add_text("grpo/0_config", json.dumps(payload, ensure_ascii=False, indent=2, default=str), 0)
    writer.flush()


def log_grpo_scalars(
    writer: Optional[SummaryWriter],
    *,
    step: int,
    loss: float,
    rewards: torch.Tensor,
    details: List[dict],
    advantages: torch.Tensor,
    seq_lp: torch.Tensor,
    texts: List[str],
    lr: float,
    params: Optional[Dict[str, float]] = None,
    grad_norm: Optional[float] = None,
    opt_step: Optional[int] = None,
    is_anomaly: Optional[bool] = None,
    extra: Optional[Dict[str, float]] = None,
) -> None:
    """Write GRPO hyperparams and metrics at the same step (same x-axis in TB)."""
    if writer is None:
        return
    r = rewards.detach().float().cpu()
    adv = advantages.detach().float().cpu()
    lp = seq_lp.detach().float().cpu()
    n = max(len(details), 1)
    mean = lambda k: float(sum(float(d.get(k, 0.0)) for d in details) / n)
    parse_ok = 0
    box_ok = 0
    for t in texts:
        p = parse_cot_output(t)
        if p.get("has_tags"):
            parse_ok += 1
        if p.get("bbox_2d") is not None:
            box_ok += 1

    pmap = dict(params or {})
    for name, val in pmap.items():
        try:
            writer.add_scalar(f"grpo/param/{name}", float(val), step)
        except (TypeError, ValueError):
            continue

    writer.add_scalar("grpo/loss", float(loss), step)
    writer.add_scalar("grpo/lr", float(lr), step)
    writer.add_scalar("grpo/reward_mean", float(r.mean()), step)
    writer.add_scalar("grpo/reward_std", float(r.std(unbiased=False)), step)
    writer.add_scalar("grpo/reward_max", float(r.max()), step)
    writer.add_scalar("grpo/reward_min", float(r.min()), step)
    writer.add_scalar("grpo/R_cov", mean("R_cov"), step)
    writer.add_scalar("grpo/R_compact", mean("R_compact"), step)
    writer.add_scalar("grpo/R_dir", mean("R_dir"), step)
    writer.add_scalar("grpo/R_iou", mean("R_iou"), step)
    writer.add_scalar("grpo/R_edge", mean("R_edge"), step)
    writer.add_scalar("grpo/R_center", mean("R_center"), step)
    writer.add_scalar("grpo/R_format", mean("R_format"), step)
    writer.add_scalar("grpo/R_ground", mean("R_ground"), step)
    writer.add_scalar("grpo/R_reason", mean("R_reason"), step)
    writer.add_scalar("grpo/R_box", mean("R_box"), step)
    writer.add_scalar("grpo/advantage_mean", float(adv.mean()), step)
    writer.add_scalar("grpo/advantage_std", float(adv.std(unbiased=False)), step)
    writer.add_scalar("grpo/logprob_mean", float(lp.mean()), step)
    writer.add_scalar("grpo/format_parse_rate", parse_ok / max(len(texts), 1), step)
    writer.add_scalar("grpo/valid_bbox_rate", box_ok / max(len(texts), 1), step)
    if is_anomaly is not None:
        writer.add_scalar("grpo/batch_is_anomaly", 1.0 if is_anomaly else 0.0, step)
    if grad_norm is not None:
        writer.add_scalar("grpo/grad_norm", float(grad_norm), step)
    if opt_step is not None:
        writer.add_scalar("grpo/opt_step", float(opt_step), step)
    for name, val in (extra or {}).items():
        try:
            writer.add_scalar(f"grpo/{name}", float(val), step)
        except (TypeError, ValueError):
            continue
    for i, det in enumerate(details[:8]):
        writer.add_scalar(f"grpo/traj_{i}/R_box", float(det.get("R_box", det.get("R", 0.0))), step)
        writer.add_scalar(f"grpo/traj_{i}/R_ground", float(det.get("R_ground", 0.0)), step)
        writer.add_scalar(f"grpo/traj_{i}/R_iou", float(det.get("R_iou", 0.0)), step)
        if i < adv.numel():
            writer.add_scalar(f"grpo/traj_{i}/advantage_box", float(adv[i]), step)
    writer.flush()

