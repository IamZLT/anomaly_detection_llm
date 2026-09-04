"""TensorBoard panels for prior-guided CoT: heatmaps, case vis, GRPO trajectories."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter

from utils.common import qwen_norm1000_to_original_pixels
from reasoning.parser import parse_cot_output, rollout_protocol_stats


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


def vstack_labeled(pairs: List[Tuple[str, Image.Image]], gap: int = 6) -> Image.Image:
    """Stack labeled images vertically (one case per row) into a single panel."""
    if not pairs:
        return Image.new("RGB", (16, 16), (0, 0, 0))
    w = max(im.width for _, im in pairs)
    imgs = []
    for title, im in pairs:
        if im.width != w:
            h = max(1, int(im.height * w / im.width))
            im = im.resize((w, h), Image.Resampling.BILINEAR)
        cap = _caption_bar(w, title)
        canvas = Image.new("RGB", (w, im.height + cap.height), (20, 20, 20))
        canvas.paste(cap, (0, 0))
        canvas.paste(im.convert("RGB"), (0, cap.height))
        imgs.append(canvas)
    total_h = sum(im.height for im in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (w, total_h), (12, 12, 12))
    y = 0
    for im in imgs:
        out.paste(im, (0, y))
        y += im.height + gap
    return out


def draw_prior_points(image: Image.Image, points_1000: Optional[Sequence[Sequence[int]]], color=(0, 220, 255)) -> Image.Image:
    im = image.copy().convert("RGB")
    if not points_1000:
        return im
    draw = ImageDraw.Draw(im)
    w, h = im.size
    for p in points_1000:
        if p is None or len(p) < 2:
            continue
        x = int(round(float(p[0]) / 1000.0 * w))
        y = int(round(float(p[1]) / 1000.0 * h))
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
        draw.line((x - 7, y, x + 7, y), fill=color, width=2)
        draw.line((x, y - 7, x, y + 7), fill=color, width=2)
    return im


def make_heatmap_panel(
    ref: Image.Image,
    test: Image.Image,
    heat: Image.Image,
    alpha: float = 0.45,
    prior_points: Optional[Sequence[Sequence[int]]] = None,
) -> Image.Image:
    heat_rs = heat.convert("RGB").resize(test.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(test.convert("RGB"), heat_rs, float(np.clip(alpha, 0.0, 1.0)))
    overlay = draw_prior_points(overlay, prior_points)
    test_pts = draw_prior_points(test, prior_points)
    return hstack_labeled(
        [
            ("REF (normal)", ref),
            ("TEST", test_pts),
            ("PRIOR H", heat_rs),
            ("TEST + H + P_H", overlay),
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
    iou_c: float = 0.0,
) -> str:
    lines = [
        f"stage={stage} step={step}",
        f"image={meta.get('image_path')}",
        f"class={meta.get('class_name')} anomaly_gt={meta.get('is_anomaly')} rec_ok={rec_ok} "
        f"iou_f={iou:.3f} iou_c={iou_c:.3f}",
        f"pred={parsed.get('is_anomaly')} bbox={parsed.get('bbox_2d')} "
        f"description={parsed.get('description') or ''}",
        "",
        response or "",
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
            f"--- τ[{i}]  "
            f"Rf={det.get('R_final', 0):.3f} "
            f"Rg={det.get('R_ground', 0):.3f} "
            f"Rr={det.get('R_reason', 0):.3f} "
            f"cov={det.get('R_cov', 0):.3f} "
            f"dir={det.get('R_dir', 0):.3f} "
            f"dense_c={det.get('R_dense_c', 0):.3f} "
            f"dense_f={det.get('R_dense_f', 0):.3f} "
            f"Δdense={det.get('delta_dense', 0):+.3f} "
            f"iou_c={det.get('R_iou_c', 0):.3f} "
            f"iou_f={det.get('R_iou', 0):.3f} "
            f"Ac={det.get('candidate_area_ratio', 0):.3f} "
            f"Af={det.get('final_area_ratio', 0):.3f} "
            f"fmt={det.get('R_fmt', 0):.3f} "
            f"edge={det.get('R_edge', 0):.3f}{adv}{lp}"
        )
        lines.append((text or "")[:1800])
        lines.append("")
    return "\n".join(lines)


def _render_case(
    *,
    step: int,
    stage: str,
    meta: dict,
    response: str,
    parsed: dict,
    iou: float,
    rec_ok: bool,
    iou_c: float,
    overlay_alpha: float,
):
    """Build (heatmap_panel, bbox_vis, cot_text) for a single case."""
    ref = meta.get("ref")
    test = meta.get("test")
    heat = meta.get("heatmap")
    orig = tuple(meta.get("orig_size") or (test.size if test is not None else (1, 1)))
    pred_px = None
    cand_px = None
    if parsed.get("bbox_2d") is not None:
        pred_px = qwen_norm1000_to_original_pixels(parsed["bbox_2d"], orig)
    cand_box = parsed.get("candidate_bbox_2d")
    if cand_box is None:
        cand_box = parsed.get("candidate_bbox")
    if cand_box is not None:
        cand_px = qwen_norm1000_to_original_pixels(cand_box, orig)
    gt = meta.get("gt_box_px")

    panel = None
    if ref is not None and test is not None and heat is not None:
        panel = make_heatmap_panel(
            ref, test, heat, alpha=overlay_alpha, prior_points=meta.get("prior_points")
        )

    vis = None
    if test is not None:
        vis = draw_case_boxes(
            test,
            gt_orig=gt,
            pred_orig=pred_px,
            cand_orig=cand_px,
            orig_wh=orig,
        )

    cot = format_case_text(
        step=step,
        stage=stage,
        meta=meta,
        response=response,
        parsed=parsed,
        iou=iou,
        rec_ok=rec_ok,
        iou_c=iou_c,
    )
    return panel, vis, cot


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
    iou_c: float = 0.0,
) -> None:
    if writer is None and not save_dir:
        return
    panel, vis, cot = _render_case(
        step=step,
        stage=tag_prefix,
        meta=meta,
        response=response,
        parsed=parsed,
        iou=iou,
        rec_ok=rec_ok,
        iou_c=iou_c,
        overlay_alpha=overlay_alpha,
    )
    if writer is not None and log_heatmap and panel is not None:
        writer.add_image(f"{tag_prefix}/1_heatmap_compare", pil_to_tb(panel), step)
    if writer is not None and log_case and vis is not None:
        writer.add_image(f"{tag_prefix}/2_bbox_vis", pil_to_tb(vis), step)
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


def log_eval_cases_grid(
    writer: Optional[SummaryWriter],
    *,
    step: int,
    cases: List[dict],
    overlay_alpha: float = 0.45,
    save_dir: Optional[str] = None,
) -> None:
    """Consolidate multiple eval samples into ONE image panel + ONE text panel."""
    if writer is None and not save_dir:
        return
    rows: List[Tuple[str, Image.Image]] = []
    cot_parts: List[str] = []
    for ci, c in enumerate(cases):
        panel, vis, cot = _render_case(
            step=step,
            stage=f"eval_case_{ci}",
            meta=c["meta"],
            response=c["response"],
            parsed=c["parsed"],
            iou=float(c.get("iou", 0.0)),
            rec_ok=bool(c.get("rec_ok", False)),
            iou_c=float(c.get("iou_c", 0.0)),
            overlay_alpha=overlay_alpha,
        )
        parts = []
        if panel is not None:
            parts.append(("H", panel))
        if vis is not None:
            parts.append(("bbox", vis))
        if parts:
            meta = c["meta"]
            title = (
                f"#{ci} {meta.get('class_name')} gt_anom={meta.get('is_anomaly')} "
                f"pred={c['parsed'].get('is_anomaly')} iou={float(c.get('iou', 0.0)):.2f} "
                f"valid={c['parsed'].get('trajectory_valid')}"
            )
            rows.append((title, hstack_labeled(parts)))
        cot_parts.append(cot)
    if rows and writer is not None:
        writer.add_image("eval/cases_grid", pil_to_tb(vstack_labeled(rows)), step)
    if writer is not None:
        writer.add_text("eval/cases_cot", "\n\n".join(cot_parts), step)
        writer.flush()
    if save_dir and rows:
        import os
        os.makedirs(save_dir, exist_ok=True)
        vstack_labeled(rows).save(os.path.join(save_dir, "cases_grid.png"))


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
    """Write the small GRPO dashboard (same x-axis). Config lives in grpo/0_config text."""
    if writer is None:
        return
    _ = (params, opt_step, is_anomaly, seq_lp, advantages)
    r = rewards.detach().float().cpu()
    n = max(len(details), 1)
    mean = lambda k: float(sum(float(d.get(k, 0.0)) for d in details) / n)
    proto = rollout_protocol_stats([parse_cot_output(t) for t in texts], texts)
    extra = extra or {}

    writer.add_scalar("grpo/loss", float(loss), step)
    writer.add_scalar("grpo/lr", float(lr), step)
    if grad_norm is not None:
        writer.add_scalar("grpo/grad_norm", float(grad_norm), step)
    writer.add_scalar("grpo/pg_loss", float(extra.get("loss_pg", extra.get("pg_loss", 0.0))), step)
    writer.add_scalar("grpo/kl", float(extra.get("loss_kl", extra.get("kl", 0.0))), step)
    writer.add_scalar("grpo/rho", float(extra.get("rho_mean", extra.get("rho", 1.0))), step)
    writer.add_scalar("grpo/clip_frac", float(extra.get("clip_frac", 0.0)), step)

    writer.add_scalar("grpo/R_ground", mean("R_ground"), step)
    writer.add_scalar("grpo/R_reason", mean("R_reason"), step)
    writer.add_scalar("grpo/R_final", mean("R_final"), step)
    writer.add_scalar("grpo/reward_std", float(r.std(unbiased=False)), step)

    writer.add_scalar("grpo/R_iou_c", mean("R_iou_c"), step)
    writer.add_scalar("grpo/R_iou", mean("R_iou"), step)
    writer.add_scalar("grpo/delta_iou", mean("delta_iou"), step)
    writer.add_scalar("grpo/R_dir", mean("R_dir"), step)
    writer.add_scalar("grpo/raw_iou_f", mean("raw_iou_f"), step)
    writer.add_scalar("grpo/raw_iou_c", mean("raw_iou_c"), step)
    writer.add_scalar("grpo/R_fmt", mean("R_fmt"), step)
    writer.add_scalar("grpo/R_dense_c", mean("R_dense_c"), step)
    writer.add_scalar("grpo/R_dense_f", mean("R_dense_f"), step)
    writer.add_scalar("grpo/delta_dense", mean("delta_dense"), step)
    writer.add_scalar("grpo/candidate_area_ratio", mean("candidate_area_ratio"), step)
    writer.add_scalar("grpo/final_area_ratio", mean("final_area_ratio"), step)
    writer.add_scalar("grpo/pred_gt_area_ratio", mean("pred_gt_area_ratio"), step)
    full_image_box_rate = float(
        sum(
            1
            for d in details
            if float(d.get("full_image_cand", 0.0)) > 0.5
            or float(d.get("full_image_final", 0.0)) > 0.5
        )
        / n
    )
    writer.add_scalar("grpo/full_image_box_rate", full_image_box_rate, step)

    writer.add_scalar("grpo/protocol_rate", float(proto.get("protocol_rate", 0.0)), step)
    writer.add_scalar("grpo/trajectory_valid_rate", float(proto.get("trajectory_valid_rate", 0.0)), step)
    writer.add_scalar("grpo/candidate_valid_rate", float(proto.get("candidate_valid_rate", 0.0)), step)
    writer.add_scalar("grpo/final_valid_rate", float(proto.get("final_valid_rate", 0.0)), step)
    writer.add_scalar("grpo/box_pair_valid_rate", float(proto.get("box_pair_valid_rate", 0.0)), step)
    writer.add_scalar("grpo/unique_response_rate", float(proto.get("unique_response_rate", 0.0)), step)
    writer.add_scalar("grpo/resample_n", float(extra.get("resample", extra.get("resample_n", 0.0))), step)
    writer.add_scalar("grpo/skip_rate", float(extra.get("skipped", extra.get("skip_rate", 0.0))), step)
    writer.flush()



import os
import re
import shutil
import signal
import socket
import subprocess
import time
from typing import Optional as _Optional


def tensorboard_event_dir(output_dir: str) -> str:
    d = os.path.abspath(os.path.join(output_dir, "tb"))
    os.makedirs(d, exist_ok=True)
    return d


def _tensorboard_logdir(cfg: dict, output_dir: str) -> str:
    tb_cfg = cfg.get("tensorboard") or {}
    override = tb_cfg.get("logdir")
    if override not in (None, "", "null", "None"):
        return os.path.abspath(os.path.expanduser(str(override)))
    return tensorboard_event_dir(output_dir)


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _pid_listening_on_port(port: int):
    try:
        out = subprocess.check_output(["ss", "-lptn", f"sport = :{port}"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    m = re.search(r"pid=(\d+)", out)
    return int(m.group(1)) if m else None


def _proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except Exception:
        return ""


def _cmdline_logdir(cmdline: str) -> str:
    m = re.search(r"--logdir(?:\s+|=)(\S+)", cmdline)
    if not m:
        return ""
    return os.path.abspath(m.group(1))


def auto_start_tensorboard(cfg: dict, output_dir: str) -> None:
    tb_cfg = cfg.get("tensorboard", {})
    if not bool(tb_cfg.get("auto_start", True)):
        return
    host = str(tb_cfg.get("host", "0.0.0.0"))
    port = int(tb_cfg.get("port", 5000))
    probe_host = "127.0.0.1" if host == "0.0.0.0" else host
    log_dir = _tensorboard_logdir(cfg, output_dir)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    if _is_port_in_use(probe_host, port):
        pid = _pid_listening_on_port(port)
        cmd_now = _proc_cmdline(pid) if pid else ""
        watching = _cmdline_logdir(cmd_now)
        if pid and "tensorboard" in cmd_now and watching == os.path.abspath(log_dir):
            print(f"[TensorBoard] 端口 {port} 已在看 {log_dir}: http://127.0.0.1:{port}")
            return
        if pid and "tensorboard" in cmd_now:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.8)
                print(f"[TensorBoard] 旧进程 pid={pid} 盯的是 {watching or '?'}，已重启为 {log_dir}")
            except Exception as e:
                print(f"[TensorBoard] 端口 {port} 被占用且无法重启 ({e}): http://127.0.0.1:{port}")
                return
        else:
            print(f"[TensorBoard] 端口 {port} 被其他进程占用，无法启动: http://127.0.0.1:{port}")
            return
    tb_bin = shutil.which("tensorboard")
    if tb_bin is None:
        print("[TensorBoard] 未找到 tensorboard 命令，跳过自动启动。")
        return
    cmd = [tb_bin, "--logdir", log_dir, "--host", host, "--port", str(port), "--reload_multifile", "true", "--reload_interval", "5"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        print(f"[TensorBoard] 已自动启动 (pid={proc.pid}) logdir={log_dir}: http://127.0.0.1:{port}")
    except Exception as e:
        print(f"[TensorBoard] 自动启动失败: {e}")
