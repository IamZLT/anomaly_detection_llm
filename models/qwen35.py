"""Load Qwen3.5-VL (processor + ImageTextToText) and freeze the vision tower."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from utils.common import is_main_process


def qwen_vision_factor(processor=None, visual=None) -> int:
    """Qwen3.5 spatial alignment unit: patch_size * spatial_merge_size (typically 16*2=32)."""
    patch, merge = 16, 2
    img = getattr(processor, "image_processor", processor) if processor is not None else None
    if img is not None:
        p = getattr(img, "patch_size", None)
        m = getattr(img, "merge_size", None) or getattr(img, "spatial_merge_size", None)
        if p is not None:
            patch = int(p[-1] if isinstance(p, (list, tuple)) else p)
        if m is not None:
            merge = int(m)
    if visual is not None:
        cfg = getattr(visual, "config", visual)
        p = getattr(cfg, "patch_size", None)
        m = getattr(cfg, "spatial_merge_size", None)
        if p is not None:
            patch = int(p[-1] if isinstance(p, (list, tuple)) else p)
        if m is not None:
            merge = int(m)
    return max(int(patch) * int(merge), 1)


def apply_processor_geometry(processor, cfg: dict, visual=None) -> int:
    """Sync the official image processor to the vision tower and our pixel budget."""
    img = getattr(processor, "image_processor", None)
    if img is None:
        return qwen_vision_factor(processor, visual)
    if visual is not None:
        vcfg = getattr(visual, "config", visual)
        patch = getattr(vcfg, "patch_size", None)
        merge = getattr(vcfg, "spatial_merge_size", None)
        if patch is not None:
            img.patch_size = int(patch[-1] if isinstance(patch, (list, tuple)) else patch)
        if merge is not None and hasattr(img, "merge_size"):
            img.merge_size = int(merge)
    factor = qwen_vision_factor(processor, visual)
    data = cfg.get("data") or {}
    max_size = int(data.get("max_image_size", 448))
    size = getattr(img, "size", None)
    official_min_pixels = None
    if isinstance(size, dict):
        official_min_pixels = size.get("shortest_edge")
    if official_min_pixels is None:
        official_min_pixels = getattr(img, "min_pixels", None)
    if official_min_pixels is None:
        official_min_pixels = 256 * 256
    cfg_min = data.get("min_pixels")
    cfg_max = data.get("max_pixels")
    min_pixels = (
        int(cfg_min) if cfg_min not in (None, "", "null", "None") else int(official_min_pixels)
    )
    max_pixels = (
        int(cfg_max) if cfg_max not in (None, "", "null", "None") else max_size * max_size
    )
    min_pixels = min(min_pixels, max_pixels)
    if hasattr(img, "min_pixels"):
        img.min_pixels = int(min_pixels)
    if hasattr(img, "max_pixels"):
        img.max_pixels = int(max_pixels)
    size = getattr(img, "size", None)
    if isinstance(size, dict):
        size["shortest_edge"] = int(min_pixels)
        size["longest_edge"] = int(max_pixels)
    cfg_factor = data.get("factor")
    if cfg_factor not in (None, "", "null", "None"):
        want = int(cfg_factor)
        if want != factor and is_main_process():
            print(f"[setup] data.factor={want} ignored; using native vision factor={factor}", flush=True)
    if is_main_process():
        print(
            f"[setup] Qwen vision geometry factor={factor} "
            f"min_pixels={min_pixels} max_pixels={max_pixels} "
            f"(patch={getattr(img, 'patch_size', '?')} merge={getattr(img, 'merge_size', '?')})",
            flush=True,
        )
    return factor


def unwrap_qwen_core(qwen: nn.Module) -> nn.Module:
    if hasattr(qwen, "get_base_model"):
        try:
            return qwen.get_base_model()
        except Exception:
            pass
    return qwen


def _unwrap_maybe_ddp(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _vision_modules(model: nn.Module) -> list:
    m = _unwrap_maybe_ddp(model)
    candidates = []
    for name in ("visual", "vision_tower", "vision_model"):
        mod = getattr(m, name, None)
        if mod is not None:
            candidates.append(mod)
    inner = getattr(m, "model", None)
    if inner is not None:
        for name in ("visual", "vision_tower", "vision_model"):
            mod = getattr(inner, name, None)
            if mod is not None:
                candidates.append(mod)
    core = unwrap_qwen_core(m)
    visual = getattr(getattr(core, "model", None), "visual", None)
    if visual is not None:
        candidates.append(visual)
    seen = set()
    out = []
    for mod in candidates:
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        out.append(mod)
    return out


def freeze_vision_encoder(model: nn.Module) -> int:
    """Freeze vision tower parameters. Call once after load (and after LoRA wrap if needed)."""
    n = 0
    for mod in _vision_modules(model):
        mod.eval()
        for p in mod.parameters():
            if p.requires_grad:
                p.requires_grad = False
                n += p.numel()
    return n


def force_vision_eval(model: nn.Module) -> None:
    """Keep the vision tower in eval() after model.train() so Dropout cannot leak into H."""
    for mod in _vision_modules(model):
        mod.eval()


def setup_model_and_processor(
    cfg: dict,
    for_inference: bool = False,
    model_name_override: Optional[str] = None,
    freeze_vision: Optional[bool] = None,
) -> Tuple[nn.Module, Any]:
    model_name = model_name_override or cfg["model"]["name"]
    local_files_only = bool(cfg.get("model", {}).get("local_files_only", False))
    torch_dtype_cfg = cfg.get("model", {}).get("torch_dtype", "auto")
    model_cfg = cfg.get("model", {}) or {}

    if isinstance(torch_dtype_cfg, str) and torch_dtype_cfg != "auto":
        torch_dtype = getattr(torch, torch_dtype_cfg)
    else:
        torch_dtype = torch_dtype_cfg

    if is_main_process():
        print(f"[setup] Qwen-VL ← {model_name}", flush=True)

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    tok = getattr(processor, "tokenizer", processor)
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
        if hasattr(tok, "pad_token_id") and hasattr(tok, "eos_token_id"):
            tok.pad_token_id = tok.eos_token_id

    random_init_llm = bool(model_cfg.get("random_init_llm", False)) and not for_inference
    dtype_kw = {} if torch_dtype == "auto" else {"dtype": torch_dtype}
    try:
        if random_init_llm:
            llm_config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=local_files_only
            )
            model = AutoModelForImageTextToText.from_config(llm_config, trust_remote_code=True)
            if is_main_process():
                print("[setup] from_config（随机初始化）", flush=True)
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=local_files_only,
                **dtype_kw,
            )
            if is_main_process():
                print("[setup] from_pretrained 权重已加载", flush=True)
    except TypeError:
        if random_init_llm:
            llm_config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=local_files_only
            )
            model = AutoModelForImageTextToText.from_config(llm_config, trust_remote_code=True)
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=local_files_only,
                torch_dtype=None if torch_dtype == "auto" else torch_dtype,
            )

    do_freeze = bool(model_cfg.get("freeze_vit", False)) if freeze_vision is None else bool(freeze_vision)
    if do_freeze:
        n_fr = freeze_vision_encoder(model)
        if is_main_process():
            print(f"[setup] 视觉塔已冻结约 {n_fr / 1e6:.1f}M", flush=True)

    if is_main_process():
        n_param = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[setup] Qwen-VL 就绪：总参数 {n_param / 1e6:.1f}M，可训练 {n_train / 1e6:.1f}M",
            flush=True,
        )
    visual = None
    mods = _vision_modules(model)
    if mods:
        visual = mods[0]
    apply_processor_geometry(processor, cfg, visual=visual)

    if for_inference:
        model.eval()
    else:
        model.train()
        force_vision_eval(model)
    return model, processor
