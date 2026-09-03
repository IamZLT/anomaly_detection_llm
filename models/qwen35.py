"""Load Qwen3.5-VL (processor + ImageTextToText) and freeze the vision tower."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from utils.common import is_main_process


def unwrap_qwen_core(qwen: nn.Module) -> nn.Module:
    if hasattr(qwen, "get_base_model"):
        try:
            return qwen.get_base_model()
        except Exception:
            pass
    return qwen


def freeze_vision_encoder(model: nn.Module) -> int:
    """Freeze vision tower parameters. Call once after load (and after LoRA wrap if needed)."""
    candidates = []
    for name in ("visual", "vision_tower", "vision_model"):
        mod = getattr(model, name, None)
        if mod is not None:
            candidates.append(mod)
    inner = getattr(model, "model", None)
    if inner is not None:
        for name in ("visual", "vision_tower", "vision_model"):
            mod = getattr(inner, name, None)
            if mod is not None:
                candidates.append(mod)
    core = unwrap_qwen_core(model)
    visual = getattr(getattr(core, "model", None), "visual", None)
    if visual is not None:
        candidates.append(visual)
        visual.eval()
    n = 0
    seen = set()
    for mod in candidates:
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        for p in mod.parameters():
            if p.requires_grad:
                p.requires_grad = False
                n += p.numel()
    return n


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
    model.eval() if for_inference else model.train()
    return model, processor
