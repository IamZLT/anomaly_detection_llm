"""LoRA on Qwen3.5 language side; vision tower stays frozen."""

from __future__ import annotations

import os

import torch.nn as nn


def _is_main_process() -> bool:
    r = os.environ.get("RANK")
    if r is not None:
        return int(r) == 0
    lr = os.environ.get("LOCAL_RANK")
    if lr is not None:
        return int(lr) == 0
    return True


def _unwrap_qwen_core(qwen: nn.Module) -> nn.Module:
    if hasattr(qwen, "get_base_model"):
        try:
            return qwen.get_base_model()
        except Exception:
            pass
    return qwen


def apply_lora_to_qwen_llm(qwen: nn.Module, cfg: dict) -> nn.Module:
    lora_cfg = cfg.get("lora", {}) or {}
    if not bool(lora_cfg.get("enabled", False)):
        return qwen
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError("启用 lora.enabled 需要安装 peft：pip install peft") from e

    target = lora_cfg.get("target_modules") or [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    skip_visual = bool(lora_cfg.get("skip_visual", True))
    target_modules: list = list(target)
    if skip_visual:
        full_names = []
        skip_tokens = ("visual", "vision_tower", "vision_model", "patch_embed", "merger")
        leaf_set = set(target)
        for name, mod in qwen.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if any(t in name for t in skip_tokens):
                continue
            if name.split(".")[-1] in leaf_set:
                full_names.append(name)
        if full_names:
            target_modules = full_names

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    core = _unwrap_qwen_core(qwen)
    visual = getattr(getattr(core, "model", None), "visual", None)
    if visual is not None:
        for p in visual.parameters():
            p.requires_grad = False

    qwen = get_peft_model(qwen, peft_config)
    core = _unwrap_qwen_core(qwen)
    visual = getattr(getattr(core, "model", None), "visual", None)
    if visual is not None:
        for p in visual.parameters():
            p.requires_grad = False
    if _is_main_process():
        print(
            f"[lora] enabled r={peft_config.r} alpha={peft_config.lora_alpha} "
            f"targets={target}",
            flush=True,
        )
        if hasattr(qwen, "print_trainable_parameters"):
            qwen.print_trainable_parameters()
    return qwen
