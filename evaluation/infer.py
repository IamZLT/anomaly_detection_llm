"""Shared generation helpers for demo.py and the Flask app."""

from __future__ import annotations

from typing import Any, Dict

import torch
from PIL import Image


def build_generation_inputs(
    cfg: dict,
    processor,
    image: Image.Image,
    prompt: str,
    **_unused,
) -> Dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }
    ]
    enable_thinking = bool((cfg.get("prompt") or {}).get("enable_thinking", False))
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        text = processor.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        text = processor.apply_chat_template(messages, **kwargs)
    return processor(text=[text], images=[image], return_tensors="pt", padding=True)


def decode_generation_output(
    processor,
    outputs: torch.Tensor,
    inputs: Dict[str, Any],
    cfg: dict,
) -> str:
    tok = getattr(processor, "tokenizer", processor)
    gen = outputs[0]
    if gen.dim() > 1:
        gen = gen[0]
    if inputs.get("input_ids") is not None:
        plen = int(inputs["input_ids"].shape[1])
        if gen.numel() > plen:
            gen = gen[plen:]
    return tok.decode(gen, skip_special_tokens=True)
