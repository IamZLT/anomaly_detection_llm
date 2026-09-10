"""Independent input channel for region tokens.

The region tokens are ``<|region|>`` placeholders positioned in the input sequence as
ordinary text (``mm_token_type_id == 0``, so they get 1-D RoPE positions like text).
Their embeddings are replaced by ``RegionAdapter`` output via a ``get_input_embeddings``
monkey-patch — the exact analogue of how ``bind_cached_image_features`` patches
``get_image_features`` for the two full images.

Because the same patch is used for generation prefill and full-sequence logprob
recomputation, rollout and logprob see an identical input.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn

from models.qwen35 import unwrap_model, unwrap_qwen_core


REGION_TOKEN = "<|region|>"

REGION_TENSOR_KEYS = (
    "region_test",
    "region_ref",
    "region_geom",
    "region_hstat",
    "region_valid",
)

# Fields the injection consumes; these must never be forwarded to the Qwen model as
# unknown kwargs. ``model_inputs`` in rl/grpo.py reads this set.
REGION_INPUT_KEYS = set(REGION_TENSOR_KEYS) | {"region_token_id"}


def _embedding_consumer(model) -> nn.Module:
    """The module whose ``get_input_embeddings()`` is actually called during forward.

    ``Qwen3_5ForConditionalGeneration.forward`` delegates to ``self.model`` (the inner
    ``Qwen3_5Model``), whose ``forward`` calls ``self.get_input_embeddings()(input_ids)``.
    Patching the outer wrapper (or the base model) therefore has NO effect on the real
    forward — we must patch the inner ``Qwen3_5Model``, exactly the same way
    ``bind_cached_image_features`` patches ``get_image_features`` on ``core.model``.
    """
    core = unwrap_qwen_core(unwrap_model(model))
    inner = getattr(core, "model", None)
    if inner is not None and inner is not core and hasattr(inner, "get_input_embeddings"):
        return inner
    return core


def ensure_region_token(processor, model) -> int:
    """Register ``<|region|>`` and resize the embedding; return its token id.

    Must be called on the base model BEFORE LoRA wrapping (so the SFT adapter and the
    LoRA model share the same vocabulary). Idempotent.
    """
    tok = getattr(processor, "tokenizer", processor)
    if REGION_TOKEN not in tok.get_vocab():
        n = tok.add_special_tokens({"additional_special_tokens": [REGION_TOKEN]})
        if n > 0:
            model.resize_token_embeddings(len(tok))
    return int(tok.convert_tokens_to_ids(REGION_TOKEN))


def region_token_id_of(processor) -> int:
    tok = getattr(processor, "tokenizer", processor)
    return int(tok.convert_tokens_to_ids(REGION_TOKEN))


def has_region(batch: dict) -> bool:
    return all(k in batch for k in REGION_TENSOR_KEYS)


def region_raw_from_batch(batch: dict) -> Dict[str, torch.Tensor]:
    return dict(
        test=batch["region_test"],
        ref=batch["region_ref"],
        geom=batch["region_geom"],
        hstat=batch["region_hstat"],
        valid=batch["region_valid"],
    )


class _RegionScatterEmbedding(nn.Module):
    """Wraps the token embedding so ``<|region|>`` positions receive adapter output."""

    def __init__(self, base: nn.Module, region_token_id: int, region_embeds: torch.Tensor):
        super().__init__()
        self.base = base
        self.region_token_id = int(region_token_id)
        self.region_embeds = region_embeds  # [n_cells, hidden]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.base(input_ids)
        if self.region_embeds is None:
            return embeds
        mask = input_ids == self.region_token_id
        n = int(mask.sum())
        if n > 0:
            flat = self.region_embeds
            if flat.shape[0] != n:
                if n % flat.shape[0] == 0:
                    flat = flat.repeat(n // flat.shape[0], 1)
                else:
                    raise RuntimeError(
                        f"region embeds {flat.shape[0]} cannot tile {n} <|region|> tokens"
                    )
            flat = flat.to(device=embeds.device, dtype=embeds.dtype)
            embeds = embeds.masked_scatter(mask.unsqueeze(-1), flat)
        return embeds


@contextmanager
def bind_region_injection(
    model,
    adapter: Optional[nn.Module],
    region_raw: Optional[Dict[str, torch.Tensor]],
    region_token_id: int,
) -> Iterator[None]:
    """Replace ``<|region|>`` placeholder embeddings with ``adapter(region_raw)``.

    ``adapter`` may be frozen (RL) or trainable (SFT); the scatter preserves gradients
    when the caller does not wrap the enclosing forward in ``torch.no_grad``.
    """
    if adapter is None or region_raw is None or not region_raw:
        yield
        return
    consumer = _embedding_consumer(model)
    # Cast inputs to the adapter's own device/dtype so a CPU (or FP32) adapter cannot
    # silently mismatch the model running on GPU (or BF16).
    ref = next(adapter.parameters(), None)
    if ref is not None:
        device, dtype = ref.device, ref.dtype
        region_raw = {
            k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device))
            for k, v in region_raw.items()
        }
    region_embeds = adapter(region_raw)
    if region_embeds.dim() == 3:
        region_embeds = region_embeds[0]
    orig = consumer.get_input_embeddings
    wrapper = _RegionScatterEmbedding(orig(), int(region_token_id), region_embeds)
    consumer.get_input_embeddings = lambda: wrapper
    try:
        yield
    finally:
        consumer.get_input_embeddings = orig


def save_region_adapter(adapter: Optional[nn.Module], path) -> None:
    """Persist a separately-mounted non-PEFT module (``model.save_pretrained`` skips it)."""
    if adapter is None:
        return
    torch.save(dict(state_dict=adapter.state_dict(), config=adapter.config_dict), str(path))


def load_region_adapter(cls, path, feature_dim: int, hidden_size: int) -> nn.Module:
    """Rebuild a region adapter from ``save_region_adapter`` output.

    ``cls`` is the adapter class (avoids a hard import cycle); config from disk wins,
    falling back to the caller-provided dimensions for legacy checkpoints.
    """
    ckpt = torch.load(str(path), map_location="cpu")
    config = dict(ckpt.get("config") or {})
    config.setdefault("feature_dim", int(feature_dim))
    config.setdefault("hidden_size", int(hidden_size))
    adapter = cls(**config)
    adapter.load_state_dict(ckpt["state_dict"])
    return adapter


def language_hidden_size(model) -> int:
    cfg = getattr(model, "config", None)
    tc = getattr(cfg, "text_config", None)
    if getattr(tc, "hidden_size", None) is not None:
        return int(tc.hidden_size)
    if getattr(cfg, "hidden_size", None) is not None:
        return int(cfg.hidden_size)
    raise ValueError("cannot determine language hidden size")


def attach_region_adapter(model, prior, cfg, sft_dir) -> nn.Module:
    """Load the frozen region adapter from ``sft_dir`` and mount it (device/dtype synced).

    Raised loudly when the weights are missing — silently falling back to a randomly
    initialized adapter would freeze an unaligned region channel and invalidate the
    experiment. The adapter is trained only during region SFT (``train_region_sft.py``).
    """
    from models.region_adapter import RegionAdapter

    if not sft_dir:
        raise ValueError(
            "outcome.sft_adapter is required: the region adapter is trained during "
            "region SFT (train_region_sft.py), not initialized in RL"
        )
    ckpt = Path(sft_dir) / "region_adapter.pt"
    if not ckpt.exists():
        raise ValueError(
            f"region adapter weights are missing: {ckpt}. Run train_region_sft.py first."
        )
    feature_dim = int(prior.visual.config.hidden_size)
    hidden_size = language_hidden_size(model)
    adapter = load_region_adapter(RegionAdapter, ckpt, feature_dim, hidden_size)
    for p in adapter.parameters():
        p.requires_grad = False
    adapter.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    adapter.to(device=device, dtype=dtype)
    model.region_adapter = adapter
    return adapter
