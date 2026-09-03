"""Process-aware spatial GRPO: log π, ρ, clip, KL, advantages."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from reasoning.segments import completion_segment_ids, mix_segment_advantage


def unwrap_model(m):
    return m.module if hasattr(m, "module") else m


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if k.startswith("_"):
            out[k] = v
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def model_inputs(batch: dict) -> dict:
    skip = {"labels", "_meta", "prompt_len", "image_embeds"}
    return {k: v for k, v in batch.items() if k not in skip and torch.is_tensor(v)}


def expand_gen_in_for_group(gen_in: dict, group: int) -> dict:
    """Repeat prompt-side multimodal metadata so one forward covers the whole GRPO group."""
    group = max(int(group), 1)
    out = {}
    for k, v in gen_in.items():
        if not torch.is_tensor(v):
            continue
        if k in ("pixel_values", "pixel_values_videos"):
            out[k] = v
            continue
        if k in ("image_grid_thw", "video_grid_thw"):
            g = v if v.ndim == 2 else v.reshape(-1, int(v.shape[-1]))
            out[k] = g.repeat(group, 1) if group > 1 else g
            continue
        t = v.unsqueeze(0) if v.ndim == 1 else v
        if group > 1 and t.shape[0] == 1:
            t = t.repeat(group, *([1] * (t.ndim - 1)))
        out[k] = t
    return out


def forward_with_vision(model, gen_in: dict, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Keep processor vision tensors; pad mm_token_type_ids to generated length."""
    n = int(input_ids.shape[0])
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    for k, v in gen_in.items():
        if k in ("input_ids", "attention_mask", "labels") or not torch.is_tensor(v):
            continue
        if k in ("mm_token_type_ids",):
            if v.ndim == 1:
                v = v.unsqueeze(0)
            if v.shape[0] == 1 and n > 1:
                v = v.repeat(n, *([1] * (v.ndim - 1)))
            seq = int(input_ids.shape[-1])
            cur = int(v.shape[-1])
            if cur < seq:
                v = F.pad(v, (0, seq - cur), value=0)
            elif cur > seq:
                v = v[..., :seq]
        kwargs[k] = v
    return model(**kwargs)


def token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = labels[:, 1:]
    gather_idx = tgt.clamp(min=0).unsqueeze(-1)
    token_lp = logp.gather(-1, gather_idx).squeeze(-1)
    mask = tgt.ne(-100)
    return token_lp * mask, mask


def disable_adapter_ctx(model):
    m = unwrap_model(model)
    fn = getattr(m, "disable_adapter", None)
    if fn is None:
        raise RuntimeError("disable_adapter() unavailable; reference policy is invalid.")
    ctx = fn()
    if not hasattr(ctx, "__enter__"):
        raise RuntimeError("disable_adapter() did not return a context manager; reference policy is invalid.")
    return ctx


@contextmanager
def dropout_eval(model):
    """Disable Dropout noise while keeping module.training=True for HF checkpointing."""
    mods = [mod for mod in unwrap_model(model).modules() if isinstance(mod, nn.Dropout)]
    states = [mod.training for mod in mods]
    for mod in mods:
        mod.eval()
    try:
        yield
    finally:
        for mod, st in zip(mods, states):
            mod.train(st)


def token_logprobs_nograd(model, gen_in: dict, outputs: torch.Tensor, attn: torch.Tensor, labels: torch.Tensor):
    m = unwrap_model(model)
    was_training = m.training
    m.eval()
    try:
        with torch.no_grad():
            out = forward_with_vision(m, gen_in, outputs, attn)
            lp, mask = token_logprobs(out.logits, labels)
    finally:
        if was_training:
            m.train()
    return lp, mask


def clipped_pg_kl(
    new_lp: torch.Tensor,
    mask: torch.Tensor,
    old_lp: torch.Tensor,
    ref_lp: torch.Tensor,
    adv: torch.Tensor,
    clip_low: float,
    clip_high: float,
    kl_beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-trajectory mean, then mean over the group (do not pool tokens across sequences)."""
    rho = torch.exp((new_lp - old_lp).clamp(-20.0, 20.0))
    clipped = rho.clamp(1.0 - float(clip_low), 1.0 + float(clip_high))
    surr = torch.minimum(rho * adv, clipped * adv)
    denom = mask.sum(dim=-1).clamp(min=1)
    pg_per_seq = -(surr * mask).sum(dim=-1) / denom
    L_pg = pg_per_seq.mean()
    delta = (ref_lp - new_lp).clamp(-20.0, 20.0)
    kl_token = torch.exp(delta) - delta - 1.0
    kl_per_seq = (kl_token * mask).sum(dim=-1) / denom
    L_kl = kl_per_seq.mean()
    clip_per_seq = (((rho < (1.0 - clip_low)) | (rho > (1.0 + clip_high))).float() * mask).sum(dim=-1) / denom
    rho_per_seq = (rho * mask).sum(dim=-1) / denom
    return (
        L_pg + float(kl_beta) * L_kl,
        L_pg,
        L_kl,
        rho_per_seq.mean().detach(),
        clip_per_seq.mean().detach(),
    )


def grpo_advantages(rewards: torch.Tensor, group: int, eps: float) -> torch.Tensor:
    r = rewards.view(-1, group)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True, unbiased=False).clamp(min=eps)
    adv = (r - mean) / std
    return adv.reshape(-1)


def avg_across_ranks(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return float(value)
    t = torch.tensor([float(value)], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return float(t.item())


def group_std(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    t = torch.tensor(vals, dtype=torch.float32)
    return float(t.std(unbiased=False))


def padded_completion_tensors(
    seqs: List[torch.Tensor],
    prompt_len: int,
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad by length, not by token identity, so pad_id==eos_id does not mask real EOS."""
    max_t = max(int(s.numel()) for s in seqs)
    group = len(seqs)
    outputs = torch.full((group, max_t), int(pad_id), device=device, dtype=torch.long)
    attn = torch.zeros((group, max_t), device=device, dtype=torch.long)
    labels = torch.full((group, max_t), -100, device=device, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = int(s.numel())
        outputs[i, :n] = s.to(device)
        attn[i, :n] = 1
        if n > int(prompt_len):
            labels[i, int(prompt_len) : n] = s[int(prompt_len) :].to(device)
    return outputs, attn, labels


def build_segment_advantages(
    tokenizer,
    seqs: List[torch.Tensor],
    prompt_len: int,
    max_t: int,
    a_ground: torch.Tensor,
    a_reason: torch.Tensor,
    a_final: torch.Tensor,
    is_anomaly: bool,
    device: torch.device,
) -> torch.Tensor:
    adv = torch.zeros(len(seqs), max(max_t - 1, 1), device=device, dtype=torch.float32)
    for i, s in enumerate(seqs):
        comp = s[prompt_len:]
        segs = completion_segment_ids(tokenizer, comp)
        ag, ar, af = float(a_ground[i]), float(a_reason[i]), float(a_final[i])
        for k, seg in enumerate(segs):
            j = prompt_len - 1 + k
            if 0 <= j < adv.shape[1]:
                adv[i, j] = mix_segment_advantage(seg, ag, ar, af, is_anomaly)
    return adv


def grpo_param_map(gcfg: dict, *, lr: float, accum: int, group: int, temperature: float, top_p: float, max_new: int) -> Dict[str, float]:
    rew = gcfg.get("reward") or {}
    return {
        "lr": float(lr),
        "group_size": float(group),
        "policy_epochs": float(gcfg.get("policy_epochs", 3)),
        "accum": float(accum),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": float(max_new),
        "max_grad_norm": float(gcfg.get("max_grad_norm", 1.0)),
        "adv_eps": float(gcfg.get("adv_eps", 1e-6)),
        "clip_low": float(gcfg.get("clip_low", 0.20)),
        "clip_high": float(gcfg.get("clip_high", 0.28)),
        "kl_beta": float(gcfg.get("kl_beta", 1.0e-4)),
        "min_reward_std": float(gcfg.get("min_reward_std", 0.02)),
        "max_resample_attempts": float(gcfg.get("max_resample_attempts", 2)),
        "epochs": float(gcfg.get("epochs", 2)),
        "w_cov": float(rew.get("w_cov", 0.6)),
        "w_cand_iou": float(rew.get("w_cand_iou", 0.4)),
        "w_iou": float(rew.get("w_iou", 0.6)),
        "w_edge": float(rew.get("w_edge", 0.4)),
        "edge_beta": float(rew.get("edge_beta", 8.0)),
        "keep_tol_norm1000": float(rew.get("keep_tol_norm1000", 8.0)),
        "normal_correct": float(rew.get("normal_correct", 1.0)),
        "wrong_decision": float(rew.get("wrong_decision", -1.0)),
        "invalid_output": float(rew.get("invalid_output", -1.0)),
        "edge_min_frac": float(rew.get("edge_min_frac", 0.05)),
    }


def resolve_max_steps(gcfg: dict, n_train: int, world: int) -> int:
    import math

    raw = gcfg.get("max_steps")
    epochs = float(gcfg.get("epochs", 2))
    if raw in (None, "null", "None", ""):
        return max(1, int(math.ceil(float(n_train) * epochs / float(max(world, 1)))))
    return max(1, int(raw))
