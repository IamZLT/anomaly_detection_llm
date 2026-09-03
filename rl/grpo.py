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
    skip = {"labels", "_meta", "prompt_len"}
    return {k: v for k, v in batch.items() if k not in skip and torch.is_tensor(v)}


def forward_with_vision(model, gen_in: dict, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Keep processor vision tensors; pad mm_token_type_ids to generated length."""
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    for k, v in gen_in.items():
        if k in ("input_ids", "attention_mask", "labels") or not torch.is_tensor(v):
            continue
        if k in ("mm_token_type_ids",) and v.shape[-1] != input_ids.shape[-1]:
            seq = int(input_ids.shape[-1])
            cur = int(v.shape[-1])
            if v.ndim == 1:
                v = v.unsqueeze(0)
            if v.shape[0] != input_ids.shape[0]:
                v = v.expand(input_ids.shape[0], *v.shape[1:])
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
    lps = []
    masks = []
    m = unwrap_model(model)
    was_training = m.training
    m.eval()
    try:
        with torch.no_grad():
            for i in range(outputs.shape[0]):
                out = forward_with_vision(m, gen_in, outputs[i : i + 1], attn[i : i + 1])
                lp, mask = token_logprobs(out.logits, labels[i : i + 1])
                lps.append(lp)
                masks.append(mask)
    finally:
        if was_training:
            m.train()
    return torch.cat(lps, dim=0), torch.cat(masks, dim=0)


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
    rho = torch.exp((new_lp - old_lp).clamp(-20.0, 20.0))
    clipped = rho.clamp(1.0 - float(clip_low), 1.0 + float(clip_high))
    surr = torch.minimum(rho * adv, clipped * adv)
    denom = mask.sum().clamp(min=1)
    L_pg = -(surr * mask).sum() / denom
    delta = (ref_lp - new_lp).clamp(-20.0, 20.0)
    kl_token = torch.exp(delta) - delta - 1.0
    L_kl = (kl_token * mask).sum() / denom
    clip_frac = (((rho < (1.0 - clip_low)) | (rho > (1.0 + clip_high))).float() * mask).sum() / denom
    rho_mean = (rho * mask).sum() / denom
    return L_pg + float(kl_beta) * L_kl, L_pg, L_kl, rho_mean.detach(), clip_frac.detach()


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
        "invalid_output": float(rew.get("invalid_output", -0.5)),
    }


def resolve_max_steps(gcfg: dict, n_train: int, world: int) -> int:
    import math

    raw = gcfg.get("max_steps")
    epochs = float(gcfg.get("epochs", 2))
    if raw in (None, "null", "None", ""):
        return max(1, int(math.ceil(float(n_train) * epochs / float(max(world, 1)))))
    return max(1, int(raw))
