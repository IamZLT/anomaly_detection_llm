"""Finite rollouts, genuine completion lengths and one outcome advantage."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from transformers import GenerationConfig, StoppingCriteriaList, StopStringCriteria

from models.qwen35 import force_vision_eval, unwrap_model
from models.vision_cache import bind_cached_image_features
from models.region_injection import bind_region_injection, has_region, region_raw_from_batch
from rl.grpo import (clipped_pg_kl, disable_adapter_ctx, dropout_eval, expand_gen_in_for_group,
                     forward_with_vision, micro_batch_ranges, model_inputs, padded_completion_tensors,
                     token_logprobs, token_logprobs_nograd)


@dataclass
class Completion:
    ids: torch.Tensor
    text: str
    stop_reason: str


def eos_ids(model, tokenizer):
    value = getattr(getattr(unwrap_model(model), 'generation_config', None), 'eos_token_id', None)
    ids = set(value if isinstance(value, (list, tuple)) else [value] if value is not None else [])
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    return sorted(ids)


def trim_completion(row, prompt_len, tokenizer, end_ids):
    """Keep first real EOS, or the token completing </answer>; never train batch padding.

    Inspect only completion tokens, because prompt contains the output template.
    A stop string can span tokens or end within a token; include that whole token.
    """
    tokens = row[prompt_len:].detach().cpu().tolist()
    prefix = []
    reason = 'length'
    for token in tokens:
        prefix.append(token)
        if '</answer>' in tokenizer.decode(prefix, skip_special_tokens=True):
            reason = 'answer'
            break
        if token in end_ids:
            reason = 'eos'
            break
    end = prompt_len+len(prefix)
    ids = row[:end].detach().clone()
    return Completion(ids, tokenizer.decode(ids[prompt_len:], skip_special_tokens=True), reason)


def group_advantages(rewards, scale=False, eps=1e-6):
    if bool((rewards.max() == rewards.min()).item()):
        return torch.zeros_like(rewards)
    adv = rewards-rewards.mean()
    if scale:
        adv = adv/rewards.std(unbiased=False).clamp(min=eps)
    return adv


def _region_bind(model, batch):
    """Region-token injection context for one forward; no-op without a mounted adapter.

    Returns a FRESH context manager on every call so it can be reused across the
    multiple ``with`` blocks inside ``optimize_group`` (a generator-based context
    manager cannot be re-entered).
    """
    adapter = getattr(unwrap_model(model), 'region_adapter', None)
    if adapter is None or not has_region(batch):
        return nullcontext()
    return bind_region_injection(model, adapter, region_raw_from_batch(batch), int(batch['region_token_id']))


def generate_group(model, processor, batch, cfg, group=1, sample=False):
    tokenizer = getattr(processor, 'tokenizer', processor)
    gcfg = cfg['grpo']
    limit = int(gcfg['max_new_tokens'])
    ends = eos_ids(model, tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else ends[0]
    # A fresh config removes inherited top-k/penalties/forced tokens. Raw-policy
    # categorical sampling is exactly the distribution used by all logprobs.
    generation = GenerationConfig(max_new_tokens=limit, do_sample=sample,
        temperature=1.0, top_p=1.0, top_k=0, typical_p=1.0, repetition_penalty=1.0,
        eos_token_id=ends or None, pad_token_id=pad, use_cache=True, num_beams=1)
    stops = StoppingCriteriaList([StopStringCriteria(tokenizer=tokenizer, stop_strings=['</answer>'])])
    core = unwrap_model(model)
    was_training = core.training
    core.eval()
    result = []
    try:
        with torch.no_grad(), bind_cached_image_features(core, batch['image_embeds']), _region_bind(model, batch):
            for start, end in micro_batch_ranges(group, int(gcfg.get('rollout_micro_batch_size', 1))):
                generation.num_return_sequences = end-start
                generated = core.generate(**model_inputs(batch), generation_config=generation,
                                          stopping_criteria=stops)
                for row in generated:
                    result.append(trim_completion(row, int(batch['prompt_len'][0]), tokenizer, ends))
    finally:
        core.train(was_training)
        force_vision_eval(core)
    return result


def optimize_group(model, processor, batch, completions, advantages, optimizer, cfg, skip=False):
    """On-policy group with multi-epoch PPO updates (clip engages after epoch 0).

    old/ref logprobs are computed once against the pre-update policy; each
    ``policy_epoch`` then recomputes ``new_lp`` against the (now-updated) policy
    so the importance ratio leaves 1.0 and the PPO clip becomes active. With
    ``policy_epochs == 1`` the ratio is pinned to 1.0 and the clip never fires,
    silently degrading to unclipped REINFORCE, so it is treated as a hard error
    unless explicitly allowed via ``allow_single_epoch``.

    ``skip=True`` runs the actor forward with a zero loss so every DDP rank still
    performs a ``backward()`` (keeping gradient all-reduce in lockstep) while
    contributing no update. This is used when a rank's group advantage collapsed
    to zero but another rank still has signal.
    """
    gc = cfg['grpo']
    policy_epochs = max(1, int(gc.get('policy_epochs', 1)))
    if policy_epochs == 1 and not bool(gc.get('allow_single_epoch', False)):
        raise ValueError('policy_epochs=1 pins ratio to 1.0 (clip never fires); '
                         'use policy_epochs>=2 or set allow_single_epoch=true')
    tokenizer = getattr(processor, 'tokenizer', processor)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device
    prompt_len = int(batch['prompt_len'][0])
    seqs = [c.ids for c in completions]
    outputs, attn, labels = padded_completion_tensors(seqs, prompt_len, pad, device)
    gen_in = model_inputs(batch)
    cache = batch['image_embeds']
    group = len(completions)
    if skip:
        old_lp = ref_lp = None
    else:
        old_parts, ref_parts = [], []
        for start, end in micro_batch_ranges(group, int(gc.get('logprob_micro_batch_size', 1))):
            chunk = expand_gen_in_for_group(gen_in, end-start)
            with bind_cached_image_features(model, cache), _region_bind(model, batch):
                old, _ = token_logprobs_nograd(model, chunk, outputs[start:end], attn[start:end], labels[start:end])
                if float(gc['kl_beta']) > 0:
                    with disable_adapter_ctx(model):
                        ref, _ = token_logprobs_nograd(model, chunk, outputs[start:end], attn[start:end], labels[start:end])
                else:
                    ref = old
            old_parts.append(old)
            ref_parts.append(ref)
        old_lp, ref_lp = torch.cat(old_parts), torch.cat(ref_parts)
    totals = dict(loss=0., pg=0., kl=0., ratio=0., clip_fraction=0., logprob_max_error=0.)
    model.train()
    force_vision_eval(model)
    last_grad_norm = 0.0
    err_tol = float(gc.get('logprob_error_tolerance', .1))
    for pe in range(policy_epochs):
        optimizer.zero_grad(set_to_none=True)
        pe_totals = dict(loss=0., pg=0., kl=0., ratio=0., clip_fraction=0., logprob_max_error=0.)
        for start, end in micro_batch_ranges(group, int(gc.get('actor_micro_batch_size', 1))):
            chunk = expand_gen_in_for_group(gen_in, end-start)
            with dropout_eval(model), bind_cached_image_features(model, cache), _region_bind(model, batch):
                out = forward_with_vision(model, chunk, outputs[start:end], attn[start:end])
                if skip:
                    # Keep the forward+backward so DDP gradient all-reduce stays in
                    # lockstep with the other ranks, but contribute a zero update.
                    loss = out.logits.float().sum() * 0.0
                    pg = kl = ratio = clipped = torch.zeros((), device=device)
                else:
                    new_lp, mask = token_logprobs(out.logits, labels[start:end])
                    error = ((new_lp.detach()-old_lp[start:end]).abs()*mask).max().item()
                    pe_totals['logprob_max_error'] = max(pe_totals['logprob_max_error'], error)
                    # On epoch 0 old_lp and new_lp come from the same policy, so any
                    # drift is a bf16/checkpointing/dropout bug. After an update the
                    # ratio is supposed to leave 1.0, so only check finiteness.
                    if not torch.isfinite(new_lp).all() or (pe == 0 and error > err_tol):
                        optimizer.zero_grad(set_to_none=True)
                        raise RuntimeError(f'behavior/new logprob mismatch before update: max={error}')
                    loss, pg, kl, ratio, clipped = clipped_pg_kl(new_lp, mask, old_lp[start:end], ref_lp[start:end],
                        advantages[start:end, None], float(gc['clip_low']), float(gc['clip_high']), float(gc['kl_beta']))
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite policy loss')
                weight = (end-start)/group
                (loss*weight).backward()
                for key, value in zip(('loss','pg','kl','ratio','clip_fraction'), (loss,pg,kl,ratio,clipped)):
                    pe_totals[key] += float(value.detach())*weight
                del out, loss
                if not skip:
                    del new_lp
        params = [p for p in model.parameters() if p.requires_grad]
        last_grad_norm = float(torch.nn.utils.clip_grad_norm_(params, float(gc['max_grad_norm']), error_if_nonfinite=True))
        optimizer.step()
        for key in ('loss', 'pg', 'kl', 'ratio', 'clip_fraction', 'logprob_max_error'):
            totals[key] += pe_totals[key]
    for key in ('loss', 'pg', 'kl', 'ratio', 'clip_fraction', 'logprob_max_error'):
        totals[key] /= policy_epochs
    totals['grad_norm'] = last_grad_norm
    totals['effective_tokens'] = sum(len(c.ids)-prompt_len for c in completions)
    return totals
