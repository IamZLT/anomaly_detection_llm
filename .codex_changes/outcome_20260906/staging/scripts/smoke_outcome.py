#!/usr/bin/env python3
"""Disposable real-model integration check. Does NOT save trained weights.

Synthetic centered advantages exercise backward even when short random responses
are invalid; this is a mechanics test, not evidence of detection improvement.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from outcome.engine import datasets, load_model, validate_config
from outcome.inputs import OutcomeCollator
from outcome.policy import generate_group, optimize_group
from rl.grpo import forward_with_vision, model_inputs, move_batch
from models.vision_cache import bind_cached_image_features
from utils.common import set_seed
from utils.config import load_yaml_config


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',default='configs/qwen35_08b_outcome.yaml')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    cfg=load_yaml_config(args.config)
    cfg['grpo'].update(group_size=2,max_new_tokens=48,rollout_micro_batch_size=2)
    validate_config(cfg);set_seed(42)
    model,processor,prior=load_model(cfg)
    _,dev,_=datasets(cfg,processor)
    index=next(i for i,s in enumerate(dev.samples) if s['metadata']['anomaly'])
    item=dev[index]
    reports=[]
    for roi in [False,True]:
        cfg['outcome']['roi']['enabled']=roi
        batch=move_batch(OutcomeCollator(processor,prior,cfg)([item]),next(model.parameters()).device)
        model.eval()
        inputs=model_inputs(batch)
        with torch.no_grad():
            raw=forward_with_vision(model,inputs,batch['input_ids'],batch['attention_mask']).logits[:,-1].float()
            with bind_cached_image_features(model,batch['image_embeds']):
                cached=forward_with_vision(model,inputs,batch['input_ids'],batch['attention_mask']).logits[:,-1].float()
        error=(raw-cached).abs().max().item()
        assert error < .03, f'cached/official visual mismatch: {error}'
        del raw,cached
        cs=generate_group(model,processor,batch,cfg,group=2,sample=True)
        before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-6)
        metrics=optimize_group(model,processor,batch,cs,torch.tensor([-1.,1.],device=next(model.parameters()).device),opt,cfg)
        change=max((p.detach()-before[n]).abs().max().item() for n,p in model.named_parameters() if p.requires_grad)
        assert change > 0, 'LoRA weights did not update'
        report=dict(roi_enabled=roi,actual_image_count=len(batch['image_grid_thw']),prompt_tokens=int(batch['prompt_len'][0]),
                    cached_official_max_error=error,weight_max_change=change,metrics=metrics,
                    completion_lengths=[len(c.ids)-int(batch['prompt_len'][0]) for c in cs],
                    stop_reasons=[c.stop_reason for c in cs],roi=batch['_meta'][0]['roi'])
        reports.append(report)
        Path(args.output).write_text(json.dumps(reports,indent=2))
        print(json.dumps(report),flush=True)
        del batch,before,opt
        if torch.cuda.is_available():torch.cuda.empty_cache()
    print('PASS: two-image and three-image cached generation + logprobs + backward; no weights saved',flush=True)

if __name__=='__main__':main()
