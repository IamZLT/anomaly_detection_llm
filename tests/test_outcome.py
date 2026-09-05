"""Regression tests for outcome reward, geometry, masks and metric denominators."""
import copy
import json

import numpy as np
import pytest
import torch
from PIL import Image

from outcome.protocol import parse_output, score_output, prompt, validate_gt
from outcome.inputs import region_proposals, crop_original, local_box_to_full
from outcome.policy import group_advantages, trim_completion
from outcome.engine import summarize, validate_config
from rl.grpo import padded_completion_tensors, token_logprobs


def answer(anomaly=True, box=None):
    return '<answer>'+json.dumps(dict(is_anomaly=anomaly,bbox_2d=box if box is not None else [100,200,300,400] if anomaly else None,description='x'))+'</answer>'


def full(anomaly=True):
    return ('<understand>x</understand><compare>y</compare>'
            '<ground>{"candidate_bbox_2d":null}</ground>'
            '<verify>{"action":"discover","evidence":"z"}</verify>'+answer(anomaly))


def meta(anomaly=True):
    return dict(is_anomaly=anomaly, orig_size=[100,100],gt_box_px=[10,20,30,40] if anomaly else None,image_path='fixture')


def test_exact_bbox_gets_full_task_reward_without_rationale():
    p = parse_output(answer())
    assert p['task_valid'] and not p['protocol_valid']
    assert score_output(p, meta()) == dict(task=1.,protocol=0.,total=1.,iou=1.,correct=True)


def test_one_character_prose_has_no_min_length_gate():
    p = parse_output(full())
    assert p['task_valid'] and p['protocol_valid']
    assert score_output(p,meta())['total'] == 1.05


@pytest.mark.parametrize('text', [answer()+answer(), answer()+'garbage', answer().replace('</answer>',''),
    '<answer>{"is_anomaly":true,"is_anomaly":false,"bbox_2d":null}</answer>',
    '<answer>{"is_anomaly":1,"bbox_2d":null}</answer>',
    '<answer>{"is_anomaly":true,"bbox_2d":[0,0,NaN,20]}</answer>',
    '<answer>{"is_anomaly":true,"bbox_2d":[false,0,20,20]}</answer>',
    answer(box=[-1,0,20,20]), answer(box=[0,0,1001,20]), answer(box=[20,0,20,20]),
    '<answer>{"is_anomaly":false}</answer>'])
def test_ambiguous_truncated_or_invalid_final_is_failure(text):
    p = parse_output(text)
    assert not p['task_valid']
    assert score_output(p,meta())['task'] == -1


def test_duplicate_prose_only_affects_protocol():
    p = parse_output('<understand>extra</understand>'+full())
    assert p['task_valid'] and not p['protocol_valid']
    assert score_output(p,meta())['task'] == 1


def test_normal_rejection_and_misclassification():
    assert score_output(parse_output(full(False)),meta(False))['task'] == 1
    assert score_output(parse_output(full(False)),meta(True))['task'] == -1
    assert score_output(parse_output(full(True)),meta(False))['task'] == -1


def test_no_dense_reward_for_displaced_tiny_box():
    m = dict(is_anomaly=True,orig_size=[1000,1000],gt_box_px=[100,100,110,110])
    assert score_output(parse_output(answer(box=[111,100,121,110])),m)['task'] == 0
    assert score_output(parse_output(answer(box=[100,100,110,110])),m)['task'] == 1


def test_gt_missing_fails_loudly():
    with pytest.raises(ValueError):
        validate_gt(dict(is_anomaly=True,orig_size=[100,100],gt_box_px=None))


def test_single_patch_and_thin_component_use_cell_edges():
    h = np.zeros((4,4));h[3,3]=1
    ps,_ = region_proposals(h,{})
    assert len(ps)==1 and ps[0]['bbox_2d']==[750,750,1000,1000]
    h[0:4,3]=1
    ps,_ = region_proposals(h,{})
    assert ps[0]['bbox_2d']==[750,0,1000,1000]


def test_flat_low_h_can_be_empty_without_forced_points():
    assert region_proposals(np.zeros((4,4)),{})[0] == []
    assert region_proposals(np.full((4,4),.01),{'raw_threshold':.2})[0] == []
    assert len(region_proposals(np.full((4,4),.5),{'raw_threshold':.2})[0]) == 1


def test_max_candidates_is_cap_not_target():
    h=np.zeros((5,5));h[0,0]=1;h[4,4]=.9
    assert len(region_proposals(h,{'max_candidates':3})[0])==2
    assert len(region_proposals(h,{'max_candidates':1})[0])==1


def test_crop_is_from_original_and_round_trips_boundaries():
    im=Image.new('RGB',(2000,1000));im.putpixel((400,200),(255,0,0))
    roi,bounds=crop_original(im,[200,200,300,300],0)
    assert roi.size==(200,100) and roi.getpixel((0,0))==(255,0,0)
    assert bounds==[400,200,600,300]
    assert local_box_to_full([0,0,1000,1000],bounds,im.size)==[200,200,300,300]
    _,bounds=crop_original(im,[0,0,100,100],.25)
    assert bounds==[0,0,250,125]


class Tokenizer:
    def decode(self,tokens,skip_special_tokens=True):
        return ''.join({1:'x',2:'</ans',3:'wer>',4:'extra',9:''}[int(t)] for t in tokens)


def test_stop_string_spanning_tokens_and_batch_padding():
    # Prefix token 4 is prompt and must never be part of completion loss.
    rows=[torch.tensor([4,1,2,3,9,9]),torch.tensor([4,1,1,1,9,9])]
    trimmed=[trim_completion(row,1,Tokenizer(),[9]) for row in rows]
    assert trimmed[0].ids.tolist()==[4,1,2,3]
    assert trimmed[1].ids.tolist()==[4,1,1,1,9]
    assert [c.stop_reason for c in trimmed]==['answer','eos']
    ids,attn,labels=padded_completion_tensors([c.ids for c in trimmed],1,9,torch.device('cpu'))
    assert labels[0].tolist()==[-100,1,2,3,-100]
    assert labels[1].tolist()==[-100,1,1,1,9]  # keep real EOS even if pad==EOS
    assert attn[0].tolist()==[1,1,1,1,0]
    logits=torch.randn(2,5,10,requires_grad=True)
    lp,mask=token_logprobs(logits,labels)
    assert mask.sum(dim=1).tolist()==[3,4]
    lp.sum().backward()
    assert torch.count_nonzero(logits.grad[0,3:])==0


def test_no_termination_at_budget_is_truncated():
    assert trim_completion(torch.tensor([4,1,1]),1,Tokenizer(),[9]).stop_reason=='length'


def test_centered_advantage_does_not_amplify_format_noise():
    reward=torch.tensor([1.,1.0001])
    assert group_advantages(reward).abs().max() < .0001
    assert torch.equal(group_advantages(torch.ones(8)),torch.zeros(8))
    assert group_advantages(torch.tensor([-1.,1.])).tolist()==[-1.,1.]


def test_metrics_count_invalid_normal_separately_from_true_negative():
    common=dict(task_valid=True,protocol_valid=False,iou=0.,class_name='a',size_bin='normal',
                candidate_to_final_delta=None,new_tokens=5,seconds=1,stop_reason='eos')
    rows=[dict(common,is_anomaly=False,pred=None,task_valid=False),
          dict(common,is_anomaly=False,pred=True),
          dict(common,is_anomaly=True,pred=True,iou=.8,size_bin='small')]
    s=summarize(rows)
    assert s['normal_fpr']==.5 and s['normal_correct_rate']==0
    assert s['anomaly_recall']==1 and s['balanced_accuracy']==.5
    assert s['anomaly_gated_miou']==.8 and s['invalid_decision_rate']==1/3
    assert s['miou_large'] is None


def test_prompt_allows_rejecting_or_searching_outside_h():
    p=prompt('bottle',True)
    assert 'search outside H' in p and 'FULL IMAGE' in p and 'ORIGINAL inspection' in p


def test_all_zero_groups_are_logged_and_consume_finite_budget(tmp_path, monkeypatch):
    import outcome.engine as engine
    from outcome.policy import Completion
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
        def save_pretrained(self,path):
            pass
    class Processor:
        def save_pretrained(self,path):
            pass
    class Dataset:
        samples=[{'image':'normal-a'},{'image':'normal-b'}]
        def __len__(self): return 2
        def __getitem__(self,i): return i
    class Empty:
        samples=[]
        def __len__(self): return 0
    m=meta(False)
    m.update(ref_path='ref',class_name='a',image_path='normal-a',prior_candidates=[],
             prompt_tokens=1,visual_tokens=0,prior_hint_tokens=0)
    batch={'input_ids':torch.tensor([[4]]),'prompt_len':torch.tensor([1]),'_meta':[m]}
    monkeypatch.setattr(engine,'OutcomeCollator',lambda *args:lambda items:batch)
    monkeypatch.setattr(engine,'generate_group',lambda *args,**kw:[Completion(torch.tensor([4,1]),'invalid','length')]*2)
    monkeypatch.setattr(engine,'optimize_group',lambda *args:pytest.fail('zero group must not update'))
    cfg={'outcome':{'protocol_weight':.05,'eval_before_train':False,'final_test':False},
         'grpo':{'max_attempts':3,'group_size':2,'learning_rate':1e-6,'save_steps':0},
         'training':{'seed':42,'eval_every_n_steps':0}}
    engine.run_train(cfg,Model(),Processor(),None,Dataset(),Empty(),Empty(),tmp_path)
    summary=json.loads((tmp_path/'training_summary.json').read_text())
    assert summary=={'attempts':3,'updates':0,'skipped':3}
    assert len((tmp_path/'rollouts.jsonl').read_text().splitlines())==3


def test_evaluation_none_is_full_split(tmp_path,monkeypatch):
    import outcome.engine as engine
    from outcome.policy import Completion
    class Dataset:
        def __len__(self):return 5
        def __getitem__(self,i):return i
    m=meta(False);m.update(ref_path='r',class_name='a')
    batch={'input_ids':torch.tensor([[4]]),'prompt_len':torch.tensor([1]),'_meta':[m]}
    monkeypatch.setattr(engine,'OutcomeCollator',lambda *args:lambda items:batch)
    monkeypatch.setattr(engine,'generate_group',lambda *args,**kwargs:[Completion(torch.tensor([4,1]),answer(False),'answer')])
    cfg={'outcome':{'protocol_weight':.05}}
    model=torch.nn.Linear(1,1)
    assert engine.evaluate(cfg,model,None,None,Dataset(),tmp_path/'full.json',None)['n']==5
    assert engine.evaluate(cfg,model,None,None,Dataset(),tmp_path/'subset.json',2)['n']==2


def test_canonical_pair_uses_one_official_forward_and_removes_hooks():
    from types import SimpleNamespace
    from outcome.inputs import encode_pair_canonical
    class Scale(torch.nn.Module):
        def __init__(self,scale):super().__init__();self.scale=scale
        def forward(self,x):return x*self.scale
    class Visual(torch.nn.Module):
        dtype=torch.float32
        def __init__(self):
            super().__init__();self.blocks=torch.nn.ModuleList([Scale(1),Scale(2)]);self.calls=0
        def forward(self,x,grid_thw):
            self.calls+=1
            for block in self.blocks:x=block(x)
            return SimpleNamespace(pooler_output=x)
    visual=Visual()
    prior=SimpleNamespace(visual=visual,block_indices=[0,1],spatial_merge_size=1,
                          temperature=.5,neighborhood_radius=0,
                          _nn_map=lambda ft,fr,ht,hr,radius:(ft-fr).abs().sum(-1).view(ht))
    pixels=torch.arange(16,dtype=torch.float32).reshape(4,4)
    encoded=encode_pair_canonical(prior,pixels,torch.tensor([[1,1,2],[1,1,2]]))
    assert visual.calls==1
    assert torch.equal(encoded['merged_embeddings'],pixels*2)
    assert torch.allclose(encoded['patch_map'],torch.full((1,2),64.))
    assert all(not block._forward_hooks for block in visual.blocks)


def test_equal_noninteger_rewards_do_not_create_roundoff_advantage():
    assert torch.equal(group_advantages(torch.full((3,),1.05)),torch.zeros(3))
