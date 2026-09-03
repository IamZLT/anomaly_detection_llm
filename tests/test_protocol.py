"""Parser, reward, and GRPO protocol checks (no GPU)."""

from __future__ import annotations

import torch

from evaluation.metrics import defect_size_bin, gt_relative_area, summarize_detection_metrics
from reasoning.parser import parse_answer_block, parse_cot_output, parse_bbox_field
from reasoning.rewards import (
    boundary_action_consistency,
    compute_rewards,
    edge_precision_reward,
)
from rl.grpo import padded_completion_tensors
from utils.common import qwen_smart_hw

ANOM = """
<compare>
The inspection image contains an irregular structural break on the bottle body that is absent from the normal reference.
</compare>
<ground>
The most suspicious region is concentrated around the damaged bottle body.
candidate_bbox_2d=[380,220,760,810]
</ground>
<verify>
The irregular structure cannot be explained by illumination or normal appearance and is consistent with a physical defect.
</verify>
<boundary>
left=inward
right=keep
top=outward
bottom=inward
</boundary>
<answer>
{"is_anomaly": true, "bbox_2d": [410,190,750,760]}
</answer>
"""

NORM = """
<compare>
The inspection sample is consistent with the reference in overall shape, texture, and surface appearance.
</compare>
<ground>
No localized difference provides sufficient evidence of a defect.
candidate_bbox_2d=null
</ground>
<verify>
The observed variations are consistent with normal appearance.
</verify>
<boundary>
not_applicable
</boundary>
<answer>
{"is_anomaly": false, "bbox_2d": null}
</answer>
"""


def test_qwen_smart_hw_is_multiple_of_32():
    h, w = qwen_smart_hw(500, 333, factor=32, min_pixels=32 * 32, max_pixels=448 * 448)
    assert h % 32 == 0 and w % 32 == 0
    assert h * w <= 448 * 448


def test_strict_answer_rejects_embedded_json():
    parsed = parse_answer_block('prefix {"is_anomaly": false, "bbox_2d": null} suffix')
    assert parsed["answer_state"] == "invalid"
    assert parsed["is_anomaly"] is None


def test_bbox_states():
    assert parse_bbox_field("no field", "candidate_bbox_2d")[0] == "missing"
    assert parse_bbox_field("candidate_bbox_2d=null", "candidate_bbox_2d")[0] == "null"
    assert parse_bbox_field("candidate_bbox_2d=[abc]", "candidate_bbox_2d")[0] == "invalid"
    st, box = parse_bbox_field("candidate_bbox_2d=[10,20,30,40]", "candidate_bbox_2d")
    assert st == "box" and box == [10.0, 20.0, 30.0, 40.0]


def test_trajectory_and_prose_gate():
    p = parse_cot_output(ANOM)
    assert p["trajectory_valid"] and p["prose_ok"]
    junk = ANOM.replace(
        "The inspection image contains an irregular structural break on the bottle body that is absent from the normal reference.",
        "anomaly anomaly anomaly",
    )
    bad = parse_cot_output(junk)
    assert not bad["trajectory_valid"]
    copied = ANOM.replace(
        "The inspection image contains an irregular structural break on the bottle body that is absent from the normal reference.",
        "Return exactly five XML blocks.",
    )
    assert not parse_cot_output(copied)["trajectory_valid"]


def test_invalid_not_cheaper_than_wrong():
    cfg = {"grpo": {"reward": {"invalid_output": -1.0, "wrong_decision": -1.0}}}
    orig = (1000, 1000)
    gt = [410, 190, 750, 760]
    broken = ANOM.replace('{"is_anomaly": true, "bbox_2d": [410,190,750,760]}', "not json")
    rb = compute_rewards(parse_cot_output(broken), gt, orig, True, cfg)
    rw = compute_rewards(parse_cot_output(NORM), gt, orig, True, cfg)
    assert rb["R_final"] == -1.0
    assert rw["R_final"] == -1.0
    assert rb["R_ground"] > 0.0
    rn = compute_rewards(parse_cot_output(NORM), None, orig, False, cfg)
    assert rn["R_final"] == 1.0


def test_scale_aware_edge():
    orig = (1000, 1000)
    gt = [100.0, 100.0, 130.0, 130.0]
    pred = [120.0, 100.0, 130.0, 130.0]
    r_old_scale = edge_precision_reward(pred, gt, orig, beta=8.0, min_frac=1.0)
    r_gt_scale = edge_precision_reward(pred, gt, orig, beta=8.0, min_frac=0.05)
    assert r_gt_scale < r_old_scale


def test_action_consistency_and_delta_iou():
    cfg = {"grpo": {"reward": {}}}
    orig = (1000, 1000)
    gt = [410, 190, 750, 760]
    det = compute_rewards(parse_cot_output(ANOM), gt, orig, True, cfg)
    assert "delta_iou" in det and "action_consistency" in det
    cons = boundary_action_consistency(
        [380, 220, 760, 810],
        [410, 190, 750, 760],
        {"L": "inward", "R": "keep", "T": "outward", "B": "inward"},
        keep_tol=8.0,
    )
    assert cons >= 0.5


def test_size_bins_and_gated_miou():
    assert defect_size_bin(0.01, True) == "small"
    assert abs(gt_relative_area([0, 0, 10, 10], (100, 100)) - 0.01) < 1e-9
    rows = [
        {"is_anomaly": True, "pred_cls": True, "class_name": "a", "iou_f": 0.6, "iou_c": 0.4, "a_gt": 0.2, "rec_ok": True},
        {"is_anomaly": True, "pred_cls": False, "class_name": "a", "iou_f": 0.9, "iou_c": 0.1, "a_gt": 0.01, "rec_ok": False},
    ]
    s = summarize_detection_metrics(rows)
    assert abs(s["mean_iou"] - 0.75) < 1e-6
    assert abs(s["mean_iou_gated"] - 0.3) < 1e-6


def test_pad_id_equals_eos_keeps_real_eos():
    eos = 2
    seqs = [torch.tensor([1, 1, 1, 9, 8, eos]), torch.tensor([1, 1, 1, 7, eos])]
    outputs, attn, labels = padded_completion_tensors(seqs, prompt_len=3, pad_id=eos, device=torch.device("cpu"))
    assert int(attn[0, 5].item()) == 1
    assert int(labels[0, 5].item()) == eos
    assert int(attn[1, 5].item()) == 0
    assert int(labels[1, 5].item()) == -100
