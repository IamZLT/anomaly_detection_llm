"""Parser, reward, and GRPO protocol checks (no GPU)."""

from __future__ import annotations

import torch

from data.prior_dataset import build_train_ref_pool, build_user_prompt, pick_ref_image
from evaluation.metrics import defect_size_bin, gt_relative_area, summarize_detection_metrics
from models.qwen35 import apply_processor_geometry
from models.vision_cache import expand_cached_image_feats, format_prior_hint, topk_spatial_points
from reasoning.parser import (
    parse_answer_block,
    parse_bbox_field,
    parse_cot_output,
    parse_cot_output_task,
)
from reasoning.rewards import (
    box_iou,
    compute_rewards,
    dense_geometry_reward,
    edge_precision_reward,
    h_anchor_reward,
    refinement_directions,
)
from models.anomaly_prior import prior_box_from_heatmap
from rl.grpo import clipped_pg_kl, expand_gen_in_for_group, micro_batch_ranges, model_inputs, padded_completion_tensors
from utils.common import qwen_smart_hw
from visualization.tensorboard import log_grpo_scalars

ANOM = """
<understand>
Image 1 shows an intact bottle body. Image 2 shows the same object with a localized irregular break.
</understand>
<compare>
The inspection image contains an irregular structural break on the bottle body that is absent from the normal reference.
</compare>
<ground>
candidate_bbox_2d=[380,220,760,810]
The most suspicious region is concentrated around the damaged bottle body.
</ground>
<verify>
The irregular structure cannot be explained by illumination or normal appearance and is consistent with a physical defect.
</verify>
<answer>
{"is_anomaly": true, "bbox_2d": [410,190,750,760], "description": "An irregular structural break is present on the bottle body and is absent from the normal reference."}
</answer>
"""

NORM = """
<understand>
Both images show a bottle with consistent overall shape and surface appearance.
</understand>
<compare>
The inspection sample is consistent with the reference in overall shape, texture, and surface appearance.
</compare>
<ground>
candidate_bbox_2d=null
No localized difference provides sufficient evidence of a defect.
</ground>
<verify>
The observed variations are consistent with normal appearance.
</verify>
<answer>
{"is_anomaly": false, "bbox_2d": null, "description": "The inspection image is consistent with the normal reference, with no clear defect observed."}
</answer>
"""

NORM_REJECT = """
<understand>
Image 2 has a localized appearance difference relative to the normal reference Image 1.
</understand>
<compare>
Image 2 contains a localized appearance difference relative to Image 1.
</compare>
<ground>
candidate_bbox_2d=[280,420,410,560]
A small region differs from the reference and requires verification.
</ground>
<verify>
After comparison with the normal reference, the candidate is consistent with normal appearance variation rather than a true defect.
</verify>
<answer>
{"is_anomaly": false, "bbox_2d": null, "description": "The localized difference is consistent with normal variation and no true defect is observed."}
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


def test_answer_description_is_required():
    p = parse_cot_output(ANOM)
    assert p["description_ok"]
    assert p["description_state"] == "ok"
    assert "bottle body" in p["description"]
    missing = ANOM.replace(
        ', "description": "An irregular structural break is present on the bottle body and is absent from the normal reference."',
        "",
    )
    bad = parse_cot_output(missing)
    assert bad["answer_state"] == "invalid"
    assert bad["is_anomaly"] is None
    assert bad["bbox_2d"] is None
    assert not bad["description_ok"]
    assert not bad["trajectory_valid"]
    copied = ANOM.replace(
        "An irregular structural break is present on the bottle body and is absent from the normal reference.",
        "Do not copy these instructions into the blocks.",
    )
    assert not parse_cot_output(copied)["trajectory_valid"]
    short = ANOM.replace(
        "An irregular structural break is present on the bottle body and is absent from the normal reference.",
        "A defect.",
    )
    assert not parse_cot_output(short)["trajectory_valid"]


def test_answer_json_rejects_nonstr_description_and_extra_keys():
    obj_desc = ANOM.replace(
        '"description": "An irregular structural break is present on the bottle body and is absent from the normal reference."',
        '"description": {"foo": "this is not actually a sentence but it is sufficiently long"}',
    )
    p = parse_cot_output(obj_desc)
    assert p["answer_state"] == "invalid"
    assert p["is_anomaly"] is None
    assert p["bbox_2d"] is None
    assert p["description"] is None
    assert not p["description_ok"]
    assert not p["trajectory_valid"]

    extra = ANOM.replace(
        '"description": "An irregular structural break is present on the bottle body and is absent from the normal reference."}',
        '"description": "An irregular structural break is present on the bottle body and is absent from the normal reference.", "defect_type": "scratch", "confidence": 0.98}',
    )
    p = parse_cot_output(extra)
    assert p["answer_state"] == "invalid"
    assert p["is_anomaly"] is None
    assert p["bbox_2d"] is None
    assert p["description"] is None
    assert not p["description_ok"]
    assert not p["trajectory_valid"]

    parsed = parse_answer_block(
        '{"is_anomaly": true, "bbox_2d": [10,20,30,40], "description": ["long enough list disguised as text"]}'
    )
    assert parsed["answer_state"] == "invalid"
    assert parsed["is_anomaly"] is None
    assert parsed["description"] is None


def test_ground_bbox_only_fails_trajectory():
    ground_only = ANOM.replace(
        "The most suspicious region is concentrated around the damaged bottle body.\n",
        "",
    )
    p = parse_cot_output(ground_only)
    assert not p["prose_ok"]
    assert not p["trajectory_valid"]


def test_invalid_not_cheaper_than_wrong():
    cfg = {"grpo": {"reward": {"invalid_output": -1.0, "wrong_decision": -1.0}}}
    orig = (1000, 1000)
    gt = [410, 190, 750, 760]
    broken = ANOM.replace(
        '{"is_anomaly": true, "bbox_2d": [410,190,750,760], "description": "An irregular structural break is present on the bottle body and is absent from the normal reference."}',
        "not json",
    )
    rb = compute_rewards(parse_cot_output(broken), gt, orig, True, cfg)
    rw = compute_rewards(parse_cot_output(NORM), gt, orig, True, cfg)
    # R_final is decoupled from R_fmt: an invalid output and a hard wrong decision both -1.
    assert rb["R_final"] == -1.0
    assert rw["R_final"] == -1.0
    assert rb["R_ground"] > 0.0
    rn = compute_rewards(parse_cot_output(NORM), None, orig, False, cfg)
    assert rn["R_final"] == 1.0


def test_unclosed_answer_is_invalid_in_strict_parser():
    # Model commonly ends its turn (EOS) right after the JSON, dropping </answer>.
    unclosed = ANOM.rsplit("</answer>", 1)[0]

    p = parse_cot_output(unclosed)
    assert not p["has_tags"]
    assert not p["trajectory_valid"]

    pt = parse_cot_output_task(unclosed)
    assert pt["answer_state"] == "ok"
    assert pt["bbox_2d"] == [410.0, 190.0, 750.0, 760.0]


def test_misspelled_candidate_gets_partial_format_reward():
    cfg = {"grpo": {"reward": {"invalid_output": -1.0, "wrong_decision": -1.0}}}
    bad = ANOM.replace(
        "candidate_bbox_2d=",
        "Candidate bbox 2d=",
    )

    det = compute_rewards(
        parse_cot_output(bad),
        [410, 190, 750, 760],
        (1000, 1000),
        True,
        cfg,
    )

    assert abs(det["R_fmt"] - 0.70) < 1e-6
    assert det["R_final"] == -1.0


def test_normal_candidate_can_be_rejected():
    cfg = {"grpo": {"reward": {}}}
    p = parse_cot_output(NORM_REJECT)

    assert p["candidate_bbox_state"] == "box"
    assert p["is_anomaly"] is False
    assert p["final_bbox_state"] == "null"
    assert p["trajectory_valid"]

    det = compute_rewards(
        p,
        None,
        (1000, 1000),
        False,
        cfg,
    )

    assert det["R_final"] == 1.0


def test_normal_null_candidate_gets_full_reward():
    cfg = {"grpo": {"reward": {}}}
    p = parse_cot_output(NORM)
    assert p["candidate_bbox_state"] == "null"
    assert p["is_anomaly"] is False
    assert p["trajectory_valid"]

    det = compute_rewards(p, None, (1000, 1000), False, cfg)
    assert det["R_final"] == 1.0


def test_normal_full_image_candidate_is_penalized():
    cfg = {"grpo": {"reward": {}}}
    text = """
<understand>
Image 1 is intact while Image 2 shows only normal appearance variation.
</understand>
<compare>
Image 2 looks the same as Image 1 with no localized defect.
</compare>
<ground>
candidate_bbox_2d=[0,0,1000,1000]
The whole image is proposed as a suspicious region.
</ground>
<verify>
The whole-image candidate is rejected as normal variation.
</verify>
<answer>
{"is_anomaly": false, "bbox_2d": null, "description": "The inspection image is consistent with the normal reference with no clear defect."}
</answer>
"""

    p = parse_cot_output(text)
    assert p["trajectory_valid"]
    assert p["is_anomaly"] is False
    assert p["candidate_bbox_state"] == "box"

    det = compute_rewards(p, None, (1000, 1000), False, cfg)
    assert det["full_image_cand"] == 1.0
    assert det["R_final"] == -1.0


def test_format_reward_monotonic():
    cfg = {"grpo": {"reward": {"invalid_output": -1.0, "wrong_decision": -1.0}}}
    orig = (1000, 1000)
    gt = [410, 190, 750, 760]
    full = compute_rewards(parse_cot_output(ANOM), gt, orig, True, cfg)
    assert full["R_fmt"] == 1.0
    assert full["R_final"] > 0.0  # ANOM is a fully valid anomaly -> positive IoU reward
    only_compare = "<compare>\nImage 2 differs from Image 1.\n</compare>\n"
    p_only = parse_cot_output(only_compare)
    det_only = compute_rewards(p_only, gt, orig, True, cfg)
    assert abs(det_only["R_fmt"] - 0.15) < 1e-6
    assert det_only["R_final"] == -1.0
    three_blocks = (
        "<understand>\nBoth images show the same object class.\n</understand>\n"
        "<compare>\nImage 2 differs from Image 1.\n</compare>\n"
        "<ground>\ncandidate_bbox_2d=null\nThe region is suspicious.\n</ground>\n"
        "<verify>\nIt is a defect.\n</verify>\n"
    )
    det_three = compute_rewards(parse_cot_output(three_blocks), gt, orig, True, cfg)
    # understand+compare+ground+verify = 0.10+0.15+0.30+0.15 = 0.70
    assert abs(det_three["R_fmt"] - 0.70) < 1e-6
    assert det_three["R_fmt"] > det_only["R_fmt"]


def test_scale_aware_edge():
    orig = (1000, 1000)
    gt = [100.0, 100.0, 130.0, 130.0]
    pred = [120.0, 100.0, 130.0, 130.0]
    r_old_scale = edge_precision_reward(pred, gt, orig, beta=8.0, min_frac=1.0)
    r_gt_scale = edge_precision_reward(pred, gt, orig, beta=8.0, min_frac=0.05)
    assert r_gt_scale < r_old_scale


def test_dense_geometry_exact_match_is_one():
    gt = [100, 200, 300, 400]

    r = dense_geometry_reward(gt, gt, (1000, 1000))

    assert abs(r - 1.0) < 1e-6


def test_dense_geometry_distinguishes_zero_iou_boxes():
    gt = [100, 600, 250, 800]
    orig = (1000, 1000)

    near = [280, 580, 430, 780]
    far = [700, 100, 850, 300]

    assert box_iou(near, gt) == 0.0
    assert box_iou(far, gt) == 0.0

    r_near = dense_geometry_reward(near, gt, orig)
    r_far = dense_geometry_reward(far, gt, orig)

    assert r_near > r_far


def test_dense_geometry_has_no_far_distance_plateau():
    orig = (1000, 1000)

    gt = [100, 600, 250, 800]

    mid = [500, 100, 650, 300]
    far = [700, 100, 850, 300]

    assert box_iou(mid, gt) == 0.0
    assert box_iou(far, gt) == 0.0

    r_mid = dense_geometry_reward(mid, gt, orig)
    r_far = dense_geometry_reward(far, gt, orig)

    assert r_mid > r_far


def test_dense_progress_rewards_better_refinement():
    gt = [400, 400, 600, 600]
    orig = (1000, 1000)

    bc = [650, 400, 850, 600]
    bf = [560, 400, 760, 600]

    rc = dense_geometry_reward(bc, gt, orig)
    rf = dense_geometry_reward(bf, gt, orig)

    assert rf > rc


def test_full_image_box_cannot_hack_dense_reward():
    orig = (1000, 1000)

    gt = [450, 450, 550, 550]

    full = [0, 0, 1000, 1000]
    near = [400, 400, 500, 500]

    r_full = dense_geometry_reward(full, gt, orig)
    r_near = dense_geometry_reward(near, gt, orig)

    assert r_full < 0.05
    assert r_near > r_full


def test_full_image_candidate_does_not_get_high_ground_reward():
    cfg = {"grpo": {"reward": {}}}
    orig = (1000, 1000)
    gt = [450, 450, 550, 550]

    text = """
<understand>
Image 1 is intact while Image 2 shows a localized defect near the center.
</understand>
<compare>
A localized difference is visible between Image 1 and Image 2.
</compare>
<ground>
candidate_bbox_2d=[0,0,1000,1000]
The whole image is proposed as a suspicious region.
</ground>
<verify>
The candidate is checked against the reference and refined to the defect.
</verify>
<answer>
{"is_anomaly": true, "bbox_2d": [0,0,1000,1000], "description": "A suspicious region is present in the inspection image relative to the reference."}
</answer>
"""

    det = compute_rewards(parse_cot_output(text), gt, orig, True, cfg)

    assert det["R_ground"] < 0.05
    assert det["R_dense_c"] < 0.05
    assert det["candidate_area_ratio"] > 0.8
    assert det["full_image_cand"] == 1.0


def test_same_center_same_area_wrong_aspect_cannot_get_perfect_dense():
    gt = [300, 450, 700, 550]  # 400 x 100
    bad = [450, 300, 550, 700]  # 100 x 400

    r = dense_geometry_reward(bad, gt, (1000, 1000))

    assert r < 0.5
    assert abs(box_iou(bad, gt)) < 0.2


def test_dense_geometry_gamma_sharpens_wrong_scale():
    gt = [450, 450, 550, 550]
    # Center matches but box is 3x too large in each side.
    bad = [300, 300, 700, 700]

    r1 = dense_geometry_reward(bad, gt, (1000, 1000), gamma=1.0)
    r2 = dense_geometry_reward(bad, gt, (1000, 1000), gamma=2.0)

    assert r2 < r1
    # An exact box is still a perfect reward regardless of gamma.
    assert abs(dense_geometry_reward(gt, gt, (1000, 1000), gamma=2.0) - 1.0) < 1e-6


def test_final_reward_includes_raw_iou():
    cfg = {
        "grpo": {
            "reward": {
                "final_iou_weight": 0.4,
                "final_dense_weight": 0.5,
                "dense_scale_gamma": 2.0,
            }
        }
    }
    orig = (1000, 1000)
    gt = [450, 450, 550, 550]
    text = ANOM.replace(
        "candidate_bbox_2d=[380,220,760,810]", "candidate_bbox_2d=[450,450,550,550]"
    ).replace('"bbox_2d": [410,190,750,760]', '"bbox_2d": [450,450,550,550]')
    det = compute_rewards(parse_cot_output(text), gt, orig, True, cfg)
    assert abs(det["R_iou"] - 1.0) < 1e-6
    assert abs(det["R_final"] - 1.0) < 1e-6


def test_final_reward_copying_candidate_is_not_free():
    # When Bf == Bc and both are imprecise, raw IoU in R_final must prevent the
    # copy from earning the same reward as a truly refined (exact) box.
    cfg = {
        "grpo": {
            "reward": {
                "final_iou_weight": 0.4,
                "final_dense_weight": 0.5,
                "dense_scale_gamma": 2.0,
            }
        }
    }
    orig = (1000, 1000)
    gt = [450, 450, 550, 550]
    # Candidate = final = a 3x oversized box centered on GT.
    copy_text = ANOM.replace(
        "candidate_bbox_2d=[380,220,760,810]", "candidate_bbox_2d=[300,300,700,700]"
    ).replace('"bbox_2d": [410,190,750,760]', '"bbox_2d": [300,300,700,700]')
    det_copy = compute_rewards(parse_cot_output(copy_text), gt, orig, True, cfg)

    exact_text = ANOM.replace(
        "candidate_bbox_2d=[380,220,760,810]", "candidate_bbox_2d=[450,450,550,550]"
    ).replace('"bbox_2d": [410,190,750,760]', '"bbox_2d": [450,450,550,550]')
    det_exact = compute_rewards(parse_cot_output(exact_text), gt, orig, True, cfg)

    assert det_copy["R_final"] < det_exact["R_final"]
    assert det_copy["raw_iou_f"] < det_exact["raw_iou_f"]


def test_dense_reward_does_not_change_normal_gate():
    cfg = {
        "grpo": {
            "reward": {
                "normal_correct": 1.0,
                "wrong_decision": -1.0,
                "invalid_output": -1.0,
            }
        }
    }

    p = parse_cot_output(NORM_REJECT)

    det = compute_rewards(p, None, (1000, 1000), False, cfg)

    assert det["R_final"] == 1.0
    assert det["R_ground"] == 0.0
    assert det["R_reason"] == 0.0


def test_h_anchor_reward_is_iou_with_prior_box():
    prior = [100.0, 100.0, 200.0, 200.0]
    assert abs(h_anchor_reward([100.0, 100.0, 200.0, 200.0], prior) - 1.0) < 1e-6
    assert h_anchor_reward(None, prior) == 0.0
    assert h_anchor_reward([100.0, 100.0, 200.0, 200.0], None) == 0.0
    assert h_anchor_reward([700.0, 700.0, 800.0, 800.0], prior) == 0.0


def test_prior_box_from_heatmap_finds_blob():
    h = torch.zeros(16, 16)
    h[4:8, 6:12] = 1.0
    box = prior_box_from_heatmap(h, thresh_frac=0.5)
    assert box is not None
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000
    # blob spans x in [6,11], y in [4,7] → centers ~ (8.5, 5.5) of 16
    assert abs(x1 - round(6.5 / 16 * 1000)) < 50
    assert abs(y2 - round(7.5 / 16 * 1000)) < 50


def test_prior_box_from_heatmap_flat_returns_none():
    assert prior_box_from_heatmap(torch.zeros(8, 8)) is None


def test_h_anchor_breaks_dead_zone_ordering():
    cfg = {"grpo": {"reward": {"h_anchor_weight": 0.2, "dense_scale_gamma": 2.0}}}
    orig = (1000, 1000)
    gt = [450, 450, 550, 550]
    prior = [100, 100, 200, 200]  # H points to the top-left corner

    def det_for(cand):
        text = ANOM.replace(
            "candidate_bbox_2d=[380,220,760,810]",
            f"candidate_bbox_2d=[{','.join(map(str, cand))}]",
        ).replace(
            '"bbox_2d": [410,190,750,760]',
            f'"bbox_2d": [{",".join(map(str, cand))}]',
        )
        return compute_rewards(parse_cot_output(text), gt, orig, True, cfg, prior_box=prior)

    d_near = det_for([100, 100, 200, 200])  # on H, zero IoU with GT
    d_far = det_for([800, 800, 900, 900])   # same size/distance from GT, away from H

    assert d_near["raw_iou_c"] == 0.0 and d_far["raw_iou_c"] == 0.0
    assert abs(d_near["R_dense_c"] - d_far["R_dense_c"]) < 1e-6
    assert d_near["h_anchor"] == 1.0
    assert d_far["h_anchor"] == 0.0
    assert d_near["R_ground"] > d_far["R_ground"]


def test_h_anchor_does_not_override_gt_truth():
    cfg = {"grpo": {"reward": {"h_anchor_weight": 0.2, "dense_scale_gamma": 2.0}}}
    orig = (1000, 1000)
    gt = [450, 450, 550, 550]
    prior = [100, 100, 200, 200]  # H points AWAY from GT (a wrong hint)

    text = ANOM.replace(
        "candidate_bbox_2d=[380,220,760,810]", "candidate_bbox_2d=[450,450,550,550]"
    ).replace('"bbox_2d": [410,190,750,760]', '"bbox_2d": [450,450,550,550]')
    det = compute_rewards(parse_cot_output(text), gt, orig, True, cfg, prior_box=prior)

    assert det["h_anchor"] == 0.0  # candidate is not on the (wrong) H
    assert abs(det["R_ground"] - 1.0) < 1e-6  # GT truth still gives a perfect score


def test_refinement_direction_from_boxes():
    cfg = {"grpo": {"reward": {}}}
    orig = (1000, 1000)
    gt = [410, 190, 750, 760]
    det = compute_rewards(parse_cot_output(ANOM), gt, orig, True, cfg)
    assert det["R_dir"] == 1.0
    # R_reason = reason_dir_weight*R_dir + (1-w)*delta_dense: still dominated by
    # the direction term, but no longer a pure discrete equality.
    assert 0.0 < det["R_reason"] <= 1.0
    assert "delta_iou" in det
    assert "delta_dense" in det
    assert "action_consistency" not in det
    dirs = refinement_directions([380, 220, 760, 810], [410, 190, 750, 760], 8.0)
    assert dirs == {"L": "inward", "R": "inward", "T": "outward", "B": "inward"}


def test_leftover_edge_numbers_do_not_invalidate_trajectory():
    leftover = ANOM.replace(
        "The irregular structure cannot be explained by illumination or normal appearance and is consistent with a physical defect.",
        "The candidate is re-checked against Image 1 and refined to the damaged region.\nleft=662\nright=662\ntop=190\nbottom=760",
    )
    p = parse_cot_output(leftover)
    assert p["trajectory_valid"]
    assert p["candidate_bbox_2d"] == [380.0, 220.0, 760.0, 810.0]
    assert p["bbox_2d"] == [410.0, 190.0, 750.0, 760.0]


def test_size_bins_and_gated_miou():
    assert defect_size_bin(0.01, True) == "small"
    assert abs(gt_relative_area([0, 0, 10, 10], (100, 100)) - 0.01) < 1e-9
    rows = [
        {"is_anomaly": True, "pred_is_anomaly": True, "class_name": "a", "iou_f": 0.6, "iou_c": 0.4, "a_gt": 0.2, "rec_ok": True},
        {"is_anomaly": True, "pred_is_anomaly": False, "class_name": "a", "iou_f": 0.9, "iou_c": 0.1, "a_gt": 0.01, "rec_ok": False},
    ]
    s = summarize_detection_metrics(rows)
    assert abs(s["mean_iou"] - 0.75) < 1e-6
    assert abs(s["mean_iou_gated"] - 0.3) < 1e-6


def test_processor_null_min_pixels_uses_official_floor():
    class _Img:
        patch_size = 16
        merge_size = 2
        size = {"shortest_edge": 65536, "longest_edge": 1280 * 28 * 28}
        min_pixels = 1024
        max_pixels = 1280 * 28 * 28

    class _Proc:
        image_processor = _Img()

    proc = _Proc()
    apply_processor_geometry(
        proc,
        {"data": {"max_image_size": 448, "min_pixels": None, "max_pixels": None}},
    )
    assert proc.image_processor.min_pixels == 65536
    assert proc.image_processor.max_pixels == 448 * 448


def test_processor_missing_official_min_defaults_to_256sq():
    class _Img:
        patch_size = 16
        merge_size = 2
        size = "square"
        min_pixels = None
        max_pixels = None

    class _Proc:
        image_processor = _Img()

    proc = _Proc()
    apply_processor_geometry(proc, {"data": {"max_image_size": 448}})
    assert proc.image_processor.min_pixels == 256 * 256
    assert proc.image_processor.max_pixels == 448 * 448


def test_train_ref_pool_excludes_dev_and_anomalies():
    train = [
        {"metadata": {"anomaly": False, "class": "pcb"}, "full_img_path": "/visa/pcb/n_train.png"},
        {"metadata": {"anomaly": True, "class": "pcb"}, "full_img_path": "/visa/pcb/a_train.png"},
    ]
    pool = build_train_ref_pool(train)
    assert pool["pcb"] == ["/visa/pcb/n_train.png"]
    picked = pick_ref_image(
        "pcb",
        "/unused",
        "/visa/pcb/query.png",
        cands=pool.get("pcb", []),
    )
    assert picked == "/visa/pcb/n_train.png"
    try:
        pick_ref_image("pcb", "/unused", "/q.png", cands=[])
        raise AssertionError("empty pool must not scan disk")
    except FileNotFoundError:
        pass


def test_topk_spatial_nms_and_prior_hint():
    h = torch.zeros(8, 8)
    h[2, 2] = 1.0
    h[2, 3] = 0.95
    h[6, 6] = 0.8
    pts = topk_spatial_points(h, k=2, nms_radius=1)
    assert len(pts) == 2
    assert pts[0] == [int(round(2.5 / 8 * 1000)), int(round(2.5 / 8 * 1000))]
    assert pts[1] == [int(round(6.5 / 8 * 1000)), int(round(6.5 / 8 * 1000))]
    text = format_prior_hint(pts)
    assert text.startswith("<prior_hint>")
    assert "high_response_points_2d=" in text
    assert "</prior_hint>" in text


def test_prompt_is_two_images_plus_spatial_hint():
    # prompt.user is now the single source of truth; there is no silent fallback.
    try:
        build_user_prompt({"prompt": {}}, "bottle")
        raise AssertionError("build_user_prompt must require prompt.user")
    except ValueError:
        pass

    tmpl = (
        "Image 1 is a defect-free normal reference of a {class_name}.\n"
        "Image 2 is the inspection image.\n"
        "These points are only spatial hints, not defect labels.\n"
        "Required structure:\n"
        "<understand>...</understand>\n"
        "<compare>...</compare>\n"
        "<ground>...</ground>\n"
        "<verify>...</verify>\n"
        "<answer>...</answer>\n"
    )
    text = build_user_prompt({"prompt": {"user": tmpl}}, "bottle")
    assert "Image 1" in text and "Image 2" in text
    assert "Image 3" not in text
    assert "bottle" in text
    assert "spatial hint" in text.lower()
    assert "<understand>" in text
    assert "<boundary>" not in text
    assert "five XML" not in text.lower()


def test_model_inputs_drops_image_embeds():
    ids = torch.zeros(1, 4, dtype=torch.long)
    cache = torch.randn(6, 8)
    out = model_inputs(
        {
            "input_ids": ids,
            "image_embeds": cache,
            "labels": torch.zeros_like(ids),
            "_meta": [{"x": 1}],
            "prompt_len": torch.tensor([4]),
        }
    )
    assert "image_embeds" not in out
    assert "labels" not in out
    assert "input_ids" in out


def test_tb_grpo_scalars_are_allowlisted():
    class _W:
        def __init__(self):
            self.tags = []

        def add_scalar(self, tag, value, step):
            self.tags.append(tag)

        def flush(self):
            pass

    w = _W()
    det = {
        "R_ground": 0.1,
        "R_reason": 0.2,
        "R_final": 0.3,
        "R_iou": 0.4,
        "R_iou_c": 0.5,
        "delta_iou": 0.05,
        "R_dir": 0.8,
    }
    log_grpo_scalars(
        w,
        step=1,
        loss=0.0,
        rewards=torch.tensor([0.3, 0.4]),
        details=[det, det],
        advantages=torch.tensor([0.1, -0.1]),
        seq_lp=torch.tensor([-1.0, -1.2]),
        texts=[ANOM, ANOM],
        lr=1e-5,
        params={"group_size": 8.0, "temperature": 0.9},
        grad_norm=0.5,
        extra={"loss_pg": 0.1, "loss_kl": 0.01, "rho_mean": 1.0, "clip_frac": 0.0, "resample": 1.0, "skipped": 0.0},
    )
    assert "grpo/param/group_size" not in w.tags
    assert "grpo/traj_0/R_final" not in w.tags
    assert "grpo/reward_mean" not in w.tags
    want = {
        "grpo/loss",
        "grpo/grad_norm",
        "grpo/lr",
        "grpo/pg_loss",
        "grpo/kl",
        "grpo/rho",
        "grpo/clip_frac",
        "grpo/R_ground",
        "grpo/R_reason",
        "grpo/R_final",
        "grpo/reward_std",
        "grpo/R_iou_c",
        "grpo/R_iou",
        "grpo/delta_iou",
        "grpo/R_dir",
        "grpo/raw_iou_f",
        "grpo/raw_iou_c",
        "grpo/R_fmt",
        "grpo/R_dense_c",
        "grpo/R_dense_f",
        "grpo/delta_dense",
        "grpo/candidate_area_ratio",
        "grpo/final_area_ratio",
        "grpo/pred_gt_area_ratio",
        "grpo/full_image_box_rate",
        "grpo/h_anchor",
        "grpo/h_follow_rate",
        "grpo/h_override_rate",
        "grpo/kl_contrib",
        "grpo/ref_gap",
        "protocol/strict_trajectory_rate",
        "protocol/task_trajectory_rate",
        "protocol/strict_answer_rate",
        "protocol/task_answer_rate",
        "protocol/strict_final_valid_rate",
        "protocol/task_final_valid_rate",
        "grpo/protocol_rate",
        "grpo/trajectory_valid_rate",
        "grpo/candidate_valid_rate",
        "grpo/final_valid_rate",
        "grpo/box_pair_valid_rate",
        "grpo/unique_response_rate",
        "grpo/resample_n",
        "grpo/skip_rate",
    }
    assert set(w.tags) == want


def test_micro_batch_pg_matches_full_group():
    assert micro_batch_ranges(8, 2) == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert micro_batch_ranges(8, 1) == [(i, i + 1) for i in range(8)]
    new = torch.randn(8, 5)
    old = torch.randn(8, 5)
    ref = torch.randn(8, 5)
    mask = torch.ones(8, 5)
    adv = torch.randn(8, 5)
    loss_full, pg_full, kl_full, rho_full, clip_full = clipped_pg_kl(new, mask, old, ref, adv, 0.2, 0.28, 0.1)
    for micro in (1, 2):
        loss_acc = pg_acc = kl_acc = rho_acc = clip_acc = 0.0
        for s, e in micro_batch_ranges(8, micro):
            loss, pg, kl, rho, clip = clipped_pg_kl(
                new[s:e], mask[s:e], old[s:e], ref[s:e], adv[s:e], 0.2, 0.28, 0.1
            )
            w = (e - s) / 8.0
            loss_acc += float(loss) * w
            pg_acc += float(pg) * w
            kl_acc += float(kl) * w
            rho_acc += float(rho) * w
            clip_acc += float(clip) * w
        assert abs(loss_acc - float(loss_full)) < 1e-5
        assert abs(pg_acc - float(pg_full)) < 1e-5
        assert abs(kl_acc - float(kl_full)) < 1e-5
        assert abs(rho_acc - float(rho_full)) < 1e-5
        assert abs(clip_acc - float(clip_full)) < 1e-5


def test_expand_cached_image_feats_repeats_group():
    cache = torch.arange(12, dtype=torch.float32).view(6, 2)
    out = expand_cached_image_feats(cache, 48)
    assert tuple(out.shape) == (48, 2)
    assert torch.equal(out[:6], cache)
    assert torch.equal(out[6:12], cache)


def test_expand_gen_in_for_group_repeats_grid_not_pixels():
    gen = {
        "input_ids": torch.arange(4).view(1, 4),
        "image_grid_thw": torch.tensor([[1, 8, 8], [1, 8, 8]]),
        "pixel_values": torch.randn(10, 16),
        "mm_token_type_ids": torch.zeros(1, 4, dtype=torch.long),
    }
    out = expand_gen_in_for_group(gen, 8)
    assert tuple(out["image_grid_thw"].shape) == (16, 3)
    assert tuple(out["pixel_values"].shape) == (10, 16)
    assert tuple(out["input_ids"].shape) == (8, 4)
    assert tuple(out["mm_token_type_ids"].shape) == (8, 4)


def test_clipped_pg_kl_averages_per_sequence():
    new = torch.zeros(2, 4)
    old = torch.zeros(2, 4)
    ref = torch.zeros(2, 4)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    adv = torch.tensor([[1.0, 1.0, 1.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
    _, L_pg, L_kl, _, _ = clipped_pg_kl(new, mask, old, ref, adv, 0.2, 0.28, 0.0)
    assert abs(float(L_pg) - (-2.0)) < 1e-5
    assert abs(float(L_kl)) < 1e-5


def test_pad_id_equals_eos_keeps_real_eos():
    eos = 2
    seqs = [torch.tensor([1, 1, 1, 9, 8, eos]), torch.tensor([1, 1, 1, 7, eos])]
    outputs, attn, labels = padded_completion_tensors(seqs, prompt_len=3, pad_id=eos, device=torch.device("cpu"))
    assert int(attn[0, 5].item()) == 1
    assert int(labels[0, 5].item()) == eos
    assert int(attn[1, 5].item()) == 0
    assert int(labels[1, 5].item()) == -100
