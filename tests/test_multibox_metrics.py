"""Tests for set-level localization metrics (Hungarian matching, component IoU)."""
import pytest

from outcome.metrics import (
    component_metrics,
    detection_metrics,
    greedy_detection_metrics,
    hungarian_matching,
    mask_iou,
    set_giou,
    union_box,
    union_iou,
)
from outcome.protocol import giou


def test_hungarian_matching_maximizes_total_score():
    # 2x2: best pairing is (0->1)=0.9 + (1->0)=0.9 = 1.8, not (0->0)+(1->1)=1.2
    scores = [[0.6, 0.9], [0.9, 0.6]]
    pairs = sorted(hungarian_matching(scores))
    assert pairs == [(0, 1), (1, 0)]


def test_hungarian_matching_rectangular():
    scores = [[0.9, 0.1], [0.1, 0.8], [0.5, 0.4]]  # 3 pred, 2 gt
    pairs = hungarian_matching(scores)
    assert len(pairs) == 2


def test_union_box_and_union_iou():
    assert union_box([]) is None
    assert union_box([[10, 10, 20, 20], [40, 40, 50, 50]]) == [10, 10, 50, 50]
    # two distant defects: union covers the gap
    assert union_iou([[10, 10, 20, 20], [80, 80, 90, 90]], [10, 10, 90, 90]) == pytest.approx(1.0)


def test_mask_iou_exact_match():
    gt = [[10, 10, 20, 20], [40, 40, 60, 60]]
    pred = [[10, 10, 20, 20], [40, 40, 60, 60]]
    assert mask_iou(pred, gt, (100, 100)) == pytest.approx(1.0)


def test_mask_iou_missed_defect():
    # 1 of 2 equal-size components covered -> mask IoU = 100 / 200 = 0.5
    gt = [[10, 10, 20, 20], [40, 40, 50, 50]]
    pred = [[10, 10, 20, 20]]
    assert mask_iou(pred, gt, (100, 100)) == pytest.approx(0.5)


def test_mask_iou_duplicate_boxes_not_penalized():
    # AD-Copilot's key property: redundant boxes over the same region collapse in
    # mask space, so duplicates do NOT lower the score (set_iou would give 1/3).
    gt = [[10, 10, 20, 20]]
    pred = [[10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20]]
    assert mask_iou(pred, gt, (100, 100)) == pytest.approx(1.0)
    assert component_metrics(pred, gt)['set_iou'] == pytest.approx(1 / 3)


def test_mask_iou_false_positive_elsewhere_penalized():
    # a stray box far away inflates the union -> IoU drops to 0.5
    gt = [[10, 10, 20, 20]]
    pred = [[10, 10, 20, 20], [50, 50, 60, 60]]
    assert mask_iou(pred, gt, (100, 100)) == pytest.approx(0.5)


def test_mask_iou_disconnected_gt_not_bridged():
    # tight-union IoU would be 1.0 here; mask IoU keeps the gap uncovered (200/6400)
    pred = [[10, 10, 20, 20], [80, 80, 90, 90]]
    gt = [[10, 10, 90, 90]]
    assert union_iou(pred, [10, 10, 90, 90]) == pytest.approx(1.0)
    assert mask_iou(pred, gt, (100, 100)) == pytest.approx(200 / 6400)


def test_mask_iou_nonoverlapping_is_zero():
    assert mask_iou([[100, 100, 110, 110]], [[10, 10, 20, 20]], (200, 200)) == pytest.approx(0.0)


def test_mask_iou_empty_returns_zero():
    assert mask_iou([], [[10, 10, 20, 20]], (100, 100)) == 0.0
    assert mask_iou([[10, 10, 20, 20]], [], (100, 100)) == 0.0


def test_component_metrics_exact_match():
    gt = [[10, 10, 20, 20], [40, 40, 60, 60]]
    pred = [[10, 10, 20, 20], [40, 40, 60, 60]]
    m = component_metrics(pred, gt)
    assert m['n_pred'] == 2 and m['n_gt'] == 2 and m['count_error'] == 0
    assert m['matched_miou'] == pytest.approx(1.0)
    assert m['recall_at_05'] == pytest.approx(1.0)
    assert m['precision_at_05'] == pytest.approx(1.0)


def test_component_metrics_missed_defect_penalized_by_recall():
    gt = [[10, 10, 20, 20], [40, 40, 60, 60]]
    pred = [[10, 10, 20, 20]]  # found only one
    m = component_metrics(pred, gt)
    assert m['count_error'] == 1
    assert m['recall_at_05'] == pytest.approx(0.5)  # only 1 of 2 matched at 0.5
    assert m['precision_at_05'] == pytest.approx(1.0)  # the one found is correct


def test_component_metrics_duplicate_box_penalized_by_precision_and_count():
    gt = [[10, 10, 20, 20]]
    pred = [[10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20]]  # 3 dup boxes
    m = component_metrics(pred, gt)
    assert m['count_error'] == 2
    assert m['precision_at_05'] == pytest.approx(1 / 3)  # only 1 of 3 matched


def test_component_metrics_empty_gt_returns_zeros():
    m = component_metrics([[1, 1, 2, 2]], [])
    assert m['matched_miou'] == 0.0 and m['recall_at_01'] == 0.0


def test_set_iou_penalizes_missing_gt():
    # 1 perfect pred, 2 GT => matched_miou=1.0 but set_iou=0.5
    gt = [[10, 10, 20, 20], [40, 40, 60, 60]]
    pred = [[10, 10, 20, 20]]
    m = component_metrics(pred, gt)
    assert m['matched_miou'] == pytest.approx(1.0)
    assert m['set_iou'] == pytest.approx(0.5)


def test_prior_component_recall_not_any_hit():
    # H hits 1 of 2 components => component recall_at_05 = 0.5, but any-hit best IoU = 1.0
    comps = [[10, 10, 20, 20], [40, 40, 60, 60]]
    h_boxes = [[10, 10, 20, 20]]  # prior candidate hits only the first component
    m = component_metrics(h_boxes, comps)
    assert m['recall_at_05'] == pytest.approx(0.5)
    assert m['set_iou'] == pytest.approx(0.5)


def test_set_iou_penalizes_duplicate_boxes():
    gt = [[10, 10, 20, 20]]
    pred = [[10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20]]
    m = component_metrics(pred, gt)
    assert m['matched_miou'] == pytest.approx(1.0)
    assert m['set_iou'] == pytest.approx(1 / 3)


def test_detection_metrics_exact_match_both_thresholds():
    gt = [[10, 10, 20, 20], [40, 40, 60, 60]]
    pred = [[10, 10, 20, 20], [40, 40, 60, 60]]
    m = detection_metrics(pred, gt)
    assert m['n_pred'] == 2 and m['n_gt'] == 2
    for key in ('50', '75'):
        assert m[f'precision_at_{key}'] == pytest.approx(1.0)
        assert m[f'recall_at_{key}'] == pytest.approx(1.0)
        assert m[f'f1_at_{key}'] == pytest.approx(1.0)


def test_detection_metrics_threshold_separates_loose_from_strict():
    # pred overlaps gt with IoU 0.6 => TP at 0.5, not at 0.75
    gt = [[0, 0, 100, 100]]
    pred = [[0, 0, 60, 100]]  # intersection 60*100 / union 100*100 = 0.6
    m = detection_metrics(pred, gt)
    assert m['tp_at_50'] == 1 and m['tp_at_75'] == 0
    assert m['recall_at_50'] == pytest.approx(1.0)
    assert m['recall_at_75'] == pytest.approx(0.0)


def test_detection_metrics_empty_vs_empty_is_perfect():
    m = detection_metrics([], [])
    for key in ('50', '75'):
        assert m[f'precision_at_{key}'] == pytest.approx(1.0)
        assert m[f'recall_at_{key}'] == pytest.approx(1.0)
        assert m[f'f1_at_{key}'] == pytest.approx(1.0)


def test_detection_metrics_false_positive_penalized():
    gt = [[10, 10, 20, 20]]
    pred = [[10, 10, 20, 20], [50, 50, 60, 60]]  # 1 correct, 1 FP
    m = detection_metrics(pred, gt)
    assert m['tp_at_50'] == 1 and m['fp_at_50'] == 1
    assert m['precision_at_50'] == pytest.approx(0.5)
    assert m['recall_at_50'] == pytest.approx(1.0)


def test_greedy_detection_metrics_backward_compatible():
    gt = [[10, 10, 20, 20]]
    pred = [[10, 10, 20, 20]]
    m = greedy_detection_metrics(pred, gt)
    assert m['precision'] == pytest.approx(1.0)
    assert m['recall'] == pytest.approx(1.0)
    assert m['f1'] == pytest.approx(1.0)
    assert m['tp'] == 1 and m['fp'] == 0 and m['fn'] == 0


def test_giou_identical_boxes_equals_one():
    assert giou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_giou_nonoverlapping_is_negative_and_directional():
    # two boxes far apart -> giou < 0
    far = giou([0, 0, 10, 10], [100, 100, 110, 110])
    assert far < 0
    # closer non-overlapping boxes -> less negative (still < 0)
    near = giou([0, 0, 10, 10], [20, 20, 30, 30])
    assert near < 0 and near > far


def test_set_giou_exact_match_equals_set_iou():
    gt = [[10, 10, 20, 20], [40, 40, 60, 60]]
    pred = [[10, 10, 20, 20], [40, 40, 60, 60]]
    assert set_giou(pred, gt) == pytest.approx(1.0)


def test_set_giou_penalizes_missing_gt():
    gt = [[10, 10, 20, 20], [40, 40, 60, 60]]
    pred = [[10, 10, 20, 20]]
    assert set_giou(pred, gt) == pytest.approx(0.5)


def test_set_giou_negative_for_fully_missed_boxes():
    # non-overlapping pred still gets a negative GIoU set score, unlike flat-0 IoU
    gt = [[10, 10, 20, 20]]
    pred = [[100, 100, 110, 110]]
    assert set_giou(pred, gt) < 0.0
