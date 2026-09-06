"""Tests for set-level localization metrics (Hungarian matching, component IoU)."""
import pytest

from outcome.metrics import component_metrics, hungarian_matching, union_box, union_iou


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
