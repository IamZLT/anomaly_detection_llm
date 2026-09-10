"""Unit tests for the H-guided region contrast token stack (no real model needed)."""
import numpy as np
import pytest
import torch

from models.anomaly_prior import AnomalyPrior
from models.region_adapter import RegionAdapter
from models.region_injection import (
    _RegionScatterEmbedding,
    load_region_adapter,
    save_region_adapter,
)
from outcome.inputs import extract_region_cells, region_proposals


# --------------------------------------------------------------------------- #
# _nn_match
# --------------------------------------------------------------------------- #
def test_nn_match_global_returns_distance_features_and_coords():
    # ref patches: two orthogonal vectors, plus the test patch equal to ref[1].
    ref = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    test = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    out = AnomalyPrior._nn_match(None, test, ref, (2, 1), (2, 1), 0)
    assert out['distance'].shape == (2, 1)
    assert out['matched_ref_features'].shape == (2, 2)
    assert out['match_coordinates'].shape == (2, 2)
    # test[0] == ref[1] -> match coordinate (y=1, x=0)
    assert out['match_coordinates'][0].tolist() == [1, 0]
    # test[1] == ref[0] -> match coordinate (y=0, x=0)
    assert out['match_coordinates'][1].tolist() == [0, 0]
    assert torch.allclose(out['matched_ref_features'][0], ref[1])


def test_nn_match_refuses_to_guess_grid_size():
    ref = torch.zeros(3, 4)
    test = torch.zeros(5, 4)
    with pytest.raises(ValueError):
        AnomalyPrior._nn_match(None, test, ref, (2, 2), (2, 2), 0)


# --------------------------------------------------------------------------- #
# region_proposals masks
# --------------------------------------------------------------------------- #
def test_region_proposals_masks_track_connected_cells():
    h = np.zeros((4, 4))
    h[0, 0] = 1.0
    h[3, 3] = 0.9
    meta, masks, mode = region_proposals(h, {'max_candidates': 2})
    assert len(meta) == 2
    assert masks.shape == (2, 4, 4)
    assert bool(masks[0][0, 0])
    assert bool(masks[1][3, 3])


# --------------------------------------------------------------------------- #
# extract_region_cells
# --------------------------------------------------------------------------- #
def test_extract_region_cells_empty_returns_single_invalid_cell():
    D = 4
    test_f = torch.zeros(4, D)
    ref_f = torch.zeros(4, D)
    hmap = torch.zeros(2, 2)
    out = extract_region_cells(np.zeros((0, 2, 2), dtype=bool), test_f, ref_f, hmap, {})
    assert out['test'].shape == (1, 1, D)
    assert out['valid'].tolist() == [[False]]


def test_extract_region_cells_full_region_splits_into_2x2():
    Ht, Wt, D = 2, 2, 3
    test_f = torch.arange(Ht * Wt * D, dtype=torch.float32).reshape(Ht * Wt, D)
    ref_f = torch.zeros_like(test_f)
    hmap = torch.tensor([[0.2, 0.8], [0.9, 0.3]])
    masks = np.ones((1, Ht, Wt), dtype=bool)
    out = extract_region_cells(masks, test_f, ref_f, hmap, {})
    assert out['test'].shape == (1, 4, D)
    assert out['geom'].shape == (1, 4, 5)
    assert out['hstat'].shape == (1, 4, 2)
    assert out['valid'].tolist() == [[True, True, True, True]]


# --------------------------------------------------------------------------- #
# RegionAdapter
# --------------------------------------------------------------------------- #
def test_region_adapter_shape_and_empty_mask():
    B, n, D = 1, 3, 8
    adapter = RegionAdapter(feature_dim=D, hidden_size=16, intermediate_dim=8)
    raw = dict(test=torch.randn(B, n, D), ref=torch.randn(B, n, D),
               geom=torch.randn(B, n, 5), hstat=torch.randn(B, n, 2),
               valid=torch.ones(B, n, dtype=torch.bool))
    out = adapter(raw)
    assert out.shape == (B, n, 16)
    raw['valid'] = torch.zeros(B, n, dtype=torch.bool)
    out_empty = adapter(raw)
    assert torch.allclose(out_empty[0, 0], adapter.empty)


def test_region_adapter_save_load_roundtrip(tmp_path):
    adapter = RegionAdapter(feature_dim=8, hidden_size=16, intermediate_dim=12)
    path = tmp_path / 'region_adapter.pt'
    save_region_adapter(adapter, path)
    loaded = load_region_adapter(RegionAdapter, path, feature_dim=8, hidden_size=16)
    raw = dict(test=torch.randn(1, 2, 8), ref=torch.randn(1, 2, 8),
               geom=torch.randn(1, 2, 5), hstat=torch.randn(1, 2, 2),
               valid=torch.ones(1, 2, dtype=torch.bool))
    assert torch.allclose(adapter(raw), loaded(raw))


# --------------------------------------------------------------------------- #
# H coverage diagnostic helper
# --------------------------------------------------------------------------- #
def test_box_union_coverage():
    from outcome.protocol import box_union_coverage
    gt = [100, 100, 200, 200]
    assert box_union_coverage(gt, [[100, 100, 200, 200]]) == 1.0
    assert box_union_coverage(gt, [[100, 100, 150, 200]]) == 0.5
    assert box_union_coverage(gt, [[100, 100, 150, 200], [150, 100, 200, 200]]) == 1.0
    assert box_union_coverage(gt, [[0, 0, 50, 50]]) == 0.0
    assert box_union_coverage(None, [[0, 0, 10, 10]]) is None


# --------------------------------------------------------------------------- #
# region injection scatter
# --------------------------------------------------------------------------- #
class _Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(10, 4)

    def forward(self, input_ids):
        return self.emb(input_ids)


def test_region_scatter_embedding_replaces_placeholder_positions():
    base = _Emb()
    region_embeds = torch.tensor([[1.0, 1, 1, 1], [2.0, 2, 2, 2]])
    wrapper = _RegionScatterEmbedding(base, region_token_id=9, region_embeds=region_embeds)
    ids = torch.tensor([[1, 9, 2, 9]])
    embeds = wrapper(ids)
    assert embeds.shape == (1, 4, 4)
    assert torch.allclose(embeds[0, 1], region_embeds[0])
    assert torch.allclose(embeds[0, 3], region_embeds[1])
    assert torch.allclose(embeds[0, 0], base.emb(torch.tensor(1)))
    assert torch.allclose(embeds[0, 2], base.emb(torch.tensor(2)))


# --------------------------------------------------------------------------- #
# SFT target construction
# --------------------------------------------------------------------------- #
def test_sft_target_is_parseable_with_gt_category_and_bbox():
    from train_region_sft import build_sft_target, _supervised_spans
    from outcome.protocol import parse_output, score_output

    anomaly_meta = dict(is_anomaly=True, orig_size=[500, 500],
                        gt_box_px=[100, 150, 200, 250], defect_type='scratch')
    target = build_sft_target(anomaly_meta)
    parsed = parse_output(target)
    assert parsed['task_valid'] and parsed['is_anomaly'] is True
    assert parsed['bbox_2d'] == [200.0, 300.0, 400.0, 500.0]
    assert parsed['candidate_bbox_2d'] == [200.0, 300.0, 400.0, 500.0]
    assert score_output(parsed, anomaly_meta)['task'] == 1.0

    normal_meta = dict(is_anomaly=False, orig_size=[500, 500],
                       gt_box_px=None, defect_type='good')
    parsed_normal = parse_output(build_sft_target(normal_meta))
    assert parsed_normal['task_valid'] and parsed_normal['is_anomaly'] is False
    assert parsed_normal['bbox_2d'] is None

    spans = _supervised_spans(target)
    assert len(spans) == 2
    # spans must not cover the neutral process blocks
    for s, e in spans:
        assert 'inspect the object' not in target[s:e]
        assert 'compare the inspection image' not in target[s:e]


class _FakeEnc(dict):
    @property
    def input_ids(self):
        return self['input_ids']


class _FakeTok:
    """Whitespace tokenizer with char offsets, sufficient for build_labels."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets = [], []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            start = i
            while i < len(text) and not text[i].isspace():
                i += 1
            ids.append(len(ids) + 1)
            offsets.append((start, i))
        out = {'input_ids': ids}
        if return_offsets_mapping:
            out['offset_mapping'] = offsets
        return _FakeEnc(out)


def test_build_labels_supervises_answer_and_ground_only():
    from train_region_sft import build_labels, build_sft_target
    meta = dict(is_anomaly=True, orig_size=[500, 500],
                gt_box_px=[100, 150, 200, 250], defect_type='scratch')
    target = build_sft_target(meta)
    prompt_ids = [0, 0, 0]
    labels = build_labels(prompt_ids, target, _FakeTok())
    assert labels[:len(prompt_ids)] == [-100] * len(prompt_ids)
    supervised = [t for t in labels[len(prompt_ids):] if t != -100]
    assert len(supervised) > 0
    # process blocks (<understand>/<compare>/<verify>) are masked
    assert -100 in labels[len(prompt_ids):]
