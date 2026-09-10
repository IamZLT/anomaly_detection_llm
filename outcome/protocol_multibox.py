"""outcome-multibox-v1: multi-box detection protocol and set-level reward.

Keeps the five-stage semantics but only <answer> is strict JSON; <ground> and
<verify> are plain text lines to lower protocol syntax entropy for direct GRPO.
Reuses the DCLR localization term and geometry helpers from ``outcome.protocol``
and the Hungarian/metrics helpers from ``outcome.metrics``.
"""
from __future__ import annotations

import json
import re

from outcome.metrics import component_metrics, hungarian_matching, mask_iou, set_giou, union_box
from outcome.protocol import iou, load_object, localization_reward, to_pixels, valid_box

VERSION = 'outcome-multibox-v1'
TAGS = ('understand', 'compare', 'ground', 'verify', 'answer')
BLOCK = re.compile(r'<(understand|compare|ground|verify|answer)>(.*?)</\1>', re.S)
VERIFY = re.compile(r'^\s*(keep|refine|reject|discover|none)\b\s*[;:,\-]?\s*(.*)$', re.I)
VERIFY_ACTIONS = ('keep', 'refine', 'reject', 'discover', 'none')

DEFAULT_MAX_BOXES = 16  # findContours GT p95=9 / p99=20 on VisA-train, see data.scan.compute_max_boxes


def parse_boxes_list(text: str):
    """Parse ``candidate_bboxes_2d=[[x1,y1,x2,y2],...]`` (or ``[]`` / ``null``).

    Returns ``(state, boxes)`` where state is one of 'list','empty','null',
    'invalid' and boxes is a list of validated [x1,y1,x2,y2] floats.
    """
    text = text.strip()
    text = re.sub(r'^\s*candidate_bboxes_2d\s*=\s*', '', text, flags=re.I)
    if text == '':
        return 'invalid', []
    if text.lower() == 'null':
        return 'null', []
    if text == '[]':
        return 'empty', []
    groups = re.findall(r'\[([^\[\]]*)\]', text)
    if not groups:
        return 'invalid', []
    boxes = []
    for g in groups:
        parts = [p.strip() for p in g.split(',')]
        if len(parts) != 4:
            return 'invalid', []
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return 'invalid', []
        if not valid_box(vals):
            return 'invalid', []
        boxes.append(vals)
    return 'list', boxes


def parse_verify(text: str):
    """Return ``(action, evidence)``; action is None when no keyword is found."""
    stripped = text.strip()
    m = VERIFY.match(stripped)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    for action in VERIFY_ACTIONS:
        if re.search(rf'\b{action}\b', text, re.I):
            return action, stripped
    return None, stripped


def parse_output(text: str, max_boxes: int = DEFAULT_MAX_BOXES) -> dict:
    """Parse the 5-block multi-box output. Only <answer> is strict JSON."""
    result = dict(task_valid=False, decision_valid=False, final_geometry_valid=False,
                  is_anomaly=None, bboxes_2d=[], candidate_bboxes_2d=[], candidate_state='missing',
                  verify_action=None, verify_evidence='', action=None, description='', tags={},
                  answer_keys=[], protocol_core=False, protocol_strict=False, num_boxes=0)
    blocks = list(BLOCK.finditer(text))
    result['tags'] = {m[1]: m[2].strip() for m in blocks}
    answers = list(re.finditer(r'<answer>(.*?)</answer>', text, re.S))
    if (len(answers) == 1 and text.count('<answer>') == 1 and text.count('</answer>') == 1
            and not text[answers[0].end():].strip()):
        try:
            obj = load_object(answers[0][1])
            result['answer_keys'] = sorted(obj)
            pred = obj.get('is_anomaly')
            result['decision_valid'] = type(pred) is bool
            result['is_anomaly'] = pred if type(pred) is bool else None
            boxes_raw = obj.get('bboxes_2d')
            if type(pred) is bool:
                if pred is True:
                    geom = (isinstance(boxes_raw, list) and 1 <= len(boxes_raw) <= max_boxes
                            and all(valid_box(b) for b in boxes_raw))
                    result['bboxes_2d'] = [list(b) for b in boxes_raw] if geom else []
                else:
                    geom = isinstance(boxes_raw, list) and len(boxes_raw) == 0
                    result['bboxes_2d'] = []
                result['final_geometry_valid'] = bool(geom)
                result['task_valid'] = result['decision_valid'] and bool(geom)
            desc = obj.get('description', '')
            result['description'] = desc if isinstance(desc, str) else ''
        except (ValueError, TypeError):
            pass
    result['num_boxes'] = len(result['bboxes_2d'])

    ground_text = result['tags'].get('ground', '')
    cstate, cboxes = parse_boxes_list(ground_text)
    if cstate == 'list' and len(cboxes) > max_boxes:
        cstate = 'invalid'
        cboxes = []
    result['candidate_state'] = cstate
    result['candidate_bboxes_2d'] = cboxes

    verify_text = result['tags'].get('verify', '')
    vaction, vevidence = parse_verify(verify_text)
    result['verify_action'] = vaction
    result['action'] = vaction
    result['verify_evidence'] = vevidence

    ordered = [m[1] for m in blocks] == list(TAGS)
    structure = (ordered
                 and not BLOCK.sub('', text).strip()
                 and all(text.count(f'<{t}>') == text.count(f'</{t}>') == 1 for t in TAGS))
    candidate_ok = cstate in ('null', 'empty', 'list')
    verify_ok = vaction is not None
    understand_ok = bool((result['tags'].get('understand') or '').strip())
    compare_ok = bool((result['tags'].get('compare') or '').strip())
    desc_ok = bool(result['description'].strip())

    result['protocol_core'] = bool(ordered and result['task_valid'] and candidate_ok and verify_ok and desc_ok)

    ground_strict = bool(re.fullmatch(r'\s*candidate_bboxes_2d\s*=\s*(null|\[.*\])\s*', ground_text, re.I))
    verify_strict = bool(re.fullmatch(r'\s*(keep|refine|reject|discover|none)\s*;\s*\S.*', verify_text, re.I))
    answer_strict = (result['task_valid']
                     and set(result['answer_keys']) == {'is_anomaly', 'bboxes_2d', 'description'}
                     and desc_ok)
    result['protocol_strict'] = bool(
        structure and result['task_valid'] and candidate_ok and verify_ok
        and ground_strict and verify_strict and answer_strict and understand_ok and compare_ok)
    return result


def count_reward(m: int, n: int) -> float:
    """Explicit box-count reward (AD-FM eq. 5): penalizes wrong enumeration.

    |m-n|=0 -> 1.0, |m-n|=1 -> 0.5, |m-n|>=2 -> -0.1. This gives GRPO a direct
    signal for detecting exactly the right number of defects, which the implicit
    max(N,M) denominator of set_iou cannot isolate.
    """
    d = abs(int(m) - int(n))
    if d == 0:
        return 1.0
    if d == 1:
        return 0.5
    return -0.1


def focus_reward(m: int) -> float:
    """Focus reward for correctly-rejected normal samples (AD-FM eq. 6).

    Encourages the model to actually localize a suspicious region before rejecting
    it: 0 boxes -> 0.0, 1 box -> 0.5, >=2 boxes -> -0.1. Uses the candidate box
    count because a normal sample's final answer must be empty.
    """
    if m == 0:
        return 0.0
    if m == 1:
        return 0.5
    return -0.1


def set_localization_reward(pred_boxes, gt_components, orig_size, iou_threshold=0.30, geometry_weight=0.30):
    """DCLR-style set reward via Hungarian matching.

    S_ij = R_loc(B_i, G_j) for every pred/GT pair; matching maximizes sum S_ij.
    R_set = sum(matched S_ij) / max(N, M) penalizes missed defects and duplicate
    boxes through the denominator without a separate count penalty.
    """
    if not pred_boxes or not gt_components:
        return dict(reward=0.0, matched_pairs=[], s_sum=0.0)
    N, M = len(pred_boxes), len(gt_components)
    S = [[localization_reward(p, g, orig_size, iou_threshold, geometry_weight)['loc_reward']
          for g in gt_components] for p in pred_boxes]
    pairs = hungarian_matching(S)
    s_sum = float(sum(S[i][j] for i, j in pairs))
    reward = s_sum / max(N, M)
    return dict(reward=reward, matched_pairs=pairs, s_sum=s_sum)


def validate_gt(meta):
    w, h = meta['orig_size']
    if w <= 0 or h <= 0:
        raise ValueError('invalid original image size')
    if meta['is_anomaly']:
        comps = meta.get('component_bboxes') or []
        gt = meta.get('gt_box_px')
        has_comps = comps and all(valid_box(b, max(w, h)) and b[2] <= w and b[3] <= h for b in comps)
        if not has_comps and not (gt is not None and valid_box(gt, max(w, h)) and gt[2] <= w and gt[3] <= h):
            raise ValueError(f"anomalous sample missing/invalid component GT: {meta.get('image_path')}")
    elif (meta.get('gt_box_px') is not None) or (meta.get('component_bboxes')):
        raise ValueError('normal sample must have null GT')


def score_output(parsed, meta, protocol_weight=0.01, localization=None, max_boxes=DEFAULT_MAX_BOXES):
    """Fine-grained reward: classification + GIoU/count/focus/refine (AD-FM style).

    ``R_task`` decomposes as:

    * wrong decision (or invalid output)  -> ``wrong_decision`` (default -1)
    * correct normal  -> ``normal_correct + focus_weight * focus_reward(n_cand)``
    * correct anomaly -> ``cls_weight + iou_weight * mask_iou
                          + count_weight * count_reward(N, M)
                          + dense_weight * loc_reward
                          + refine_weight * min(0, delta_refine)``

    ``mask_iou`` is the AD-Copilot BBox-Mask IoU: the predicted and GT boxes are
    each rasterized to a binary mask and IoU is computed in mask space, so it does
    not pair boxes nor care how many boxes the model emits (robust to irregular/
    disconnected defects and to output granularity). ``count_reward`` still gives a
    direct signal for box enumeration, ``loc_reward`` is the DCLR dense term, and the
    refine term only penalizes degrading a good candidate (min(0, delta_refine)).
    ``focus_reward`` rewards localizing-then-rejecting on normal samples. Hungarian
    ``set_giou`` / ``set_iou`` / ``union_iou`` are reported as diagnostics only,
    never in reward.
    """
    validate_gt(meta)
    if not 0 <= protocol_weight <= 0.1:
        raise ValueError('protocol_weight must be in [0, 0.1]')
    loc = localization or {}
    iou_threshold = float(loc.get('iou_threshold', 0.30))
    geometry_weight = float(loc.get('geometry_weight', 0.30))
    normal_correct = float(loc.get('normal_correct', 1.0))
    wrong_decision = float(loc.get('wrong_decision', -1.0))
    cls_weight = float(loc.get('cls_weight', 0.0))
    iou_weight = float(loc.get('iou_weight', 0.5))
    count_weight = float(loc.get('count_weight', 0.2))
    dense_weight = float(loc.get('dense_weight', 0.3))
    focus_weight = float(loc.get('focus_weight', 0.2))
    refine_weight = float(loc.get('refine_weight', 0.1))
    correct = parsed['task_valid'] and parsed['is_anomaly'] == bool(meta['is_anomaly'])
    pred_px = [to_pixels(b, meta['orig_size']) for b in parsed['bboxes_2d']]
    union_iou_val = (iou(union_box(pred_px), meta.get('gt_box_px'))
                     if (meta['is_anomaly'] and pred_px) else 0.0)
    comps = meta.get('component_bboxes') or ([meta['gt_box_px']] if meta.get('gt_box_px') else [])
    mask_iou_val = mask_iou(pred_px, comps, meta['orig_size']) if meta['is_anomaly'] else 0.0
    setd = dict(reward=0.0, matched_pairs=[], s_sum=0.0)
    candd = dict(reward=0.0, matched_pairs=[], s_sum=0.0)
    raw_set_iou = 0.0
    set_giou_val = 0.0
    count_val = 0.0
    focus_val = 0.0
    if meta['is_anomaly'] and correct:
        setd = set_localization_reward(parsed['bboxes_2d'], comps, meta['orig_size'], iou_threshold, geometry_weight)
        raw_set_iou = float(component_metrics(pred_px, comps)['set_iou'])
        set_giou_val = float(set_giou(pred_px, comps))
        count_val = float(count_reward(len(pred_px), len(comps)))
    if meta['is_anomaly']:
        candd = set_localization_reward(parsed['candidate_bboxes_2d'], comps, meta['orig_size'], iou_threshold, geometry_weight)
    loc_reward = setd['reward']
    delta_refine = loc_reward - candd['reward']
    if not correct:
        task = wrong_decision
    elif not meta['is_anomaly']:
        focus_val = float(focus_reward(len(parsed['candidate_bboxes_2d'])))
        task = normal_correct + focus_weight * focus_val
    else:
        task = (cls_weight
                + iou_weight * mask_iou_val
                + count_weight * count_val
                + dense_weight * loc_reward
                + refine_weight * min(0.0, delta_refine))
    protocol_core = float(parsed['protocol_core'])
    return dict(task=task, protocol=protocol_core, protocol_core=protocol_core,
                protocol_strict=float(parsed['protocol_strict']),
                total=task + protocol_weight * protocol_core,
                loc_reward=loc_reward, set_c_reward=candd['reward'], set_f_reward=loc_reward,
                delta_refine=delta_refine,
                mask_iou=mask_iou_val, union_iou=union_iou_val,
                raw_iou=raw_set_iou, set_iou=raw_set_iou, set_giou=set_giou_val,
                count_reward=count_val, focus_reward=focus_val,
                correct=bool(correct), matched_pairs=setd['matched_pairs'], s_sum=setd['s_sum'])
