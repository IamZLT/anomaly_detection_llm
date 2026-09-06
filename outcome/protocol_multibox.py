"""outcome-multibox-v1: multi-box detection protocol and set-level reward.

Keeps the five-stage semantics but only <answer> is strict JSON; <ground> and
<verify> are plain text lines to lower protocol syntax entropy for direct GRPO.
Reuses the DCLR localization term and geometry helpers from ``outcome.protocol``
and the Hungarian/metrics helpers from ``outcome.metrics``.
"""
from __future__ import annotations

import json
import re

from outcome.metrics import hungarian_matching, union_box
from outcome.protocol import iou, load_object, localization_reward, to_pixels, valid_box

VERSION = 'outcome-multibox-v1'
TAGS = ('understand', 'compare', 'ground', 'verify', 'answer')
BLOCK = re.compile(r'<(understand|compare|ground|verify|answer)>(.*?)</\1>', re.S)
VERIFY = re.compile(r'^\s*(keep|refine|reject|discover|none)\b\s*[;:,\-]?\s*(.*)$', re.I)
VERIFY_ACTIONS = ('keep', 'refine', 'reject', 'discover', 'none')

DEFAULT_MAX_BOXES = 16  # VisA-train p95=13 (frozen), see data.scan.compute_max_boxes


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
                    geom = (isinstance(boxes_raw, list) and len(boxes_raw) >= 1
                            and all(valid_box(b) for b in boxes_raw))
                    result['bboxes_2d'] = [b for b in boxes_raw] if geom else []
                else:
                    geom = isinstance(boxes_raw, list) and len(boxes_raw) == 0
                    result['bboxes_2d'] = []
                result['final_geometry_valid'] = bool(geom)
                result['task_valid'] = result['decision_valid'] and bool(geom)
            desc = obj.get('description', '')
            result['description'] = desc if isinstance(desc, str) else ''
        except (ValueError, TypeError):
            pass
    result['bboxes_2d'] = result['bboxes_2d'][:max_boxes]
    result['num_boxes'] = len(result['bboxes_2d'])

    ground_text = result['tags'].get('ground', '')
    cstate, cboxes = parse_boxes_list(ground_text)
    result['candidate_state'] = cstate
    result['candidate_bboxes_2d'] = cboxes[:max_boxes]

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
    """R_task = -1 (wrong/invalid) | 0 (correct normal) | R_set (correct anomaly).

    Union IoU is reported for the official benchmark but never enters the reward.
    """
    validate_gt(meta)
    if not 0 <= protocol_weight <= 0.1:
        raise ValueError('protocol_weight must be in [0, 0.1]')
    loc = localization or {}
    iou_threshold = float(loc.get('iou_threshold', 0.30))
    geometry_weight = float(loc.get('geometry_weight', 0.30))
    correct = parsed['task_valid'] and parsed['is_anomaly'] == bool(meta['is_anomaly'])
    pred_px = [to_pixels(b, meta['orig_size']) for b in parsed['bboxes_2d']]
    union_iou_val = (iou(union_box(pred_px), meta.get('gt_box_px'))
                     if (meta['is_anomaly'] and pred_px) else 0.0)
    comps = meta.get('component_bboxes') or ([meta['gt_box_px']] if meta.get('gt_box_px') else [])
    setd = dict(reward=0.0, matched_pairs=[], s_sum=0.0)
    candd = dict(reward=0.0, matched_pairs=[], s_sum=0.0)
    if meta['is_anomaly'] and correct:
        setd = set_localization_reward(parsed['bboxes_2d'], comps, meta['orig_size'], iou_threshold, geometry_weight)
    if meta['is_anomaly']:
        candd = set_localization_reward(parsed['candidate_bboxes_2d'], comps, meta['orig_size'], iou_threshold, geometry_weight)
    loc_reward = setd['reward']
    if not correct:
        task = -1.0
    elif not meta['is_anomaly']:
        task = 0.0
    else:
        task = loc_reward
    protocol_core = float(parsed['protocol_core'])
    return dict(task=task, protocol=protocol_core, protocol_core=protocol_core,
                protocol_strict=float(parsed['protocol_strict']),
                total=task + protocol_weight * protocol_core,
                loc_reward=loc_reward, set_c_reward=candd['reward'], set_f_reward=loc_reward,
                delta_refine=loc_reward - candd['reward'],
                union_iou=union_iou_val, raw_iou=union_iou_val,
                correct=bool(correct), matched_pairs=setd['matched_pairs'], s_sum=setd['s_sum'])


def prompt(class_name: str, roi: bool) -> str:
    return f'''Image 1 is a defect-free reference of {class_name}. Image 2 is the inspection image.
H contains coarse discrepancy proposals, not labels or anomaly probabilities. H may be empty or wrong.
Compare the images; reject normal variations and search outside H too.
{('Image 3, if supplied, is a crop from the ORIGINAL inspection image. Its full-image bounds are given in roi. The reference is not registered: do not assume matching pixel positions.' if roi else 'Use the two full images to check candidate regions.')}
Return these five SHORT blocks, in this exact order:
<understand>
brief object/structure observation
</understand>
<compare>
brief reference-test difference
</compare>
<ground>
candidate_bboxes_2d=[[x1,y1,x2,y2],[x1,y1,x2,y2]]
</ground>
<verify>
action; brief reference-based evidence
</verify>
<answer>
{{"is_anomaly": false, "bboxes_2d": [], "description": "brief result"}}
</answer>
Replace examples with your observations.
candidate_bboxes_2d is a provisional list of boxes, or [] if nothing suspicious. It is a hypothesis set.
Verification action is one of: keep, refine, reject, discover, none, followed by a semicolon and short evidence.
For anomaly=true, bboxes_2d lists every detected defect (one box per defect, no duplicates).
For anomaly=false, bboxes_2d MUST be [].
All boxes are integers [x1,y1,x2,y2] in [0,1000] for Image 2 FULL IMAGE, with x1<x2 and y1<y2.
Do not repeat blocks. Stop immediately after </answer>.'''
