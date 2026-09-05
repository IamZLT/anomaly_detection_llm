"""One final-answer contract for training, validation and prediction."""
from __future__ import annotations

import json
import math
import re
from typing import Optional

VERSION = 'outcome-v1'
TAGS = ('understand', 'compare', 'ground', 'verify', 'answer')
BLOCK = re.compile(r'<(understand|compare|ground|verify|answer)>(.*?)</\1>', re.S)


def valid_box(box, upper=1000.0):
    return (isinstance(box, list) and len(box) == 4
            and all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= upper for v in box)
            and box[0] < box[2] and box[1] < box[3])


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f'duplicate JSON key: {key}')
        obj[key] = value
    return obj


def load_object(text):
    obj = json.loads(text, object_pairs_hook=unique_object,
                     parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    if not isinstance(obj, dict):
        raise ValueError('expected JSON object')
    return obj


def parse_output(text: str) -> dict:
    """Final fields do not depend on rationale lengths or candidate correctness.

    Exactly one closed answer is required, with no trailing non-whitespace.
    No recovery of truncated answers or ambiguous duplicate JSON fields.
    """
    result = dict(task_valid=False, protocol_valid=False, decision_valid=False,
                  final_geometry_valid=False, is_anomaly=None, bbox_2d=None,
                  candidate_bbox_2d=None, description='', action=None, tags={})
    blocks = list(BLOCK.finditer(text))
    result['tags'] = {m[1]: m[2].strip() for m in blocks}
    answers = list(re.finditer(r'<answer>(.*?)</answer>', text, re.S))
    if (len(answers) != 1 or text.count('<answer>') != 1 or text.count('</answer>') != 1
            or text[answers[0].end():].strip()):
        return result
    try:
        obj = load_object(answers[0][1])
    except (ValueError, TypeError):
        return result
    pred = obj.get('is_anomaly')
    result['decision_valid'] = type(pred) is bool
    result['is_anomaly'] = pred if type(pred) is bool else None
    box = obj.get('bbox_2d')
    geom = ('bbox_2d' in obj and ((pred is True and valid_box(box))
                                or (pred is False and box is None)))
    result['final_geometry_valid'] = geom
    result['bbox_2d'] = box if valid_box(box) else None
    result['task_valid'] = result['decision_valid'] and geom
    desc = obj.get('description', '')
    result['description'] = desc if isinstance(desc, str) else ''
    structure = ([m[1] for m in blocks] == list(TAGS)
                 and not BLOCK.sub('', text).strip()
                 and all(text.count(f'<{t}>') == text.count(f'</{t}>') == 1 for t in TAGS)
                 and all('<' not in m[2] and '>' not in m[2] for m in blocks))
    candidate_ok = verify_ok = False
    try:
        ground = load_object(result['tags'].get('ground', ''))
        c = ground.get('candidate_bbox_2d')
        candidate_ok = set(ground) == {'candidate_bbox_2d'} and (c is None or valid_box(c))
        result['candidate_bbox_2d'] = c if valid_box(c) else None
        verify = load_object(result['tags'].get('verify', ''))
        result['action'] = verify.get('action')
        verify_ok = (set(verify) == {'action', 'evidence'}
                     and verify.get('action') in ('keep', 'refine', 'reject', 'discover', 'none')
                     and isinstance(verify.get('evidence'), str) and bool(verify['evidence'].strip()))
    except (ValueError, TypeError):
        pass
    result['protocol_valid'] = bool(structure and result['task_valid'] and candidate_ok and verify_ok
        and result['tags'].get('understand') and result['tags'].get('compare')
        and isinstance(desc, str) and desc.strip()
        and set(obj) == {'is_anomaly', 'bbox_2d', 'description'})
    return result


def iou(a, b):
    if a is None or b is None:
        return 0.0
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(x1-x0, 0) * max(y1-y0, 0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/union if union > 0 else 0.0


def to_pixels(box, wh):
    if box is None:
        return None
    return [box[0]*wh[0]/1000, box[1]*wh[1]/1000,
            box[2]*wh[0]/1000, box[3]*wh[1]/1000]


def validate_gt(meta):
    w, h = meta['orig_size']
    gt = meta.get('gt_box_px')
    if w <= 0 or h <= 0:
        raise ValueError('invalid original image size')
    if meta['is_anomaly']:
        if not (valid_box(gt, max(w, h)) and gt[2] <= w and gt[3] <= h):
            raise ValueError(f"anomalous sample has missing/invalid GT: {meta.get('image_path')}")
    elif gt is not None:
        raise ValueError('normal sample must have null GT bbox')


def score_output(parsed, meta, protocol_weight=0.05):
    validate_gt(meta)
    if not 0 <= protocol_weight <= 0.1:
        raise ValueError('protocol_weight must be in [0, 0.1]')
    correct = parsed['task_valid'] and parsed['is_anomaly'] == bool(meta['is_anomaly'])
    overlap = iou(to_pixels(parsed['bbox_2d'], meta['orig_size']), meta.get('gt_box_px'))
    task = (-1.0 if not correct else overlap if meta['is_anomaly'] else 1.0)
    protocol = float(parsed['protocol_valid'])
    return dict(task=task, protocol=protocol, total=task+protocol_weight*protocol,
                iou=overlap if correct and meta['is_anomaly'] else 0.0,
                correct=bool(correct))


def prompt(class_name: str, roi: bool) -> str:
    return f'''Image 1 is a defect-free reference of {class_name}. Image 2 is the inspection image.
H contains coarse discrepancy proposals, not labels or anomaly probabilities. H may be empty or wrong.
Compare the images; reject normal variations and search outside H too.
{('Image 3, if supplied, is a crop from the ORIGINAL inspection image. Its full-image bounds are given in roi. The reference is not registered: do not assume matching pixel positions.' if roi else 'Use the two full images to check candidate regions.')}
Return these five SHORT blocks, with at most one short observation per prose field:
<understand>object and relevant structure</understand>
<compare>specific visible difference or consistency</compare>
<ground>{{"candidate_bbox_2d": null}}</ground>
<verify>{{"action": "none", "evidence": "brief visible evidence"}}</verify>
<answer>{{"is_anomaly": false, "bbox_2d": null, "description": "brief result"}}</answer>
Replace examples with your observations. Candidate is a provisional full-image box or null.
Verification action: keep, refine, reject, discover (outside the supplied candidate), or none.
ALL boxes refer to Image 2 FULL IMAGE, coordinates [x1,y1,x2,y2] in [0,1000], x1<x2 and y1<y2.
For anomaly=true, final bbox encloses ALL detected defects under the single union-box benchmark.
For anomaly=false, final bbox MUST be null. Empty H/absent crop does not establish normality.
Do not repeat blocks. Stop immediately after </answer>.'''
