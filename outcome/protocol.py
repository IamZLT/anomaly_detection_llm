"""One final-answer contract for training, validation and prediction."""
from __future__ import annotations

import json
import math
import re
from typing import Optional

VERSION = 'outcome-v1'
TAGS = ('understand', 'compare', 'ground', 'verify', 'answer')
BLOCK = re.compile(r'<(understand|compare|ground|verify|answer)>(.*?)</\1>', re.S)
CAND = re.compile(r'candidate_bbox_2d\s*=\s*(null|\[[^\[\]]*\])', re.I)
VERIFY = re.compile(r'^\s*(keep|refine|reject|discover|none)\b\s*[;:,\-]?\s*(.*)$', re.I)
VERIFY_ACTIONS = ('keep', 'refine', 'reject', 'discover', 'none')


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


def parse_candidate(text: str):
    """Return (state, box) where state in {'box','null','missing','invalid'}.

    The <ground> block is a single plain line: candidate_bbox_2d=[x1,y1,x2,y2]
    or candidate_bbox_2d=null. Prose may surround it in lenient parsing.
    """
    m = CAND.search(text)
    if not m:
        return 'missing', None
    raw = m.group(1).strip()
    if raw.lower() == 'null':
        return 'null', None
    inner = raw[1:-1]
    parts = [p.strip() for p in inner.split(',')]
    if len(parts) != 4:
        return 'invalid', None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return 'invalid', None
    if valid_box(vals):
        return 'box', vals
    return 'invalid', None


def parse_verify(text: str):
    """Return (action, evidence). Action is None when no known keyword found.

    The <verify> block is 'action; evidence' (e.g. 'refine; tighten to the edge').
    """
    stripped = text.strip()
    m = VERIFY.match(stripped)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    for action in VERIFY_ACTIONS:
        if re.search(rf'\b{action}\b', text, re.I):
            return action, stripped
    return None, stripped


def parse_output(text: str) -> dict:
    """Parse the 5-block output. Only <answer> is strict JSON.

    ``task_valid`` reflects only the final answer decision + geometry.
    ``protocol_core`` is a lenient structural check (5 stages in order, candidate
    parseable, verify action parseable, valid answer) used for a weak format
    reward. ``protocol_strict`` is the canonical exact-format check, recorded for
    diagnosis only and never fed into the reward.
    """
    result = dict(task_valid=False, decision_valid=False, final_geometry_valid=False,
                  is_anomaly=None, bbox_2d=None, candidate_bbox_2d=None, candidate_state='missing',
                  verify_action=None, verify_evidence='', action=None, description='', tags={},
                  answer_keys=[], protocol_core=False, protocol_strict=False)
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
            box = obj.get('bbox_2d')
            geom = ('bbox_2d' in obj and ((pred is True and valid_box(box))
                                          or (pred is False and box is None)))
            result['final_geometry_valid'] = geom
            result['bbox_2d'] = box if valid_box(box) else None
            result['task_valid'] = result['decision_valid'] and geom
            desc = obj.get('description', '')
            result['description'] = desc if isinstance(desc, str) else ''
        except (ValueError, TypeError):
            pass

    ground_text = result['tags'].get('ground', '')
    cstate, cbox = parse_candidate(ground_text)
    result['candidate_state'] = cstate
    result['candidate_bbox_2d'] = cbox

    verify_text = result['tags'].get('verify', '')
    vaction, vevidence = parse_verify(verify_text)
    result['verify_action'] = vaction
    result['action'] = vaction
    result['verify_evidence'] = vevidence

    ordered = [m[1] for m in blocks] == list(TAGS)
    structure = (ordered
                 and not BLOCK.sub('', text).strip()
                 and all(text.count(f'<{t}>') == text.count(f'</{t}>') == 1 for t in TAGS))
    candidate_ok = cstate in ('box', 'null')
    verify_ok = vaction is not None
    understand_ok = bool((result['tags'].get('understand') or '').strip())
    compare_ok = bool((result['tags'].get('compare') or '').strip())
    desc_ok = bool(result['description'].strip())

    # Lenient: presence + order + parseable candidate/action + valid answer.
    result['protocol_core'] = bool(
        ordered and result['task_valid'] and candidate_ok and verify_ok and desc_ok)

    # Canonical exact-format: nothing outside blocks, one occurrence each, and
    # ground/verify exactly in the machine line form.
    ground_strict = bool(re.fullmatch(r'\s*candidate_bbox_2d\s*=\s*(null|\[[^\[\]]*\])\s*', ground_text, re.I))
    verify_strict = bool(re.fullmatch(r'\s*(keep|refine|reject|discover|none)\s*;\s*\S.*', verify_text, re.I))
    answer_strict = (result['task_valid'] and set(result['answer_keys']) == {'is_anomaly', 'bbox_2d', 'description'}
                     and desc_ok)
    result['protocol_strict'] = bool(
        structure and result['task_valid'] and candidate_ok and verify_ok
        and ground_strict and verify_strict and answer_strict
        and understand_ok and compare_ok)
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


def _loc_empty():
    return dict(loc_reward=0.0, raw_iou=0.0, s_center=0.0, s_w=0.0, s_h=0.0, s_geo=0.0)


def localization_reward(pred_box, gt_box, orig_size, iou_threshold=0.30, geometry_weight=0.30):
    """DCLR-style dense localization, active only below an IoU threshold.

    Returns a dict of all internal components so the training loop can log them
    and distinguish a strong center signal from a vanishing width/height match.
    """
    if pred_box is None or gt_box is None:
        return _loc_empty()
    pred_px = to_pixels(pred_box, orig_size)
    if pred_px is None:
        return _loc_empty()
    iou_val = iou(pred_px, gt_box)
    w, h = orig_size[0], orig_size[1]
    img_diag = math.hypot(w, h)
    pcx = (pred_px[0] + pred_px[2]) / 2.0
    pcy = (pred_px[1] + pred_px[3]) / 2.0
    gcx = (gt_box[0] + gt_box[2]) / 2.0
    gcy = (gt_box[1] + gt_box[3]) / 2.0
    dist = math.hypot(pcx - gcx, pcy - gcy)
    s_center = 1.0 - min(1.0, dist / (img_diag + 1e-6))
    pw = pred_px[2] - pred_px[0]
    ph = pred_px[3] - pred_px[1]
    gw = gt_box[2] - gt_box[0]
    gh = gt_box[3] - gt_box[1]
    s_w = min(pw, gw) / max(pw, gw) if pw > 0 and gw > 0 else 0.0
    s_h = min(ph, gh) / max(ph, gh) if ph > 0 and gh > 0 else 0.0
    s_geo = s_center * s_w * s_h
    if iou_val >= iou_threshold:
        reward = iou_val
    else:
        reward = iou_val + geometry_weight * (1.0 - iou_val) * s_geo
    return dict(loc_reward=float(max(0.0, min(1.0, reward))), raw_iou=float(iou_val),
                s_center=float(s_center), s_w=float(s_w), s_h=float(s_h), s_geo=float(s_geo))


def score_output(parsed, meta, protocol_weight=0.01, localization=None):
    """Reward plumbing.

    R_task = -1 (wrong/invalid), 0 (correct normal), R_loc (correct anomaly).
    R_total = R_task + protocol_weight * protocol_core.
    protocol_strict is recorded but never enters the reward.
    """
    validate_gt(meta)
    if not 0 <= protocol_weight <= 0.1:
        raise ValueError('protocol_weight must be in [0, 0.1]')
    loc = localization or {}
    iou_threshold = float(loc.get('iou_threshold', 0.30))
    geometry_weight = float(loc.get('geometry_weight', 0.30))
    correct = parsed['task_valid'] and parsed['is_anomaly'] == bool(meta['is_anomaly'])
    raw_iou = iou(to_pixels(parsed['bbox_2d'], meta['orig_size']), meta.get('gt_box_px'))
    locd = localization_reward(parsed['bbox_2d'], meta.get('gt_box_px'), meta['orig_size'],
                               iou_threshold, geometry_weight)
    loc_reward = locd['loc_reward'] if (meta['is_anomaly'] and correct) else 0.0
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
                iou=raw_iou if correct and meta['is_anomaly'] else 0.0,
                raw_iou=raw_iou, loc_reward=loc_reward, correct=bool(correct),
                s_center=locd['s_center'], s_w=locd['s_w'], s_h=locd['s_h'], s_geo=locd['s_geo'])


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
candidate_bbox_2d=[x1,y1,x2,y2]
</ground>
<verify>
action; brief reference-based evidence
</verify>
<answer>
{{"is_anomaly": false, "bbox_2d": null, "description": "brief result"}}
</answer>
Replace examples with your observations.
candidate_bbox_2d is a provisional box in [0,1000], or null if nothing suspicious.
Verification action is one of: keep, refine, reject, discover, none, followed by a semicolon and a short evidence.
For anomaly=true, bbox_2d is the single union box covering ALL defects; for anomaly=false, bbox_2d MUST be null.
Coordinates are integers [x1,y1,x2,y2] in [0,1000] for Image 2 FULL IMAGE, with x1<x2 and y1<y2.
Do not repeat blocks. Stop immediately after </answer>.'''
