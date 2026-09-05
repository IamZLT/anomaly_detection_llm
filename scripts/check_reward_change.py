#!/usr/bin/env python3
"""Show the effect of the reward changes on a small grid of boxes (no GPU).

Compares OLD vs NEW reward geometry:
  1. dense reward plateau: gamma=1 (old) vs gamma=2 (new) across a sweep of
     boxes that all have the right center but wrong scale.
  2. R_final under the copy-shortcut: Bf=Bc (imprecise) vs Bf=exact, with the
     old weights (no raw IoU) vs the new weights (final_iou_weight present).

Usage:
    python scripts/check_reward_change.py
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reasoning.rewards import (
    box_iou,
    compute_rewards,
    dense_geometry_reward,
)
from reasoning.parser import parse_cot_output

ORIG = (1000, 1000)
GT = [450, 450, 550, 550]  # 100x100 defect near center


def _fmt_row(label, *vals):
    cells = [f"{label:<34s}"]
    for v in vals:
        cells.append(f"{v:>8}" if isinstance(v, str) else f"{v:8.3f}")
    return " ".join(cells)


def part1_dense_plateau():
    print("=" * 78)
    print("PART 1: dense_geometry_reward — does gamma break the size plateau?")
    print(f"GT = {GT}  (center 500,500; size 100x100)")
    print("=" * 78)
    # All boxes are centered on GT but have increasing size. Old reward keeps
    # them clustered high; new reward (gamma=2) collapses the wrong-size ones.
    boxes = {
        "exact (100x100)": [450, 450, 550, 550],
        "1.5x (150x150)": [425, 425, 575, 575],
        "2x (200x200)": [400, 400, 600, 600],
        "3x (300x300)": [350, 350, 650, 650],
        "5x (500x500)": [250, 250, 750, 750],
        "full image": [0, 0, 1000, 1000],
    }
    print(_fmt_row("box", "IoU", "dense(g=1)", "dense(g=2)"))
    print("-" * 78)
    for name, box in boxes.items():
        iou = box_iou(box, GT)
        d1 = dense_geometry_reward(box, GT, ORIG, gamma=1.0)
        d2 = dense_geometry_reward(box, GT, ORIG, gamma=2.0)
        print(_fmt_row(name, iou, d1, d2))
    print()
    print("Reading: gamma=2 makes a wrong-size box drop much faster than gamma=1,")
    print("so raw IoU (which is plateau-free) becomes the dominant signal once boxes")
    print("start to overlap. The full-image box is already ~0 either way.")
    print()


def part2_copy_shortcut():
    print("=" * 78)
    print("PART 2: R_final — is 'copy Bc into Bf' still free?")
    print(f"GT = {GT}")
    print("=" * 78)

    new_cfg = {
        "grpo": {
            "reward": {
                "final_iou_weight": 0.40,
                "final_dense_weight": 0.50,
                "final_edge_weight": 0.10,
                "dense_scale_gamma": 2.0,
            }
        }
    }

    def build(cand, final):
        return (
            "<understand>\nImage 1 shows an intact bottle body. Image 2 shows the same object "
            "with a localized irregular break.\n</understand>\n"
            "<compare>\nThe inspection image contains an irregular structural break that is "
            "absent from the normal reference.\n</compare>\n"
            f"<ground>\ncandidate_bbox_2d={cand}\nThe most suspicious region is the damaged body.\n</ground>\n"
            "<verify>\nThe irregular structure is a physical defect.\n</verify>\n"
            f'<answer>\n{{"is_anomaly": true, "bbox_2d": {final}, '
            '"description": "An irregular structural break is present on the bottle body."}\n</answer>\n'
        )

    # "copy" = candidate and final are both a 3x oversized box centered on GT.
    copy_text = build("[300,300,700,700]", "[300,300,700,700]")
    # "refined" = candidate imprecise, final exact.
    refined_text = build("[300,300,700,700]", "[450,450,550,550]")

    # Honest OLD formula (pre-change): R_final = 0.8 * dense(gamma=1) + 0.2 * edge.
    from reasoning.rewards import edge_precision_reward

    def old_r_final(text):
        p = parse_cot_output(text)
        final = p["bbox_2d"]
        dense = dense_geometry_reward(final, GT, ORIG, gamma=1.0)
        edge = edge_precision_reward(final, GT, ORIG, beta=8.0, min_frac=0.05)
        return 0.8 * dense + 0.2 * edge

    for label, text in [("copy (Bf=Bc, 3x box)", copy_text), ("refined (Bf=exact)", refined_text)]:
        p = parse_cot_output(text)
        new = compute_rewards(p, GT, ORIG, True, new_cfg)
        old = old_r_final(text)
        print(f"[{label}]")
        print(f"    raw_iou_f={new['raw_iou_f']:.3f}   R_dense_f(new, g=2)={new['R_dense_f']:.3f}")
        print(f"    R_final OLD (0.8*dense+0.2*edge) = {old:.3f}")
        print(f"    R_final NEW (iou+dense+edge)    = {new['R_final']:.3f}")
        print()

    old_copy = old_r_final(copy_text)
    old_refined = old_r_final(refined_text)
    new_copy = compute_rewards(parse_cot_output(copy_text), GT, ORIG, True, new_cfg)["R_final"]
    new_refined = compute_rewards(parse_cot_output(refined_text), GT, ORIG, True, new_cfg)["R_final"]

    print("Summary (R_final):")
    print(f"    OLD: copy={old_copy:.3f}  refined={old_refined:.3f}  "
          f"gap(refined-copy)={old_refined - old_copy:+.3f}")
    print(f"    NEW: copy={new_copy:.3f}  refined={new_refined:.3f}  "
          f"gap(refined-copy)={new_refined - new_copy:+.3f}")
    print()
    print("Reading: under OLD weights the copy shortcut and the true refinement are close,")
    print("so the model learns 'copy Bc' as a lazy optimum. NEW weights put raw IoU into")
    print("R_final, so a refined (higher-IoU) Bf is rewarded strictly more than copying.")
    print()


if __name__ == "__main__":
    part1_dense_plateau()
    part2_copy_shortcut()
