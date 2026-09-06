"""Quantify how much the single union-box GT hurts localization metrics.

The current GT for an anomaly image is ONE bounding box that spans ALL
foreground pixels in the mask (``extract_bbox_from_mask`` uses global
min/max over rows/cols). When a mask contains several disconnected defect
components, this union box covers the empty gaps too, which:

  * inflates the "defect area" (corrupting small/medium/large size bins),
  * lowers the max achievable IoU for any single-box prediction,
  * depresses iou_h_top1/iou_h_bestk (H proposes per-component boxes).

This script reports, for VisA (train) and MVTec (eval):

  * component-count distribution,
  * union-box area inflation (vs sum-of-components and vs largest component),
  * the "single-defect oracle" ceiling: IoU(largest_component, union_box),
  * how much size_bin shifts when switching from union to largest component.

Usage:  python scripts/diagnose_gt_multibox.py
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image
from scipy import ndimage


def collect_masks(root: str, kind: str):
    paths = []
    sep = os.sep
    for dp, dn, fn in os.walk(root):
        norm = dp.replace(sep, "/")
        if kind == "visa":
            # VisA: {cls}/Masks/Anomaly/*.png (masks directly inside "Anomaly")
            if os.path.basename(dp) == "Anomaly":
                for f in fn:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        paths.append(os.path.join(dp, f))
        else:
            # MVTec: {cls}/ground_truth/{defect}/*_mask.png (one level deeper)
            if "/ground_truth/" in norm:
                for f in fn:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        paths.append(os.path.join(dp, f))
    return paths


def analyze(mask_path: str):
    m = np.array(Image.open(mask_path).convert("L")) > 0
    if not m.any():
        return None
    H, W = m.shape
    lbl, n = ndimage.label(m, structure=np.ones((3, 3)))
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    rows = np.any(m, axis=1)
    cols = np.any(m, axis=0)
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    union = (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)
    return dict(n=n, boxes=boxes, areas=areas, union=union, H=H, W=W, npix=int(m.sum()))


def box_area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(x1 - x0, 0) * max(y1 - y0, 0)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def size_bin(area_frac):
    return "small" if area_frac < 0.02 else "medium" if area_frac < 0.10 else "large"


def report(name, paths):
    dist = {}
    total = 0
    n_multi = 0
    inflate_sum = []
    inflate_largest = []
    oracle_ious = []          # IoU(largest_component, union)
    best_single_ious = []     # max_i IoU(component_i, union)
    bin_shifts = 0
    largest_cov = []          # largest component pixels / total defect pixels
    for p in paths:
        r = analyze(p)
        if r is None:
            continue
        total += 1
        n = r["n"]
        dist[n] = dist.get(n, 0) + 1
        if n >= 2:
            n_multi += 1
            ua = box_area(r["union"])
            sum_areas = sum(r["areas"])
            largest = max(r["boxes"], key=box_area)
            la = box_area(largest)
            inflate_sum.append(ua / sum_areas if sum_areas else 1.0)
            inflate_largest.append(ua / la if la else 1.0)
            oracle_ious.append(iou(largest, r["union"]))
            best_single_ious.append(max(iou(b, r["union"]) for b in r["boxes"]))
            # size_bin under union GT vs largest-component GT
            img_area = r["H"] * r["W"]
            if size_bin(ua / img_area) != size_bin(la / img_area):
                bin_shifts += 1
            # largest component coverage of defect mass
            largest_mask = ndimage.label(np.array(Image.open(p).convert("L")) > 0,
                                         structure=np.ones((3, 3)))[0]
            largest_cov.append(int((largest_mask == 1).sum()) / r["npix"])

    print(f"\n===== {name} =====")
    print(f"  anomaly masks: {total}")
    print(f"  multi-component (>=2): {n_multi} ({n_multi/max(total,1)*100:.1f}%)")
    print(f"  component-count dist: {dict(sorted(dist.items(), key=lambda x:-x[1])[:8])}")
    if n_multi:
        def stats(xs):
            xs = np.array(xs)
            return f"mean={xs.mean():.2f}  median={np.median(xs):.2f}  p90={np.percentile(xs,90):.2f}  max={xs.max():.2f}"
        print(f"  [multi-comp] union_area / sum_component_area:   {stats(inflate_sum)}")
        print(f"  [multi-comp] union_area / largest_component:    {stats(inflate_largest)}")
        print(f"  [multi-comp] IoU(largest_comp, union GT):       {stats(oracle_ious)}")
        print(f"  [multi-comp] best single-comp IoU vs union GT:  {stats(best_single_ious)}")
        print(f"  [multi-comp] largest-comp defect-mass coverage: {stats(largest_cov)}")
        print(f"  [multi-comp] size_bin changes (union->largest): {bin_shifts}/{n_multi} ({bin_shifts/n_multi*100:.1f}%)")
        # how many multi-comp samples have a very low oracle ceiling
        low = sum(1 for v in oracle_ious if v < 0.3)
        print(f"  [multi-comp] samples where a single largest-defect box can NEVER exceed IoU 0.3 vs union GT: {low}/{n_multi} ({low/n_multi*100:.1f}%)")


if __name__ == "__main__":
    report("VisA (train)", collect_masks("/data2/zlt/anomaly_detection_llm/datasets/VisA", "visa"))
    report("MVTec (eval)", collect_masks("/data2/zlt/anomaly_detection_llm/datasets/mvtec_anomaly_detection", "mvtec"))
