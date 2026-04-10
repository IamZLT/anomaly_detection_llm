#!/usr/bin/env python3
"""
统计 datasets/mvtec_zero_shot.json 数据量（仅用标准库，可在 conda clip 环境运行）。

用法:
  python scripts/stats_mvtec_zero_shot_json.py
  python scripts/stats_mvtec_zero_shot_json.py --json /path/to/mvtec_zero_shot.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import List, Tuple


def parse_image_path(image_rel: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    mvtec_anomaly_detection/capsule/test/poke/004.png
    -> (class_name, split, defect_type)
    """
    parts = image_rel.replace("\\", "/").strip("/").split("/")
    if len(parts) < 4:
        return None, None, None
    cls_name = parts[-4]
    split = parts[-3]
    defect = parts[-2]
    return cls_name, split, defect


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_json = os.path.join(root, "datasets", "mvtec_zero_shot.json")

    ap = argparse.ArgumentParser(description="统计 mvtec_zero_shot.json")
    ap.add_argument(
        "--json",
        type=str,
        default=default_json,
        help=f"JSON 路径（默认: {default_json}）",
    )
    args = ap.parse_args()
    path = os.path.abspath(os.path.expanduser(args.json))

    if not os.path.isfile(path):
        raise SystemExit(f"文件不存在: {path}")

    print(f"读取: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit(f"期望顶层为 JSON 数组，实际: {type(data)}")

    n = len(data)
    print(f"\n{'='*60}")
    print(f"总条数（JSON 对象数）: {n}")
    print(f"{'='*60}")

    no_image = 0
    no_conv = 0
    turn_counts: List[int] = []
    by_split = Counter()
    by_class = Counter()
    by_defect = Counter()
    by_anomaly = Counter()
    by_class_split = defaultdict(Counter)  # class -> split -> count
    unique_images: set[str] = set()
    duplicate_image_records = 0  # how many extra rows for same image

    for item in data:
        if not isinstance(item, dict):
            continue
        img = (item.get("image") or "").replace("\\", "/").strip()
        if not img:
            no_image += 1
            continue
        unique_images.add(img)

        conv = item.get("conversations") or []
        if not conv:
            no_conv += 1
        turn_counts.append(len(conv))

        meta = item.get("metadata") or {}
        anom = meta.get("anomaly")
        if anom is True:
            by_anomaly["true"] += 1
        elif anom is False:
            by_anomaly["false"] += 1
        else:
            by_anomaly["missing_or_other"] += 1

        cls_name, split, defect = parse_image_path(img)
        if cls_name:
            by_class[cls_name] += 1
        if split:
            by_split[split] += 1
        if defect:
            by_defect[defect] += 1
        if cls_name and split:
            by_class_split[cls_name][split] += 1

    # 同一 image 出现多次：条数 - 唯一图数
    image_to_count = Counter()
    for item in data:
        if not isinstance(item, dict):
            continue
        img = (item.get("image") or "").replace("\\", "/").strip()
        if img:
            image_to_count[img] += 1
    multi = sum(1 for c in image_to_count.values() if c > 1)
    duplicate_image_records = sum(c - 1 for c in image_to_count.values() if c > 1)

    print(f"\n唯一 image 路径数: {len(unique_images)}")
    print(f"出现多次的 image 数: {multi}（多出来的 JSON 行数: {duplicate_image_records}）")

    print(f"\n无 image 字段或空: {no_image}")
    print(f"conversations 为空列表: {no_conv}")

    if turn_counts:
        tc = Counter(turn_counts)
        print(f"\n每样本 conversations 条数（轮次/消息数）:")
        print(f"  min={min(turn_counts)}  max={max(turn_counts)}  mean={sum(turn_counts)/len(turn_counts):.2f}")
        print(f"  分布（前 15 个频次）: {tc.most_common(15)}")

    print(f"\n按路径解析的 split（train/test/…）:")
    for k, v in sorted(by_split.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v}")

    print(f"\nmetadata.anomaly:")
    for k, v in by_anomaly.items():
        print(f"  {k}: {v}")

    print(f"\n按类别 class（路径倒数第 4 段）共 {len(by_class)} 类:")
    for k, v in sorted(by_class.items(), key=lambda x: (-x[1], x[0])):
        tr = by_class_split[k].get("train", 0)
        te = by_class_split[k].get("test", 0)
        print(f"  {k}: 总计={v} (train={tr}, test={te})")

    print(f"\n按缺陷子目录 defect（路径倒数第 2 段）Top 30:")
    for k, v in by_defect.most_common(30):
        print(f"  {k}: {v}")
    if len(by_defect) > 30:
        print(f"  … 共 {len(by_defect)} 个不同 defect 名")

    print(f"\n完成。")


if __name__ == "__main__":
    main()
