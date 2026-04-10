#!/usr/bin/env python3
"""
检查 datasets/mvtec_zero_shot.json 中是否存在「要求 GPT 输出 JSON 框 / bbox」的对话，
以及 GPT 回复里是否真出现 bbox 类 JSON。

用法:
  python scripts/check_mvtec_json_bbox_in_conversations.py
  python scripts/check_mvtec_json_bbox_in_conversations.py --json /path/to/mvtec_zero_shot.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple


def _turn_text(conv: Dict[str, Any]) -> str:
    return str(conv.get("value") or conv.get("content") or "").strip()


def _role(conv: Dict[str, Any]) -> str:
    r = conv.get("from") or conv.get("role") or ""
    return str(r).lower()


def _is_human(conv: Dict[str, Any]) -> bool:
    return _role(conv) in ("human", "user")


def _is_gpt(conv: Dict[str, Any]) -> bool:
    return _role(conv) in ("gpt", "assistant")


# GPT 侧：像 JSON 里带 bbox 的写法（避免把 electrical "grounding" 当 bbox）
_RE_BBOX_TOKEN = re.compile(r"\bbbox\b|bbox_2d|\"bbox\"|'bbox'", re.IGNORECASE)
_RE_JSON_NUM_LIST = re.compile(
    r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"  # [x1,y1,x2,y2]
)

# Human 侧：是否在要「JSON + 定位框」类输出
_RE_ASK_JSON = re.compile(r"\bjson\b", re.IGNORECASE)
_RE_ASK_BBOX = re.compile(
    r"\bbbox\b|bounding\s*box|bbox_2d|框|坐标|coordinate(s)?\s*(of|for)?\s*(the)?\s*(anomaly|defect|region|object)",
    re.IGNORECASE,
)


def human_asks_json_bbox(text: str) -> bool:
    t = text
    if not _RE_ASK_JSON.search(t):
        return False
    if _RE_BBOX_TOKEN.search(t) or _RE_ASK_BBOX.search(t):
        return True
    return False


def gpt_looks_like_bbox_json(text: str) -> Tuple[bool, List[str]]:
    """返回 (是否命中, 命中原因标签列表)"""
    reasons: List[str] = []
    if _RE_BBOX_TOKEN.search(text):
        reasons.append("bbox_token")
    if _RE_JSON_NUM_LIST.search(text) and ("{" in text or "[" in text):
        reasons.append("quad_bracket_list")
    # 含 { ... "bbox" ... } 或类似
    if "{" in text and _RE_BBOX_TOKEN.search(text):
        reasons.append("brace_and_bbox")
    return (len(reasons) > 0, reasons)


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_json = os.path.join(root, "datasets", "mvtec_zero_shot.json")

    ap = argparse.ArgumentParser(description="检查 mvtec_zero_shot.json 是否含 JSON bbox 类对话")
    ap.add_argument(
        "--json",
        type=str,
        default=default_json,
        help=f"JSON 路径（默认: {default_json}）",
    )
    ap.add_argument(
        "--show",
        type=int,
        default=5,
        help="若有命中，最多打印几条示例（human/gpt 各侧）",
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

    n_records = len(data)
    n_human = 0
    n_gpt = 0
    human_ask_json_bbox = 0
    human_ask_json_bbox_examples: List[Tuple[str, str]] = []  # (image, snippet)

    gpt_bbox_like = 0
    gpt_bbox_examples: List[Tuple[str, str, str]] = []  # (image, reasons, text preview)

    meta_target_2d = 0
    meta_with_mask = 0

    for item in data:
        if not isinstance(item, dict):
            continue
        img = (item.get("image") or "").replace("\\", "/").strip()
        meta = item.get("metadata") or {}
        if isinstance(meta, dict):
            if str(meta.get("target", "")).upper() == "2D" or meta.get("target") == "2D":
                meta_target_2d += 1
            if meta.get("mask"):
                meta_with_mask += 1

        convs = item.get("conversations") or []
        if not isinstance(convs, list):
            continue

        for conv in convs:
            if not isinstance(conv, dict):
                continue
            text = _turn_text(conv)
            if not text:
                continue
            if _is_human(conv):
                n_human += 1
                if human_asks_json_bbox(text):
                    human_ask_json_bbox += 1
                    if len(human_ask_json_bbox_examples) < max(args.show, 8):
                        human_ask_json_bbox_examples.append(
                            (img, text[:500] + ("…" if len(text) > 500 else ""))
                        )
            elif _is_gpt(conv):
                n_gpt += 1
                ok, reasons = gpt_looks_like_bbox_json(text)
                if ok:
                    gpt_bbox_like += 1
                    if len(gpt_bbox_examples) < max(args.show, 8):
                        gpt_bbox_examples.append(
                            (
                                img,
                                ",".join(reasons),
                                text[:600] + ("…" if len(text) > 600 else ""),
                            )
                        )

    print("\n" + "=" * 72)
    print("结论摘要")
    print("=" * 72)
    print(f"JSON 记录数:              {n_records}")
    print(f"metadata 含 mask 字段:   {meta_with_mask}（有像素级 GT，不代表对话里要求输出 bbox JSON）")
    print(f"metadata.target == 2D:    {meta_target_2d}")
    print(f"human 轮次总数:           {n_human}")
    print(f"gpt 轮次总数:             {n_gpt}")
    print()
    print(f"【Human】同时提到 json 且（bbox/框/坐标类）的轮次: {human_ask_json_bbox}")
    print(f"【GPT】  回复像含 bbox/JSON 框的轮次:               {gpt_bbox_like}")

    if human_ask_json_bbox == 0 and gpt_bbox_like == 0:
        print("\n→ 在本文件的启发式扫描下：**未发现**要求 GPT 输出 JSON bbox 的对话，GPT 侧也**未出现**典型 bbox JSON 片段。")
        print("  （全文件字面量搜索「bbox」通常也为 0；若需 grounding 训练需另建带 bbox JSON 的标注/模板。）")
    else:
        print("\n→ 存在疑似相关轮次，示例见下。")

    if human_ask_json_bbox_examples and args.show > 0:
        print("\n" + "-" * 72)
        print(f"Human 示例（最多 {args.show} 条）")
        print("-" * 72)
        for i, (img, snip) in enumerate(human_ask_json_bbox_examples[: args.show], 1):
            print(f"\n[{i}] image: {img}\n{snip}\n")

    if gpt_bbox_examples and args.show > 0:
        print("-" * 72)
        print(f"GPT 示例（最多 {args.show} 条）")
        print("-" * 72)
        for i, (img, reasons, snip) in enumerate(gpt_bbox_examples[: args.show], 1):
            print(f"\n[{i}] image: {img}  tags={reasons}\n{snip}\n")

    print("=" * 72)


if __name__ == "__main__":
    main()
