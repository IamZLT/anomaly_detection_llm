#!/usr/bin/env python3
"""
在 conda 环境 `clip` 下测试 Qwen3-VL 与本地 DINOv3 的配置维度，便于设计 DINO -> MLP/Linear -> Qwen 的映射。

用法（在项目根目录）:
  conda activate clip
  cd /data2/zlt/anomaly_detection_llm
  python scripts/test_qwen_dino_dims.py

可选:
  python scripts/test_qwen_dino_dims.py --smoke-dino          # 加载 DINO，随机图前向，打印 CLS/patch 形状
  python scripts/test_qwen_dino_dims.py --load-processor    # 仅加载 Qwen AutoProcessor（校验路径）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def print_qwen_dims(qwen_config_path: Path) -> None:
    cfg = _load_json(qwen_config_path)
    text = cfg.get("text_config") or {}
    vision = cfg.get("vision_config") or {}
    print("=== Qwen3-VL (来自 config.json) ===")
    print(f"  路径: {qwen_config_path}")
    print(f"  text_config.hidden_size (LLM 隐层 / 词嵌入维度): {text.get('hidden_size')}")
    print(f"  vision_config.hidden_size (视觉 backbone 内部维): {vision.get('hidden_size')}")
    print(f"  vision_config.out_hidden_size (视觉到多模态融合输出维): {vision.get('out_hidden_size')}")
    print(f"  vision_config.patch_size: {vision.get('patch_size')}")
    print()
    print("  设计 DINO 特征接入 Qwen 时，常见目标是把 token 映射到 **text_config.hidden_size**")
    print(f"  （本机为 {text.get('hidden_size')}），与多模态序列在同一向量空间对齐。")
    print()


def print_dino_dims(dino_config_path: Path) -> None:
    cfg = _load_json(dino_config_path)
    print("=== DINOv3 (来自 config.json) ===")
    print(f"  路径: {dino_config_path}")
    print(f"  hidden_size (CLS / patch token 维): {cfg.get('hidden_size')}")
    print(f"  patch_size: {cfg.get('patch_size')}")
    print(f"  num_register_tokens: {cfg.get('num_register_tokens')}")
    print(f"  image_size (processor 默认): {cfg.get('image_size')}")
    print()


def print_projection_hint(dino_h: int | None, qwen_text_h: int | None) -> None:
    if dino_h is None or qwen_text_h is None:
        return
    print("=== 映射提示（示例） ===")
    print(f"  若用 DINOv3 的 CLS 或 patch 维 {dino_h}，可接:")
    print(f"    nn.Linear({dino_h}, {qwen_text_h})  或  MLP({dino_h} -> {qwen_text_h})")
    print("  再与 Qwen 的 input embeddings / 视觉 token 融合方式需结合模型 forward 自定义。")
    print()


def smoke_dino(dino_dir: Path, device: str) -> None:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    print("=== DINOv3 冒烟（前向一张随机图） ===")
    proc = AutoImageProcessor.from_pretrained(str(dino_dir))
    model = AutoModel.from_pretrained(str(dino_dir))
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    # 224x224 与 config 一致即可
    from PIL import Image
    import numpy as np

    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    batch = proc(images=pil, return_tensors="pt")
    batch = {k: v.to(dev) for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)
    last = out.last_hidden_state
    b, t, d = last.shape
    print(f"  last_hidden_state: [{b}, {t}, {d}]  (batch, tokens, dim)")
    print(f"  CLS 建议取 [:, 0, :] 形状: [{b}, {d}]")
    num_reg = getattr(model.config, "num_register_tokens", 0)
    patch_tokens = t - 1 - num_reg
    print(f"  register tokens: {num_reg}, patch 数约: {patch_tokens}")
    print()


def load_qwen_processor_only(qwen_dir: Path, local_files_only: bool) -> None:
    print("=== Qwen AutoProcessor 加载测试 ===")
    from transformers import AutoProcessor

    p = AutoProcessor.from_pretrained(
        str(qwen_dir),
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local_files_only,
    )
    print(f"  processor 类型: {type(p).__name__}")
    if hasattr(p, "tokenizer"):
        print(f"  tokenizer 词表大小: {getattr(p.tokenizer, 'vocab_size', 'N/A')}")
    print("  OK")
    print()


def main() -> int:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Qwen3-VL + DINOv3 维度与冒烟测试（clip 环境）")
    parser.add_argument(
        "--qwen-dir",
        type=str,
        default=str(root / "model_card" / "Qwen3-VL-8B-Instruct"),
        help="Qwen3-VL 本地目录（含 config.json）",
    )
    parser.add_argument(
        "--dino-dir",
        type=str,
        default=str(root / "model_card" / "dinov3-vitl16-pretrain-lvd1689m"),
        help="DINOv3 本地目录（含 config.json）",
    )
    parser.add_argument("--load-processor", action="store_true", help="加载 Qwen AutoProcessor（不加载大模型权重）")
    parser.add_argument("--smoke-dino", action="store_true", help="DINO 随机图前向")
    parser.add_argument("--device", type=str, default="cuda" if _cuda_available() else "cpu", help="smoke-dino 设备")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Processor 仅本地文件")
    args = parser.parse_args()

    qwen_config = Path(args.qwen_dir) / "config.json"
    dino_config = Path(args.dino_dir) / "config.json"
    if not qwen_config.is_file():
        print(f"未找到: {qwen_config}", file=sys.stderr)
        return 1
    if not dino_config.is_file():
        print(f"未找到: {dino_config}", file=sys.stderr)
        return 1

    q_cfg = _load_json(qwen_config)
    d_cfg = _load_json(dino_config)
    print_qwen_dims(qwen_config)
    print_dino_dims(dino_config)
    print_projection_hint(d_cfg.get("hidden_size"), (q_cfg.get("text_config") or {}).get("hidden_size"))

    if args.load_processor:
        load_qwen_processor_only(Path(args.qwen_dir), args.local_files_only)

    if args.smoke_dino:
        smoke_dino(Path(args.dino_dir), args.device)

    return 0


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
