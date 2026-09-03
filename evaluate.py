#!/usr/bin/env python3
"""Official VisA-trained checkpoint evaluation on MVTec test."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.prior_dataset import PriorCollator, PriorCoTDataset
from data.scan import load_prior_split
from evaluation.evaluator import run_simple_eval
from models.anomaly_prior import AnomalyPrior
from models.qwen35 import setup_model_and_processor
from utils.config import apply_runtime_overrides, load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Evaluate GRPO checkpoint on MVTec")
    p.add_argument("--config", type=str, default="configs/qwen35_2b_grpo.yaml")
    p.add_argument("--ckpt", type=str, required=True, help="LoRA / GRPO 输出目录")
    p.add_argument("--num-samples", type=int, default=None, help="覆盖 training.final_eval_num_samples")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_runtime_overrides(cfg, args)
    ckpt = os.path.abspath(os.path.expanduser(args.ckpt))
    if not os.path.isdir(ckpt):
        raise FileNotFoundError(ckpt)

    _, eval_samples = load_prior_split(cfg)
    model, processor = setup_model_and_processor(
        cfg, for_inference=True, model_name_override=ckpt, freeze_vision=True
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    prior = AnomalyPrior.from_qwen(model, cfg)
    eval_set = PriorCoTDataset(eval_samples, cfg, processor, mode="eval")
    loader = DataLoader(eval_set, batch_size=1, shuffle=False, collate_fn=PriorCollator(processor, prior, cfg))
    n_max = args.num_samples
    if n_max is None:
        n_max = (cfg.get("training") or {}).get("final_eval_num_samples")
    writer = None
    out_dir = os.path.join(ckpt, "eval_mvtec")
    os.makedirs(out_dir, exist_ok=True)
    tb = os.path.join(out_dir, "tb")
    os.makedirs(tb, exist_ok=True)
    writer = SummaryWriter(log_dir=tb)
    stats = run_simple_eval(cfg, model, processor, loader, writer=writer, global_step=0, n_max=n_max)
    writer.close()
    compact = {k: v for k, v in stats.items() if k != "records"}
    print(
        f"[evaluate] n={stats['n']} rec={stats['rec_acc']:.3f} "
        f"gated_mIoU={stats.get('mean_iou_gated', 0.0):.3f} acc@0.5={stats.get('acc_at_05', 0.0):.3f} "
        f"iou03={stats['iou_at_03']:.3f} mean_iou_f={stats['mean_iou']:.3f} "
        f"mean_iou_c={stats['mean_iou_c']:.3f}",
        flush=True,
    )
    import json

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)
    print(f"[evaluate] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
