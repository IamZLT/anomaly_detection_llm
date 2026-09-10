#!/usr/bin/env python3
"""outcome-multibox-v1: multi-box detection, train / eval / predict."""
from __future__ import annotations

import os

# Disable the NCCL heartbeat monitor (torch>=2.4) before importing torch.
# During mid-training evaluation only the main process runs `evaluate` (which can
# take tens of minutes over a large stratified sample) while the other ranks block
# at dist.barrier(); the heartbeat monitor otherwise aborts those idle ranks with
# SIGABRT after ~480s of no collective progress.
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from outcome.engine_multibox import datasets, load_model, run_train, validate_config
from outcome.evaluate_multibox import evaluate
from outcome.inputs_multibox import OutcomeMultiboxCollator
from outcome.policy import generate_group
from outcome.protocol_multibox import VERSION, parse_output, to_pixels
from rl.grpo import move_batch
from utils.common import is_main_process, set_seed
from utils.config import load_yaml_config


def _in_distributed_worker() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


def _maybe_relaunch_multi_gpu_train(cfg: dict) -> None:
    """Re-run this script under torchrun when distributed.num_gpu > 1.

    Only the outermost (non-worker) process relaunches; each spawned worker then
    sees LOCAL_RANK set and proceeds directly into the single-process code path.
    """
    num_gpu = int(cfg.get("distributed", {}).get("num_gpu", 1))
    if num_gpu <= 1 or _in_distributed_worker():
        return
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(
        f"[multibox] relaunch check: distributed.num_gpu={num_gpu} cuda_count={cuda_count} "
        f"CUDA_VISIBLE_DEVICES={cvd}",
        flush=True,
    )
    script = os.path.abspath(sys.argv[0])
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpu}",
        script,
        *sys.argv[1:],
    ]
    print(f"[multibox] distributed.num_gpu={num_gpu}，正在启动: {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.call(cmd))


def start_tensorboard(logdir: Path, cfg: dict) -> None:
    tb_cfg = cfg.get('tensorboard') or {}
    port = int(tb_cfg.get('port', 5003))
    host = str(tb_cfg.get('host', '0.0.0.0'))
    logdir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.Popen(
            [sys.executable, '-m', 'tensorboard.main',
             '--logdir', str(logdir), '--port', str(port), '--host', host],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f'[tensorboard] 已启动 pid={proc.pid} http://{host}:{port} (logdir={logdir})', flush=True)
    except Exception as exc:
        print(f'[tensorboard] 启动失败: {exc}', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='configs/qwen35_2b_outcome_multibox.yaml')
    parser.add_argument('--mode', choices=['train','eval','predict'], default='train')
    parser.add_argument('--split', choices=['dev','test'], default='dev')
    parser.add_argument('--adapter', help='Evaluation/prediction LoRA; not an optimizer resume')
    parser.add_argument('--num-gpu', type=int, default=1)
    parser.add_argument('--max-attempts', type=int)
    parser.add_argument('--eval-limit', type=int, help='Diagnostic subset; omitted means complete split')
    parser.add_argument('--output-dir')
    parser.add_argument('--image')
    parser.add_argument('--reference')
    parser.add_argument('--class-name', default='object')
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    cfg.setdefault('distributed', {})['num_gpu'] = args.num_gpu
    if args.adapter and args.mode == 'train':
        parser.error('--adapter is evaluation only; use outcome.sft_adapter for an SFT reference')
    if args.eval_limit is not None and args.eval_limit <= 0:
        parser.error('--eval-limit must be positive')
    if args.max_attempts is not None:
        cfg['grpo']['max_attempts'] = args.max_attempts
    if args.eval_limit is not None:
        cfg['training']['eval_num_samples'] = args.eval_limit
        cfg['training']['final_eval_num_samples'] = args.eval_limit
    if args.mode == 'train':
        _maybe_relaunch_multi_gpu_train(cfg)
    validate_config(cfg)
    set_seed(int(cfg['training']['seed']))
    main_proc = is_main_process()
    name = f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output = Path(args.output_dir) if args.output_dir else Path(cfg['paths']['output_dir'])/name
    if main_proc:
        output.mkdir(parents=True, exist_ok=False)
        (output/'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    root = Path(__file__).resolve().parent
    sources = list((root/'outcome').glob('*.py')) + [root/'train_outcome_multibox.py']
    sources += [root/p for p in ['rl/grpo.py','models/qwen35.py','models/vision_cache.py','models/anomaly_prior.py','data/scan.py','data/prior_dataset.py']]
    if main_proc:
        (output/'provenance.json').write_text(json.dumps(dict(protocol_version=VERSION,
            source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            reference_policy='merged_sft' if cfg['outcome'].get('sft_adapter') else 'frozen_base',
            adapter=args.adapter, mode=args.mode, split=args.split, eval_limit=args.eval_limit), indent=2))
    model, processor, prior = load_model(cfg, args.adapter)
    if args.mode == 'predict':
        if not args.image or not args.reference:
            parser.error('predict requires --image and --reference')
        if Path(args.image).resolve() == Path(args.reference).resolve():
            parser.error('reference and inspection image must differ')
        test = Image.open(args.image).convert('RGB')
        item = dict(test=test, ref=Image.open(args.reference).convert('RGB'), image_path=args.image,
                    ref_path=args.reference, orig_size=test.size, class_name=args.class_name,
                    gt_box_px=None, component_bboxes=[], num_components=0,
                    mask_area_fraction=0.0, union_area_fraction=0.0, is_anomaly=False, defect_type=None)
        max_boxes = int(cfg['outcome'].get('max_boxes', 16))
        batch = move_batch(OutcomeMultiboxCollator(processor, prior, cfg)([item]), next(model.parameters()).device)
        completion = generate_group(model, processor, batch, cfg)[0]
        result = parse_output(completion.text, max_boxes=max_boxes)
        result.update(bboxes_original_px=[to_pixels(b, test.size) for b in result['bboxes_2d']],
                      text=completion.text, stop_reason=completion.stop_reason, input=batch['_meta'][0])
        result['input'].pop('is_anomaly', None); result['input'].pop('gt_box_px', None)
        (output/'prediction.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        train_set, dev_set, test_set = datasets(cfg, processor)
        if args.mode == 'eval':
            selected = dev_set if args.split == 'dev' else test_set
            stats = evaluate(cfg, model, processor, prior, selected, output/f'{args.split}.json', args.eval_limit, namespace=args.split)
            if main_proc:
                print(json.dumps(stats, ensure_ascii=False, indent=2))
        else:
            if main_proc:
                start_tensorboard(output / 'tb', cfg)
            run_train(cfg, model, processor, prior, train_set, dev_set, test_set, output)
    if main_proc:
        print(f'Output: {output}', flush=True)


if __name__ == '__main__':
    main()
