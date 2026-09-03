# Prior-Guided Process-Aware Spatial GRPO

Industrial anomaly **localization** with a frozen Qwen3.5 vision tower, a training-free multi-layer difference prior, and LoRA GRPO on Grounded Comparative CoT.

Official protocol: **VisA train → MVTec test**.

## Method (code ↔ paper)

| Paper | Code |
| --- | --- |
| Visual difference prior (layers 12/16/20/24, per-layer NN, softmax fusion of distance maps) | `models/anomaly_prior.py` |
| Qwen3.5 load + freeze vision | `models/qwen35.py` |
| Language-side LoRA | `models/lora.py` |
| Grounded Comparative CoT parser | `reasoning/parser.py` |
| Process / outcome rewards \(R_{ground}, R_{reason}, R_{cls}, R_{box}\) | `reasoning/rewards.py` |
| Segment-level credit assignment | `reasoning/segments.py` |
| Clipped GRPO + KL | `rl/grpo.py` |
| Rollout → reward → optimize | `rl/trainer.py` |
| VisA / MVTec scan + dataset | `data/scan.py`, `data/prior_dataset.py` |

## Layout

```
anomaly_detection_llm/
├── configs/qwen35_2b_grpo.yaml
├── data/{scan,prior_dataset}.py
├── models/{qwen35,lora,anomaly_prior}.py
├── reasoning/{parser,rewards,segments}.py
├── rl/{grpo,trainer}.py
├── evaluation/{evaluator,metrics,infer}.py
├── visualization/tensorboard.py
├── utils/{common,config}.py          # shared helpers only
├── train.py
├── evaluate.py
├── demo.py
├── README.md
└── requirements.txt
```

Optional: `app/` (Flask UI), `scripts/eval_qwen35_9b_zeroshot.py` (vanilla 9B baseline).

## Usage

Train (2 GPUs; `--num-gpu` must equal visible GPU count):

```bash
CUDA_VISIBLE_DEVICES=2,7 python train.py --config configs/qwen35_2b_grpo.yaml --num-gpu 2
```

Evaluate a GRPO checkpoint on MVTec test:

```bash
python evaluate.py --config configs/qwen35_2b_grpo.yaml --ckpt outputs/train/qwen35_2b_prior/<run>/grpo_final
```

Single-image demo:

```bash
python demo.py --model_path outputs/train/qwen35_2b_prior/<run>/grpo_final --image_path /path/to/image.png
```

TensorBoard:

```bash
tensorboard --logdir ./outputs/train/qwen35_2b_prior --port 5002 --host 0.0.0.0
```

## Data

- Train: VisA (`data.train_layout: visa`)
- Test: MVTec test (`data.eval_layout: mvtec`)
- Each prompt is a triple: normal reference, test image, prior heatmap H

Checkpoints, logs, and backups live under `outputs/` (gitignored).
