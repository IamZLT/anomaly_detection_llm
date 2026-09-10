#!/bin/bash
# Run region-SFT checkpoint eval in parallel across free GPUs.
# GPU1 hosts 2 processes (1000, 2000), GPU3 hosts 1 (3000);
# final relays onto GPU1 as soon as 1000 finishes.
set -u
cd /data2/zlt/anomaly_detection_llm || exit 1

CFG=configs/qwen35_2b_outcome.yaml
SFT=outputs/train/region_sft
OUT=outputs/train/region_sft/_checkpoint_eval_mvtec
mkdir -p "$OUT"

run() {
  local gpu="$1" name="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/eval_sft_checkpoints.py \
    --config "$CFG" --sft-dir "$SFT" --split test --per-class 8 --save-viz \
    --checkpoints "$name" --out "$OUT/run_${name}" \
    > "$OUT/run_${name}.log" 2>&1
}

run 1 checkpoint-1000 &
P1=$!
run 1 checkpoint-2000 &
P2=$!
run 3 checkpoint-3000 &
P3=$!

# final waits for the first GPU1 slot to free up
wait "$P1"
run 1 final &
P4=$!

wait "$P2" "$P3" "$P4"
echo "ALL DONE"
