#!/bin/bash
# Student pilot (strategy work 2026-09-24): LoRA r64 on Qwen3-1.7B, same 2k puzzles, different training text.
# Usage: results/run_pilot_train.sh arm1 arm2 ...   (arm = results/sft/<arm>.jsonl -> ckpt/<arm>)
cd ~/chess-distillation
ts() { date '+%F %T'; }
for arm in "$@"; do
  if [ -f ckpt/$arm/adapter_model.safetensors ]; then echo "$(ts) $arm already trained"; continue; fi
  echo "$(ts) TRAIN $arm"
  docker exec trainenv python -u train_sft.py --data results/sft/$arm.jsonl --out ckpt/$arm \
      --model Qwen/Qwen3-1.7B --epochs 1 --batch-tokens 8192 --accum 4 --no-grad-ckpt || echo "$(ts) TRAIN $arm FAILED"
done
echo "$(ts) PILOT TRAIN DONE: $*"
