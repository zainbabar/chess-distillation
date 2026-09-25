#!/bin/bash
# Serve Qwen3-1.7B + the pilot LoRA adapters and evaluate each on the 500-puzzle test set (eval only).
# Usage: results/run_pilot_eval.sh arm1 arm2 ...
cd ~/chess-distillation
ts() { date '+%F %T'; }
mods=""
for arm in "$@"; do mods="$mods $arm=/ckpt/$arm"; done
docker rm -f student >/dev/null 2>&1
docker run -d --name student --gpus all --ipc=host -p 8001:8001 \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset \
  -e HF_HUB_OFFLINE=1 -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/chess-distillation/ckpt:/ckpt \
  nvcr.io/nvidia/vllm:26.05-py3 \
  vllm serve Qwen/Qwen3-1.7B --port 8001 --gpu-memory-utilization 0.08 --max-model-len 8192 --max-num-seqs 32 \
  --enable-lora --max-lora-rank 64 --max-loras 2 --lora-modules $mods > /dev/null
until curl -sf localhost:8001/v1/models >/dev/null; do docker ps --filter name=student -q | grep -q . || { echo "$(ts) server died"; exit 1; }; sleep 5; done
echo "$(ts) server up with: $*"
for arm in "$@"; do
  .venv/bin/python -u eval_student.py --model Qwen/Qwen3-1.7B --served-name $arm --tag pilot_$arm \
      --puzzles ${EVAL_PUZZLES:-pilot_set.jsonl} --max-tokens 1024 --temperature 0 --concurrency 32 || echo "$(ts) EVAL $arm FAILED"
done
echo "$(ts) PILOT EVAL DONE: $*"
