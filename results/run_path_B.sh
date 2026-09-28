#!/bin/bash
# THE PATH step 2, student B (teacher explanations) — user's go 2026-09-26 ~13:00.
# Identical recipe to student A (results/run_path_A.sh): Qwen3-1.7B, full fine-tune, 2 passes, lr 1e-5 (src/train_sft.py
# default for --full), fp32 master weights + bf16 autocast, FA2, batch-tokens 8192 x accum 4, seed 0, --save-epochs.
# Same 37,543 puzzles and P1L prompt as A; only the target differs (explanation + FINAL_LINE + FINAL_MOVE).
# Then: eval the final model and the pass-1 checkpoint on the 500 test puzzles (max-tokens 1024, greedy, as A), and
# claim-check B's own explanations (src/student_explain_check.py). Teacher stays STOPPED.
# 14:00 restart: run 1 (log results/path_B.log.run1_fragmentation) grew 94.8 -> 99.3 GB in 1 h (allocator fragmentation;
# free memory 19 -> 14 GiB); now PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (allocator only, same math).
# Launch: (nohup results/run_path_B.sh >> results/path_B.log 2>&1 &)
cd ~/chess-distillation
PY=.venv/bin/python
ts() { date '+%F %T'; }
has_model() { ls $1/*.safetensors > /dev/null 2>&1; }
memgb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
echo "$(ts) PATH_B START ($(wc -l < results/sft/path_B.jsonl) examples)"

# never train while the teacher serves
if docker ps --format '{{.Names}}' | grep -qx gptoss; then
  docker stop gptoss > /dev/null && echo "$(ts) teacher stopped"
fi
docker rm -f student > /dev/null 2>&1
echo "$(ts) MemAvailable $(memgb) GiB"

# memory watchdog for training: restart trainenv if < 8 GiB free
( minm=999; while ps -p $$ > /dev/null; do m=$(memgb); [ $m -lt $minm ] && { minm=$m; echo $minm > results/path_B.minmem; }
    if [ $m -lt 8 ]; then echo "$(ts) WATCHDOG: MemAvailable $m GiB -> restarting trainenv"; docker restart trainenv > /dev/null; sleep 60; fi
    sleep 5; done ) &

serve() {  # serve <name> <dir relative to ckpt/>
  docker rm -f student > /dev/null 2>&1
  docker run -d --name student --gpus all --ipc=host -p 8001:8001 \
    --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
    --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset \
    -e HF_HUB_OFFLINE=1 -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/chess-distillation/ckpt:/ckpt \
    nvcr.io/nvidia/vllm:26.05-py3 vllm serve /ckpt/$2 --served-model-name $1 --dtype bfloat16 \
    --port 8001 --gpu-memory-utilization 0.30 --max-model-len 8192 --max-num-seqs 64 > /dev/null
  for i in $(seq 1 120); do curl -sf localhost:8001/v1/models > /dev/null && return 0
    docker ps --filter name=student -q | grep -q . || break; sleep 5; done
  echo "$(ts) student server failed to start ($*)"; docker logs --tail 20 student 2>&1; return 1
}

# 1. train B (skip if already trained)
if has_model ckpt/path_B; then
  echo "$(ts) ckpt/path_B already trained -> skipping to eval"
else
  echo "$(ts) TRAIN path_B (full FT, FA2, 2 passes, identical to A)"
  docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trainenv python -u src/train_sft.py --data results/sft/path_B.jsonl --out ckpt/path_B --model Qwen/Qwen3-1.7B \
      --full --epochs 2 --batch-tokens 8192 --accum 4 --no-grad-ckpt --save-epochs --attn flash_attention_2 \
    || echo "$(ts) TRAIN path_B FAILED"
fi

# 2. evaluate on the 500 test puzzles: final, then after pass 1
for pair in "path_B path_B" "path_B_ep1 path_B/epoch1"; do
  set -- $pair
  if has_model ckpt/$2; then
    echo "$(ts) EVAL $1"
    serve $1 $2 && $PY -u src/eval_student.py --model ckpt/$1 --served-name $1 --tag $1 --puzzles pilot_set.jsonl \
        --max-tokens 1024 --temperature 0 --concurrency 32 || echo "$(ts) EVAL $1 FAILED"
  else
    echo "$(ts) no checkpoint ckpt/$2 -> skip eval"
  fi
done
docker rm -f student > /dev/null 2>&1

# 3. are B's own explanations true? (claim checker on its outputs)
for tag in path_B path_B_ep1; do
  f=results/student_${tag}_nothink.jsonl
  [ -f $f ] && { echo "$(ts) EXPLAIN-CHECK $tag"; $PY src/student_explain_check.py $f; }
done
echo "$(ts) PATH_B DONE (teacher left stopped; min MemAvailable $(cat results/path_B.minmem 2>/dev/null) GiB)"
