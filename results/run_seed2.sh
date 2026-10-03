#!/bin/bash
# 09-30 (user's go: "A with the second seed and the pass-1 evaluations"). Replication only; teacher stays stopped.
# (1) Pass-1 checkpoints of A and B on the fresh set (greedy) -> results/fresh_student_path_{A,B}_ep1.jsonl, completing
#     the pass-1 comparison with B-answer-first.
# (2) Student A again with a different seed: identical to ckpt/path_A (train_meta: full FT, 2 passes, lr 1e-5,
#     batch-tokens 8192 x accum 4, FA2, no grad ckpt, seed 0) except --seed 1, which changes the shuffled batch order
#     -> ckpt/path_A_seed1 (+ epoch1), ~5.8 h; log results/path_A_seed1.log.
# (3) Greedy evals of path_A_seed1 (final, pass 1) on the development and fresh sets.
# Log prints "SEED2 DONE". Resumable: re-run (finished steps are skipped).
# Launch: (nohup results/run_seed2.sh >> results/seed2.log 2>&1 &)
cd ~/chess-distillation
PY=.venv/bin/python
DEV=test_set.jsonl
FRESH=fresh_test_set.jsonl
ts() { date '+%F %T'; }
has_model() { ls $1/*.safetensors > /dev/null 2>&1; }
memgb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
done500() { [ -s $1 ] && [ $($PY -c "import json,sys; print(len({json.loads(l)['puzzle_id'] for l in open(sys.argv[1]) if json.loads(l).get('status') != 'error'}))" $1) -eq 500 ]; }

echo "$(ts) SEED2 START"
if docker ps --format '{{.Names}}' | grep -qx gptoss; then docker stop gptoss > /dev/null && echo "$(ts) teacher stopped"; fi
docker rm -f student > /dev/null 2>&1
( minm=999; while ps -p $$ > /dev/null; do m=$(memgb); [ $m -lt $minm ] && { minm=$m; echo $minm > results/seed2.minmem; }
    if [ $m -lt 8 ]; then echo "$(ts) WATCHDOG: MemAvailable $m GiB -> restarting trainenv"; docker restart trainenv > /dev/null; sleep 60; fi
    sleep 5; done ) &

serve() {  # serve <served name> <model path inside the container>
  docker rm -f student > /dev/null 2>&1
  docker run -d --name student --gpus all --ipc=host -p 8001:8001 \
    --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
    --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset \
    -e HF_HUB_OFFLINE=1 -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/chess-distillation/ckpt:/ckpt \
    nvcr.io/nvidia/vllm:26.05-py3 vllm serve $2 --served-model-name $1 --dtype bfloat16 \
    --port 8001 --gpu-memory-utilization 0.30 --max-model-len 8192 --max-num-seqs 64 > /dev/null
  for i in $(seq 1 120); do curl -sf localhost:8001/v1/models > /dev/null && return 0
    docker ps --filter name=student -q | grep -q . || break; sleep 5; done
  echo "$(ts) student server failed to start ($*)"; docker logs --tail 20 student 2>&1; return 1
}
run_eval() {  # run_eval <tag> <puzzles> <out>   (greedy; the server must be up)
  if done500 $3; then echo "$(ts) $3 already done -> skip"; return; fi
  $PY -u src/eval_student.py --model ckpt/$1 --served-name $1 --tag $1 --puzzles $2 --out $3 \
      --max-tokens 1024 --temperature 0 --concurrency 32 | tail -n 2 || echo "$(ts) EVAL $3 FAILED"
}

# 1. pass-1 checkpoints of A and B on the fresh set
for pair in "path_A_ep1 path_A/epoch1" "path_B_ep1 path_B/epoch1"; do
  set -- $pair
  out=results/fresh_student_$1.jsonl
  if has_model ckpt/$2 && ! done500 $out; then
    echo "$(ts) EVAL $1 on the fresh set"
    serve $1 /ckpt/$2 && run_eval $1 $FRESH $out
  fi
done
docker rm -f student > /dev/null 2>&1

# 2. A with seed 1
if has_model ckpt/path_A_seed1; then echo "$(ts) ckpt/path_A_seed1 already trained -> skip"; else
  echo "$(ts) TRAIN path_A_seed1 (A's exact recipe, --seed 1)"
  docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trainenv python -u src/train_sft.py \
      --data results/sft/path_A.jsonl --out ckpt/path_A_seed1 --model Qwen/Qwen3-1.7B --full --epochs 2 \
      --batch-tokens 8192 --accum 4 --no-grad-ckpt --save-epochs --attn flash_attention_2 --seed 1 \
      > results/path_A_seed1.log 2>&1 || echo "$(ts) TRAIN path_A_seed1 FAILED (see results/path_A_seed1.log)"
  grep -aE "^saved|optimizer steps" results/path_A_seed1.log | tail -n 3
fi

# 3. evals of the new seed on both sets
for pair in "path_A_seed1 path_A_seed1" "path_A_seed1_ep1 path_A_seed1/epoch1"; do
  set -- $pair
  if has_model ckpt/$2 && ! { done500 results/student_$1_nothink.jsonl && done500 results/fresh_student_$1.jsonl; }; then
    echo "$(ts) EVAL $1"
    if serve $1 /ckpt/$2; then
      run_eval $1 $DEV results/student_$1_nothink.jsonl
      run_eval $1 $FRESH results/fresh_student_$1.jsonl
    fi
  fi
done
docker rm -f student > /dev/null 2>&1
echo "$(ts) SEED2 DONE (teacher left stopped; min MemAvailable $(cat results/seed2.minmem 2>/dev/null) GiB)"
