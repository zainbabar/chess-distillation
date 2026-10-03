#!/bin/bash
# 09-30 night (user: "run all DGX things, GPU and CPU ... machine algorithms only"). Tightening checks, nothing new.
# CPU, in the background (one after the other, 12 Stockfish threads):
#   experiments/stockfish_audit.py  -> results/stockfish_audit.{jsonl,md}: every wrong answer of the current models
#                                      re-checked by Stockfish (same rule as the original 567-answer check)
#   experiments/reply_quality.py    -> results/reply_quality.{jsonl,md}: are the opponent replies in written lines real
#                                      defenses (Stockfish), with the Lichess solutions as the method check
# GPU:
#   (1) 4 more sampled runs (samples 2-5; sample 1 exists) of A, B, A+200k and B-answer-first on both sets
#   (2) student B again with --seed 1 (B's exact recipe, train_meta: seed 0) -> ckpt/path_B_seed1 (~7.2 h)
#   (3) greedy evals of path_B_seed1 (final, pass 1) on both sets
# Then experiments/tighten_report.py -> results/tighten_report.md; the log prints "TIGHTEN DONE".
# Resumable: re-run (finished steps are skipped). Launch: (nohup results/run_tighten.sh >> results/tighten.log 2>&1 &)
cd ~/chess-distillation
PY=.venv/bin/python
DEV=test_set.jsonl
FRESH=fresh_test_set.jsonl
ts() { date '+%F %T'; }
has_model() { ls $1/*.safetensors > /dev/null 2>&1; }
memgb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
done500() { [ -s $1 ] && [ $($PY -c "import json,sys; print(len({json.loads(l)['puzzle_id'] for l in open(sys.argv[1]) if json.loads(l).get('status') != 'error'}))" $1) -eq 500 ]; }

echo "$(ts) TIGHTEN START"
if docker ps --format '{{.Names}}' | grep -qx gptoss; then docker stop gptoss > /dev/null && echo "$(ts) teacher stopped"; fi
docker rm -f student > /dev/null 2>&1
( minm=999; while ps -p $$ > /dev/null; do m=$(memgb); [ $m -lt $minm ] && { minm=$m; echo $minm > results/tighten.minmem; }
    if [ $m -lt 8 ]; then echo "$(ts) WATCHDOG: MemAvailable $m GiB -> restarting trainenv"; docker restart trainenv > /dev/null; sleep 60; fi
    sleep 5; done ) &

# CPU checks in the background
( [ -s results/stockfish_audit.md ] && grep -q "depth 18" results/stockfish_audit.md || nice -n 5 $PY -u experiments/stockfish_audit.py > results/tighten_cpu_audit.log 2>&1
  [ -s results/reply_quality.md ] && grep -q "Lichess rows" results/reply_quality.md && [ "$(grep -c '^| ' results/reply_quality.md)" -ge 19 ] \
    || nice -n 5 $PY -u experiments/reply_quality.py > results/tighten_cpu_reply.log 2>&1
  echo "$(ts) CPU checks done" ) &
CPU_PID=$!

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
run_eval() {  # run_eval <tag> <puzzles> <out> [extra eval_student args]   (the server must be up)
  local tag=$1 puzzles=$2 out=$3; shift 3
  if done500 $out; then return; fi
  $PY -u src/eval_student.py --model ckpt/$tag --served-name $tag --tag $tag --puzzles $puzzles --out $out \
      --max-tokens 1024 --concurrency 32 "$@" | tail -n 1 || echo "$(ts) EVAL $out FAILED"
}

# 1. samples 2-5 (eval_student's default sampling: temperature 0.7, top-p 0.8, top-k 20)
for tag in path_A path_B path_A200k path_Baf; do
  outs=""; for k in 2 3 4 5; do outs="$outs results/student_${tag}_sampled${k}_nothink.jsonl results/fresh_student_${tag}_sampled${k}.jsonl"; done
  need=0; for f in $outs; do done500 $f || need=1; done
  if [ $need = 1 ] && has_model ckpt/$tag; then
    echo "$(ts) SAMPLES 2-5 $tag"
    if serve $tag /ckpt/$tag; then
      for k in 2 3 4 5; do
        run_eval $tag $DEV results/student_${tag}_sampled${k}_nothink.jsonl
        run_eval $tag $FRESH results/fresh_student_${tag}_sampled${k}.jsonl
      done
    fi
  fi
done
docker rm -f student > /dev/null 2>&1

# 2. B with seed 1
if has_model ckpt/path_B_seed1; then echo "$(ts) ckpt/path_B_seed1 already trained -> skip"; else
  echo "$(ts) TRAIN path_B_seed1 (B's exact recipe, --seed 1)"
  docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trainenv python -u src/train_sft.py \
      --data results/sft/path_B.jsonl --out ckpt/path_B_seed1 --model Qwen/Qwen3-1.7B --full --epochs 2 \
      --batch-tokens 8192 --accum 4 --no-grad-ckpt --save-epochs --attn flash_attention_2 --seed 1 \
      > results/path_B_seed1.log 2>&1 || echo "$(ts) TRAIN path_B_seed1 FAILED (see results/path_B_seed1.log)"
  grep -aE "^saved|optimizer steps" results/path_B_seed1.log | tail -n 3
fi

# 3. greedy evals of B seed 1
for pair in "path_B_seed1 path_B_seed1" "path_B_seed1_ep1 path_B_seed1/epoch1"; do
  set -- $pair
  if has_model ckpt/$2 && ! { done500 results/student_$1_nothink.jsonl && done500 results/fresh_student_$1.jsonl; }; then
    echo "$(ts) EVAL $1"
    if serve $1 /ckpt/$2; then
      run_eval $1 $DEV results/student_$1_nothink.jsonl --temperature 0
      run_eval $1 $FRESH results/fresh_student_$1.jsonl --temperature 0
    fi
  fi
done
docker rm -f student > /dev/null 2>&1

wait $CPU_PID
$PY experiments/tighten_report.py > /dev/null && echo "$(ts) report: results/tighten_report.md"
echo "$(ts) TIGHTEN DONE (teacher left stopped; min MemAvailable $(cat results/tighten.minmem 2>/dev/null) GiB)"
