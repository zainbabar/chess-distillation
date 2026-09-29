#!/bin/bash
# Overnight 09-28 (user's go ~02:40; user back ~18:00): (1) teacher gpt-oss-120b at MEDIUM effort on the 500 test puzzles
# (P1L prompt, max 32,768 tokens, 32 concurrent, one attempt — same as night1's medium run) -> results/test500_med_formatP1L.jsonl;
# hard cutoff BASE_DEADLINE (resumable later; unfinished puzzles are simply missing). (2) teacher stopped. (3) RL for B with
# --reward truth3 (same start/puzzles/settings as the other three B runs) -> ckpt/rlT3_B (+ step130/260), eval all, reports.
# Launch: (nohup results/run_overnight3.sh >> results/overnight3.log 2>&1 &)
cd ~/chess-distillation
PY=.venv/bin/python
ts() { date '+%F %T'; }
has_model() { ls $1/*.safetensors > /dev/null 2>&1; }
memgb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
BASE_DEADLINE=${BASE_DEADLINE:-"2026-09-28 10:30"}
echo "$(ts) OVERNIGHT3 START (teacher medium baseline until $BASE_DEADLINE at the latest, then truth3 RL for B)"

# 1. teacher up (mem_guard must be running; the container keeps the known-good config)
ps -eo comm | grep -qx mem_guard.sh || echo "$(ts) WARNING: mem_guard.sh not running"
docker rm -f student > /dev/null 2>&1
if ! curl -sf localhost:8000/v1/models > /dev/null; then
  docker start gptoss > /dev/null && echo "$(ts) teacher starting"
  for i in $(seq 1 120); do curl -sf localhost:8000/v1/models > /dev/null && break; sleep 10; done
fi
if curl -sf localhost:8000/v1/models > /dev/null; then
  echo "$(ts) teacher up; MEDIUM baseline on the 500"
  left=$(( $(date -d "$BASE_DEADLINE" +%s) - $(date +%s) ))
  timeout $left $PY -u src/run_pilot.py --run test500_med --formats P1L --puzzles test_set.jsonl --effort medium \
      --max-tokens 32768 --concurrency 32 > results/test500_med.log 2>&1
  rc=$?; [ $rc = 124 ] && echo "$(ts) baseline hit the deadline (partial results kept; resumable)"
  echo "$(ts) BASELINE DONE: $(grep -c . results/test500_med_formatP1L.jsonl 2>/dev/null) records"
  $PY src/puzzle_rating.py results/test500_formatP1L.jsonl results/test500_med_formatP1L.jsonl 2>&1 | tail -n 6
else
  echo "$(ts) teacher failed to start -> skipping the baseline"
fi
docker stop gptoss > /dev/null 2>&1 && echo "$(ts) teacher stopped"
for i in $(seq 1 30); do [ $(memgb) -gt 80 ] && break; sleep 5; done

# 2. truth3 RL for B
( minm=999; while ps -p $$ > /dev/null; do m=$(memgb); [ $m -lt $minm ] && { minm=$m; echo $minm > results/overnight3.minmem; }
    if [ $m -lt 8 ]; then echo "$(ts) WATCHDOG: MemAvailable $m GiB -> restarting trainenv"; docker restart trainenv > /dev/null; sleep 60; fi
    sleep 5; done ) &
serve() {
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
out=ckpt/rlT3_B
if has_model $out; then echo "$(ts) $out already trained -> skip"; else
  echo "$(ts) RL B (truth3 reward) from ckpt/path_B: 390 steps"
  docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trainenv python -u src/rl_grpo.py --model ckpt/path_B \
      --puzzles pool_collect1.jsonl --skip 40800 --out $out --max-steps 390 --lr 2e-6 --save-every 130 --reward truth3 \
      > results/rlT3_B.log 2>&1 || echo "$(ts) RL B truth3 FAILED (see results/rlT3_B.log)"
  grep -aE "^saved" results/rlT3_B.log | tail -n 4
fi
for ck in step130 step260 final; do
  d=rlT3_B/$ck; tag=rlT3_B_$ck; [ $ck = final ] && d=rlT3_B
  if has_model ckpt/$d && [ ! -s results/student_${tag}_nothink.jsonl ]; then
    echo "$(ts) EVAL $tag"
    serve $tag $d && $PY -u src/eval_student.py --model ckpt/$d --served-name $tag --tag $tag --puzzles test_set.jsonl \
        --max-tokens 1024 --temperature 0 --concurrency 32 | tail -n 2 || echo "$(ts) EVAL $tag FAILED"
  fi
done
docker rm -f student > /dev/null 2>&1
$PY experiments/rl_truth_report.py | tail -n 5
echo "$(ts) OVERNIGHT3 DONE (teacher left stopped; min MemAvailable $(cat results/overnight3.minmem 2>/dev/null) GiB)"
