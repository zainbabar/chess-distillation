#!/bin/bash
# 10-01 (user: "do all of them, I'm not using the Spark"). Machine-only checks, nothing new. Waits for
# results/run_tighten.sh to finish (B seed 1 training), then:
# CPU (background): experiments/line_soundness.py -> results/line_soundness.{jsonl,md} (Stockfish judges whole written
#   lines; method check: Lichess's own lines pass 98% at depth 18, measured 10-01)
# GPU: (1) untrained Qwen3-1.7B (pinned revision) on the development set at the students' exact settings
#      -> results/student_base_nothink_dev.jsonl
#      (2) teacher: docker start gptoss (known-good container, mem_guard running), on the fresh set: a second medium
#      sample (results/fresh_med_s2_formatP1L.jsonl, ~6 h), then low samples 2 and 3 (~40 min each); teacher stopped.
# Then experiments/final_checks_report.py -> results/final_checks_report.md; the log prints "FINAL CHECKS DONE".
# Resumable: re-run (finished steps skipped; run_pilot resumes). Launch: (nohup results/run_final_checks.sh >> results/final_checks.log 2>&1 &)
cd ~/chess-distillation
PY=.venv/bin/python
FRESH=fresh_test_set.jsonl
BASE=/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
ts() { date '+%F %T'; }
done500() { [ -s $1 ] && [ $($PY -c "import json,sys; print(len({json.loads(l)['puzzle_id'] for l in open(sys.argv[1]) if json.loads(l).get('status') != 'error'}))" $1) -eq 500 ]; }

echo "$(ts) FINAL CHECKS START (waiting for run_tighten.sh)"
while ps -eo comm | grep -qx run_tighten.sh; do sleep 60; done
echo "$(ts) run_tighten.sh finished: $(grep -a 'TIGHTEN DONE' results/tighten.log | tail -n 1)"

( [ -s results/line_soundness.md ] || nice -n 5 $PY -u experiments/line_soundness.py > results/final_checks_cpu.log 2>&1
  echo "$(ts) line soundness done" ) &
CPU_PID=$!

# 1. untrained model on the development set (teacher stopped)
if ! done500 results/student_base_nothink_dev.jsonl; then
  echo "$(ts) EVAL untrained Qwen3-1.7B on the development set"
  docker rm -f student > /dev/null 2>&1
  docker run -d --name student --gpus all --ipc=host -p 8001:8001 \
    --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
    --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset \
    -e HF_HUB_OFFLINE=1 -v ~/.cache/huggingface:/root/.cache/huggingface \
    nvcr.io/nvidia/vllm:26.05-py3 vllm serve $BASE --served-model-name base --dtype bfloat16 \
    --port 8001 --gpu-memory-utilization 0.30 --max-model-len 8192 --max-num-seqs 64 > /dev/null
  for i in $(seq 1 120); do curl -sf localhost:8001/v1/models > /dev/null && break; sleep 5; done
  $PY -u src/eval_student.py --model Qwen/Qwen3-1.7B@70d244cc --served-name base --tag base_dev --puzzles test_set.jsonl \
      --out results/student_base_nothink_dev.jsonl --max-tokens 1024 --temperature 0 --concurrency 32 | tail -n 2 \
      || echo "$(ts) EVAL base FAILED"
  docker rm -f student > /dev/null 2>&1
fi

# 2. teacher on the fresh set: medium sample 2, then low samples 2 and 3
if ! { done500 results/fresh_med_s2_formatP1L.jsonl && done500 results/fresh_low_s2_formatP1L.jsonl && done500 results/fresh_low_s3_formatP1L.jsonl; }; then
  if ! ps -eo comm | grep -qx mem_guard.sh; then
    (nohup src/mem_guard.sh >> results/mem_guard.log.3 2>&1 &); echo "$(ts) mem_guard started"; fi
  echo "$(ts) starting the teacher (docker start gptoss)"
  docker start gptoss > /dev/null
  for i in $(seq 1 180); do curl -sf localhost:8000/v1/models > /dev/null && break; sleep 10; done
  if curl -sf localhost:8000/v1/models > /dev/null; then
    echo "$(ts) teacher up; GPU devices allowed: $(systemctl show docker-$(docker inspect -f '{{.Id}}' gptoss).scope -p DeviceAllow | tr ' ' '\n' | grep -c '195\|499')"
    $PY -u src/run_pilot.py --run fresh_med_s2 --formats P1L --puzzles $FRESH --effort medium --max-tokens 32768 \
        --concurrency 32 | tail -n 3
    for k in 2 3; do
      $PY -u src/run_pilot.py --run fresh_low_s$k --formats P1L --puzzles $FRESH --effort low --max-tokens 8192 \
          --concurrency 32 | tail -n 3
    done
  else
    echo "$(ts) TEACHER FAILED TO START"; docker logs --tail 30 gptoss 2>&1
  fi
  docker stop gptoss > /dev/null && echo "$(ts) teacher stopped"
fi

wait $CPU_PID
$PY experiments/final_checks_report.py > /dev/null && echo "$(ts) report: results/final_checks_report.md"
echo "$(ts) FINAL CHECKS DONE"
