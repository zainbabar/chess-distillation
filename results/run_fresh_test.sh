#!/bin/bash
# 09-29 (user's go: "run the fresh test set as soon as the Spark becomes idle again"). The one-time evaluation fixed in
# reports/fresh_test_protocol.md, on fresh_test_set.jsonl. Evaluation only, no training.
# (0) waits until results/run_scale_v4.sh has finished (the Spark is busy until then);
# (1) students, teacher stopped (same serving/settings as the main evaluation; outputs results/fresh_student_<tag>.jsonl):
#     A, B, A+200k, B+RL v4 (final, if it exists), untrained Qwen3-1.7B (pinned revision, same settings);
# (2) Stockfish 16 at 0.1 s per puzzle (CPU) -> results/fresh_stockfish.jsonl;
# (3) teacher: docker start gptoss (the known-good container, mem_guard running), low then medium effort
#     -> results/fresh_{low,med}_formatP1L.jsonl; then the teacher is stopped again;
# (4) experiments/fresh_test_report.py -> results/fresh_test_report.md; the log prints "FRESH DONE".
# Resumable: re-run the script (finished steps are skipped; run_pilot/eval_student resume where they stopped).
# Launch: (nohup results/run_fresh_test.sh >> results/fresh_test.log 2>&1 &)
cd ~/chess-distillation
PY=.venv/bin/python
SET=fresh_test_set.jsonl
BASE=/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
ts() { date '+%F %T'; }
has_model() { ls $1/*.safetensors > /dev/null 2>&1; }
done500() { [ -s $1 ] && [ $($PY -c "import json,sys; print(len({json.loads(l)['puzzle_id'] for l in open(sys.argv[1]) if json.loads(l).get('status') != 'error'}))" $1) -eq 500 ]; }

echo "$(ts) FRESH START (protocol: reports/fresh_test_protocol.md)"
while ps -eo comm | grep -qx run_scale_v4.sh; do sleep 60; done
echo "$(ts) run_scale_v4.sh finished: $(grep -a 'SCALE_V4 DONE' results/scale_v4.log | tail -n 1)"
if docker ps --format '{{.Names}}' | grep -qx gptoss; then docker stop gptoss > /dev/null && echo "$(ts) teacher stopped"; fi

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
evaluate() {  # evaluate <tag> <model path inside the container> <label for the record>
  local out=results/fresh_student_$1.jsonl
  if done500 $out; then echo "$(ts) $1 already evaluated -> skip"; return; fi
  echo "$(ts) EVAL $1"
  serve $1 $2 && $PY -u src/eval_student.py --model $3 --served-name $1 --tag fresh_$1 --puzzles $SET --out $out \
      --max-tokens 1024 --temperature 0 --concurrency 32 | tail -n 2 || echo "$(ts) EVAL $1 FAILED"
}

# 1. students (teacher stopped)
for m in path_A path_B path_A200k; do
  has_model ckpt/$m && evaluate $m /ckpt/$m ckpt/$m || echo "$(ts) ckpt/$m missing -> skip"
done
has_model ckpt/rlT4_B && evaluate rlT4_B_final /ckpt/rlT4_B ckpt/rlT4_B || echo "$(ts) ckpt/rlT4_B missing -> skip"
evaluate base $BASE Qwen/Qwen3-1.7B@70d244cc
docker rm -f student > /dev/null 2>&1

# 2. Stockfish
done500 results/fresh_stockfish.jsonl || $PY experiments/stockfish_baseline.py --puzzles $SET --out results/fresh_stockfish.jsonl

# 3. teacher, low then medium effort
if ! done500 results/fresh_med_formatP1L.jsonl; then
  if ! ps -eo comm | grep -qx mem_guard.sh; then
    (nohup src/mem_guard.sh >> results/mem_guard.log.3 2>&1 &); echo "$(ts) mem_guard started"; fi
  echo "$(ts) starting the teacher (docker start gptoss)"
  docker start gptoss > /dev/null
  for i in $(seq 1 180); do curl -sf localhost:8000/v1/models > /dev/null && break; sleep 10; done
  if curl -sf localhost:8000/v1/models > /dev/null; then
    echo "$(ts) teacher up; DeviceAllow: $(systemctl show docker-$(docker inspect -f '{{.Id}}' gptoss).scope -p DeviceAllow | cut -c1-80)"
    done500 results/fresh_low_formatP1L.jsonl || $PY -u src/run_pilot.py --run fresh_low --formats P1L --puzzles $SET \
        --effort low --max-tokens 8192 --concurrency 32 | tail -n 3
    $PY -u src/run_pilot.py --run fresh_med --formats P1L --puzzles $SET --effort medium --max-tokens 32768 \
        --concurrency 32 | tail -n 3
  else
    echo "$(ts) TEACHER FAILED TO START"; docker logs --tail 30 gptoss 2>&1
  fi
  docker stop gptoss > /dev/null && echo "$(ts) teacher stopped"
fi

# 4. report
$PY experiments/fresh_test_report.py | tail -n 30
echo "$(ts) FRESH DONE"
