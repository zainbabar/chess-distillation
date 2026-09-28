#!/bin/bash
# THE PATH step 3, first RL test on the Spark (user's go 2026-09-26 ~21:40, "a 2-3 hr RL test").
# GRPO (src/rl_grpo.py: binary reward, colocated vLLM sampling, fp32 master weights, DAPO loss, no KL) on A and on B with
# identical settings: the same 800 fresh puzzles (pool_collect1 rows 40,001-40,800, never used for SFT), 100 steps x
# 8 puzzles x 8 samples, lr 2e-6 constant, temperature 1.0, max 384 new tokens. Then evaluate both on the 500 test
# puzzles (greedy, max-tokens 1024, as the SFT evals). Smoke tests (3 steps each) passed: A ~35 s/step, B ~50 s/step.
# Launch: (nohup results/run_rl_test.sh >> results/rl_test.log 2>&1 &)
cd ~/chess-distillation
PY=.venv/bin/python
ts() { date '+%F %T'; }
has_model() { ls $1/*.safetensors > /dev/null 2>&1; }
memgb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
STEPS=${STEPS:-100}; LR=${LR:-2e-6}
echo "$(ts) RL_TEST START (steps $STEPS, lr $LR)"
if docker ps --format '{{.Names}}' | grep -qx gptoss; then docker stop gptoss > /dev/null && echo "$(ts) teacher stopped"; fi
docker rm -f student > /dev/null 2>&1

( minm=999; while ps -p $$ > /dev/null; do m=$(memgb); [ $m -lt $minm ] && { minm=$m; echo $minm > results/rl_test.minmem; }
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

for arm in A B; do
  out=ckpt/rl_$arm
  if has_model $out; then
    echo "$(ts) $out already trained -> skip"
  else
    echo "$(ts) RL $arm from ckpt/path_$arm"
    docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trainenv python -u src/rl_grpo.py --model ckpt/path_$arm \
        --puzzles pool_collect1.jsonl --skip 40000 --out $out --max-steps $STEPS --lr $LR > results/rl_$arm.log 2>&1 \
      || echo "$(ts) RL $arm FAILED (see results/rl_$arm.log)"
    grep -E "^rl step|^saved" results/rl_$arm.log | tail -n 2
  fi
  if has_model $out; then
    echo "$(ts) EVAL rl_$arm"
    serve rl_$arm rl_$arm && $PY -u src/eval_student.py --model $out --served-name rl_$arm --tag rl_$arm \
        --puzzles pilot_set.jsonl --max-tokens 1024 --temperature 0 --concurrency 32 | tail -n 2 \
      || echo "$(ts) EVAL rl_$arm FAILED"
    docker rm -f student > /dev/null 2>&1
  fi
done
$PY experiments/rl_test_report.py | tail -n 25
echo "$(ts) RL_TEST DONE (teacher left stopped; min MemAvailable $(cat results/rl_test.minmem 2>/dev/null) GiB)"
