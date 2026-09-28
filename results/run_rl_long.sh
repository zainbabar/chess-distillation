#!/bin/bash
# THE PATH step 3, the ~12 h RL run (user's go 2026-09-27 ~01:10: "a 12 hr run, up until 1pm").
# Same settings as the RL test (src/rl_grpo.py: GRPO, binary reward, colocated vLLM, fp32 master weights, DAPO loss, no KL,
# 8 puzzles x 8 samples per step, lr 2e-6 constant, temperature 1.0, max 384 new tokens), but 390 steps per student on
# 3,120 NEW fresh puzzles (pool_collect1 rows 40,801-43,920: not used for SFT or the RL test), starting again from the
# SFT checkpoints; checkpoints every 130 steps. B first (the question is about B; ~5.4 h), then A (~4.1 h). If A can't
# finish 390 steps by DEADLINE minus the eval buffer, it runs fewer steps (the 130/260 checkpoints still compare equal
# steps). Then eval every checkpoint on the 500 test puzzles and write results/rl_long_report.md.
# Launch: (nohup results/run_rl_long.sh >> results/rl_long.log 2>&1 &)
cd ~/chess-distillation
PY=.venv/bin/python
ts() { date '+%F %T'; }
has_model() { ls $1/*.safetensors > /dev/null 2>&1; }
memgb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
STEPS=${STEPS:-390}; LR=${LR:-2e-6}; SKIP=${SKIP:-40800}; SAVE=${SAVE:-130}
DEADLINE=${DEADLINE:-"2026-09-27 13:00"}; EVAL_BUFFER_MIN=${EVAL_BUFFER_MIN:-25}; A_SEC_PER_STEP=${A_SEC_PER_STEP:-38}
echo "$(ts) RL_LONG START (steps $STEPS, lr $LR, puzzles from row $((SKIP + 1)), checkpoints every $SAVE, deadline $DEADLINE)"
if docker ps --format '{{.Names}}' | grep -qx gptoss; then docker stop gptoss > /dev/null && echo "$(ts) teacher stopped"; fi
docker rm -f student > /dev/null 2>&1

( minm=999; while ps -p $$ > /dev/null; do m=$(memgb); [ $m -lt $minm ] && { minm=$m; echo $minm > results/rl_long.minmem; }
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

for arm in B A; do
  out=ckpt/rlL_$arm
  if has_model $out; then echo "$(ts) $out already trained -> skip"; continue; fi
  steps=$STEPS
  if [ $arm = A ]; then
    left=$(( ( $(date -d "$DEADLINE" +%s) - $(date +%s) ) / 60 - EVAL_BUFFER_MIN - 3 ))
    fit=$(( left * 60 / A_SEC_PER_STEP ))
    if [ $fit -lt $steps ]; then
      steps=$(( fit > 0 ? fit : 0 ))
      echo "$(ts) WARNING: only $left min left before the eval buffer -> A runs $steps steps instead of $STEPS"
    fi
  fi
  if [ $steps -lt 1 ]; then echo "$(ts) no time left for RL $arm -> skip"; continue; fi
  echo "$(ts) RL $arm from ckpt/path_$arm: $steps steps"
  docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trainenv python -u src/rl_grpo.py --model ckpt/path_$arm \
      --puzzles pool_collect1.jsonl --skip $SKIP --out $out --max-steps $steps --lr $LR --save-every $SAVE \
      > results/rlL_$arm.log 2>&1 || echo "$(ts) RL $arm FAILED (see results/rlL_$arm.log)"
  grep -aE "^saved" results/rlL_$arm.log | tail -n 4
done

# evaluate every checkpoint on the 500 test puzzles
for arm in B A; do
  for ck in step130 step260 final; do
    d=rlL_$arm/$ck; tag=rlL_${arm}_$ck
    [ $ck = final ] && d=rlL_$arm
    if has_model ckpt/$d && [ ! -s results/student_${tag}_nothink.jsonl ]; then
      echo "$(ts) EVAL $tag"
      serve $tag $d && $PY -u src/eval_student.py --model ckpt/$d --served-name $tag --tag $tag --puzzles pilot_set.jsonl \
          --max-tokens 1024 --temperature 0 --concurrency 32 | tail -n 2 || echo "$(ts) EVAL $tag FAILED"
    fi
  done
done
docker rm -f student > /dev/null 2>&1
$PY experiments/rl_long_report.py | tail -n 30
echo "$(ts) RL_LONG DONE (teacher left stopped; min MemAvailable $(cat results/rl_long.minmem 2>/dev/null) GiB)"
