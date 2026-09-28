#!/bin/bash
# Night 2, full (launched 2026-09-25 ~02:30 with the user's OK; they're asleep). ~10 h, free, local.
#   1. night2 core (results/run_night2.sh): teacher baseline P1L-low on the 500 test puzzles; FDF-low traces for
#      N fresh puzzles; teacher stopped; answer-only vs LLM students on the same puzzles; eval on the 500.
#   2. board-tracking arm: the same answers + one board-tracking task per puzzle (src/aux_tasks.py), same epochs.
#   3. answer-only scaling arm: the same puzzles + up to SCALE_MORE more from pool_collect1, 1 epoch,
#      sized to finish before DEADLINE (HHMM, today).
#   4. held-out board-tracking test for the answer / aux arms; summary table; teacher restarted at the end.
# Every step resumes if re-run. Outputs are named after RUN.
cd ~/chess-distillation
export PYTHONPATH=src  # shared modules live in src/
PY=.venv/bin/python
ts() { date '+%F %T'; }
export RUN=${RUN:-night2} N=${N:-8000} EPOCHS=${EPOCHS:-1}
SCALE_MORE=${SCALE_MORE:-16000}
DEADLINE=${DEADLINE:-1230}
TOKS=${TOKS:-1750}   # measured training speed (tokens/s) for Qwen3-1.7B LoRA without gradient checkpointing
TEACHER_CMD='docker run -d --name gptoss --gpus all --ipc=host -p 8000:8000 --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset -e HF_HUB_OFFLINE=1 -e VLLM_MXFP4_BACKEND=marlin -e VLLM_MARLIN_USE_ATOMIC_ADD=1 -v /home/zainbabar/.cache/huggingface:/root/.cache/huggingface nvcr.io/nvidia/vllm:26.05-py3 vllm serve openai/gpt-oss-120b --moe-backend marlin --attention-backend TRITON_ATTN --gpu-memory-utilization 0.70 --max-model-len 65536 --max-num-seqs 32'
minutes_left() { echo $(( ( $(date -d "today ${DEADLINE:0:2}:${DEADLINE:2:2}" +%s) - $(date +%s) ) / 60 )); }
est_minutes() {  # est_minutes FILE EPOCHS -> training minutes (chars/2.2 = tokens) + 12 min load/eval
  $PY -c "import sys; c=sum(len(l) for l in open('$1')); print(int(c/2.2*$2/$TOKS/60)+12)"
}
echo "$(ts) NIGHT2-FULL START (run $RUN, N=$N, epochs=$EPOCHS, scale +$SCALE_MORE, deadline $DEADLINE, $(minutes_left) min left)"

# 1. core
NO_TEACHER_RESTART=1 results/run_night2.sh
docker stop gptoss >/dev/null 2>&1; docker rm gptoss >/dev/null 2>&1   # (already stopped by the core; make sure)

# 2. board-tracking arm
if [ ! -s results/sft/${RUN}_aux.jsonl ]; then
  $PY src/aux_tasks.py --puzzles results/$RUN/pool.jsonl --n $(wc -l < results/sft/${RUN}_answer.jsonl) \
      --out results/$RUN/aux.jsonl --seed 21
  cat results/sft/${RUN}_answer.jsonl results/$RUN/aux.jsonl | shuf --random-source=<(yes) > results/sft/${RUN}_aux.jsonl
fi
need=$(est_minutes results/sft/${RUN}_aux.jsonl $EPOCHS)
if [ $need -lt $(minutes_left) ]; then
  echo "$(ts) AUX arm (est. $need min, $(minutes_left) left)"
  sed -i "s/--epochs [0-9]*/--epochs $EPOCHS/" results/run_pilot_train.sh
  results/run_pilot_train.sh ${RUN}_aux
  results/run_pilot_eval.sh ${RUN}_aux
  docker rm -f student >/dev/null 2>&1
else
  echo "$(ts) AUX arm skipped (needs $need min, $(minutes_left) left)"
fi

# 3. answer-only scaling arm, sized to the time left (1 epoch)
if [ ! -f ckpt/${RUN}_scale/adapter_model.safetensors ]; then
  avail=$(( $(minutes_left) - 30 ))   # keep 30 min for the board-tracking test, summary and teacher restart
  RUN=$RUN AVAIL=$avail SCALE_MORE=$SCALE_MORE TOKS=$TOKS N=$N $PY - <<'EOF'
import json, os
RUN, avail, more, toks, N = os.environ["RUN"], int(os.environ["AVAIL"]), int(os.environ["SCALE_MORE"]), int(os.environ["TOKS"]), int(os.environ["N"])
base = [json.loads(l) for l in open(f"results/sft/{RUN}_answer.jsonl")]
chars = sum(len(json.dumps(b)) for b in base) / len(base)
fit = int(max(0, avail - 12) * 60 * toks / (chars / 2.2))          # examples that fit in 1 epoch
from run_pilot import build_prompt
extra = []
for i, l in enumerate(open("pool_collect1.jsonl")):
    if i < N:
        continue
    if len(extra) >= min(more, fit - len(base)):
        break
    p = json.loads(l)
    extra.append({"puzzle_id": p["puzzle_id"], "prompt": build_prompt(p, "P1L"),
                  "target": f"FINAL_LINE: {' '.join(p['full_solution'])}\nFINAL_MOVE: {p['correct_move']}"})
total = len(base) + len(extra)
if total < 1.5 * len(base):
    print(f"scale arm skipped: only {total} examples fit ({avail} min)"); open(f"results/{RUN}/scale_skipped", "w").write(str(total))
else:
    with open(f"results/sft/{RUN}_scale.jsonl", "w") as f:
        for r in base + extra:
            f.write(json.dumps(r) + "\n")
    print(f"scale arm: {len(base)} + {len(extra)} = {total} examples (fits {fit} in {avail} min)")
EOF
fi
if [ -s results/sft/${RUN}_scale.jsonl ] && [ ! -f results/$RUN/scale_skipped ]; then
  echo "$(ts) SCALE arm ($(wc -l < results/sft/${RUN}_scale.jsonl) examples, 1 epoch, $(minutes_left) min left)"
  sed -i "s/--epochs [0-9]*/--epochs 1/" results/run_pilot_train.sh
  results/run_pilot_train.sh ${RUN}_scale
  results/run_pilot_eval.sh ${RUN}_scale
  docker rm -f student >/dev/null 2>&1
fi

# 4. held-out board-tracking test (answer vs aux arm) + summary
if [ ! -s results/$RUN/aux_heldout.jsonl ]; then
  tail -n 300 pool_collect1.jsonl > results/$RUN/heldout_pool.jsonl
  $PY src/aux_tasks.py --puzzles results/$RUN/heldout_pool.jsonl --n 200 --out results/$RUN/aux_heldout.jsonl --seed 77
fi
arms=""; for a in ${RUN}_answer ${RUN}_aux; do [ -f ckpt/$a/adapter_model.safetensors ] && arms="$arms $a"; done
if [ -n "$arms" ]; then
  mods=""; for a in $arms; do mods="$mods $a=/ckpt/$a"; done
  docker rm -f student >/dev/null 2>&1
  docker run -d --name student --gpus all --ipc=host -p 8001:8001 --device /dev/nvidia0 --device /dev/nvidiactl \
    --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset -e HF_HUB_OFFLINE=1 \
    -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/chess-distillation/ckpt:/ckpt nvcr.io/nvidia/vllm:26.05-py3 \
    vllm serve Qwen/Qwen3-1.7B --port 8001 --gpu-memory-utilization 0.08 --max-model-len 8192 --max-num-seqs 32 \
    --enable-lora --max-lora-rank 64 --max-loras 2 --lora-modules $mods > /dev/null
  until curl -sf localhost:8001/v1/models >/dev/null; do docker ps --filter name=student -q | grep -q . || break; sleep 5; done
  for a in $arms; do $PY experiments/aux_eval.py --tasks results/$RUN/aux_heldout.jsonl --served-name $a --out results/$RUN/auxeval_$a.jsonl; done
  docker rm -f student >/dev/null 2>&1
fi
RUN=$RUN $PY experiments/night2_summary.py || echo "$(ts) SUMMARY FAILED"

# 5. teacher back
eval $TEACHER_CMD >/dev/null && echo "$(ts) teacher restarting"
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 10; done
echo "$(ts) NIGHT2-FULL DONE"
