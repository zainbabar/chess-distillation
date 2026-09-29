#!/bin/bash
# Night 2 (proposed 2026-09-24 ~20:45; NOT launched — the user approves): baseline + a 4x-larger student comparison.
#
# Why: tonight's pilot (1,892 puzzles, Qwen3-1.7B LoRA) gave answer-only 48.4%, code-built 49.4%, LLM (FDF-low) 41.2%
# (p = 0.002). Does LLM reasoning catch up with 4x the data (as the Master Distillation scaling curve suggests)?
#   A. teacher (running): baseline P1L-low on the 500 test puzzles (~1 h; evaluation only)
#   B. teacher: FDF-low traces for the first N=8,000 puzzles of pool_collect1 (~4 h) -> packaged (filtered)
#   C. stop the teacher; train Qwen3-1.7B LoRA r64, 2 epochs, on the SAME puzzles (those with an accepted trace):
#      answer-only (~1.5 h) and FDF-low (~2 h); evaluate both on the 500 test puzzles
#   D. restart the teacher (so the machine is back to normal in the morning)
# Every step resumes if re-run. Expected end ~7-8 h after launch.
# Launch: (nohup results/run_night2.sh >> results/night2.log 2>&1 &)     (mem_guard running, nothing else on the GPU)
# Options: N=8000  EPOCHS=2  SKIP_BASELINE=1
cd ~/chess-distillation
export PYTHONPATH=src  # shared modules live in src/
PY=.venv/bin/python
ts() { date '+%F %T'; }
N=${N:-8000}
RUN=${RUN:-night2}   # all outputs are named after RUN (smoke tests use another name)
EPOCHS=${EPOCHS:-2}
TEACHER_CMD='docker run -d --name gptoss --gpus all --ipc=host -p 8000:8000 --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset -e HF_HUB_OFFLINE=1 -e VLLM_MXFP4_BACKEND=marlin -e VLLM_MARLIN_USE_ATOMIC_ADD=1 -v /home/zainbabar/.cache/huggingface:/root/.cache/huggingface nvcr.io/nvidia/vllm:26.05-py3 vllm serve openai/gpt-oss-120b --moe-backend marlin --attention-backend TRITON_ATTN --gpu-memory-utilization 0.70 --max-model-len 65536 --max-num-seqs 32'
mkdir -p results/$RUN results/sft
echo "$(ts) NIGHT2 START (run $RUN, N=$N, epochs=$EPOCHS)"
ps -eo comm | grep -q '^mem_guard.sh' || { echo "$(ts) mem_guard not running - start it first"; exit 1; }

# A + B need the teacher
if [ ! -f ckpt/${RUN}_llm/adapter_model.safetensors ]; then
  curl -sf localhost:8000/v1/models >/dev/null || { eval $TEACHER_CMD >/dev/null; echo "$(ts) teacher starting";
    until curl -sf localhost:8000/v1/models >/dev/null; do sleep 10; done; }
  if [ "${SKIP_BASELINE:-0}" != "1" ]; then
    echo "$(ts) A: teacher baseline, P1L low, 500 test puzzles"
    $PY -u src/run_pilot.py --run test500 --formats P1L --puzzles test_set.jsonl --effort low --max-tokens 8192 \
        --concurrency 32 || echo "$(ts) A FAILED"
  fi
  head -n $N pool_collect1.jsonl > results/$RUN/pool.jsonl
  echo "$(ts) B: FDF-low traces for $N puzzles"
  $PY -u src/write_traces.py --run $RUN --variant FDF --puzzles results/$RUN/pool.jsonl --effort low \
      --concurrency 48 --max-tokens 4096 || echo "$(ts) B FAILED"
  $PY -u src/write_traces.py --run $RUN --variant FDF --puzzles results/$RUN/pool.jsonl --effort low \
      --concurrency 48 --max-tokens 4096 || true   # second pass retries any errors
  $PY src/package_sft.py --traces results/traces_${RUN}_FDF.jsonl --puzzles results/$RUN/pool.jsonl \
      --out results/$RUN/sft_fdf.jsonl
fi

# C: arms on the same puzzles, teacher stopped
RUN=$RUN $PY - <<'EOF'
import json, os
RUN = os.environ['RUN']
acc = [json.loads(l) for l in open(f"results/{RUN}/sft_fdf.jsonl")]
ids = {a["puzzle_id"] for a in acc}
pz = [json.loads(l) for l in open(f"results/{RUN}/pool.jsonl") if json.loads(l)["puzzle_id"] in ids]
with open(f"results/sft/{RUN}_llm.jsonl", "w") as f:
    for a in acc:
        f.write(json.dumps({"puzzle_id": a["puzzle_id"], "prompt": a["prompt"], "target": a["target"]}) + "\n")
from run_pilot import build_prompt
with open(f"results/sft/{RUN}_answer.jsonl", "w") as f:
    for p in pz:
        f.write(json.dumps({"puzzle_id": p["puzzle_id"], "prompt": build_prompt(p, "P1L"),
                            "target": f"FINAL_LINE: {' '.join(p['full_solution'])}\nFINAL_MOVE: {p['correct_move']}"}) + "\n")
print(len(ids), "puzzles in both arms")
EOF
docker stop gptoss >/dev/null 2>&1; docker rm gptoss >/dev/null 2>&1; echo "$(ts) teacher stopped for training"
sed -i "s/--epochs [0-9]*/--epochs $EPOCHS/" results/run_pilot_train.sh
results/run_pilot_train.sh ${RUN}_answer ${RUN}_llm
results/run_pilot_eval.sh ${RUN}_answer ${RUN}_llm
docker rm -f student >/dev/null 2>&1
RUN=$RUN .venv/bin/python - <<'EOF'
import json, os
RUN = os.environ['RUN']
from analyze_pilot import mcnemar
from puzzle_rating import rate
D = {}
import os.path
for arm in (f"{RUN}_answer", f"{RUN}_llm"):
    R = {}
    f = f"results/student_pilot_{arm}_nothink.jsonl"
    if not os.path.exists(f):
        print(arm, "not evaluated"); continue
    for l in open(f):
        r = json.loads(l); R[r["puzzle_id"]] = r
    D[arm] = R
    rt = rate(list(R.values()))
    print(arm, sum(r["status"] == "correct" for r in R.values()), "/", len(R), "rating", rt["rating"], (rt["ci_low"], rt["ci_high"]))
if len(D) < 2:
    raise SystemExit(0)
a, b = D[f"{RUN}_answer"], D[f"{RUN}_llm"]
ids = set(a) & set(b)
x = sum(a[k]["status"] == "correct" and b[k]["status"] != "correct" for k in ids)
y = sum(b[k]["status"] == "correct" and a[k]["status"] != "correct" for k in ids)
print("answer-only only right:", x, "LLM only right:", y, "p =", round(mcnemar(x, y), 4))
EOF

# D: teacher back (skipped when a wrapper continues with more training: NO_TEACHER_RESTART=1)
if [ "${NO_TEACHER_RESTART:-0}" != "1" ]; then
  eval $TEACHER_CMD >/dev/null && echo "$(ts) teacher restarting"
  until curl -sf localhost:8000/v1/models >/dev/null; do sleep 10; done
fi
echo "$(ts) NIGHT2 DONE"
