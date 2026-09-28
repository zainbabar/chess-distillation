#!/bin/bash
# Orchestrates the rest of the student pilot (strategy work, 2026-09-24 evening).
cd ~/chess-distillation
PY=.venv/bin/python
ts() { date '+%F %T'; }
TEACHER_CMD='docker run -d --name gptoss --gpus all --ipc=host -p 8000:8000 --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset -e HF_HUB_OFFLINE=1 -e VLLM_MXFP4_BACKEND=marlin -e VLLM_MARLIN_USE_ATOMIC_ADD=1 -v /home/zainbabar/.cache/huggingface:/root/.cache/huggingface nvcr.io/nvidia/vllm:26.05-py3 vllm serve openai/gpt-oss-120b --moe-backend marlin --attention-backend TRITON_ATTN --gpu-memory-utilization 0.70 --max-model-len 65536 --max-num-seqs 32'
echo "$(ts) PILOT-ALL START"

# 1. teacher window: skip the 4 slow X2 stragglers, wait for WINDOW DONE
until grep -q "WINDOW DONE" results/teacher_window.log; do
  for pid in $(ps -eo pid,args | awk '/[r]un_pilot.py --run exp2_E_high/ {print $1}'); do
    kill $pid && echo "$(ts) skipped the X2 straggler re-run (pid $pid)"
  done
  sleep 15
done
# wait for the judge comparison too (it uses the teacher)
while ps -eo args | grep -q "[j]udge_eval.py"; do sleep 15; done
echo "$(ts) teacher work done"

# 2. arms on the same puzzles: those with an accepted LLM trace
$PY experiments/make_sft_data.py --puzzles pool_pilot2k.jsonl --arm llm --traces results/traces_pilot2k_FDF.jsonl --out results/sft/llm_2k_all.jsonl
$PY - <<'EOF'
import json
ids = {json.loads(l)["puzzle_id"] for l in open("results/sft/llm_2k_all.jsonl")}
with open("pool_pilot_llm.jsonl", "w") as f:
    for l in open("pool_pilot2k.jsonl"):
        if json.loads(l)["puzzle_id"] in ids:
            f.write(l)
print(len(ids), "puzzles with an accepted LLM trace")
EOF
for arm in answer code llm; do
  $PY experiments/make_sft_data.py --puzzles pool_pilot_llm.jsonl --arm $arm --traces results/traces_pilot2k_FDF.jsonl --out results/sft/p_$arm.jsonl
done

# 3. stop the teacher (never train while it serves), train, evaluate
docker stop gptoss > /dev/null && docker rm gptoss > /dev/null && echo "$(ts) teacher stopped for training"
results/run_pilot_train.sh p_answer p_code p_llm
results/run_pilot_eval.sh p_answer p_code p_llm
docker rm -f student > /dev/null 2>&1

# 4. Qwen3.5-2B trainability smoke test (20 steps, LoRA) + load the adapter in vLLM
echo "$(ts) QWEN3.5 SMOKE"
docker exec trainenv python -u src/train_sft.py --data results/sft/p_llm.jsonl --out ckpt/q35_smoke --model Qwen/Qwen3.5-2B \
    --max-steps 20 --epochs 1 --batch-tokens 8192 --accum 1 --no-grad-ckpt || echo "$(ts) QWEN3.5 TRAIN FAILED"

# 5. restart the teacher so the overnight run can be launched
eval $TEACHER_CMD > /dev/null && echo "$(ts) teacher restarting"
until curl -sf localhost:8000/v1/models > /dev/null; do sleep 10; done
echo "$(ts) teacher up"
echo "$(ts) PILOT-ALL DONE"
