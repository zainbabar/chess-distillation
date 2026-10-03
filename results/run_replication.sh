#!/bin/bash
# 10-01 night (user: "queue all 4"). Replications of the single-run results + a second fresh test. Nothing new.
# (1) A + 200k replicate: ckpt/path_A_seed1 continued on results/sft/ans_p1l_200k.jsonl with --seed 1, otherwise exactly
#     A200k's recipe (train_meta: full, 1 pass, lr 1e-5, batch-tokens 8192 x accum 4, FA2, no grad ckpt) -> ckpt/path_A200k_seed1
#     (~15 h); greedy evals on dev + fresh 1.
# (2) B answer-first replicate: path_Baf's recipe with --seed 1 -> ckpt/path_Baf_seed1 (+epoch1) (~7.2 h); evals dev + fresh 1.
# (3) RL v4 replicate: v4's exact command with --seed 1 -> ckpt/rlT4_B_seed1 (+step130/260) (~5.4 h); evals of the three
#     checkpoints on dev, final on fresh 1; Stockfish reply quality (background) -> results/replication_reply_quality.md.
# (4) Second fresh test (reports/fresh_test_2_protocol.md): all students on fresh_test_set_2.jsonl, Stockfish, then the
#     teacher (docker start gptoss, mem_guard running) at medium then low effort; teacher stopped.
# Then experiments/replication_report.py -> results/replication_report.md; the log prints "REPLICATION DONE".
# Resumable: re-run (finished steps skipped). Launch: (nohup results/run_replication.sh >> results/replication.log 2>&1 &)
cd ~/chess-distillation
PY=.venv/bin/python
DEV=test_set.jsonl
F1=fresh_test_set.jsonl
F2=fresh_test_set_2.jsonl
BASE=/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
ts() { date '+%F %T'; }
has_model() { ls $1/*.safetensors > /dev/null 2>&1; }
memgb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
done500() { [ -s $1 ] && [ $($PY -c "import json,sys; print(len({json.loads(l)['puzzle_id'] for l in open(sys.argv[1]) if json.loads(l).get('status') != 'error'}))" $1) -eq 500 ]; }

echo "$(ts) REPLICATION START"
if docker ps --format '{{.Names}}' | grep -qx gptoss; then docker stop gptoss > /dev/null && echo "$(ts) teacher stopped"; fi
docker rm -f student > /dev/null 2>&1
( minm=999; while ps -p $$ > /dev/null; do m=$(memgb); [ $m -lt $minm ] && { minm=$m; echo $minm > results/replication.minmem; }
    if [ $m -lt 8 ] && ! docker ps --format '{{.Names}}' | grep -qx gptoss; then
      echo "$(ts) WATCHDOG: MemAvailable $m GiB -> restarting trainenv"; docker restart trainenv > /dev/null; sleep 60; fi
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
  if done500 $3; then return; fi
  $PY -u src/eval_student.py --model ckpt/$1 --served-name $1 --tag $1 --puzzles $2 --out $3 \
      --max-tokens 1024 --temperature 0 --concurrency 32 | tail -n 1 || echo "$(ts) EVAL $3 FAILED"
}
evaluate() {  # evaluate <tag> <path in container> <puzzles:out> ...
  local tag=$1 path=$2; shift 2
  local need=0; for po in "$@"; do done500 ${po#*:} || need=1; done
  [ $need = 0 ] && return
  echo "$(ts) EVAL $tag"
  serve $tag $path || return
  for po in "$@"; do run_eval $tag ${po%%:*} ${po#*:}; done
}
train() {  # train <out ckpt> <log> <train_sft args...>
  local out=$1 log=$2; shift 2
  if has_model $out; then echo "$(ts) $out already trained -> skip"; return; fi
  echo "$(ts) TRAIN $out"
  docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trainenv python -u src/train_sft.py --out $out "$@" \
      > $log 2>&1 || echo "$(ts) TRAIN $out FAILED (see $log)"
  grep -aE "^saved|optimizer steps" $log | tail -n 3
}

# 1. A + 200k replicate
train ckpt/path_A200k_seed1 results/path_A200k_seed1.log --data results/sft/ans_p1l_200k.jsonl --model ckpt/path_A_seed1 \
    --full --epochs 1 --lr 1e-5 --batch-tokens 8192 --accum 4 --no-grad-ckpt --attn flash_attention_2 --seed 1
has_model ckpt/path_A200k_seed1 && evaluate path_A200k_seed1 /ckpt/path_A200k_seed1 \
    $DEV:results/student_path_A200k_seed1_nothink.jsonl $F1:results/fresh_student_path_A200k_seed1.jsonl
docker rm -f student > /dev/null 2>&1

# 2. B answer-first replicate
train ckpt/path_Baf_seed1 results/path_Baf_seed1.log --data results/sft/path_B_answerfirst.jsonl --model Qwen/Qwen3-1.7B \
    --full --epochs 2 --batch-tokens 8192 --accum 4 --no-grad-ckpt --save-epochs --attn flash_attention_2 --seed 1
has_model ckpt/path_Baf_seed1 && evaluate path_Baf_seed1 /ckpt/path_Baf_seed1 \
    $DEV:results/student_path_Baf_seed1_nothink.jsonl $F1:results/fresh_student_path_Baf_seed1.jsonl
has_model ckpt/path_Baf_seed1/epoch1 && evaluate path_Baf_seed1_ep1 /ckpt/path_Baf_seed1/epoch1 \
    $DEV:results/student_path_Baf_seed1_ep1_nothink.jsonl $F1:results/fresh_student_path_Baf_seed1_ep1.jsonl
docker rm -f student > /dev/null 2>&1

# 3. RL v4 replicate
if has_model ckpt/rlT4_B_seed1; then echo "$(ts) ckpt/rlT4_B_seed1 already trained -> skip"; else
  echo "$(ts) RL B (truth4 reward, seed 1) from ckpt/path_B: 390 steps"
  docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trainenv python -u src/rl_grpo.py --model ckpt/path_B \
      --puzzles pool_collect1.jsonl --skip 40800 --out ckpt/rlT4_B_seed1 --max-steps 390 --lr 2e-6 --save-every 130 \
      --reward truth4 --seed 1 > results/rlT4_B_seed1.log 2>&1 || echo "$(ts) RL truth4 seed 1 FAILED (see results/rlT4_B_seed1.log)"
  grep -aE "^saved" results/rlT4_B_seed1.log | tail -n 4
fi
for ck in step130 step260; do
  has_model ckpt/rlT4_B_seed1/$ck && evaluate rlT4_B_seed1_$ck /ckpt/rlT4_B_seed1/$ck $DEV:results/student_rlT4_B_seed1_${ck}_nothink.jsonl
done
has_model ckpt/rlT4_B_seed1 && evaluate rlT4_B_seed1_final /ckpt/rlT4_B_seed1 \
    $DEV:results/student_rlT4_B_seed1_final_nothink.jsonl $F1:results/fresh_student_rlT4_B_seed1_final.jsonl
docker rm -f student > /dev/null 2>&1
( [ -s results/replication_reply_quality.md ] || nice -n 5 $PY - > results/replication_cpu.log 2>&1 <<'PYEOF'
import sys
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
import reply_quality as rq
rq.OUT_STEM = "results/replication_reply_quality"
rq.RUNS = [("dev: Lichess solution (method check)", rq.DEV, "LICHESS"),
           ("dev: B + RL v4, seed 0, final", rq.DEV, "results/student_rlT4_B_final_nothink.jsonl"),
           ("dev: B + RL v4, seed 1, final", rq.DEV, "results/student_rlT4_B_seed1_final_nothink.jsonl"),
           ("fresh: B + RL v4, seed 0, final", rq.FRESH, "results/fresh_student_rlT4_B_final.jsonl"),
           ("fresh: B + RL v4, seed 1, final", rq.FRESH, "results/fresh_student_rlT4_B_seed1_final.jsonl")]
rq.main()
PYEOF
  echo "$(ts) reply quality (v4 replicate) done" ) &
CPU_PID=$!

# 4. second fresh test: students, Stockfish, then the teacher
for pair in "path_A /ckpt/path_A" "path_A_seed1 /ckpt/path_A_seed1" "path_A200k /ckpt/path_A200k" \
            "path_A200k_seed1 /ckpt/path_A200k_seed1" "path_B /ckpt/path_B" "path_B_seed1 /ckpt/path_B_seed1" \
            "path_Baf /ckpt/path_Baf" "path_Baf_seed1 /ckpt/path_Baf_seed1" "rlT4_B_final /ckpt/rlT4_B" \
            "rlT4_B_seed1_final /ckpt/rlT4_B_seed1"; do
  set -- $pair
  has_model ckpt/${2#/ckpt/} && evaluate $1 $2 $F2:results/fresh2_student_$1.jsonl
done
evaluate base $BASE $F2:results/fresh2_student_base.jsonl
docker rm -f student > /dev/null 2>&1
done500 results/fresh2_stockfish.jsonl || $PY experiments/stockfish_baseline.py --puzzles $F2 --out results/fresh2_stockfish.jsonl
if ! { done500 results/fresh2_med_formatP1L.jsonl && done500 results/fresh2_low_formatP1L.jsonl; }; then
  if ! ps -eo comm | grep -qx mem_guard.sh; then
    (nohup src/mem_guard.sh >> results/mem_guard.log.3 2>&1 &); echo "$(ts) mem_guard started"; fi
  echo "$(ts) starting the teacher (docker start gptoss)"
  docker start gptoss > /dev/null
  for i in $(seq 1 180); do curl -sf localhost:8000/v1/models > /dev/null && break; sleep 10; done
  if curl -sf localhost:8000/v1/models > /dev/null; then
    echo "$(ts) teacher up; GPU devices allowed: $(systemctl show docker-$(docker inspect -f '{{.Id}}' gptoss).scope -p DeviceAllow | tr ' ' '\n' | grep -c '195\|499')"
    $PY -u src/run_pilot.py --run fresh2_med --formats P1L --puzzles $F2 --effort medium --max-tokens 32768 --concurrency 32 | tail -n 3
    $PY -u src/run_pilot.py --run fresh2_low --formats P1L --puzzles $F2 --effort low --max-tokens 8192 --concurrency 32 | tail -n 3
  else
    echo "$(ts) TEACHER FAILED TO START"; docker logs --tail 30 gptoss 2>&1
  fi
  docker stop gptoss > /dev/null && echo "$(ts) teacher stopped"
fi

wait $CPU_PID
$PY experiments/replication_report.py > /dev/null && echo "$(ts) report: results/replication_report.md"
echo "$(ts) REPLICATION DONE (teacher left stopped; min MemAvailable $(cat results/replication.minmem 2>/dev/null) GiB)"
