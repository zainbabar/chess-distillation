#!/bin/bash
# THE PATH step 2, student A (answers only) overnight — user's go 2026-09-26 ~02:45.
# Waits for the explanation collection (COLLECT40K DONE) -> builds A/B data on the same puzzles -> stops the teacher
# -> full-fine-tune smoke test (400 examples, 2 passes, + a 20-puzzle serving check) -> trains A (Qwen3-1.7B, full FT,
# 2 passes, lr 1e-5, fp32 master weights) -> evaluates A (final + after pass 1) on the 500 test puzzles.
# Fallback (pre-approved by the user): if full FT fails with FA2 and with SDPA, or runs < 800 tok/s, use LoRA r64
# (lr 2e-4, as in night 2), same data and passes. The teacher stays STOPPED at the end (B trains next).
# Launch: (nohup results/run_path_A.sh >> results/path_A.log 2>&1 &)
cd ~/chess-distillation
PY=.venv/bin/python
ts() { date '+%F %T'; }
has_model() { ls $1/*.safetensors > /dev/null 2>&1; }   # full model (maybe sharded) or LoRA adapter
memgb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
echo "$(ts) PATH_A START"

# 0. wait for the collection
until grep -q "COLLECT40K DONE" results/collect40k.log; do
  if ! ps -eo comm | grep -qx "run_collect40k."; then
    sleep 5; grep -q "COLLECT40K DONE" results/collect40k.log && break
    echo "$(ts) collector stopped without COLLECT40K DONE -> PATH_A FAILED"; exit 1
  fi
  sleep 60
done
echo "$(ts) collection done: $(tail -n 1 results/collect40k.log)"

# 1. data (A and B on the same puzzles)
$PY src/make_path_data.py || { echo "$(ts) data build FAILED -> PATH_A FAILED"; exit 1; }

# 2. stop the teacher (never train while it serves)
if docker ps --format '{{.Names}}' | grep -qx gptoss; then
  docker stop gptoss > /dev/null && echo "$(ts) teacher stopped"
fi
for i in $(seq 1 30); do [ $(memgb) -gt 80 ] && break; sleep 5; done
echo "$(ts) MemAvailable $(memgb) GiB"
docker rm -f student > /dev/null 2>&1

# memory watchdog for training (the mem guard only protects the teacher): restart trainenv if < 8 GiB free
( minm=999; while ps -p $$ > /dev/null; do m=$(memgb); [ $m -lt $minm ] && { minm=$m; echo $minm > results/path_A.minmem; }
    if [ $m -lt 8 ]; then echo "$(ts) WATCHDOG: MemAvailable $m GiB -> restarting trainenv"; docker restart trainenv > /dev/null; sleep 60; fi
    sleep 5; done ) &

train() {  # train <data> <out> <mode full|lora> <attn> [extra args]
  local data=$1 out=$2 mode=$3 attn=$4; shift 4
  local m=""; [ $mode = full ] && m="--full"
  docker exec trainenv python -u src/train_sft.py --data $data --out $out --model Qwen/Qwen3-1.7B $m --epochs 2 \
      --batch-tokens 8192 --accum 4 --no-grad-ckpt --save-epochs --attn $attn "$@"
}

serve() {  # serve <mode> <name> <dir> [<name2> <dir2>]  (dirs relative to ckpt/)
  local mode=$1; shift
  docker rm -f student > /dev/null 2>&1
  local common="--port 8001 --gpu-memory-utilization 0.30 --max-model-len 8192 --max-num-seqs 64"
  if [ $mode = full ]; then
    docker run -d --name student --gpus all --ipc=host -p 8001:8001 \
      --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
      --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset \
      -e HF_HUB_OFFLINE=1 -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/chess-distillation/ckpt:/ckpt \
      nvcr.io/nvidia/vllm:26.05-py3 vllm serve /ckpt/$2 --served-model-name $1 --dtype bfloat16 $common > /dev/null
  else
    local mods="$1=/ckpt/$2"; [ -n "$3" ] && mods="$mods $3=/ckpt/$4"
    docker run -d --name student --gpus all --ipc=host -p 8001:8001 \
      --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
      --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset \
      -e HF_HUB_OFFLINE=1 -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/chess-distillation/ckpt:/ckpt \
      nvcr.io/nvidia/vllm:26.05-py3 vllm serve Qwen/Qwen3-1.7B $common \
      --enable-lora --max-lora-rank 64 --max-loras 2 --lora-modules $mods > /dev/null
  fi
  for i in $(seq 1 120); do curl -sf localhost:8001/v1/models > /dev/null && return 0
    docker ps --filter name=student -q | grep -q . || break; sleep 5; done
  echo "$(ts) student server failed to start ($*)"; docker logs --tail 20 student 2>&1; return 1
}

evalm() {  # evalm <served name> <tag> <puzzles>
  $PY -u src/eval_student.py --model ckpt/$1 --served-name $1 --tag $2 --puzzles $3 \
      --max-tokens 1024 --temperature 0 --concurrency 32 || echo "$(ts) EVAL $2 FAILED"
}

# 3. smoke test of full fine-tuning (FA2, then SDPA); LoRA fallback
MODE=""; ATTN=""
if has_model ckpt/path_A; then
  [ -f ckpt/path_A/adapter_model.safetensors ] && MODE=lora || MODE=full
  echo "$(ts) ckpt/path_A already trained ($MODE) -> skipping to eval"
else
  for attn in flash_attention_2 sdpa; do
    echo "$(ts) SMOKE full FT, attn $attn (400 examples, 2 passes)"
    [ -d ckpt/path_A_smoke ] && mv ckpt/path_A_smoke ckpt/path_A_smoke.old.$(date +%H%M%S)
    if train results/sft/path_A_smoke.jsonl ckpt/path_A_smoke full $attn > results/path_A_smoke_$attn.log 2>&1 \
        && has_model ckpt/path_A_smoke && has_model ckpt/path_A_smoke/epoch1; then
      tps=$(grep -oE '[0-9,]+ tok/s' results/path_A_smoke_$attn.log | tail -n 1 | tr -d ', tok/s')
      echo "$(ts) smoke $attn OK: $(grep -E '^step|examples' results/path_A_smoke_$attn.log | tr '\n' ' ') | min MemAvailable $(cat results/path_A.minmem) GiB"
      if [ "${tps:-0}" -ge 800 ]; then MODE=full; ATTN=$attn; break; fi
      echo "$(ts) smoke $attn too slow (${tps:-?} tok/s < 800)"
    else
      echo "$(ts) smoke $attn FAILED:"; tail -n 15 results/path_A_smoke_$attn.log
    fi
  done
  if [ "$MODE" = full ]; then
    # serving check: the saved full model loads in vLLM and answers (20 test puzzles, evaluation only)
    if serve full path_A_smoke path_A_smoke; then
      head -n 20 test_set.jsonl > results/path_A_smoke_eval_puzzles.jsonl
      evalm path_A_smoke path_A_smoke results/path_A_smoke_eval_puzzles.jsonl
    else
      echo "$(ts) WARNING: full-model serving failed in the smoke test (training continues; fix eval in the morning)"
    fi
    docker rm -f student > /dev/null 2>&1
  else
    MODE=lora; ATTN=flash_attention_2
    echo "$(ts) FALLBACK to LoRA r64 (full FT failed or too slow)"
  fi

  # 4. train A
  echo "$(ts) TRAIN path_A mode=$MODE attn=$ATTN ($(wc -l < results/sft/path_A.jsonl) examples, 2 passes)"
  if [ $MODE = full ]; then train results/sft/path_A.jsonl ckpt/path_A full $ATTN
  else train results/sft/path_A.jsonl ckpt/path_A lora $ATTN; fi \
    || { echo "$(ts) TRAIN path_A FAILED"; }
fi

# 5. evaluate A on the 500 test puzzles: final model, then after pass 1
for pair in "path_A path_A" "path_A_ep1 path_A/epoch1"; do
  set -- $pair
  if has_model ckpt/$2; then
    echo "$(ts) EVAL $1"
    serve $MODE $1 $2 && evalm $1 $1 test_set.jsonl
  else
    echo "$(ts) no checkpoint ckpt/$2 -> skip eval"
  fi
done
docker rm -f student > /dev/null 2>&1
echo "$(ts) PATH_A DONE (mode $MODE; teacher left stopped; min MemAvailable $(cat results/path_A.minmem 2>/dev/null) GiB)"
