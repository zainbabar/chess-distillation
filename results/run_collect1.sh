#!/bin/bash
# Overnight collection "collect1" (recommended recipe, 2026-09-24): NOT launched automatically — the user approves it.
#
# What it does, in order (every step is resumable: re-running skips finished work):
#   0. teacher baseline on the 500 test puzzles (P1L, low effort, single attempt; ~1 h; BASELINE=0 skips)
#   1. teacher (gpt-oss-120b, low effort) writes a facts-grounded "feigned discovery" trace (FDF) for each
#      puzzle of pool_collect1.jsonl (60k, theme- and rating-balanced; any prefix is balanced), in chunks
#   2. after each chunk: filter (right move, line = Lichess solution, no leak, claim-check clean) and
#      package SFT data -> results/collect1/sft_fdf.jsonl (+ rejected list)
#   3. in the background, Stockfish multi-PV analysis of each chunk (CPU, 10 threads) for later variants
#   4. code-built traces + board-tracking tasks for the whole pool already exist (results/collect1/sft_code.jsonl,
#      aux_20k.jsonl; free)
# Stop it any time (Ctrl-C / kill): whatever finished is kept and usable.
#
# Launch (teacher running, mem_guard running, nothing else on the GPU):
#   (nohup results/run_collect1.sh >> results/collect1.log 2>&1 &)
# Options via env: CHUNK=5000 (puzzles per chunk), MAXN=60000, CONC=48
cd ~/chess-distillation
PY=.venv/bin/python
ts() { date '+%F %T'; }
CHUNK=${CHUNK:-5000}
MAXN=${MAXN:-60000}
CONC=${CONC:-48}
mkdir -p results/collect1
echo "$(ts) COLLECT1 START (chunk $CHUNK, max $MAXN, concurrency $CONC)"
curl -sf localhost:8000/v1/models >/dev/null || { echo "$(ts) teacher not running on :8000 — start it first (CLAUDE.md)"; exit 1; }
# 0. the fair baseline for the headline: the teacher, single attempt, same P1L prompt, on the whole 500-puzzle
#    test set (evaluation only; ~1 h at low effort). Skip with BASELINE=0.
if [ "${BASELINE:-1}" = "1" ]; then
  echo "$(ts) baseline: teacher P1L low on the 500 test puzzles"
  $PY -u run_pilot.py --run test500 --formats P1L --puzzles pilot_set.jsonl --effort low --max-tokens 8192 \
      --concurrency 32 || echo "$(ts) baseline FAILED (re-run to resume)"
fi
for ((start = 0; start < MAXN; start += CHUNK)); do
  part=$(printf "%05d" $((start / CHUNK)))
  f=results/collect1/pool_part$part.jsonl
  [ -s $f ] || sed -n "$((start + 1)),$((start + CHUNK))p" pool_collect1.jsonl > $f
  [ -s $f ] || break
  echo "$(ts) chunk $part: puzzles $start..$((start + CHUNK - 1))"
  # Stockfish multi-PV analysis of the chunk on the (otherwise idle) CPU, in the background: enables the
  # refuted-try variant (FDFT) and code-built search traces later. Written at the end of each chunk.
  a=results/collect1/engine_analysis_part$part.jsonl
  [ -s $a ] || (nohup $PY engine_analysis.py --puzzles $f --out $a --workers 10 > /dev/null 2>&1 &)
  $PY -u write_traces.py --run collect1_part$part --variant FDF --puzzles $f --effort low \
      --concurrency $CONC --max-tokens 4096 || echo "$(ts) chunk $part writer FAILED (re-run to resume)"
  $PY -u package_sft.py --traces results/traces_collect1_part${part}_FDF.jsonl --puzzles $f \
      --out results/collect1/sft_fdf_part$part.jsonl || echo "$(ts) chunk $part packaging FAILED"
done
cat results/collect1/sft_fdf_part?????.jsonl > results/collect1/sft_fdf.jsonl 2>/dev/null
echo "$(ts) COLLECT1 DONE: $(wc -l < results/collect1/sft_fdf.jsonl) accepted traces"
