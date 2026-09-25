#!/bin/bash
# Step 1 of THE PATH (launched 2026-09-25 ~15:45 with the user's OK): fact-grounded "as if discovering" explanations
# (FDF, low effort) for pool_collect1 puzzles 6,001-40,000 (the first 6,000 were done in night 2), ~12 h.
# Per 5k chunk: write (resumable) -> package/filter (claim check) ; Stockfish multi-PV analysis of each chunk runs on the
# idle CPU in the background (for the refuted-try format later). Stop any time; everything finished is kept.
# Launch: (nohup results/run_collect40k.sh >> results/collect40k.log 2>&1 &)
cd ~/chess-distillation
PY=.venv/bin/python
ts() { date '+%F %T'; }
START=${START:-6000}; END=${END:-40000}; CHUNK=${CHUNK:-5000}; CONC=${CONC:-48}
D=results/collect40k
mkdir -p $D
echo "$(ts) COLLECT40K START (puzzles $((START+1))-$END, chunks of $CHUNK)"
curl -sf localhost:8000/v1/models >/dev/null || { echo "$(ts) teacher not running"; exit 1; }
for ((s = START; s < END; s += CHUNK)); do
  e=$(( s + CHUNK < END ? s + CHUNK : END ))
  part=$(printf "%05d" $s)
  f=$D/pool_$part.jsonl
  [ -s $f ] || sed -n "$((s + 1)),${e}p" pool_collect1.jsonl > $f
  a=$D/engine_analysis_$part.jsonl
  [ -s $a ] || (nohup $PY engine_analysis.py --puzzles $f --out $a --workers 8 > /dev/null 2>&1 &)
  echo "$(ts) chunk $part: puzzles $((s + 1))-$e"
  $PY -u write_traces.py --run c40k_$part --variant FDF --puzzles $f --effort low --concurrency $CONC --max-tokens 4096 \
      || echo "$(ts) chunk $part writer FAILED (re-run to resume)"
  $PY -u write_traces.py --run c40k_$part --variant FDF --puzzles $f --effort low --concurrency $CONC --max-tokens 4096 \
      > /dev/null 2>&1 || true   # retry any errored requests
  $PY package_sft.py --traces results/traces_c40k_${part}_FDF.jsonl --puzzles $f --out $D/sft_fdf_$part.jsonl \
      || echo "$(ts) chunk $part packaging FAILED"
done
cat $D/sft_fdf_?????.jsonl > $D/sft_fdf_all.jsonl
echo "$(ts) COLLECT40K DONE: $(wc -l < $D/sft_fdf_all.jsonl) accepted (plus night 2's $(wc -l < results/night2/sft_fdf.jsonl))"
