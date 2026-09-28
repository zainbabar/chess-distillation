#!/bin/bash
# night1 chain (see CLAUDE.md "OVERNIGHT RUN night1"). Resumable: re-running skips finished answers.
# Stage B format: S (structured solver). Documented fallback: B_FMT=P1L results/run_night1.sh
cd ~/chess-distillation
PY=.venv/bin/python
R=results
B_FMT=${B_FMT:-P1L}  # S failed its smoke test on 2026-09-24 (0/5), documented fallback P1L
ts() { date '+%F %T'; }
echo "$(ts) NIGHT1 START (stage B format $B_FMT)"

echo "$(ts) STAGE A: engine analysis + engine-written traces (E, low) on 98 puzzles"
{ [ -s $R/night1_engine_analysis.jsonl ] || $PY src/engine_analysis.py --puzzles pool_night1_engine98.jsonl --out $R/night1_engine_analysis.jsonl; } \
  && $PY -u src/run_pilot.py --run night1_A --formats E --puzzles pool_night1_engine98.jsonl --effort low --max-tokens 8192 --concurrency 32 \
  || echo "$(ts) STAGE A FAILED"

echo "$(ts) STAGE B: solver traces ($B_FMT, medium) on 504 fresh puzzles"
$PY -u src/run_pilot.py --run night1_B --formats $B_FMT --puzzles pool_night1.jsonl --effort medium --max-tokens 32768 --concurrency 32 \
  || echo "$(ts) STAGE B FAILED"

echo "$(ts) STAGE C: high effort (P1L, cap 60k) on 20 of B's misses"
$PY src/select_unsolved.py --puzzles pool_night1.jsonl --results $R/night1_B_format$B_FMT.jsonl --out $R/night1_C_puzzles.jsonl --sample 20 --seed 1 \
  && $PY -u src/run_pilot.py --run night1_C --formats P1L --puzzles $R/night1_C_puzzles.jsonl --effort high --max-tokens 60000 --concurrency 7 \
  || echo "$(ts) STAGE C FAILED"

echo "$(ts) STAGE D: step verification + report"
# step verification needs the structured format: always stage A (E); stage B only if it used S
VFILES="$R/night1_A_formatE.jsonl"; [ "$B_FMT" = S ] && VFILES="$VFILES $R/night1_B_format$B_FMT.jsonl"
$PY src/verify_trace.py --results $VFILES --out $R/night1_verified.jsonl || echo "$(ts) VERIFY FAILED"
$PY experiments/overnight_report.py > /dev/null || echo "$(ts) REPORT FAILED"
echo "$(ts) NIGHT1 DONE"
