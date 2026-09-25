#!/bin/bash
# Rest of the exp2 chain after the user dropped X3 (2026-09-24 15:00). Replaces run_exp2.sh from X2 on:
# waits for the running X2 answers (run_pilot PID = $1), then X2 resume/verify/judge, the pool analysis +
# judging stage B's solver reasoning, and the report. X3 now lives in results/run_x3.sh (maybe overnight).
# Resumable: re-running skips finished answers/judgments.
cd ~/chess-distillation
PY=.venv/bin/python
R=results
ts() { date '+%F %T'; }
echo "$(ts) X3 DROPPED (user decision); rest of the chain continues in run_exp2_rest.sh"
if [ -n "$1" ]; then
  echo "$(ts) waiting for the running X2 answers (PID $1) to finish"
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi
$PY -u run_pilot.py --run exp2_E_high --formats E --puzzles pool_night1_engine98.jsonl --effort high --max-tokens 32768 --concurrency 12 || echo "$(ts) X2 FAILED"
$PY verify_trace.py --results $R/exp2_E_high_formatE.jsonl --out $R/exp2_verified_E_high.jsonl || echo "$(ts) X2 VERIFY FAILED"
$PY judge.py --results $R/exp2_E_high_formatE.jsonl --analysis $R/night1_engine_analysis.jsonl --puzzles pool_night1.jsonl --out $R/judge_E_high.jsonl || echo "$(ts) X2 JUDGE FAILED"

echo "$(ts) POST: Stockfish analysis of the pool + judging stage B's solver reasoning"
{ [ -s $R/exp2_engine_analysis_pool.jsonl ] || $PY engine_analysis.py --puzzles pool_night1.jsonl --out $R/exp2_engine_analysis_pool.jsonl; } || echo "$(ts) POOL ANALYSIS FAILED"
$PY judge.py --results $R/night1_B_formatP1L.jsonl --only-correct --field reasoning --analysis $R/exp2_engine_analysis_pool.jsonl --puzzles pool_night1.jsonl --out $R/judge_B.jsonl || echo "$(ts) JUDGE B FAILED"

echo "$(ts) REPORT"
$PY exp2_report.py > /dev/null || echo "$(ts) REPORT FAILED"
echo "$(ts) EXP2 DONE (without X3)"
