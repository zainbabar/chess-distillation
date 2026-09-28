#!/bin/bash
# exp2 chain (see CLAUDE.md "NEXT TASK"): engine-grounded writer effort (X1 medium, X2 high) and a
# second medium solving attempt on stage B's misses (X3), with step verification + the explanation
# judge. Resumable: re-running skips finished answers/judgments.
cd ~/chess-distillation
PY=.venv/bin/python
R=results
ts() { date '+%F %T'; }
echo "$(ts) EXP2 START"

echo "$(ts) X1: engine-grounded, MEDIUM writer (98 puzzles)"
$PY -u src/run_pilot.py --run exp2_E_med --formats E --puzzles pool_night1_engine98.jsonl --effort medium --max-tokens 32768 --concurrency 32 || echo "$(ts) X1 FAILED"
$PY src/verify_trace.py --results $R/exp2_E_med_formatE.jsonl --out $R/exp2_verified_E_med.jsonl || echo "$(ts) X1 VERIFY FAILED"
$PY src/judge.py --results $R/exp2_E_med_formatE.jsonl --analysis $R/night1_engine_analysis.jsonl --puzzles pool_night1.jsonl --out $R/judge_E_med.jsonl || echo "$(ts) X1 JUDGE FAILED"

echo "$(ts) X2: engine-grounded, HIGH writer (98 puzzles)"
$PY -u src/run_pilot.py --run exp2_E_high --formats E --puzzles pool_night1_engine98.jsonl --effort high --max-tokens 32768 --concurrency 12 || echo "$(ts) X2 FAILED"
$PY src/verify_trace.py --results $R/exp2_E_high_formatE.jsonl --out $R/exp2_verified_E_high.jsonl || echo "$(ts) X2 VERIFY FAILED"
$PY src/judge.py --results $R/exp2_E_high_formatE.jsonl --analysis $R/night1_engine_analysis.jsonl --puzzles pool_night1.jsonl --out $R/judge_E_high.jsonl || echo "$(ts) X2 JUDGE FAILED"

echo "$(ts) X3: second MEDIUM attempt on stage B's misses"
$PY src/select_unsolved.py --puzzles pool_night1.jsonl --results $R/night1_B_formatP1L.jsonl --out $R/exp2_B2_puzzles.jsonl \
  && $PY -u src/run_pilot.py --run exp2_B2 --formats P1L --puzzles $R/exp2_B2_puzzles.jsonl --effort medium --max-tokens 32768 --concurrency 32 \
  || echo "$(ts) X3 FAILED"

echo "$(ts) POST: Stockfish analysis of the pool + judging the solver traces' hidden reasoning"
{ [ -s $R/exp2_engine_analysis_pool.jsonl ] || $PY src/engine_analysis.py --puzzles pool_night1.jsonl --out $R/exp2_engine_analysis_pool.jsonl; } || echo "$(ts) POOL ANALYSIS FAILED"
$PY src/judge.py --results $R/night1_B_formatP1L.jsonl --only-correct --field reasoning --analysis $R/exp2_engine_analysis_pool.jsonl --puzzles pool_night1.jsonl --out $R/judge_B.jsonl || echo "$(ts) JUDGE B FAILED"
$PY src/judge.py --results $R/exp2_B2_formatP1L.jsonl --only-correct --field reasoning --analysis $R/exp2_engine_analysis_pool.jsonl --puzzles pool_night1.jsonl --out $R/judge_B2.jsonl || echo "$(ts) JUDGE B2 FAILED"

echo "$(ts) REPORT"
[ -f experiments/exp2_report.py ] && { $PY experiments/exp2_report.py > /dev/null || echo "$(ts) REPORT FAILED"; } || echo "$(ts) experiments/exp2_report.py not written yet"
echo "$(ts) EXP2 DONE"
