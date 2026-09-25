#!/bin/bash
# X3, dropped from exp2 by the user (2026-09-24) and kept for a possible overnight run: a second MEDIUM
# solving attempt on night1 stage B's misses (~257 puzzles, ~3.5-4 h on the Spark), then judge the
# correct traces' hidden reasoning and refresh the exp2 report. Resumable: re-running skips finished work.
# Launch (mem_guard.sh running, exp2 finished):  (nohup results/run_x3.sh >> results/x3_run.log 2>&1 &)
cd ~/chess-distillation
PY=.venv/bin/python
R=results
ts() { date '+%F %T'; }
echo "$(ts) X3 START: second MEDIUM attempt on stage B's misses"
$PY select_unsolved.py --puzzles pool_night1.jsonl --results $R/night1_B_formatP1L.jsonl --out $R/exp2_B2_puzzles.jsonl \
  && $PY -u run_pilot.py --run exp2_B2 --formats P1L --puzzles $R/exp2_B2_puzzles.jsonl --effort medium --max-tokens 32768 --concurrency 32 \
  || echo "$(ts) X3 FAILED"
{ [ -s $R/exp2_engine_analysis_pool.jsonl ] || $PY engine_analysis.py --puzzles pool_night1.jsonl --out $R/exp2_engine_analysis_pool.jsonl; } || echo "$(ts) POOL ANALYSIS FAILED"
$PY judge.py --results $R/exp2_B2_formatP1L.jsonl --only-correct --field reasoning --analysis $R/exp2_engine_analysis_pool.jsonl --puzzles pool_night1.jsonl --out $R/judge_B2.jsonl || echo "$(ts) JUDGE B2 FAILED"
$PY exp2_report.py > /dev/null || echo "$(ts) REPORT FAILED"
echo "$(ts) X3 DONE"
