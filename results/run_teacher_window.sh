#!/bin/bash
# Teacher window 17:30 (after the mem-guard restart): finish the pilot LLM traces, the exp2 leftovers,
# and a 20-puzzle smoke test of the overnight collection script.
cd ~/chess-distillation
PY=.venv/bin/python
ts() { date '+%F %T'; }
echo "$(ts) WINDOW START"
$PY -u src/write_traces.py --run pilot2k --variant FDF --puzzles pool_pilot2k.jsonl --effort low --concurrency 48 || echo "$(ts) FDF resume FAILED"
# exp2 leftovers (the 4 X2 answers lost to the restart; judge the high writer; judge solver reasoning)
$PY -u src/run_pilot.py --run exp2_E_high --formats E --puzzles pool_night1_engine98.jsonl --effort high --max-tokens 32768 --concurrency 12 &
X2PID=$!
{ [ -s results/exp2_engine_analysis_pool.jsonl ] || $PY src/engine_analysis.py --puzzles pool_night1.jsonl --out results/exp2_engine_analysis_pool.jsonl; }
$PY src/judge.py --results results/night1_B_formatP1L.jsonl --only-correct --field reasoning --analysis results/exp2_engine_analysis_pool.jsonl --puzzles pool_night1.jsonl --out results/judge_B.jsonl || echo "$(ts) JUDGE B FAILED"
# overnight script smoke test: 20 puzzles, separate output dir
CHUNK=20 MAXN=20 CONC=20 bash -c 'sed -e "s#results/collect1#results/collect1_smoke#g" -e "s#collect1_part#collect1smoke_part#g" results/run_collect1.sh > /tmp/claude-1001/-home-zainbabar-chess-distillation/e24f66bb-3624-449a-b248-5f4236a23ea9/scratchpad/run_collect1_smoke.sh; bash /tmp/claude-1001/-home-zainbabar-chess-distillation/e24f66bb-3624-449a-b248-5f4236a23ea9/scratchpad/run_collect1_smoke.sh'
wait $X2PID
$PY src/verify_trace.py --results results/exp2_E_high_formatE.jsonl --out results/exp2_verified_E_high.jsonl || echo "$(ts) X2 VERIFY FAILED"
$PY src/judge.py --results results/exp2_E_high_formatE.jsonl --analysis results/night1_engine_analysis.jsonl --puzzles pool_night1.jsonl --out results/judge_E_high.jsonl || echo "$(ts) X2 JUDGE FAILED"
$PY experiments/exp2_report.py > /dev/null || echo "$(ts) REPORT FAILED"
echo "$(ts) WINDOW DONE"
