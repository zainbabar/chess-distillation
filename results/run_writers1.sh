#!/bin/bash
# Strategy work 2026-09-24 evening: feigned-discovery writers on the 98 engine-grounded puzzles.
cd ~/chess-distillation
PY=.venv/bin/python
ts() { date '+%F %T'; }
echo "$(ts) WRITERS1 START"
$PY -u src/write_traces.py --run w98 --variant FD --puzzles pool_night1_engine98.jsonl --effort low --concurrency 24 || echo "$(ts) FD low FAILED"
mv -n results/traces_w98_FD.jsonl results/traces_w98_FD_low.jsonl
$PY -u src/write_traces.py --run w98 --variant FDF --puzzles pool_night1_engine98.jsonl --effort low --concurrency 24 || echo "$(ts) FDF low FAILED"
mv -n results/traces_w98_FDF.jsonl results/traces_w98_FDF_low.jsonl
$PY -u src/write_traces.py --run w98 --variant FDF --puzzles pool_night1_engine98.jsonl --effort medium --concurrency 24 || echo "$(ts) FDF medium FAILED"
mv -n results/traces_w98_FDF.jsonl results/traces_w98_FDF_medium.jsonl
echo "$(ts) WRITERS1 DONE"
