#!/bin/bash
# Strategy work: LLM arm of the student pilot = FDF (facts-grounded feigned discovery), low effort, 2k pilot puzzles.
cd ~/chess-distillation
PY=.venv/bin/python
ts() { date '+%F %T'; }
while ps -eo comm,args | grep -q "[r]un_writers1.sh"; do sleep 20; done
echo "$(ts) WRITERS2 START (FDF low on pool_pilot2k)"
$PY -u write_traces.py --run pilot2k --variant FDF --puzzles pool_pilot2k.jsonl --effort low --concurrency 32 || echo "$(ts) FDF pilot2k FAILED"
echo "$(ts) WRITERS2 DONE"
