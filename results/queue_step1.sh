#!/bin/bash
# Cascade test step 1: wait for the high+P1 probe, then run low + P1L (full line) on the 140 subset.
cd ~/chess-distillation
while ps -eo comm,args | awk '$1 ~ /^python/ && /run_pilot/ && /probe_high/' | grep -q .; do sleep 15; done
echo "$(date '+%F %T') high+P1 done, starting step 1 (low + P1L, 140 puzzles)"
exec .venv/bin/python -u src/run_pilot.py --run cascade_t1a --formats P1L --puzzles pilot_subset140.jsonl --effort low --max-tokens 8192 --concurrency 32
