#!/bin/bash
# Wait for the high-effort probe to finish, then run the medium-effort probe on the same 10 puzzles.
cd ~/chess-distillation
while ps -eo comm,args | awk '$1 ~ /^python/ && /run_pilot/ && /probe_high/' | grep -q .; do sleep 15; done
echo "$(date '+%F %T') high probe done, starting medium probe"
exec .venv/bin/python -u src/run_pilot.py --run probe_medium --formats B --puzzles pilot_subset140.jsonl --limit 10 --effort medium --max-tokens 32768 --concurrency 10
