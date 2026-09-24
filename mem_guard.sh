#!/bin/bash
# Kill the vLLM container if free RAM drops below a threshold, so the machine never freezes.
# Usage: nohup ./mem_guard.sh [min_gib=8] > results/mem_guard.log 2>&1 &
MIN=${1:-8}
echo "$(date '+%F %T') guard started (threshold ${MIN} GiB)"
while true; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$a" -lt "$MIN" ]; then docker kill gptoss; echo "$(date '+%F %T') KILLED gptoss: ${a} GiB available"; exit 1; fi
  sleep 1
done
