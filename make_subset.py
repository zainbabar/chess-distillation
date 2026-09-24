"""Pick a rating-stratified subset of pilot_set.jsonl (for the reasoning-effort comparison).

Usage: .venv/bin/python make_subset.py [--per-band 20] [--seed 42] [--out pilot_subset140.jsonl]
"""
import argparse
import json
import random
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--per-band", type=int, default=20)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="pilot_subset140.jsonl")
ap.add_argument("--input", default="pilot_set.jsonl")
args = ap.parse_args()

by_band = defaultdict(list)
for line in open(args.input):
    p = json.loads(line)
    by_band[p["rating_band"]].append(p)

rng = random.Random(args.seed)
subset = []
for band in sorted(by_band, key=lambda b: int(b.split("-")[0])):
    subset += rng.sample(by_band[band], args.per_band)

subset.sort(key=lambda p: (p["rating"], p["puzzle_id"]))
with open(args.out, "w") as f:
    for p in subset:
        f.write(json.dumps(p) + "\n")
print(f"wrote {len(subset)} puzzles ({args.per_band} per band x {len(by_band)} bands) to {args.out}")
