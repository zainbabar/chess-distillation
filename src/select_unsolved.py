"""Cascade helper: write the puzzles that NO earlier attempt solved, to feed the next tier.

Usage: .venv/bin/python select_unsolved.py --puzzles pilot_subset140.jsonl \
           --results results/a.jsonl results/b.jsonl --out tier2_puzzles.jsonl
A puzzle counts as solved if any result line for it has status "correct".
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--puzzles", required=True)
ap.add_argument("--results", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sample", type=int, default=0, help="keep only N unsolved, spread across rating bands")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

puzzles = [json.loads(l) for l in open(args.puzzles)]
seen, solved = set(), set()
for path in args.results:
    for line in open(path):
        r = json.loads(line)
        seen.add(r["puzzle_id"])
        if r["status"] == "correct":
            solved.add(r["puzzle_id"])

missing = [p["puzzle_id"] for p in puzzles if p["puzzle_id"] not in seen]
if missing:
    raise SystemExit(f"{len(missing)} puzzles have no result yet (e.g. {missing[:3]}); finish that tier first")

unsolved = [p for p in puzzles if p["puzzle_id"] not in solved]
if args.sample and len(unsolved) > args.sample:  # round-robin over bands, random within a band
    import random
    rng = random.Random(args.seed)
    by_band = {}
    for p in unsolved:
        by_band.setdefault(p["rating_band"], []).append(p)
    for v in by_band.values():
        rng.shuffle(v)
    bands = sorted(by_band, key=lambda b: int(b.split("-")[0]))
    picked = []
    while len(picked) < args.sample:
        for b in bands:
            if by_band[b] and len(picked) < args.sample:
                picked.append(by_band[b].pop())
    unsolved = sorted(picked, key=lambda p: (p["rating"], p["puzzle_id"]))
with open(args.out, "w") as f:
    for p in unsolved:
        f.write(json.dumps(p) + "\n")
print(f"{len(puzzles)} puzzles, {len(solved)} solved by earlier attempts -> {len(unsolved)} unsolved "
      f"written to {args.out}")
