"""Theme- and rating-balanced collection pool (after Master Distillation's theme-balanced sampling).

- Excludes the test set (pilot_set.jsonl) and any other --exclude files (by puzzle id).
- Quality filter: RatingDeviation <= 100, NbPlays >= 100 (well-tested puzzles).
- Ratings 600-2600 in 200-point bands (test set is 800-2200; a little wider on both ends).
- Balance: tactical themes are sampled up to --per-theme each, rarest theme first; the rest is filled
  rating-stratified. The output order is interleaved (round-robin over bands, shuffled) so ANY PREFIX
  of the file is itself balanced: an overnight run can stop anywhere.

Usage: .venv/bin/python build_collect_pool.py --n 60000 --out pool_collect1.jsonl \
          --exclude pilot_set.jsonl pool_night1.jsonl pool_pilot2k.jsonl
"""
import argparse
import csv
import json
import random
from collections import defaultdict

import chess

from code_trace import MOTIF_THEMES

BANDS = [(lo, lo + 200) for lo in range(600, 2600, 200)]


def to_record(row):
    moves = row["Moves"].split()
    board = chess.Board(row["FEN"])
    opp = chess.Move.from_uci(moves[0])
    board.push(opp)
    rating = int(row["Rating"])
    lo = next(lo for lo, hi in BANDS if lo <= rating < hi)
    return {"puzzle_id": row["PuzzleId"], "fen": board.fen(), "side_to_move": "white" if board.turn else "black",
            "correct_move": moves[1], "rating": rating, "rating_band": f"{lo}-{lo + 200}",
            "themes": row["Themes"].split(), "original_fen": row["FEN"], "opponent_move": moves[0],
            "full_solution": moves[1:], "game_url": row["GameUrl"], "nb_plays": int(row["NbPlays"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--per-theme", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--exclude", nargs="*", default=["pilot_set.jsonl"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    excl = {json.loads(l)["puzzle_id"] for f in args.exclude for l in open(f)}
    rows = []
    with open("data/lichess_db_puzzle.csv") as f:
        for row in csv.DictReader(f):
            if row["PuzzleId"] in excl or int(row["RatingDeviation"]) > 100 or int(row["NbPlays"]) < 100:
                continue
            r = int(row["Rating"])
            if 600 <= r < 2600:
                rows.append(row)
    rng.shuffle(rows)
    print(f"{len(rows):,} eligible puzzles")
    by_theme = defaultdict(list)
    for i, row in enumerate(rows):
        for t in row["Themes"].split():
            if t in MOTIF_THEMES:
                by_theme[t].append(i)
    chosen, used = [], set()
    for t in sorted(by_theme, key=lambda t: len(by_theme[t])):  # rarest first
        k = 0
        for i in by_theme[t]:
            if k >= args.per_theme or len(chosen) >= args.n // 2:
                break
            if i not in used:
                used.add(i)
                chosen.append(i)
                k += 1
    # fill the rest rating-stratified
    per_band = defaultdict(list)
    for i, row in enumerate(rows):
        if i not in used:
            per_band[next(lo for lo, hi in BANDS if lo <= int(row["Rating"]) < hi)].append(i)
    while len(chosen) < args.n and any(per_band.values()):
        for lo, _ in BANDS:
            if per_band[lo] and len(chosen) < args.n:
                i = per_band[lo].pop()
                used.add(i)
                chosen.append(i)
    recs = [to_record(rows[i]) for i in chosen]
    # also exclude any puzzle whose position (piece placement) equals one in the excluded files (e.g. the test
    # set): different puzzle ids can share a board (2026-09-25: 3 of 200k did)
    boards = {json.loads(l)["fen"].split(" ")[0] for f in args.exclude for l in open(f) if '"fen"' in l}
    before = len(recs)
    recs = [r for r in recs if r["fen"].split(" ")[0] not in boards]
    print(f"removed {before - len(recs)} puzzles sharing a position with an excluded file")
    # interleave by band so any prefix is balanced
    bands = defaultdict(list)
    for r in recs:
        bands[r["rating_band"]].append(r)
    for b in bands.values():
        rng.shuffle(b)
    out, keys = [], sorted(bands)
    while any(bands[k] for k in keys):
        for k in keys:
            if bands[k]:
                out.append(bands[k].pop())
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    themes = defaultdict(int)
    for r in out:
        for t in r["themes"]:
            if t in MOTIF_THEMES:
                themes[t] += 1
    print(f"wrote {len(out)} puzzles to {args.out}; bands: " +
          ", ".join(f"{k}:{sum(1 for r in out if r['rating_band'] == k)}" for k in keys))
    print("motif themes:", dict(sorted(themes.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
