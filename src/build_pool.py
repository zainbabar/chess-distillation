"""Build a fresh rating-stratified puzzle pool for data collection, EXCLUDING the test set.

Same rules as build_test_set.py: RatingDeviation <= 100, 7 bands 800-2200, apply the opponent's first
move to get the real puzzle position; the answer is Moves[1]; full_solution = Moves[1:].
Puzzle ids in any --exclude file (default: test_set.jsonl, the held-out test set) are never picked.

Usage: .venv/bin/python build_pool.py --per-band 72 --seed 7 --out pool_night1.jsonl
"""
import argparse
import csv
import json
import random

import chess

CSV_PATH = "data/lichess_db_puzzle.csv"
BANDS = [(lo, lo + 200) for lo in range(800, 2200, 200)]
MAX_RATING_DEVIATION = 100


def band_of(rating):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= rating < hi:
            return i
    return None


def to_record(row, band):
    moves = row["Moves"].split()
    board = chess.Board(row["FEN"])
    opp = chess.Move.from_uci(moves[0])
    assert opp in board.legal_moves
    board.push(opp)
    answer = chess.Move.from_uci(moves[1])
    assert answer in board.legal_moves
    themes = row["Themes"].split()
    if "mateIn1" in themes:
        after = board.copy()
        after.push(answer)
        assert after.is_checkmate()
    lo, hi = BANDS[band]
    return {
        "puzzle_id": row["PuzzleId"], "fen": board.fen(),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "correct_move": moves[1], "rating": int(row["Rating"]), "rating_band": f"{lo}-{hi}",
        "themes": themes, "original_fen": row["FEN"], "opponent_move": moves[0],
        "full_solution": moves[1:], "game_url": row["GameUrl"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-band", type=int, default=72)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude", nargs="*", default=["test_set.jsonl"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    excluded = set()
    for path in args.exclude:
        excluded |= {json.loads(l)["puzzle_id"] for l in open(path)}

    rows_by_band = [[] for _ in BANDS]
    with open(CSV_PATH, newline="") as f:
        for rownum, row in enumerate(csv.DictReader(f)):
            if int(row["RatingDeviation"]) > MAX_RATING_DEVIATION or row["PuzzleId"] in excluded:
                continue
            b = band_of(int(row["Rating"]))
            if b is not None:
                rows_by_band[b].append(rownum)

    rng = random.Random(args.seed)
    chosen = {}
    for b, rows in enumerate(rows_by_band):
        for r in rng.sample(rows, args.per_band):
            chosen[r] = b

    records = []
    with open(CSV_PATH, newline="") as f:
        for rownum, row in enumerate(csv.DictReader(f)):
            if rownum in chosen:
                records.append(to_record(row, chosen[rownum]))

    assert not {r["puzzle_id"] for r in records} & excluded
    records.sort(key=lambda r: (r["rating"], r["puzzle_id"]))
    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(records)} puzzles to {args.out} (excluded {len(excluded)} ids)")


if __name__ == "__main__":
    main()
