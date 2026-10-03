"""Build fresh_test_set.jsonl (defaults) or another fresh set (--seed, --out): 500 new puzzles for a one-time
evaluation (see reports/fresh_test_protocol.md, reports/fresh_test_2_protocol.md).

Same recipe as build_test_set.py (same Lichess snapshot, 7 rating bands 800-2200, RatingDeviation <= 100, 72/72/72/71/
71/71/71), a new seed, and a strict exclusion: a candidate is dropped if its puzzle id, its source game, or any position
along its solution (piece placement, side to move, castling, en passant; move counters ignored) matches ANY puzzle that
appears anywhere in this project's files (every pool, training set, test set and output under the repo root and
results/), not only the training sets.
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path

import chess

from build_test_set import BANDS, CSV_PATH, MAX_RATING_DEVIATION, band_of

OUT_PATH = "fresh_test_set.jsonl"
SEED = 20260929
PER_BAND = [72, 72, 72, 71, 71, 71, 71]
ID_RE = re.compile(rb'"puzzle_id": "([A-Za-z0-9]{5})"')


def key(board):
    return " ".join(board.fen().split()[:4])


def positions(fen, moves):
    b = chess.Board(fen)
    out = [key(b)]
    for m in moves:
        b.push_uci(m)
        out.append(key(b))
    return out


def seen_ids():
    """Every puzzle id in any .jsonl file of the project (root and results/, recursively)."""
    files = list(Path(".").glob("*.jsonl")) + list(Path("results").rglob("*.jsonl"))
    ids = set()
    for f in files:
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                ids.update(m.decode() for m in ID_RE.findall(chunk))
    return ids, len(files)


def main():
    global OUT_PATH, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=OUT_PATH)
    a = ap.parse_args()
    OUT_PATH, SEED = a.out, a.seed
    ids, n_files = seen_ids()
    # Pass 1: positions and games of every puzzle we have ever used; eligible rows per band.
    games, pos = set(), set()
    eligible = [[] for _ in BANDS]
    with open(CSV_PATH, newline="") as f:
        for rownum, row in enumerate(csv.DictReader(f)):
            if row["PuzzleId"] in ids:
                games.add(row["GameUrl"].split("#")[0])
                pos.update(positions(row["FEN"], row["Moves"].split()))
            elif int(row["RatingDeviation"]) <= MAX_RATING_DEVIATION:
                b = band_of(int(row["Rating"]))
                if b is not None:
                    eligible[b].append(rownum)
    print(f"excluding {len(ids):,} puzzle ids seen in {n_files} files: {len(games):,} games, {len(pos):,} positions")
    # Pass 2: walk each band in a seeded random order and keep the first candidates that pass the exclusion.
    rng = random.Random(SEED)
    order = {}
    for b, rows in enumerate(eligible):
        rng.shuffle(rows)
        for rank, r in enumerate(rows[:PER_BAND[b] * 20]):  # plenty of spares; exclusions are rare
            order[r] = (b, rank)
    cand = [[] for _ in BANDS]
    with open(CSV_PATH, newline="") as f:
        for rownum, row in enumerate(csv.DictReader(f)):
            if rownum in order:
                b, rank = order[rownum]
                cand[b].append((rank, row))
    records, dropped = [], 0
    for b, rows in enumerate(cand):
        kept = 0
        for _, row in sorted(rows, key=lambda x: x[0]):
            if kept == PER_BAND[b]:
                break
            moves = row["Moves"].split()
            game = row["GameUrl"].split("#")[0]
            if game in games or pos.intersection(positions(row["FEN"], moves)):
                dropped += 1
                continue
            board = chess.Board(row["FEN"])
            board.push(chess.Move.from_uci(moves[0]))
            answer = chess.Move.from_uci(moves[1])
            assert answer in board.legal_moves, row["PuzzleId"]
            themes = row["Themes"].split()
            if "mateIn1" in themes:
                after = board.copy()
                after.push(answer)
                assert after.is_checkmate(), row["PuzzleId"]
            games.add(game)  # no two fresh puzzles from the same game either
            lo, hi = BANDS[b]
            records.append({"puzzle_id": row["PuzzleId"], "fen": board.fen(),
                            "side_to_move": "white" if board.turn == chess.WHITE else "black",
                            "correct_move": moves[1], "rating": int(row["Rating"]), "rating_band": f"{lo}-{hi}",
                            "themes": themes, "original_fen": row["FEN"], "opponent_move": moves[0],
                            "full_solution": moves[1:], "game_url": row["GameUrl"]})
            kept += 1
        assert kept == PER_BAND[b], f"band {BANDS[b]}: only {kept}"
    records.sort(key=lambda r: (r["rating"], r["puzzle_id"]))
    with open(OUT_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(records)} puzzles to {OUT_PATH} ({dropped} candidates dropped by the exclusion)")


if __name__ == "__main__":
    main()
