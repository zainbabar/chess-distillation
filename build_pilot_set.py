"""Build the 500-puzzle pilot set from the Lichess puzzle database.

Lichess CSV quirk: `FEN` is the position BEFORE the opponent's move, and the first UCI move in
`Moves` is that opponent move. We apply it to get the real puzzle position; the correct answer is
the SECOND move in `Moves`.
"""
import csv
import json
import random

import chess

CSV_PATH = "data/lichess_db_puzzle.csv"
OUT_PATH = "pilot_set.jsonl"
SEED = 42
TOTAL = 500
BANDS = [(lo, lo + 200) for lo in range(800, 2200, 200)]  # 800-1000, ..., 2000-2200
MAX_RATING_DEVIATION = 100  # skip puzzles whose rating is still uncertain


def band_of(rating):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= rating < hi:
            return i
    return None


def main():
    # Pass 1: row numbers per band (cheap on memory).
    rows_by_band = [[] for _ in BANDS]
    with open(CSV_PATH, newline="") as f:
        for rownum, row in enumerate(csv.DictReader(f)):
            if int(row["RatingDeviation"]) > MAX_RATING_DEVIATION:
                continue
            b = band_of(int(row["Rating"]))
            if b is not None:
                rows_by_band[b].append(rownum)

    # Stratified sample: split TOTAL as evenly as possible (extra puzzles go to the lowest bands).
    rng = random.Random(SEED)
    per_band = [TOTAL // len(BANDS) + (1 if i < TOTAL % len(BANDS) else 0) for i in range(len(BANDS))]
    chosen = {}
    for b, (rows, k) in enumerate(zip(rows_by_band, per_band)):
        print(f"band {BANDS[b][0]}-{BANDS[b][1]}: {len(rows):,} eligible, sampling {k}")
        for r in rng.sample(rows, k):
            chosen[r] = b

    # Pass 2: extract the chosen rows and build the real puzzle positions.
    records = []
    with open(CSV_PATH, newline="") as f:
        for rownum, row in enumerate(csv.DictReader(f)):
            if rownum not in chosen:
                continue
            moves = row["Moves"].split()
            board = chess.Board(row["FEN"])
            opp_move = chess.Move.from_uci(moves[0])
            assert opp_move in board.legal_moves, f"{row['PuzzleId']}: opponent move illegal"
            board.push(opp_move)
            answer = chess.Move.from_uci(moves[1])
            assert answer in board.legal_moves, f"{row['PuzzleId']}: answer illegal"
            themes = row["Themes"].split()
            if "mateIn1" in themes:
                after = board.copy()
                after.push(answer)
                assert after.is_checkmate(), f"{row['PuzzleId']}: mateIn1 answer is not mate"
            lo, hi = BANDS[chosen[rownum]]
            records.append({
                "puzzle_id": row["PuzzleId"],
                "fen": board.fen(),
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "correct_move": moves[1],
                "rating": int(row["Rating"]),
                "rating_band": f"{lo}-{hi}",
                "themes": themes,
                # Kept for reference / later use:
                "original_fen": row["FEN"],
                "opponent_move": moves[0],
                "full_solution": moves[1:],
                "game_url": row["GameUrl"],
            })

    records.sort(key=lambda r: (r["rating"], r["puzzle_id"]))
    with open(OUT_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(records)} puzzles to {OUT_PATH}")


if __name__ == "__main__":
    main()
