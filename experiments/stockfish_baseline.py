"""Stockfish as a reference point: its first move on each puzzle at a fixed budget, graded by the same grader as the
models (run_pilot.grade: first move vs the Lichess solution; in mate-in-one puzzles any mate counts).

    python experiments/stockfish_baseline.py --puzzles fresh_test_set.jsonl --out results/fresh_stockfish.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from run_pilot import grade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", default="fresh_test_set.jsonl")
    ap.add_argument("--out", default="results/fresh_stockfish.jsonl")
    ap.add_argument("--seconds", type=float, default=0.1)
    ap.add_argument("--engine", default="/usr/games/stockfish")
    a = ap.parse_args()
    eng = chess.engine.SimpleEngine.popen_uci(a.engine)
    eng.configure({"Threads": 1, "Hash": 64})
    n = ok = 0
    with open(a.out, "w") as f:
        for line in open(a.puzzles):
            p = json.loads(line)
            move = eng.play(chess.Board(p["fen"]), chess.engine.Limit(time=a.seconds)).move
            status, _, uci = grade(p, f"FINAL_MOVE: {move.uci()}")
            f.write(json.dumps({"puzzle_id": p["puzzle_id"], "rating": p["rating"], "rating_band": p["rating_band"],
                                "themes": p["themes"], "move": uci, "status": status,
                                "engine": f"stockfish, 1 thread, {a.seconds} s"}) + "\n")
            n += 1
            ok += status == "correct"
    eng.quit()
    print(f"stockfish {a.seconds} s: {ok}/{n} ({100 * ok / n:.1f}%) -> {a.out}")


if __name__ == "__main__":
    main()
