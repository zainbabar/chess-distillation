"""Stockfish analysis for engine-grounded traces (night1 stage A).

For each puzzle: Stockfish's top-3 candidate moves (multipv), each with its principal line (up to
--pv-plies half-moves) and evaluation from the solver's point of view. The Lichess answer is always
included (analysed on its own if it isn't in the top 3). Records whether Stockfish's best move equals
the Lichess answer.

Usage: .venv/bin/python engine_analysis.py --puzzles pool_night1_engine98.jsonl \
           --out results/night1_engine_analysis.jsonl [--depth 20] [--workers 16]
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine

STOCKFISH = "/usr/games/stockfish"
_local = threading.local()
_engines, _lock = [], threading.Lock()


def _engine():
    if not hasattr(_local, "engine"):
        eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        eng.configure({"Threads": 1, "Hash": 128})
        _local.engine = eng
        with _lock:
            _engines.append(eng)
    return _local.engine


def describe(board, info, pv_plies):
    """One candidate: move, SAN, line (UCI + SAN), score from the mover's POV."""
    pv = info["pv"][:pv_plies]
    b = board.copy()
    sans = []
    for m in pv:
        sans.append(b.san(m))
        b.push(m)
    score = info["score"].pov(board.turn)
    return {
        "move": pv[0].uci(), "san": sans[0],
        "line_uci": [m.uci() for m in pv], "line_san": sans,
        "cp": score.score(), "mate": score.mate(),
        "win_pct": round(100 * score.wdl(model="lichess").expectation(), 1),
    }


def analyse(args_tuple):
    puzzle, depth, pv_plies = args_tuple
    board = chess.Board(puzzle["fen"])
    eng = _engine()
    infos = eng.analyse(board, chess.engine.Limit(depth=depth), multipv=3)
    cands = [describe(board, i, pv_plies) for i in infos if i.get("pv")]
    answer = puzzle["correct_move"]
    if answer not in [c["move"] for c in cands]:
        info = eng.analyse(board, chess.engine.Limit(depth=depth),
                           root_moves=[chess.Move.from_uci(answer)])
        cands.append(describe(board, info, pv_plies))
    return {
        "puzzle_id": puzzle["puzzle_id"], "fen": puzzle["fen"], "depth": depth,
        "correct_move": answer, "sf_best": cands[0]["move"],
        "sf_best_is_answer": cands[0]["move"] == answer, "candidates": cands,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--pv-plies", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    puzzles = [json.loads(l) for l in open(args.puzzles)]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(analyse, [(p, args.depth, args.pv_plies) for p in puzzles]))
    finally:
        for eng in _engines:
            eng.quit()
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    agree = sum(r["sf_best_is_answer"] for r in results)
    print(f"wrote {len(results)} analyses to {args.out}; Stockfish best = Lichess answer in "
          f"{agree}/{len(results)}")


if __name__ == "__main__":
    main()
