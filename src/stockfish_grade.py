"""Re-grade "wrong" pilot answers with Stockfish.

Is a wrong answer actually bad, or just not the Lichess solution? For each wrong answer, Stockfish
scores the model's move and the Lichess solution (same position, same depth, search restricted to
that one move). Scores become win chances (Lichess's formula, 0-100).

Verdicts:
  also_correct  - solution forces mate and so does the model move, or model move is within
                  ALT_TOLERANCE win-% points of the solution
  small         - loses < 10 win-% points vs the solution
  mistake       - loses 10-30
  blunder       - loses >= 30
Correct / illegal / parse_fail answers are carried over unchanged.

Output: results/<run>_stockfish.jsonl (raw result files are not modified).
Usage: .venv/bin/python stockfish_grade.py [--run pilot] [--depth 18] [--workers 16]
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine

STOCKFISH = "/usr/games/stockfish"
ALT_TOLERANCE = 5.0  # win-% points

# One Stockfish process per worker thread (threads only wait on the engines, which do the work).
# Engines are closed explicitly at the end: open engines keep worker processes alive, which is
# why a ProcessPoolExecutor version hung on shutdown.
_local = threading.local()
_engines = []
_engines_lock = threading.Lock()


def _engine():
    if not hasattr(_local, "engine"):
        eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        eng.configure({"Threads": 1, "Hash": 64})
        _local.engine = eng
        with _engines_lock:
            _engines.append(eng)
    return _local.engine


def _eval_move(args):
    """Score one move in one position, from the mover's point of view."""
    fen, uci, depth = args
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    info = _engine().analyse(board, chess.engine.Limit(depth=depth), root_moves=[move])
    score = info["score"].pov(board.turn)
    return {
        "cp": score.score(),              # None if mate
        "mate": score.mate(),             # +n: mover mates in n; -n: mover gets mated
        "win_pct": round(100 * score.wdl(model="lichess").expectation(), 2),
    }


def verdict(model_ev, sol_ev):
    if sol_ev["mate"] is not None and sol_ev["mate"] > 0:
        if model_ev["mate"] is not None and model_ev["mate"] > 0:
            return "also_correct"
    loss = sol_ev["win_pct"] - model_ev["win_pct"]
    if loss <= ALT_TOLERANCE:
        return "also_correct"
    if loss < 10:
        return "small"
    if loss < 30:
        return "mistake"
    return "blunder"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot")
    ap.add_argument("--formats", nargs="+", default=["A", "B"])
    ap.add_argument("--depth", type=int, default=18)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    rows = []
    for f in args.formats:
        rows += [json.loads(l) for l in open(f"results/{args.run}_format{f}.jsonl")]

    # Unique (fen, move) pairs to evaluate: model move + solution for every wrong answer.
    jobs = set()
    for r in rows:
        if r["status"] == "wrong":
            jobs.add((r["fen"], r["move_uci"]))
            jobs.add((r["fen"], r["correct_move"]))
    jobs = sorted(jobs)
    print(f"{len(rows)} answers, {sum(r['status'] == 'wrong' for r in rows)} wrong -> "
          f"{len(jobs)} Stockfish evaluations at depth {args.depth} with {args.workers} workers")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            evals = dict(zip(jobs, ex.map(_eval_move, [(fen, uci, args.depth) for fen, uci in jobs])))
    finally:
        for eng in _engines:
            eng.quit()

    out_path = f"results/{args.run}_stockfish.jsonl"
    with open(out_path, "w") as out:
        for r in rows:
            rec = {k: r.get(k) for k in ("puzzle_id", "format", "rating", "rating_band", "themes", "fen",
                                         "correct_move", "move_uci", "status")}
            rec["sf_depth"] = args.depth
            if r["status"] == "wrong":
                m, s = evals[(r["fen"], r["move_uci"])], evals[(r["fen"], r["correct_move"])]
                rec.update(model_eval=m, solution_eval=s,
                           win_pct_loss=round(s["win_pct"] - m["win_pct"], 2),
                           sf_verdict=verdict(m, s))
            else:
                rec["sf_verdict"] = r["status"]  # correct / illegal / parse_fail / error
            rec["sf_correct"] = rec["sf_verdict"] in ("correct", "also_correct")
            out.write(json.dumps(rec) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
