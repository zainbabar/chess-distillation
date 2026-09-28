"""Step verification for structured traces (formats S and E): check every step, not just the answer.

Per answer: parse the structured write-up (structured.py), then check
  - structure_ok:      all sections present
  - candidates_legal:  every listed candidate is a legal move
  - each LINE:         legal from the puzzle position; its VERDICT agrees with Stockfish's evaluation
                       at the end of the line (solver's point of view, Lichess win%):
                       WINS >= 70, LOSES <= 30, EQUAL 30-70, UNCLEAR always accepted
  - consistent:        CHOSEN == FINAL_MOVE == first move of FINAL_LINE, and a LINE starts with it
Labels: answer_correct (final move matches Lichess, from the run's grading), process_verified (every
check passes), severity: none | minor (problem only in a side line / unchosen candidate) | serious
(missing structure, inconsistent, or the chosen line is illegal or has a wrong verdict).

Usage: .venv/bin/python verify_trace.py --results results/night1_A_formatE.jsonl results/night1_B_formatS.jsonl \
           --out results/night1_verified.jsonl [--depth 14] [--workers 16]
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine

from structured import parse_structured

STOCKFISH = "/usr/games/stockfish"
_local = threading.local()
_engines, _lock = [], threading.Lock()


def _engine():
    if not hasattr(_local, "engine"):
        eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        eng.configure({"Threads": 1, "Hash": 64})
        _local.engine = eng
        with _lock:
            _engines.append(eng)
    return _local.engine


def win_pct_after(fen, moves, solver, depth):
    """Solver's win% (0-100) after playing `moves` from `fen`."""
    board = chess.Board(fen)
    for u in moves:
        board.push_uci(u)
    if board.is_checkmate():
        return 0.0 if board.turn == solver else 100.0
    if board.is_game_over():
        return 50.0
    info = _engine().analyse(board, chess.engine.Limit(depth=depth))
    return round(100 * info["score"].pov(solver).wdl(model="lichess").expectation(), 1)


def verdict_ok(verdict, win):
    return {"WINS": win >= 70, "LOSES": win <= 30, "EQUAL": 30 < win < 70}.get(verdict, True)


def legal_prefix(fen, moves):
    if not moves:
        return False
    board = chess.Board(fen)
    for u in moves:
        try:
            board.push(board.parse_uci(u))
        except ValueError:
            return False
    return True


def verify(args_tuple):
    rec, depth = args_tuple
    fen = rec["fen"]
    solver = chess.Board(fen).turn
    s = parse_structured(rec.get("content"), fen)
    out = {"puzzle_id": rec["puzzle_id"], "format": rec["format"], "rating": rec["rating"],
           "rating_band": rec["rating_band"], "status": rec["status"],
           "answer_correct": rec["status"] == "correct", "structure_ok": s["structure_ok"],
           "n_candidates": len(s["candidates"]), "n_lines": len(s["lines"])}
    legal_moves = {m.uci() for m in chess.Board(fen).legal_moves}
    cand_ok = [c in legal_moves for c in s["candidates"]]
    out["candidates_legal"] = bool(cand_ok) and all(cand_ok)

    chosen = s["final_move"]
    line_results = []
    for ln in s["lines"]:
        r = {"n": ln["n"], "moves": ln["moves"], "verdict": ln["verdict"], "legal": legal_prefix(fen, ln["moves"])}
        if r["legal"] and ln["moves"]:
            r["win_pct"] = win_pct_after(fen, ln["moves"], solver, depth)
            r["verdict_ok"] = verdict_ok(ln["verdict"], r["win_pct"])
        else:
            r["win_pct"], r["verdict_ok"] = None, False
        r["is_chosen"] = bool(ln["moves"]) and ln["moves"][0] == chosen
        line_results.append(r)
    out["lines"] = line_results
    chosen_lines = [r for r in line_results if r["is_chosen"]]
    out["consistent"] = bool(chosen and s["chosen"] == chosen and s["final_line"]
                             and s["final_line"][0] == chosen and chosen_lines)
    out["lines_legal"] = bool(line_results) and all(r["legal"] for r in line_results)
    out["verdicts_ok"] = bool(line_results) and all(r["verdict_ok"] for r in line_results)
    out["chosen_line_ok"] = bool(chosen_lines) and all(r["legal"] and r["verdict_ok"] for r in chosen_lines)

    if not s["structure_ok"] or not out["consistent"] or not out["chosen_line_ok"]:
        sev = "serious"
    elif not (out["candidates_legal"] and out["lines_legal"] and out["verdicts_ok"]):
        sev = "minor"
    else:
        sev = "none"
    out["severity"] = sev
    out["process_verified"] = sev == "none"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    recs = []
    import os
    for path in args.results:
        if not os.path.exists(path):
            print(f"skipping missing {path}")
            continue
        recs += [json.loads(l) for l in open(path)]
    recs = [r for r in recs if r.get("status") != "error"]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(verify, [(r, args.depth) for r in recs]))
    finally:
        for eng in _engines:
            eng.quit()
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    for fmt in sorted({r["format"] for r in results}):
        rs = [r for r in results if r["format"] == fmt]
        n = len(rs)
        print(f"[{fmt}] {n} answers: answer_correct {sum(r['answer_correct'] for r in rs)}, "
              f"process_verified {sum(r['process_verified'] for r in rs)}, "
              f"both {sum(r['answer_correct'] and r['process_verified'] for r in rs)}, "
              f"severity none/minor/serious {sum(r['severity'] == 'none' for r in rs)}/"
              f"{sum(r['severity'] == 'minor' for r in rs)}/{sum(r['severity'] == 'serious' for r in rs)}")


if __name__ == "__main__":
    main()
