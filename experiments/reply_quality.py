"""Are the opponent replies in a model's written line real defenses? An engine check of "invented" lines.

For every multi-move puzzle (solution of 3+ moves) where the model's first move is right and its written line contains a
legal second move (the opponent's reply), Stockfish scores, from the defender's side at depth 18: the best defense in
that position (full search) and the written reply (search restricted to it). The reply is a **real defense** if it loses
at most 5 win-percentage points against the best defense (the same tolerance as the grading check in
src/stockfish_grade.py). Also reported: how often the reply is the Lichess reply, and the median loss.

    python experiments/reply_quality.py   -> results/reply_quality.jsonl + results/reply_quality.md
"""
import json
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stockfish_grade as sg  # noqa: E402
from path_A_report import load  # noqa: E402

DEPTH = 18
OUT_STEM = "results/reply_quality"  # <stem>.jsonl and <stem>.md; other scripts may point it elsewhere
TOL = sg.ALT_TOLERANCE  # 5 win-% points
DEV, FRESH = "test_set.jsonl", "fresh_test_set.jsonl"
RUNS = [  # (label, puzzle file, outputs); "LICHESS" = the real solutions, a check of the method (should be ~100%)
    ("dev: Lichess solution (method check)", DEV, "LICHESS"),
    ("fresh: Lichess solution (method check)", FRESH, "LICHESS"),
    ("dev: student B (before RL)", DEV, "results/student_path_B_nothink.jsonl"),
    ("dev: B + RL answer-only", DEV, "results/student_rlL_B_final_nothink.jsonl"),
    ("dev: B + RL truth v1", DEV, "results/student_rlT_B_final_nothink.jsonl"),
    ("dev: B + RL strict v2", DEV, "results/student_rlT2_B_final_nothink.jsonl"),
    ("dev: B + RL v3", DEV, "results/student_rlT3_B_final_nothink.jsonl"),
    ("dev: B + RL v4, step 130", DEV, "results/student_rlT4_B_step130_nothink.jsonl"),
    ("dev: B + RL v4, step 260", DEV, "results/student_rlT4_B_step260_nothink.jsonl"),
    ("dev: B + RL v4, final", DEV, "results/student_rlT4_B_final_nothink.jsonl"),
    ("dev: student A", DEV, "results/student_path_A_nothink.jsonl"),
    ("dev: student A + 200k", DEV, "results/student_path_A200k_nothink.jsonl"),
    ("dev: teacher, medium effort", DEV, "results/test500_med_formatP1L.jsonl"),
    ("fresh: student B", FRESH, "results/fresh_student_path_B.jsonl"),
    ("fresh: B + RL v4, final", FRESH, "results/fresh_student_rlT4_B_final.jsonl"),
    ("fresh: student A", FRESH, "results/fresh_student_path_A.jsonl"),
    ("fresh: student A + 200k", FRESH, "results/fresh_student_path_A200k.jsonl"),
    ("fresh: teacher, medium effort", FRESH, "results/fresh_med_formatP1L.jsonl"),
]
_best_cache, _lock = {}, threading.Lock()


def win(info, board):
    return 100 * info["score"].pov(board.turn).wdl(model="lichess").expectation()


def best_defense(fen):
    with _lock:
        if fen in _best_cache:
            return _best_cache[fen]
    board = chess.Board(fen)
    info = sg._engine().analyse(board, chess.engine.Limit(depth=DEPTH))
    val = (win(info, board), info["pv"][0].uci())
    with _lock:
        _best_cache[fen] = val
    return val


def job(args):
    label, p, r = args
    board = chess.Board(p["fen"])
    board.push_uci(p["correct_move"])  # position after the (right) first move; the defender is to move
    reply = r["line_moves"][1]
    best_w, best_uci = best_defense(board.fen())
    info = sg._engine().analyse(board, chess.engine.Limit(depth=DEPTH), root_moves=[chess.Move.from_uci(reply)])
    reply_w = win(info, board)
    loss = max(0.0, best_w - reply_w)
    return {"run": label, "puzzle_id": p["puzzle_id"], "reply": reply, "lichess_reply": p["full_solution"][1],
            "engine_best": best_uci, "best_win": round(best_w, 2), "reply_win": round(reply_w, 2),
            "loss": round(loss, 2), "real_defense": loss <= TOL, "within_10": loss <= 10}


def eligible(p, r):
    mv = r.get("line_moves") or []
    if r["status"] != "correct" or len(p["full_solution"]) < 3 or len(mv) < 2 or mv[0] != p["correct_move"]:
        return False
    b = chess.Board(p["fen"])
    b.push_uci(p["correct_move"])
    try:
        return chess.Move.from_uci(mv[1]) in b.legal_moves
    except ValueError:
        return False


def main():
    jobs, counts = [], {}
    for label, pf, rf in RUNS:
        P = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open(pf)}
        R = ({i: {"status": "correct", "line_moves": p["full_solution"]} for i, p in P.items()} if rf == "LICHESS"
             else load(rf)[0])
        if not R:
            continue
        right_multi = [i for i, r in R.items() if r["status"] == "correct" and len(P[i]["full_solution"]) >= 3]
        js = [(label, P[i], R[i]) for i in right_multi if eligible(P[i], R[i])]
        counts[label] = (len(right_multi), len(js))
        jobs += js
    print(f"{len(jobs)} written replies to check at depth {DEPTH}", flush=True)
    with ThreadPoolExecutor(12) as ex:
        out = list(ex.map(job, jobs))
    for eng in sg._engines:
        eng.quit()
    with open(OUT_STEM + ".jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    L = ["# Are the opponent replies in written lines real defenses? (Stockfish, depth 18)", "",
         "Multi-move puzzles where the first move is right. A reply is a real defense if it loses at most 5 win-% points",
         "against Stockfish's best defense, from the defender's side. The Lichess rows score the real solutions the same way:",
         "they are the ceiling for this method (some positions are lost whatever the defender plays).", "",
         "| Run | Right first move (multi-move) | Lines with a legal reply | Real defense (≤ 5) | Within 10 | Lichess reply | Median loss (win-%) |",
         "|---|---|---|---|---|---|---|"]
    for label, _, _ in RUNS:
        if label not in counts:
            continue
        rs = [o for o in out if o["run"] == label]
        n_right, n = counts[label]
        real = sum(o["real_defense"] for o in rs)
        lich = sum(o["reply"] == o["lichess_reply"] for o in rs)
        w10 = sum(o["within_10"] for o in rs)
        med = statistics.median([o["loss"] for o in rs]) if rs else float("nan")
        pct = lambda x: f"{x}/{n} ({100 * x / n:.0f}%)" if n else "–"  # noqa: E731
        L.append(f"| {label} | {n_right} | {n} | {pct(real)} | {pct(w10)} | {pct(lich)} | {med:.1f} |")
    Path(OUT_STEM + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
