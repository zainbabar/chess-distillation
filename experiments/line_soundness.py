"""Are the written lines sound? An engine judgement of whole lines, fairer than an exact match with Lichess.

"Full line right" elsewhere means the written line equals the Lichess solution move for move, which marks a different
line that also wins as wrong. Here Stockfish (depth 18) judges each line written by a model whose first move is right,
on multi-move puzzles (solution of 3+ moves):
- every move must be legal in order;
- each solver move must be as good as Stockfish's best move in that position (it also forces mate, or loses at most 5
  win-% points), the rule of src/stockfish_grade.py;
- each opponent move must be a real defense (at most 5 win-% points worse than the best defense, defender's side);
- **sound full line** = all of that, and the line is at least as long as the solution or ends in checkmate.
The Lichess solutions go through the same check as the method check (they should pass almost always).

    python experiments/line_soundness.py   -> results/line_soundness.{jsonl,md}
"""
import json
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
OUT_STEM = "results/line_soundness"  # <stem>.jsonl and <stem>.md; other scripts may point it elsewhere
TOL = sg.ALT_TOLERANCE
DEV, FRESH = "test_set.jsonl", "fresh_test_set.jsonl"
MODELS = [  # (label, development outputs, fresh outputs)
    ("Lichess solution (method check)", "LICHESS", "LICHESS"),
    ("teacher, low effort", "results/test500_formatP1L.jsonl", "results/fresh_low_formatP1L.jsonl"),
    ("teacher, medium effort", "results/test500_med_formatP1L.jsonl", "results/fresh_med_formatP1L.jsonl"),
    ("student A", "results/student_path_A_nothink.jsonl", "results/fresh_student_path_A.jsonl"),
    ("student A, seed 1", "results/student_path_A_seed1_nothink.jsonl", "results/fresh_student_path_A_seed1.jsonl"),
    ("student A + 200k", "results/student_path_A200k_nothink.jsonl", "results/fresh_student_path_A200k.jsonl"),
    ("student B", "results/student_path_B_nothink.jsonl", "results/fresh_student_path_B.jsonl"),
    ("student B, seed 1", "results/student_path_B_seed1_nothink.jsonl", "results/fresh_student_path_B_seed1.jsonl"),
    ("B answer-first", "results/student_path_Baf_nothink.jsonl", "results/fresh_student_path_Baf.jsonl"),
    ("B + RL v4", "results/student_rlT4_B_final_nothink.jsonl", "results/fresh_student_rlT4_B_final.jsonl"),
]
_cache, _lock = {}, threading.Lock()


def win(info, board):
    return 100 * info["score"].pov(board.turn).wdl(model="lichess").expectation()


def best(fen):
    with _lock:
        if fen in _cache:
            return _cache[fen]
    board = chess.Board(fen)
    v = win(sg._engine().analyse(board, chess.engine.Limit(depth=DEPTH)), board)
    with _lock:
        _cache[fen] = v
    return v


def judge(args):
    label, sname, p, moves = args
    board = chess.Board(p["fen"])
    out = {"model": label, "set": sname, "puzzle_id": p["puzzle_id"], "moves": moves, "bad_ply": None, "why": None}
    for ply, uci in enumerate(moves):
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            mv = None
        if mv is None or mv not in board.legal_moves:
            out.update(bad_ply=ply, why="illegal")
            break
        b = best(board.fen())
        played = win(sg._engine().analyse(board, chess.engine.Limit(depth=DEPTH), root_moves=[mv]), board)
        if b - played > TOL:
            out.update(bad_ply=ply, why="solver move worse than best" if ply % 2 == 0 else "not a real defense")
            break
        board.push(mv)
    out["sound"] = out["bad_ply"] is None
    out["full"] = out["sound"] and (len(moves) >= len(p["full_solution"]) or board.is_checkmate())
    out["exact"] = moves[:len(p["full_solution"])] == p["full_solution"] and len(moves) >= len(p["full_solution"])
    return out


def main():
    jobs, counts = [], {}
    for label, dev, fresh in MODELS:
        for sname, pf, rf in (("development", DEV, dev), ("fresh", FRESH, fresh)):
            P = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open(pf)}
            R = ({i: {"status": "correct", "line_moves": p["full_solution"]} for i, p in P.items()} if rf == "LICHESS"
                 else load(rf)[0])
            if not R:
                continue
            ids = [i for i, r in R.items() if r["status"] == "correct" and len(P[i]["full_solution"]) >= 3]
            counts[(label, sname)] = len(ids)
            for i in ids:
                mv = R[i].get("line_moves") or [P[i]["correct_move"]]  # judged as written; no line = just the move
                jobs.append((label, sname, P[i], mv))
    print(f"{len(jobs)} lines to judge at depth {DEPTH}", flush=True)
    with ThreadPoolExecutor(12) as ex:
        out = list(ex.map(judge, jobs))
    for eng in sg._engines:
        eng.quit()
    with open(OUT_STEM + ".jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    L = ["# Are the written lines sound? (Stockfish, depth 18)", "",
         "Multi-move puzzles where the model's first move is right. Sound full line: legal, every solver move as good as",
         "Stockfish's best (≤ 5 win-% points), every opponent move a real defense, and as long as the solution or ending in",
         "mate. Exact match = the old \"full line right\" (identical to the Lichess solution). The Lichess rows are the",
         "method check.", "",
         "| Model | Set | First move right (multi-move) | Sound full line | Exact match | First problem: illegal / weak solver move / not a real defense / too short |",
         "|---|---|---|---|---|---|"]
    for label, _, _ in MODELS:
        for sname in ("development", "fresh"):
            if (label, sname) not in counts:
                continue
            rs = [o for o in out if o["model"] == label and o["set"] == sname]
            n = counts[(label, sname)]
            full = sum(o["full"] for o in rs)
            exact = sum(o["exact"] for o in rs)
            ill = sum(o["why"] == "illegal" for o in rs)
            weak = sum(o["why"] == "solver move worse than best" for o in rs)
            nd = sum(o["why"] == "not a real defense" for o in rs)
            short = sum(o["sound"] and not o["full"] for o in rs)
            L.append(f"| {label} | {sname} | {n} | {full} ({100 * full / n:.0f}%) | {exact} | {ill} / {weak} / {nd} / {short} |")
    Path(OUT_STEM + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
