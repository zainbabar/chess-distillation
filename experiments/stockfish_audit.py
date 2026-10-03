"""Is first-move grading fair to every model? Stockfish re-checks every wrong (legal) answer of the current models.

Same rule as the original check on the first teacher run (src/stockfish_grade.py): Stockfish scores the model's move
and the Lichess move in the same position at depth 18, each search restricted to that one move; a wrong answer is an
equally good alternative ("also_correct") if both force mate, or if it loses at most 5 win-percentage points against
the Lichess move. Otherwise small (< 10), mistake (10-30) or blunder (>= 30).

    python experiments/stockfish_audit.py            -> results/stockfish_audit.jsonl + results/stockfish_audit.md
"""
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stockfish_grade as sg  # noqa: E402
from path_A_report import load  # noqa: E402

DEPTH = 18
RUNS = [  # (label, puzzle file, graded outputs)
    ("teacher, medium effort (dev)", "test_set.jsonl", "results/test500_med_formatP1L.jsonl"),
    ("teacher, medium effort (fresh)", "fresh_test_set.jsonl", "results/fresh_med_formatP1L.jsonl"),
    ("teacher, low effort (fresh)", "fresh_test_set.jsonl", "results/fresh_low_formatP1L.jsonl"),
    ("student A (dev)", "test_set.jsonl", "results/student_path_A_nothink.jsonl"),
    ("student A (fresh)", "fresh_test_set.jsonl", "results/fresh_student_path_A.jsonl"),
    ("student A + 200k (dev)", "test_set.jsonl", "results/student_path_A200k_nothink.jsonl"),
    ("student A + 200k (fresh)", "fresh_test_set.jsonl", "results/fresh_student_path_A200k.jsonl"),
    ("student B (dev)", "test_set.jsonl", "results/student_path_B_nothink.jsonl"),
    ("student B (fresh)", "fresh_test_set.jsonl", "results/fresh_student_path_B.jsonl"),
]


def job(args):
    label, p, r = args
    m = sg._eval_move((p["fen"], r["parsed_move"], DEPTH))
    s = sg._eval_move((p["fen"], p["correct_move"], DEPTH))
    return {"run": label, "puzzle_id": p["puzzle_id"], "model_move": r["parsed_move"], "solution": p["correct_move"],
            "model_eval": m, "solution_eval": s, "verdict": sg.verdict(m, s)}


def main():
    jobs = []
    for label, pf, rf in RUNS:
        P = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open(pf)}
        R = load(rf)[0]
        jobs += [(label, P[i], r) for i, r in R.items() if r["status"] == "wrong" and r.get("parsed_move")]
    print(f"{len(jobs)} wrong answers to check at depth {DEPTH}", flush=True)
    with ThreadPoolExecutor(12) as ex:
        out = list(ex.map(job, jobs))
    for eng in sg._engines:
        eng.quit()
    with open("results/stockfish_audit.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    L = ["# Stockfish re-check of every wrong answer (current models)", "",
         f"Rule as in the original check: depth {DEPTH}; a wrong answer is an equally good alternative if both moves force "
         "mate or it loses at most 5 win-% points against the Lichess move.", "",
         "| Run | Wrong answers checked | Equally good alternative | Small (< 10) | Mistake (10-30) | Blunder (>= 30) |",
         "|---|---|---|---|---|---|"]
    for label, _, _ in RUNS:
        c = Counter(o["verdict"] for o in out if o["run"] == label)
        L.append(f"| {label} | {sum(c.values())} | {c['also_correct']} | {c['small']} | {c['mistake']} | {c['blunder']} |")
    alts = [o for o in out if o["verdict"] == "also_correct"]
    L += ["", f"Total: {len(out)} wrong answers, {len(alts)} equally good alternatives."]
    for o in alts:
        L.append(f"- {o['run']}: {o['puzzle_id']} played {o['model_move']} (solution {o['solution']}); "
                 f"win% {o['model_eval']['win_pct']} vs {o['solution_eval']['win_pct']}")
    Path("results/stockfish_audit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
