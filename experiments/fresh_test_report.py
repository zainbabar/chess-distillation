"""The one-time fresh test (reports/fresh_test_protocol.md) -> results/fresh_test_report.md.

Reports exactly what the protocol fixed in advance: per model, first move right, full line right, puzzle rating
(95% bootstrap interval), illegal / unreadable answers and output tokens; the three primary comparisons (A vs B, A vs
the teacher at low and at medium effort) with exact McNemar tests, a Holm correction and the gap in points with a 95%
paired bootstrap interval; secondary comparisons; and each model's score on the development set (the original 500).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chess  # noqa: E402

from analyze_pilot import mcnemar  # noqa: E402
from path_A_report import load  # noqa: E402
from puzzle_rating import rate  # noqa: E402
from teacher_report import _mates, pair  # noqa: E402

MODELS = [  # (label, fresh-test output, development-set output)
    ("student A: answers only", "results/fresh_student_path_A.jsonl", "results/student_path_A_nothink.jsonl"),
    ("student B: teacher explanations", "results/fresh_student_path_B.jsonl", "results/student_path_B_nothink.jsonl"),
    ("student A + 200k more puzzles", "results/fresh_student_path_A200k.jsonl",
     "results/student_path_A200k_nothink.jsonl"),
    ("student B + RL, reward v4", "results/fresh_student_rlT4_B_final.jsonl", "results/student_rlT4_B_final_nothink.jsonl"),
    ("untrained Qwen3-1.7B", "results/fresh_student_base.jsonl", None),
    ("teacher gpt-oss-120b, low effort", "results/fresh_low_formatP1L.jsonl", "results/test500_formatP1L.jsonl"),
    ("teacher gpt-oss-120b, medium effort", "results/fresh_med_formatP1L.jsonl", "results/test500_med_formatP1L.jsonl"),
    ("Stockfish 16, 0.1 s", "results/fresh_stockfish.jsonl", None),
]
A, B, LOW, MED = MODELS[0][0], MODELS[1][0], MODELS[5][0], MODELS[6][0]
PRIMARY = [(A, B), (A, LOW), (A, MED)]
SECONDARY = [(MODELS[2][0], MED), (MODELS[2][0], LOW), (MODELS[2][0], A), (MODELS[3][0], B), (B, LOW), (B, MED), (MED, LOW)]


def holm(ps):
    """Holm-adjusted p-values, in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def random_expected(puzzles):
    total = 0.0
    for p in puzzles:
        b = chess.Board(p["fen"])
        moves = list(b.legal_moves)
        mate1 = "mateIn1" in p["themes"]
        total += sum(m.uci() == p["correct_move"] or (mate1 and _mates(b, m)) for m in moves) / len(moves)
    return 100 * total / len(puzzles)


def report(puzzle_file, models, out):
    P = [json.loads(line) for line in open(puzzle_file)]
    ids = {p["puzzle_id"] for p in P}
    D = {}
    for label, path, _ in models:
        R = load(path)[0]
        if R:
            assert set(R) == ids, f"{path}: {len(R)} puzzles, not the {len(ids)} in {puzzle_file}"
            D[label] = R
    L = ["# Fresh test: 500 puzzles no model or experiment had seen", "",
         f"Puzzles: `{puzzle_file}`. Protocol, fixed before the run: `reports/fresh_test_protocol.md`. Same prompt,",
         "grader and settings as the main evaluation. \"First move right\" is the main score.", "",
         "| Model | First move right | Puzzle rating (95% CI) | Illegal | No readable answer | Full line right | "
         "Output tokens (mean) |", "|---|---|---|---|---|---|---|"]
    for label, R in D.items():
        st = Counter(r["status"] for r in R.values())
        rt = rate(list(R.values()))
        toks = [r["usage"]["completion_tokens"] for r in R.values() if r.get("usage")]
        full = sum(bool(r.get("full_line_correct")) for r in R.values()) if "Stockfish" not in label else "–"
        L.append(f"| {label} | **{st['correct']}/{len(R)} ({100 * st['correct'] / len(R):.1f}%)** | {rt['rating']} "
                 f"({rt['ci_low']}–{rt['ci_high']}) | {st['illegal']} | {st['parse_fail']} | {full} | "
                 f"{f'{mean(toks):,.0f}' if toks else '–'} |")
    L.append(f"| random legal move (expected) | {random_expected(P):.1f}% | | | | | |")
    have = [(x, y) for x, y in PRIMARY if x in D and y in D]
    if have:
        ps = []
        for x, y in have:
            b = sum(D[x][i]["status"] == "correct" and D[y][i]["status"] != "correct" for i in ids)
            c = sum(D[y][i]["status"] == "correct" and D[x][i]["status"] != "correct" for i in ids)
            ps.append(mcnemar(b, c))
        L += ["", "Primary comparisons (fixed in the protocol; exact McNemar, Holm-adjusted over these three; the gap is",
              "the first model's score minus the second's, in points, with a 95% paired bootstrap interval):", ""]
        for (x, y), adj in zip(have, holm(ps)):
            L.append(f"- {x} vs {y}: {pair(D[x], D[y])}; Holm-adjusted p = {adj:.2g}")
    sec = [(x, y) for x, y in SECONDARY if x in D and y in D]
    if sec:
        L += ["", "Secondary comparisons (not corrected):", ""] + [f"- {x} vs {y}: {pair(D[x], D[y])}" for x, y in sec]
    L += ["", "Development set (the original 500) vs fresh test, first move right:", "",
          "| Model | Development set | Fresh test |", "|---|---|---|"]
    for label, _, dev in models:
        if label in D and dev:
            R = load(dev)[0]
            if R:
                d = 100 * sum(r["status"] == "correct" for r in R.values()) / len(R)
                f = 100 * sum(r["status"] == "correct" for r in D[label].values()) / len(D[label])
                L.append(f"| {label} | {d:.1f}% | {f:.1f}% |")
    Path(out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", default="fresh_test_set.jsonl")
    ap.add_argument("--out", default="results/fresh_test_report.md")
    a = ap.parse_args()
    report(a.puzzles, MODELS, a.out)


if __name__ == "__main__":
    main()
