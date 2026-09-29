"""The teacher at low vs medium reasoning effort on the 500 test puzzles, the students against both, and how the prompt
changes the teacher (position only / + legal moves / + board description) -> results/teacher_report.md.

Teacher: gpt-oss-120b, P1L prompt (board description + "calculate the forcing line"), one attempt per puzzle, max 8,192
tokens at low effort and 32,768 at medium. Students: Qwen3-1.7B full fine-tunes (A = answers only, B = teacher
explanations), greedy, same prompt.
"""
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from analyze_pilot import mcnemar
from path_A_report import load
from puzzle_rating import rate

BANDS = ["800-1000", "1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", "2000-2200"]
MODELS = [("teacher gpt-oss-120b, low effort", "results/test500_formatP1L.jsonl"),
          ("teacher gpt-oss-120b, medium effort", "results/test500_med_formatP1L.jsonl"),
          ("student A: answers only (1.7B)", "results/student_path_A_nothink.jsonl"),
          ("student B: teacher explanations (1.7B)", "results/student_path_B_nothink.jsonl"),
          ("student A + 200k more puzzles (1.7B)", "results/student_path_A200k_nothink.jsonl")]
PROMPTS = [("position only (FEN + side to move)", "results/pilot_formatA.jsonl"),
           ("+ list of legal moves", "results/pilot_formatB.jsonl"),
           ("+ board description + \"calculate the forcing line\" (P1L)", "results/test500_formatP1L.jsonl")]


def _mates(board, move):
    after = board.copy()
    after.push(move)
    return after.is_checkmate()


def pair(a, b):
    """McNemar counts and p, plus the gap in points with a 95% paired bootstrap interval (puzzles resampled together)."""
    ids = sorted(set(a) & set(b))
    x = sum(a[i]["status"] == "correct" and b[i]["status"] != "correct" for i in ids)
    y = sum(b[i]["status"] == "correct" and a[i]["status"] != "correct" for i in ids)
    d = np.array([(a[i]["status"] == "correct") - (b[i]["status"] == "correct") for i in ids], dtype=float)
    boots = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(axis=1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return f"{x} vs {y}, p = {mcnemar(x, y):.2g}; gap {100 * d.mean():+.1f} points (95% interval {lo:+.1f} to {hi:+.1f})"


def main():
    P = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("test_set.jsonl")}
    D = {name: load(path)[0] for name, path in MODELS}
    D = {name: R for name, R in D.items() if R}  # the 200k run appears once it has been evaluated
    L = ["# The teacher at low vs medium effort, and the students against both", "",
         "500 held-out test puzzles, the same prompt for every model (board description + \"calculate the forcing line\").",
         "Teacher: gpt-oss-120b, one attempt per puzzle at its default sampling settings, max 8,192 tokens at low effort and",
         "32,768 at medium. Students:",
         "Qwen3-1.7B, full fine-tune on 37,543 puzzles, greedy, max 1,024 new tokens.", "",
         "| Model | Solved | Puzzle rating (95% CI) | Illegal | No readable answer | Cut off | Legal line | "
         "Full line right | Output tokens (mean / median / max) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name in D:
        R = D[name]
        st = Counter(r["status"] for r in R.values())
        rt = rate(list(R.values()))
        toks = [r["usage"]["completion_tokens"] for r in R.values() if r.get("usage")]
        cut = sum(r.get("finish_reason") == "length" for r in R.values())
        L.append(f"| {name} | **{st['correct']}/{len(R)} ({100 * st['correct'] / len(R):.1f}%)** | {rt['rating']} "
                 f"({rt['ci_low']}–{rt['ci_high']}) | {st['illegal']} | {st['parse_fail']} | {cut} | "
                 f"{sum(bool(r.get('line_legal')) for r in R.values())} | "
                 f"{sum(bool(r.get('full_line_correct')) for r in R.values())} | "
                 f"{mean(toks):,.0f} / {median(toks):,.0f} / {max(toks):,} |")
    names = list(D)
    low, med, a, b = (D[n] for n in names[:4])
    L += ["", "Output tokens for the teacher include its hidden reasoning.", "",
          "Paired comparisons (exact McNemar: puzzles only the first solved vs only the second solved; the gap is the first's",
          "score minus the second's, in points, with a 95% paired bootstrap interval):", "",
          f"- medium vs low effort: {pair(med, low)}",
          f"- student A vs teacher, low effort: {pair(a, low)}",
          f"- student A vs teacher, medium effort: {pair(a, med)}",
          f"- student B vs teacher, low effort: {pair(b, low)}",
          f"- student B vs teacher, medium effort: {pair(b, med)}",
          f"- student A vs student B: {pair(a, b)}"]
    if len(names) > 4:
        a2 = D[names[4]]
        L += [f"- student A + 200k vs teacher, medium effort: {pair(a2, med)}",
              f"- student A + 200k vs teacher, low effort: {pair(a2, low)}",
              f"- student A + 200k vs student A: {pair(a2, a)}"]
    L += ["",
          "Solved by rating band (of 72 / 72 / 72 / 71 / 71 / 71 / 71):", "",
          "| Model | " + " | ".join(BANDS) + " |", "|---|" + "---|" * len(BANDS)]
    for name in names:
        R = D[name]
        cells = [str(sum(R[i]["status"] == "correct" for i, p in P.items() if p["rating_band"] == band)) for band in BANDS]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    # the prompt: the teacher reading the board (low effort, 8,192 tokens, the same 500 puzzles; nothing was cut off)
    Q = {name: load(path)[0] for name, path in PROMPTS}
    L += ["", "## How the prompt changes the teacher", "",
          "gpt-oss-120b at low effort, one attempt per puzzle, max 8,192 tokens, the same 500 puzzles; each row adds to the",
          "prompt above it. \"Illegal\" = the answer is not a legal move in the position.", "",
          "| Teacher prompt | Solved | Puzzle rating (95% CI) | Illegal | No readable answer | Output tokens (mean) |",
          "|---|---|---|---|---|---|"]
    for name, R in Q.items():
        st = Counter(r["status"] for r in R.values())
        rt = rate(list(R.values()))
        toks = [r["usage"]["completion_tokens"] for r in R.values() if r.get("usage")]
        L.append(f"| {name} | **{st['correct']}/{len(R)} ({100 * st['correct'] / len(R):.1f}%)** | {rt['rating']} "
                 f"({rt['ci_low']}–{rt['ci_high']}) | {st['illegal']} ({100 * st['illegal'] / len(R):.1f}%) | "
                 f"{st['parse_fail']} | {mean(toks):,.0f} |")
    fen, legal, p1l = Q.values()
    rnd = 0.0  # expected score of a uniformly random legal move (in mate-in-one puzzles any mate counts)
    for p in P.values():
        b = chess.Board(p["fen"])
        moves = list(b.legal_moves)
        mate1 = len(p["full_solution"]) == 1 and "mateIn1" in p["themes"]
        rnd += sum(m.uci() == p["correct_move"] or (mate1 and _mates(b, m))
                   for m in moves) / len(moves)
    L += ["", "Paired comparisons (exact McNemar):", "",
          f"- + legal moves vs position only: {pair(legal, fen)}",
          f"- P1L vs + legal moves: {pair(p1l, legal)}", "",
          "The last step changes two things at once (the board description and the request for the full line).", "",
          f"For scale: a random legal move would solve {100 * rnd / len(P):.1f}% of these puzzles (expected value)."]
    Path("results").mkdir(exist_ok=True)
    Path("results/teacher_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
