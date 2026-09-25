"""Analyse the student pilot: same base (Qwen3-1.7B + LoRA), same ~1.9k training puzzles, different training
text (answer only / code-built explanation / LLM-written FDF explanation), evaluated on the 500-puzzle test set.

Prints a markdown report (also written to results/pilot_report.md): accuracy, illegal moves, format,
full-line accuracy, puzzle rating with bootstrap CI, per-band accuracy, and paired comparisons (exact
McNemar test on the puzzles where two arms disagree).
"""
import json
import statistics as st
from collections import Counter
from math import comb
from pathlib import Path

from puzzle_rating import rate

ARMS = [("p_answer", "answer only"), ("p_code", "code-built explanation"), ("p_llm", "LLM (FDF-low) explanation")]
BANDS = [f"{lo}-{lo + 200}" for lo in range(800, 2200, 200)]


def load(arm):
    p = Path(f"results/student_pilot_{arm}_nothink.jsonl")
    return {json.loads(l)["puzzle_id"]: json.loads(l) for l in open(p)} if p.exists() else {}


def mcnemar(a, b):
    """Exact two-sided McNemar p-value from discordant counts a (only arm1 right) and b (only arm2 right)."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    data = {arm: load(arm) for arm, _ in ARMS}
    L = ["# Student pilot report", "",
         "Qwen3-1.7B + LoRA r64, 3 epochs, the same 1,892 training puzzles (pool_pilot_llm.jsonl, 800–2200),"
         " differing only in the training text. Test: pilot_set.jsonl (500 puzzles, held out), greedy decoding,"
         " the teacher's P1L prompt. Zero-shot baseline: 1/140.", "",
         "| Arm | n | Correct | Illegal | No answer | Exact format | Legal line | Full line right | Mean tokens | Puzzle rating (95% CI) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, label in ARMS:
        R = list(data[arm].values())
        if not R:
            L.append(f"| {label} | (not run) |" + " |" * 8)
            continue
        c = Counter(r["status"] for r in R)
        rt = rate([{"rating": r["rating"], "status": r["status"]} for r in R])
        L.append(f"| {label} | {len(R)} | **{c['correct']} ({100 * c['correct'] / len(R):.1f}%)** | {c['illegal']} | "
                 f"{c['parse_fail'] + c['truncated']} | {sum(bool(r.get('final_line_strict')) for r in R)} | "
                 f"{sum(bool(r.get('line_legal')) for r in R)} | {sum(bool(r.get('full_line_correct')) for r in R)} | "
                 f"{st.mean((r.get('usage') or {}).get('completion_tokens', 0) for r in R):.0f} | "
                 f"{rt['rating']} ({rt['ci_low']}–{rt['ci_high']}) |")
    L += ["", "Per band (correct of ~71):", "", "| Arm | " + " | ".join(BANDS) + " |", "|---|" + "---|" * len(BANDS)]
    for arm, label in ARMS:
        R = list(data[arm].values())
        if R:
            L.append(f"| {label} | " + " | ".join(str(sum(r["status"] == "correct" for r in R if r["rating_band"] == b))
                                                for b in BANDS) + " |")
    L += ["", "Paired comparisons (same 500 puzzles; exact McNemar):", ""]
    for i in range(len(ARMS)):
        for j in range(i + 1, len(ARMS)):
            (a1, l1), (a2, l2) = ARMS[i], ARMS[j]
            ids = set(data[a1]) & set(data[a2])
            if not ids:
                continue
            only1 = sum(data[a1][k]["status"] == "correct" and data[a2][k]["status"] != "correct" for k in ids)
            only2 = sum(data[a2][k]["status"] == "correct" and data[a1][k]["status"] != "correct" for k in ids)
            L.append(f"- {l1} vs {l2}: only the first right on {only1}, only the second on {only2} "
                     f"(p = {mcnemar(only1, only2):.3f})")
    L += ["", "Reference: the teacher gpt-oss-120b, low effort, P1L prompt: 45% on the 140-puzzle subset "
          "(rating ~1425); FEN + legal moves on these 500: 31.2%."]
    Path("results/pilot_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
