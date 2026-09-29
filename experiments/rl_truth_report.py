"""RL for student B under five rewards (THE PATH step 3b) -> results/rl_truth4_report.md (v4 present), else the v3 / v2 /
v1 report names used earlier. Pass --out to write somewhere else.

Compares, on the 500 held-out test puzzles: B after SFT, and B after RL with each reward (ckpt/rlL_B, rlT_B, rlT2_B,
rlT3_B, rlT4_B). All RL runs: same start, same 3,120 puzzles in the same order, same settings; only the reward differs.
Checkpoints at 130/260/390 steps.
"""
import argparse
import chess
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from analyze_pilot import mcnemar
from claim_check import check
from path_A_report import load
from puzzle_rating import rate

MATE_RE = re.compile(r"checkmate|is mate\b|mate in \w+|forced mate|delivers mate|mating", re.I)
ARMS = [("B after SFT", "results/student_path_B_nothink.jsonl", "sft")]
for ck in ("step130", "step260", "final"):
    n = {"step130": "130", "step260": "260", "final": "390"}[ck]
    ARMS.append((f"outcome-only RL, {n} steps", f"results/student_rlL_B_{ck}_nothink.jsonl", f"L{n}"))
    ARMS.append((f"truth-aware RL, {n} steps", f"results/student_rlT_B_{ck}_nothink.jsonl", f"T{n}"))
    ARMS.append((f"strict truth-aware RL (v2), {n} steps", f"results/student_rlT2_B_{ck}_nothink.jsonl", f"S{n}"))
    ARMS.append((f"calc-rewarding truth RL (v3), {n} steps", f"results/student_rlT3_B_{ck}_nothink.jsonl", f"U{n}"))
    ARMS.append((f"v4: replays the line, {n} steps", f"results/student_rlT4_B_{ck}_nothink.jsonl", f"Q{n}"))


def legal_replies(R, P):
    """Written lines whose first move and the opponent's reply are both legal (a 1-move line doesn't count)."""
    n = 0
    for i, r in R.items():
        mv = r.get("line_moves") or []
        if len(mv) < 2:
            continue
        b = chess.Board(P[i]["fen"])
        try:
            for m in mv[:2]:
                move = chess.Move.from_uci(m)
                if move not in b.legal_moves:
                    break
                b.push(move)
            else:
                n += 1
        except ValueError:
            pass
    return n


def stats(R, P):
    rs = list(R.items())
    st = Counter(r["status"] for _, r in rs)
    right = [(i, r) for i, r in rs if r["status"] == "correct"]
    clean = sum(check((r.get("content") or "").split("FINAL_LINE")[0], P[i])["clean"] for i, r in right)
    nonmate = [(i, r) for i, r in rs if "mate" not in " ".join(P[i]["themes"])]
    fmate = sum(bool(MATE_RE.search((r.get("content") or "").split("FINAL_LINE")[0])) for _, r in nonmate)
    bodies = [(i, (r.get("content") or "").split("FINAL_LINE")[0]) for i, r in rs]
    chk = [check(b, P[i]) for i, b in bodies]
    return {"n": len(rs), "correct": st["correct"], "rating": rate([r for _, r in rs]),
            "legal": sum(bool(r.get("line_legal")) for _, r in rs), "full": sum(bool(r.get("full_line_correct")) for _, r in rs),
            "line_len": mean(len(r.get("line_moves") or []) for _, r in rs),
            "tokens": mean(r["usage"]["completion_tokens"] for _, r in rs if r.get("usage")),
            "clean_right": (clean, len(right)), "false_mate": (fmate, len(nonmate)),
            "trunc": st["truncated"], "parse": st["parse_fail"],
            "words": mean(len(b.split()) for _, b in bodies), "claims": mean(c["n_claims"] for c in chk),
            "flagged": sum(any(x[0] == "move" for x in c["soft"]) for c in chk),
            "padded": sum(len(r.get("line_moves") or []) > (r.get("solution_len") or 0) for _, r in rs),
            "replies": legal_replies(R, P)}


def pair(a, b):
    ids = set(a) & set(b)
    x = sum(a[i]["status"] == "correct" and b[i]["status"] != "correct" for i in ids)
    y = sum(b[i]["status"] == "correct" and a[i]["status"] != "correct" for i in ids)
    return f"{x} vs {y}, p = {mcnemar(x, y):.2g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write the report here instead of the default results/ name")
    args = ap.parse_args()
    P = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("test_set.jsonl")}
    D, S = {}, {}
    for label, path, key in ARMS:
        R, _ = load(path)
        if R:
            D[key], S[key] = R, stats(R, P)
    L = ["# RL for student B — can RL make imitated reasoning real?", "",
         "RL runs from the same B checkpoint on the same 3,120 fresh puzzles, identical settings (390 steps × 8 puzzles ×",
         "8 samples, lr 2e-6, no KL); only the reward differs:", "",
         "- **Outcome-only**: right move 1, else 0.",
         "- **Truth-aware (v1)**: +1 right move; +0.5 × fraction of the solution line written correctly; −0.5 if the claim",
         "  checker finds a hard false claim (piece not on that square, false \"wins the X\", false mate); −0.5 if the",
         "  explanation is under 25 words.",
         "- **Strict (v2)**: line credit 0.5 × correct prefix / max(line length, solution length); −0.5 hard false claim;",
         "  −0.5 if the text names an invented move; −0.5 if it has fewer than 2 verified moves, is under 40 words, or has",
         "  no answer.",
         "- **v3**: as v2 with line credit 1.0, invented-move penalty −0.25, and the floor counting distinct moves.",
         "- **v4**: as v3, plus +0.25 for a complete, playable written line and −0.25 for one that breaks down when",
         "  replayed or stops short; move sequences in the text must be playable in order (`src/rewards.py`).", "",
         "Test: 500 held-out puzzles, greedy, max 1,024 new tokens.",
         "\"False mate claims\" = explanations that claim checkmate or a forced mate (checkmate, mate in N, forced mate, …)",
         "on puzzles with no mate in the solution.", "",
         "| Model | Correct | Rating (95% CI) | Full line right | Mean line length (moves) | Right answers with no false claim | "
         "False mate claims (non-mate puzzles) | Mean tokens |",
         "|---|---|---|---|---|---|---|---|"]
    V = ["", "Honesty checks (is it truer, or vaguer / padded?):", "",
         "| Model | Words | Checkable claims per text | Texts naming an invented move | Legal line | "
         "Lines with a legal reply | Line longer than solution |",
         "|---|---|---|---|---|---|---|"]
    for label, _, key in ARMS:
        if key in S:
            s = S[key]
            cr, fm = s["clean_right"], s["false_mate"]
            L.append(f"| {label} | **{s['correct']}/{s['n']} ({100 * s['correct'] / s['n']:.1f}%)** | {s['rating']['rating']} "
                     f"({s['rating']['ci_low']}–{s['rating']['ci_high']}) | {s['full']} | {s['line_len']:.2f} | "
                     f"{cr[0]}/{cr[1]} ({100 * cr[0] / max(1, cr[1]):.0f}%) | {fm[0]}/{fm[1]} ({100 * fm[0] / max(1, fm[1]):.0f}%) | "
                     f"{s['tokens']:.0f} |")
            V.append(f"| {label} | {s['words']:.0f} | {s['claims']:.1f} | {s['flagged']}/{s['n']} | {s['legal']} | "
                     f"{s['replies']} | {s['padded']} |")
    L += V
    L += ["", "Paired accuracy comparisons (exact McNemar):", ""]
    for n in ("130", "260", "390"):
        for other, name in (("sft", "SFT"), (f"U{n}", f"v3 {n}"), (f"S{n}", f"strict v2 {n}"), (f"L{n}", f"outcome-only {n}")):
            if f"Q{n}" in D and other in D:
                L.append(f"- v4 {n} vs {name}: {pair(D[f'Q{n}'], D[other])}")
        if f"U{n}" in D and "sft" in D:
            L.append(f"- v3 {n} vs SFT: {pair(D[f'U{n}'], D['sft'])}")
        if f"U{n}" in D and f"S{n}" in D:
            L.append(f"- v3 {n} vs strict v2 {n}: {pair(D[f'U{n}'], D[f'S{n}'])}")
        if f"U{n}" in D and f"L{n}" in D:
            L.append(f"- v3 {n} vs outcome-only {n}: {pair(D[f'U{n}'], D[f'L{n}'])}")
        if f"S{n}" in D and "sft" in D:
            L.append(f"- strict v2 {n} vs SFT: {pair(D[f'S{n}'], D['sft'])}")
        if f"S{n}" in D and f"L{n}" in D:
            L.append(f"- strict v2 {n} vs outcome-only {n}: {pair(D[f'S{n}'], D[f'L{n}'])}")
        if f"S{n}" in D and f"T{n}" in D:
            L.append(f"- strict v2 {n} vs truth v1 {n}: {pair(D[f'S{n}'], D[f'T{n}'])}")
        if f"T{n}" in D and "sft" in D:
            L.append(f"- truth-aware {n} vs SFT: {pair(D[f'T{n}'], D['sft'])}")
        if f"T{n}" in D and f"L{n}" in D:
            L.append(f"- truth-aware {n} vs outcome-only {n}: {pair(D[f'T{n}'], D[f'L{n}'])}")
    A, _ = load("results/student_path_A_nothink.jsonl")
    for n in ("390",):
        if f"T{n}" in D and A:
            L.append(f"- truth-aware {n} vs A after SFT (answers only, 284): {pair(D[f'T{n}'], A)}")
        if f"S{n}" in D and A:
            L.append(f"- strict v2 {n} vs A after SFT (answers only, 284): {pair(D[f'S{n}'], A)}")
        if f"U{n}" in D and A:
            L.append(f"- v3 {n} vs A after SFT (answers only, 284): {pair(D[f'U{n}'], A)}")
        if f"Q{n}" in D and A:
            L.append(f"- v4 {n} vs A after SFT (answers only, 284): {pair(D[f'Q{n}'], A)}")
    for tag, p in (("truth v1", Path("results/rlT_B.log")), ("strict v2", Path("results/rlT2_B.log")),
                   ("v3", Path("results/rlT3_B.log")), ("v4", Path("results/rlT4_B.log"))):
      if p.exists():
        s = p.read_text(errors="ignore")
        rew = [float(x) for x in re.findall(r"'reward': '([-0-9.e]+)'", s)]
        zero = [float(x) for x in re.findall(r"'frac_reward_zero_std': '([0-9.e]+)'", s)]
        prog = re.findall(r"rl step (\d+)/\d+ sample-accuracy so far ([0-9.]+) .*?claim errors ([0-9.]+)", s)
        if rew:
            L += ["", f"## Training ({tag} run)", "",
                  "- mean reward in 30-step blocks: " + " → ".join(f"{mean(rew[i:i + 30]):.3f}" for i in range(0, len(rew), 30)),
                  f"- groups with no learning signal: {mean(zero[:30]):.0%} → {mean(zero[-30:]):.0%} "
                  f"(outcome-only run: 28% → 66%)"]
        if prog:
            L.append("- running sample accuracy / claim-error rate: " +
                     ", ".join(f"step {a}: {float(b):.3f} / {float(c):.3f}" for a, b, c in prog[::13] + prog[-1:]))
    out = args.out or ("results/rl_truth4_report.md" if any(k.startswith("Q") for k in D) else
           "results/rl_truth3_report.md" if any(k.startswith("U") for k in D) else
           "results/rl_truth2_report.md" if any(k.startswith("S") for k in D) else "results/rl_truth_report.md")
    if Path(out).exists() and "Sanity notes" in Path(out).read_text():
        out = out.replace(".md", ".regenerated.md")  # never overwrite a report that has my notes
    Path(out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
