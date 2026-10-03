"""Replications and the second fresh test (results/run_replication.sh) -> results/replication_report.md. Computed from
saved outputs; missing files are skipped, so it can run on a partial queue.

(1) A + 200k replicate (A's seed-1 model continued on the 200k with seed 1) vs the original, and vs the teacher.
(2) B answer-first replicate (seed 1) vs the original, A and B.
(3) RL v4 replicate (seed 1): accuracy and the template measurements of v4 (lines exactly 3 moves, quiet second own move,
    opponent reply = Lichess, template wording, false "only legal" claims), seed 0 next to seed 1; Stockfish reply
    quality from results/replication_reply_quality.md.
(4) The second fresh test (reports/fresh_test_2_protocol.md): every model, the four primary comparisons with a Holm
    correction, on the set alone and pooled with the first fresh set (1,000 puzzles).
"""
import json
import re
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_pilot import mcnemar  # noqa: E402
from fresh_test_report import holm, random_expected  # noqa: E402
from path_A_report import load  # noqa: E402
from puzzle_rating import rate  # noqa: E402
from teacher_report import pair  # noqa: E402

R = lambda p: load(p)[0]  # noqa: E731
DEV, F1, F2 = "test_set.jsonl", "fresh_test_set.jsonl", "fresh_test_set_2.jsonl"
ONLY = re.compile(r"only legal (reply|response|move)", re.I)
TEMPLATE = re.compile(r"(improve (the|its|his) position|thus the forced sequence is)", re.I)


def puzzles(f):
    return {json.loads(l)["puzzle_id"]: json.loads(l) for l in open(f)}


def acc(D):
    c = sum(r["status"] == "correct" for r in D.values())
    return f"{c}/{len(D)} ({100 * c / len(D):.1f}%)"


def template_stats(D, P):
    """v4's template measurements on multi-move puzzles with the first move right."""
    n = three = quiet = rep = tmpl = 0
    only_claims = only_false = 0
    for pid, r in D.items():
        p = P[pid]
        if r["status"] != "correct":
            continue
        text = (r.get("content") or "").split("FINAL_LINE")[0]
        if ONLY.search(text):
            b = chess.Board(p["fen"])
            b.push_uci(p["correct_move"])
            only_claims += 1
            only_false += b.legal_moves.count() > 1
        if len(p["full_solution"]) < 3:
            continue
        n += 1
        mv = r.get("line_moves") or []
        three += len(mv) == 3
        tmpl += bool(TEMPLATE.search(text))
        if len(mv) >= 2 and mv[1] == p["full_solution"][1]:
            rep += 1
        if len(mv) >= 3:
            b = chess.Board(p["fen"])
            try:
                for m in mv[:2]:
                    b.push_uci(m)
                m3 = chess.Move.from_uci(mv[2])
                if m3 in b.legal_moves:
                    quiet += (not b.is_capture(m3)) and (not b.gives_check(m3))
            except ValueError:
                pass
    f = lambda x: f"{100 * x / n:.0f}%" if n else "–"  # noqa: E731
    return [f(three), f(quiet), f(rep), f(tmpl), f"{only_false}/{only_claims}"]


def main():
    PD, P1, P2 = puzzles(DEV), puzzles(F1), puzzles(F2)
    L = ["# Replications and the second fresh test", ""]

    # 1. A + 200k replicate
    L += ["## 1. A + 200k, replicated", "",
          "Replicate: A's second-seed model (`ckpt/path_A_seed1`) continued on the same 200k answers with seed 1, so both",
          "stages of the pipeline differ in their randomness from the original.", ""]
    sets = [("development", "results/student_path_A200k_nothink.jsonl", "results/student_path_A200k_seed1_nothink.jsonl",
             "results/test500_med_formatP1L.jsonl"),
            ("fresh 1", "results/fresh_student_path_A200k.jsonl", "results/fresh_student_path_A200k_seed1.jsonl",
             "results/fresh_med_formatP1L.jsonl"),
            ("fresh 2", "results/fresh2_student_path_A200k.jsonl", "results/fresh2_student_path_A200k_seed1.jsonl",
             "results/fresh2_med_formatP1L.jsonl")]
    for name, f0, f1, fm in sets:
        a0, a1, m = R(f0), R(f1), R(fm)
        if not a1:
            L.append(f"- {name}: replicate not evaluated yet")
            continue
        L.append(f"- {name}: original {acc(a0) if a0 else '–'}, **replicate {acc(a1)}**"
                 + (f"; replicate vs original: {pair(a1, a0)}" if a0 else "")
                 + (f"; replicate vs teacher medium: {pair(a1, m)}" if m else ""))
    L.append("")

    # 2. B answer-first replicate
    L += ["## 2. B answer-first, replicated (seed 1)", ""]
    for name, tag_dev, tag_fresh in (("development", "results/student_{}_nothink.jsonl", None),
                                     ("fresh 1", None, "results/fresh_student_{}.jsonl")):
        tpl = tag_dev or tag_fresh
        b1 = R(tpl.format("path_Baf_seed1"))
        if not b1:
            L.append(f"- {name}: replicate not evaluated yet")
            continue
        b0, A0, A1, B0, B1 = (R(tpl.format(t)) for t in ("path_Baf", "path_A", "path_A_seed1", "path_B", "path_B_seed1"))
        L.append(f"- {name}: original {acc(b0)}, **replicate {acc(b1)}**; replicate vs original: {pair(b1, b0)}")
        for lab, X in (("A seed 0", A0), ("A seed 1", A1), ("B seed 0", B0), ("B seed 1", B1)):
            if X:
                L.append(f"  - replicate vs {lab}: {pair(b1, X)}")
    ep = R("results/student_path_Baf_seed1_ep1_nothink.jsonl")
    if ep:
        L.append(f"- after pass 1 (development): replicate {acc(ep)} (original 53.0%)")
    L.append("")

    # 3. RL v4 replicate
    L += ["## 3. RL reward v4, replicated (seed 1)", "",
          "| Run | Dev, step 130 | Dev, step 260 | Dev, final | Fresh 1, final |", "|---|---|---|---|---|"]
    for lab, tag in (("v4, seed 0", "rlT4_B"), ("v4, seed 1", "rlT4_B_seed1")):
        cells = [R(f"results/student_{tag}_{ck}_nothink.jsonl") for ck in ("step130", "step260", "final")]
        cells.append(R(f"results/fresh_student_{tag}_final.jsonl"))
        L.append(f"| {lab} | " + " | ".join(acc(c) if c else "–" for c in cells) + " |")
    L += ["", "Template measurements (multi-move puzzles, first move right; development set, final checkpoint):", "",
          "| Run | Lines exactly 3 moves | Quiet 2nd own move | Reply = Lichess | Template wording | False \"only legal\" claims |",
          "|---|---|---|---|---|---|"]
    for lab, f in (("B before RL", "results/student_path_B_nothink.jsonl"),
                   ("v4, seed 0", "results/student_rlT4_B_final_nothink.jsonl"),
                   ("v4, seed 1", "results/student_rlT4_B_seed1_final_nothink.jsonl")):
        D = R(f)
        if D:
            L.append(f"| {lab} | " + " | ".join(template_stats(D, PD)) + " |")
    rq = Path("results/replication_reply_quality.md")
    if rq.exists():
        L += ["", "Stockfish reply quality (results/replication_reply_quality.md):", ""] + \
             [l for l in rq.read_text().splitlines() if l.startswith("|")]
    L.append("")

    # 4. second fresh test
    L += ["## 4. The second fresh test (500 more untouched puzzles)", "",
          "Protocol fixed before the run: `reports/fresh_test_2_protocol.md`.", "",
          "| Model | Fresh 2: first move right | Puzzle rating (95% CI) | Full line right | Fresh 1 | Pooled (1,000) |",
          "|---|---|---|---|---|---|"]
    models = [("student A", "path_A"), ("student A, seed 1", "path_A_seed1"), ("student A + 200k", "path_A200k"),
              ("student A + 200k, replicate", "path_A200k_seed1"), ("student B", "path_B"), ("student B, seed 1", "path_B_seed1"),
              ("B answer-first", "path_Baf"), ("B answer-first, replicate", "path_Baf_seed1"),
              ("B + RL v4", "rlT4_B_final"), ("B + RL v4, replicate", "rlT4_B_seed1_final"), ("untrained Qwen3-1.7B", "base")]
    M2, M1 = {}, {}
    for lab, tag in models:
        M2[lab], M1[lab] = R(f"results/fresh2_student_{tag}.jsonl"), R(f"results/fresh_student_{tag}.jsonl")
    for lab, f2, f1 in (("teacher, low effort", "results/fresh2_low_formatP1L.jsonl", "results/fresh_low_formatP1L.jsonl"),
                        ("teacher, medium effort", "results/fresh2_med_formatP1L.jsonl", "results/fresh_med_formatP1L.jsonl"),
                        ("Stockfish 16, 0.1 s", "results/fresh2_stockfish.jsonl", "results/fresh_stockfish.jsonl")):
        M2[lab], M1[lab] = R(f2), R(f1)
    for lab in M2:
        D2, D1 = M2[lab], M1[lab]
        if not D2:
            continue
        rt = rate(list(D2.values()))
        full = sum(bool(r.get("full_line_correct")) for r in D2.values()) if "Stockfish" not in lab else "–"
        pooled = {**D1, **D2} if D1 else None
        L.append(f"| {lab} | **{acc(D2)}** | {rt['rating']} ({rt['ci_low']}–{rt['ci_high']}) | {full} | "
                 f"{acc(D1) if D1 else '–'} | {acc(pooled) if pooled else '–'} |")
    L.append(f"| random legal move (expected) | {random_expected(list(P2.values())):.1f}% | | | | |")
    prim = [("student A", "student B"), ("student A", "teacher, low effort"), ("student A", "teacher, medium effort"),
            ("student A + 200k", "teacher, medium effort")]
    for scope, get in (("fresh 2", lambda l: M2[l]), ("pooled over both fresh sets (1,000)",
                                                       lambda l: {**M1[l], **M2[l]} if M1.get(l) and M2.get(l) else None)):
        have = [(x, y) for x, y in prim if get(x) and get(y)]
        if not have:
            continue
        ps = []
        for x, y in have:
            X, Y = get(x), get(y)
            b = sum(X[i]["status"] == "correct" and Y[i]["status"] != "correct" for i in X)
            c = sum(Y[i]["status"] == "correct" and X[i]["status"] != "correct" for i in X)
            ps.append(mcnemar(b, c))
        L += ["", f"Primary comparisons, {scope} (exact McNemar, Holm over the four; gap with a 95% paired bootstrap interval):", ""]
        for (x, y), adj in zip(have, holm(ps)):
            L.append(f"- {x} vs {y}: {pair(get(x), get(y))}; Holm-adjusted p = {adj:.2g}")
    Path("results/replication_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
