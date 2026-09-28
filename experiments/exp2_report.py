"""Experiment 2 report -> results/exp2_report.md and results/exp2_examples.md.

1. Engine-grounded traces by writer effort (low = night1 stage A, medium = X1, high = X2), same 98
   puzzles: correctness, step verification, full lines, leaks, tokens, and the explanation judge.
2. Medium x2 coverage: night1 stage B (one medium attempt) + X3 (a second attempt on B's misses).
3. Solver traces' hidden reasoning, judged, vs engine-grounded write-ups.
4. Side-by-side examples, this time with the solver's hidden reasoning (not its near-empty answer).
Works on partial data (missing files are skipped).
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from puzzle_rating import rate

R = Path("results")
BANDS = [f"{lo}-{lo + 200}" for lo in range(800, 2200, 200)]


def load(name):
    p = R / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def by_id(rows, fmt=None):
    return {r["puzzle_id"]: r for r in rows if fmt is None or r.get("format") == fmt}


def wall(run):
    p = R / f"{run}_meta.jsonl"
    return sum(json.loads(l)["wall_time_s"] for l in open(p)) if p.exists() else 0.0


def tok(r):
    return (r.get("usage") or {}).get("completion_tokens") or 0


def pct(k, n):
    return f"{100 * k / n:.0f}%" if n else "–"


def judge_summary(J):
    ok = [j for j in J if j.get("parse_ok")]
    v = Counter(j["verdict"] for j in ok)
    mean = lambda k: statistics.mean(j[k] for j in ok) if ok else float("nan")
    return ok, v, mean


WRITERS = [  # label, results file, verification file (+ format filter), judge file, run name
    ("low (night1 A)", "night1_A_formatE.jsonl", "night1_verified.jsonl", "judge_E_low.jsonl", "night1_A"),
    ("medium (X1)", "exp2_E_med_formatE.jsonl", "exp2_verified_E_med.jsonl", "judge_E_med.jsonl", "exp2_E_med"),
    ("high (X2)", "exp2_E_high_formatE.jsonl", "exp2_verified_E_high.jsonl", "judge_E_high.jsonl", "exp2_E_high"),
]


def main():
    L = ["# Experiment 2 report", "",
         "Engine-grounded traces: Stockfish supplies the verified lines, gpt-oss writes the explanation. "
         "Judge = gpt-oss checking each text against Stockfish's analysis (1–5: accuracy, coherence, "
         "explains why; verdict good/ok/bad).", ""]

    # ---- 1. writer effort
    L += ["## 1. Engine-grounded traces: writer effort (same 98 puzzles)", "",
          "| Writer | Traces | Answer correct | Step-verified | Full line = Lichess | Mean tokens | Truncated | "
          "Judge good / ok / bad | Accuracy | Coherence | Explains why | Correct + verified + judged good | Writer tokens per such trace |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    writer_data = {}
    for label, rf, vf, jf, run in WRITERS:
        rs = load(rf)
        if not rs:
            L.append(f"| {label} | (not run yet) |" + " |" * 11)
            continue
        V = by_id(load(vf), "E")
        J = by_id([j for j in load(jf) if j.get("parse_ok")])
        _, v, mean = judge_summary(list(J.values()))
        n = len(rs)
        usable = [r for r in rs if r["status"] == "correct" and V.get(r["puzzle_id"], {}).get("process_verified")
                  and J.get(r["puzzle_id"], {}).get("verdict") == "good"]
        total_tok = sum(tok(r) for r in rs)
        writer_data[label] = (rs, V, J)
        L.append(
            f"| {label} | {n} | {sum(r['status'] == 'correct' for r in rs)} | "
            f"{sum(V.get(r['puzzle_id'], {}).get('process_verified', False) for r in rs)} | "
            f"{sum(r.get('full_line_correct', False) for r in rs)} | {statistics.mean(tok(r) for r in rs):,.0f} | "
            f"{sum(r['status'] == 'truncated' for r in rs)} | {v['good']} / {v['ok']} / {v['bad']} | "
            f"{mean('accuracy'):.2f} | {mean('coherence'):.2f} | {mean('explains_why'):.2f} | {len(usable)} | "
            f"{(total_tok / len(usable)) if usable else float('nan'):,.0f} |")
    leaks = {label: sum(bool(r.get("mentions_hint")) for r in d[0]) for label, d in writer_data.items()}
    L += ["", f"Written solutions mentioning an engine / given analysis: {leaks}.", ""]

    # ---- 2. medium x2
    B = load("night1_B_formatP1L.jsonl")
    B2 = load("exp2_B2_formatP1L.jsonl")
    if B:
        solved1 = {r["puzzle_id"] for r in B if r["status"] == "correct"}
        solved2 = {r["puzzle_id"] for r in B2 if r["status"] == "correct"}
        pool = [json.loads(l) for l in open("pool_night1.jsonl")]
        n = len(pool)
        both = solved1 | solved2
        L += ["## 2. Medium ×2: second medium attempt on stage B's misses (X3)", ""]
        if B2:
            rescue = len(solved2) / len(B2)
            est3 = len(both) + (n - len(both)) * rescue
            r1 = rate([{"rating": p["rating"], "status": "correct" if p["puzzle_id"] in solved1 else "x"} for p in pool])
            r2 = rate([{"rating": p["rating"], "status": "correct" if p["puzzle_id"] in both else "x"} for p in pool])
            okB2 = [r for r in B2 if r["status"] == "correct"]
            multi = [r for r in okB2 if r.get("solution_len", 1) > 1]
            L += [f"- Attempt 1 (night1 B): {len(solved1)}/{n} ({pct(len(solved1), n)}), rating {r1['rating']} "
                  f"({r1['ci_low']}–{r1['ci_high']}).",
                  f"- Attempt 2 on the {len(B2)} misses: **{len(solved2)} rescued ({pct(len(solved2), len(B2))})**; "
                  f"status {dict(Counter(r['status'] for r in B2))}; mean {statistics.mean(tok(r) for r in B2):,.0f} tokens; "
                  f"wall {wall('exp2_B2') / 3600:.1f} h.",
                  f"- **Medium ×2: {len(both)}/{n} ({pct(len(both), n)})**, rating {r2['rating']} ({r2['ci_low']}–{r2['ci_high']}). "
                  f"Extrapolated ×3 (same rescue rate): ~{est3:.0f}/{n} ({100 * est3 / n:.0f}%).",
                  f"- Attempt-2 line quality (correct answers): legal line {pct(sum(bool(r.get('line_legal')) for r in okB2), len(okB2))}, "
                  f"full line on multi-move puzzles {sum(r.get('full_line_correct', False) for r in multi)}/{len(multi)}.",
                  "", "| Band | Attempt 1 | + attempt 2 | of |", "|---|---|---|---|"]
            for band in BANDS:
                ids = [p["puzzle_id"] for p in pool if p["rating_band"] == band]
                L.append(f"| {band} | {sum(i in solved1 for i in ids)} | {sum(i in both for i in ids)} | {len(ids)} |")
            L.append("")
        else:
            L += ["*(X3 not run yet)*", ""]

    # ---- 3. solver hidden reasoning, judged
    JB, JB2 = load("judge_B.jsonl"), load("judge_B2.jsonl")
    if JB or JB2:
        L += ["## 3. Solver traces: the hidden reasoning, judged", "",
              "| Traces | Judged | Good / ok / bad | Accuracy | Coherence | Explains why |", "|---|---|---|---|---|---|"]
        for label, J in (("night1 B (medium, attempt 1), correct", JB), ("X3 B2 (medium, attempt 2), correct", JB2)):
            ok, v, mean = judge_summary(J)
            if ok:
                L.append(f"| {label} | {len(ok)} | {v['good']} / {v['ok']} / {v['bad']} | {mean('accuracy'):.2f} | "
                         f"{mean('coherence'):.2f} | {mean('explains_why'):.2f} |")
        L.append("")
        if writer_data:
            shared = {j["puzzle_id"] for j in JB if j.get("parse_ok")} & set(next(iter(writer_data.values()))[2])
            L.append(f"On the {len(shared)} puzzles that have both a correct solver trace and engine-grounded traces, "
                     "judge verdicts (good/ok/bad):")
            jb = by_id([j for j in JB if j.get("parse_ok")])
            L.append(f"- solver (hidden reasoning): {dict(Counter(jb[i]['verdict'] for i in shared))}")
            for label, (_, _, J) in writer_data.items():
                L.append(f"- engine-grounded, {label}: {dict(Counter(J[i]['verdict'] for i in shared if i in J))}")
            L.append("")

    (R / "exp2_report.md").write_text("\n".join(L))
    print("\n".join(L))
    write_examples(writer_data, B)


def excerpt(text, head=3000, tail=1500):
    text = (text or "").strip()
    if len(text) <= head + tail:
        return text
    return text[:head] + f"\n\n[... {len(text) - head - tail:,} characters omitted ...]\n\n" + text[-tail:]


def write_examples(writer_data, B):
    if not writer_data:
        return
    Bm = by_id(B)
    JB = by_id([j for j in load("judge_B.jsonl") if j.get("parse_ok")])
    analysis = by_id(load("night1_engine_analysis.jsonl"))
    first = next(iter(writer_data.values()))[0]
    ids = [r["puzzle_id"] for r in first]
    with_solver = [i for i in ids if Bm.get(i, {}).get("status") == "correct"]
    without = [i for i in ids if Bm.get(i, {}).get("status") != "correct"]
    pick = sorted(with_solver, key=lambda i: -first[ids.index(i)]["rating"])[:3] + without[:1]
    E = ["# Experiment 2: side-by-side examples", "",
         "For each puzzle: Stockfish's best line, the engine-grounded write-ups at each writer effort (their "
         "written explanation, which is what a student would train on), and the medium solver's **hidden "
         "reasoning** (what a student would train on for solver traces). Judge verdicts in brackets.", ""]
    for pid in pick:
        r0 = first[ids.index(pid)]
        board = chess.Board(r0["fen"])
        a = analysis.get(pid)
        best = next((c for c in a["candidates"] if c["move"] == r0["correct_move"]), None) if a else None
        E += [f"## Puzzle {pid} (rating {r0['rating']}): answer {board.san(chess.Move.from_uci(r0['correct_move']))}", "",
              f"FEN `{r0['fen']}`" + (f"; Stockfish line: {' '.join(best['line_san'])}" if best else ""), ""]
        for label, (rs, V, J) in writer_data.items():
            r = next((x for x in rs if x["puzzle_id"] == pid), None)
            if not r:
                continue
            j = J.get(pid, {})
            E += [f"### Engine-grounded, {label} writer — {tok(r):,} tokens; step-verified: "
                  f"{V.get(pid, {}).get('process_verified')}; judge: {j.get('verdict')} "
                  f"({j.get('accuracy')}/{j.get('coherence')}/{j.get('explains_why')})", "",
                  "```", (r.get("content") or "").strip(), "```", ""]
        b = Bm.get(pid)
        if b:
            j = JB.get(pid, {})
            E += [f"### Solver, medium (status {b['status']}; {tok(b):,} tokens; full line right: "
                  f"{b.get('full_line_correct')}; judge on hidden reasoning: {j.get('verdict')} "
                  f"({j.get('accuracy')}/{j.get('coherence')}/{j.get('explains_why')}))", "",
                  "Hidden reasoning (excerpt):", "", "```", excerpt(b.get("reasoning")), "```", "",
                  "Written answer:", "", "```", (b.get("content") or "").strip(), "```", ""]
    (R / "exp2_examples.md").write_text("\n".join(E))


if __name__ == "__main__":
    main()
