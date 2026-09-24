"""Morning report for night1 -> results/night1_report.md and results/night1_examples.md.

Reads (whatever exists; stages may have failed):
  results/night1_A_formatE.jsonl   stage A: engine-grounded traces (low), 98 puzzles
  results/night1_B_format*.jsonl   stage B: solver traces (medium, S or fallback P1L), fresh pool
  results/night1_C_formatP1L.jsonl stage C: high on B's misses
  results/night1_verified.jsonl    verify_trace.py output for A and B
  results/*_meta.jsonl             wall times
"""
import glob
import json
import statistics
from collections import Counter
from pathlib import Path

import chess

from puzzle_rating import rate

R = Path("results")
BANDS = [f"{lo}-{lo + 200}" for lo in range(800, 2200, 200)]


def load(path):
    p = Path(path)
    return [json.loads(l) for l in open(p)] if p.exists() else []


def wall(run):
    p = R / f"{run}_meta.jsonl"
    return sum(json.loads(l)["wall_time_s"] for l in open(p)) if p.exists() else 0.0


def out_tok(r):
    return (r.get("usage") or {}).get("completion_tokens") or 0


def pct(k, n):
    return f"{100 * k / n:.0f}%" if n else "–"


def verification_block(label, ver):
    n = len(ver)
    if not n:
        return [f"*No verification results for {label}.*", ""]
    ac = [v for v in ver if v["answer_correct"]]
    sev = Counter(v["severity"] for v in ac)
    return [
        f"| {label} | {n} | {len(ac)} ({pct(len(ac), n)}) | {sum(v['structure_ok'] for v in ver)} | "
        f"{sum(v['process_verified'] for v in ac)} ({pct(sum(v['process_verified'] for v in ac), len(ac))}) | "
        f"{sev['none']} / {sev['minor']} / {sev['serious']} | "
        f"{sum(v['lines_legal'] for v in ver)} | {sum(v['verdicts_ok'] for v in ver)} |"]


def main():
    A = load(R / "night1_A_formatE.jsonl")
    b_files = sorted(glob.glob(str(R / "night1_B_format*.jsonl")))
    B = load(b_files[0]) if b_files else []
    b_fmt = B[0]["format"] if B else "?"
    C = load(R / "night1_C_formatP1L.jsonl")
    ver = load(R / "night1_verified.jsonl")
    verA = [v for v in ver if v["format"] == "E"]
    verB = [v for v in ver if v["format"] == b_fmt]
    L = ["# Night1 report", ""]

    # ---- stage B
    if B:
        w = wall("night1_B")
        st = Counter(r["status"] for r in B)
        ok = st["correct"]
        L += [f"## Stage B: solver traces (medium effort, format {b_fmt}), fresh pool", "",
              f"- **{ok}/{len(B)} correct ({pct(ok, len(B))})**; status: {dict(st)}.",
              f"- Output tokens: mean {statistics.mean(out_tok(r) for r in B):,.0f}, median "
              f"{statistics.median(out_tok(r) for r in B):,.0f}, max {max(out_tok(r) for r in B):,}.",
              f"- Wall time {w / 3600:.1f} h; throughput {sum(out_tok(r) for r in B) / w:.0f} output tok/s; "
              f"**{ok / w * 3600:.0f} correct traces/hour**." if w else "- (no wall time recorded)",
              "", "| Band | Correct / puzzles |", "|---|---|"]
        for band in BANDS:
            rs = [r for r in B if r["rating_band"] == band]
            L.append(f"| {band} | {sum(r['status'] == 'correct' for r in rs)}/{len(rs)} |")
        okr = [r for r in B if r["status"] == "correct"]
        multi = [r for r in okr if r.get("solution_len", 1) > 1]
        L += ["", f"- Full line (vs Lichess) among correct answers: {sum(r.get('full_line_correct', False) for r in okr)}"
              f"/{len(okr)}; multi-move puzzles {sum(r.get('full_line_correct', False) for r in multi)}/{len(multi)}; "
              f"legal lines {sum(bool(r.get('line_legal')) for r in okr)}/{len(okr)}.", ""]

    # ---- verification table
    L += ["## Step verification (answer correct vs process verified)", "",
          "Process verified = every candidate and line legal, consistent with the final answer, and every "
          "verdict agrees with Stockfish. Severity among answer-correct traces: none / minor (side line) / "
          "serious (chosen line or structure).", "",
          "| Stage | Answers | Answer correct | Structure OK | Correct AND verified | Severity none/minor/serious | All lines legal | All verdicts OK |",
          "|---|---|---|---|---|---|---|---|"]
    L += verification_block(f"A (engine writer, E)", verA)
    L += verification_block(f"B (solver, {b_fmt})", verB)
    L.append("")

    # ---- stage A
    if A:
        w = wall("night1_A")
        leaks = [r for r in A if r.get("mentions_hint")]
        L += ["## Stage A: engine-grounded traces (low effort)", "",
              f"- {len(A)} puzzles; final move correct {sum(r['status'] == 'correct' for r in A)}/{len(A)} "
              f"(expected ~all: the analysis is given).",
              f"- Output tokens mean {statistics.mean(out_tok(r) for r in A):,.0f}; wall {w / 60:.0f} min.",
              f"- Written solution mentions an engine/given analysis: {len(leaks)}/{len(A)}"
              + (f" (e.g. \"{leaks[0]['mentions_hint']}\")" if leaks else "") + "; hidden reasoning mentions it: "
              f"{sum(bool(r.get('reasoning_mentions_hint')) for r in A)}/{len(A)} (fine, the student trains on the "
              f"written solution or it gets filtered).", ""]
        if B:
            bmap = {r["puzzle_id"]: r for r in B}
            shared = [r["puzzle_id"] for r in A if r["puzzle_id"] in bmap]
            vA = {v["puzzle_id"]: v for v in verA}
            vB = {v["puzzle_id"]: v for v in verB}
            b_usable = (f"from B: {sum(vB.get(i, {}).get('process_verified', False) and vB[i]['answer_correct'] for i in shared)}"
                        if vB else f"from B ({b_fmt}, not step-verifiable; answer correct AND full line right): "
                        f"{sum(bmap[i]['status'] == 'correct' and bmap[i].get('full_line_correct', False) for i in shared)}")
            L += [f"**Head-to-head on the {len(shared)} shared puzzles:** usable trace (answer correct AND process "
                  f"verified) from A: {sum(vA.get(i, {}).get('process_verified', False) and vA[i]['answer_correct'] for i in shared)}; "
                  f"{b_usable}; "
                  f"A full line right: {sum(r.get('full_line_correct', False) for r in A if r['puzzle_id'] in bmap)}; "
                  f"B answer correct: {sum(bmap[i]['status'] == 'correct' for i in shared)}. "
                  f"Tokens per trace: A {statistics.mean(out_tok(r) for r in A):,.0f} vs B "
                  f"{statistics.mean(out_tok(bmap[i]) for i in shared):,.0f}.", ""]

    # ---- stage C
    if C:
        L += ["## Stage C: can high effort solve what medium missed? (cap 60k tokens)", "",
              f"- {sum(r['status'] == 'correct' for r in C)}/{len(C)} solved; "
              f"{sum(r['status'] == 'truncated' for r in C)} ran out of tokens; status {dict(Counter(r['status'] for r in C))}.",
              "", "| Puzzle | Rating | Result | Output tokens | Minutes |", "|---|---|---|---|---|"]
        for r in sorted(C, key=lambda r: r["rating"]):
            L.append(f"| {r['puzzle_id']} | {r['rating']} | {r['status']} | {out_tok(r):,} | {r['latency_s'] / 60:.0f} |")
        L.append("")

    # ---- puzzle ratings
    L += ["## Puzzle ratings (Elo fit, 95% bootstrap CI)", "",
          "| Solver | Puzzle set | Puzzles | Solved | Rating | 95% CI |", "|---|---|---|---|---|---|"]
    specs = [("low, A (FEN only)", "results/pilot_formatA.jsonl", "test 500"),
             ("low, B (+legal moves)", "results/pilot_formatB.jsonl", "test 500"),
             ("low, P1 (perception)", "results/perc_low_formatP1.jsonl", "test 140"),
             ("low, P2", "results/perc_low_formatP2.jsonl", "test 140"),
             ("low, P1L (full line)", "results/cascade_t1a_formatP1L.jsonl", "test 140")]
    if b_files:
        specs.append((f"**medium, {b_fmt}**", b_files[0], "fresh pool 504"))
    for label, path, pset in specs:
        recs = load(path)
        if recs:
            r = rate(recs)
            L.append(f"| {label} | {pset} | {r['n']} | {r['solved']} | **{r['rating']}** | {r['ci_low']}–{r['ci_high']} |")
    L += ["", "All sets are rating-balanced 800–2200, so ratings are comparable; the fresh pool is not the "
          "test set, so the medium rating is indicative (test-set rating comes with the student evaluation).", ""]

    (R / "night1_report.md").write_text("\n".join(L))
    print("\n".join(L))

    # ---- examples: engine-written vs solver-written on the same puzzle
    if A and B:
        bmap = {r["puzzle_id"]: r for r in B}
        vA = {v["puzzle_id"]: v for v in verA}
        vB = {v["puzzle_id"]: v for v in verB}
        both = [r for r in A if r["puzzle_id"] in bmap and bmap[r["puzzle_id"]]["status"] == "correct"
                and r["status"] == "correct"]
        both.sort(key=lambda r: (not vB.get(r["puzzle_id"], {}).get("process_verified", False), -r["rating"]))
        only_a = [r for r in A if r["puzzle_id"] in bmap and bmap[r["puzzle_id"]]["status"] != "correct"
                  and vA.get(r["puzzle_id"], {}).get("process_verified")]
        E = ["# Night1 examples: engine-grounded (A) vs solver (B) written solutions", ""]
        for r in both[:2] + only_a[:1]:
            b = bmap[r["puzzle_id"]]
            board = chess.Board(r["fen"])
            ans = board.san(chess.Move.from_uci(r["correct_move"]))
            E += [f"## Puzzle {r['puzzle_id']} (rating {r['rating']}), answer {ans}", "",
                  f"FEN `{r['fen']}`", "",
                  f"### A: engine-grounded (verified: {vA.get(r['puzzle_id'], {}).get('process_verified')}, "
                  f"{out_tok(r):,} tokens)", "", "```", (r["content"] or "").strip(), "```", "",
                  f"### B: solver, medium (status {b['status']}, verified: "
                  f"{vB.get(b['puzzle_id'], {}).get('process_verified')}, {out_tok(b):,} tokens)", "",
                  "```", (b["content"] or "").strip(), "```", ""]
        (R / "night1_examples.md").write_text("\n".join(E))


if __name__ == "__main__":
    main()
