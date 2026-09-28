"""Step 5 of the cascade test: summarize tiers 1-3 on the 140 subset -> results/cascade_report.md.

Tier files (from results/run_cascade.sh):
  tier 1: cascade_t1a_formatP1L.jsonl, cascade_t1b_formatP1L.jsonl  (low + P1L, 2 attempts)
  tier 2: cascade_t2_formatP1L.jsonl  (medium + P1L on puzzles tier 1 missed)
  tier 3: cascade_t3_formatR1.jsonl   (rationalization, low, on puzzles tiers 1-2 missed)
"""
import json
import statistics
from collections import Counter
from pathlib import Path

R = Path("results")
PUZZLES = [json.loads(l) for l in open("pilot_subset140.jsonl")]
BANDS = [f"{lo}-{lo + 200}" for lo in range(800, 2200, 200)]
H200_TOK_S = (2500, 5000)  # estimate until benchmarked
SPARK_TOK_S = 250


def load(name):
    p = R / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def wall(run, fmt):
    total = 0.0
    p = R / f"{run}_meta.jsonl"
    if p.exists():
        for l in open(p):
            m = json.loads(l)
            if m["format"] == fmt:
                total += m["wall_time_s"]
    return total


def out_tok(r):
    return (r.get("usage") or {}).get("completion_tokens") or 0


def all_tok(r):
    return (r.get("usage") or {}).get("total_tokens") or 0


def pct(k, n):
    return f"{100 * k / n:.0f}%" if n else "–"


def line_stats(rs):
    """Line quality among first-move-correct answers."""
    ok = [r for r in rs if r["status"] == "correct"]
    multi = [r for r in ok if r.get("solution_len", 1) > 1]
    return {
        "n_correct": len(ok),
        "legal_line": sum(bool(r.get("line_legal")) for r in ok),
        "full_line": sum(bool(r.get("full_line_correct")) for r in ok),
        "multi": len(multi),
        "multi_full": sum(bool(r.get("full_line_correct")) for r in multi),
        "mean_match_frac": statistics.mean(r["line_match_len"] / r["solution_len"] for r in multi) if multi else 0,
    }


def main():
    t1a, t1b = load("cascade_t1a_formatP1L.jsonl"), load("cascade_t1b_formatP1L.jsonl")
    t2, t3 = load("cascade_t2_formatP1L.jsonl"), load("cascade_t3_formatR1.jsonl")
    tiers = [
        ("Tier 1a: low + P1L, attempt 1", t1a, wall("cascade_t1a", "P1L")),
        ("Tier 1b: low + P1L, attempt 2", t1b, wall("cascade_t1b", "P1L")),
        ("Tier 2: medium + P1L on tier-1 failures", t2, wall("cascade_t2", "P1L")),
    ]
    band_of = {p["puzzle_id"]: p["rating_band"] for p in PUZZLES}
    solved_by = {}  # puzzle_id -> first tier label that solved it (tiers 1-2 only; tier 3 judged separately)
    L = ["# Cascade test report (140-puzzle subset)", "",
         "Tiers run in order; each later tier only gets the puzzles every earlier tier missed. "
         "\"Solved\" = first move matches Lichess (any mate for mate-in-1). "
         "Tier 3 (rationalization) is reported separately because it is given the answer.", ""]

    # ---- per-tier table
    L += ["## Yield per tier", "",
          "| Tier | Puzzles in | Newly solved | Cumulative | Output tokens (total) | Mean output tokens | Wall time | New correct / hour |",
          "|---|---|---|---|---|---|---|---|"]
    tot_tokens, tot_wall = 0, 0.0
    for label, rs, w in tiers:
        new = [r for r in rs if r["status"] == "correct" and r["puzzle_id"] not in solved_by]
        for r in new:
            solved_by[r["puzzle_id"]] = label
        toks = sum(out_tok(r) for r in rs)
        tot_tokens += toks
        tot_wall += w
        L.append(f"| {label} | {len(rs)} | {len(new)} | {len(solved_by)} ({pct(len(solved_by), 140)}) | "
                 f"{toks:,} | {statistics.mean(out_tok(r) for r in rs):,.0f} | {w / 60:.1f} min | "
                 f"{len(new) / w * 3600:.0f} |" if rs and w else f"| {label} | {len(rs)} | – | – | – | – | – | – |")
    L.append("")

    # ---- best-of-2 detail
    if t1a and t1b:
        a = {r["puzzle_id"]: r["status"] == "correct" for r in t1a}
        b = {r["puzzle_id"]: r["status"] == "correct" for r in t1b}
        b = {i: b.get(i, False) for i in a}  # tolerate a partial attempt-2 file
        both = sum(a[i] and b[i] for i in a)
        one = sum(a[i] != b[i] for i in a)
        L += ["## Tier 1: best-of-2 at low", "",
              f"- Attempt 1: {sum(a.values())}/140, attempt 2: {sum(b.values())}/140.",
              f"- Solved in **both** attempts: {both} (more trustworthy); in **only one**: {one} "
              f"(could include lucky guesses); either: {both + one}/140.",
              f"- Attempt 2 added {sum(b[i] and not a[i] for i in a)} new puzzles for "
              f"{sum(out_tok(r) for r in t1b):,} output tokens.", ""]

    # ---- coverage by band
    L += ["## Coverage by rating band (cumulative after each tier, out of 20)", "",
          "| Band | after tier 1a | after 1b | after tier 2 | still unsolved → tier 3 |", "|---|---|---|---|---|"]
    for band in BANDS:
        cells, acc = [], set()
        for label, rs, _ in tiers:
            acc |= {pid for pid, lab in solved_by.items() if lab == label and band_of[pid] == band}
            cells.append(str(len(acc)))
        L.append(f"| {band} | " + " | ".join(cells) + f" | {20 - len(acc)} |")
    L.append("")

    # ---- line quality
    L += ["## Full-line quality (among first-move-correct answers)", "",
          "| Tier | Correct answers | Legal line | Full line matches Lichess | Multi-move puzzles: full line | Mean fraction of line matched (multi-move) |",
          "|---|---|---|---|---|---|"]
    for label, rs, _ in tiers:
        if not rs:
            continue
        s = line_stats(rs)
        L.append(f"| {label} | {s['n_correct']} | {s['legal_line']} ({pct(s['legal_line'], s['n_correct'])}) | "
                 f"{s['full_line']} ({pct(s['full_line'], s['n_correct'])}) | {s['multi_full']}/{s['multi']} | "
                 f"{100 * s['mean_match_frac']:.0f}% |")
    L.append("")

    # ---- tier 3
    if t3:
        n = len(t3)
        move_ok = sum(r["status"] == "correct" for r in t3)
        legal = sum(bool(r.get("line_legal")) for r in t3)
        full = sum(bool(r.get("full_line_correct")) for r in t3)
        two = sum(r.get("line_match_len", 0) >= min(3, r["solution_len"]) for r in t3)
        hint = [r for r in t3 if r.get("mentions_hint")]
        w3 = wall("cascade_t3", "R1")
        L += ["## Tier 3: rationalization (model is given the answer)", "",
              f"- Puzzles: {n}; wall time {w3 / 60:.1f} min; mean output tokens "
              f"{statistics.mean(out_tok(r) for r in t3):,.0f}.",
              f"- Final move = the given answer: {move_ok}/{n} (sanity check).",
              f"- Line legal: {legal}/{n}. **Full line matches Lichess: {full}/{n}** (strict). "
              f"Line correct through the solver's 2nd move (or complete if shorter): {two}/{n} (lenient).",
              f"- Mentions being told the answer (hint leak): {len(hint)}/{n}"
              + (f", e.g. \"{hint[0]['mentions_hint']}\"" if hint else "") + ".",
              "", "Verified rationalizations (strict) would be added as *tagged* traces; lenient ones need a "
              "decision.", ""]
        tot_tokens += sum(out_tok(r) for r in t3)
        tot_wall += w3

    # ---- totals and projection
    solved12 = len(solved_by)
    if tot_wall:
        L += ["## Totals and projection", "",
              f"- Tiers 1–2 solved **{solved12}/140 ({pct(solved12, 140)})** with real (non-hinted) reasoning.",
              f"- All tiers: {tot_tokens:,} output tokens, {tot_wall / 60:.0f} min wall time on the Spark.",
              f"- Tiers 1–2 yield: **{solved12 / (sum(w for _, _, w in tiers)) * 3600:.0f} verified traces per "
              f"hour** on the Spark (these 140 puzzles are rating-balanced 800–2200).",
              ]
        per_trace = sum(sum(out_tok(r) for r in rs) for _, rs, _ in tiers) / max(solved12, 1)
        spark_h = 10000 * per_trace / SPARK_TOK_S / 3600
        L += [f"- Output tokens per verified trace (tiers 1–2): {per_trace:,.0f}. For **10,000 traces**: "
              f"~{spark_h:.0f} h on the Spark at ~{SPARK_TOK_S} tok/s; ~{10000 * per_trace / H200_TOK_S[1] / 3600:.0f}–"
              f"{10000 * per_trace / H200_TOK_S[0] / 3600:.0f} h on an H200 (estimated {H200_TOK_S[0]:,}–{H200_TOK_S[1]:,} tok/s, "
              f"unbenchmarked). Note the real Spark rate depends on how full the batches are; this is a rough guide.", ""]

    text = "\n".join(L)
    (R / "cascade_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
