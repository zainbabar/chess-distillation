"""Step 4: summarize results/pilot_format{A,B}.jsonl into results/report.md (also printed)."""
import json
import math
import statistics
from pathlib import Path

RUN = "pilot"
FORMATS = {"A": "FEN + side to move", "B": "FEN + side to move + legal moves"}
BANDS = [f"{lo}-{lo + 200}" for lo in range(800, 2200, 200)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def pct(k, n):
    return f"{100 * k / n:.1f}%" if n else "–"


def load():
    res = {f: [json.loads(l) for l in open(f"results/{RUN}_format{f}.jsonl")] for f in FORMATS}
    meta = {}
    for l in open(f"results/{RUN}_meta.jsonl"):
        m = json.loads(l)
        meta[m["format"]] = m  # last entry per format wins
    return res, meta


def out_tokens(r):
    return (r.get("usage") or {}).get("completion_tokens") or 0


def main():
    res, meta = load()
    L = [f"# Pilot report: gpt-oss-120b on 500 Lichess puzzles", ""]
    L += ["Model `openai/gpt-oss-120b` (vLLM 0.20.1, `nvcr.io/nvidia/vllm:26.05-py3`, DGX Spark), "
          "reasoning_effort `low`, max_tokens 8192, 32 concurrent requests. "
          "Grading: exact match with the Lichess move; for `mateIn1` any checkmating move counts.", ""]

    # 1. Accuracy by band x format
    L += ["## Accuracy by rating band", "",
          "| Rating band | n | A: FEN | B: FEN + legal moves | B − A |", "|---|---|---|---|---|"]
    for band in BANDS + ["**All**"]:
        row = [band]
        accs = {}
        for f in FORMATS:
            rs = [r for r in res[f] if band == "**All**" or r["rating_band"] == band]
            k = sum(r["correct"] for r in rs)
            accs[f] = (k, len(rs))
        n = accs["A"][1]
        row.append(str(n))
        for f in FORMATS:
            k, n = accs[f]
            cell = f"{k}/{n} ({pct(k, n)})"
            row.append(f"**{cell}**" if band == "**All**" else cell)
        diff = 100 * (accs["B"][0] / accs["B"][1] - accs["A"][0] / accs["A"][1])
        row.append(f"{diff:+.1f} pp")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    for f in FORMATS:
        k, n = sum(r["correct"] for r in res[f]), len(res[f])
        lo, hi = wilson(k, n)
        L.append(f"- Overall {f}: {pct(k, n)} (95% CI {100*lo:.1f}–{100*hi:.1f}%).")
    lo, hi = wilson(35, 71)
    L.append(f"- Per band n≈71, so each band cell is noisy: e.g. 35/71 has a 95% CI of "
             f"{100*lo:.0f}–{100*hi:.0f}%. Read band trends, not single cells.")
    L.append("")

    # 2. Outcome breakdown
    L += ["## Outcomes", "", "| Format | Correct | Wrong (legal) | Illegal | Parse failure | Error | Exact final-line format |",
          "|---|---|---|---|---|---|---|"]
    for f in FORMATS:
        rs = res[f]
        n = len(rs)
        c = {s: sum(r["status"] == s for r in rs) for s in ["correct", "wrong", "illegal", "parse_fail", "error"]}
        strict = sum(bool(r.get("final_line_strict")) for r in rs)
        L.append(f"| {f} | {c['correct']} ({pct(c['correct'], n)}) | {c['wrong']} ({pct(c['wrong'], n)}) | "
                 f"{c['illegal']} ({pct(c['illegal'], n)}) | {c['parse_fail']} ({pct(c['parse_fail'], n)}) | "
                 f"{c['error']} | {pct(strict, n)} |")
    L.append("")
    L += ["Illegal-move rate by band:", "", "| Rating band | A | B |", "|---|---|---|"]
    for band in BANDS:
        cells = []
        for f in FORMATS:
            rs = [r for r in res[f] if r["rating_band"] == band]
            cells.append(pct(sum(r["status"] == "illegal" for r in rs), len(rs)))
        L.append(f"| {band} | {cells[0]} | {cells[1]} |")
    L.append("")

    # 3. Tokens, latency, throughput
    L += ["## Tokens, latency, throughput", "",
          "| Format | Mean output tokens | Median | p90 | Max | Mean prompt tokens | Mean latency | Wall time | Output tok/s (aggregate) |",
          "|---|---|---|---|---|---|---|---|---|"]
    tot_out, tot_wall = 0, 0.0
    for f in FORMATS:
        rs = res[f]
        ot = [out_tokens(r) for r in rs]
        pt = [(r.get("usage") or {}).get("prompt_tokens") or 0 for r in rs]
        lat = [r["latency_s"] for r in rs]
        wall = meta[f]["wall_time_s"]
        tot_out += sum(ot)
        tot_wall += wall
        p90 = sorted(ot)[int(0.9 * len(ot)) - 1]
        L.append(f"| {f} | {statistics.mean(ot):.0f} | {statistics.median(ot):.0f} | {p90} | {max(ot)} | "
                 f"{statistics.mean(pt):.0f} | {statistics.mean(lat):.0f} s | {wall/60:.1f} min | {sum(ot)/wall:.0f} |")
    L.append(f"| **Total** | | | | | | | **{tot_wall/60:.1f} min** | **{tot_out/tot_wall:.0f}** "
             f"({tot_out:,} output tokens) |")
    L.append("")
    L += ["Mean output tokens: correct vs incorrect answers:", "", "| Format | Correct | Wrong/illegal |", "|---|---|---|"]
    for f in FORMATS:
        ok = [out_tokens(r) for r in res[f] if r["correct"]]
        bad = [out_tokens(r) for r in res[f] if not r["correct"]]
        L.append(f"| {f} | {statistics.mean(ok):.0f} | {statistics.mean(bad):.0f} |")
    L.append("")

    # 4. Extras
    a = {r["puzzle_id"]: r["correct"] for r in res["A"]}
    b = {r["puzzle_id"]: r["correct"] for r in res["B"]}
    both = sum(a[i] and b[i] for i in a)
    only_a = sum(a[i] and not b[i] for i in a)
    only_b = sum(b[i] and not a[i] for i in a)
    neither = sum(not a[i] and not b[i] for i in a)
    L += ["## Extras", "",
          f"- **Head-to-head (same 500 puzzles):** both correct {both}, only A {only_a}, only B {only_b}, "
          f"neither {neither}. Solved by at least one format: {both + only_a + only_b}/500.",
          ]
    for f in FORMATS:
        m1 = [r for r in res[f] if "mateIn1" in r["themes"]]
        mate = [r for r in res[f] if "mate" in r["themes"] and "mateIn1" not in r["themes"]]
        other = [r for r in res[f] if "mate" not in r["themes"]]
        L.append(f"- **By puzzle type, {f}:** mate-in-1 {pct(sum(r['correct'] for r in m1), len(m1))} (n={len(m1)}), "
                 f"longer mates {pct(sum(r['correct'] for r in mate), len(mate))} (n={len(mate)}), "
                 f"non-mate tactics {pct(sum(r['correct'] for r in other), len(other))} (n={len(other)}).")
    n_ok = sum(r["correct"] for f in FORMATS for r in res[f])
    L.append(f"- **Usable training traces:** {n_ok} correct reasoning traces out of 1,000 "
             f"({n_ok/10:.1f}%), covering {both + only_a + only_b} distinct puzzles.")
    L.append("")

    # 5. Stockfish re-grade of wrong answers (from stockfish_grade.py), if present
    sf_path = Path(f"results/{RUN}_stockfish.jsonl")
    if sf_path.exists():
        sf = [json.loads(l) for l in sf_path.open()]
        depth = sf[0]["sf_depth"]
        L += ["## Stockfish re-grade of wrong answers", "",
              f"Stockfish 16, depth {depth}. A wrong answer counts as *also correct* if it forces mate when "
              "the solution does, or is within 5 win-% points of the solution (Lichess win-% formula). "
              "Otherwise graded by win-% lost vs the solution: small <10, mistake 10–30, blunder ≥30.", "",
              "| Format | Lichess-exact accuracy | Stockfish-adjusted | Also correct | Small | Mistake | Blunder |",
              "|---|---|---|---|---|---|---|"]
        for f in FORMATS:
            x = [r for r in sf if r["format"] == f]
            v = {k: sum(r["sf_verdict"] == k for r in x) for k in ("also_correct", "small", "mistake", "blunder")}
            L.append(f"| {f} | {pct(sum(r['status'] == 'correct' for r in x), len(x))} | "
                     f"{pct(sum(r['sf_correct'] for r in x), len(x))} | {v['also_correct']} | {v['small']} | "
                     f"{v['mistake']} | {v['blunder']} |")
        wrong = [r for r in sf if r["status"] == "wrong"]
        L += ["", f"- Wrong answers ({len(wrong)}): median win chance {statistics.median(r['model_eval']['win_pct'] for r in wrong):.0f}% "
              f"after the model's move vs {statistics.median(r['solution_eval']['win_pct'] for r in wrong):.0f}% after the solution; "
              f"median loss {statistics.median(r['win_pct_loss'] for r in wrong):.0f} win-% points.", ""]

    text = "\n".join(L)
    Path("results/report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
