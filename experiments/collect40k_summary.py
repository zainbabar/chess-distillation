"""Summary of THE PATH step 1 (results/run_collect40k.sh) -> results/collect40k/summary.md.

Per chunk and overall: accepted explanations, rejection reasons, acceptance by rating band, writer tokens, wall time
(from the log's timestamps) and whether each chunk's Stockfish analysis is complete. Adds night 2's accepted set.
"""
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

D = Path("results/collect40k")
LOG = Path("results/collect40k.log")
NIGHT2 = Path("results/night2/sft_fdf.jsonl")


def jl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def main():
    parts = sorted(p.name[5:10] for p in D.glob("pool_?????.jsonl"))
    rows, reasons, band_tot, band_ok, tok = [], Counter(), Counter(), Counter(), []
    for part in parts:
        pool = {r["puzzle_id"]: r for r in jl(D / f"pool_{part}.jsonl")}
        acc = jl(D / f"sft_fdf_{part}.jsonl")
        rej = jl(D / f"sft_fdf_{part}.rejected.jsonl")
        tr = {}
        for r in jl(f"results/traces_c40k_{part}_FDF.jsonl"):
            tr[r["puzzle_id"]] = r  # last record wins (retries)
        errors = sum(r.get("status") == "error" for r in tr.values())
        tok += [r["usage"]["completion_tokens"] for r in tr.values() if r.get("usage")]
        rc = Counter()
        for r in rej:
            for x in r["reason"].split(","):
                rc[x.replace("claims:", "")] += 1
        reasons += rc
        for pid, p in pool.items():
            band_tot[p["rating_band"]] += 1
        for r in acc:
            band_ok[r["rating_band"]] += 1
        ea = D / f"engine_analysis_{part}.jsonl"
        rows.append((part, len(pool), len(tr), errors, len(acc), len(rej), rc,
                     sum(1 for _ in open(ea)) if ea.exists() else 0))

    # wall time per chunk from the log
    ts = {}
    for l in open(LOG):
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (chunk (\d{5})|COLLECT40K (START|DONE))", l)
        if m:
            ts[m.group(3) or m.group(4)] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    order = parts + ["DONE"]
    dur = {p: (ts[order[i + 1]] - ts[p]).total_seconds() / 3600 for i, p in enumerate(parts)
           if p in ts and order[i + 1] in ts}

    n_pool = sum(r[1] for r in rows)
    n_acc = sum(r[4] for r in rows)
    n2 = sum(1 for _ in open(NIGHT2)) if NIGHT2.exists() else 0
    hours = (ts["DONE"] - ts["START"]).total_seconds() / 3600 if "DONE" in ts and "START" in ts else None
    L = ["# Explanation collection (THE PATH step 1) — summary", "",
         f"FDF-low (\"as if discovering\" + python-chess FACTS, v1 prompt) for pool_collect1 puzzles 6,001–40,000; "
         f"claim-checked by `package_sft.py`.", "",
         f"**Accepted: {n_acc:,} of {n_pool:,} ({100 * n_acc / max(n_pool, 1):.1f}%)**; with night 2's {n2:,} "
         f"(puzzles 1–6,000, same prompt): **{n_acc + n2:,}** explanation examples.", ""]
    if hours:
        L.append(f"Wall time {hours:.1f} h ({ts['START']:%m-%d %H:%M} → {ts['DONE']:%m-%d %H:%M}); "
                 f"{n_pool / hours:,.0f} puzzles/h written, {n_acc / hours:,.0f} accepted/h.")
    if tok:
        L.append(f"Writer output: mean {sum(tok) / len(tok):.0f} tokens per explanation "
                 f"(incl. hidden reasoning), {sum(tok) / 1e6:.1f}M tokens total.")
    L += ["", "| Chunk (puzzles) | Written | Errors left | Accepted | Rejected | Main reasons | Hours | Stockfish analysis |",
          "|---|---|---|---|---|---|---|---|"]
    for part, n, nt, err, a, rj, rc, ea in rows:
        s = int(part)
        L.append(f"| {s + 1:,}–{s + n:,} | {nt:,} | {err} | {a:,} ({100 * a / max(n, 1):.1f}%) | {rj} | "
                 + ", ".join(f"{k} {v}" for k, v in rc.most_common(4))
                 + f" | {dur.get(part, float('nan')):.2f} | {ea:,}/{n:,}{' ✓' if ea == n else ' INCOMPLETE'} |")
    L += ["", "Rejection reasons overall (a text can have several): "
          + ", ".join(f"{k} {v}" for k, v in reasons.most_common()) + ".",
          "(gain = false \"wins the X\" claim, piece = names a piece that isn't on that square, mate = false mate claim, "
          "line = written line differs from the solution, wrong_move = wrong first move.)", "",
          "| Rating band | Puzzles | Accepted | % |", "|---|---|---|---|"]
    for b in sorted(band_tot, key=lambda b: int(b.split("-")[0])):
        L.append(f"| {b} | {band_tot[b]:,} | {band_ok[b]:,} | {100 * band_ok[b] / band_tot[b]:.1f}% |")
    out = D / "summary.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
