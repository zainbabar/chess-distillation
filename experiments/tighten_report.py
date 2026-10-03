"""Tightening checks, 09-30 night (results/run_tighten.sh) -> results/tighten_report.md. All numbers are computed here
from saved outputs; missing files are skipped.

(1) Five sampled runs per student (temperature 0.7, top-p 0.8, top-k 20) instead of one: mean, spread, and the gap to the
    medium-effort teacher in each sample.
(2) Student B with a second seed, and the A vs B comparison across both seeds of each (2 x 2).
(3) The Stockfish re-check of wrong answers and the reply-quality check (results/stockfish_audit.md,
    results/reply_quality.md), appended as written.
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_pilot import mcnemar  # noqa: E402
from path_A_report import load  # noqa: E402
from teacher_report import pair  # noqa: E402

R = lambda p: load(p)[0]  # noqa: E731


def pct(D):
    return 100 * sum(r["status"] == "correct" for r in D.values()) / len(D)


def acc(D):
    return f"{sum(r['status'] == 'correct' for r in D.values())}/{len(D)} ({pct(D):.1f}%)"


def main():
    L = ["# Tightening checks (09-30 night): five samples, B's second seed, Stockfish checks", ""]
    # 1. five samples
    L += ["## 1. Five sampled runs per student", "",
          "Each student sampled 5 times per puzzle set at Qwen3's recommended non-thinking settings (temperature 0.7, top-p",
          "0.8, top-k 20), the way the teacher is run (the teacher: one sample). Greedy = the main evaluation.", "",
          "| Student | Set | Greedy | Sampled: mean ± sd (min–max), n | Gap to teacher medium per sample (points; * = p < 0.05) |",
          "|---|---|---|---|---|"]
    sets = {"development": ("results/student_{t}_nothink.jsonl", "results/student_{t}_sampled{k}_nothink.jsonl",
                            "results/test500_med_formatP1L.jsonl"),
            "fresh": ("results/fresh_student_{t}.jsonl", "results/fresh_student_{t}_sampled{k}.jsonl",
                      "results/fresh_med_formatP1L.jsonl")}
    for label, tag in [("A", "path_A"), ("A + 200k", "path_A200k"), ("B", "path_B"), ("B answer-first", "path_Baf")]:
        for sname, (greedy_f, samp_f, med_f) in sets.items():
            g = R(greedy_f.format(t=tag))
            first = (f"results/student_{tag}_sampled_nothink.jsonl" if sname == "development"
                     else f"results/fresh_student_{tag}_sampled.jsonl")
            samples = [s for s in [R(first)] + [R(samp_f.format(t=tag, k=k)) for k in range(2, 6)] if s]
            med = R(med_f)
            if not samples:
                continue
            accs = [pct(s) for s in samples]
            gaps = []
            for s in samples:
                b = sum(s[i]["status"] == "correct" and med[i]["status"] != "correct" for i in s)
                c = sum(med[i]["status"] == "correct" and s[i]["status"] != "correct" for i in s)
                gaps.append(f"{pct(s) - pct(med):+.1f}{'*' if mcnemar(b, c) < 0.05 else ''}")
            sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
            L.append(f"| {label} | {sname} | {pct(g):.1f}% | {statistics.mean(accs):.1f} ± {sd:.1f} "
                     f"({min(accs):.1f}–{max(accs):.1f}), {len(accs)} | {', '.join(gaps)} |")
    L.append("")
    # 2. B second seed, 2 x 2
    L += ["## 2. Student B with a second seed", ""]
    files = {("A", 0): ("results/student_path_A_nothink.jsonl", "results/fresh_student_path_A.jsonl"),
             ("A", 1): ("results/student_path_A_seed1_nothink.jsonl", "results/fresh_student_path_A_seed1.jsonl"),
             ("B", 0): ("results/student_path_B_nothink.jsonl", "results/fresh_student_path_B.jsonl"),
             ("B", 1): ("results/student_path_B_seed1_nothink.jsonl", "results/fresh_student_path_B_seed1.jsonl")}
    D = {k: (R(v[0]), R(v[1])) for k, v in files.items()}
    if D[("B", 1)][0]:
        L.append(f"B seed 1: development {acc(D[('B', 1)][0])}, fresh {acc(D[('B', 1)][1]) if D[('B', 1)][1] else '–'} "
                 f"(B seed 0: {acc(D[('B', 0)][0])}, {acc(D[('B', 0)][1])})")
        b1ep = R("results/student_path_B_seed1_ep1_nothink.jsonl")
        if b1ep:
            L.append(f"B seed 1 after pass 1 (development): {acc(b1ep)}")
        L += ["", "| Comparison | Development | Fresh |", "|---|---|---|"]
        for (x, y) in [(("B", 1), ("B", 0)), (("A", 0), ("B", 0)), (("A", 0), ("B", 1)), (("A", 1), ("B", 0)), (("A", 1), ("B", 1))]:
            cells = [pair(D[x][j], D[y][j]) if D[x][j] and D[y][j] else "–" for j in (0, 1)]
            L.append(f"| {x[0]} seed {x[1]} vs {y[0]} seed {y[1]} | " + " | ".join(cells) + " |")
    else:
        L.append("B seed 1 not evaluated yet.")
    L.append("")
    # 3. Stockfish checks
    for f in ("results/stockfish_audit.md", "results/reply_quality.md"):
        p = Path(f)
        L += ["## " + ("3" if "audit" in f else "4") + ". " + (p.read_text().split("\n", 1)[0].lstrip("# ") if p.exists() else f),
              ""] + ((p.read_text().split("\n", 1)[1].strip().splitlines()) if p.exists() else ["not run yet"]) + [""]
    Path("results/tighten_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
