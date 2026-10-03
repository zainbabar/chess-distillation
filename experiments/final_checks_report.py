"""Final machine checks, 10-01 (results/run_final_checks.sh) -> results/final_checks_report.md. Computed from saved
outputs; missing files are skipped.

(1) The untrained Qwen3-1.7B on the development set at the exact student settings.
(2) The teacher sampled again on the fresh set (medium x 2, low x 3 in total): spread, and the key comparisons against
    each teacher sample.
(3) Whole-line soundness by Stockfish (results/line_soundness.md), appended as written.
"""
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from path_A_report import load  # noqa: E402
from teacher_report import pair  # noqa: E402

R = lambda p: load(p)[0]  # noqa: E731


def pct(D):
    return 100 * sum(r["status"] == "correct" for r in D.values()) / len(D)


def main():
    L = ["# Final machine checks (10-01): untrained baseline, teacher resampled, whole-line soundness", ""]
    base = R("results/student_base_nothink_dev.jsonl")
    L += ["## 1. Untrained Qwen3-1.7B, development set (greedy, thinking off, 1,024 tokens: the students' settings)", ""]
    if base:
        st = Counter(r["status"] for r in base.values())
        L.append(f"{st['correct']}/500 ({pct(base):.1f}%); unreadable {st['parse_fail']}, cut off {st['truncated']}, "
                 f"illegal {st['illegal']}, wrong {st['wrong']}. (Fresh set: 6/500, 1.2%.)")
    else:
        L.append("not run")
    L += ["", "## 2. The teacher sampled again (fresh set)", ""]
    med = [d for d in (R("results/fresh_med_formatP1L.jsonl"), R("results/fresh_med_s2_formatP1L.jsonl")) if d]
    low = [d for d in (R("results/fresh_low_formatP1L.jsonl"), R("results/fresh_low_s2_formatP1L.jsonl"),
                       R("results/fresh_low_s3_formatP1L.jsonl")) if d]
    for name, runs in (("medium effort", med), ("low effort", low)):
        accs = [pct(d) for d in runs]
        sd = f" ± {statistics.stdev(accs):.1f}" if len(accs) > 1 else ""
        L.append(f"- Teacher, {name}: samples {', '.join(f'{a:.1f}%' for a in accs)}; mean {statistics.mean(accs):.1f}{sd}")
    L.append("")
    students = {"student A + 200k": R("results/fresh_student_path_A200k.jsonl"), "student A": R("results/fresh_student_path_A.jsonl"),
                "student A, seed 1": R("results/fresh_student_path_A_seed1.jsonl"), "student B": R("results/fresh_student_path_B.jsonl")}
    L += ["Each student (greedy) against each teacher sample (exact McNemar; gap with a 95% paired bootstrap interval):", ""]
    for sname, S in students.items():
        if not S:
            continue
        for k, M in enumerate(med, 1):
            L.append(f"- {sname} vs teacher medium, sample {k}: {pair(S, M)}")
    L.append("")
    p = Path("results/line_soundness.md")
    L += ["## 3. " + (p.read_text().split("\n", 1)[0].lstrip("# ") if p.exists() else "Line soundness"), ""]
    L += (p.read_text().split("\n", 1)[1].strip().splitlines() if p.exists() else ["not run yet"]) + [""]
    Path("results/final_checks_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
