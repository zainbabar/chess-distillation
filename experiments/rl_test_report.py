"""First RL test (THE PATH step 3) -> results/rl_test_report.md.

Before/after RL for A (answers only) and B (teacher explanations) on the 500 test puzzles, plus the training curve:
every RL puzzle is fresh and used once, so the per-step sample accuracy (temperature 1.0) is itself a held-out learning
curve that never touches the test set.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from analyze_pilot import mcnemar
from path_A_report import load
from path_AB_report import explain_stats
from puzzle_rating import rate

ARMS = {
    "A": ("A (answers only), SFT", "results/student_path_A_nothink.jsonl"),
    "rlA": ("A + RL (100 steps)", "results/student_rl_A_nothink.jsonl"),
    "B": ("B (teacher explanations), SFT", "results/student_path_B_nothink.jsonl"),
    "rlB": ("B + RL (100 steps)", "results/student_rl_B_nothink.jsonl"),
    "T": ("teacher gpt-oss-120b (low, 1 attempt)", "results/test500_formatP1L.jsonl"),
}
PAIRS = [("rlA", "A"), ("rlB", "B"), ("rlB", "rlA"), ("B", "A"), ("rlA", "T"), ("rlB", "T")]


def curve(log):
    s = Path(log).read_text() if Path(log).exists() else ""
    rew = [float(x) for x in re.findall(r"'reward': '([-0-9.e]+)'", s)]
    ln = [float(x) for x in re.findall(r"'completions/mean_length': '([0-9.e]+)'", s)]
    zero = [float(x) for x in re.findall(r"'frac_reward_zero_std': '([0-9.e]+)'", s)]
    return rew, ln, zero


def main():
    P = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("test_set.jsonl")}
    D, notes = {}, []
    L = ["# First RL test on the Spark — A vs B, before and after RL", "",
         "GRPO, binary reward (right move 1, else 0; no FINAL_MOVE −0.1), 100 steps × 8 fresh puzzles × 8 samples",
         "(the same 800 puzzles for both, never used in SFT), lr 2e-6, temperature 1.0, no KL, DAPO loss.",
         "Test: 500 held-out puzzles, greedy, max 1,024 new tokens.", "",
         "| Model | Correct | Rating (95% CI) | Illegal | Legal line | Full line right | Mean output tokens |",
         "|---|---|---|---|---|---|---|"]
    for k, (label, path) in ARMS.items():
        R, n_lines = load(path)
        if not R:
            notes.append(f"- missing: {label} ({path})")
            continue
        D[k] = R
        rs = list(R.values())
        st = Counter(r["status"] for r in rs)
        rt = rate(rs)
        toks = [r["usage"]["completion_tokens"] for r in rs if r.get("usage")]
        L.append(f"| {label} | **{st['correct']}/{len(rs)} ({100 * st['correct'] / len(rs):.1f}%)** | "
                 f"{rt['rating']} ({rt['ci_low']}–{rt['ci_high']}) | {st['illegal']} | "
                 f"{sum(bool(r.get('line_legal')) for r in rs)} | {sum(bool(r.get('full_line_correct')) for r in rs)} | "
                 f"{mean(toks):.0f} |")
        if len(R) != 500:
            notes.append(f"- {label}: {len(R)} puzzles graded")
    L += ["", "Paired comparisons (exact McNemar; only-first-right vs only-second-right):", ""]
    for a, b in PAIRS:
        if a in D and b in D:
            ids = set(D[a]) & set(D[b])
            x = sum(D[a][i]["status"] == "correct" and D[b][i]["status"] != "correct" for i in ids)
            y = sum(D[b][i]["status"] == "correct" and D[a][i]["status"] != "correct" for i in ids)
            L.append(f"- {ARMS[a][0]} vs {ARMS[b][0]}: {x} vs {y}, p = {mcnemar(x, y):.2g}")
    L += ["", "Training curve (sample accuracy on fresh puzzles at temperature 1.0 = mean reward, in blocks of 20 steps):", ""]
    for arm in ("A", "B"):
        rew, ln, zero = curve(f"results/rl_{arm}.log")
        if rew:
            blocks = [rew[i:i + 20] for i in range(0, len(rew), 20)]
            L.append(f"- {arm}: " + " → ".join(f"{mean(b):.3f}" for b in blocks) + f" ({len(rew)} steps); "
                     f"mean length {mean(ln[:20]):.0f} → {mean(ln[-20:]):.0f} tokens; groups with no signal "
                     f"(all 8 samples equal) {mean(zero):.0%}")
        m = Path(f"ckpt/rl_{arm}/rl_meta.json")
        if m.exists():
            d = json.load(open(m))
            L.append(f"  - {d['minutes']} min for {d['max_steps']} steps ({60 * d['minutes'] / d['max_steps']:.0f} s/step), "
                     f"{d['samples']:,} samples, {d['parse_fails']} parse fails")
    if "rlB" in D:
        L += ["", "Is B's reasoning truer after RL? (claim checker on the text before FINAL_LINE)", ""]
        for k in ("B", "rlB"):
            s = explain_stats(D[k], P)
            for key in ("right", "wrong"):
                nb = s[f"{key}_body"]
                if nb:
                    L.append(f"- {ARMS[k][0]}, {key} answers ({nb}): claim-clean {100 * s[f'{key}_clean'] / nb:.0f}%, "
                             f"move flag {100 * s[f'{key}_move_flag'] / nb:.0f}%")
    if notes:
        L += ["", "Notes:", ""] + notes
    Path("results/rl_test_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
