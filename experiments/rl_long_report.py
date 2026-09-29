"""The ~12 h RL run (THE PATH step 3) -> results/rl_long_report.md.

A (answers only) and B (teacher explanations), each RL'd from its SFT checkpoint with identical settings on the same
3,120 fresh puzzles; checkpoints every 130 steps are evaluated on the 500 held-out test puzzles. Step 0 = the SFT model.
Training curve = sample accuracy on fresh puzzles at temperature 1.0 (each puzzle used once, so it is held-out data).
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

CKS = [("0 (SFT)", "results/student_path_{arm}_nothink.jsonl"),
       ("130", "results/student_rlL_{arm}_step130_nothink.jsonl"),
       ("260", "results/student_rlL_{arm}_step260_nothink.jsonl"),
       ("final", "results/student_rlL_{arm}_final_nothink.jsonl")]


def row(label, R):
    rs = list(R.values())
    st = Counter(r["status"] for r in rs)
    rt = rate(rs)
    toks = [r["usage"]["completion_tokens"] for r in rs if r.get("usage")]
    return (f"| {label} | **{st['correct']}/{len(rs)} ({100 * st['correct'] / len(rs):.1f}%)** | "
            f"{rt['rating']} ({rt['ci_low']}–{rt['ci_high']}) | {st['illegal']} | {sum(bool(r.get('line_legal')) for r in rs)} | "
            f"{sum(bool(r.get('full_line_correct')) for r in rs)} | {mean(toks):.0f} |")


def pair(a, b):
    ids = set(a) & set(b)
    x = sum(a[i]["status"] == "correct" and b[i]["status"] != "correct" for i in ids)
    y = sum(b[i]["status"] == "correct" and a[i]["status"] != "correct" for i in ids)
    return f"{x} vs {y}, p = {mcnemar(x, y):.2g}"


def main():
    P = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("test_set.jsonl")}
    D = {}
    for arm in ("A", "B"):
        for ck, pat in CKS:
            R, _ = load(pat.format(arm=arm))
            if R:
                D[(arm, ck)] = R
    meta = {arm: json.load(open(f"ckpt/rlL_{arm}/rl_meta.json")) for arm in ("A", "B")
            if Path(f"ckpt/rlL_{arm}/rl_meta.json").exists()}
    L = ["# RL run (~12 h on the Spark) — does imitated reasoning pay off after RL?", "",
         "GRPO from each SFT checkpoint, identical settings: binary reward (right move 1, wrong 0, no FINAL_MOVE −0.1),",
         "8 fresh puzzles × 8 samples per step (the same puzzles for A and B, never used before), lr 2e-6, temperature 1.0,",
         "no KL, DAPO loss. Test: 500 held-out puzzles, greedy, max 1,024 new tokens. Step 0 = the SFT model.", ""]
    for arm, name in (("A", "A: answers only"), ("B", "B: teacher explanations")):
        steps = meta.get(arm, {}).get("max_steps", "?")
        L += [f"## {name} (RL steps: {steps})", "",
              "| RL steps | Correct | Rating (95% CI) | Illegal | Legal line | Full line right | Mean output tokens |",
              "|---|---|---|---|---|---|---|"]
        for ck, _ in CKS:
            if (arm, ck) in D:
                L.append(row(ck if ck != "final" else str(steps), D[(arm, ck)]))
        base = D.get((arm, "0 (SFT)"))
        for ck, _ in CKS[1:]:
            if base and (arm, ck) in D:
                L.append(f"- after {ck if ck != 'final' else steps} steps vs SFT: {pair(D[(arm, ck)], base)}")
        L.append("")
    L += ["## B vs A at equal RL steps", ""]
    for ck, _ in CKS:
        if ("A", ck) in D and ("B", ck) in D:
            L.append(f"- step {ck}: B vs A {pair(D[('B', ck)], D[('A', ck)])}")
    t, _ = load("results/test500_formatP1L.jsonl")
    if t:
        L += ["", "Teacher gpt-oss-120b (low, 1 attempt): " + str(sum(r["status"] == "correct" for r in t.values())) + "/500."]
        for arm in ("A", "B"):
            if (arm, "final") in D:
                L.append(f"- {arm} final vs teacher: {pair(D[(arm, 'final')], t)}")
    L += ["", "## Training curve (sample accuracy on fresh puzzles at temperature 1.0, 30-step blocks)", ""]
    for arm in ("A", "B"):
        p = Path(f"results/rlL_{arm}.log")
        if p.exists():
            s = p.read_text(errors="ignore")
            rew = [float(x) for x in re.findall(r"'reward': '([-0-9.e]+)'", s)]
            ln = [float(x) for x in re.findall(r"'completions/mean_length': '([0-9.e]+)'", s)]
            zero = [float(x) for x in re.findall(r"'frac_reward_zero_std': '([0-9.e]+)'", s)]
            if rew:
                blocks = [rew[i:i + 30] for i in range(0, len(rew), 30)]
                L.append(f"- {arm} ({len(rew)} steps): " + " → ".join(f"{mean(b):.3f}" for b in blocks))
                L.append(f"  - output length {mean(ln[:30]):.0f} → {mean(ln[-30:]):.0f} tokens; groups with no signal "
                         f"{mean(zero[:30]):.0%} → {mean(zero[-30:]):.0%}")
        if arm in meta:
            d = meta[arm]
            L.append(f"  - {d['minutes']} min, {60 * d['minutes'] / d['max_steps']:.0f} s/step, {d['samples']:,} samples, "
                     f"{d['parse_fails']} parse fails")
    L += ["", "## Is B's reasoning truer after RL? (claim checker on the text before FINAL_LINE)", ""]
    for ck, _ in CKS:
        if ("B", ck) in D:
            s = explain_stats(D[("B", ck)], P)
            parts = []
            for key in ("right", "wrong"):
                nb = s[f"{key}_body"]
                if nb:
                    parts.append(f"{key} answers ({nb}): clean {100 * s[f'{key}_clean'] / nb:.0f}%, "
                                 f"move flag {100 * s[f'{key}_move_flag'] / nb:.0f}%")
            L.append(f"- step {ck}: " + "; ".join(parts))
    L += ["", "Earlier 100-step test (different puzzles, `results/rl_test_report.md`): A 284 → 275, B 250 → 264."]
    Path("results/rl_long_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
