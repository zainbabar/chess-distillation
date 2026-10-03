"""Overnight 09-29/30 (results/run_overnight4.sh) -> results/overnight4_report.md.

(1) Student B with the answer first (B's explanations, reordered: A's exact target, then the explanation) against A
    and B, on the development set (the original 500) and the fresh set; pass-1 checkpoints on the development set.
    Its explanations are claim-checked like B's.
(2) Sampled decoding (temperature 0.7, top-p 0.8, top-k 20; one sample) vs greedy for A, B, A+200k, B-answer-first,
    and against the teacher (which is always sampled).
(3) The earlier RL finals on the fresh set, next to their development-set scores.
Missing files are skipped, so it can run on a partial night.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim_check import check  # noqa: E402
from path_A_report import load  # noqa: E402
from teacher_report import pair  # noqa: E402

DEV, FRESH = "test_set.jsonl", "fresh_test_set.jsonl"
R = lambda p: load(p)[0]  # noqa: E731


def acc(D):
    return f"{sum(r['status'] == 'correct' for r in D.values())}/{len(D)} ({100 * sum(r['status'] == 'correct' for r in D.values()) / len(D):.1f}%)"


def explanation(r):
    """The explanation text: before FINAL_LINE (B), or after the FINAL_MOVE line (answer first)."""
    c = (r.get("content") or "").strip()
    if c.startswith("FINAL_LINE"):
        m = re.search(r"FINAL_MOVE:[^\n]*\n(.*)", c, re.S)
        return m.group(1).strip() if m else ""
    return re.split(r"\n\s*FINAL_LINE:", c)[0]


def honesty(D, P):
    right = [r for r in D.values() if r["status"] == "correct"]
    res = [check(explanation(r), P[r["puzzle_id"]]) for r in right]
    clean = sum(x["clean"] for x in res)
    flags = sum(any(s[0] == "move" for s in x["soft"]) for x in res)
    empty = sum(not explanation(r) for r in right)
    return f"{clean}/{len(right)} claim-clean, {flags}/{len(right)} with a move flag, {empty} with no explanation"


def main():
    P = {**{json.loads(l)["puzzle_id"]: json.loads(l) for l in open(DEV)},
         **{json.loads(l)["puzzle_id"]: json.loads(l) for l in open(FRESH)}}
    L = ["# Overnight 09-29/30: answer-first explanations, sampled decoding, RL on the fresh set", ""]

    # 1. answer first
    sets = {"development set": {"A": "results/student_path_A_nothink.jsonl", "B": "results/student_path_B_nothink.jsonl",
                                "Baf": "results/student_path_Baf_nothink.jsonl"},
            "fresh set": {"A": "results/fresh_student_path_A.jsonl", "B": "results/fresh_student_path_B.jsonl",
                          "Baf": "results/fresh_student_path_Baf.jsonl"}}
    L += ["## 1. B's explanations with the answer first", "",
          "Same 37,543 puzzles, texts and recipe as B; each target is A's exact target (FINAL_LINE + FINAL_MOVE) followed",
          "by the teacher's explanation. Greedy, up to 1,024 new tokens.", ""]
    for name, paths in sets.items():
        D = {k: R(v) for k, v in paths.items()}
        if not D["Baf"]:
            L.append(f"- {name}: B-answer-first not evaluated yet")
            continue
        L += [f"**{name}:** A {acc(D['A'])}, B {acc(D['B'])}, **B answer-first {acc(D['Baf'])}**", "",
              f"- B answer-first vs A: {pair(D['Baf'], D['A'])}",
              f"- B answer-first vs B: {pair(D['Baf'], D['B'])}",
              f"- explanations (right answers): B {honesty(D['B'], P)}; B answer-first {honesty(D['Baf'], P)}", ""]
    ep = {k: R(v) for k, v in {"A1": "results/student_path_A_ep1_nothink.jsonl", "B1": "results/student_path_B_ep1_nothink.jsonl",
                                "Baf1": "results/student_path_Baf_ep1_nothink.jsonl"}.items()}
    if ep["Baf1"]:
        L += [f"After pass 1 (development set): A {acc(ep['A1'])}, B {acc(ep['B1'])}, B answer-first {acc(ep['Baf1'])}; "
              f"B answer-first vs A: {pair(ep['Baf1'], ep['A1'])}", ""]

    # 2. sampled vs greedy
    L += ["## 2. Sampled decoding vs greedy", "",
          "Students sampled once at Qwen3's recommended non-thinking settings (temperature 0.7, top-p 0.8, top-k 20), the",
          "way the teacher is always run; greedy = the main evaluation.", "",
          "| Student | Development: greedy | Development: sampled | Fresh: greedy | Fresh: sampled |", "|---|---|---|---|---|"]
    rows = [("A", "path_A", "results/fresh_student_path_A.jsonl"), ("B", "path_B", "results/fresh_student_path_B.jsonl"),
            ("A + 200k", "path_A200k", "results/fresh_student_path_A200k.jsonl"),
            ("B answer-first", "path_Baf", "results/fresh_student_path_Baf.jsonl")]
    S = {}
    for label, tag, fg in rows:
        cells = [R(f"results/student_{tag}_nothink.jsonl"), R(f"results/student_{tag}_sampled_nothink.jsonl"),
                 R(fg), R(f"results/fresh_student_{tag}_sampled.jsonl")]
        S[label] = cells
        L.append(f"| {label} | " + " | ".join(acc(c) if c else "–" for c in cells) + " |")
    L.append("")
    med = {"development": R("results/test500_med_formatP1L.jsonl"), "fresh": R("results/fresh_med_formatP1L.jsonl")}
    for label in ("A", "A + 200k"):
        dev_s, fresh_s = S[label][1], S[label][3]
        if dev_s:
            L.append(f"- {label} sampled vs teacher medium, development set: {pair(dev_s, med['development'])}")
        if fresh_s:
            L.append(f"- {label} sampled vs teacher medium, fresh set: {pair(fresh_s, med['fresh'])}")
        if S[label][0] and dev_s:
            L.append(f"- {label} sampled vs greedy, development set: {pair(dev_s, S[label][0])}")
    L.append("")

    # 3. RL finals on the fresh set
    L += ["## 3. The RL runs on the fresh set (final checkpoints, greedy; evaluated after the protocol)", "",
          "| Run | Development set | Fresh set |", "|---|---|---|"]
    for label, tag, fresh in [("A, answer-only reward", "rlL_A_final", "results/fresh_student_rlL_A_final.jsonl"),
                              ("B, answer-only reward", "rlL_B_final", "results/fresh_student_rlL_B_final.jsonl"),
                              ("B, truth v1", "rlT_B_final", "results/fresh_student_rlT_B_final.jsonl"),
                              ("B, strict v2", "rlT2_B_final", "results/fresh_student_rlT2_B_final.jsonl"),
                              ("B, v3", "rlT3_B_final", "results/fresh_student_rlT3_B_final.jsonl"),
                              ("B, v4", "rlT4_B_final", "results/fresh_student_rlT4_B_final.jsonl")]:
        d, f = R(f"results/student_{tag}_nothink.jsonl"), R(fresh)
        L.append(f"| {label} | {acc(d) if d else '–'} | {acc(f) if f else '–'} |")
    Path("results/overnight4_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
