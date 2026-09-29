"""THE PATH step 2 result: student A (answers only) vs student B (teacher explanations) -> results/path_AB_report.md.

Same base (Qwen3-1.7B), same 37,543 puzzles, same prompt, same recipe (full fine-tune, 2 passes, lr 1e-5); only the
training target differs. Both evaluated greedily on the 500 held-out test puzzles (max 1,024 new tokens).
Also: is B's own reasoning true? (claim_check on its written explanation, as student_explain_check.py.)
"""
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from analyze_pilot import mcnemar
from claim_check import check
from path_A_report import load
from puzzle_rating import rate

ARMS = {
    "A2": ("A: answers only, 2 passes", "results/student_path_A_nothink.jsonl", "ckpt/path_A"),
    "A1": ("A after pass 1", "results/student_path_A_ep1_nothink.jsonl", None),
    "B2": ("B: teacher explanations, 2 passes", "results/student_path_B_nothink.jsonl", "ckpt/path_B"),
    "B1": ("B after pass 1", "results/student_path_B_ep1_nothink.jsonl", None),
    "T": ("teacher gpt-oss-120b (low, 1 attempt)", "results/test500_formatP1L.jsonl", None),
    "L": ("night 2: answers only, LoRA 21.6k × 1", "results/student_pilot_night2_scale_nothink.jsonl", None),
}
PAIRS = [("B2", "A2"), ("B1", "A1"), ("B2", "B1"), ("B2", "T"), ("B2", "L"), ("A2", "T")]


def explain_stats(R, P):
    s = Counter()
    for pid, r in R.items():
        body = (r.get("content") or "").split("FINAL_LINE")[0].strip()
        key = "right" if r["status"] == "correct" else "wrong"
        s[f"{key}_n"] += 1
        if not body:
            continue
        c = check(body, P[pid])
        s[f"{key}_body"] += 1
        s[f"{key}_clean"] += c["clean"]
        s[f"{key}_move_flag"] += any(x[0] == "move" for x in c["soft"])
    return s


def main():
    test = [json.loads(l) for l in open("test_set.jsonl")]
    P = {p["puzzle_id"]: p for p in test}
    band = {p["puzzle_id"]: p["rating_band"] for p in test}
    D, notes = {}, []
    L = ["# Student A (answers only) vs student B (teacher explanations)", "",
         "Qwen3-1.7B, full fine-tune, 2 passes, lr 1e-5, the same 37,543 training puzzles and prompt; only the target text",
         "differs. 500 held-out test puzzles, greedy, max 1,024 new tokens.", "",
         "| Model | Correct | Rating (95% CI) | Illegal | Parse fail / truncated | Legal line | Full line right | Mean output tokens |",
         "|---|---|---|---|---|---|---|---|"]
    for k, (label, path, _) in ARMS.items():
        R, n_lines = load(path)
        if not R:
            notes.append(f"- missing: {label} ({path})")
            continue
        D[k] = R
        rs = list(R.values())
        st = Counter(r["status"] for r in rs)
        rt = rate(rs)
        toks = [r["usage"]["completion_tokens"] for r in rs if r.get("usage")]
        mt = f"{mean(toks):.0f}" if toks else "–"
        L.append(f"| {label} | **{st['correct']}/{len(rs)} ({100 * st['correct'] / len(rs):.1f}%)** | "
                 f"{rt['rating']} ({rt['ci_low']}–{rt['ci_high']}) | {st['illegal']} | {st['parse_fail']} / {st['truncated']} | "
                 f"{sum(bool(r.get('line_legal')) for r in rs)} | {sum(bool(r.get('full_line_correct')) for r in rs)} | "
                 f"{mt} |")
        if len(R) != 500 or n_lines != len(R) or set(R) - set(P):
            notes.append(f"- {label}: {len(R)} puzzles graded from {n_lines} records")
    L += ["", "Paired comparisons (exact McNemar; only-first-right vs only-second-right):", ""]
    for a, b in PAIRS:
        if a in D and b in D:
            ids = set(D[a]) & set(D[b])
            x = sum(D[a][i]["status"] == "correct" and D[b][i]["status"] != "correct" for i in ids)
            y = sum(D[b][i]["status"] == "correct" and D[a][i]["status"] != "correct" for i in ids)
            L.append(f"- {ARMS[a][0]} vs {ARMS[b][0]}: {x} vs {y}, p = {mcnemar(x, y):.2g}")
    bands = sorted(set(band.values()), key=lambda b: int(b.split("-")[0]))
    L += ["", "Correct by rating band:", "", "| Model | " + " | ".join(bands) + " |", "|---|" + "---|" * len(bands)]
    for k in D:
        cnt = Counter(band[i] for i, r in D[k].items() if r["status"] == "correct")
        tot = Counter(band[i] for i in D[k])
        L.append(f"| {ARMS[k][0]} | " + " | ".join(f"{cnt[b]}/{tot[b]}" for b in bands) + " |")
    L += ["", "Is B's own reasoning true? (claim checker on the text before FINAL_LINE; hard errors = piece not on that",
          "square, false \"wins the X\", false mate; move flag = mentions a move that is illegal/mis-annotated where written)", ""]
    for k in ("B2", "B1"):
        if k in D:
            s = explain_stats(D[k], P)
            for key in ("right", "wrong"):
                n, nb = s[f"{key}_n"], s[f"{key}_body"]
                if nb:
                    L.append(f"- {ARMS[k][0]}, {key} answers ({n}): wrote an explanation {nb}; claim-clean {s[f'{key}_clean']} "
                             f"({100 * s[f'{key}_clean'] / nb:.0f}%); move flag {s[f'{key}_move_flag']} "
                             f"({100 * s[f'{key}_move_flag'] / nb:.0f}%)")
    L += ["", "Training:", ""]
    for k in ("A2", "B2"):
        m = Path(ARMS[k][2], "train_meta.json")
        if m.exists():
            d = json.load(open(m))
            L.append(f"- {ARMS[k][0]}: {d['examples']:,} examples, {d['tokens']:,} tokens/pass ({d['target_tokens']:,} target), "
                     f"{d['minutes']} min, final loss {d['final_loss']:.3f}")
    if notes:
        L += ["", "Notes / sanity checks:", ""] + notes
    Path("results/path_AB_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
