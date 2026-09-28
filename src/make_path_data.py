"""THE PATH step 2 data: two SFT sets on exactly the same puzzles, differing only in the target text.

  B (explanations): the accepted FDF-low explanations (night 2's results/night2/sft_fdf.jsonl + the 34k collection's
    results/collect40k/sft_fdf_all.jsonl); target = explanation + FINAL_LINE + FINAL_MOVE.
  A (answers only): the same rows with the explanation removed; target = FINAL_LINE + FINAL_MOVE (the solution).
Prompt = the P1L evaluation prompt (identical in A and B). Checks: no duplicate puzzles, no test-set puzzles.
Also writes results/sft/path_A_smoke.jsonl (first --smoke rows of A) for the full-fine-tune smoke test.
"""
import argparse
import json

NIGHT2 = "results/night2/sft_fdf.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=400)
    ap.add_argument("--collect", default="results/collect40k/sft_fdf_all.jsonl")
    args = ap.parse_args()
    test_ids = {json.loads(l)["puzzle_id"] for l in open("pilot_set.jsonl")}
    rows, seen = [], set()
    for src in (NIGHT2, args.collect):
        for l in open(src):
            r = json.loads(l)
            assert r["puzzle_id"] not in seen, f"duplicate puzzle {r['puzzle_id']}"
            assert r["puzzle_id"] not in test_ids, f"test puzzle {r['puzzle_id']} in training data"
            seen.add(r["puzzle_id"])
            rows.append(r)
    nb = na = 0
    with open("results/sft/path_B.jsonl", "w") as fb, open("results/sft/path_A.jsonl", "w") as fa:
        for r in rows:
            lines = r["target"].rstrip().split("\n")
            assert lines[-2].startswith("FINAL_LINE: ") and lines[-1].startswith("FINAL_MOVE: "), r["puzzle_id"]
            assert len(lines) > 2 and lines[0].strip(), f"empty explanation {r['puzzle_id']}"
            fb.write(json.dumps(r) + "\n")
            fa.write(json.dumps({**r, "source": "answer", "target": "\n".join(lines[-2:])}) + "\n")
            nb += 1
            na += 1
    with open("results/sft/path_A.jsonl") as f, open("results/sft/path_A_smoke.jsonl", "w") as g:
        for i, l in enumerate(f):
            if i >= args.smoke:
                break
            g.write(l)
    print(f"path_A {na} rows, path_B {nb} rows (same puzzles, 0 test puzzles, 0 duplicates); smoke {args.smoke}")


if __name__ == "__main__":
    main()
