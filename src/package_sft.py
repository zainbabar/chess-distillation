"""Filter LLM-written traces and package them as SFT examples (used by results/run_collect1.sh).

Keep a trace only if: final move right, FINAL_LINE = Lichess solution, no engine/answer leak,
claim_check clean (no false piece-on-square / illegal move / wrong +,# / false "wins the X" / false mate).
Every trace is kept on disk either way (raw file untouched); this writes accepted examples + a reject log.

Output rows: {puzzle_id, rating, rating_band, themes, source: "fdf_low", prompt (student prompt = P1L), target}
"""
import argparse
import json
from collections import Counter

from claim_check import check
from run_pilot import build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    P = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open(args.puzzles)}
    why, n = Counter(), 0
    with open(args.out, "w") as f, open(args.out.replace(".jsonl", ".rejected.jsonl"), "w") as rej:
        for line in open(args.traces):
            t = json.loads(line)
            p = P.get(t["puzzle_id"])
            if p is None or t.get("status") == "error":
                continue
            reason = None
            if t.get("status") != "correct":
                reason = "wrong_move"
            elif not t.get("full_line_correct"):
                reason = "line"
            elif t.get("mentions_hint"):
                reason = "leak"
            else:
                c = check(t["content"], p)
                if not c["clean"]:
                    reason = "claims:" + ",".join(sorted({e[0] for e in c["errors"]}))
            if reason:
                why[reason] += 1
                rej.write(json.dumps({"puzzle_id": t["puzzle_id"], "reason": reason}) + "\n")
                continue
            body = t["content"].split("FINAL_LINE:")[0].strip()
            target = body + f"\nFINAL_LINE: {' '.join(p['full_solution'])}\nFINAL_MOVE: {p['correct_move']}"
            f.write(json.dumps({"puzzle_id": p["puzzle_id"], "rating": p["rating"], "rating_band": p["rating_band"],
                                "themes": p["themes"], "source": "fdf_low", "prompt": build_prompt(p, "P1L"),
                                "target": target}) + "\n")
            n += 1
    print(f"accepted {n}; rejected {dict(why)} -> {args.out}")


if __name__ == "__main__":
    main()
