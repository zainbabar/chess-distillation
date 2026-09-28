"""Build SFT data (prompt/target pairs) for the student pilot arms, all on the same puzzles.

  --arm code     code-built explanation (code_trace.py), no LLM
  --arm answer   answer only: FINAL_LINE + FINAL_MOVE, no reasoning (control)
  --arm llm      LLM-written trace from --traces (kept only if: move right, line = solution, claim-check clean)
Prompt = run_pilot.build_prompt(puzzle, "P1L"): exactly the evaluation prompt.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from claim_check import check
from code_trace import code_trace
from run_pilot import build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--arm", required=True, choices=["code", "answer", "llm"])
    ap.add_argument("--traces", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pz = [json.loads(l) for l in open(args.puzzles)]
    T = {}
    if args.arm == "llm":
        T = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open(args.traces)}
    n, dropped = 0, {}
    with open(args.out, "w") as f:
        for p in pz:
            if args.arm == "code":
                target = code_trace(p)
            elif args.arm == "answer":
                target = f"FINAL_LINE: {' '.join(p['full_solution'])}\nFINAL_MOVE: {p['correct_move']}"
            else:
                t = T.get(p["puzzle_id"])
                reason = (None if t else "missing") or (None if t.get("status") == "correct" else "wrong_move") or \
                         (None if t.get("full_line_correct") else "line") or \
                         (None if not t.get("mentions_hint") else "leak") or \
                         (None if check(t["content"], p)["clean"] else "claims")
                if reason:
                    dropped[reason] = dropped.get(reason, 0) + 1
                    continue
                body = t["content"].split("FINAL_LINE:")[0].strip()
                target = body + f"\nFINAL_LINE: {' '.join(p['full_solution'])}\nFINAL_MOVE: {p['correct_move']}"
            f.write(json.dumps({"puzzle_id": p["puzzle_id"], "prompt": build_prompt(p, "P1L"), "target": target}) + "\n")
            n += 1
    print(f"{args.arm}: wrote {n} examples to {args.out}; dropped {dropped}")


if __name__ == "__main__":
    main()
