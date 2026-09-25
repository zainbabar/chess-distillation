"""How correct are a trained student's own explanations? Claim-check its outputs on the test set.
Usage: student_explain_check.py results/student_pilot_<arm>_nothink.jsonl"""
import json
import sys
from collections import Counter

from claim_check import check

P = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("pilot_set.jsonl")}
R = {}
for l in open(sys.argv[1]):
    r = json.loads(l)
    R[r["puzzle_id"]] = r  # last record wins (retries)
stats = Counter()
for r in R.values():
    if r.get("status") == "error":
        continue
    body = (r.get("content") or "").split("FINAL_LINE")[0].strip()
    if not body:
        continue
    c = check(body, P[r["puzzle_id"]])
    key = "right" if r["status"] == "correct" else "wrong"
    stats[f"{key}_n"] += 1
    stats[f"{key}_clean"] += c["clean"]
    stats[f"{key}_soft_move"] += any(s[0] == "move" for s in c["soft"])
    stats[f"{key}_legal_line"] += bool(r.get("line_legal"))
for key in ("right", "wrong"):
    n = stats[f"{key}_n"]
    if n:
        print(f"{key} answers: {n}; explanation claim-clean {stats[f'{key}_clean']} ({100 * stats[f'{key}_clean'] / n:.0f}%); "
              f"mentions a move that is illegal/wrongly annotated {stats[f'{key}_soft_move']} ({100 * stats[f'{key}_soft_move'] / n:.0f}%); "
              f"legal FINAL_LINE {stats[f'{key}_legal_line']} ({100 * stats[f'{key}_legal_line'] / n:.0f}%)")
