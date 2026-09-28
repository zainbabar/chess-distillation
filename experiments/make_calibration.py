"""Blind calibration set for the explanation judge.

Picks ~10 engine-grounded traces spanning the judge's verdicts (good/ok/bad) from the low and medium
writers, shuffles them, and writes:
  results/judge_calibration.md        - for the user to rate (no judge scores, no writer label)
  results/judge_calibration_key.json  - the hidden judge scores + writer for each item
Compare the user's ratings with the key to see whether the judge can be trusted.
"""
from pathlib import Path
import json
import random
import sys

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from judge import answer_key

SEED = 11
PICK = [  # (writer label, results file, judge file, verdict, how many)
    ("medium", "results/exp2_E_med_formatE.jsonl", "results/judge_E_med.jsonl", "good", 3),
    ("medium", "results/exp2_E_med_formatE.jsonl", "results/judge_E_med.jsonl", "ok", 2),
    ("medium", "results/exp2_E_med_formatE.jsonl", "results/judge_E_med.jsonl", "bad", 1),
    ("low", "results/night1_A_formatE.jsonl", "results/judge_E_low.jsonl", "good", 1),
    ("low", "results/night1_A_formatE.jsonl", "results/judge_E_low.jsonl", "ok", 2),
    ("low", "results/night1_A_formatE.jsonl", "results/judge_E_low.jsonl", "bad", 1),
]


def main():
    rng = random.Random(SEED)
    puzzles = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("pool_night1.jsonl")}
    analysis = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("results/night1_engine_analysis.jsonl")}
    items, used = [], set()
    for writer, rf, jf, verdict, n in PICK:
        recs = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open(rf)}
        judged = [json.loads(l) for l in open(jf)]
        pool = [j for j in judged if j.get("verdict") == verdict and j["puzzle_id"] not in used]
        for j in rng.sample(pool, min(n, len(pool))):
            used.add(j["puzzle_id"])
            items.append({"writer": writer, "judge": {k: j.get(k) for k in
                          ("accuracy", "coherence", "explains_why", "verdict", "reason", "false_claims")},
                          "puzzle_id": j["puzzle_id"], "text": recs[j["puzzle_id"]]["content"]})
    rng.shuffle(items)

    md = ["# Judge calibration: rate these explanations (blind)", "",
          "For each item: read the position and the verified answer key, then the explanation. Rate it:",
          "- **good**: a strong player could learn from it (correct, clear, explains *why*)",
          "- **ok**: usable but weak (e.g. correct but mostly just lists the moves)",
          "- **bad**: wrong, garbled, or empty",
          "",
          "Judge only the *words*: the move lines have already been machine-checked. Write your rating "
          "after each item (or just tell Claude, e.g. \"1 good, 2 ok, ...\"). The judge's verdicts are "
          "hidden in `judge_calibration_key.json`; don't peek until you've rated.", ""]
    key = []
    for i, it in enumerate(items, 1):
        p = puzzles[it["puzzle_id"]]
        board = chess.Board(p["fen"])
        md += [f"---", "", f"## Item {i} (puzzle rating {p['rating']})", "",
               "```", answer_key(p, analysis[it["puzzle_id"]]), "```", "",
               "**Explanation to rate:**", "", "```", (it["text"] or "").strip(), "```", "",
               "**Your rating (good / ok / bad):** ", "", "**Comment (optional):** ", ""]
        key.append({"item": i, "puzzle_id": it["puzzle_id"], "writer": it["writer"], "judge": it["judge"]})
    open("results/judge_calibration.md", "w").write("\n".join(md))
    json.dump(key, open("results/judge_calibration_key.json", "w"), indent=1)
    print(f"wrote {len(items)} items to results/judge_calibration.md (+ hidden key)")
    print("judge verdicts in the set:", {v: sum(k['judge']['verdict'] == v for k in key) for v in ('good', 'ok', 'bad')})


if __name__ == "__main__":
    main()
