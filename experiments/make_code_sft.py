"""Code-built SFT data (no LLM) for a whole pool: results/collect1/sft_code.jsonl (+ board-tracking tasks)."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from code_trace import code_trace
from run_pilot import build_prompt

pool, out = sys.argv[1], sys.argv[2]
n = 0
with open(out, "w") as f:
    for line in open(pool):
        p = json.loads(line)
        try:
            target = code_trace(p)
        except Exception as e:  # keep going; count failures
            print("skip", p["puzzle_id"], type(e).__name__, e)
            continue
        f.write(json.dumps({"puzzle_id": p["puzzle_id"], "rating": p["rating"], "rating_band": p["rating_band"],
                            "themes": p["themes"], "source": "code", "prompt": build_prompt(p, "P1L"),
                            "target": target}) + "\n")
        n += 1
print(f"wrote {n} code-built examples to {out}")
