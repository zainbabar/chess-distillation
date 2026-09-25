"""Blind rating set for Claude: same puzzles, texts from several writers, shuffled, writer hidden.
Writes results/rating_set2.md (with a facts block per puzzle to check claims against) + hidden key."""
import json, random
from code_trace import code_trace
from line_facts import facts_text
from perception import board_diagram
import chess
rng = random.Random(23)
P = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("pool_night1_engine98.jsonl")}
src = {"E_med": "results/exp2_E_med_formatE.jsonl", "FD_low": "results/traces_w98_FD_low.jsonl",
       "FDF_low": "results/traces_w98_FDF_low.jsonl", "FDF_med": "results/traces_w98_FDF_medium.jsonl"}
T = {k: {json.loads(l)["puzzle_id"]: json.loads(l) for l in open(v)} for k, v in src.items()}
picks = []
for band in ["1000-1200", "1400-1600", "1600-1800", "1800-2000", "2000-2200"]:
    ids = sorted(i for i, p in P.items() if p["rating_band"] == band and all(i in T[k] for k in T))
    picks.append(rng.choice(ids))
items = []
for pid in picks:
    for k in T:
        items.append((pid, k, T[k][pid]["content"]))
    items.append((pid, "code", code_trace(P[pid])))
rng.shuffle(items)
md, key = ["# Rating set 2 (blind): rate good / ok / bad; count false claims", ""], []
for i, (pid, k, text) in enumerate(items, 1):
    p = P[pid]
    md += [f"## Item {i} — puzzle {pid} ({p['rating']}, {p['side_to_move']} to move)", "```", board_diagram(chess.Board(p["fen"])),
           facts_text(p["fen"], p["full_solution"], p["themes"]), "```", "Text:", "```", text.strip(), "```", ""]
    key.append({"item": i, "puzzle_id": pid, "writer": k})
open("results/rating_set2.md", "w").write("\n".join(md))
json.dump(key, open("results/rating_set2_key.json", "w"), indent=1)
print(len(items), "items; puzzles", picks)
