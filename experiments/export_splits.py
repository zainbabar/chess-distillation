"""Write splits/train_puzzles.jsonl.gz: every puzzle any published model was trained on, with what's needed to check
it against the test set (the position before the opponent's move, the moves, the source game) and which training
sets used it. Reads the local pools and training files (not in the repo); `experiments/check_overlap.py` reads the
published file.
"""
import gzip
import json
from pathlib import Path

POOLS = ["pool_collect1.jsonl", "pool_collect2.jsonl", "pool_pilot2k.jsonl", "pool_pilot_llm.jsonl",
         "pool_train1.jsonl", "results/night2/pool.jsonl"]
SETS = {  # training set -> SFT files (results/sft/) that used it
    "sft_1.9k": ["p_answer", "p_code", "p_llm"],
    "sft_5.6k": ["night2_answer", "night2_llm", "night2_code", "night2_aux"],
    "sft_21.6k": ["night2_scale"],
    "sft_37.5k_A_B": ["path_A", "path_B"],
    "sft_200k_A": ["ans_p1l_200k"],
}
RL = ("rl_3120", "pool_collect1.jsonl", 40800, 3120)  # every RL run: rows 40,801-43,920 of pool_collect1, in order


def main():
    pool = {}
    for f in POOLS:
        for line in open(f):
            p = json.loads(line)
            pool.setdefault(p["puzzle_id"], p)
    used = {}
    for name, files in SETS.items():
        for f in files:
            for line in open(f"results/sft/{f}.jsonl"):
                used.setdefault(json.loads(line)["puzzle_id"], set()).add(name)
    name, f, skip, n = RL
    for line in list(open(f))[skip:skip + n]:
        used.setdefault(json.loads(line)["puzzle_id"], set()).add(name)
    Path("splits").mkdir(exist_ok=True)
    with gzip.GzipFile("splits/train_puzzles.jsonl.gz", "wb", compresslevel=9, mtime=0) as g:
        for pid in sorted(used):
            p = pool[pid]
            g.write((json.dumps({"id": pid, "fen": p["original_fen"],
                                 "moves": " ".join([p["opponent_move"]] + p["full_solution"]),
                                 "game": p["game_url"].split("#")[0], "sets": sorted(used[pid])}) + "\n").encode())
    counts = {s: sum(s in v for v in used.values()) for s in list(SETS) + [RL[0]]}
    print(f"splits/train_puzzles.jsonl.gz: {len(used):,} puzzles", counts)


if __name__ == "__main__":
    main()
