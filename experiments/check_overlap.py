"""Check a test set against every published training puzzle (splits/train_puzzles.jsonl.gz): shared puzzle ids,
shared source games, and shared positions anywhere along the two solutions. A position is the piece placement, side to
move, castling rights and en-passant square (move counters ignored), so a transposition counts as shared.

    python experiments/check_overlap.py                       # the 500 test puzzles in test_set.jsonl
    python experiments/check_overlap.py --test other.jsonl    # any file of puzzles in the same format

For each shared position it prints how many moves into the test solution it occurs (ply 0 = the position the model
answers from, -1 = before the opponent's move).
"""
import argparse
import gzip
import json
from collections import defaultdict

import chess


def key(board):
    return " ".join(board.fen().split()[:4])


def positions(fen, moves):
    """(position key, ply) along a Lichess puzzle: ply -1 before the opponent's move, 0 = the puzzle position."""
    b = chess.Board(fen)
    out = [(key(b), -1)]
    for i, m in enumerate(moves):
        b.push_uci(m)
        out.append((key(b), i))
    return out


def load_test(path):
    test = [json.loads(line) for line in open(path)]
    return [{"id": p["puzzle_id"], "fen": p["original_fen"], "moves": [p["opponent_move"]] + p["full_solution"],
             "game": p["game_url"].split("#")[0]} for p in test]


def overlap(test, train_path="splits/train_puzzles.jsonl.gz", sets=None):
    """Per training set: shared ids, shared games, and shared positions as (train id, test id, test ply)."""
    ids = {p["id"] for p in test}
    games = {p["game"] for p in test}
    where = defaultdict(list)
    for p in test:
        for k, ply in positions(p["fen"], p["moves"]):
            where[k].append((p["id"], ply))
    res = defaultdict(lambda: {"puzzles": 0, "ids": set(), "games": set(), "positions": set()})
    for line in gzip.open(train_path, "rt"):
        t = json.loads(line)
        names = [s for s in t["sets"] if sets is None or s in sets]
        if not names:
            continue
        hits = {(t["id"], tid, ply) for k, _ in positions(t["fen"], t["moves"].split()) for tid, ply in where.get(k, [])}
        for s in names:
            r = res[s]
            r["puzzles"] += 1
            if t["id"] in ids:
                r["ids"].add(t["id"])
            if t["game"] in games:
                r["games"].add(t["game"])
            r["positions"] |= hits
    return dict(res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default="test_set.jsonl")
    ap.add_argument("--train", default="splits/train_puzzles.jsonl.gz")
    a = ap.parse_args()
    test = load_test(a.test)
    for s, r in sorted(overlap(test, a.train).items()):
        tp = sorted({(tid, ply) for _, tid, ply in r["positions"]})
        print(f"{s}: {r['puzzles']:,} training puzzles | shared ids {len(r['ids'])} | shared games {len(r['games'])} | "
              f"shared positions: {len({h[0] for h in r['positions']})} training puzzles, "
              f"{len({t for t, _ in tp})} test puzzles" + (f", at test plies {sorted({p for _, p in tp})}" if tp else ""))
        for tid, ply in tp:
            print(f"    test {tid} ply {ply}")


if __name__ == "__main__":
    main()
