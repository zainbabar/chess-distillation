"""Answer-only SFT data (move + line, no reasoning) from any puzzle pool — the recommended core data.

--prompt p1l      the teacher's P1L prompt (board diagram + piece lists + legal moves grouped, SAN+UCI; ~680 tokens)
--prompt compact  board diagram + a flat legal-move list "SAN (uci)" + the instruction (~40% of the tokens);
                  answers are still UCI, so the same grader works
Usage: .venv/bin/python make_answer_data.py --pool pool_collect2.jsonl --prompt compact --out results/sft/ans_compact_200k.jsonl [--limit N]
"""
import argparse
import json

import chess

from perception import board_diagram
from run_pilot import build_prompt

COMPACT_TAIL = ("Find the best move and the forcing line. End with two lines: FINAL_LINE: <the move and the expected "
                "continuation as UCI moves> and FINAL_MOVE: <the move in UCI>.")


def compact_prompt(p):
    b = chess.Board(p["fen"])
    moves = sorted(b.legal_moves, key=b.san)
    return "\n".join([f"Chess puzzle, {p['side_to_move'].capitalize()} to move.", board_diagram(b),
                      "Legal moves: " + ", ".join(f"{b.san(m)} ({m.uci()})" for m in moves), COMPACT_TAIL])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--prompt", default="p1l", choices=["p1l", "compact"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    n = 0
    with open(a.out, "w") as f:
        for line in open(a.pool):
            p = json.loads(line)
            prompt = build_prompt(p, "P1L") if a.prompt == "p1l" else compact_prompt(p)
            f.write(json.dumps({"puzzle_id": p["puzzle_id"], "rating": p["rating"], "source": f"answer_{a.prompt}",
                                "prompt": prompt, "target": f"FINAL_LINE: {' '.join(p['full_solution'])}\n"
                                                            f"FINAL_MOVE: {p['correct_move']}"}) + "\n")
            n += 1
            if a.limit and n >= a.limit:
                break
    print(f"wrote {n} examples ({a.prompt} prompt) to {a.out}")


if __name__ == "__main__":
    main()
