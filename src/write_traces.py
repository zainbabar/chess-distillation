"""LLM-written training traces in the "feigned discovery" style (Master Distillation, arXiv 2603.20510).

The writer (gpt-oss) gets the position (perception P1) and the verified solution line (Lichess's full
solution, best line only, not several engine lines), and writes the reasoning a strong player would go
through to *find* it, as if it didn't know the answer. Variants:
  FD   feigned discovery, solution line only (as in the paper)
  FDF  FD + a python-chess FACTS block (line_facts.py) so claims about captures/checks/material are grounded
  FDFT FDF + one tempting wrong move with Stockfish's refutation (needs --analysis): the text first tries it,
       shows why it fails, then finds the solution (a verified "search" trace)

Usage: .venv/bin/python write_traces.py --run NAME --variant FD|FDF --puzzles FILE --effort low|medium
       [--concurrency 32] [--max-tokens 8192] [--limit N]
Output: results/traces_<run>_<variant>.jsonl (resumable; errors retried on re-run).
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import chess
from openai import AsyncOpenAI

from line_facts import facts_text
from perception import perception_block
from run_pilot import ENGINE_HINT_RE, grade, grade_line

BASE_URL, MODEL = "http://localhost:8000/v1", "openai/gpt-oss-120b"

FD_INSTRUCTIONS = (
    "Write the reasoning a strong chess player would go through to FIND this solution over the board, "
    "as if you did not know it yet: notice what stands out in the position, consider the forcing moves, "
    "calculate the solution line, and arrive at the move. Use 4 to 10 sentences, scaled to how complex "
    "the puzzle is. Refer to pieces with their squares (e.g. \"the queen on h5\") and give moves in "
    "standard notation (e.g. Qxh7+). Only state things that are true in this position.{facts_rule} "
    "Never mention an engine, a computer, evaluations, ratings, themes, a database, or that you were "
    "given the solution.\n"
    "Then end with two lines exactly of the form:\n"
    "FINAL_LINE: <the solution line as UCI moves separated by spaces>\n"
    "FINAL_MOVE: <the first move in UCI>"
)
FACTS_RULE = (" Every claim about captures, checks, threats, pins, forks, material and mate must agree "
              "with the FACTS above.")
# FDF2 (2026-09-25): stricter rules aimed at the claims writers invented most (defended/hanging pieces, "only reply",
# "double check", squares a piece "controls"), shorter text, and the v2 FACTS block (attackers/defenders, reply counts).
FDF2_INSTRUCTIONS = (
    "Write the reasoning a strong chess player would go through to FIND this solution over the board, as if you did "
    "not know it yet: notice what stands out, consider the forcing moves, calculate the solution line, and arrive at "
    "the move. Use 3 to 7 sentences. Accuracy rules: (1) every fact you state about the position or a move (captures, "
    "checks, threats, what attacks or defends what, material, mate) must be in the FACTS; (2) use the words pin, "
    "fork, skewer, discovered, double check, hanging, trapped, overloaded, sacrifice, forced, only move or only reply "
    "only when the FACTS support them (\"forced\" or \"only\" only when the FACTS say there is 1 legal move); "
    "(3) do not describe which squares a piece controls or covers; (4) mention only pieces, squares and moves that "
    "appear on the board or in the FACTS. Refer to pieces with their squares (e.g. \"the queen on h5\") and give moves "
    "in standard notation. Never mention an engine, a computer, evaluations, ratings, themes, a database, or that you "
    "were given the solution.\n"
    "Then end with two lines exactly of the form:\n"
    "FINAL_LINE: <the solution line as UCI moves separated by spaces>\n"
    "FINAL_MOVE: <the first move in UCI>"
)


def build_prompt(p, variant):
    board = chess.Board(p["fen"])
    sol_san, b = [], board.copy()
    for u in p["full_solution"]:
        m = chess.Move.from_uci(u)
        sol_san.append(b.san(m))
        b.push(m)
    lines = ["Here is a chess puzzle.", f"Position (FEN): {p['fen']}",
             f"Side to move: {p['side_to_move'].capitalize()}", "", perception_block(board, level=1), "",
             f"The solution (verified correct): {' '.join(sol_san)}  [UCI: {' '.join(p['full_solution'])}]", ""]
    if variant in ("FDF", "FDFT"):
        lines += [facts_text(p["fen"], p["full_solution"], p.get("themes")), ""]
    if variant == "FDF2":
        lines += [facts_text(p["fen"], p["full_solution"], p.get("themes"), v2=True), "", FDF2_INSTRUCTIONS]
        return "\n".join(lines)
    alt = tempting_alternative(p) if variant == "FDFT" else None
    if alt:
        b2, alt_san = board.copy(), []
        for u in alt["line_uci"]:
            m = chess.Move.from_uci(u)
            alt_san.append(b2.san(m))
            b2.push(m)
        outcome = "only about equal" if alt["win_pct"] > 30 else "losing"
        lines += [f"A tempting move that does NOT work (verified): {alt['san']}. Best play after it: "
                  f"{' '.join(alt_san)} -> {outcome} for the side to move.",
                  facts_text(p["fen"], alt["line_uci"][:4], None, with_start=False).replace(
                      "- The solution line, move by move:", "- That failing line, move by move:"), ""]
    instr = FD_INSTRUCTIONS.format(facts_rule=FACTS_RULE if variant in ("FDF", "FDFT") else "")
    if alt:
        instr = instr.replace("calculate the solution line, and arrive at the move.",
                              f"first consider the tempting {alt['san']} and show briefly why it fails, then "
                              "calculate the solution line and arrive at the move.")
    lines.append(instr)
    return "\n".join(lines)


ANALYSIS = {}


def tempting_alternative(p):
    """A forcing-looking wrong candidate from Stockfish's multi-PV analysis (check/capture preferred)."""
    a = ANALYSIS.get(p["puzzle_id"])
    if not a:
        return None
    board = chess.Board(p["fen"])
    alts = [c for c in a["candidates"] if c["move"] != p["correct_move"] and c["win_pct"] < 70 and len(c["line_uci"]) >= 2]
    if not alts:
        return None
    def forcing(c):
        m = chess.Move.from_uci(c["move"])
        return board.gives_check(m) or board.is_capture(m)
    alts.sort(key=lambda c: (not forcing(c), -c["win_pct"]))
    return alts[0]


async def write_one(client, sem, args, p, out_f, lock, prog):
    prompt = build_prompt(p, args.variant)
    rec = {k: p[k] for k in ("puzzle_id", "rating", "rating_band", "themes", "fen", "correct_move", "full_solution")}
    rec.update(variant=args.variant, effort=args.effort, writer=MODEL, prompt=prompt)
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                reasoning_effort=args.effort, max_tokens=args.max_tokens)
            raw = resp.model_dump()
            msg = raw["choices"][0]["message"]
            rec.update(reasoning=msg.get("reasoning") or msg.get("reasoning_content"), content=msg.get("content"),
                       finish_reason=raw["choices"][0]["finish_reason"], usage=raw.get("usage"),
                       latency_s=round(time.perf_counter() - t0, 2))
            status, parsed, uci = grade(p, rec["content"])
            rec.update(status=status, **grade_line(p, rec["content"]))
            hit = ENGINE_HINT_RE.search(rec["content"] or "")
            rec["mentions_hint"] = hit.group(0) if hit else None
        except Exception as e:
            rec.update(status="error", error=f"{type(e).__name__}: {e}")
    async with lock:
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        prog["done"] += 1
        if prog["done"] % 25 == 0 or prog["done"] == prog["total"]:
            print(f"[{args.variant}/{args.effort}] {prog['done']}/{prog['total']}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", required=True, choices=["FD", "FDF", "FDFT", "FDF2"])
    ap.add_argument("--analysis", default=None, help="engine_analysis.py output (FDFT)")
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    puzzles = [json.loads(l) for l in open(args.puzzles)]
    if args.analysis:
        ANALYSIS.update({json.loads(l)["puzzle_id"]: json.loads(l) for l in open(args.analysis)})
    if args.limit:
        puzzles = puzzles[:: max(1, len(puzzles) // args.limit)][: args.limit]
    out = Path(f"results/traces_{args.run}_{args.variant}.jsonl")
    done = set()
    if out.exists():
        done = {json.loads(l)["puzzle_id"] for l in open(out) if json.loads(l).get("status") != "error"}
    todo = [p for p in puzzles if p["puzzle_id"] not in done]
    print(f"[{args.variant}/{args.effort}] {len(puzzles)} puzzles, {len(done)} done, {len(todo)} to run -> {out}")
    client = AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY", timeout=3 * 3600, max_retries=0)
    sem, lock, prog = asyncio.Semaphore(args.concurrency), asyncio.Lock(), {"done": 0, "total": len(todo)}
    t0 = time.perf_counter()
    with out.open("a") as f:
        await asyncio.gather(*(write_one(client, sem, args, p, f, lock, prog) for p in todo))
    wall = time.perf_counter() - t0
    with open(f"results/traces_{args.run}_meta.jsonl", "a") as f:
        f.write(json.dumps({"variant": args.variant, "effort": args.effort, "n": len(todo), "wall_s": round(wall, 1),
                            "concurrency": args.concurrency}) + "\n")
    print(f"[{args.variant}/{args.effort}] wall {wall:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
