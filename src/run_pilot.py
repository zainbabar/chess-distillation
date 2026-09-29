"""Run the pilot: ask gpt-oss-120b every puzzle in test_set.jsonl, in prompt format A and/or B.

Every result is appended to results/<run>_format<X>.jsonl as soon as it arrives (raw response,
reasoning, final answer, tokens, latency, grade). Re-running the same command resumes: puzzles
already present in the output file are skipped.

Usage: .venv/bin/python run_pilot.py --run pilot --formats A B|P1|P2|P1L|R1|S|E [--limit N] [--concurrency 32]
       [--effort low|medium|high] [--max-tokens 8192] [--puzzles test_set.jsonl]
"""
import argparse
import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import chess
from openai import AsyncOpenAI

import random

from perception import perception_block
from structured import FORMAT_SPEC, parse_structured

BASE_URL = "http://localhost:8000/v1"
MODEL = "openai/gpt-oss-120b"
REASONING_EFFORT = "low"  # overridden by --effort
MAX_TOKENS = 8192  # overridden by --max-tokens
FINAL_RE = re.compile(r"FINAL_MOVE:[\s*`]*([^\s*`]+)", re.IGNORECASE)
UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")

INSTRUCTIONS = (
    "Find the best move. Think it through, then end your reply with a final line exactly of the form:\n"
    "FINAL_MOVE: <move in UCI notation>\n"
    "UCI notation is the from-square followed by the to-square, e.g. e2e4 or g1f3; "
    "add the promotion piece for promotions, e.g. e7e8q."
)


UCI_HELP = ("UCI notation is the from-square followed by the to-square, e.g. e2e4 or g1f3; "
            "add the promotion piece for promotions, e.g. e7e8q.")
LINE_ENDING = (
    "Then end your reply with two final lines exactly of the form:\n"
    "FINAL_LINE: <the move followed by the expected continuation (both sides' moves), as UCI moves "
    "separated by spaces>\n"
    "FINAL_MOVE: <the move in UCI notation>\n" + UCI_HELP
)
# P1L: P1 + calculate the whole forcing line
LINE_INSTRUCTIONS = (
    "Find the best move and calculate the forcing continuation until the win is clear "
    "(checkmate or a decisive material gain). Think it through. " + LINE_ENDING
)
# R1 (rationalization, cascade tier 3): the answer is given; explain it as a discovery + give the line
RATIONALIZE_INSTRUCTIONS = (
    "The best move in this position is {san} ({uci}). Explain the solution as if you were finding it "
    "yourself: note the key features of the position, consider the main candidate moves, show why "
    "{san} works, and calculate the forcing continuation until the win is clear (checkmate or a "
    "decisive material gain). Do not say that you were given the answer. " + LINE_ENDING
)


# S (structured solver, night1 stage B): solve, then write the checkable structured solution
SOLVER_INSTRUCTIONS = (
    "Find the best move. Look at the forcing moves first (checks, captures, threats), then calculate "
    "each serious candidate until its outcome is clear. " + FORMAT_SPEC
)
# E (engine writer, night1 stage A): verified Stockfish analysis given; write it up as own reasoning
ENGINE_WRITER_INSTRUCTIONS = (
    "Write the solution to this puzzle as a strong chess coach who calculated these lines themselves: "
    "explain the key features of the position, why the best move works, and why the other candidates "
    "fail. Use exactly the candidates and lines given above (no others) as your CANDIDATES and LINEs. "
    "Do not mention an engine, a computer, or being given any analysis. "
    + FORMAT_SPEC
)
ENGINE_ANALYSIS_PATH = "results/night1_engine_analysis.jsonl"  # overridden by --engine-analysis
_engine_cache = None


def engine_block(puzzle, board):
    """Verified candidate lines for format E (order shuffled per puzzle so 'first' != 'best')."""
    global _engine_cache
    if _engine_cache is None:
        _engine_cache = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open(ENGINE_ANALYSIS_PATH)}
    cands = list(_engine_cache[puzzle["puzzle_id"]]["candidates"])
    random.Random(puzzle["puzzle_id"]).shuffle(cands)
    out = ["Verified analysis of this position (all of it is correct):"]
    for c in cands:
        if c["mate"] is not None:
            ev = f"mate in {c['mate']}" if c["mate"] > 0 else f"gets mated in {-c['mate']}"
        else:
            ev = f"{c['cp'] / 100:+.1f} pawns"
        verdict = "winning" if c["win_pct"] >= 70 else "losing" if c["win_pct"] <= 30 else "about equal"
        out.append(f"- {c['san']} ({c['move']}): {' '.join(c['line_san'])}  [UCI: {' '.join(c['line_uci'])}]"
                   f" -> {verdict} ({ev})")
    return "\n".join(out) + "\n"


def build_prompt(puzzle, fmt):
    side = puzzle["side_to_move"].capitalize()
    board = chess.Board(puzzle["fen"])
    lines = [
        "Solve this chess puzzle.",
        f"Position (FEN): {puzzle['fen']}",
        f"Side to move: {side}",
    ]
    if fmt == "B":
        legal = sorted(m.uci() for m in board.legal_moves)
        lines.append(f"Legal moves (UCI): {' '.join(legal)}")
    elif fmt in ("P1", "P2", "P1L", "R1", "S", "E"):  # B's legal moves + perception (perception.py)
        lines += ["", perception_block(board, level=2 if fmt == "P2" else 1), ""]
    if fmt == "S":
        lines.append(SOLVER_INSTRUCTIONS)
    elif fmt == "E":
        lines.append(engine_block(puzzle, board))
        lines.append(ENGINE_WRITER_INSTRUCTIONS)
    elif fmt == "P1L":
        lines.append(LINE_INSTRUCTIONS)
    elif fmt == "R1":
        answer = chess.Move.from_uci(puzzle["correct_move"])
        lines.append(RATIONALIZE_INSTRUCTIONS.format(san=board.san(answer), uci=answer.uci()))
    else:
        lines.append(INSTRUCTIONS)
    return "\n".join(lines)


LINE_RE = re.compile(r"FINAL_LINE:([^\n]*)", re.IGNORECASE)
LINE_MOVE_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")
HINT_RE = re.compile(r"(we (are|were) told|we're told|told that|the (prompt|problem|puzzle|question) "
                     r"(says|states|tells|gives)|as (given|stated)|the (given|provided) (answer|move)|"
                     r"(given|provided) (as )?the (best|correct) move)", re.IGNORECASE)


ENGINE_HINT_RE = re.compile(r"(\bengine\b|stockfish|computer|analysis (shows|says|given|provided)|"
                            r"(given|provided|verified) analysis|we (are|were) told|we're told|"
                            r"the (prompt|problem) (says|states|tells|gives))", re.IGNORECASE)


def grade_line(puzzle, content):
    """Compare the model's FINAL_LINE with Lichess's full solution (starting from the puzzle position).

    line_match_len: how many leading moves agree with the solution (any checkmating move is accepted
    on the solution's final move, as Lichess does); full_line_correct: all of them agree.
    """
    solution = puzzle["full_solution"]
    out = {"line_moves": None, "line_legal": None, "line_match_len": 0,
           "solution_len": len(solution), "full_line_correct": False}
    found = LINE_RE.findall(content or "")
    if not found:
        return out
    moves = LINE_MOVE_RE.findall(found[-1].lower())
    out["line_moves"] = moves
    board, legal, match = chess.Board(puzzle["fen"]), True, 0
    for i, uci in enumerate(moves):
        try:
            move = board.parse_uci(uci)
        except ValueError:
            legal = False
            break
        if match == i and i < len(solution):
            if move.uci() == solution[i]:
                match += 1
            elif i == len(solution) - 1:  # last solver move: any mate counts
                after = board.copy()
                after.push(move)
                if after.is_checkmate():
                    match += 1
        board.push(move)
    out.update(line_legal=legal, line_match_len=match, full_line_correct=match == len(solution))
    return out


def grade(puzzle, content):
    """Return (status, parsed_move_text, normalized_uci). status: correct|wrong|illegal|parse_fail."""
    matches = FINAL_RE.findall(content or "")
    if not matches:
        return "parse_fail", None, None
    # tolerate markdown/punctuation and a trailing check/mate marker (e.g. "f4f8+");
    # final_line_strict still records whether the exact format was followed
    raw = matches[-1].strip().strip("`*.,;:'\"()[]").rstrip("+#").lower()
    if not UCI_RE.match(raw):
        return "parse_fail", raw, None
    board = chess.Board(puzzle["fen"])
    try:
        move = board.parse_uci(raw)  # also normalizes king-takes-rook castling notation
    except ValueError:
        return "illegal", raw, None
    if move.uci() == puzzle["correct_move"]:
        return "correct", raw, move.uci()
    if "mateIn1" in puzzle["themes"]:
        board.push(move)
        if board.is_checkmate():
            return "correct", raw, move.uci()
    return "wrong", raw, move.uci()


def final_line_strict(content):
    lines = [l for l in (content or "").strip().splitlines() if l.strip()]
    return bool(lines) and re.fullmatch(r"FINAL_MOVE: [a-h][1-8][a-h][1-8][qrbn]?", lines[-1].strip()) is not None


async def solve(client, sem, puzzle, fmt, out_f, lock, progress):
    prompt = build_prompt(puzzle, fmt)
    rec = {
        "puzzle_id": puzzle["puzzle_id"], "format": fmt, "rating": puzzle["rating"],
        "rating_band": puzzle["rating_band"], "themes": puzzle["themes"], "fen": puzzle["fen"],
        "side_to_move": puzzle["side_to_move"], "correct_move": puzzle["correct_move"],
        "model": MODEL, "reasoning_effort": REASONING_EFFORT, "max_tokens": MAX_TOKENS,
        "prompt": prompt,
    }
    async with sem:
        rec["started_at"] = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=REASONING_EFFORT,
                max_tokens=MAX_TOKENS,
            )
            rec["latency_s"] = round(time.perf_counter() - t0, 3)
            raw = resp.model_dump()
            msg = raw["choices"][0]["message"]
            rec["reasoning"] = msg.get("reasoning") or msg.get("reasoning_content")
            rec["content"] = msg.get("content")
            rec["finish_reason"] = raw["choices"][0]["finish_reason"]
            rec["usage"] = raw.get("usage")
            rec["raw_response"] = raw
            status, parsed, uci = grade(puzzle, rec["content"])
            rec["truncated"] = rec["finish_reason"] == "length"
            if rec["truncated"] and status == "parse_fail":
                status = "truncated"  # hit max_tokens before writing an answer
            if fmt in ("P1L", "R1", "S", "E"):
                rec.update(grade_line(puzzle, rec["content"]))
            if fmt in ("S", "E"):
                rec["structured"] = parse_structured(rec["content"], puzzle["fen"])
            if fmt == "E":  # the written solution must not reveal it was given an analysis
                hit = ENGINE_HINT_RE.search(rec["content"] or "")
                rec["mentions_hint"] = hit.group(0) if hit else None
                hit = ENGINE_HINT_RE.search(rec["reasoning"] or "")
                rec["reasoning_mentions_hint"] = hit.group(0) if hit else None
            if fmt == "R1":  # rationalization should read as a discovery, not "we were told..."
                hit = HINT_RE.search((rec["reasoning"] or "") + "\n" + (rec["content"] or ""))
                rec["mentions_hint"] = hit.group(0) if hit else None
            rec.update(status=status, parsed_move=parsed, move_uci=uci,
                       correct=(status == "correct"), final_line_strict=final_line_strict(rec["content"]))
        except Exception as e:  # keep going; record the failure
            rec["latency_s"] = round(time.perf_counter() - t0, 3)
            rec.update(status="error", error=f"{type(e).__name__}: {e}", correct=False)
    async with lock:
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        progress["done"] += 1
        progress[rec["status"]] = progress.get(rec["status"], 0) + 1
        if progress["done"] % 25 == 0 or progress["done"] == progress["total"]:
            el = time.perf_counter() - progress["t0"]
            counts = {k: v for k, v in progress.items() if k not in ("done", "total", "t0")}
            print(f"[{fmt}] {progress['done']}/{progress['total']} in {el:.0f}s {counts}", flush=True)


async def run_format(puzzles, fmt, out_path, concurrency, timeout):
    done_ids = set()
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            if r.get("status") != "error":  # retry errored requests on resume
                done_ids.add(r["puzzle_id"])
    todo = [p for p in puzzles if p["puzzle_id"] not in done_ids]
    print(f"[{fmt}] {len(puzzles)} puzzles, {len(done_ids)} already done, {len(todo)} to run -> {out_path}")
    if not todo:
        return None
    # No automatic retries: a retry restarts generation from scratch (at high effort that threw away
    # 3 x 15 min per request). Errored requests are re-run by re-running the command (resume).
    client = AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY", timeout=timeout, max_retries=0)
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    progress = {"done": 0, "total": len(todo), "t0": time.perf_counter()}
    t0 = time.perf_counter()
    with out_path.open("a") as out_f:
        await asyncio.gather(*(solve(client, sem, p, fmt, out_f, lock, progress) for p in todo))
    return round(time.perf_counter() - t0, 2)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot", help="run name (output file prefix)")
    ap.add_argument("--formats", nargs="+", default=["A", "B"], choices=["A", "B", "P1", "P2", "P1L", "R1", "S", "E"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N puzzles (smoke test)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--puzzles", default="test_set.jsonl")
    ap.add_argument("--engine-analysis", default="results/night1_engine_analysis.jsonl", help="for format E")
    ap.add_argument("--timeout", type=float, default=3 * 3600, help="per-request timeout, seconds")
    args = ap.parse_args()
    global REASONING_EFFORT, MAX_TOKENS
    REASONING_EFFORT, MAX_TOKENS = args.effort, args.max_tokens
    global ENGINE_ANALYSIS_PATH
    ENGINE_ANALYSIS_PATH = args.engine_analysis

    puzzles = [json.loads(l) for l in open(args.puzzles)]
    if args.limit:
        # spread the smoke test across rating bands
        step = max(1, len(puzzles) // args.limit)
        puzzles = puzzles[::step][: args.limit]
    Path("results").mkdir(exist_ok=True)

    for fmt in args.formats:
        out_path = Path("results") / f"{args.run}_format{fmt}.jsonl"
        started = datetime.now(timezone.utc).isoformat()
        wall = await run_format(puzzles, fmt, out_path, args.concurrency, args.timeout)
        if wall is not None:
            meta = {"run": args.run, "format": fmt, "started_at": started,
                    "finished_at": datetime.now(timezone.utc).isoformat(), "wall_time_s": wall,
                    "n_requested": len(puzzles), "puzzles_file": args.puzzles,
                    "concurrency": args.concurrency, "model": MODEL,
                    "reasoning_effort": REASONING_EFFORT, "max_tokens": MAX_TOKENS,
                    "server_image": "nvcr.io/nvidia/vllm:26.05-py3"}
            with (Path("results") / f"{args.run}_meta.jsonl").open("a") as f:
                f.write(json.dumps(meta) + "\n")
            print(f"[{fmt}] wall time {wall}s")


if __name__ == "__main__":
    asyncio.run(main())
