"""Explanation judge: rate how good a chess explanation (or reasoning transcript) is.

The judge is gpt-oss itself, so it must NOT rely on its own chess (it's only ~1400-1500 at puzzles).
It gets an answer key: Stockfish's verified analysis of the position (top candidates, lines,
evaluations) plus the Lichess solution. Its job is to CHECK the text against that key:
  accuracy      - are the concrete claims (captures, threats, material, which moves win/lose) right?
  coherence     - is it clear and consistent (no garbled sentences, no contradictions)?
  explains_why  - does it explain WHY the best move works and the alternatives fail?
and give a verdict: good / ok / bad. Output per trace -> JSONL (raw judge response kept).

Which text is judged: engine-grounded traces (format E) -> the written explanation (`content`);
solver traces (P1L etc.) -> the hidden `reasoning` (a thinking transcript; judged leniently on ideas
it later rejects).

Usage:
  .venv/bin/python judge.py --results results/night1_A_formatE.jsonl \
      --analysis results/night1_engine_analysis.jsonl --puzzles pool_night1.jsonl \
      --out results/judge_E_low.jsonl [--only-correct] [--effort low] [--concurrency 32]
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import chess
from openai import AsyncOpenAI

from perception import board_diagram, piece_list

BASE_URL = "http://localhost:8000/v1"
MODEL = "openai/gpt-oss-120b"
MAX_TEXT_CHARS = 120_000  # ~30k tokens; longer texts keep head + tail
VERDICTS = ("good", "ok", "bad")

KIND_EXPLANATION = (
    "The text is a written explanation of the solution, meant to teach a student."
)
KIND_REASONING = (
    "The text is a THINKING TRANSCRIPT: the solver thinking aloud before answering. It may try ideas "
    "and reject them; that is fine. Judge the claims it relies on for its conclusions; mistakes that "
    "it notices and corrects itself are not errors. For 'coherence', judge whether the thinking is "
    "followable, not whether it is polished prose."
)

JUDGE_VERSION = 3  # v2: don't judge move legality / move sequences (python-chess does that exactly;
# v1 hallucinated "illegal" moves in 13 of 33 'bad' verdicts on lines python-chess had verified).
# v3: also give the judge the exact MACHINE CHECK of each line (legal? verdict agrees with Stockfish?)
# because an instruction alone didn't stop it contradicting verified lines.

INSTRUCTIONS = """IMPORTANT: every move sequence in the TEXT (LINE entries, FINAL_LINE, UCI moves like e2e4) has already been checked exactly by a chess program for legality and against the engine; where available the results are in the MACHINE CHECK section, which is correct. Do NOT check move legality or move sequences yourself, never contradict the MACHINE CHECK, and never report move sequences as false claims. Judge the WORDS.
Rate the TEXT on three criteria, each from 1 (very poor) to 5 (excellent):
- accuracy: are its claims in words (material won or lost, threats, checks, mate, which moves win or lose, evaluations and verdict words like winning/equal/losing) consistent with the VERIFIED ANALYSIS? List every such claim that is false.
- coherence: is it clear and internally consistent (no garbled or meaningless sentences, no contradictions such as calling a winning line equal)?
- explains_why: does it explain WHY the best move works and why the alternatives fail (threats, weaknesses, the key reply), rather than only listing moves?
Then an overall verdict: "good" (a strong player could learn from it), "ok" (usable but weak), or "bad" (wrong, garbled, or empty).
Do not solve the position yourself: treat the VERIFIED ANALYSIS as correct.
Reply with ONLY a JSON object, no other text:
{"accuracy": 1-5, "coherence": 1-5, "explains_why": 1-5, "false_claims": ["..."], "verdict": "good|ok|bad", "reason": "one sentence"}"""


def fmt_eval(c):
    if c.get("mate") is not None:
        return f"mate in {c['mate']}" if c["mate"] > 0 else f"gets mated in {-c['mate']}"
    return f"{c['cp'] / 100:+.1f} pawns"


def verdict_word(win_pct):
    return "winning" if win_pct >= 70 else "losing" if win_pct <= 30 else "about equal"


def solution_san(fen, uci_moves):
    board, out = chess.Board(fen), []
    for u in uci_moves:
        m = chess.Move.from_uci(u)
        out.append(board.san(m))
        board.push(m)
    return out


def answer_key(puzzle, analysis):
    """Text block: board, verified candidate lines (best first), Lichess solution."""
    board = chess.Board(puzzle["fen"])
    me = board.turn
    side = "White" if me == chess.WHITE else "Black"
    lines = [
        f"Position (FEN): {puzzle['fen']}",
        f"Side to move: {side}",
        "Board (White uppercase, Black lowercase; rank 8 at the top):",
        board_diagram(board),
        f"{side}'s pieces: {piece_list(board, me)}",
        f"Opponent's pieces: {piece_list(board, not me)}",
        "",
        "VERIFIED ANALYSIS (from a strong engine; correct), evaluations from the side to move:",
    ]
    cands = sorted(analysis["candidates"], key=lambda c: -c["win_pct"])
    for i, c in enumerate(cands):
        tag = "BEST MOVE" if c["move"] == puzzle["correct_move"] else f"alternative {i}"
        lines.append(f"- {tag}: {c['san']} ({c['move']}). Main line: {' '.join(c['line_san'])} "
                     f"-> {verdict_word(c['win_pct'])} ({fmt_eval(c)})")
    sol = puzzle.get("full_solution")
    if sol:
        lines.append(f"Known puzzle solution: {' '.join(solution_san(puzzle['fen'], sol))}")
    return "\n".join(lines)


def clip(text):
    text = (text or "").strip()
    if len(text) <= MAX_TEXT_CHARS:
        return text
    half = MAX_TEXT_CHARS // 2
    return text[:half] + "\n[... middle omitted for length ...]\n" + text[-half:]


def machine_check_block(check):
    """Exact facts from verify_trace.verify() about the text's LINE entries (None -> no block)."""
    if not check or not check.get("lines"):
        return None
    out = ["MACHINE CHECK of the move sequences in the TEXT (exact, by a chess program + the engine):"]
    for ln in check["lines"]:
        if not ln["legal"]:
            out.append(f"- LINE {ln['n']}: contains an illegal or unreadable move")
        else:
            agree = "agrees" if ln["verdict_ok"] else "DISAGREES"
            out.append(f"- LINE {ln['n']}: legal; its verdict {ln['verdict']} {agree} with the engine "
                       f"(side to move's win chance at the end of the line: {ln['win_pct']:.0f}%)")
    out.append(f"- chosen move consistent with FINAL_MOVE/FINAL_LINE: {'yes' if check.get('consistent') else 'NO'}")
    return "\n".join(out)


def build_prompt(puzzle, analysis, text, kind, check=None):
    mc = machine_check_block(check)
    return "\n".join([
        "You are checking the quality of a chess text. " + (KIND_REASONING if kind == "reasoning"
                                                             else KIND_EXPLANATION),
        "",
        answer_key(puzzle, analysis),
        "",
        *([mc, ""] if mc else []),
        "TEXT TO CHECK:",
        "<<<",
        clip(text),
        ">>>",
        "",
        INSTRUCTIONS,
    ])


def parse_verdict(content):
    """Last JSON object in the reply with the expected keys, validated. Returns dict or None."""
    content = content or ""
    dec = json.JSONDecoder()
    for i in reversed([m.start() for m in re.finditer(r"\{", content)]):
        try:
            obj, _ = dec.raw_decode(content[i:])
        except ValueError:
            continue
        if not isinstance(obj, dict) or "verdict" not in obj:
            continue
        try:
            out = {k: int(obj[k]) for k in ("accuracy", "coherence", "explains_why")}
        except (KeyError, TypeError, ValueError):
            continue
        if not all(1 <= v <= 5 for v in out.values()):
            continue
        verdict = str(obj["verdict"]).strip().lower()
        if verdict not in VERDICTS:
            continue
        claims = obj.get("false_claims") or []
        out.update(verdict=verdict, reason=str(obj.get("reason", "")),
                   false_claims=[str(c) for c in claims] if isinstance(claims, list) else [str(claims)])
        return out
    return None


def pick_text(rec, field):
    if field == "auto":
        field = "content" if rec.get("format") == "E" else "reasoning"
    return field, rec.get(field) or ""


async def judge_one(client, sem, rec, puzzle, analysis, field, effort, max_tokens, out_f, lock, stats, check=None):
    field, text = pick_text(rec, field)
    kind = "explanation" if field == "content" else "reasoning"
    prompt = build_prompt(puzzle, analysis, text, kind, check if field == "content" else None)
    out = {"puzzle_id": rec["puzzle_id"], "format": rec.get("format"), "rating": rec.get("rating"),
           "rating_band": rec.get("rating_band"), "status": rec.get("status"), "judged_field": field,
           "text_chars": len(text), "judge_effort": effort, "judge_version": JUDGE_VERSION, "prompt": prompt}
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                reasoning_effort=effort, max_tokens=max_tokens)
            raw = resp.model_dump()
            msg = raw["choices"][0]["message"]
            out.update(judge_content=msg.get("content"),
                       judge_reasoning=msg.get("reasoning") or msg.get("reasoning_content"),
                       finish_reason=raw["choices"][0]["finish_reason"], usage=raw.get("usage"))
            parsed = parse_verdict(msg.get("content"))
            out["parse_ok"] = parsed is not None
            if parsed:
                out.update(parsed)
        except Exception as e:
            out.update(parse_ok=False, error=f"{type(e).__name__}: {e}")
        out["latency_s"] = round(time.perf_counter() - t0, 2)
    async with lock:
        out_f.write(json.dumps(out) + "\n")
        out_f.flush()
        stats["done"] += 1
        stats[out.get("verdict", "unparsed")] = stats.get(out.get("verdict", "unparsed"), 0) + 1


async def main_async(args):
    recs = []
    for path in args.results:
        recs += [json.loads(l) for l in open(path)]
    if args.only_correct:
        recs = [r for r in recs if r.get("status") == "correct"]
    analysis = {}
    for path in args.analysis:
        analysis.update({json.loads(l)["puzzle_id"]: json.loads(l) for l in open(path)})
    puzzles = {}
    for path in args.puzzles:
        puzzles.update({json.loads(l)["puzzle_id"]: json.loads(l) for l in open(path)})
    done = set()
    if Path(args.out).exists():  # resume
        done = {(json.loads(l)["puzzle_id"], json.loads(l)["format"]) for l in open(args.out)
                if json.loads(l).get("parse_ok") or not json.loads(l).get("error")}
    todo, missing = [], 0
    for r in recs:
        if (r["puzzle_id"], r.get("format")) in done:
            continue
        if r["puzzle_id"] not in analysis or r["puzzle_id"] not in puzzles:
            missing += 1
            continue
        todo.append(r)
    print(f"{len(recs)} traces, {len(done)} already judged, {missing} without analysis/puzzle, "
          f"{len(todo)} to judge -> {args.out}", flush=True)
    checks = {}
    structured = [r for r in todo if re.search(r"^[\s*`\->#]*LINE\s*\d", r.get("content") or "", re.M | re.I)]
    if structured and args.field in ("auto", "content"):
        import verify_trace
        from concurrent.futures import ThreadPoolExecutor
        try:
            with ThreadPoolExecutor(max_workers=16) as ex:
                # verify() needs the position etc.: take them from the puzzle file, not the trace
                res = list(ex.map(verify_trace.verify, [
                    ({**r, "fen": puzzles[r["puzzle_id"]]["fen"], "rating": puzzles[r["puzzle_id"]]["rating"],
                      "rating_band": puzzles[r["puzzle_id"]]["rating_band"], "status": r.get("status"),
                      "format": r.get("format")}, 14) for r in structured]))
        finally:
            for eng in verify_trace._engines:
                eng.quit()
        checks = {id(r): c for r, c in zip(structured, res)}
        print(f"machine-checked {len(checks)} structured texts", flush=True)
    client = AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY", timeout=3600, max_retries=0)
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    stats = {"done": 0}
    t0 = time.perf_counter()
    with open(args.out, "a") as out_f:
        await asyncio.gather(*(judge_one(client, sem, r, puzzles[r["puzzle_id"]], analysis[r["puzzle_id"]],
                                         args.field, args.effort, args.max_tokens, out_f, lock, stats,
                                         checks.get(id(r)))
                               for r in todo))
    print(f"done in {time.perf_counter() - t0:.0f}s: {stats}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="trace files from run_pilot.py")
    ap.add_argument("--analysis", nargs="+", required=True, help="engine_analysis.py output(s)")
    ap.add_argument("--puzzles", nargs="+", required=True, help="puzzle files (for the Lichess solution)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--field", default="auto", choices=["auto", "content", "reasoning"])
    ap.add_argument("--only-correct", action="store_true")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=32)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
