"""Zero-shot (or fine-tuned) evaluation of a small student model on puzzles, same prompt + grading as the teacher.

The student is served by a second vLLM container (OpenAI API on another port). Uses run_pilot's
build_prompt / grade / grade_line, so numbers are comparable with the teacher's P1L runs.

Usage: .venv/bin/python eval_student.py --model Qwen/Qwen3-1.7B --port 8001 --tag qwen3_1.7b \
          [--thinking] [--max-tokens 4096] [--puzzles pilot_subset140.jsonl] [--concurrency 16] [--format P1L]
Output: results/student_<tag>_<think|nothink>.jsonl (resumable).
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from openai import AsyncOpenAI

from run_pilot import build_prompt, grade, grade_line, final_line_strict

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_thinking(text):
    """Return (thinking, answer). An unclosed <think> means the model never finished thinking."""
    text = text or ""
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip()
    if "<think>" in text:
        return text.split("<think>", 1)[1], ""
    return None, text


async def solve(client, sem, args, p, out_f, lock, prog):
    if args.prompt == "compact":
        from make_answer_data import compact_prompt
        prompt = compact_prompt(p)
    else:
        prompt = build_prompt(p, args.format)
    rec = {k: p[k] for k in ("puzzle_id", "rating", "rating_band", "themes", "fen", "correct_move")}
    rec.update(model=args.model, tag=args.tag, thinking=args.thinking, format=args.format, prompt=prompt)
    sampling = ({"temperature": 0.6, "top_p": 0.95} if args.thinking else {"temperature": 0.7, "top_p": 0.8})
    if args.temperature is not None:
        sampling["temperature"] = args.temperature
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=args.served_name or args.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens, **sampling,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": args.thinking}})
            raw = resp.model_dump()
            ch = raw["choices"][0]
            text = ch["message"].get("content") or ""
            reasoning = ch["message"].get("reasoning") or ch["message"].get("reasoning_content")
            thinking, answer = split_thinking(text)
            rec.update(raw_text=text, reasoning=reasoning or thinking, content=answer,
                       finish_reason=ch["finish_reason"], usage=raw.get("usage"),
                       latency_s=round(time.perf_counter() - t0, 2))
            status, parsed, uci = grade(p, answer)
            if ch["finish_reason"] == "length" and status == "parse_fail":
                status = "truncated"
            rec.update(status=status, parsed_move=parsed, move_uci=uci, correct=status == "correct",
                       final_line_strict=final_line_strict(answer))
            if args.format in ("P1L",):
                rec.update(grade_line(p, answer))
        except Exception as e:
            rec.update(status="error", error=f"{type(e).__name__}: {e}", correct=False)
    async with lock:
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        prog["done"] += 1
        prog[rec["status"]] = prog.get(rec["status"], 0) + 1
        if prog["done"] % 20 == 0 or prog["done"] == prog["total"]:
            print(f"[{args.tag}] {prog['done']}/{prog['total']} "
                  f"{ {k: v for k, v in prog.items() if k not in ('done', 'total')} }", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-name", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--puzzles", default="pilot_subset140.jsonl")
    ap.add_argument("--format", default="P1L")
    ap.add_argument("--prompt", default="p1l", choices=["p1l", "compact"], help="compact = make_answer_data.compact_prompt")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    puzzles = [json.loads(l) for l in open(args.puzzles)]
    out = Path(args.out or f"results/student_{args.tag}_{'think' if args.thinking else 'nothink'}.jsonl")
    done = set()
    if out.exists():
        done = {json.loads(l)["puzzle_id"] for l in open(out) if json.loads(l).get("status") != "error"}
    todo = [p for p in puzzles if p["puzzle_id"] not in done]
    print(f"[{args.tag}] {len(puzzles)} puzzles, {len(done)} done, {len(todo)} to run -> {out}")
    client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY", timeout=3600, max_retries=0)
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    prog = {"done": 0, "total": len(todo)}
    t0 = time.perf_counter()
    with out.open("a") as f:
        await asyncio.gather(*(solve(client, sem, args, p, f, lock, prog) for p in todo))
    print(f"[{args.tag}] wall {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
