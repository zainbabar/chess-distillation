"""Did a student learn board tracking? Exact-match accuracy on held-out board-tracking tasks (aux_tasks.py format).
Usage: aux_eval.py --tasks FILE --served-name NAME [--port 8001] --out FILE
'board' tasks: the 8-line diagram must match exactly; 'square' tasks: every 'sq: answer' line must match."""
import argparse
import asyncio
import json

from openai import AsyncOpenAI


def norm(s):
    return [l.strip() for l in (s or "").strip().splitlines() if l.strip()]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = [json.loads(l) for l in open(a.tasks)]
    client = AsyncOpenAI(base_url=f"http://localhost:{a.port}/v1", api_key="EMPTY", timeout=600, max_retries=1)
    sem = asyncio.Semaphore(32)

    async def one(t):
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=a.served_name, messages=[{"role": "user", "content": t["prompt"]}], max_tokens=300,
                    temperature=0, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                out = r.choices[0].message.content or ""
            except Exception as e:
                out = f"ERROR {e}"
        kind = "board" if "Write the board" in t["prompt"] else "square"
        return {"kind": kind, "ok": norm(out) == norm(t["target"]), "out": out[:400]}

    R = await asyncio.gather(*(one(t) for t in T))
    with open(a.out, "w") as f:
        for r in R:
            f.write(json.dumps(r) + "\n")
    for k in ("board", "square"):
        S = [r for r in R if r["kind"] == k]
        if S:
            print(f"{a.served_name} {k}: {sum(r['ok'] for r in S)}/{len(S)} exact")


if __name__ == "__main__":
    asyncio.run(main())
