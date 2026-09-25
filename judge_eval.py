"""How reliable are LLM judges? Compare judge variants with Claude's blind ratings (35 items).

Items: results/judge_calibration.md (10, rated in judge_calibration_claude.json) and
results/rating_set2.md (25, rated in rating_set2_claude.json); texts are parsed from the .md files, so
they are exactly what was rated.
Variants:
  v3   the existing judge prompt (answer key + MACHINE CHECK for structured texts)
  v4   v3 + the python-chess FACTS block (line_facts.facts_text) + an instruction to check every
       claim about what a move does against the FACTS
at reasoning effort low / medium. Output: results/judge_eval.jsonl + a summary table on stdout.
"""
import asyncio
import json
import re
import sys
from collections import Counter

from openai import AsyncOpenAI

import judge
from line_facts import facts_text

V4_EXTRA = ("ALSO: the FACTS block below was computed by a chess program and is correct. Check every claim "
            "in words about what a move does or what a piece attacks, defends or pins, sacrifices, material "
            "won or lost, checks and mate, against the FACTS and the VERIFIED ANALYSIS. A claim they "
            "contradict, or a concrete claim they don't support (e.g. a pin, fork, sacrifice or attack that "
            "isn't there), is a false claim. A text with a false claim about the key idea is \"bad\"; a text "
            "that is correct but only restates moves without explaining why is \"ok\".")


def parse_md(path):
    txt = open(path).read()
    items = {}
    for m in re.finditer(r"## Item (\d+)[^\n]*\n(.*?)(?=\n## Item |\Z)", txt, re.S):
        blocks = re.findall(r"```\n(.*?)```", m.group(2), re.S)
        items[int(m.group(1))] = blocks[-1].strip()
    return items


def load_items():
    out = []
    cal_text = parse_md("results/judge_calibration.md")
    cal_key = {k["item"]: k for k in json.load(open("results/judge_calibration_key.json"))}
    cal_me = json.load(open("results/judge_calibration_claude.json"))["ratings"]
    for i, text in cal_text.items():
        k = cal_key[i]
        out.append({"set": "cal", "item": i, "puzzle_id": k["puzzle_id"], "writer": "E_" + k["writer"],
                    "text": text, "claude": cal_me[str(i)][0], "v3_low_existing": k["judge"]["verdict"]})
    s2_text = parse_md("results/rating_set2.md")
    s2_key = {k["item"]: k for k in json.load(open("results/rating_set2_key.json"))}
    s2_me = json.load(open("results/rating_set2_claude.json"))["ratings"]
    for i, text in s2_text.items():
        k = s2_key[i]
        out.append({"set": "s2", "item": i, "puzzle_id": k["puzzle_id"], "writer": k["writer"], "text": text,
                    "claude": s2_me[str(i)]})
    return out


async def run(variant, effort, items, puzzles, analysis, checks):
    client = AsyncOpenAI(base_url=judge.BASE_URL, api_key="EMPTY", timeout=3600, max_retries=0)
    sem = asyncio.Semaphore(16)

    async def one(it):
        p, a = puzzles[it["puzzle_id"]], analysis[it["puzzle_id"]]
        prompt = judge.build_prompt(p, a, it["text"], "explanation", checks.get((it["set"], it["item"])))
        if variant == "v4":
            prompt = prompt.replace(judge.INSTRUCTIONS, facts_text(p["fen"], p["full_solution"], p.get("themes"))
                                    + "\n\n" + judge.INSTRUCTIONS + "\n" + V4_EXTRA)
        async with sem:
            r = await client.chat.completions.create(model=judge.MODEL, messages=[{"role": "user", "content": prompt}],
                                                     reasoning_effort=effort, max_tokens=8192)
        content = r.choices[0].message.content
        v = judge.parse_verdict(content) or {}
        return {**{k: it[k] for k in ("set", "item", "puzzle_id", "writer", "claude")}, "variant": variant,
                "effort": effort, "verdict": v.get("verdict"), "false_claims": v.get("false_claims"),
                "tokens": r.usage.completion_tokens}
    return await asyncio.gather(*(one(it) for it in items))


def main():
    items = load_items()
    puzzles = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("pool_night1.jsonl")}
    analysis = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("results/night1_engine_analysis.jsonl")}
    # machine check for structured texts (as judge.py does)
    import verify_trace
    checks = {}
    for it in items:
        if re.search(r"^[\s*`\->#]*LINE\s*\d", it["text"], re.M | re.I):
            p = puzzles[it["puzzle_id"]]
            checks[(it["set"], it["item"])] = verify_trace.verify(({"puzzle_id": p["puzzle_id"], "content": it["text"],
                                                                   "fen": p["fen"], "rating": p["rating"],
                                                                   "rating_band": p["rating_band"], "status": "correct",
                                                                   "format": "E", "correct_move": p["correct_move"],
                                                                   "full_solution": p["full_solution"]}, 14))
    for eng in verify_trace._engines:
        eng.quit()
    runs = [("v3", "low"), ("v4", "low"), ("v4", "medium")]
    if len(sys.argv) > 1:
        runs = [tuple(x.split(":")) for x in sys.argv[1:]]
    rows = []
    for variant, effort in runs:
        rows += asyncio.run(run(variant, effort, items, puzzles, analysis, checks))
    with open("results/judge_eval.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("| Judge | n | agree with Claude | 'good' precision (Claude also good) | Claude-bad caught (judge bad) | Claude-bad passed as good | mean tokens |")
    print("|---|---|---|---|---|---|---|")
    for variant, effort in runs:
        R = [r for r in rows if r["variant"] == variant and r["effort"] == effort and r["verdict"]]
        agree = sum(r["verdict"] == r["claude"] for r in R)
        good = [r for r in R if r["verdict"] == "good"]
        bad = [r for r in R if r["claude"] == "bad"]
        print(f"| {variant} {effort} | {len(R)} | {agree}/{len(R)} | {sum(r['claude'] == 'good' for r in good)}/{len(good)} | "
              f"{sum(r['verdict'] == 'bad' for r in bad)}/{len(bad)} | {sum(r['verdict'] == 'good' for r in bad)}/{len(bad)} | "
              f"{sum(r['tokens'] for r in R) / max(1, len(R)):.0f} |")
    print("Claude's ratings:", Counter(it["claude"] for it in items))


if __name__ == "__main__":
    main()
