"""Build the Puzzle Explorer: one self-contained web page over the 500 test puzzles and every model's graded answer.

Reads test_set.jsonl and the graded outputs (results/, or the committed copies in reports/outputs/), and writes
  docs/index.html                  full page for GitHub Pages (Settings -> Pages -> branch master, folder /docs)
  <--artifact-out>                 the same page without the <html>/<head> wrapper (for a claude.ai preview)
from the template experiments/explorer_template.html (its <!--BODY--> marker splits head from body).

Per puzzle it stores the start position, the Lichess solution, and for each model: the grade, the move, the written line
replayed on the board (stopping at the first illegal move) and, for the explanation students, the text. Newer runs
(A on 200k more puzzles, the v4 reward) appear automatically once their evaluation files exist.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import chess
import chess.svg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from path_A_report import load
from puzzle_rating import rate

MODELS = [  # key, label, short label, file
    ("tl", "gpt-oss-120b, low effort", "Teacher · low", "results/test500_formatP1L.jsonl"),
    ("tm", "gpt-oss-120b, medium effort", "Teacher · medium", "results/test500_med_formatP1L.jsonl"),
    ("A", "Student A: answers only", "Student A", "results/student_path_A_nothink.jsonl"),
    ("A2", "Student A + 200k more puzzles", "A + 200k", "results/student_path_A200k_nothink.jsonl"),
    ("B", "Student B: teacher explanations", "Student B", "results/student_path_B_nothink.jsonl"),
    ("rL", "B + RL, answer-only reward", "B + RL answer", "results/student_rlL_B_final_nothink.jsonl"),
    ("r1", "B + RL, truth v1", "B + RL v1", "results/student_rlT_B_final_nothink.jsonl"),
    ("r2", "B + RL, strict v2", "B + RL v2", "results/student_rlT2_B_final_nothink.jsonl"),
    ("r3", "B + RL, v3 after 390 steps", "B + RL v3", "results/student_rlT3_B_final_nothink.jsonl"),
    ("r4", "B + RL, v4 (replays the line)", "B + RL v4", "results/student_rlT4_B_final_nothink.jsonl"),
]
MATE_RE = re.compile(r"checkmate|is mate\b|mate in \w+|forced mate|delivers mate|mating", re.I)
FINAL_RE = re.compile(r"\n?\s*FINAL_(LINE|MOVE):.*", re.S)


def replay(fen, ucis):
    """SAN, placements and moves (from, to) for a line; stops at the first illegal move (index in 'bad')."""
    b = chess.Board(fen)
    san, pl, mv, bad = [], [], [], -1
    for i, u in enumerate(ucis or []):
        try:
            m = b.parse_uci(u)
        except ValueError:
            bad = i
            break
        san.append(b.san(m))
        mv.append([m.from_square, m.to_square])
        b.push(m)
        pl.append(b.board_fen())
    return {"san": san, "pl": pl, "mv": mv, "bad": bad, "raw": list(ucis or [])}


def own_side_twice(fen, ucis):
    b = chess.Board(fen)
    try:
        return len(ucis) >= 2 and chess.Move.from_uci(ucis[1]) in b.legal_moves
    except ValueError:
        return False


def text_of(content, limit):
    t = FINAL_RE.sub("", content or "").strip()
    return t[:limit] + ("…" if len(t) > limit else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="experiments/explorer_template.html")
    ap.add_argument("--pages-out", default="docs/index.html")
    ap.add_argument("--artifact-out", default=None)
    a = ap.parse_args()

    P = [json.loads(l) for l in open("test_set.jsonl")]
    D = {}
    for key, label, short, path in MODELS:
        R, _ = load(path)
        if R:
            D[key] = R
    models = [{"k": k, "label": l, "short": s} for k, l, s, _ in MODELS if k in D]

    puzzles = []
    for p in sorted(P, key=lambda p: p["rating"]):
        i, fen = p["puzzle_id"], p["fen"]
        entry = {"id": i, "r": p["rating"], "band": p["rating_band"], "th": p["themes"], "fen": fen,
                 "sol": replay(fen, p["full_solution"]), "m": {}}
        for key, R in D.items():
            r = R.get(i)
            if not r:
                continue
            ln = replay(fen, r.get("line_moves") or ([r["move_uci"]] if r.get("move_uci") else []))
            e = {"st": r["status"], "ln": ln, "tok": (r.get("usage") or {}).get("completion_tokens")}
            if key not in ("tl", "tm", "A", "A2"):
                e["tx"] = text_of(r.get("content"), 1800)
            elif key in ("tl", "tm"):
                t = text_of(r.get("content"), 700)
                if t:
                    e["tx"] = t
            entry["m"][key] = e
        m = entry["m"]
        flags = []
        if "A" in m and "tm" in m:
            if m["A"]["st"] == "correct" and m["tm"]["st"] != "correct":
                flags.append("beat")
            if m["A"]["st"] != "correct" and m["tm"]["st"] == "correct":
                flags.append("lost")
        if "r3" in m and own_side_twice(fen, m["r3"]["ln"]["raw"]):
            flags.append("hack")
        if "rL" in m and "mate" not in " ".join(p["themes"]) and MATE_RE.search(m["rL"].get("tx", "")):
            flags.append("fmate")
        entry["f"] = flags
        puzzles.append(entry)

    summary = []
    for mod in models:
        R = D[mod["k"]]
        rt = rate(list(R.values()))
        summary.append({"k": mod["k"], "solved": sum(r["status"] == "correct" for r in R.values()), "n": len(R),
                        "rating": rt["rating"]})
    pieces = {s: chess.svg.PIECES[s] for s in chess.svg.PIECES}
    data = {"models": models, "summary": summary, "puzzles": puzzles, "pieces": pieces}
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")

    tpl = Path(a.template).read_text()
    head, body = tpl.split("<!--BODY-->")
    body = body.replace("/*__DATA__*/null", blob)
    Path(a.pages_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.pages_out).write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"{head}</head>\n<body>\n{body}</body>\n</html>\n")
    if a.artifact_out:
        Path(a.artifact_out).write_text(head + body)
    print(f"{len(puzzles)} puzzles, {len(models)} models ({', '.join(m['k'] for m in models)}); "
          f"data {len(blob) / 1e6:.2f} MB -> {a.pages_out}" + (f", {a.artifact_out}" if a.artifact_out else ""))


if __name__ == "__main__":
    main()
