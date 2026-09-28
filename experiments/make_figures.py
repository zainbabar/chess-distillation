"""Figures for the README write-up -> figures/*.png. Reads the graded outputs in results/ (run after the evals).

1. ratings.png: puzzle rating (95% bootstrap CI) of the teacher (low / medium effort) and the two full-fine-tuned students,
   annotated with accuracy and output tokens per answer.
2. by_band.png: share solved per rating band: teacher low, teacher medium, student A.
3. rl_rewards.png: student B during RL under the four rewards (checkpoints 0 = SFT, 130, 260, 390 steps): accuracy and
   the text/line checks that show how each reward failed.
4. example_sacrifice.svg / example_reward_hack.svg: two test puzzles drawn with python-chess (moves as arrows).
5. social_preview.png: 1280x640 link-preview image for GitHub.
"""
import json
import sys
from pathlib import Path
from statistics import mean

import chess
import chess.svg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from path_A_report import load
from puzzle_rating import rate
from rl_truth_report import stats

OUT = Path("figures")
P = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("pilot_set.jsonl")}
BANDS = ["800-1000", "1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", "2000-2200"]
TEACHER_LOW, TEACHER_MED = "results/test500_formatP1L.jsonl", "results/test500_med_formatP1L.jsonl"
STUDENT_A, STUDENT_B = "results/student_path_A_nothink.jsonl", "results/student_path_B_nothink.jsonl"
GREY, DARK, BLUE, ORANGE = "#9e9e9e", "#424242", "#1f77b4", "#e07b39"
REWARDS = [("answer only", "rlL_B", "#d62728"), ("truth v1", "rlT_B", "#9467bd"),
           ("strict v2", "rlT2_B", "#2ca02c"), ("v3", "rlT3_B", "#1f77b4")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def solved(R):
    return sum(r["status"] == "correct" for r in R.values())


def legal_replies(R):
    """Written lines whose first move and the opponent's reply are both legal (a 1-move line doesn't count)."""
    n = 0
    for i, r in R.items():
        mv = r.get("line_moves") or []
        if len(mv) < 2:
            continue
        b = chess.Board(P[i]["fen"])
        try:
            for m in mv[:2]:
                move = chess.Move.from_uci(m)
                if move not in b.legal_moves:
                    break
                b.push(move)
            else:
                n += 1
        except ValueError:
            pass
    return n


def approx(n):
    return f"{round(n, -2):,.0f}" if n >= 1000 else f"{n:.0f}"


def fig_ratings():
    rows = [("gpt-oss-120b, low effort", TEACHER_LOW, GREY), ("gpt-oss-120b, medium effort", TEACHER_MED, DARK),
            ("Student B: teacher explanations", STUDENT_B, ORANGE), ("Student A: answers only", STUDENT_A, BLUE)]
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    for y, (label, path, color) in enumerate(rows):
        R, _ = load(path)
        rt = rate(list(R.values()))
        toks = mean(r["usage"]["completion_tokens"] for r in R.values() if r.get("usage"))
        ax.errorbar(rt["rating"], y, xerr=[[rt["rating"] - rt["ci_low"]], [rt["ci_high"] - rt["rating"]]], fmt="o",
                    color=color, capsize=4, ms=8, lw=2)
        ax.text(1680, y, f"{100 * solved(R) / len(R):.1f}% solved, ~{approx(toks)} tokens/answer", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(1320, 2010)
    ax.set_xlabel("Puzzle rating on the 500 held-out puzzles (95% CI)")
    ax.set_title("A 1.7B student trained on answers matches the 120B teacher at medium effort", fontsize=10.5, loc="left")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ratings.png", dpi=180)


def fig_by_band():
    series = [("gpt-oss-120b, low", TEACHER_LOW, GREY), ("gpt-oss-120b, medium", TEACHER_MED, DARK),
              ("Student A (1.7B, answers only)", STUDENT_A, BLUE)]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    w = 0.27
    for k, (label, path, color) in enumerate(series):
        R, _ = load(path)
        vals = []
        for b in BANDS:
            ids = [i for i, p in P.items() if p["rating_band"] == b]
            vals.append(100 * sum(R[i]["status"] == "correct" for i in ids) / len(ids))
        ax.bar([x + (k - 1) * w for x in range(len(BANDS))], vals, w, label=label, color=color)
    ax.set_xticks(range(len(BANDS)), [b.replace("-", "–") for b in BANDS])
    ax.set_xlabel("Lichess puzzle rating band (71–72 puzzles each)")
    ax.set_ylabel("Solved (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Solved by difficulty", fontsize=10.5, loc="left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "by_band.png", dpi=180)


def fig_rl():
    sft, _ = load(STUDENT_B)
    base = stats(sft, P) | {"replies": legal_replies(sft)}
    panels = [("Solved (of 500)", lambda s: s["correct"]),
              ("False checkmate claims\n(% of 356 non-mate puzzles)", lambda s: 100 * s["false_mate"][0] / s["false_mate"][1]),
              ("Checkable claims per explanation", lambda s: s["claims"]),
              ("Written line length (half-moves)", lambda s: s["line_len"]),
              ("Lines longer than the solution\n(of 500)", lambda s: s["padded"]),
              ("Lines with a legal opponent reply\n(of 500; real calculation)", lambda s: s["replies"])]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    steps = [0, 130, 260, 390]
    for label, tag, color in REWARDS:
        S = [base]
        for ck in ("step130", "step260", "final"):
            R, _ = load(f"results/student_{tag}_{ck}_nothink.jsonl")
            S.append(stats(R, P) | {"replies": legal_replies(R)})
        for ax, (title, f) in zip(axes.flat, panels):
            ax.plot(steps, [f(s) for s in S], "o-", color=color, label=label, lw=2, ms=5)
    A, _ = load(STUDENT_A)
    T, _ = load(TEACHER_MED)
    ref = [axes[0, 0].axhline(solved(A), color=BLUE, ls="--", lw=1, label="student A (answers only)"),
           axes[0, 0].axhline(solved(T), color=DARK, ls=":", lw=1.2, label="teacher, medium effort")]
    axes[0, 0].legend(handles=ref, loc="lower center", frameon=False, fontsize=8)
    axes[0, 0].set_ylim(215, 300)
    for ax, (title, _) in zip(axes.flat, panels):
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks(steps)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("RL steps (0 = after imitation)")
    axes[0, 1].legend(title="reward", frameon=False, fontsize=8.5, title_fontsize=8.5)
    fig.suptitle("Student B under RL: each reward improved one check and broke another", fontsize=11, x=0.02,
                 ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "rl_rewards.png", dpi=180)


def board_svg(name, puzzle_id, arrows=(), play=(), size=300):
    """Draw a test puzzle from the side to move, optionally after playing `play` (uci moves), with (uci, color) arrows."""
    board = chess.Board(P[puzzle_id]["fen"])
    side = board.turn
    for u in play:
        board.push_uci(u)
    kw = {"lastmove": board.peek()} if play else {}
    if board.is_check():
        kw["check"] = board.king(board.turn)
    svg = chess.svg.board(board, orientation=side, size=size, coordinates=True,
                          arrows=[chess.svg.Arrow(chess.parse_square(u[:2]), chess.parse_square(u[2:4]), color=c)
                                  for u, c in arrows], **kw)
    (OUT / name).write_text(svg)


def fig_boards():
    GOOD, BAD = "#15781bcc", "#d62728cc"
    # uURSu: the student plays the forced mate Qxd1+ Kxd1 Re1#; the medium teacher played Qf4+ (about level per Stockfish)
    board_svg("example_sacrifice_1.svg", "uURSu", arrows=[("f3d1", GOOD)])
    board_svg("example_sacrifice_2.svg", "uURSu", play=["f3d1", "c1d1", "e7e1"])
    # yl9Uw: v3's final model writes Rxf2 and then Rd3, a second move by its own side, as the "line"
    board_svg("example_reward_hack.svg", "yl9Uw", arrows=[("e2f2", GOOD), ("d1d3", BAD)])


def fig_social():
    """1280x640 image for GitHub's link preview: the headline plus the rating chart's numbers."""
    rows = [("gpt-oss-120b, low effort", TEACHER_LOW, GREY), ("gpt-oss-120b, medium effort", TEACHER_MED, DARK),
            ("1.7B student, answers only", STUDENT_A, BLUE)]
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.text(0.05, 0.86, "A 1.7B model matches a 120B model at chess tactics", fontsize=25, weight="bold")
    fig.text(0.05, 0.775, "with ~300× fewer tokens. Under RL, the explanation student kept finding ways around our fact-checker.",
             fontsize=16, color="#444444")
    ax = fig.add_axes([0.33, 0.14, 0.62, 0.52])
    for y, (label, path, color) in enumerate(rows):
        R, _ = load(path)
        rt = rate(list(R.values()))
        ax.errorbar(rt["rating"], y, xerr=[[rt["rating"] - rt["ci_low"]], [rt["ci_high"] - rt["rating"]]], fmt="o",
                    color=color, capsize=6, ms=13, lw=3)
        ax.text(rt["ci_high"] + 12, y, f"{100 * solved(R) / len(R):.1f}%", va="center", fontsize=16, color=color)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=16)
    ax.set_xlim(1320, 1760)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlabel("Puzzle rating on 500 held-out Lichess puzzles (95% CI)", fontsize=14)
    ax.grid(axis="x", alpha=0.3)
    fig.text(0.05, 0.04, "github.com/zainbabar/chess-llm-reasoning", fontsize=13, color="#666666")
    fig.savefig(OUT / "social_preview.png", dpi=100)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_ratings()
    fig_by_band()
    fig_rl()
    fig_boards()
    fig_social()
    print("wrote", ", ".join(str(p) for p in sorted(OUT.glob("*.*"))))
