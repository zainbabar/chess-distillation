"""RL reward v4 ("truth4") for explanation students: replay what the student wrote, in order.

Why (2026-09-28): under v3 the model learned to write the right first move and then another move by its OWN side as the
"reply" (e.g. "Rxf2 Rd3"). The claim checker accepted any move that is legal somewhere along the real solution, and the
written line's legality wasn't rewarded (an illegal-line penalty had been left out because, on its own, it pushes lines
down to one move, as v2 showed). v4 checks the student's own calculation move by move and pays for a playable line:

  +1          right move (FINAL_MOVE)
  +1.0 x      correct prefix of FINAL_LINE / max(line length, solution length)          (right answers only; as v3)
  +0.25       FINAL_LINE is complete and playable: legal when replayed and at least min(3, solution length) half-moves
  -0.25       otherwise: the line breaks down when replayed (an illegal move, or a side moving twice), or it stops
              short (e.g. first move only). Symmetric on purpose: an offline check on the saved test outputs showed that
              with -0.5 for a broken line, v2's first-move-only policy still scored best (0.72 vs 0.47 for v3's
              calculating checkpoint), because the student's continuations are legal only about half the time.
  -0.5        hard false claim (claim checker: piece not on that square, false "wins the X", false mate)
  -0.25       the text names an invented move (v3's test) OR writes a run of moves that can't be played in that
              order: 2+ moves separated only by spaces or move numbers ("Rxf2 Rd3", "1. Rxf2 Nxe4 2. Bxe4"), replayed
              from every position on the real solution and on the student's own legal line. Comma or "or" lists
              ("Qe8, Qc4") are alternatives, not sequences, and are not replayed.
  -0.5        floor: fewer than min(2, solution length) distinct verified moves in the text, under 40 words, or no
              FINAL_MOVE (as v3)

For a right answer on a 3-half-move puzzle: a legal 3-move line with wrong continuations 1.58; the fully right line
2.25; first move only, or a line that breaks down (v3's trick), 1.08 (minus 0.25 if the text also writes the broken
sequence). So a playable calculation always pays 0.5 more than stopping or faking it.
"""
import re

import chess

from claim_check import SAN_RE, _boards, check
from run_pilot import grade, grade_line

SEQ_GAP_RE = re.compile(r"^\s*(?:\d+\s*\.(?:\s*\.\.)?\s*|\.\.\.\s*)?$")  # only spaces and move numbers between moves


def _runs(text):
    """Runs of 2+ SAN moves written as a sequence (separated only by whitespace / move numbers)."""
    ms = list(SAN_RE.finditer(text))
    runs, cur = [], ms[:1]
    for prev, m in zip(ms, ms[1:]):
        if SEQ_GAP_RE.match(text[prev.end():m.start()]):
            cur.append(m)
        else:
            if len(cur) > 1:
                runs.append([x.group(1) for x in cur])
            cur = [m]
    if len(cur) > 1:
        runs.append([x.group(1) for x in cur])
    return runs


def _playable(run, starts):
    for start in starts:
        b = start.copy()
        try:
            for san in run:
                b.push(b.parse_san(san.rstrip("+#")))
        except ValueError:
            continue
        return True
    return False


def own_line_boards(puzzle, moves):
    """Positions along the legal prefix of the student's own FINAL_LINE."""
    b, out = chess.Board(puzzle["fen"]), []
    for u in moves or []:
        try:
            b.push(b.parse_uci(u))
        except ValueError:
            break
        out.append(b.copy())
    return out


def truth4(puzzle, text):
    """Return (reward, info) for one completion. info has the components, for logging."""
    status, _, _ = grade(puzzle, text)
    g = grade_line(puzzle, text)
    body = text.split("FINAL_LINE")[0].strip()
    c = check(body, puzzle)
    moves, sol_len = g["line_moves"] or [], g["solution_len"]
    info = {"status": status, "line_frac": 0.0, "illegal_line": False, "legal_bonus": False, "claim_err": bool(c["errors"]),
            "move_flag": False, "seq_flag": False, "short": False}
    r = 0.0
    if status == "correct":
        info["line_frac"] = g["line_match_len"] / max(1, len(moves), sol_len)
        r += 1.0 + 1.0 * info["line_frac"]
    if moves and g["line_legal"] and len(moves) >= min(3, sol_len):
        info["legal_bonus"] = True
        r += 0.25
    else:
        info["illegal_line"] = len(moves) >= 2 and not g["line_legal"]
        r -= 0.25
    if c["errors"]:
        r -= 0.5
    flagged = {x[1] for x in c["soft"] if x[0] == "move"}
    starts = _boards(puzzle) + own_line_boards(puzzle, moves)
    info["move_flag"] = bool(flagged)
    info["seq_flag"] = any(not _playable(run, starts) for run in _runs(body))
    if info["move_flag"] or info["seq_flag"]:
        r -= 0.25
    n_ok = len(set(SAN_RE.findall(body))) - len(flagged)
    if n_ok < min(2, sol_len) or len(body.split()) < 40 or status == "parse_fail":
        info["short"] = True
        r -= 0.5
    return r, info
