"""Tool-based claim checker for chess texts (no LLM): extract checkable claims and verify them with python-chess.

In the spirit of ACT-Eval ("Hallucinations on the Board", arXiv 2608.04240): LLM judges can't reliably
spot false chess claims, so decompose the text into atomic claims and check each with a tool.

Checked (hard errors -> the text is not "clean"; "move" is soft since 2026-09-24, see below):
  piece  "the queen on h5" / "White's rook on d1": that piece (and colour, if named) stands on that square
         in the start position or somewhere along a verified line
  move   SAN moves (Qxh7+, exd5, O-O): legal somewhere along a verified line (start position, solution
         line, or Stockfish's candidate lines); a '#' must be mate and a '+' must be check there
  gain   "wins the queen / a rook / the exchange / a piece": the solution line captures such a piece and
         ends ahead in material
  mate   "checkmate" / "mate": the solution line ends in mate (or the puzzle is tagged mate)
Soft (reported, not counted as errors): motif words (fork, pin, skewer, discovered, sacrifice) that no
fact or puzzle tag supports.

check(text, puzzle, analysis=None) -> dict(errors=[...], soft=[...], n_claims=..., clean=bool)
"""
import re

import chess

from line_facts import facts_for_line, start_facts

PIECES = "king|queen|rook|bishop|knight|pawn"
PIECE_RE = re.compile(rf"\b(?:(white|black)(?:'s)?\s+)?({PIECES})s?\s+(?:on|at|from|sitting on|standing on)\s+([a-h][1-8])\b", re.I)
PIECE_RE2 = re.compile(rf"\b(white|black)?\s*\b([a-h][1-8])[-\u2010\u2011\u2012\u2013\u2014 ]({PIECES})\b", re.I)
SAN_RE = re.compile(r"(?<![\w/])((?:[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h]x[a-h][1-8](?:=[QRBN])?|[a-h][18]=[QRBN]|O-O(?:-O)?)[+#]?)(?![\w])")
GAIN_RE = re.compile(r"\b(?:win|wins|winning|won|nets?|netting|picks? up|picking up)\s+(?:back\s+)?(?:the |a |an |his |her |their |white's |black's |the opponent's |an entire |a whole )?"
                     r"(?:(white|black)(?:'s)?\s+)?(queen|rook|bishop|knight|piece|exchange)\b", re.I)
MATE_RE = re.compile(r"\b(checkmate|checkmates|mate|mates|mating)\b", re.I)
NEG_RE = re.compile(r"\b(no|not|n't|never|without|avoid|avoids|isn't|doesn't|cannot|can't)\b[^.;]{0,40}$", re.I)
MOTIFS = {"fork": r"\bfork", "pin": r"\bpin(s|ned|ning)?\b", "skewer": r"\bskewer", "discovered": r"\bdiscover",
          "sacrifice": r"\bsacrific"}
VALUE = {"queen": 9, "rook": 5, "bishop": 3, "knight": 3}


def _boards(puzzle, analysis=None):
    """All positions on verified lines: start, after each solution ply, and along Stockfish's candidate lines."""
    start = chess.Board(puzzle["fen"])
    boards = [start.copy()]
    lines = [puzzle["full_solution"]]
    if analysis:
        lines += [c["line_uci"] for c in analysis.get("candidates", [])]
    for line in lines:
        b = start.copy()
        for u in line:
            m = chess.Move.from_uci(u)
            if m not in b.legal_moves:
                break
            b.push(m)
            boards.append(b.copy())
    return boards


def _negated(text, start):
    return bool(NEG_RE.search(text[max(0, start - 60):start]))


def check(text, puzzle, analysis=None):
    text = re.split(r"\n\s*FINAL_LINE:", text or "")[0]
    boards = _boards(puzzle, analysis)
    facts, end = facts_for_line(puzzle["fen"], puzzle["full_solution"])
    hero = chess.Board(puzzle["fen"]).turn
    themes = set(puzzle.get("themes", []))
    errors, soft, n = [], [], 0

    # piece-on-square claims
    for m in list(PIECE_RE.finditer(text)) + list(PIECE_RE2.finditer(text)):
        if m.re is PIECE_RE:
            color, ptype, sq = m.group(1), m.group(2).lower(), m.group(3)
        else:
            color, sq, ptype = m.group(1), m.group(2), m.group(3).lower()
        n += 1
        pt = chess.PIECE_NAMES.index(ptype)
        s = chess.parse_square(sq)
        ok = False
        for b in boards:
            p = b.piece_at(s)
            if p and p.piece_type == pt and (color is None or p.color == (color.lower() == "white")):
                ok = True
                break
        if not ok:
            errors.append(("piece", m.group(0)))

    # SAN moves
    for m in SAN_RE.finditer(text):
        san = m.group(1)
        n += 1
        core = san.rstrip("+#")
        legal_somewhere, annotation_ok = False, False
        for b in boards:  # OK if legal in some verified position where its +/# annotation is also right
            try:
                mv = b.parse_san(core)
            except ValueError:
                continue
            legal_somewhere = True
            after = b.copy()
            after.push(mv)
            if ((san.endswith("#") and after.is_checkmate()) or (san.endswith("+") and after.is_check())
                    or not san.endswith(("+", "#"))):
                annotation_ok = True
                break
        # soft, not hard: prose often names moves to say they're impossible or hypothetical ("Kg2 is covered",
        # "Qg4+ is illegal because the pawn is pinned", "the threat of ...Rxf1+") - 2026-09-24 review of 6
        # rejected pilot traces: most "move" flags were such false alarms; the line itself is verified separately
        if not legal_somewhere:
            soft.append(("move", san))
        elif not annotation_ok:
            soft.append(("move", san + (" (not mate)" if san.endswith("#") else " (not check)")))

    # material gain claims
    hero_caps = [f["captures"]["piece"] for f in facts if f.get("hero_move") and f.get("captures")]
    bal = facts[-1].get("balance_after", 0) if facts and "illegal" not in facts[-1] else 0
    mate_line = bool(facts) and bool(facts[-1].get("checkmate"))
    for m in GAIN_RE.finditer(text):
        if _negated(text, m.start()) or re.match(r"\s*(exchange|trade|swap|endgame|ending)", text[m.end():]):
            continue
        n += 1
        what = m.group(2).lower()
        if mate_line:
            continue  # material talk in a mating line is secondary; don't police it
        if what == "exchange":
            ok = "rook" in hero_caps and bal > 0
        elif what == "piece":
            ok = any(c in ("bishop", "knight", "rook", "queen") for c in hero_caps) and bal > 0
        else:
            ok = what in hero_caps and bal > 0
        if not ok:
            errors.append(("gain", m.group(0)))

    # mate claims
    mate_tag = bool({"mate", "mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5"} & themes)
    for m in MATE_RE.finditer(text):
        if _negated(text, m.start()) or re.search(r"(threat|threaten|avoid|stalemate|mating net|mate threat)",
                                                   text[max(0, m.start() - 25):m.end() + 10], re.I):
            continue
        n += 1
        if not (mate_line or mate_tag):
            errors.append(("mate", text[max(0, m.start() - 30):m.end() + 10].replace("\n", " ")))
        break  # one mate claim is enough to check

    # motif words (soft)
    sf = start_facts(puzzle["fen"])
    support = {
        "fork": any(f.get("fork") and f.get("hero_move") for f in facts) or "fork" in themes,
        "pin": bool(sf["pins"]) or bool({"pin", "skewer", "xRayAttack"} & themes) or any(
            b.is_pinned(c, s) for b in boards[:len(puzzle["full_solution"]) + 1] for c in (True, False)
            for s in chess.SquareSet(b.occupied_co[c] & ~b.kings)),
        "skewer": "skewer" in themes or "xRayAttack" in themes,
        "discovered": any(f.get("discovered_check") or f.get("discovered_attacks") for f in facts
                          if f.get("hero_move")) or bool({"discoveredAttack", "doubleCheck"} & themes),
        "sacrifice": any(f.get("sacrifice") for f in facts) or bool({"sacrifice", "attraction", "deflection"} & themes),
    }
    for motif, pat in MOTIFS.items():
        for m in re.finditer(pat, text, re.I):
            if _negated(text, m.start()):
                continue
            if not support[motif]:
                soft.append((motif, text[max(0, m.start() - 30):m.end() + 20].replace("\n", " ")))
            break
    return {"errors": errors, "soft": soft, "n_claims": n, "clean": not errors}


if __name__ == "__main__":
    import json
    import sys
    # usage: claim_check.py results_file [text_field]
    field = sys.argv[2] if len(sys.argv) > 2 else "content"
    pz = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("pool_night1.jsonl")}
    an = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("results/night1_engine_analysis.jsonl")}
    rows = [json.loads(l) for l in open(sys.argv[1])]
    clean = 0
    for r in rows:
        res = check(r.get(field) or "", pz.get(r["puzzle_id"], r), an.get(r["puzzle_id"]))
        clean += res["clean"]
        if not res["clean"]:
            print(r["puzzle_id"], res["errors"][:4], res["soft"][:2])
    print(f"clean {clean}/{len(rows)}")
