"""Structured write-up format shared by prompt formats S (solver) and E (engine writer).

The model ends its answer with:
    CANDIDATES: Qxe5 (c3e5), Ng6+ (h4g6)
    LINE 1: c3e5 h5e5 f1f8 | VERDICT: WINS | wins the queen
    LINE 2: h4g6 h8g8 | VERDICT: LOSES | runs out of checks
    CHOSEN: c3e5
    FINAL_LINE: c3e5 h5e5 f1f8
    FINAL_MOVE: c3e5
parse_structured() extracts it; verify_trace.py checks it against python-chess and Stockfish.
"""
import re

UCI = r"[a-h][1-8][a-h][1-8][qrbn]?"
VERDICTS = ("WINS", "LOSES", "EQUAL", "UNCLEAR")

FORMAT_SPEC = (
    "Then write your solution in exactly this format (a short explanation may come before it):\n"
    "CANDIDATES: <only the 2 to 4 most serious candidate moves, not every legal move, in standard "
    "notation with UCI in brackets, e.g. Qxe5 (c3e5), Ng6+ (h4g6)>\n"
    "LINE 1: <the first candidate followed by the expected continuation for both sides, every move in "
    "UCI, e.g. c3e5 h5e5 f1f8> | VERDICT: <WINS, LOSES, EQUAL or UNCLEAR> | <one short reason>\n"
    "LINE 2: <the same for the next candidate> (exactly one numbered LINE per candidate)\n"
    "CHOSEN: <the chosen move in UCI>\n"
    "FINAL_LINE: <the chosen move and its expected continuation, every move in UCI>\n"
    "FINAL_MOVE: <the chosen move in UCI>\n"
    "Every move inside LINE and FINAL_LINE must be UCI: the from-square followed by the to-square, "
    "e.g. e2e4 or g1f3, plus the promotion piece for promotions, e.g. e7e8q. Verdicts are from the "
    "point of view of the side to move."
)

_P = r"^[\s*`\->#]*"  # tolerated line prefix: spaces, markdown bold/code, bullets, headings
_CAND_RE = re.compile(_P + r"CANDIDATES[\s*`]*:(.*)$", re.IGNORECASE | re.MULTILINE)
_LINE_RE = re.compile(
    _P + r"LINE\s*(\d+)[\s*`]*:(.*?)\|[\s*`]*VERDICT[\s*`]*:[\s*`]*(WINS?|LOSES?|EQUAL|DRAW|UNCLEAR)\b[*`]*"
    r"\s*(?:\|(.*))?$", re.IGNORECASE | re.MULTILINE)
_CHOSEN_RE = re.compile(_P + r"CHOSEN[\s*`]*:[\s*`]*(" + UCI + r")", re.IGNORECASE | re.MULTILINE)
_FLINE_RE = re.compile(_P + r"FINAL_LINE[\s*`]*:(.*)$", re.IGNORECASE | re.MULTILINE)
_FMOVE_RE = re.compile(_P + r"FINAL_MOVE[\s*`]*:[\s*`]*(" + UCI + r")", re.IGNORECASE | re.MULTILINE)


def _uci_moves(text):
    return re.findall(UCI, (text or "").lower())


_TOKEN_STRIP = re.compile(r"^\d+\.+|[+#!?,;()`*]|\.+$")


def _line_moves(text, fen):
    """Moves of a LINE as UCI. With a FEN, replay the tokens on the board accepting UCI *or* SAN
    (models mix them: 'Qh3 f6g6 ...'); an unreadable token becomes the marker '??' (line illegal)."""
    if fen is None:
        return _uci_moves(text)
    import chess
    board, out = chess.Board(fen), []
    # "Ne6+ (c5e6) Kg8 (g7g8)" -> "c5e6 g7g8": a move written twice (SAN + bracketed UCI) counts once
    text = re.sub(r"\S+\s*\((" + UCI + r")\)", r"\1", text or "", flags=re.IGNORECASE)
    for raw in text.split():
        tok = _TOKEN_STRIP.sub("", raw)
        if not tok or tok in ("...", "..", "--"):
            continue
        move = None
        for parse in (board.parse_uci, board.parse_san):
            try:
                move = parse(tok.lower() if parse == board.parse_uci else tok)
                break
            except ValueError:
                continue
        if move is None:
            out.append("??")
            break
        out.append(move.uci())
        board.push(move)
    return out


def parse_structured(content, fen=None):
    """Return the structured fields (lower-case UCI) plus structure_ok. Pass the puzzle FEN to accept
    SAN inside LINE / FINAL_LINE."""
    content = content or ""
    cand = _CAND_RE.findall(content)
    cands = []
    if cand:
        bracketed = re.findall(r"\((" + UCI + r")\)", cand[-1].lower())
        cands = bracketed or _uci_moves(cand[-1])
    norm = {"WIN": "WINS", "LOSE": "LOSES", "DRAW": "EQUAL"}
    lines = [{"n": int(n), "moves": _line_moves(mv, fen), "verdict": norm.get(v.upper(), v.upper()),
              "reason": (r or "").strip()}
             for n, mv, v, r in _LINE_RE.findall(content)]
    chosen = _CHOSEN_RE.findall(content)
    fline = _FLINE_RE.findall(content)
    fmove = _FMOVE_RE.findall(content)
    out = {
        "candidates": cands,
        "lines": lines,
        "chosen": chosen[-1].lower() if chosen else None,
        "final_line": _line_moves(fline[-1], fen) if fline else [],
        "final_move": fmove[-1].lower() if fmove else None,
    }
    out["structure_ok"] = bool(cands and lines and out["chosen"] and out["final_line"] and out["final_move"]
                               and all(l["moves"] for l in lines))
    return out
