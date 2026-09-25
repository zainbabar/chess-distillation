"""Machine-computed facts about a puzzle position and a move sequence (python-chess only, so always correct).

Used three ways (strategy work, 2026-09-24 evening):
  1. grounding: a FACTS block given to an LLM writer / judge, so it doesn't have to track the board itself
  2. explanations built by code (code_trace.py), with no LLM at all
  3. claim checking (claim_check.py): compare what a text says with what actually happens

facts_for_line(fen, uci_moves) -> list of per-move dicts; facts_text(...) -> a readable block.
"""
import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
SIDE = {chess.WHITE: "White", chess.BLACK: "Black"}

THEME_WORDS = {  # Lichess theme tags -> plain words (only the tactical ones; phase/length tags skipped)
    "fork": "fork", "pin": "pin", "skewer": "skewer", "discoveredAttack": "discovered attack",
    "doubleCheck": "double check", "deflection": "deflection", "attraction": "attraction (luring a piece)",
    "sacrifice": "sacrifice", "clearance": "clearance", "interference": "interference",
    "hangingPiece": "a hanging (undefended) piece", "trappedPiece": "a trapped piece",
    "capturingDefender": "removing the defender", "xRayAttack": "x-ray attack", "intermezzo": "in-between move",
    "quietMove": "a quiet move", "defensiveMove": "a defensive move", "zugzwang": "zugzwang",
    "promotion": "promotion", "underPromotion": "under-promotion", "advancedPawn": "an advanced pawn",
    "exposedKing": "an exposed king", "kingsideAttack": "kingside attack", "queensideAttack": "queenside attack",
    "backRankMate": "back-rank mate", "smotheredMate": "smothered mate", "anastasiaMate": "Anastasia's mate",
    "arabianMate": "Arabian mate", "bodenMate": "Boden's mate", "doubleBishopMate": "double-bishop mate",
    "dovetailMate": "dovetail mate", "hookMate": "hook mate", "killBoxMate": "kill-box mate",
    "vukovicMate": "Vukovic mate", "mate": "checkmate", "mateIn1": "mate in 1", "mateIn2": "mate in 2",
    "mateIn3": "mate in 3", "mateIn4": "mate in 4", "mateIn5": "mate in 5", "enPassant": "en passant",
    "castling": "castling", "equality": "saving a draw", "crushing": "a decisive advantage",
    "advantage": "a clear advantage",
}


def pname(pt):
    return chess.piece_name(pt)


def desc(board, sq):
    """'White's rook on d4'"""
    p = board.piece_at(sq)
    return f"{SIDE[p.color]}'s {pname(p.piece_type)} on {chess.square_name(sq)}"


def material(board, color):
    return sum(VAL[p.piece_type] for p in board.piece_map().values() if p.color == color)


def attacked_targets(board, sq):
    """Opponent pieces attacked by the piece on sq (kings included)."""
    color = board.piece_at(sq).color
    return [t for t in board.attacks(sq) if board.piece_at(t) and board.piece_at(t).color != color]


def move_facts(board, move):
    """Facts about one move played from `board` (board is not modified)."""
    mover = board.piece_at(move.from_square)
    f = {"san": board.san(move), "uci": move.uci(), "side": SIDE[mover.color],
         "piece": pname(mover.piece_type), "from": chess.square_name(move.from_square),
         "to": chess.square_name(move.to_square), "n_legal_before": board.legal_moves.count()}
    if board.is_castling(move):
        f["castles"] = "kingside" if board.is_kingside_castling(move) else "queenside"
    if board.is_capture(move):
        if board.is_en_passant(move):
            f["captures"] = {"piece": "pawn", "square": chess.square_name(move.to_square), "value": 1, "en_passant": True}
        else:
            cap = board.piece_at(move.to_square)
            f["captures"] = {"piece": pname(cap.piece_type), "square": chess.square_name(move.to_square),
                             "value": VAL[cap.piece_type]}
    if move.promotion:
        f["promotes_to"] = pname(move.promotion)
    after = board.copy()
    after.push(move)
    to = move.to_square
    if after.is_check():
        checkers = list(after.checkers())
        f["check"] = True
        f["checkers"] = [desc(after, s) for s in checkers]
        f["discovered_check"] = to not in checkers
        f["double_check"] = len(checkers) > 1
    f["checkmate"] = after.is_checkmate()
    f["stalemate"] = after.is_stalemate()
    # what the moved piece attacks now (opponent pieces), with defence status
    targets = []
    for t in attacked_targets(after, to):
        p = after.piece_at(t)
        targets.append({"desc": desc(after, t), "piece": pname(p.piece_type), "value": VAL[p.piece_type],
                        "king": p.piece_type == chess.KING,
                        "defended": bool(after.attackers(p.color, t))})
    f["attacks"] = targets
    # discovered attacks: other pieces of the mover that attack an opponent piece (value >= 3 or king)
    # after the move but not before (the moved piece was in the way)
    disc = []
    for sq in chess.SquareSet(after.occupied_co[mover.color]):
        if sq == to:
            continue
        for t in attacked_targets(after, sq):
            p = after.piece_at(t)
            if (VAL[p.piece_type] >= 3 or p.piece_type == chess.KING) and t not in board.attacks(sq):
                disc.append({"by": desc(after, sq), "target": desc(after, t)})
    f["discovered_attacks"] = disc
    # exposed: the move opens a line so that an opponent piece now attacks one of the mover's pieces
    # (e.g. the king steps off a diagonal and the bishop behind it is skewered)
    exposed = []
    for sq in chess.SquareSet(after.occupied_co[not mover.color]):
        for t in attacked_targets(after, sq):
            p = after.piece_at(t)
            if t != to and VAL[p.piece_type] >= 3 and t not in board.attacks(sq):
                exposed.append({"by": desc(after, sq), "target": desc(after, t)})
    f["exposed"] = exposed
    my_val = VAL[mover.piece_type if not move.promotion else move.promotion]
    serious = [t for t in targets if t["king"] or t["value"] > my_val or not t["defended"]]
    f["fork"] = len(serious) >= 2
    # is the moved piece now en prise (attacked by the opponent)?
    attackers = after.attackers(not mover.color, to)
    f["moved_piece_attacked"] = bool(attackers)
    f["moved_piece_defended"] = bool(after.attackers(mover.color, to))
    f["n_replies"] = after.legal_moves.count()
    return f


def facts_for_line(fen, uci_moves):
    """Per-move facts for a line from fen; also material balance for the side to move at the start."""
    board = chess.Board(fen)
    hero = board.turn
    start_bal = material(board, hero) - material(board, not hero)
    out = []
    for i, uci in enumerate(uci_moves):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            out.append({"illegal": uci, "ply": i})
            break
        f = move_facts(board, move)
        f["ply"] = i
        f["hero_move"] = board.turn == hero
        board.push(move)
        f["balance_after"] = material(board, hero) - material(board, not hero) - start_bal
        # sacrifice: hero's moved piece can be taken and the opponent does take it next move
        out.append(f)
    for i, f in enumerate(out[:-1]):
        nxt = out[i + 1]
        if "illegal" in f or "illegal" in nxt:
            continue
        if f["hero_move"] and nxt.get("captures") and nxt["captures"]["square"] == f["to"]:
            gained = f["captures"]["value"] if f.get("captures") else 0
            lost = VAL[chess.PIECE_NAMES.index(f["piece"])] if f["piece"] != "king" else 0
            if f.get("promotes_to"):
                lost = VAL[chess.PIECE_NAMES.index(f["promotes_to"])]
            f["sacrifice"] = lost > gained  # gives up more than it took, on purpose
    return out, board


def start_facts(fen):
    """Salient facts about the start position (for the side to move)."""
    board = chess.Board(fen)
    me, them = board.turn, not board.turn
    facts = {"side": SIDE[me], "in_check": board.is_check(),
             "checks": [board.san(m) for m in board.legal_moves if board.gives_check(m)],
             "captures": [board.san(m) for m in board.legal_moves if board.is_capture(m)]}
    pins, hanging, threatened = [], [], []
    for color in (me, them):
        for sq in chess.SquareSet(board.occupied_co[color] & ~board.kings):
            if board.is_pinned(color, sq):
                pins.append(desc(board, sq) + " is pinned to its king")
    for sq in chess.SquareSet(board.occupied_co[them] & ~board.kings):
        if board.attackers(me, sq) and not board.attackers(them, sq):
            hanging.append(desc(board, sq))
    for sq in chess.SquareSet(board.occupied_co[me] & ~board.kings):
        if board.attackers(them, sq):
            threatened.append(desc(board, sq) + (" (undefended)" if not board.attackers(me, sq) else ""))
    facts.update(pins=pins, hanging=hanging, threatened=threatened,
                 material={"White": material(board, chess.WHITE), "Black": material(board, chess.BLACK)})
    return facts


def move_sentence(f):
    """One factual clause for a move, e.g. 'Rxd4+ (rook d1 to d4) takes White's rook on d4, check'."""
    s = f"{f['side']} {f['san']} ({f['piece']} {f['from']} to {f['to']})"
    bits = []
    if f.get("castles"):
        bits.append(f"castles {f['castles']}")
    if f.get("captures"):
        c = f["captures"]
        bits.append(f"takes the {c['piece']} on {c['square']}" + (" en passant" if c.get("en_passant") else ""))
    if f.get("promotes_to"):
        bits.append(f"promotes to a {f['promotes_to']}")
    if f.get("checkmate"):
        bits.append("CHECKMATE")
    elif f.get("check"):
        kind = "double check" if f.get("double_check") else "discovered check" if f.get("discovered_check") else "check"
        bits.append(f"{kind} (from {', '.join(f['checkers'])})")
    if f.get("fork") and f.get("hero_move", True):
        bits.append("forks " + " and ".join(t["desc"] for t in f["attacks"]
                                             if t["king"] or t["value"] > 0))
    elif f.get("attacks") and not f.get("check"):
        big = [t["desc"] for t in f["attacks"] if t["value"] >= 3 or t["king"]]
        if big:
            bits.append("now attacks " + " and ".join(big))
    if f.get("discovered_attacks") and not f.get("discovered_check"):
        bits.append("uncovers an attack: " + "; ".join(f"{d['by']} now attacks {d['target']}"
                                                     for d in f["discovered_attacks"][:2]))
    if f.get("sacrifice"):
        bits.append("a sacrifice: the piece is taken next move")
    if f.get("moved_piece_attacked") and not f.get("moved_piece_defended") and not f.get("sacrifice"):
        bits.append("the piece can be taken and is undefended")
    if not f.get("checkmate") and f.get("n_replies") == 1:
        bits.append("the opponent has only one legal reply")
    return s + (": " + "; ".join(bits) if bits else "")


def facts_text(fen, uci_moves, themes=None, with_start=True):
    """Readable FACTS block for prompts."""
    board = chess.Board(fen)
    hero = SIDE[board.turn]
    lines = []
    if with_start:
        s = start_facts(fen)
        lines += [f"FACTS (computed exactly by a chess program; all correct). {hero} to move.",
                  f"- Material: White {s['material']['White']}, Black {s['material']['Black']} "
                  "(pawn 1, knight 3, bishop 3, rook 5, queen 9)."]
        if s["in_check"]:
            lines.append(f"- {hero} is in check.")
        lines.append(f"- {hero}'s checks: {', '.join(s['checks']) or 'none'}; captures: {', '.join(s['captures']) or 'none'}.")
        if s["pins"]:
            lines.append("- Pins: " + "; ".join(s["pins"]) + ".")
        if s["hanging"]:
            lines.append("- Undefended opponent pieces that " + hero + " attacks: " + ", ".join(s["hanging"]) + ".")
        if s["threatened"]:
            lines.append(f"- {hero}'s pieces under attack: " + ", ".join(s["threatened"]) + ".")
    facts, end_board = facts_for_line(fen, uci_moves)
    lines.append("- The solution line, move by move:")
    for f in facts:
        if "illegal" in f:
            lines.append(f"  * (move {f['illegal']} is illegal here)")
            break
        lines.append("  * " + move_sentence(f))
    last = facts[-1] if facts else {}
    if last.get("checkmate"):
        lines.append(f"- Result: {hero} checkmates.")
    elif facts and "illegal" not in last:
        bal = last["balance_after"]
        lines.append(f"- Result: material change over the line for {hero}: {bal:+d} "
                     f"({'wins material' if bal > 0 else 'loses material' if bal < 0 else 'material level'}).")
    if themes:
        words = [THEME_WORDS[t] for t in themes if t in THEME_WORDS]
        if words:
            lines.append("- Tactical motifs (from the puzzle database): " + ", ".join(words) + ".")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys
    pz = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "pool_night1.jsonl")]
    for p in pz[:: max(1, len(pz) // 3)][:3]:
        print(p["puzzle_id"], p["rating"], p["themes"])
        print(facts_text(p["fen"], p["full_solution"], p["themes"]))
        print()
