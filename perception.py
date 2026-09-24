"""Board-perception text for chess prompts, computed exactly with python-chess.

The teacher misreads positions given only a FEN (illegal moves, moving the opponent's pieces, missing
simple captures). These blocks spell the position out so the model doesn't have to decode it:

  level 1 ("P1", see the board): board diagram, piece lists by side, legal moves grouped by piece in
          standard notation (SAN) with UCI in brackets
  level 2 ("P2", see the tactics): level 1 + check status, available checks and captures, attacked
          pieces with defender counts, pins

Everything here is derived from python-chess, so it is always correct; test_perception.py parses the
text back and checks it against python-chess.
"""
import chess

PIECE_ORDER = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
SIDE = {chess.WHITE: "White", chess.BLACK: "Black"}


def _name(pt):
    return chess.piece_name(pt)


def _piece_at(board, sq):
    """e.g. 'queen h5'"""
    return f"{_name(board.piece_type_at(sq))} {chess.square_name(sq)}"


def board_diagram(board):
    rows = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            p = board.piece_at(chess.square(file, rank))
            cells.append(p.symbol() if p else ".")
        rows.append(f"{rank + 1} | " + " ".join(cells))
    rows.append("    " + "-" * 15)
    rows.append("    a b c d e f g h")
    return "\n".join(rows)


def piece_list(board, color):
    parts = []
    for pt in PIECE_ORDER:
        squares = [chess.square_name(s) for s in sorted(board.pieces(pt, color))]
        if squares:
            label = _name(pt).capitalize() + ("s" if len(squares) > 1 else "")
            parts.append(f"{label} {', '.join(squares)}")
    return "; ".join(parts)


def grouped_legal_moves(board):
    """One line per moving piece: 'Queen c3: Qa5 (c3a5), Qxe5 (c3e5), ...'"""
    by_from = {}
    for m in board.legal_moves:
        by_from.setdefault(m.from_square, []).append(m)
    lines = []
    order = sorted(by_from, key=lambda s: (PIECE_ORDER.index(board.piece_type_at(s)), s))
    for sq in order:
        moves = sorted(by_from[sq], key=lambda m: board.san(m))
        lines.append(f"{_piece_at(board, sq).capitalize()}: " +
                     ", ".join(f"{board.san(m)} ({m.uci()})" for m in moves))
    return lines


def attacked_pieces(board, color):
    """Non-king pieces of `color` attacked by the other side, with attackers and defender count."""
    out = []
    for sq in sorted(chess.SquareSet(board.occupied_co[color] & ~board.kings)):
        attackers = board.attackers(not color, sq)
        if attackers:
            defenders = len(board.attackers(color, sq))
            att = ", ".join(_piece_at(board, a) for a in sorted(attackers))
            flag = " - UNDEFENDED" if defenders == 0 else ""
            out.append(f"{_piece_at(board, sq)} (attacked by {att}; defended {defenders} time"
                       f"{'s' if defenders != 1 else ''}{flag})")
    return out


def pinned_pieces(board):
    out = []
    for color in (board.turn, not board.turn):
        for sq in sorted(chess.SquareSet(board.occupied_co[color] & ~board.kings)):
            if board.is_pinned(color, sq):
                who = "your" if color == board.turn else "opponent's"
                out.append(f"{who} {_piece_at(board, sq)} is pinned to its king")
    return out


def perception_block(board, level):
    me, them = board.turn, not board.turn
    lines = [
        "Board (White pieces uppercase, Black pieces lowercase; rank 8 at the top):",
        board_diagram(board),
        "",
        f"Your pieces ({SIDE[me]}): {piece_list(board, me)}",
        f"Opponent's pieces ({SIDE[them]}): {piece_list(board, them)}",
    ]
    if level >= 2:
        legal = list(board.legal_moves)
        if board.is_check():
            checkers = ", ".join(_piece_at(board, s) for s in sorted(board.checkers()))
            check_line = f"Your king is IN CHECK from {checkers}."
        else:
            check_line = "Your king is not in check."
        checks = [m for m in legal if board.gives_check(m)]
        captures = [m for m in legal if board.is_capture(m)]

        def cap_desc(m):
            if board.is_en_passant(m):
                return f"{board.san(m)} ({m.uci()}) takes pawn en passant"
            return f"{board.san(m)} ({m.uci()}) takes {_name(board.piece_type_at(m.to_square))}"

        lines += [
            "",
            "Tactical facts:",
            f"- {check_line}",
            "- Checks you can give: " + (", ".join(f"{board.san(m)} ({m.uci()})" for m in
                                                   sorted(checks, key=board.san)) or "none"),
            "- Captures you can make: " + (", ".join(cap_desc(m) for m in
                                                    sorted(captures, key=board.san)) or "none"),
            "- Your pieces under attack: " + ("; ".join(attacked_pieces(board, me)) or "none"),
            "- Opponent's pieces under attack: " + ("; ".join(attacked_pieces(board, them)) or "none"),
            "- Pinned pieces: " + ("; ".join(pinned_pieces(board)) or "none"),
        ]
    moves = grouped_legal_moves(board)
    lines += ["", f"Legal moves ({board.legal_moves.count()}), grouped by piece, "
                  "standard notation with UCI in brackets:"] + moves
    return "\n".join(lines)
