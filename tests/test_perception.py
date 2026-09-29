"""Check perception.py: parse its text back independently and compare every fact with python-chess.

Positions: the 500 pilot puzzles + N random puzzles from the Lichess DB (after the opponent's move),
plus positions picked to contain en passant, promotions, castling and check.
Usage: .venv/bin/python test_perception.py [N]
"""
import csv
import json
import random
import re
import sys

import chess

from perception import perception_block

NAMES = {chess.piece_name(pt): pt for pt in chess.PIECE_TYPES}


def section(text, header):
    for line in text.splitlines():
        if line.startswith(header):
            return line[len(header):].strip()
    raise AssertionError(f"missing line {header!r}")


def moves_in(s):
    """UCI moves in brackets, e.g. 'Qxe5 (c3e5), Ng6+ (h4g6)'"""
    return set(re.findall(r"\(([a-h][1-8][a-h][1-8][qrbn]?)\)", s))


def parse_pieces(s):
    """'King g1; Queens d1, d8' -> {square: piece_type}"""
    out = {}
    for part in s.split("; "):
        name, squares = part.split(" ", 1)
        pt = NAMES[name.lower().rstrip("s")]
        for sq in squares.split(", "):
            out[chess.parse_square(sq)] = pt
    return out


def check(board, level):
    text = perception_block(board, level)
    me = board.turn

    # 1. diagram round-trip
    rows = [l for l in text.splitlines() if re.match(r"^[1-8] \| ", l)]
    assert len(rows) == 8
    fen_rows = []
    for row in rows:
        cells, run, s = row[4:].split(" "), 0, ""
        for c in cells:
            if c == ".":
                run += 1
            else:
                s += (str(run) if run else "") + c
                run = 0
        fen_rows.append(s + (str(run) if run else ""))
    assert "/".join(fen_rows) == board.board_fen(), "diagram mismatch"

    # 2. piece lists
    for header, color in (("Your pieces", me), ("Opponent's pieces", not me)):
        line = next(l for l in text.splitlines() if l.startswith(header))
        got = parse_pieces(line.split("): ", 1)[1])
        want = {sq: p.piece_type for sq, p in board.piece_map().items() if p.color == color}
        assert got == want, f"{header} mismatch"

    # 3. legal moves (UCI in brackets on the grouped lines) + SAN matches
    move_lines = text.split("standard notation with UCI in brackets:\n", 1)[1].splitlines()
    listed = set()
    for line in move_lines:
        for san, uci in re.findall(r"(\S+) \(([a-h][1-8][a-h][1-8][qrbn]?)\)", line):
            assert board.san(chess.Move.from_uci(uci)) == san, f"SAN mismatch {san} {uci}"
            listed.add(uci)
    assert listed == {m.uci() for m in board.legal_moves}, "legal move set mismatch"

    if level >= 2:
        legal = list(board.legal_moves)
        assert ("IN CHECK" in section(text, "- Your king")) == board.is_check()
        assert moves_in(section(text, "- Checks you can give:")) == {m.uci() for m in legal if board.gives_check(m)}
        assert moves_in(section(text, "- Captures you can make:")) == {m.uci() for m in legal if board.is_capture(m)}
        for header, color in (("- Your pieces under attack:", me), ("- Opponent's pieces under attack:", not me)):
            s = section(text, header)
            got = set() if s == "none" else {
                chess.parse_square(m) for m in re.findall(r"(?:^|; )\w+ ([a-h][1-8]) \(attacked by", s)}
            want = {sq for sq in chess.SquareSet(board.occupied_co[color] & ~board.kings)
                    if board.attackers(not color, sq)}
            assert got == want, f"{header} mismatch"
            for sq_name, defended in re.findall(r"\w+ ([a-h][1-8]) \(attacked by [^;]*; defended (\d+) time", s):
                sq = chess.parse_square(sq_name)
                assert int(defended) == len(board.attackers(color, sq)), "defender count mismatch"
        s = section(text, "- Pinned pieces:")
        got = set() if s == "none" else {chess.parse_square(x) for x in re.findall(r"\w+ ([a-h][1-8]) is pinned", s)}
        want = {sq for c in (True, False) for sq in chess.SquareSet(board.occupied_co[c] & ~board.kings)
                if board.is_pinned(c, sq)}
        assert got == want, "pins mismatch"
    return text


def main():
    n_random = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    boards = [chess.Board(json.loads(l)["fen"]) for l in open("test_set.jsonl")]
    rng = random.Random(0)
    with open("data/lichess_db_puzzle.csv", newline="") as f:
        rows = [r for i, r in enumerate(csv.DictReader(f)) if i % 1000 == 0]  # ~6k spread rows
    for r in rng.sample(rows, min(n_random, len(rows))):
        b = chess.Board(r["FEN"])
        b.push_uci(r["Moves"].split()[0])
        boards.append(b)
    edge = {"en passant": 0, "promotion": 0, "castling": 0, "in check": 0}
    for b in boards:
        for level in (1, 2):
            check(b, level)
        legal = list(b.legal_moves)
        edge["en passant"] += any(b.is_en_passant(m) for m in legal)
        edge["promotion"] += any(m.promotion for m in legal)
        edge["castling"] += any(b.is_castling(m) for m in legal)
        edge["in check"] += b.is_check()
    print(f"OK: {len(boards)} positions x 2 levels, every fact matches python-chess. "
          f"Edge cases covered: {edge}")


if __name__ == "__main__":
    main()
