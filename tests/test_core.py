"""Fast unit tests for the pieces every result depends on (no model, no Stockfish, no Lichess download).

Positions are made up for the tests (never taken from the held-out test set):
- MATE: White mates with Ra8#.
- WIN_QUEEN: White's rook takes the queen, then gives check (a 3-ply "solution").
"""
import chess
import pytest

from analyze_pilot import mcnemar
from claim_check import check
from puzzle_rating import rate
from run_pilot import grade, grade_line
from test_perception import check as perception_roundtrip

MATE = {"fen": "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "correct_move": "a1a8", "full_solution": ["a1a8"],
        "themes": ["mateIn1", "backRankMate"]}
WIN_QUEEN = {"fen": "4k3/8/8/8/8/8/q7/R3K3 w - - 0 1", "correct_move": "a1a2",
             "full_solution": ["a1a2", "e8d7", "a2a7"], "themes": ["crushing"]}


@pytest.mark.parametrize("text, status", [
    ("FINAL_LINE: a1a2 e8d7 a2a7\nFINAL_MOVE: a1a2", "correct"),
    ("FINAL_MOVE: **a1a2**", "correct"),          # markdown around the move is tolerated
    ("FINAL_MOVE: e1d1", "wrong"),                # legal, but not the solution
    ("FINAL_MOVE: a1a8", "illegal"),              # the queen on a2 blocks the file
    ("I think Rxa2 is best.", "parse_fail"),       # no FINAL_MOVE line
])
def test_grade(text, status):
    assert grade(WIN_QUEEN, text)[0] == status


def test_grade_accepts_the_mate():
    assert grade(MATE, "FINAL_MOVE: a1a8")[0] == "correct"


def test_grade_line_full_partial_illegal():
    full = grade_line(WIN_QUEEN, "FINAL_LINE: a1a2 e8d7 a2a7")
    assert full["line_legal"] and full["full_line_correct"] and full["line_match_len"] == 3
    partial = grade_line(WIN_QUEEN, "FINAL_LINE: a1a2 e8e7 a2a7")
    assert partial["line_legal"] and not partial["full_line_correct"] and partial["line_match_len"] == 1
    own_side_twice = grade_line(WIN_QUEEN, "FINAL_LINE: a1a2 e1d1")  # the v3 reward-hacking pattern
    assert own_side_twice["line_legal"] is False


def test_claim_check_accepts_true_claims():
    out = check("The black queen on a2 is undefended, so Rxa2 wins the queen.", WIN_QUEEN)
    assert out["clean"] and out["n_claims"] >= 1


def test_claim_check_flags_a_piece_that_is_not_there():
    out = check("The black queen on b3 is undefended, so the rook takes it.", WIN_QUEEN)
    assert not out["clean"]


def test_claim_check_flags_a_false_mate():
    out = check("Rxa2 is checkmate.", WIN_QUEEN)
    assert not out["clean"]


@pytest.mark.parametrize("fen", [
    chess.STARTING_FEN,
    MATE["fen"],
    WIN_QUEEN["fen"],
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",   # en passant available
    "r3k2r/pppq1ppp/2n2n2/3pp3/1b1PP3/2N2N2/PPPQ1PPP/R3K2R w KQkq - 0 1",  # both sides can castle
    "8/P7/8/8/8/8/5k2/7K w - - 0 1",                                  # promotion
    "rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 3",  # side to move is in check
])
def test_board_description_round_trips(fen):
    board = chess.Board(fen)
    for level in (1, 2):
        perception_roundtrip(board, level)  # raises if any listed fact disagrees with python-chess


def test_mcnemar():
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(90, 71) == pytest.approx(mcnemar(71, 90))
    assert 0.1 < mcnemar(90, 71) < 0.2       # student A vs the medium teacher: p = 0.16
    assert mcnemar(111, 49) < 1e-5           # student A vs the low-effort teacher


def test_rating_fit():
    records = [{"rating": r, "status": "correct" if r < 1500 else "wrong"} for r in range(800, 2200, 5)]
    out = rate(records, n_boot=200)
    assert 1400 < out["rating"] < 1600
    assert out["ci_low"] <= out["rating"] <= out["ci_high"]


# RL reward v4 (src/rewards.py): replays what the student wrote, in order
from rewards import truth4  # noqa: E402

WHY = ("The black queen on a2 is undefended and sits right in front of my rook on the a-file, so Rxa2 simply wins "
       "the queen. After Rxa2 the black king has nothing better than Kd7, and Ra7+ keeps chasing it with checks "
       "while White stays a whole queen ahead in a completely winning ending.")


def test_truth4_pays_for_a_complete_playable_line():
    full, info = truth4(WIN_QUEEN, WHY + "\nFINAL_LINE: a1a2 e8d7 a2a7\nFINAL_MOVE: a1a2")
    first, _ = truth4(WIN_QUEEN, WHY + "\nFINAL_LINE: a1a2\nFINAL_MOVE: a1a2")
    assert info["legal_bonus"] and not info["short"] and not info["claim_err"]
    assert full == pytest.approx(2.25) and first == pytest.approx(1 + 1 / 3 - 0.25)


def test_truth4_catches_the_v3_trick():
    # right move, then another move by White as the "reply": no better than stopping at the first move
    r, info = truth4(WIN_QUEEN, WHY + "\nFINAL_LINE: a1a2 e1d1\nFINAL_MOVE: a1a2")
    assert info["illegal_line"] and not info["legal_bonus"]
    assert r == pytest.approx(truth4(WIN_QUEEN, WHY + "\nFINAL_LINE: a1a2\nFINAL_MOVE: a1a2")[0])
    # writing the broken sequence in the text costs extra; a comma list of alternatives does not
    _, seq = truth4(WIN_QUEEN, WHY + " The forced sequence is Rxa2 Kd1." + "\nFINAL_LINE: a1a2 e8d7 a2a7\nFINAL_MOVE: a1a2")
    _, alts = truth4(WIN_QUEEN, WHY + " Other tries such as Ra8+, Kd2 achieve less." + "\nFINAL_LINE: a1a2 e8d7 a2a7\nFINAL_MOVE: a1a2")
    assert seq["seq_flag"] and not alts["seq_flag"]
