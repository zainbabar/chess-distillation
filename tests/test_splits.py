"""The README's test-set claims, checked against the published training puzzles (splits/train_puzzles.jsonl.gz)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from check_overlap import load_test, overlap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_students_A_B_and_rl_never_saw_a_test_puzzle():
    test = load_test(ROOT / "test_set.jsonl")
    res = overlap(test, ROOT / "splits/train_puzzles.jsonl.gz", sets={"sft_37.5k_A_B", "rl_3120"})
    for name, r in res.items():
        assert not r["ids"] and not r["games"], name
    assert not res["rl_3120"]["positions"]
    # the one known case: a late pawn-endgame position shared with test puzzle PPhFd, 6-7 moves into its solution,
    # never a position a model answers from
    assert {(tid, ply) for _, tid, ply in res["sft_37.5k_A_B"]["positions"]} == {("PPhFd", 6), ("PPhFd", 7)}
