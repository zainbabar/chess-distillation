# Fresh test: protocol, fixed before the run

The 500 puzzles in `test_set.jsonl` were held out from training but used while developing the experiments (the prompt
was chosen on some of them and the RL rewards were redesigned after reading outputs on them). This is a one-time
evaluation on 500 puzzles no model and no experiment has seen, with everything below fixed before any model answers
them. This file was written on 2026-09-29, before the run; its git commit time is the timestamp.

## The puzzles

- `fresh_test_set.jsonl` (sha256 `312e1e56df42e5b8fe7af3571bfe0861b0e7c49bbe96b0227926287061315eb2`), built by
  `src/build_fresh_test_set.py` from the same Lichess snapshot as everything else (see `REPRODUCING.md`), with the same
  recipe as the original test set: 7 rating bands from 800 to 2200 (72/72/72/71/71/71/71), rating deviation ≤ 100,
  and a new seed (20260929).
- Exclusion: a candidate was dropped if its puzzle id, its source game, or any position along its solution matched any
  of the 281,802 puzzles that appear anywhere in this project's files (every pool, training set, test set and output),
  not only the training sets. 2 candidates were dropped.
- Checked independently with `experiments/check_overlap.py --test fresh_test_set.jsonl`: no shared ids, games or
  positions with any training set (including the 200k puzzles of A+200k) or with the original test set.

## What is evaluated (all frozen)

Same prompt (P1L), grader and settings as the main evaluation:

| Model | Setting |
|---|---|
| Student A (answers only), `ckpt/path_A` | greedy, thinking off, up to 1,024 new tokens |
| Student B (teacher explanations), `ckpt/path_B` | same |
| Student A + 200k more puzzles, `ckpt/path_A200k` | same |
| Student B after RL with reward v4, `ckpt/rlT4_B` (final checkpoint), if that run finishes | same |
| Untrained Qwen3-1.7B (revision `70d244cc`) | same (greedy, thinking off, 1,024 tokens) |
| gpt-oss-120b, low effort | default sampling, up to 8,192 tokens, one attempt |
| gpt-oss-120b, medium effort | default sampling, up to 32,768 tokens, one attempt |
| Random legal move | expected score (exact, no sampling) |
| Stockfish 16 | 1 thread, 0.1 s per puzzle, first move only |

## How results are reported

- Per model: first move right (the main score; in mate-in-one puzzles any mate counts), full solution line right,
  puzzle rating with a 95% bootstrap interval, output tokens.
- Primary comparisons, fixed here: **A vs B**, **A vs the teacher at low effort**, **A vs the teacher at medium
  effort**. Exact McNemar tests with a Holm correction over these three, and the gap in points with a 95% paired
  bootstrap interval (10,000 resamples, seed 0). Everything else is secondary.
- Expectations from the development set: A ahead of B; A ahead of the low-effort teacher; A and the medium-effort
  teacher within a few points of each other.
- Every result is reported, whichever way it comes out. The original 500 are then called the development set.

## Rules

- One run. A crash or server failure is resumed (answers already written are kept); nothing is re-run because of its
  result.
- No change to prompts, grading, settings or checkpoints after this file is committed.
