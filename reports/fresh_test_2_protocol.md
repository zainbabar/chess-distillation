# Second fresh test: protocol, fixed before the run

A second one-time evaluation on 500 more puzzles no model or experiment has seen, so the untouched sample doubles to
1,000 puzzles (with the first fresh test, `reports/fresh_test_protocol.md`). Written on 2026-10-01, before any model
answered these puzzles; its git commit time is the timestamp. The models are evaluated on this set only after the
replication runs below finish (not before 2026-10-02).

## The puzzles

- `fresh_test_set_2.jsonl` (sha256 `a86bd35a3e6588cf7fffe5bc28299b1784d03d779614e8f6e20cd88d6e7f56f7`), built by
  `src/build_fresh_test_set.py --seed 20261001 --out fresh_test_set_2.jsonl`: same snapshot and recipe as before (7 bands
  800-2200, 72/72/72/71/71/71/71, rating deviation ≤ 100).
- Exclusion: puzzle id, source game and every solution position of all 282,302 puzzles found in this project's files,
  which now include the first fresh set. 3 candidates were dropped.
- Checked with `experiments/check_overlap.py --test fresh_test_set_2.jsonl` and directly against the development set and
  the first fresh set: no shared ids, games or positions.
- The committed file is the set. Re-running the build command later gives a different set, because the exclusion then
  also covers these 500 puzzles.

## What is evaluated (all frozen; same prompt, grader and settings as before)

Students, greedy, thinking off, up to 1,024 new tokens: A (`ckpt/path_A`), A with seed 1 (`ckpt/path_A_seed1`), A + 200k
(`ckpt/path_A200k`), A + 200k replicate (`ckpt/path_A200k_seed1`, if it finishes), B (`ckpt/path_B`), B with seed 1
(`ckpt/path_B_seed1`), B answer-first (`ckpt/path_Baf`) and its replicate (`ckpt/path_Baf_seed1`, if it finishes), B + RL
v4 (`ckpt/rlT4_B`) and its replicate (`ckpt/rlT4_B_seed1`, if it finishes), and the untrained Qwen3-1.7B (revision
`70d244cc`). The teacher gpt-oss-120b at low effort (up to 8,192 tokens) and medium effort (up to 32,768), one attempt
each, default sampling. Stockfish 16 (1 thread, 0.1 s) and the random-legal-move expectation as references.

## How results are reported

- Per model: first move right (the main score), full line right, puzzle rating with a 95% bootstrap interval.
- **Primary comparisons, fixed here** (exact McNemar, Holm correction over these four; gaps in points with a 95% paired
  bootstrap interval): A vs B; A vs the teacher at low effort; A vs the teacher at medium effort; **A + 200k vs the
  teacher at medium effort**. Each is reported on this set and **pooled over both fresh sets (1,000 puzzles)**; the
  Holm correction is applied separately within each of these two scopes.
- The pooled results use the original one-attempt teacher runs from the first fresh test (`fresh_low`, `fresh_med`),
  not the extra teacher samples taken on that set afterwards.
- Expectations from the earlier sets: A ahead of B; A ahead of the low-effort teacher; A and the medium-effort teacher
  within a few points; A + 200k ahead of the medium-effort teacher.
- Every result is reported, whichever way it comes out. Everything else is secondary.

## Rules

- One run. A crash or server failure is resumed (answers already written are kept); nothing is re-run because of its
  result. No change to prompts, grading, settings or checkpoints after this file is committed.
