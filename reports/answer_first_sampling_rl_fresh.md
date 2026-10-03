# Overnight 09-29/30: answer-first explanations, sampled decoding, RL on the fresh set

## 1. B's explanations with the answer first

Same 37,543 puzzles, texts and recipe as B; each target is A's exact target (FINAL_LINE + FINAL_MOVE) followed
by the teacher's explanation. Greedy, up to 1,024 new tokens.

**development set:** A 284/500 (56.8%), B 250/500 (50.0%), **B answer-first 265/500 (53.0%)**

- B answer-first vs A: 25 vs 44, p = 0.029; gap -3.8 points (95% interval -7.0 to -0.6)
- B answer-first vs B: 67 vs 52, p = 0.2; gap +3.0 points (95% interval -1.4 to +7.4)
- explanations (right answers): B 170/250 claim-clean, 140/250 with a move flag, 0 with no explanation; B answer-first 176/265 claim-clean, 115/265 with a move flag, 0 with no explanation

**fresh set:** A 280/500 (56.0%), B 233/500 (46.6%), **B answer-first 245/500 (49.0%)**

- B answer-first vs A: 18 vs 53, p = 3.9e-05; gap -7.0 points (95% interval -10.2 to -3.8)
- B answer-first vs B: 69 vs 57, p = 0.33; gap +2.4 points (95% interval -2.0 to +6.8)
- explanations (right answers): B 140/233 claim-clean, 127/233 with a move flag, 0 with no explanation; B answer-first 168/245 claim-clean, 108/245 with a move flag, 0 with no explanation

After pass 1 (development set): A 267/500 (53.4%), B 232/500 (46.4%), B answer-first 265/500 (53.0%); B answer-first vs A: 41 vs 43, p = 0.91; gap -0.4 points (95% interval -4.0 to +3.2)

## 2. Sampled decoding vs greedy

Students sampled once at Qwen3's recommended non-thinking settings (temperature 0.7, top-p 0.8, top-k 20), the
way the teacher is always run; greedy = the main evaluation.

| Student | Development: greedy | Development: sampled | Fresh: greedy | Fresh: sampled |
|---|---|---|---|---|
| A | 284/500 (56.8%) | 270/500 (54.0%) | 280/500 (56.0%) | 250/500 (50.0%) |
| B | 250/500 (50.0%) | 245/500 (49.0%) | 233/500 (46.6%) | 239/500 (47.8%) |
| A + 200k | 319/500 (63.8%) | 311/500 (62.2%) | 314/500 (62.8%) | 296/500 (59.2%) |
| B answer-first | 265/500 (53.0%) | 249/500 (49.8%) | 245/500 (49.0%) | 244/500 (48.8%) |

- A sampled vs teacher medium, development set: 88 vs 83, p = 0.76; gap +1.0 points (95% interval -4.2 to +6.2)
- A sampled vs teacher medium, fresh set: 79 vs 93, p = 0.32; gap -2.8 points (95% interval -8.0 to +2.4)
- A sampled vs greedy, development set: 37 vs 51, p = 0.17; gap -2.8 points (95% interval -6.4 to +1.0)
- A + 200k sampled vs teacher medium, development set: 105 vs 59, p = 0.00041; gap +9.2 points (95% interval +4.2 to +14.2)
- A + 200k sampled vs teacher medium, fresh set: 101 vs 69, p = 0.017; gap +6.4 points (95% interval +1.2 to +11.4)
- A + 200k sampled vs greedy, development set: 34 vs 42, p = 0.42; gap -1.6 points (95% interval -5.0 to +1.8)

## 3. The RL runs on the fresh set (final checkpoints, greedy; evaluated after the protocol)

| Run | Development set | Fresh set |
|---|---|---|
| A, answer-only reward | 267/500 (53.4%) | 235/500 (47.0%) |
| B, answer-only reward | 264/500 (52.8%) | 246/500 (49.2%) |
| B, truth v1 | 271/500 (54.2%) | 248/500 (49.6%) |
| B, strict v2 | 262/500 (52.4%) | 234/500 (46.8%) |
| B, v3 | 246/500 (49.2%) | 221/500 (44.2%) |
| B, v4 | 264/500 (52.8%) | 248/500 (49.6%) |

## Notes (added by hand 2026-10-03, computed from the graded outputs with `teacher_report.pair`)

Each RL-trained B (final checkpoint) against A and against B before RL, fresh set (puzzles only the first model solved
vs only the second; exact McNemar; gap with a 95% paired bootstrap interval):

| Run | vs A | vs B before RL |
|---|---|---|
| B, answer-only | 38 vs 72, p = 0.0015; −6.8 (−10.8 to −2.8) | 71 vs 58, p = 0.29; +2.6 (−1.8 to +7.0) |
| B, truth v1 | 33 vs 65, p = 0.0016; −6.4 (−10.2 to −2.6) | 60 vs 45, p = 0.17; +3.0 (−1.0 to +7.0) |
| B, strict v2 | 34 vs 80, p = 2e-05; −9.2 (−13.2 to −5.2) | 70 vs 69, p = 1; +0.2 (−4.4 to +4.8) |
| B, v3 | 35 vs 94, p = 2e-07; −11.8 (−16.2 to −7.4) | 70 vs 82, p = 0.37; −2.4 (−7.2 to +2.4) |
| B, v4 | 42 vs 74, p = 0.0038; −6.4 (−10.6 to −2.2) | 78 vs 63, p = 0.24; +3.0 (−1.6 to +7.6) |

The answer-first conclusion in §1 was revised on 2026-10-03 with a second seed and the second fresh set: see
`replication_and_fresh_test_2.md` (pooled answer-first comparisons in its notes).

