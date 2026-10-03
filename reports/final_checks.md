# Final machine checks (10-01): untrained baseline, teacher resampled, whole-line soundness

## 1. Untrained Qwen3-1.7B, development set (greedy, thinking off, 1,024 tokens: the students' settings)

6/500 (1.2%); unreadable 224, cut off 189, illegal 11, wrong 70. (Fresh set: 6/500, 1.2%.)

## 2. The teacher sampled again (fresh set)

- Teacher, medium effort: samples 52.8%, 52.4%; mean 52.6 ± 0.3
- Teacher, low effort: samples 40.4%, 41.8%, 42.8%; mean 41.7 ± 1.2

Each student (greedy) against each teacher sample (exact McNemar; gap with a 95% paired bootstrap interval):

- student A + 200k vs teacher medium, sample 1: 111 vs 61, p = 0.00017; gap +10.0 points (95% interval +5.0 to +15.0)
- student A + 200k vs teacher medium, sample 2: 114 vs 62, p = 0.00011; gap +10.4 points (95% interval +5.2 to +15.6)
- student A vs teacher medium, sample 1: 88 vs 72, p = 0.24; gap +3.2 points (95% interval -1.8 to +8.2)
- student A vs teacher medium, sample 2: 94 vs 76, p = 0.19; gap +3.6 points (95% interval -1.6 to +8.8)
- student A, seed 1 vs teacher medium, sample 1: 93 vs 79, p = 0.32; gap +2.8 points (95% interval -2.4 to +8.0)
- student A, seed 1 vs teacher medium, sample 2: 97 vs 81, p = 0.26; gap +3.2 points (95% interval -2.0 to +8.6)
- student B vs teacher medium, sample 1: 68 vs 99, p = 0.02; gap -6.2 points (95% interval -11.4 to -1.2)
- student B vs teacher medium, sample 2: 69 vs 98, p = 0.03; gap -5.8 points (95% interval -11.0 to -0.6)

## 3. Are the written lines sound? (Stockfish, depth 18)

Multi-move puzzles where the model's first move is right. Sound full line: legal, every solver move as good as
Stockfish's best (≤ 5 win-% points), every opponent move a real defense, and as long as the solution or ending in
mate. Exact match = the old "full line right" (identical to the Lichess solution). The Lichess rows are the
method check.

| Model | Set | First move right (multi-move) | Sound full line | Exact match | First problem: illegal / weak solver move / not a real defense / too short |
|---|---|---|---|---|---|
| Lichess solution (method check) | development | 449 | 441 (98%) | 449 | 0 / 2 / 6 / 0 |
| Lichess solution (method check) | fresh | 449 | 442 (98%) | 449 | 0 / 0 / 7 / 0 |
| teacher, low effort | development | 176 | 12 (7%) | 11 | 89 / 19 / 24 / 32 |
| teacher, low effort | fresh | 157 | 9 (6%) | 6 | 87 / 17 / 15 / 29 |
| teacher, medium effort | development | 214 | 65 (30%) | 45 | 36 / 31 / 44 / 38 |
| teacher, medium effort | fresh | 215 | 73 (34%) | 64 | 40 / 37 / 33 / 32 |
| student A | development | 233 | 49 (21%) | 38 | 69 / 52 / 43 / 20 |
| student A | fresh | 229 | 59 (26%) | 45 | 78 / 48 / 30 / 14 |
| student A, seed 1 | development | 230 | 57 (25%) | 47 | 70 / 53 / 34 / 16 |
| student A, seed 1 | fresh | 227 | 55 (24%) | 39 | 75 / 50 / 35 / 12 |
| student A + 200k | development | 268 | 90 (34%) | 75 | 51 / 57 / 48 / 22 |
| student A + 200k | fresh | 263 | 88 (33%) | 66 | 59 / 60 / 39 / 17 |
| student B | development | 199 | 16 (8%) | 15 | 123 / 29 / 21 / 10 |
| student B | fresh | 182 | 15 (8%) | 16 | 119 / 27 / 17 / 4 |
| student B, seed 1 | development | 194 | 15 (8%) | 11 | 122 / 29 / 23 / 5 |
| student B, seed 1 | fresh | 184 | 11 (6%) | 10 | 118 / 36 / 15 / 4 |
| B answer-first | development | 214 | 38 (18%) | 31 | 81 / 48 / 34 / 13 |
| B answer-first | fresh | 194 | 33 (17%) | 28 | 73 / 48 / 30 / 10 |
| B + RL v4 | development | 213 | 3 (1%) | 0 | 85 / 58 / 61 / 6 |
| B + RL v4 | fresh | 197 | 7 (4%) | 0 | 69 / 71 / 47 / 3 |

