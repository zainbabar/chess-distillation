# The teacher at low vs medium effort, and the students against both

500 held-out test puzzles, the same prompt for every model (board description + "calculate the forcing line").
Teacher: gpt-oss-120b, one attempt per puzzle at its default sampling settings, max 8,192 tokens at low effort and
32,768 at medium. Students:
Qwen3-1.7B, full fine-tune on 37,543 puzzles, greedy, max 1,024 new tokens.

| Model | Solved | Puzzle rating (95% CI) | Illegal | No readable answer | Cut off | Legal line | Full line right | Output tokens (mean / median / max) |
|---|---|---|---|---|---|---|---|---|
| teacher gpt-oss-120b, low effort | **222/500 (44.4%)** | 1416 (1349–1482) | 5 | 0 | 0 | 245 | 57 | 1,199 / 1,122 / 2,963 |
| teacher gpt-oss-120b, medium effort | **265/500 (53.0%)** | 1540 (1481–1596) | 11 | 2 | 0 | 355 | 96 | 8,427 / 8,490 / 19,233 |
| student A: answers only (1.7B) | **284/500 (56.8%)** | 1594 (1530–1660) | 0 | 0 | 0 | 317 | 90 | 26 / 24 / 57 |
| student B: teacher explanations (1.7B) | **250/500 (50.0%)** | 1497 (1428–1567) | 1 | 0 | 0 | 131 | 66 | 179 / 179 / 270 |

Output tokens for the teacher include its hidden reasoning.

Paired comparisons (exact McNemar: puzzles only the first solved vs only the second solved; the gap is the first's
score minus the second's, in points, with a 95% paired bootstrap interval):

- medium vs low effort: 96 vs 53, p = 0.00054; gap +8.6 points (95% interval +4.0 to +13.4)
- student A vs teacher, low effort: 111 vs 49, p = 1.1e-06; gap +12.4 points (95% interval +7.6 to +17.4)
- student A vs teacher, medium effort: 90 vs 71, p = 0.16; gap +3.8 points (95% interval -1.2 to +9.0)
- student B vs teacher, low effort: 100 vs 72, p = 0.039; gap +5.6 points (95% interval +0.6 to +10.8)
- student B vs teacher, medium effort: 79 vs 94, p = 0.29; gap -3.0 points (95% interval -8.2 to +2.2)
- student A vs student B: 77 vs 43, p = 0.0024; gap +6.8 points (95% interval +2.6 to +11.0)

Solved by rating band (of 72 / 72 / 72 / 71 / 71 / 71 / 71):

| Model | 800-1000 | 1000-1200 | 1200-1400 | 1400-1600 | 1600-1800 | 1800-2000 | 2000-2200 |
|---|---|---|---|---|---|---|---|
| teacher gpt-oss-120b, low effort | 49 | 44 | 39 | 30 | 20 | 19 | 21 |
| teacher gpt-oss-120b, medium effort | 59 | 57 | 47 | 36 | 26 | 21 | 19 |
| student A: answers only (1.7B) | 63 | 56 | 45 | 32 | 39 | 28 | 21 |
| student B: teacher explanations (1.7B) | 54 | 47 | 37 | 31 | 39 | 21 | 21 |

## How the prompt changes the teacher

gpt-oss-120b at low effort, one attempt per puzzle, max 8,192 tokens, the same 500 puzzles; each row adds to the
prompt above it. "Illegal" = the answer is not a legal move in the position.

| Teacher prompt | Solved | Puzzle rating (95% CI) | Illegal | No readable answer | Output tokens (mean) |
|---|---|---|---|---|---|
| position only (FEN + side to move) | **121/500 (24.2%)** | 1114 (1043–1178) | 111 (22.2%) | 1 | 1,440 |
| + list of legal moves | **156/500 (31.2%)** | 1223 (1158–1286) | 44 (8.8%) | 0 | 1,559 |
| + board description + "calculate the forcing line" (P1L) | **222/500 (44.4%)** | 1416 (1349–1482) | 5 (1.0%) | 0 | 1,199 |

Paired comparisons (exact McNemar):

- + legal moves vs position only: 94 vs 59, p = 0.0058; gap +7.0 points (95% interval +2.2 to +11.8)
- P1L vs + legal moves: 129 vs 63, p = 2.2e-06; gap +13.2 points (95% interval +7.8 to +18.4)

The last step changes two things at once (the board description and the request for the full line).

For scale: a random legal move would solve 5.2% of these puzzles (expected value).
