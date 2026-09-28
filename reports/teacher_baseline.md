# The teacher at low vs medium effort, and the students against both

500 held-out test puzzles, the same prompt for every model (board description + "calculate the forcing line").
Teacher: gpt-oss-120b, one attempt per puzzle at its default sampling settings, max 32,768 tokens. Students:
Qwen3-1.7B, full fine-tune on 37,543 puzzles, greedy, max 1,024 new tokens.

| Model | Solved | Puzzle rating (95% CI) | Illegal | No readable answer | Cut off | Legal line | Full line right | Output tokens (mean / median / max) |
|---|---|---|---|---|---|---|---|---|
| teacher gpt-oss-120b, low effort | **222/500 (44.4%)** | 1416 (1349–1482) | 5 | 0 | 0 | 245 | 57 | 1,199 / 1,122 / 2,963 |
| teacher gpt-oss-120b, medium effort | **265/500 (53.0%)** | 1540 (1481–1596) | 11 | 2 | 0 | 355 | 96 | 8,427 / 8,490 / 19,233 |
| student A: answers only (1.7B) | **284/500 (56.8%)** | 1594 (1530–1660) | 0 | 0 | 0 | 317 | 90 | 26 / 24 / 57 |
| student B: teacher explanations (1.7B) | **250/500 (50.0%)** | 1497 (1428–1567) | 1 | 0 | 0 | 131 | 66 | 179 / 179 / 270 |

Output tokens for the teacher include its hidden reasoning.

Paired comparisons (exact McNemar; puzzles only the first solved vs only the second solved):

- medium vs low effort: 96 vs 53, p = 0.00054
- student A vs teacher, low effort: 111 vs 49, p = 1.1e-06
- student A vs teacher, medium effort: 90 vs 71, p = 0.16
- student B vs teacher, low effort: 100 vs 72, p = 0.039
- student B vs teacher, medium effort: 79 vs 94, p = 0.29
- student A vs student B: 77 vs 43, p = 0.0024

Solved by rating band (of 72 / 72 / 72 / 71 / 71 / 71 / 71):

| Model | 800-1000 | 1000-1200 | 1200-1400 | 1400-1600 | 1600-1800 | 1800-2000 | 2000-2200 |
|---|---|---|---|---|---|---|---|
| teacher gpt-oss-120b, low effort | 49 | 44 | 39 | 30 | 20 | 19 | 21 |
| teacher gpt-oss-120b, medium effort | 59 | 57 | 47 | 36 | 26 | 21 | 19 |
| student A: answers only (1.7B) | 63 | 56 | 45 | 32 | 39 | 28 | 21 |
| student B: teacher explanations (1.7B) | 54 | 47 | 37 | 31 | 39 | 21 | 21 |
