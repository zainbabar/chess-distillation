# LoRA students on 5.6k and 21.6k puzzles

500 held-out test puzzles, same prompt (P1L) for everyone.

| Model | Training | Correct | Rating (95% CI) | Illegal | Legal line | Full line right |
|---|---|---|---|---|---|---|
| teacher gpt-oss-120b, low effort, 1 attempt | — | **222/500 (44.4%)** | 1416 (1349–1482) | 5 | 245 | 57 |
| pilot: answer only (1.9k) | 1,892 ex × 3.0 ep (37.0 min) | **242/500 (48.4%)** | 1474 (1408–1539) | 0 | 179 | 61 |
| pilot: code-built explanation (1.9k) | 1,892 ex × 3.0 ep (43.5 min) | **247/500 (49.4%)** | 1488 (1427–1551) | 0 | 127 | 54 |
| pilot: LLM explanation (1.9k) | 1,892 ex × 3.0 ep (47.8 min) | **206/500 (41.2%)** | 1370 (1303–1433) | 12 | 96 | 51 |
| night 2: answer only | 5,645 ex × 1.0 ep (36.2 min) | **248/500 (49.6%)** | 1491 (1426–1555) | 0 | 176 | 63 |
| night 2: LLM explanation | 5,645 ex × 1.0 ep (45.7 min) | **217/500 (43.4%)** | 1402 (1337–1468) | 12 | 98 | 48 |
| night 2: answer + board tracking | 11,290 ex × 1.0 ep (49.4 min) | **241/500 (48.2%)** | 1471 (1408–1530) | 0 | 201 | 59 |
| night 2: code-built explanation | 5,645 ex × 1.0 ep (41.6 min) | **232/500 (46.4%)** | 1445 (1384–1509) | 1 | 114 | 55 |
| night 2: answer only, scaled | 21,645 ex × 1.0 ep (136.4 min) | **256/500 (51.2%)** | 1514 (1449–1575) | 0 | 279 | 74 |

Paired comparisons (exact McNemar on the puzzles where the two disagree):

- night 2: answer only vs night 2: LLM explanation: only first right 80, only second right 49, p = 0.008
- night 2: answer only vs night 2: answer + board tracking: only first right 29, only second right 22, p = 0.401
- night 2: answer only vs night 2: answer only, scaled: only first right 28, only second right 36, p = 0.382
- night 2: answer only vs night 2: code-built explanation: only first right 76, only second right 60, p = 0.198
- pilot: code-built explanation (1.9k) vs night 2: code-built explanation: only first right 56, only second right 41, p = 0.155
- pilot: answer only (1.9k) vs night 2: answer only: only first right 42, only second right 48, p = 0.598
- teacher gpt-oss-120b, low effort, 1 attempt vs night 2: answer only: only first right 67, only second right 93, p = 0.048
- teacher gpt-oss-120b, low effort, 1 attempt vs night 2: answer only, scaled: only first right 68, only second right 102, p = 0.011
- pilot: LLM explanation (1.9k) vs night 2: LLM explanation: only first right 51, only second right 62, p = 0.347

Held-out board-tracking test (200 tasks from unseen puzzles):

- night2_answer, board tasks: 0/95 exactly right
- night2_answer, square tasks: 0/105 exactly right
- night2_aux, board tasks: 1/95 exactly right
- night2_aux, square tasks: 53/105 exactly right
