# Student A (answers only, full fine-tune) — THE PATH step 2

500 held-out test puzzles, same P1L prompt for every model; greedy decoding for students.

| Model | Correct | Rating (95% CI) | Illegal | Parse fail / truncated | Legal line | Full line right |
|---|---|---|---|---|---|---|
| A: answers only, full FT, 37.5k × 2 passes | **284/500 (56.8%)** | 1594 (1530–1660) | 0 | 0 / 0 | 317 | 90 |
| A after pass 1 | **267/500 (53.4%)** | 1545 (1477–1604) | 0 | 0 / 0 | 288 | 80 |
| teacher gpt-oss-120b (low, 1 attempt) | **222/500 (44.4%)** | 1416 (1349–1482) | 5 | 0 / 0 | 245 | 57 |
| night 2: answers only, LoRA, 21.6k × 1 | **256/500 (51.2%)** | 1514 (1449–1575) | 0 | 0 / 0 | 279 | 74 |
| night 2: answers only, LoRA, 5.6k × 1 | **248/500 (49.6%)** | 1491 (1426–1555) | 0 | 0 / 0 | 176 | 63 |

Paired comparisons (exact McNemar; only-first-right vs only-second-right):

- A: answers only, full FT, 37.5k × 2 passes vs teacher gpt-oss-120b (low, 1 attempt): 111 vs 49, p = 1.1e-06 (500 puzzles)
- A: answers only, full FT, 37.5k × 2 passes vs night 2: answers only, LoRA, 21.6k × 1: 56 vs 28, p = 0.003 (500 puzzles)
- A after pass 1 vs teacher gpt-oss-120b (low, 1 attempt): 98 vs 53, p = 0.00031 (500 puzzles)
- A after pass 1 vs A: answers only, full FT, 37.5k × 2 passes: 27 vs 44, p = 0.057 (500 puzzles)
- A after pass 1 vs night 2: answers only, LoRA, 21.6k × 1: 43 vs 32, p = 0.25 (500 puzzles)
- A: answers only, full FT, 37.5k × 2 passes vs night 2: answers only, LoRA, 5.6k × 1: 69 vs 33, p = 0.00047 (500 puzzles)

Correct by rating band:

| Model | 800-1000 | 1000-1200 | 1200-1400 | 1400-1600 | 1600-1800 | 1800-2000 | 2000-2200 |
|---|---|---|---|---|---|---|---|
| A: answers only, full FT, 37.5k × 2 passes | 63/72 | 56/72 | 45/72 | 32/71 | 39/71 | 28/71 | 21/71 |
| A after pass 1 | 62/72 | 49/72 | 47/72 | 29/71 | 35/71 | 24/71 | 21/71 |
| teacher gpt-oss-120b (low, 1 attempt) | 49/72 | 44/72 | 39/72 | 30/71 | 20/71 | 19/71 | 21/71 |
| night 2: answers only, LoRA, 21.6k × 1 | 61/72 | 50/72 | 46/72 | 27/71 | 29/71 | 25/71 | 18/71 |
| night 2: answers only, LoRA, 5.6k × 1 | 56/72 | 49/72 | 44/72 | 27/71 | 25/71 | 26/71 | 21/71 |

Training:

- A: answers only, full FT, 37.5k × 2 passes: full FT, 37,543 examples × 2 passes, lr 1e-05, 344.1 min (~2,641 tok/s), final loss 0.256
- night 2: answers only, LoRA, 21.6k × 1: LoRA r64, 21,645 examples × 1 passes, lr 0.0002, 136.4 min (~1,916 tok/s), final loss 0.333
- night 2: answers only, LoRA, 5.6k × 1: LoRA r64, 5,645 examples × 1 passes, lr 0.0002, 36.2 min (~1,889 tok/s), final loss 0.406
