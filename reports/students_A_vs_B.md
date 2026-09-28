# Student A (answers only) vs student B (teacher explanations) — THE PATH step 2

Qwen3-1.7B, full fine-tune, 2 passes, lr 1e-5, the same 37,543 training puzzles and prompt; only the target text
differs. 500 held-out test puzzles, greedy, max 1,024 new tokens.

| Model | Correct | Rating (95% CI) | Illegal | Parse fail / truncated | Legal line | Full line right | Mean output tokens |
|---|---|---|---|---|---|---|---|
| A: answers only, 2 passes | **284/500 (56.8%)** | 1594 (1530–1660) | 0 | 0 / 0 | 317 | 90 | 26 |
| A after pass 1 | **267/500 (53.4%)** | 1545 (1477–1604) | 0 | 0 / 0 | 288 | 80 | 26 |
| B: teacher explanations, 2 passes | **250/500 (50.0%)** | 1497 (1428–1567) | 1 | 0 / 0 | 131 | 66 | 179 |
| B after pass 1 | **232/500 (46.4%)** | 1445 (1383–1500) | 3 | 0 / 0 | 153 | 64 | 176 |
| teacher gpt-oss-120b (low, 1 attempt) | **222/500 (44.4%)** | 1416 (1349–1482) | 5 | 0 / 0 | 245 | 57 | 1199 |
| night 2: answers only, LoRA 21.6k × 1 | **256/500 (51.2%)** | 1514 (1449–1575) | 0 | 0 / 0 | 279 | 74 | 26 |

Paired comparisons (exact McNemar; only-first-right vs only-second-right):

- B: teacher explanations, 2 passes vs A: answers only, 2 passes: 43 vs 77, p = 0.0024
- B after pass 1 vs A after pass 1: 42 vs 77, p = 0.0017
- B: teacher explanations, 2 passes vs B after pass 1: 52 vs 34, p = 0.066
- B: teacher explanations, 2 passes vs teacher gpt-oss-120b (low, 1 attempt): 100 vs 72, p = 0.039
- B: teacher explanations, 2 passes vs night 2: answers only, LoRA 21.6k × 1: 59 vs 65, p = 0.65
- A: answers only, 2 passes vs teacher gpt-oss-120b (low, 1 attempt): 111 vs 49, p = 1.1e-06

Correct by rating band:

| Model | 800-1000 | 1000-1200 | 1200-1400 | 1400-1600 | 1600-1800 | 1800-2000 | 2000-2200 |
|---|---|---|---|---|---|---|---|
| A: answers only, 2 passes | 63/72 | 56/72 | 45/72 | 32/71 | 39/71 | 28/71 | 21/71 |
| A after pass 1 | 62/72 | 49/72 | 47/72 | 29/71 | 35/71 | 24/71 | 21/71 |
| B: teacher explanations, 2 passes | 54/72 | 47/72 | 37/72 | 31/71 | 39/71 | 21/71 | 21/71 |
| B after pass 1 | 58/72 | 45/72 | 40/72 | 27/71 | 28/71 | 17/71 | 17/71 |
| teacher gpt-oss-120b (low, 1 attempt) | 49/72 | 44/72 | 39/72 | 30/71 | 20/71 | 19/71 | 21/71 |
| night 2: answers only, LoRA 21.6k × 1 | 61/72 | 50/72 | 46/72 | 27/71 | 29/71 | 25/71 | 18/71 |

Is B's own reasoning true? (claim checker on the text before FINAL_LINE; hard errors = piece not on that
square, false "wins the X", false mate; move flag = mentions a move that is illegal/mis-annotated where written)

- B: teacher explanations, 2 passes, right answers (250): wrote an explanation 250; claim-clean 170 (68%); move flag 140 (56%)
- B: teacher explanations, 2 passes, wrong answers (250): wrote an explanation 250; claim-clean 74 (30%); move flag 193 (77%)
- B after pass 1, right answers (232): wrote an explanation 232; claim-clean 152 (66%); move flag 121 (52%)
- B after pass 1, wrong answers (268): wrote an explanation 268; claim-clean 83 (31%); move flag 199 (74%)

Training:

- A: answers only, 2 passes: 37,543 examples, 27,261,198 tokens/pass (1,060,901 target), 344.1 min, final loss 0.256
- B: teacher explanations, 2 passes: 37,543 examples, 33,359,989 tokens/pass (7,159,692 target), 430.9 min, final loss 0.545
