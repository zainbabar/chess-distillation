# First RL test on the Spark — A vs B, before and after RL

GRPO, binary reward (right move 1, else 0; no FINAL_MOVE −0.1), 100 steps × 8 fresh puzzles × 8 samples
(the same 800 puzzles for both, never used in SFT), lr 2e-6, temperature 1.0, no KL, DAPO loss.
Test: 500 held-out puzzles, greedy, max 1,024 new tokens.

| Model | Correct | Rating (95% CI) | Illegal | Legal line | Full line right | Mean output tokens |
|---|---|---|---|---|---|---|
| A (answers only), SFT | **284/500 (56.8%)** | 1594 (1530–1660) | 0 | 317 | 90 | 26 |
| A + RL (100 steps) | **275/500 (55.0%)** | 1568 (1505–1634) | 0 | 281 | 79 | 28 |
| B (teacher explanations), SFT | **250/500 (50.0%)** | 1497 (1428–1567) | 1 | 131 | 66 | 179 |
| B + RL (100 steps) | **264/500 (52.8%)** | 1537 (1476–1598) | 1 | 94 | 63 | 186 |
| teacher gpt-oss-120b (low, 1 attempt) | **222/500 (44.4%)** | 1416 (1349–1482) | 5 | 245 | 57 | 1199 |

Paired comparisons (exact McNemar; only-first-right vs only-second-right):

- A + RL (100 steps) vs A (answers only), SFT: 22 vs 31, p = 0.27
- B + RL (100 steps) vs B (teacher explanations), SFT: 63 vs 49, p = 0.22
- B + RL (100 steps) vs A + RL (100 steps): 32 vs 43, p = 0.25
- B (teacher explanations), SFT vs A (answers only), SFT: 43 vs 77, p = 0.0024
- A + RL (100 steps) vs teacher gpt-oss-120b (low, 1 attempt): 115 vs 62, p = 8.3e-05
- B + RL (100 steps) vs teacher gpt-oss-120b (low, 1 attempt): 103 vs 61, p = 0.0013

Training curve (sample accuracy on fresh puzzles at temperature 1.0 = mean reward, in blocks of 20 steps):

- A: 0.485 → 0.503 → 0.453 → 0.530 → 0.523 (100 steps); mean length 29 → 30 tokens; groups with no signal (all 8 samples equal) 53%
  - 59.3 min for 100 steps (36 s/step), 6,400 samples, 0 parse fails
- B: 0.458 → 0.437 → 0.411 → 0.455 → 0.448 (100 steps); mean length 193 → 201 tokens; groups with no signal (all 8 samples equal) 39%
  - 83.1 min for 100 steps (50 s/step), 6,400 samples, 3 parse fails

Is B's reasoning truer after RL? (claim checker on the text before FINAL_LINE)

- B (teacher explanations), SFT, right answers (250): claim-clean 68%, move flag 56%
- B (teacher explanations), SFT, wrong answers (250): claim-clean 30%, move flag 77%
- B + RL (100 steps), right answers (264): claim-clean 56%, move flag 65%
- B + RL (100 steps), wrong answers (236): claim-clean 29%, move flag 81%
