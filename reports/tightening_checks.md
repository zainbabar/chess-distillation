# Tightening checks (09-30 night): five samples, B's second seed, Stockfish checks

## 1. Five sampled runs per student

Each student sampled 5 times per puzzle set at Qwen3's recommended non-thinking settings (temperature 0.7, top-p
0.8, top-k 20), the way the teacher is run (the teacher: one sample). Greedy = the main evaluation.

| Student | Set | Greedy | Sampled: mean ± sd (min–max), n | Gap to teacher medium per sample (points; * = p < 0.05) |
|---|---|---|---|---|
| A | development | 56.8% | 52.2 ± 1.1 (51.2–54.0), 5 | +1.0, -1.0, -1.6, -1.8, -0.8 |
| A | fresh | 56.0% | 51.5 ± 1.4 (50.0–53.2), 5 | -2.8, -2.2, -1.8, +0.4, +0.0 |
| A + 200k | development | 63.8% | 61.6 ± 0.9 (60.6–62.8), 5 | +9.2*, +9.8*, +8.2*, +8.2*, +7.6* |
| A + 200k | fresh | 62.8% | 60.2 ± 1.2 (59.0–61.8), 5 | +6.4*, +8.0*, +6.2*, +7.2*, +9.0* |
| B | development | 50.0% | 48.4 ± 1.1 (46.8–49.6), 5 | -4.0, -4.2, -6.2*, -3.4, -5.2* |
| B | fresh | 46.6% | 46.6 ± 1.5 (45.0–47.8), 5 | -5.0, -5.0, -7.8*, -7.8*, -5.2 |
| B answer-first | development | 53.0% | 49.8 ± 1.4 (48.2–51.8), 5 | -3.2, -1.2, -4.8, -2.8, -4.0 |
| B answer-first | fresh | 49.0% | 49.2 ± 0.6 (48.8–50.2), 5 | -4.0, -2.6, -3.8, -3.8, -3.8 |

## 2. Student B with a second seed

B seed 1: development 245/500 (49.0%), fresh 235/500 (47.0%) (B seed 0: 250/500 (50.0%), 233/500 (46.6%))
B seed 1 after pass 1 (development): 220/500 (44.0%)

| Comparison | Development | Fresh |
|---|---|---|
| B seed 1 vs B seed 0 | 28 vs 33, p = 0.61; gap -1.0 points (95% interval -4.0 to +2.0) | 24 vs 22, p = 0.88; gap +0.4 points (95% interval -2.2 to +3.2) |
| A seed 0 vs B seed 0 | 77 vs 43, p = 0.0024; gap +6.8 points (95% interval +2.6 to +11.0) | 87 vs 40, p = 3.7e-05; gap +9.4 points (95% interval +5.0 to +13.8) |
| A seed 0 vs B seed 1 | 83 vs 44, p = 0.00068; gap +7.8 points (95% interval +3.6 to +12.2) | 82 vs 37, p = 4.5e-05; gap +9.0 points (95% interval +4.8 to +13.2) |
| A seed 1 vs B seed 0 | 76 vs 45, p = 0.0062; gap +6.2 points (95% interval +2.0 to +10.4) | 88 vs 43, p = 0.0001; gap +9.0 points (95% interval +4.6 to +13.4) |
| A seed 1 vs B seed 1 | 82 vs 46, p = 0.0019; gap +7.2 points (95% interval +2.8 to +11.6) | 78 vs 35, p = 6.4e-05; gap +8.6 points (95% interval +4.6 to +12.8) |

## 3. Stockfish re-check of every wrong answer (current models)

Rule as in the original check: depth 18; a wrong answer is an equally good alternative if both moves force mate or it loses at most 5 win-% points against the Lichess move.

| Run | Wrong answers checked | Equally good alternative | Small (< 10) | Mistake (10-30) | Blunder (>= 30) |
|---|---|---|---|---|---|
| teacher, medium effort (dev) | 222 | 0 | 0 | 4 | 218 |
| teacher, medium effort (fresh) | 225 | 0 | 0 | 8 | 217 |
| teacher, low effort (fresh) | 288 | 0 | 0 | 7 | 281 |
| student A (dev) | 216 | 0 | 0 | 7 | 209 |
| student A (fresh) | 220 | 0 | 0 | 6 | 214 |
| student A + 200k (dev) | 181 | 0 | 0 | 6 | 175 |
| student A + 200k (fresh) | 186 | 0 | 0 | 9 | 177 |
| student B (dev) | 249 | 0 | 0 | 4 | 245 |
| student B (fresh) | 265 | 0 | 0 | 8 | 257 |

Total: 2052 wrong answers, 0 equally good alternatives.

## 4. Are the opponent replies in written lines real defenses? (Stockfish, depth 18)

Multi-move puzzles where the first move is right. A reply is a real defense if it loses at most 5 win-% points
against Stockfish's best defense, from the defender's side. The Lichess rows score the real solutions the same way:
they are the ceiling for this method (some positions are lost whatever the defender plays).

| Run | Right first move (multi-move) | Lines with a legal reply | Real defense (≤ 5) | Within 10 | Lichess reply | Median loss (win-%) |
|---|---|---|---|---|---|---|
| dev: Lichess solution (method check) | 449 | 449 | 440/449 (98%) | 447/449 (100%) | 449/449 (100%) | 0.3 |
| fresh: Lichess solution (method check) | 449 | 449 | 438/449 (98%) | 446/449 (99%) | 449/449 (100%) | 0.2 |
| dev: student B (before RL) | 199 | 100 | 75/100 (75%) | 86/100 (86%) | 57/100 (57%) | 0.3 |
| dev: B + RL answer-only | 214 | 11 | 7/11 (64%) | 9/11 (82%) | 5/11 (45%) | 2.3 |
| dev: B + RL truth v1 | 220 | 82 | 64/82 (78%) | 72/82 (88%) | 51/82 (62%) | 0.6 |
| dev: B + RL strict v2 | 211 | 0 | – | – | – | nan |
| dev: B + RL v3 | 195 | 18 | 13/18 (72%) | 16/18 (89%) | 7/18 (39%) | 1.6 |
| dev: B + RL v4, step 130 | 208 | 120 | 66/120 (55%) | 86/120 (72%) | 36/120 (30%) | 3.3 |
| dev: B + RL v4, step 260 | 215 | 131 | 65/131 (50%) | 89/131 (68%) | 32/131 (24%) | 5.1 |
| dev: B + RL v4, final | 213 | 130 | 67/130 (52%) | 87/130 (67%) | 39/130 (30%) | 5.0 |
| dev: student A | 233 | 178 | 131/178 (74%) | 148/178 (83%) | 101/178 (57%) | 0.4 |
| dev: student A + 200k | 268 | 232 | 184/232 (79%) | 205/232 (88%) | 144/232 (62%) | 0.3 |
| dev: teacher, medium effort | 214 | 168 | 129/168 (77%) | 142/168 (85%) | 82/168 (49%) | 0.1 |
| fresh: student B | 182 | 95 | 78/95 (82%) | 83/95 (87%) | 47/95 (49%) | 0.3 |
| fresh: B + RL v4, final | 197 | 128 | 77/128 (60%) | 95/128 (74%) | 41/128 (32%) | 2.2 |
| fresh: student A | 229 | 173 | 145/173 (84%) | 153/173 (88%) | 105/173 (61%) | 0.3 |
| fresh: student A + 200k | 263 | 224 | 177/224 (79%) | 194/224 (87%) | 139/224 (62%) | 0.2 |
| fresh: teacher, medium effort | 215 | 165 | 133/165 (81%) | 148/165 (90%) | 100/165 (61%) | 0.0 |

