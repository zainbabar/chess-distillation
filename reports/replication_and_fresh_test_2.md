# Replications and the second fresh test

## 1. A + 200k, replicated

Replicate: A's second-seed model (`ckpt/path_A_seed1`) continued on the same 200k answers with seed 1, so both
stages of the pipeline differ in their randomness from the original.

- development: original 319/500 (63.8%), **replicate 319/500 (63.8%)**; replicate vs original: 22 vs 22, p = 1; gap +0.0 points (95% interval -2.6 to +2.6); replicate vs teacher medium: 114 vs 60, p = 5.2e-05; gap +10.8 points (95% interval +5.8 to +16.0)
- fresh 1: original 314/500 (62.8%), **replicate 311/500 (62.2%)**; replicate vs original: 20 vs 23, p = 0.76; gap -0.6 points (95% interval -3.2 to +2.0); replicate vs teacher medium: 105 vs 58, p = 0.00029; gap +9.4 points (95% interval +4.4 to +14.2)
- fresh 2: original 319/500 (63.8%), **replicate 333/500 (66.6%)**; replicate vs original: 26 vs 12, p = 0.034; gap +2.8 points (95% interval +0.4 to +5.2); replicate vs teacher medium: 111 vs 42, p = 2.2e-08; gap +13.8 points (95% interval +9.2 to +18.6)

## 2. B answer-first, replicated (seed 1)

- development: original 265/500 (53.0%), **replicate 271/500 (54.2%)**; replicate vs original: 20 vs 14, p = 0.39; gap +1.2 points (95% interval -1.0 to +3.4)
  - replicate vs A seed 0: 24 vs 37, p = 0.12; gap -2.6 points (95% interval -5.6 to +0.4)
  - replicate vs A seed 1: 24 vs 34, p = 0.24; gap -2.0 points (95% interval -5.0 to +1.0)
  - replicate vs B seed 0: 68 vs 47, p = 0.062; gap +4.2 points (95% interval +0.0 to +8.4)
  - replicate vs B seed 1: 75 vs 49, p = 0.024; gap +5.2 points (95% interval +0.8 to +9.6)
- fresh 1: original 245/500 (49.0%), **replicate 256/500 (51.2%)**; replicate vs original: 20 vs 9, p = 0.061; gap +2.2 points (95% interval +0.2 to +4.4)
  - replicate vs A seed 0: 25 vs 49, p = 0.0071; gap -4.8 points (95% interval -8.2 to -1.4)
  - replicate vs A seed 1: 24 vs 46, p = 0.012; gap -4.4 points (95% interval -7.6 to -1.2)
  - replicate vs B seed 0: 79 vs 56, p = 0.058; gap +4.6 points (95% interval +0.0 to +9.2)
  - replicate vs B seed 1: 75 vs 54, p = 0.078; gap +4.2 points (95% interval -0.2 to +8.6)
- after pass 1 (development): replicate 266/500 (53.2%) (original 53.0%)

## 3. RL reward v4, replicated (seed 1)

| Run | Dev, step 130 | Dev, step 260 | Dev, final | Fresh 1, final |
|---|---|---|---|---|
| v4, seed 0 | 259/500 (51.8%) | 266/500 (53.2%) | 264/500 (52.8%) | 248/500 (49.6%) |
| v4, seed 1 | 257/500 (51.4%) | 268/500 (53.6%) | 269/500 (53.8%) | 247/500 (49.4%) |

Template measurements (multi-move puzzles, first move right; development set, final checkpoint):

| Run | Lines exactly 3 moves | Quiet 2nd own move | Reply = Lichess | Template wording | False "only legal" claims |
|---|---|---|---|---|---|
| B before RL | 53% | 9% | 29% | 0% | 20/37 |
| v4, seed 0 | 100% | 54% | 18% | 98% | 104/151 |
| v4, seed 1 | 91% | 45% | 16% | 25% | 91/135 |

Stockfish reply quality (results/replication_reply_quality.md):

| Run | Right first move (multi-move) | Lines with a legal reply | Real defense (≤ 5) | Within 10 | Lichess reply | Median loss (win-%) |
|---|---|---|---|---|---|---|
| dev: Lichess solution (method check) | 449 | 449 | 440/449 (98%) | 447/449 (100%) | 449/449 (100%) | 0.3 |
| dev: B + RL v4, seed 0, final | 213 | 130 | 65/130 (50%) | 88/130 (68%) | 39/130 (30%) | 5.0 |
| dev: B + RL v4, seed 1, final | 218 | 133 | 73/133 (55%) | 97/133 (73%) | 35/133 (26%) | 3.1 |
| fresh: B + RL v4, seed 0, final | 197 | 128 | 80/128 (62%) | 94/128 (73%) | 41/128 (32%) | 2.2 |
| fresh: B + RL v4, seed 1, final | 196 | 121 | 82/121 (68%) | 91/121 (75%) | 37/121 (31%) | 2.5 |

## 4. The second fresh test (500 more untouched puzzles)

Protocol fixed before the run: `reports/fresh_test_2_protocol.md`.

| Model | Fresh 2: first move right | Puzzle rating (95% CI) | Full line right | Fresh 1 | Pooled (1,000) |
|---|---|---|---|---|---|
| student A | **275/500 (55.0%)** | 1567 (1506–1631) | 94 | 280/500 (56.0%) | 555/1000 (55.5%) |
| student A, seed 1 | **283/500 (56.6%)** | 1590 (1525–1654) | 91 | 278/500 (55.6%) | 561/1000 (56.1%) |
| student A + 200k | **319/500 (63.8%)** | 1693 (1632–1753) | 130 | 314/500 (62.8%) | 633/1000 (63.3%) |
| student A + 200k, replicate | **333/500 (66.6%)** | 1735 (1675–1797) | 129 | 311/500 (62.2%) | 644/1000 (64.4%) |
| student B | **237/500 (47.4%)** | 1459 (1397–1522) | 62 | 233/500 (46.6%) | 470/1000 (47.0%) |
| student B, seed 1 | **232/500 (46.4%)** | 1444 (1380–1504) | 71 | 235/500 (47.0%) | 467/1000 (46.7%) |
| B answer-first | **270/500 (54.0%)** | 1552 (1491–1613) | 86 | 245/500 (49.0%) | 515/1000 (51.5%) |
| B answer-first, replicate | **271/500 (54.2%)** | 1555 (1487–1619) | 80 | 256/500 (51.2%) | 527/1000 (52.7%) |
| B + RL v4 | **247/500 (49.4%)** | 1487 (1430–1546) | 52 | 248/500 (49.6%) | 495/1000 (49.5%) |
| B + RL v4, replicate | **260/500 (52.0%)** | 1524 (1458–1586) | 54 | 247/500 (49.4%) | 507/1000 (50.7%) |
| untrained Qwen3-1.7B | **6/500 (1.2%)** | 415 (214–534) | 1 | 6/500 (1.2%) | 12/1000 (1.2%) |
| teacher, low effort | **214/500 (42.8%)** | 1393 (1333–1461) | 63 | 202/500 (40.4%) | 416/1000 (41.6%) |
| teacher, medium effort | **264/500 (52.8%)** | 1535 (1474–1598) | 103 | 264/500 (52.8%) | 528/1000 (52.8%) |
| Stockfish 16, 0.1 s | **498/500 (99.6%)** | 2783 (2613–4000) | – | 495/500 (99.0%) | 993/1000 (99.3%) |
| random legal move (expected) | 5.4% | | | | |

Primary comparisons, fresh 2 (exact McNemar, Holm over the four; gap with a 95% paired bootstrap interval):

- student A vs student B: 80 vs 42, p = 0.00074; gap +7.6 points (95% interval +3.4 to +12.0); Holm-adjusted p = 0.0015
- student A vs teacher, low effort: 124 vs 63, p = 9.7e-06; gap +12.2 points (95% interval +7.0 to +17.4); Holm-adjusted p = 3.1e-05
- student A vs teacher, medium effort: 88 vs 77, p = 0.44; gap +2.2 points (95% interval -2.8 to +7.4); Holm-adjusted p = 0.44
- student A + 200k vs teacher, medium effort: 102 vs 47, p = 7.7e-06; gap +11.0 points (95% interval +6.4 to +15.8); Holm-adjusted p = 3.1e-05

Primary comparisons, pooled over both fresh sets (1,000) (exact McNemar, Holm over the four; gap with a 95% paired bootstrap interval):

- student A vs student B: 167 vs 82, p = 7.7e-08; gap +8.5 points (95% interval +5.5 to +11.5); Holm-adjusted p = 1.5e-07
- student A vs teacher, low effort: 246 vs 107, p = 9.9e-14; gap +13.9 points (95% interval +10.4 to +17.4); Holm-adjusted p = 4e-13
- student A vs teacher, medium effort: 176 vs 149, p = 0.15; gap +2.7 points (95% interval -0.8 to +6.3); Holm-adjusted p = 0.15
- student A + 200k vs teacher, medium effort: 213 vs 108, p = 4.7e-09; gap +10.5 points (95% interval +7.1 to +14.0); Holm-adjusted p = 1.4e-08

## Notes (added by hand 2026-10-03, computed with `teacher_report.pair`)

B answer-first pooled over both fresh sets (1,000 puzzles), every pairing of the two seeds of each model (secondary,
not in the protocol):

| | vs A seed 0 | vs A seed 1 | vs B seed 0 | vs B seed 1 |
|---|---|---|---|---|
| B answer-first, seed 0 | 55 vs 95, p = 0.0014; −4.0 (−6.4 to −1.6) | 51 vs 97, p = 0.00019; −4.6 (−7.0 to −2.3) | 148 vs 103, p = 0.0054; +4.5 (+1.5 to +7.6) | 148 vs 100, p = 0.0028; +4.8 (+1.8 to +7.8) |
| B answer-first, seed 1 | 61 vs 89, p = 0.027; −2.8 (−5.2 to −0.4) | 51 vs 85, p = 0.0045; −3.4 (−5.7 to −1.2) | 159 vs 102, p = 0.0005; +5.7 (+2.6 to +8.9) | 158 vs 98, p = 0.00021; +6.0 (+3.0 to +9.1) |

Answer-first sits significantly between A and B in all eight comparisons: about half of the A–B gap (8.5 points pooled)
comes from answering after the explanation, half from training on the prose. This revises the one-seed reading in
`answer_first_sampling_rl_fresh.md` ("mostly the prose").

The v4 replicate's "template wording" share (25%) is low because the regex matches seed 0's exact phrasing; seed 1
writes the same structure in other words ("Thus the correct continuation is…"), as the three-move-line share (91%) and
the false "only legal" claims (91 of 135) show. Spot puzzles yl9Uw (Rxf2 Kh7 Bd3), 4bBW0 (Rxg2+ Kh1 Rd8), nXlKS (g5+ Kh7
Qxe8) follow the template.

