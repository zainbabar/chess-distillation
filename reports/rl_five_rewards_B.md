# RL for student B — can RL make imitated reasoning real?

RL runs from the same B checkpoint on the same 3,120 fresh puzzles, identical settings (390 steps × 8 puzzles ×
8 samples, lr 2e-6, no KL); only the reward differs:

- **Outcome-only**: right move 1, else 0.
- **Truth-aware (v1)**: +1 right move; +0.5 × fraction of the solution line written correctly; −0.5 if the claim
  checker finds a hard false claim (piece not on that square, false "wins the X", false mate); −0.5 if the
  explanation is under 25 words.
- **Strict (v2)**: line credit 0.5 × correct prefix / max(line length, solution length); −0.5 hard false claim;
  −0.5 if the text names an invented move; −0.5 if it has fewer than 2 verified moves, is under 40 words, or has
  no answer.
- **v3**: as v2 with line credit 1.0, invented-move penalty −0.25, and the floor counting distinct moves.
- **v4**: as v3, plus +0.25 for a complete, playable written line and −0.25 for one that breaks down when
  replayed or stops short; move sequences in the text must be playable in order (`src/rewards.py`).

Test: 500 held-out puzzles, greedy, max 1,024 new tokens.
"False mate claims" = explanations that claim checkmate or a forced mate (checkmate, mate in N, forced mate, …)
on puzzles with no mate in the solution.

| Model | Correct | Rating (95% CI) | Full line right | Mean line length (moves) | Right answers with no false claim | False mate claims (non-mate puzzles) | Mean tokens |
|---|---|---|---|---|---|---|---|
| B after SFT | **250/500 (50.0%)** | 1497 (1428–1567) | 66 | 3.76 | 170/250 (68%) | 40/356 (11%) | 179 |
| outcome-only RL, 130 steps | **277/500 (55.4%)** | 1574 (1514–1637) | 58 | 3.36 | 123/277 (44%) | 157/356 (44%) | 211 |
| truth-aware RL, 130 steps | **272/500 (54.4%)** | 1560 (1495–1623) | 59 | 3.29 | 261/272 (96%) | 2/356 (1%) | 147 |
| strict truth-aware RL (v2), 130 steps | **262/500 (52.4%)** | 1531 (1465–1588) | 52 | 1.61 | 248/262 (95%) | 12/356 (3%) | 163 |
| calc-rewarding truth RL (v3), 130 steps | **263/500 (52.6%)** | 1534 (1465–1594) | 60 | 2.81 | 257/263 (98%) | 9/356 (3%) | 161 |
| v4: replays the line, 130 steps | **259/500 (51.8%)** | 1522 (1456–1587) | 53 | 2.79 | 247/259 (95%) | 8/356 (2%) | 179 |
| outcome-only RL, 260 steps | **252/500 (50.4%)** | 1502 (1438–1565) | 52 | 2.39 | 116/252 (46%) | 268/356 (75%) | 211 |
| truth-aware RL, 260 steps | **274/500 (54.8%)** | 1565 (1501–1629) | 57 | 3.51 | 272/274 (99%) | 1/356 (0%) | 114 |
| strict truth-aware RL (v2), 260 steps | **251/500 (50.2%)** | 1499 (1435–1568) | 49 | 1.00 | 249/251 (99%) | 0/356 (0%) | 177 |
| calc-rewarding truth RL (v3), 260 steps | **261/500 (52.2%)** | 1528 (1460–1593) | 55 | 2.78 | 261/261 (100%) | 1/356 (0%) | 153 |
| v4: replays the line, 260 steps | **266/500 (53.2%)** | 1542 (1477–1605) | 55 | 2.83 | 264/266 (99%) | 2/356 (1%) | 225 |
| outcome-only RL, 390 steps | **264/500 (52.8%)** | 1537 (1474–1603) | 50 | 1.19 | 135/264 (51%) | 286/356 (80%) | 179 |
| truth-aware RL, 390 steps | **271/500 (54.2%)** | 1557 (1494–1622) | 57 | 5.42 | 271/271 (100%) | 1/356 (0%) | 133 |
| strict truth-aware RL (v2), 390 steps | **262/500 (52.4%)** | 1531 (1461–1597) | 51 | 1.00 | 261/262 (100%) | 0/356 (0%) | 147 |
| calc-rewarding truth RL (v3), 390 steps | **246/500 (49.2%)** | 1485 (1417–1555) | 51 | 2.35 | 246/246 (100%) | 1/356 (0%) | 156 |
| v4: replays the line, 390 steps | **264/500 (52.8%)** | 1537 (1469–1605) | 51 | 2.80 | 264/264 (100%) | 1/356 (0%) | 196 |

Honesty checks (is it truer, or vaguer / padded?):

| Model | Words | Checkable claims per text | Texts naming an invented move | Legal line | Lines with a legal reply | Line longer than solution |
|---|---|---|---|---|---|---|
| B after SFT | 110 | 10.9 | 333/500 | 131 | 231 | 136 |
| outcome-only RL, 130 steps | 134 | 13.0 | 388/500 | 128 | 209 | 77 |
| truth-aware RL, 130 steps | 89 | 6.9 | 277/500 | 187 | 230 | 68 |
| strict truth-aware RL (v2), 130 steps | 113 | 7.0 | 77/500 | 410 | 89 | 0 |
| calc-rewarding truth RL (v3), 130 steps | 95 | 10.3 | 215/500 | 265 | 260 | 10 |
| v4: replays the line, 130 steps | 111 | 11.0 | 187/500 | 285 | 264 | 5 |
| outcome-only RL, 260 steps | 138 | 12.3 | 355/500 | 217 | 112 | 14 |
| truth-aware RL, 260 steps | 64 | 4.9 | 275/500 | 103 | 210 | 87 |
| strict truth-aware RL (v2), 260 steps | 127 | 8.2 | 2/500 | 499 | 0 | 0 |
| calc-rewarding truth RL (v3), 260 steps | 89 | 9.9 | 141/500 | 238 | 234 | 6 |
| v4: replays the line, 260 steps | 139 | 12.0 | 152/500 | 299 | 277 | 7 |
| outcome-only RL, 390 steps | 117 | 9.4 | 57/500 | 454 | 26 | 0 |
| truth-aware RL, 390 steps | 67 | 5.7 | 376/500 | 83 | 198 | 298 |
| strict truth-aware RL (v2), 390 steps | 100 | 7.5 | 1/500 | 500 | 0 | 0 |
| calc-rewarding truth RL (v3), 390 steps | 91 | 9.7 | 15/500 | 95 | 36 | 6 |
| v4: replays the line, 390 steps | 117 | 10.7 | 144/500 | 328 | 290 | 0 |

Paired accuracy comparisons (exact McNemar):

- v4 130 vs SFT: 67 vs 58, p = 0.47
- v4 130 vs v3 130: 39 vs 43, p = 0.74
- v4 130 vs strict v2 130: 43 vs 46, p = 0.83
- v4 130 vs outcome-only 130: 29 vs 47, p = 0.05
- v3 130 vs SFT: 62 vs 49, p = 0.25
- v3 130 vs strict v2 130: 37 vs 36, p = 1
- v3 130 vs outcome-only 130: 21 vs 35, p = 0.081
- strict v2 130 vs SFT: 70 vs 58, p = 0.33
- strict v2 130 vs outcome-only 130: 28 vs 43, p = 0.096
- strict v2 130 vs truth v1 130: 34 vs 44, p = 0.31
- truth-aware 130 vs SFT: 70 vs 48, p = 0.053
- truth-aware 130 vs outcome-only 130: 31 vs 36, p = 0.63
- v4 260 vs SFT: 71 vs 55, p = 0.18
- v4 260 vs v3 260: 27 vs 22, p = 0.57
- v4 260 vs strict v2 260: 37 vs 22, p = 0.067
- v4 260 vs outcome-only 260: 37 vs 23, p = 0.092
- v3 260 vs SFT: 71 vs 60, p = 0.38
- v3 260 vs strict v2 260: 34 vs 24, p = 0.24
- v3 260 vs outcome-only 260: 38 vs 29, p = 0.33
- strict v2 260 vs SFT: 68 vs 67, p = 1
- strict v2 260 vs outcome-only 260: 35 vs 36, p = 1
- strict v2 260 vs truth v1 260: 22 vs 45, p = 0.0067
- truth-aware 260 vs SFT: 72 vs 48, p = 0.035
- truth-aware 260 vs outcome-only 260: 40 vs 18, p = 0.0054
- v4 390 vs SFT: 71 vs 57, p = 0.25
- v4 390 vs v3 390: 54 vs 36, p = 0.073
- v4 390 vs strict v2 390: 41 vs 39, p = 0.91
- v4 390 vs outcome-only 390: 33 vs 33, p = 1
- v3 390 vs SFT: 73 vs 77, p = 0.81
- v3 390 vs strict v2 390: 38 vs 54, p = 0.12
- v3 390 vs outcome-only 390: 40 vs 58, p = 0.085
- strict v2 390 vs SFT: 73 vs 61, p = 0.34
- strict v2 390 vs outcome-only 390: 42 vs 44, p = 0.91
- strict v2 390 vs truth v1 390: 37 vs 46, p = 0.38
- truth-aware 390 vs SFT: 80 vs 59, p = 0.089
- truth-aware 390 vs outcome-only 390: 31 vs 24, p = 0.42
- truth-aware 390 vs A after SFT (answers only, 284): 45 vs 58, p = 0.24
- strict v2 390 vs A after SFT (answers only, 284): 44 vs 66, p = 0.045
- v3 390 vs A after SFT (answers only, 284): 50 vs 88, p = 0.0015
- v4 390 vs A after SFT (answers only, 284): 46 vs 66, p = 0.072

## Training (truth v1 run)

- mean reward in 30-step blocks: 0.233 → 0.315 → 0.401 → 0.490 → 0.516 → 0.558 → 0.574 → 0.626 → 0.606 → 0.567 → 0.592 → 0.587 → 0.577
- groups with no learning signal: 3% → 52% (outcome-only run: 28% → 66%)
- running sample accuracy / claim-error rate: step 5: 0.425 / 0.497, step 70: 0.393 / 0.388, step 135: 0.416 / 0.253, step 200: 0.432 / 0.179, step 265: 0.451 / 0.137, step 330: 0.459 / 0.111, step 390: 0.463 / 0.094

## Training (strict v2 run)

- mean reward in 30-step blocks: -0.269 → -0.116 → 0.017 → 0.231 → 0.386 → 0.474 → 0.528 → 0.552 → 0.595 → 0.567 → 0.562 → 0.529 → 0.523
- groups with no learning signal: 1% → 59% (outcome-only run: 28% → 66%)
- running sample accuracy / claim-error rate: step 5: 0.428 / 0.519, step 70: 0.408 / 0.393, step 135: 0.416 / 0.260, step 200: 0.435 / 0.186, step 265: 0.452 / 0.145, step 330: 0.461 / 0.119, step 390: 0.462 / 0.102

## Training (v3 run)

- mean reward in 30-step blocks: -0.045 → 0.158 → 0.203 → 0.422 → 0.471 → 0.505 → 0.567 → 0.638 → 0.641 → 0.581 → 0.584 → 0.579 → 0.584
- groups with no learning signal: 2% → 42% (outcome-only run: 28% → 66%)
- running sample accuracy / claim-error rate: step 5: 0.422 / 0.509, step 70: 0.404 / 0.355, step 135: 0.421 / 0.230, step 200: 0.436 / 0.164, step 265: 0.456 / 0.126, step 330: 0.458 / 0.103, step 390: 0.457 / 0.089

## Training (v4 run)

- mean reward in 30-step blocks: -0.235 → -0.020 → 0.065 → 0.347 → 0.399 → 0.472 → 0.565 → 0.602 → 0.697 → 0.619 → 0.648 → 0.670 → 0.620
- groups with no learning signal: 0% → 30% (outcome-only run: 28% → 66%)
- running sample accuracy / claim-error rate: step 5: 0.438 / 0.484, step 70: 0.401 / 0.405, step 135: 0.420 / 0.274, step 200: 0.437 / 0.202, step 265: 0.455 / 0.158, step 330: 0.457 / 0.129, step 390: 0.459 / 0.110

## Sanity notes (Claude, 2026-09-29 17:10, after reading sample outputs)

- **v4 was gamed too, in a new way: legal but invented lines.** It closed v3's loophole (no more moves by the model's own
  side written as the reply: lines with a legal reply 290, legal lines 328, nothing padded past the solution), but the
  reward pays for a complete, *legal* line, not a *correct* one. By the final checkpoint the model writes one template:
  the right first move, an invented "only legal response" for the opponent, a quiet filler move, "Thus the forced
  sequence is …".
- Measured on multi-move puzzles with the first move right (B after SFT → v4 130 → 260 → 390; A for reference):
  lines exactly 3 moves long (the reward's minimum) 53% → 92% → 95% → **100%** (A 70%); the model's own second move is
  quiet, no capture or check, 9% → 25% → 40% → **54%** (A 17%); the opponent's reply matches the Lichess solution
  29% → 17% → 15% → **18%** (A 43%); template wording ("improve the position" / "Thus the forced sequence is")
  0% → 23% → 49% → **98%**.
- "Only legal response/move" claims about the opponent's reply: 151 texts at the final checkpoint (SFT 37); in **104
  (69%)** the opponent had more than one legal move. The claim checker doesn't check "only" claims, so they went
  unpunished.
- Spot-read (yl9Uw, 4bBW0, nXlKS): all three follow the template. yl9Uw: "Rxf2 … the only legal move is Kh8 … the only
  safe king move is Kh2" (the solution is a knight trade); 4bBW0: "White's only legal response is Kh1" (the solution
  goes Kf1), then Ke7 as filler; nXlKS: "Black's only legal response is Kh7" (the solution is Kxg5).
- Lesson: checking that reasoning is *legal* in order isn't enough; the reward has to check that it's *right* (e.g. the
  opponent's reply must be a real defense by Stockfish). Accuracy: 52.8% final (A 56.8%, 66 vs 46, p = 0.072).
