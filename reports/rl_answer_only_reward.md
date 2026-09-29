# RL run (~12 h on the Spark) — does imitated reasoning pay off after RL?

GRPO from each SFT checkpoint, identical settings: binary reward (right move 1, wrong 0, no FINAL_MOVE −0.1),
8 fresh puzzles × 8 samples per step (the same puzzles for A and B, never used before), lr 2e-6, temperature 1.0,
no KL, DAPO loss. Test: 500 held-out puzzles, greedy, max 1,024 new tokens. Step 0 = the SFT model.

## A: answers only (RL steps: 390)

| RL steps | Correct | Rating (95% CI) | Illegal | Legal line | Full line right | Mean output tokens |
|---|---|---|---|---|---|---|
| 0 (SFT) | **284/500 (56.8%)** | 1594 (1530–1660) | 0 | 317 | 90 | 26 |
| 130 | **271/500 (54.2%)** | 1557 (1493–1623) | 0 | 232 | 78 | 27 |
| 260 | **289/500 (57.8%)** | 1608 (1543–1670) | 0 | 255 | 81 | 26 |
| 390 | **267/500 (53.4%)** | 1545 (1477–1614) | 0 | 241 | 69 | 24 |
- after 130 steps vs SFT: 46 vs 59, p = 0.24
- after 260 steps vs SFT: 38 vs 33, p = 0.64
- after 390 steps vs SFT: 49 vs 66, p = 0.14

## B: teacher explanations (RL steps: 390)

| RL steps | Correct | Rating (95% CI) | Illegal | Legal line | Full line right | Mean output tokens |
|---|---|---|---|---|---|---|
| 0 (SFT) | **250/500 (50.0%)** | 1497 (1428–1567) | 1 | 131 | 66 | 179 |
| 130 | **277/500 (55.4%)** | 1574 (1514–1637) | 0 | 128 | 58 | 211 |
| 260 | **252/500 (50.4%)** | 1502 (1438–1565) | 0 | 217 | 52 | 211 |
| 390 | **264/500 (52.8%)** | 1537 (1474–1603) | 1 | 454 | 50 | 179 |
- after 130 steps vs SFT: 75 vs 48, p = 0.019
- after 260 steps vs SFT: 65 vs 63, p = 0.93
- after 390 steps vs SFT: 73 vs 59, p = 0.26

## B vs A at equal RL steps

- step 0 (SFT): B vs A 43 vs 77, p = 0.0024
- step 130: B vs A 48 vs 42, p = 0.6
- step 260: B vs A 25 vs 62, p = 9.1e-05
- step final: B vs A 47 vs 50, p = 0.84

Teacher gpt-oss-120b (low, 1 attempt): 222/500.
- A final vs teacher: 122 vs 77, p = 0.0017
- B final vs teacher: 107 vs 65, p = 0.0017

## Training curve (sample accuracy on fresh puzzles at temperature 1.0, 30-step blocks)

- A (390 steps): 0.443 → 0.505 → 0.450 → 0.505 → 0.536 → 0.508 → 0.530 → 0.554 → 0.589 → 0.503 → 0.504 → 0.477 → 0.460
  - output length 31 → 26 tokens; groups with no signal 34% → 83%
  - 225.3 min, 35 s/step, 24,960 samples, 13 parse fails
- B (390 steps): 0.394 → 0.415 → 0.411 → 0.455 → 0.461 → 0.473 → 0.476 → 0.510 → 0.530 → 0.464 → 0.462 → 0.463 → 0.462
  - output length 201 → 190 tokens; groups with no signal 28% → 66%
  - 334.1 min, 51 s/step, 24,960 samples, 42 parse fails

## Is B's reasoning truer after RL? (claim checker on the text before FINAL_LINE)

- step 0 (SFT): right answers (250): clean 68%, move flag 56%; wrong answers (250): clean 30%, move flag 77%
- step 130: right answers (277): clean 44%, move flag 70%; wrong answers (223): clean 18%, move flag 87%
- step 260: right answers (252): clean 46%, move flag 65%; wrong answers (248): clean 19%, move flag 77%
- step final: right answers (264): clean 51%, move flag 8%; wrong answers (236): clean 28%, move flag 15%

An earlier 100-step test on different puzzles (not included here) gave A 284 → 275, B 250 → 264.

## Sanity notes (Claude, 12:15)

- All 8 evals graded 500/500; parse fails 0; B final has 4 truncated answers.
- **B's "legal line" 454 and "move flag 8%" at the final checkpoint are artifacts, not better reasoning:** RL taught B to
  drop the continuation — 446/500 final lines are the first move only (SFT: mean line 3.8 moves, final 1.2; the reward
  only checks the first move), and a 1-move line is trivially legal; fewer written moves also means fewer move flags.
  A's lines shortened a little too (3.6 → 3.0 moves; full line right 90 → 69).
- **B's explanations drifted into confident false mates:** they claim checkmate/forced mate on most of the 356 non-mate test
  puzzles after RL: 286 (SFT: 40) by the mate-word count used in `rl_four_rewards_B.md` and the README, 293 (SFT: 55)
  by the claim checker. (This note first gave 255 vs 37 from an earlier, looser count.) Claim-clean on right answers 68% → 51%.
- Spot-read of 3 puzzles B solves before and after RL (j05AW 830, b8msd 894, ggDHA 1526): SFT explanations 3/3 good
  (right idea + right continuation); after RL 3/3 bad (invented "checkmate", invented square control; step 260 invents
  whole lines like "Ne7+ Qxe7 Qxe7#").
- **Exploration collapsed:** groups where all 8 samples got the same reward (no learning signal) rose from 34% → 83% (A)
  and 28% → 66% (B); the training curves peak mid-run (A 0.59, B 0.53) and sag after — the policies became nearly
  deterministic. No entropy/KL control in this recipe (beta 0, symmetric clip).
- Checkpoint-to-checkpoint swings of ±20–25 puzzles (e.g. B 277 → 252 → 264) are bigger than eval noise alone (~±15), so
  the policies themselves oscillate; single seed.
