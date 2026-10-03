# Reports and raw outputs

The results in the README come from the reports in this folder. The development-set numbers are computed from the
graded model outputs in `outputs/`. The fresh-test reports and those of the newer models are here too, but their graded
outputs aren't published yet; until they are, those reports are the record. A few extras, such as RL training curves
and a small board-tracking quiz, come from training logs and files that aren't included, as do a few side numbers in
the README from early exploratory runs (described with their setup where cited). The reports were written by the
scripts in `experiments/` (notes marked "Sanity notes" or "added" were written by hand after reading outputs) and copied
here by `experiments/export_reports.py`.

## Which report backs which claim

| Claim in the README | Report |
|---|---|
| The teacher plays illegal moves from a raw position; the board description fixes that | `teacher_baseline.md` ("How the prompt changes the teacher") |
| The teacher at low vs medium effort; students A, B and A + 200k against both (development set) | `teacher_baseline.md` |
| Fresh test 1: every model on 500 untouched puzzles, the primary comparisons | `fresh_test.md` (protocol: `fresh_test_protocol.md`) |
| Fresh test 2, the pooled 1,000-puzzle comparisons, and the second-seed copies of A + 200k, B answer-first and RL v4 | `replication_and_fresh_test_2.md` (protocol: `fresh_test_2_protocol.md`) |
| Answers-only accuracy rises with more data (1.9k → 5.6k → 21.6k → 37.5k); the early steps are within noise and the last also switched LoRA → full fine-tune | `students_pilot_1.9k.md`, `students_5.6k_21.6k.md`, `student_A.md` |
| Training on teacher explanations made students worse at every size | `students_pilot_1.9k.md`, `students_5.6k_21.6k.md`, `students_A_vs_B.md`, `fresh_test.md` |
| B answer-first; sampled vs greedy decoding; the RL models on fresh set 1 | `answer_first_sampling_rl_fresh.md` (pooled answer-first comparisons: `replication_and_fresh_test_2.md` and the README) |
| Five samples per student, B's second seed (all four seed pairings), the Stockfish check of 2,052 wrong answers, Stockfish check of written replies | `tightening_checks.md` |
| The untrained model at the students' settings, the teacher resampled, sound full lines | `final_checks.md` |
| RL with a right-move reward gave no significant gain by the end of the run (B gained at step 130, p = 0.019 uncorrected, then fell back) | `rl_answer_only_reward.md` |
| Four rewards on the explanation student and how each was reward-hacked (the "Sanity notes" were written after reading sample outputs) | `rl_four_rewards_B.md` |
| The fifth reward (v4, replays the line) and how it was hacked; each reward's final B checkpoint vs A | `rl_five_rewards_B.md` |
| Test puzzles were kept out of training (ids, source games, positions) | `../splits/train_puzzles.jsonl.gz`, checked by `experiments/check_overlap.py` (also a test) |

## Terms used in the reports

- **Development set / fresh test sets**: `test_set.jsonl` (the original 500, called the "test set" in the older
  reports) / `fresh_test_set.jsonl` and `fresh_test_set_2.jsonl` (500 untouched puzzles each).
- **P1L**: the prompt. The position, a python-chess description of the board (diagram, pieces, legal moves) and a
  request for the forcing line (`FINAL_LINE`) and the move (`FINAL_MOVE`).
- **FDF-low**: the teacher's explanation format. gpt-oss-120b at low effort writes "as if discovering" the Lichess
  solution, with python-chess facts about the line in its prompt.
- **night 2**: the earlier LoRA experiments at 5.6k and 21.6k puzzles.
- **Legal line / full line right**: the written `FINAL_LINE` is legal from the puzzle position / matches the whole
  Lichess solution. **Sound full line** (Stockfish): legal, every solver move as good as Stockfish's best, every
  opponent move a real defense, and as long as the solution or ending in mate.
- **Claim-clean / move flag**: the claim checker found no false claim (piece on a square, "wins the X", mate) / the text
  names a move that is illegal or wrongly marked as check or mate.
- **pilot_formatA / pilot_formatB**: the first teacher runs on the 500 (low effort), with the position only (FEN) / the
  position + a list of legal moves. `test500_formatP1L` / `test500_med_formatP1L` = the P1L prompt at low / medium.
- **Students**: `path_A` = A, `path_B` = B, `path_A200k` = A + 200k, `path_Baf` = B answer-first; `_seed1` = the
  second-seed copy.
- **RL runs**: `rlL` = right-move reward, `rlT` = truth v1, `rlT2` = strict v2, `rlT3` = v3, `rlT4` = v4; `step130`,
  `step260`, `final` (= 390 steps) are checkpoints. `ep1` = after the first of two training passes.

## Outputs

`outputs/<name>.jsonl.gz` is `results/<name>.jsonl` in the reports: exactly one JSON record per development-set puzzle
(where a puzzle was retried after an error, the last record, as the reports read it), with the prompt, the model's full
response (`content`; the teacher's hidden reasoning is in `reasoning`), the parsed move and line, the grade (`status`:
correct / wrong / illegal / parse_fail), the line checks and token usage. For example:

```python
import gzip, json
rows = [json.loads(l) for l in gzip.open("reports/outputs/student_path_A_nothink.jsonl.gz", "rt")]
print(sum(r["status"] == "correct" for r in rows), "of", len(rows))   # 284 of 500
```

The puzzles themselves are in `test_set.jsonl`, `fresh_test_set.jsonl` and `fresh_test_set_2.jsonl` at the top of the
repo.
