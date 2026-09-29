# Reports and raw outputs

The main results in the README come from the reports in this folder, and their test-puzzle numbers are computed from the
graded model outputs in `outputs/`. A few extras in the reports, such as RL training curves and a small board-tracking
quiz, come from training logs and files that aren't included, as do a few side numbers in the README from early
exploratory runs (described with their setup where cited). The reports were written by the scripts in `experiments/` and
copied here by `experiments/export_reports.py`.

## Which report backs which claim

| Claim in the README | Report |
|---|---|
| The teacher plays illegal moves from a raw position; the board description fixes that | `teacher_baseline.md` ("How the prompt changes the teacher") |
| The teacher at low vs medium effort; student A against both | `teacher_baseline.md` |
| Answers-only students improve with more data (1.9k → 5.6k → 21.6k → 37.5k puzzles) | `students_pilot_1.9k.md`, `students_5.6k_21.6k.md`, `student_A.md` |
| Training on teacher explanations made students worse at every size | `students_pilot_1.9k.md`, `students_5.6k_21.6k.md`, `students_A_vs_B.md` |
| RL with a right-move reward didn't help either student | `rl_answer_only_reward.md` |
| Test puzzles were kept out of training (ids, source games, positions) | `../splits/train_puzzles.jsonl.gz`, checked by `experiments/check_overlap.py` (also a test) |
| Four rewards on the explanation student, and how each was reward-hacked | `rl_four_rewards_B.md` (the "Sanity notes" at the end were written after reading sample outputs) |

## Terms used in the reports

- **P1L**: the prompt. The position, a python-chess description of the board (diagram, pieces, legal moves) and a
  request for the forcing line (`FINAL_LINE`) and the move (`FINAL_MOVE`).
- **FDF-low**: the teacher's explanation format. gpt-oss-120b at low effort writes "as if discovering" the Lichess
  solution, with python-chess facts about the line in its prompt.
- **night 2**: the earlier LoRA experiments at 5.6k and 21.6k puzzles.
- **Legal line / full line right**: the written `FINAL_LINE` is legal from the puzzle position / matches the whole
  Lichess solution.
- **Claim-clean / move flag**: the claim checker found no false claim (piece on a square, "wins the X", mate) / the text
  names a move that is illegal or wrongly marked as check or mate.
- **pilot_formatA / pilot_formatB**: the first teacher runs on the 500 (low effort), with the position only (FEN) / the
  position + a list of legal moves. `test500_formatP1L` / `test500_med_formatP1L` = the P1L prompt at low / medium.
- **RL runs**: `rlL` = right-move reward, `rlT` = truth v1, `rlT2` = strict v2, `rlT3` = v3; `step130`, `step260`,
  `final` (= 390 steps) are checkpoints. `ep1` = after the first of two training passes.

## Outputs

`outputs/<name>.jsonl.gz` is `results/<name>.jsonl` in the reports: exactly one JSON record per test puzzle (where a
puzzle was retried after an error, the last record, as the reports read it), with the prompt, the model's full response
(`content`; the teacher's hidden reasoning is in `reasoning`), the parsed move and line, the grade (`status`: correct /
wrong / illegal / parse_fail), the line checks and token usage. For example:

```python
import gzip, json
rows = [json.loads(l) for l in gzip.open("reports/outputs/student_path_A_nothink.jsonl.gz", "rt")]
print(sum(r["status"] == "correct" for r in rows), "of", len(rows))   # 284 of 500
```

The 500 test puzzles themselves are in `test_set.jsonl` at the top of the repo.
