# Reports and raw outputs

Every number in the main README comes from a report in this folder, and every report is computed from the graded model
outputs in `outputs/`. The reports were written by the scripts in `experiments/` and copied here by
`experiments/export_reports.py`.

## Which report backs which claim

| Claim in the README | Report |
|---|---|
| The teacher at low vs medium effort; student A matches the medium teacher | `teacher_baseline.md` |
| Answers-only students improve with more data (1.9k → 5.6k → 21.6k → 37.5k puzzles) | `students_pilot_1.9k.md`, `students_5.6k_21.6k.md`, `student_A.md` |
| Training on teacher explanations made students worse at every size | `students_pilot_1.9k.md`, `students_5.6k_21.6k.md`, `students_A_vs_B.md` |
| RL with a right-move reward didn't help either student | `rl_first_test.md`, `rl_answer_only_reward.md` |
| Four rewards on the explanation student, and how each was gamed | `rl_four_rewards_B.md` (the "Sanity notes" at the end were written after reading sample outputs) |

## Terms used in the reports

- **P1L**: the prompt. The position, a python-chess description of the board (diagram, pieces, legal moves) and a
  request for the forcing line (`FINAL_LINE`) and the move (`FINAL_MOVE`).
- **FDF-low**: the teacher's explanation format. gpt-oss-120b at low effort writes "as if discovering" the Lichess
  solution, with python-chess facts about the line in its prompt.
- **THE PATH**: the main experiment (student A = answers only vs student B = teacher explanations, then RL).
- **night 2**: the earlier LoRA experiments at 5.6k and 21.6k puzzles.
- **Legal line / full line right**: the written `FINAL_LINE` is legal from the puzzle position / matches the whole
  Lichess solution.
- **Claim-clean / move flag**: the claim checker found no false claim (piece on a square, "wins the X", mate) / the
  text names a move that is illegal or wrongly marked as check or mate.
- **RL runs**: `rlL` = right-move reward, `rlT` = truth v1, `rlT2` = strict v2, `rlT3` = v3; `step130`, `step260`,
  `final` (= 390 steps) are checkpoints. `ep1` = after the first of two training passes.

## Outputs

`outputs/<name>.jsonl.gz` is `results/<name>.jsonl` in the reports: one JSON record per test puzzle, with the prompt,
the model's full response (`content`; the teacher's hidden reasoning is in `reasoning`), the parsed move and line, the
grade (`status`: correct / wrong / illegal / parse_fail), the line checks and token usage. For example:

```python
import gzip, json
rows = [json.loads(l) for l in gzip.open("reports/outputs/student_path_A_nothink.jsonl.gz", "rt")]
print(sum(r["status"] == "correct" for r in rows), "of", len(rows))   # 284 of 500
```

The 500 test puzzles themselves are in `pilot_set.jsonl` at the top of the repo.
