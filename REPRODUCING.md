# Code and reproducing

The pipeline, where everything lives, and the commands to rerun it. Results and findings are in the [README](README.md);
the reports behind them are in [`reports/`](reports/).

## How it works

```
Lichess puzzles ─► development set (500) + two fresh test sets (2 × 500), all held out
      │            + balanced training pools (60k / 200k), test puzzles excluded by id, game and position
      │
      ├─► teacher (gpt-oss-120b) ─► baseline scores, and explanations of the Lichess solution
      │        (python-chess facts in the prompt; a claim checker filters false statements)
      │
      ├─► answers-only data (move + line) straight from Lichess
      ▼
students (Qwen3-1.7B) ─► imitation (full fine-tune) ─► RL (GRPO, checked rewards) ─► evaluation on the test sets
                                                                                     (+ Stockfish checks of grading and lines)
```

| Folder | What's in it |
|---|---|
| [`src/`](src/) | The pipeline: puzzle sets, the board description, teacher runs and grading, the fact and claim checkers, training data, student training, evaluation, RL rewards and statistics |
| [`experiments/`](experiments/) | Reports and analyses for each experiment, the Stockfish checks, the figures (`make_figures.py`), the Puzzle Explorer builder and the export to `reports/` |
| [`reports/`](reports/) | The reports behind the README's results, and the graded development-set outputs they come from |
| [`figures/`](figures/) | The charts and board pictures in the README |
| [`docs/`](docs/) | The Puzzle Explorer page (built by `experiments/build_explorer.py`, served by GitHub Pages) |
| [`tests/`](tests/) | Fast tests of the grader, the claim checker, the RL reward, the board description, the statistics and the train/test overlap (run on every push) |
| [`splits/`](splits/) | Every puzzle any model was trained on (243,872, with moves and source game), to check the test sets against: `experiments/check_overlap.py` |
| [`results/`](results/) | The run scripts behind each experiment (`run_*.sh`); outputs and logs written here stay local |
| `test_set.jsonl` | The development set: 500 puzzles held out from training (seed 42) |
| `fresh_test_set.jsonl`, `fresh_test_set_2.jsonl` | The fresh test sets: 500 untouched puzzles each (seeds 20260929 and 20261001) |

Key files in `src/`:

| File | What it does |
|---|---|
| `build_test_set.py`, `build_fresh_test_set.py` | The development set, and the fresh test sets (excluding every puzzle found in the project's files, their games and solution positions) |
| `build_pool.py`, `build_collect_pool.py` | Theme- and rating-balanced training pools (test puzzles excluded by id and by position) |
| `perception.py` | Board description for prompts (diagram, pieces, legal moves; optional tactics) |
| `run_pilot.py` | Runs the teacher with a chosen prompt and effort, grades answers and lines, resumes where it stopped |
| `line_facts.py`, `claim_check.py` | Machine-computed facts about a line; a tool-based checker for the claims in a text |
| `write_traces.py`, `package_sft.py`, `make_path_data.py`, `make_answer_data.py` | Teacher explanations, filtering, the two students' training sets, and answer-only data from any pool (the 200k set) |
| `train_sft.py`, `eval_student.py`, `rl_grpo.py`, `rewards.py` | Student training (LoRA or full fine-tune), evaluation, and RL; `rewards.py` holds reward v4 |
| `puzzle_rating.py`, `analyze_pilot.py` | Puzzle ratings with confidence intervals; paired McNemar tests |
| `stockfish_grade.py` | The Stockfish rule for "equally good alternative" (used by the audits in `experiments/`) |

Report scripts in `experiments/` (each writes `results/<name>.md`, copied to `reports/` by `export_reports.py`):
`teacher_report.py`, `path_A_report.py`, `path_AB_report.py`, `rl_long_report.py`, `rl_truth_report.py`,
`fresh_test_report.py`, `overnight4_report.py` (answer-first, sampling, RL on the fresh set), `tighten_report.py`,
`final_checks_report.py`, `replication_report.py`; Stockfish checks `stockfish_audit.py`, `reply_quality.py`,
`line_soundness.py`; `stockfish_baseline.py` (Stockfish as a reference player).

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # plus Stockfish 16: apt install stockfish
.venv/bin/python -m pytest tests                  # fast checks, no model needed (22 tests)
.venv/bin/python experiments/make_figures.py      # redraws every figure from reports/outputs/
.venv/bin/python experiments/check_overlap.py     # development set vs every training puzzle (~30 s)
.venv/bin/python experiments/check_overlap.py --test fresh_test_set.jsonl     # same for each fresh set
```

The full runs need the Lichess database and a GPU:

```bash
curl -LO https://database.lichess.org/lichess_db_puzzle.csv.zst && mkdir -p data \
  && zstd -d lichess_db_puzzle.csv.zst -o data/lichess_db_puzzle.csv
# serve gpt-oss-120b with vLLM on localhost:8000, then e.g. the teacher baseline:
.venv/bin/python src/run_pilot.py --run test500_med --formats P1L --puzzles test_set.jsonl --effort medium --max-tokens 32768
# training data: src/build_collect_pool.py (puzzle pools), src/write_traces.py + src/package_sft.py (teacher
# explanations, needs the teacher), src/make_path_data.py (the two students' sets),
# src/make_answer_data.py --pool pool_collect2.jsonl --prompt p1l (the 200k answer-only set)
# student training and RL run inside the vLLM container (requirements-train.txt), e.g.:
# (FlashAttention 2 is the default; --no-grad-ckpt only with the teacher stopped)
python src/train_sft.py --data results/sft/path_A.jsonl --out ckpt/path_A --full --epochs 2 --lr 1e-5 \
  --batch-tokens 8192 --accum 4 --no-grad-ckpt --save-epochs [--seed 1]
python src/train_sft.py --data results/sft/ans_p1l_200k.jsonl --model ckpt/path_A --out ckpt/path_A200k --full \
  --epochs 1 --lr 1e-5 --batch-tokens 8192 --accum 4 --no-grad-ckpt
python src/rl_grpo.py --model ckpt/path_B --puzzles pool_collect1.jsonl --skip 40800 --out ckpt/rlT4_B \
  --max-steps 390 --lr 2e-6 --save-every 130 --reward truth4 [--seed 1]
# evaluation: serve the student with vLLM on :8001 (util 0.30), then
.venv/bin/python src/eval_student.py --model ckpt/path_A --served-name path_A --tag path_A --puzzles fresh_test_set.jsonl \
  --out results/fresh_student_path_A.jsonl --max-tokens 1024 --temperature 0 --concurrency 32
.venv/bin/python experiments/stockfish_baseline.py --puzzles fresh_test_set.jsonl --out results/fresh_stockfish.jsonl
```

B answer-first's training set (`results/sft/path_B_answerfirst.jsonl`) is B's set with each target reordered: A's
exact target (`FINAL_LINE` + `FINAL_MOVE`), a blank line, then B's explanation. It was built with a one-off command
that isn't in the repo; the recipe above is enough to rebuild it from `path_A.jsonl` and `path_B.jsonl`.

**Exact inputs.** Lichess updates its puzzle database every month, so a new download differs from ours. The committed
puzzle files and `splits/train_puzzles.jsonl.gz` pin the puzzles that were actually used. Our snapshot was downloaded
on 2026-09-23: 6,100,952 puzzles, `lichess_db_puzzle.csv.zst` sha256
`95fd454bec9efe8f940d5863d5db4c57474f281a865834997bd8cb5d6a149bb9` (the unpacked CSV:
`b7e1661b4b87fddb2afff1323ba872c08a7f700f2e359d9a909feee42e965640`). Model revisions on Hugging Face:
`Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, `openai/gpt-oss-120b` at
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.

**The fresh test sets.** `src/build_fresh_test_set.py` (defaults: seed 20260929, `fresh_test_set.jsonl`; the second
set used `--seed 20261001 --out fresh_test_set_2.jsonl`) excludes every puzzle id found in any `.jsonl` file of the
project at build time. Re-running it now gives different sets, because the exclusion would then cover these puzzles
too: the committed files are the sets (sha256 `312e1e56…` and `a86bd35a…`; full hashes in the protocols in
`reports/`).

**The run scripts are records, not tools.** `results/run_*.sh` are the exact chains that produced each result, kept as a
record. They assume our machine (paths, a `trainenv` training container, a `student` serving container, offline model
caches) and some contain one-off settings such as a deadline for that night's run, so use them as a reference for the
steps and settings rather than running them as they are. The latest ones: `run_scale_v4.sh` (A + 200k, reward v4),
`run_fresh_test.sh` (fresh test 1), `run_overnight4.sh` (answer-first, sampling, RL on the fresh set), `run_seed2.sh`
(A's second seed), `run_tighten.sh` (five samples, B's second seed, Stockfish checks), `run_final_checks.sh`
(untrained model, teacher resampled, line soundness), `run_replication.sh` (second seeds of A + 200k, answer-first and
v4; fresh test 2).

On the ASUS Ascent GX10 (NVIDIA GB10, the DGX Spark design; 128 GB unified memory) we serve the teacher with NVIDIA's
`nvcr.io/nvidia/vllm:26.05-py3` image (vLLM 0.20.1), the Marlin MoE backend and `--gpu-memory-utilization 0.70`.
Training uses the same image with the packages in `requirements-train.txt`. The teacher and training never run at the
same time (memory).
