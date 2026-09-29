# Code and reproducing

The pipeline, where everything lives, and the commands to rerun it. Results and findings are in the [README](README.md);
the graded outputs behind them are in [`reports/`](reports/).

## How it works

```
Lichess puzzles ─► test set (500, held out) + balanced training pools (60k / 200k)
      │
      ├─► teacher (gpt-oss-120b) ─► baseline scores, and explanations of the Lichess solution
      │        (python-chess facts in the prompt; a claim checker filters false statements)
      │
      ├─► answers-only data (move + line) straight from Lichess
      ▼
students (Qwen3-1.7B) ─► imitation (full fine-tune) ─► RL (GRPO, checked rewards) ─► test on the 500
```

| Folder | What's in it |
|---|---|
| [`src/`](src/) | The pipeline: puzzle sets, the board description, teacher runs and grading, the fact and claim checkers, training data, student training, evaluation, RL and statistics |
| [`experiments/`](experiments/) | Reports and analyses for each experiment, the figures (`make_figures.py`) and the export to `reports/` |
| [`reports/`](reports/) | The reports behind the README's main results, and the graded outputs they come from (all 500 test puzzles) |
| [`figures/`](figures/) | The charts and board pictures in the README |
| [`docs/`](docs/) | The Puzzle Explorer page (built by `experiments/build_explorer.py`, served by GitHub Pages) |
| [`tests/`](tests/) | Fast tests of the grader, the claim checker, the board description and the statistics (run on every push) |
| [`splits/`](splits/) | Every puzzle the published models were trained on (243,872, with moves and source game), to check the test set against: `experiments/check_overlap.py` |
| [`results/`](results/) | The run scripts behind each experiment (`run_*.sh`) |
| `test_set.jsonl` | The 500 test puzzles (held out from training) |

Key files in `src/`:

| File | What it does |
|---|---|
| `build_test_set.py`, `build_pool.py`, `build_collect_pool.py` | The test set, and theme- and rating-balanced training pools (test puzzles excluded by id and by position) |
| `perception.py` | Board description for prompts (diagram, pieces, legal moves; optional tactics) |
| `run_pilot.py` | Runs the teacher with a chosen prompt and effort, grades answers and lines, resumes where it stopped |
| `line_facts.py`, `claim_check.py` | Machine-computed facts about a line; a tool-based checker for the claims in a text |
| `write_traces.py`, `package_sft.py`, `make_path_data.py` | Teacher explanations, filtering, and the two students' training sets |
| `train_sft.py`, `eval_student.py`, `rl_grpo.py` | Student training (LoRA or full fine-tune), evaluation, and RL with the four rewards |
| `puzzle_rating.py`, `analyze_pilot.py` | Puzzle ratings with confidence intervals; paired McNemar tests |

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # plus Stockfish 16: apt install stockfish
.venv/bin/python -m pytest tests                  # fast checks, no model needed
.venv/bin/python experiments/make_figures.py      # redraws every figure from reports/outputs/
.venv/bin/python experiments/check_overlap.py     # test set vs every training puzzle (ids, games, positions; ~30 s)
```

The full runs need the Lichess database and a GPU:

```bash
curl -LO https://database.lichess.org/lichess_db_puzzle.csv.zst && mkdir -p data \
  && zstd -d lichess_db_puzzle.csv.zst -o data/lichess_db_puzzle.csv
# serve gpt-oss-120b with vLLM on localhost:8000, then e.g. the teacher baseline:
.venv/bin/python src/run_pilot.py --run test500_med --formats P1L --puzzles test_set.jsonl --effort medium --max-tokens 32768
# training data: src/build_collect_pool.py (puzzle pools), src/write_traces.py + src/package_sft.py (teacher
# explanations, needs the teacher), src/make_path_data.py (the two students' sets)
# student training and RL run inside the vLLM container (requirements-train.txt), e.g.:
python src/train_sft.py --data results/sft/path_A.jsonl --out ckpt/path_A --full --epochs 2 --lr 1e-5 --batch-tokens 8192
python src/rl_grpo.py --model ckpt/path_B --puzzles pool_collect1.jsonl --skip 40800 --out ckpt/rlT3_B --max-steps 390 --lr 2e-6 --reward truth3
```

**Exact inputs.** Lichess updates its puzzle database every month, so a new download differs from ours; `test_set.jsonl`
and `splits/train_puzzles.jsonl.gz` pin the puzzles that were actually used. Our snapshot was downloaded on 2026-09-23:
6,100,952 puzzles, `lichess_db_puzzle.csv.zst` sha256 `95fd454bec9efe8f940d5863d5db4c57474f281a865834997bd8cb5d6a149bb9`
(the unpacked CSV: `b7e1661b4b87fddb2afff1323ba872c08a7f700f2e359d9a909feee42e965640`). Model revisions on Hugging Face:
`Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, `openai/gpt-oss-120b` at
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.

**The run scripts are records, not tools.** `results/run_*.sh` are the exact chains that produced each result, kept as a
record. They assume our machine (paths, a `trainenv` training container, offline model caches) and some contain one-off
settings such as a deadline for that night's run, so use them as a reference for the steps and settings rather than
running them as they are.

On the ASUS Ascent GX10 (NVIDIA GB10, the DGX Spark design; 128 GB unified memory) we serve the teacher with NVIDIA's
`nvcr.io/nvidia/vllm:26.05-py3` image (vLLM 0.20.1), the Marlin MoE backend and `--gpu-memory-utilization 0.70`.
Training uses the same image with the packages in `requirements-train.txt`. The full chains, including evaluation, are
in `results/run_*.sh`.

