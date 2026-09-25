# Chess distillation

> Work in progress. Teacher evaluation, data-collection experiments and the first student pilots are done. The main
> experiment is next: students that imitate the teacher's reasoning versus students trained on answers only, compared
> before and after RL.

Can a small language model (1 to 3B parameters), trained on a single desktop GPU, beat a 120B model at chess tactics?
And does learning the big model's reasoning help it, either directly or after reinforcement learning?

The teacher is [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), served locally with vLLM on an NVIDIA
DGX Spark. Everything is checked automatically, with [python-chess](https://python-chess.readthedocs.io/) for the rules
and [Stockfish](https://stockfishchess.org/) for move quality. Puzzles come from the
[Lichess puzzle database](https://database.lichess.org/#puzzles), and 500 rating-stratified puzzles (rated 800 to 2200)
are held out as the test set.

## Findings so far

### 1. The teacher's main weakness is perception

Given only the position (FEN), gpt-oss-120b plays an illegal move 22% of the time. Adding a python-chess description of
the board (a diagram, the pieces and the legal moves) removes the illegal moves and raises its score. With it, the
teacher solves 44.4% of the 500 test puzzles (puzzle rating 1416) in one low-effort attempt. Thinking harder has
diminishing returns: medium effort scores about 49% on a fresh pool, and high effort often runs out of budget. Retrying
helps more.

### 2. A small student already beats the teacher with answers only

We trained Qwen3-1.7B (LoRA) on Lichess puzzles with just the move and its line, no reasoning text, and gave it the same
board description in its prompt:

| Student training data (same prompt, same 500 test puzzles) | Solved | Puzzle rating |
|---|---|---|
| gpt-oss-120b (teacher), low effort, one attempt | 44.4% | 1416 |
| answers only, 1.9k puzzles | 48.4% | 1474 |
| answers only, 5.6k puzzles | 49.6% | 1491 |
| **answers only, 21.6k puzzles** | **51.2%** | **1514** (beats the teacher, paired p = 0.011) |

### 3. Imitating the teacher's reasoning made students worse at these sizes

We followed the recipe of [Master Distillation](https://arxiv.org/abs/2603.20510), where the teacher writes an
explanation "as if discovering" the engine's best line. We added machine-checked facts to the teacher's prompt and
filtered out explanations with false claims. A student trained on those explanations solved 43.4%, against 49.6% for
answers only on the same 5.6k puzzles (p = 0.008). It copies the teacher's confident style but can't follow the board:
most of its own explanations contain an illegal or impossible move, even when its answer is right. Explanations built
by code and board-tracking practice didn't raise accuracy either, though board-tracking practice does make the
student's lines legal more often.

### 4. LLM judges can't grade chess explanations reliably

Even when given the facts, an LLM judge agreed with careful human-style ratings only about 55% of the time. Preventing
errors by putting facts in the writer's prompt worked better than trying to catch them afterwards.

## Next: the main experiment

1. Collect about 40k fact-grounded teacher explanations (the paper's scale), plus answers-only data for the same
   puzzles.
2. Train two students the same way (full fine-tune), one on answers only and one on the teacher's explanations.
3. Give both the same RL, with the correct move as the reward, and see whether the reasoning-trained student overtakes
   the other after practice.
4. Measure accuracy, and also whether the students' reasoning is real (true claims, legal lines).
5. Add a stronger baseline: the teacher at medium effort on the test set.

## How it works

```
Lichess puzzles ─► test set (500, held out) + balanced training pools (60k / 200k)
      │
      ├─► teacher (gpt-oss-120b) ─► baseline scores, and explanations of the engine's best line
      │        (python-chess facts in the prompt; claim checker filters false statements)
      │
      ├─► answers-only data (move + line) straight from Lichess
      ▼
students (Qwen3-1.7B) ─► imitation (SFT) ─► RL with a verifiable reward ─► test on the 500
```

| File | What it does |
|---|---|
| `build_pilot_set.py`, `build_pool.py`, `build_collect_pool.py`, `make_subset.py` | Puzzle sets: the test set, and theme- and rating-balanced training pools (test puzzles excluded by id and by position) |
| `perception.py` | Board description for prompts (diagram, pieces, legal moves; optional tactics) |
| `run_pilot.py` | Runs the teacher with a chosen prompt format and effort, grades the answers, resumes where it stopped |
| `line_facts.py`, `claim_check.py` | Machine-computed facts about a line; a tool-based checker for the claims in a text |
| `write_traces.py`, `package_sft.py` | Teacher explanations ("as if discovering" + facts) and filtering into training data |
| `code_trace.py`, `aux_tasks.py`, `make_answer_data.py` | Code-built explanations, board-tracking tasks, answers-only data |
| `train_sft.py`, `eval_student.py`, `rl_grpo.py` | Student training (LoRA or full), evaluation, RL (draft) |
| `engine_analysis.py`, `verify_trace.py`, `judge.py`, `judge_eval.py` | Stockfish analysis, step verification, the LLM judge and its calibration |
| `puzzle_rating.py`, `analyze_pilot.py`, `night2_summary.py` | Puzzle ratings with confidence intervals, paired comparisons, reports |
| `pilot_set.jsonl` | The held-out 500-puzzle test set |

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install chess openai   # Stockfish: apt install stockfish
curl -LO https://database.lichess.org/lichess_db_puzzle.csv.zst && mkdir -p data \
  && zstd -d lichess_db_puzzle.csv.zst -o data/lichess_db_puzzle.csv
# serve gpt-oss-120b with vLLM on localhost:8000, then e.g.:
.venv/bin/python run_pilot.py --run test500 --formats P1L --puzzles pilot_set.jsonl --effort low
```

On the DGX Spark (GB10, 128 GB unified memory) we serve the teacher with NVIDIA's `nvcr.io/nvidia/vllm:26.05-py3`
image (vLLM 0.20.1), the Marlin MoE backend and `--gpu-memory-utilization 0.70`. Student training runs in the same
image with `peft` added.
