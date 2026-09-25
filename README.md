# Chess distillation

> **Work in progress.** Teacher evaluation, data-collection experiments and first student pilots are done; the
> main experiment (imitation of teacher reasoning vs answers-only, before and after RL) is next.

**Question:** can a small language model (1–3B parameters), trained on a single desktop GPU, beat a 120B model at
chess tactics — and does learning the big model's *reasoning* help it, either directly or after reinforcement learning?

The teacher is [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), served locally with vLLM on an NVIDIA
DGX Spark. Everything is checked automatically: [python-chess](https://python-chess.readthedocs.io/) for the rules,
[Stockfish](https://stockfishchess.org/) for move quality. Puzzles come from the
[Lichess puzzle database](https://database.lichess.org/#puzzles); 500 rating-stratified puzzles (800–2200) are a
held-out test set.

## Findings so far

**1. The teacher's main weakness is perception.** Given only the position (FEN), gpt-oss-120b plays an illegal
move 22% of the time. A python-chess description of the board (diagram, pieces, legal moves) removes illegal moves
and raises its score; with it, the teacher solves **44.4% of the 500 test puzzles (puzzle rating 1416)** in one
low-effort attempt. Thinking harder has diminishing returns (medium ≈ 49% on a fresh pool; high effort often runs
out of budget); retrying helps more.

**2. A small student already beats the teacher — with answers only.** Qwen3-1.7B (LoRA) trained on Lichess puzzles
with just the move and its line, no reasoning text, using the same board description in the prompt:

| Student training data (same prompt, same 500 test puzzles) | Solved | Puzzle rating |
|---|---|---|
| gpt-oss-120b (teacher), low effort, one attempt | 44.4% | 1416 |
| answers only, 1.9k puzzles | 48.4% | 1474 |
| answers only, 5.6k puzzles | 49.6% | 1491 |
| **answers only, 21.6k puzzles** | **51.2%** | **1514** (beats the teacher, paired p = 0.011) |

**3. Imitating the teacher's reasoning made students worse at these sizes.** Following the recipe of
[Master Distillation](https://arxiv.org/abs/2603.20510), the teacher writes an explanation "as if discovering" the
engine's best line; we add machine-checked facts to its prompt and filter out explanations with false claims.
Trained on those explanations, the student solved **43.4%** vs 49.6% for answers-only on the same 5.6k puzzles
(p = 0.008). The student copies the confident style but can't follow the board: most of its own explanations
contain an illegal or impossible move, even when its answer is right. Explanations built by code, and
board-tracking practice, didn't raise accuracy either (board-tracking practice does make its lines more legal).

**4. LLM judges can't grade chess explanations reliably** (even given the facts, ~55% agreement with careful
human-style ratings); preventing errors (facts in the writer's prompt) worked better than detecting them.

## Next (the main experiment)

1. ~40k fact-grounded teacher explanations (the paper's scale) + answers-only data for the same puzzles.
2. Two students trained the same way (full fine-tune): **answers-only** vs **teacher explanations**.
3. The same RL (reward = correct move) on both. Does the reasoning-trained student overtake after practice?
4. Measure accuracy *and* whether the students' reasoning is real (true claims, legal lines).
5. A stronger baseline: the teacher at medium effort on the test set.

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
| `build_pilot_set.py`, `build_pool.py`, `build_collect_pool.py`, `make_subset.py` | Puzzle sets (test set, theme/rating-balanced training pools; test puzzles excluded by id and position) |
| `perception.py` | Board description for prompts (diagram, pieces, legal moves; optional tactics) |
| `run_pilot.py` | Teacher runner: prompt formats, effort, grading, resumable |
| `line_facts.py`, `claim_check.py` | Machine-computed facts about a line; tool-based checker of claims in a text |
| `write_traces.py`, `package_sft.py` | Teacher explanations ("as if discovering" + facts) and filtering into training data |
| `code_trace.py`, `aux_tasks.py`, `make_answer_data.py` | Code-built explanations, board-tracking tasks, answers-only data |
| `train_sft.py`, `eval_student.py`, `rl_grpo.py` | Student training (LoRA/full), evaluation, RL (draft) |
| `engine_analysis.py`, `verify_trace.py`, `judge.py`, `judge_eval.py` | Stockfish analysis, step verification, LLM judge and its calibration |
| `puzzle_rating.py`, `analyze_pilot.py`, `night2_summary.py` | Puzzle rating with CIs, paired comparisons, reports |
| `pilot_set.jsonl` | The held-out 500-puzzle **test set** |

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install chess openai   # Stockfish: apt install stockfish
curl -LO https://database.lichess.org/lichess_db_puzzle.csv.zst && mkdir -p data \
  && zstd -d lichess_db_puzzle.csv.zst -o data/lichess_db_puzzle.csv
# serve gpt-oss-120b with vLLM on localhost:8000, then e.g.:
.venv/bin/python run_pilot.py --run test500 --formats P1L --puzzles pilot_set.jsonl --effort low
```

Serving notes for the DGX Spark (GB10, 128 GB unified memory): NVIDIA's `nvcr.io/nvidia/vllm:26.05-py3` image
(vLLM 0.20.1), the Marlin MoE backend, `--gpu-memory-utilization 0.70`. Student training runs in the same image
with `peft` added.
