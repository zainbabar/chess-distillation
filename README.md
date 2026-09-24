# Chess distillation

> **Work in progress.** Teacher evaluation and data collection are underway; student training hasn't started yet.

**Question:** can a small language model (1–3B parameters), trained on a single desktop GPU, beat a 120B model at chess tactics by learning to *see the board* before it reasons?

The teacher is [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), served locally with vLLM on an NVIDIA DGX Spark. Every answer is checked automatically: [python-chess](https://python-chess.readthedocs.io/) for the rules and [Stockfish](https://stockfishchess.org/) for move quality. Puzzles come from the [Lichess puzzle database](https://database.lichess.org/#puzzles).

## Findings so far

**1. The teacher's main weakness is perception, not chess knowledge.** Given only the position (FEN), gpt-oss-120b plays an illegal move 22% of the time: it moves pieces from empty squares, moves the opponent's pieces, and slides through blockers. Giving it a python-chess-generated description of the board removes illegal moves entirely and cuts tokens by ~25%.

| Prompt (low reasoning effort) | Puzzles | Solved | Illegal moves | Puzzle rating* |
|---|---|---|---|---|
| FEN only | 500 | 24% | 22% | 1114 |
| + list of legal moves | 500 | 31% | 9% | 1223 |
| + board diagram, piece lists, readable moves | 140 | 42% | 0% | 1383 |
| + the same, asking for the full solution line | 140 | 45% | 0% | 1425 |

\*Elo fitted to which rated puzzles were solved. 95% intervals are roughly ±60 on 500 puzzles and ±130 on 140.

**2. Thinking harder has sharply diminishing returns.** Medium effort (one attempt) solves 49% of 504 fresh puzzles, at ~8× the tokens of low effort. High effort mostly spends its budget checking squares one by one and often runs out: 7 of 10 answers were cut off at 32k tokens. Retrying is cheaper than thinking longer: low effort's best-of-2 reaches 56%.

**3. Stockfish never rescued a "wrong" answer.** Of 567 wrong answers, none was an equally good alternative move; 556 were outright blunders.

**4. Engine-grounded traces look promising.** Instead of making the teacher *solve* the puzzle, Stockfish provides the verified analysis and the teacher *writes it up* as reasoning. In a 98-puzzle pilot:
- every final move was correct
- 80% of traces passed step-by-step verification
- 74 of 98 had the full solution line matching Lichess
- each trace cost ~425 tokens, vs ~9k for solving at medium effort

The alternative, telling the teacher only the correct move and asking it to explain, failed: its continuations matched the real solution in 0 of 40 cases.

## How it works

```
Lichess puzzles ─► pilot/test sets (fixed seeds, 7 rating bands 800–2200)
                        │
                        ▼
      prompts (FEN + python-chess perception) ─► gpt-oss-120b (vLLM)
                        │
                        ▼
   grading: python-chess (legal? correct?) + Stockfish (how good?)
   step verification: every candidate line legal, every verdict matches Stockfish
                        │
                        ▼
        verified reasoning traces ─► (next) student training
```

| File | What it does |
|---|---|
| `build_pilot_set.py`, `build_pool.py`, `make_subset.py` | Build puzzle sets from the Lichess CSV (applies the opponent's setup move) |
| `perception.py` | Board perception text (diagram, pieces, legal moves; optional tactics) — tested by `test_perception.py` |
| `run_pilot.py` | Async runner: prompt formats, reasoning effort, grading, resumable output |
| `engine_analysis.py` | Stockfish top-3 lines for engine-grounded traces |
| `structured.py`, `verify_trace.py` | Checkable write-up format and step-by-step verifier |
| `puzzle_rating.py` | Puzzle rating (Elo) with bootstrap confidence intervals |
| `stockfish_grade.py` | Re-grades wrong answers with Stockfish |
| `*_report.py` | Summary reports |
| `pilot_set.jsonl` | The held-out 500-puzzle **test set** (never used for training) |

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install chess openai   # Stockfish: apt install stockfish
curl -LO https://database.lichess.org/lichess_db_puzzle.csv.zst && mkdir -p data \
  && zstd -d lichess_db_puzzle.csv.zst -o data/lichess_db_puzzle.csv
# serve gpt-oss-120b with vLLM on localhost:8000, then e.g.:
.venv/bin/python run_pilot.py --run pilot --formats B P1 --effort low
```

Serving notes for the DGX Spark (GB10, 128 GB unified memory): use NVIDIA's `nvcr.io/nvidia/vllm:26.05-py3` image (vLLM 0.20.1), the Marlin MoE backend, and `--gpu-memory-utilization 0.70`. The newer `26.05.post1` image froze the machine while loading the model.

## Next

- Pick the data recipe: engine-grounded traces vs. medium-effort solver traces (or both)
- Test candidate student models (1–3B) zero-shot to choose a base
- Train the student: perception pre-training → distillation → RL with verifiable rewards
- Baselines: an answer-only student, and the teacher prompted to "describe the board first"
