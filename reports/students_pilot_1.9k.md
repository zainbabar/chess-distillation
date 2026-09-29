# Student pilot report

Qwen3-1.7B + LoRA r64, 3 epochs, the same 1,892 training puzzles (pool_pilot_llm.jsonl, 800–2200), differing only in the training text. Test: test_set.jsonl (500 puzzles, held out), greedy decoding, the teacher's P1L prompt. Zero-shot baseline: 1/140.

| Arm | n | Correct | Illegal | No answer | Exact format | Legal line | Full line right | Mean tokens | Puzzle rating (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| answer only | 500 | **242 (48.4%)** | 0 | 0 | 500 | 179 | 61 | 25 | 1474 (1408–1539) |
| code-built explanation | 500 | **247 (49.4%)** | 0 | 0 | 500 | 127 | 54 | 111 | 1488 (1427–1551) |
| LLM (FDF-low) explanation | 500 | **206 (41.2%)** | 12 | 0 | 500 | 96 | 51 | 176 | 1370 (1303–1433) |

Per band (correct of ~71):

| Arm | 800-1000 | 1000-1200 | 1200-1400 | 1400-1600 | 1600-1800 | 1800-2000 | 2000-2200 |
|---|---|---|---|---|---|---|---|
| answer only | 59 | 46 | 39 | 27 | 29 | 22 | 20 |
| code-built explanation | 53 | 48 | 40 | 33 | 27 | 27 | 19 |
| LLM (FDF-low) explanation | 51 | 36 | 35 | 26 | 21 | 20 | 17 |

Paired comparisons (same 500 puzzles; exact McNemar):

- answer only vs code-built explanation: only the first right on 57, only the second on 62 (p = 0.714)
- answer only vs LLM (FDF-low) explanation: only the first right on 82, only the second on 46 (p = 0.002)
- code-built explanation vs LLM (FDF-low) explanation: only the first right on 95, only the second on 54 (p = 0.001)

Reference: the teacher gpt-oss-120b, low effort, P1L prompt: 45% on the 140-puzzle subset (rating ~1425); FEN + legal moves on these 500: 31.2%.