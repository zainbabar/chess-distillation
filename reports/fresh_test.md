# Fresh test: 500 puzzles no model or experiment had seen

Puzzles: `fresh_test_set.jsonl`. Protocol, fixed before the run: `reports/fresh_test_protocol.md`. Same prompt,
grader and settings as the main evaluation. "First move right" is the main score.

| Model | First move right | Puzzle rating (95% CI) | Illegal | No readable answer | Full line right | Output tokens (mean) |
|---|---|---|---|---|---|---|
| student A: answers only | **280/500 (56.0%)** | 1582 (1520–1640) | 0 | 0 | 96 | 27 |
| student B: teacher explanations | **233/500 (46.6%)** | 1447 (1384–1510) | 2 | 0 | 67 | 180 |
| student A + 200k more puzzles | **314/500 (62.8%)** | 1682 (1618–1738) | 0 | 0 | 117 | 26 |
| student B + RL, reward v4 | **248/500 (49.6%)** | 1490 (1426–1557) | 0 | 0 | 51 | 195 |
| untrained Qwen3-1.7B | **6/500 (1.2%)** | 419 (211–545) | 7 | 232 | 1 | 645 |
| teacher gpt-oss-120b, low effort | **202/500 (40.4%)** | 1359 (1293–1425) | 10 | 0 | 51 | 1,212 |
| teacher gpt-oss-120b, medium effort | **264/500 (52.8%)** | 1536 (1476–1595) | 11 | 0 | 114 | 8,294 |
| Stockfish 16, 0.1 s | **495/500 (99.0%)** | 2620 (2494–2901) | 0 | 0 | – | – |
| random legal move (expected) | 5.3% | | | | | |

Primary comparisons (fixed in the protocol; exact McNemar, Holm-adjusted over these three; the gap is
the first model's score minus the second's, in points, with a 95% paired bootstrap interval):

- student A: answers only vs student B: teacher explanations: 87 vs 40, p = 3.7e-05; gap +9.4 points (95% interval +5.0 to +13.8); Holm-adjusted p = 7.4e-05
- student A: answers only vs teacher gpt-oss-120b, low effort: 122 vs 44, p = 1.1e-09; gap +15.6 points (95% interval +10.6 to +20.6); Holm-adjusted p = 3.4e-09
- student A: answers only vs teacher gpt-oss-120b, medium effort: 88 vs 72, p = 0.24; gap +3.2 points (95% interval -1.8 to +8.2); Holm-adjusted p = 0.24

Secondary comparisons (not corrected):

- student A + 200k more puzzles vs teacher gpt-oss-120b, medium effort: 111 vs 61, p = 0.00017; gap +10.0 points (95% interval +5.0 to +15.0)
- student A + 200k more puzzles vs teacher gpt-oss-120b, low effort: 153 vs 41, p = 2.1e-16; gap +22.4 points (95% interval +17.2 to +27.6)
- student A + 200k more puzzles vs student A: answers only: 61 vs 27, p = 0.00037; gap +6.8 points (95% interval +3.2 to +10.4)
- student B + RL, reward v4 vs student B: teacher explanations: 78 vs 63, p = 0.24; gap +3.0 points (95% interval -1.6 to +7.6)
- student B: teacher explanations vs teacher gpt-oss-120b, low effort: 100 vs 69, p = 0.021; gap +6.2 points (95% interval +1.2 to +11.4)
- student B: teacher explanations vs teacher gpt-oss-120b, medium effort: 68 vs 99, p = 0.02; gap -6.2 points (95% interval -11.4 to -1.2)
- teacher gpt-oss-120b, medium effort vs teacher gpt-oss-120b, low effort: 109 vs 47, p = 7.5e-07; gap +12.4 points (95% interval +7.6 to +17.2)

Development set (the original 500) vs fresh test, first move right:

| Model | Development set | Fresh test |
|---|---|---|
| student A: answers only | 56.8% | 56.0% |
| student B: teacher explanations | 50.0% | 46.6% |
| student A + 200k more puzzles | 63.8% | 62.8% |
| student B + RL, reward v4 | 52.8% | 49.6% |
| teacher gpt-oss-120b, low effort | 44.4% | 40.4% |
| teacher gpt-oss-120b, medium effort | 53.0% | 52.8% |
