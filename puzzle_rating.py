"""Puzzle rating of a solver: the Elo rating that best explains which rated puzzles it solved.

Model (Elo): P(solve puzzle i) = 1 / (1 + 10^((r_i - R) / 400)). R is the maximum-likelihood fit
(Newton's method; the log-likelihood is concave), with a bootstrap 95% interval (resampling puzzles).
Headline use: "student rated 1650 vs teacher 1300". Only compare ratings fitted on puzzles from the
same rating range (all our sets are balanced over 800-2200).

Usage: .venv/bin/python puzzle_rating.py results/pilot_formatB.jsonl results/perc_low_formatP1.jsonl ...
"""
import json
import math
import random
import sys

K = math.log(10) / 400


def fit(ratings, solved, iters=50):
    """MLE rating R. Returns +-inf-ish bounds (clamped to 0..4000) if all/none solved."""
    n_ok = sum(solved)
    if n_ok == 0:
        return 0.0
    if n_ok == len(solved):
        return 4000.0
    R = sum(ratings) / len(ratings)
    for _ in range(iters):
        g = h = 0.0
        for r, y in zip(ratings, solved):
            p = 1 / (1 + math.exp(-K * (R - r)))
            g += y - p
            h += p * (1 - p)
        step = g / (K * h)
        R += max(-400, min(400, step))
        if abs(step) < 0.01:
            break
    return max(0.0, min(4000.0, R))


def rate(records, n_boot=1000, seed=0):
    """records: dicts with 'rating' and 'status' (correct = solved; everything else = not solved)."""
    ratings = [r["rating"] for r in records]
    solved = [1 if r["status"] == "correct" else 0 for r in records]
    R = fit(ratings, solved)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(ratings)) for _ in ratings]
        boots.append(fit([ratings[i] for i in idx], [solved[i] for i in idx]))
    boots.sort()
    return {"rating": round(R), "ci_low": round(boots[int(0.025 * n_boot)]),
            "ci_high": round(boots[int(0.975 * n_boot) - 1]), "n": len(records), "solved": sum(solved)}


def main():
    print("| Results file | Puzzles | Solved | Puzzle rating | 95% CI |\n|---|---|---|---|---|")
    for path in sys.argv[1:]:
        recs = [json.loads(l) for l in open(path)]
        r = rate(recs)
        print(f"| {path} | {r['n']} | {r['solved']} | **{r['rating']}** | {r['ci_low']}–{r['ci_high']} |")


if __name__ == "__main__":
    main()
