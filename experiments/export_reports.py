"""Copy the reports behind the README's numbers, and the graded outputs they are computed from, into reports/
(the committed, public copy; results/ stays local). Re-run after regenerating any report.

Outputs are gzipped JSONL, exactly one record per test puzzle (checked here; a retried puzzle keeps its last record):
the prompt, the model's full response, the parsed move and line, and the grading (status, line_legal,
full_line_correct, ...). File names match the paths the reports cite (results/<name>.jsonl ->
reports/outputs/<name>.jsonl.gz).
"""
import gzip
import json
import shutil
from pathlib import Path

REPORTS = {  # public name <- local report
    "teacher_baseline.md": "results/teacher_report.md",
    "students_pilot_1.9k.md": "results/pilot_report.md",
    "students_5.6k_21.6k.md": "results/night2/results.md",
    "student_A.md": "results/path_A_report.md",
    "students_A_vs_B.md": "results/path_AB_report.md",
    "rl_answer_only_reward.md": "results/rl_long_report.md",
    "rl_four_rewards_B.md": "results/rl_truth3_report.md",
}
OUTPUTS = ["pilot_formatA", "pilot_formatB", "test500_formatP1L", "test500_med_formatP1L"] + [f"student_{t}_nothink" for t in (
    "pilot_p_answer", "pilot_p_code", "pilot_p_llm",
    "pilot_night2_answer", "pilot_night2_llm", "pilot_night2_aux", "pilot_night2_code", "pilot_night2_scale",
    "path_A", "path_A_ep1", "path_B", "path_B_ep1",
    *(f"{run}_{ck}" for run in ("rlL_A", "rlL_B", "rlT_B", "rlT2_B", "rlT3_B") for ck in ("step130", "step260", "final")))]


def main():
    out = Path("reports")
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    for name, src in REPORTS.items():
        shutil.copyfile(src, out / name)
    test_ids = {json.loads(line)["puzzle_id"] for line in open("test_set.jsonl")}
    total = 0
    for name in OUTPUTS:
        dst = out / "outputs" / f"{name}.jsonl.gz"
        # one record per test puzzle: a retried puzzle keeps its last record, as the report scripts read it
        last = {}
        for line in open(f"results/{name}.jsonl", "rb"):
            last[json.loads(line)["puzzle_id"]] = line
        assert set(last) == test_ids, f"{name}: {len(last)} puzzles, not the 500 test puzzles"
        assert not any(json.loads(v).get("status") == "error" for v in last.values()), f"{name}: unresolved errors"
        src = b"".join(last.values())
        # leave the file alone when its content hasn't changed: gzip headers differ between writers (git churn)
        if not (dst.exists() and gzip.decompress(dst.read_bytes()) == src):
            with gzip.GzipFile(dst, "wb", compresslevel=9, mtime=0) as g:
                g.write(src)
        total += dst.stat().st_size
    print(f"reports/: {len(REPORTS)} reports, {len(OUTPUTS)} output files ({total / 1e6:.1f} MB compressed)")


if __name__ == "__main__":
    main()
