"""RL after SFT: GRPO with a binary reward from python-chess (THE PATH step 3).

Reward (--reward move, default): 1.0 if FINAL_MOVE is the Lichess answer (any mate accepted on the last move), 0
otherwise, -0.1 if there is no parsable FINAL_MOVE. Master Distillation found a binary reward beat partial credit.
--reward truth (for explanation students): +1 right move; + 0.5 x (fraction of the Lichess line written correctly in
FINAL_LINE) when the move is right; -0.5 if the claim checker finds a hard false claim in the explanation (piece not on
that square, false "wins the X", false mate); -0.5 if the explanation is under 25 words or FINAL_MOVE is missing
(so dodging the checker doesn't pay). The prompt is the P1L prompt the students were SFT-trained and evaluated on.
--reward truth2 (stricter, after truth was gamed: vaguer text + padded lines): +1 right move; + 0.5 x correct prefix /
max(line length, solution length) when the move is right (padding dilutes the credit; attempting the full line never
scores below a 1-move line); -0.5 hard false claim; -0.5 if the text names a move that is illegal in every position
along the real solution or has a wrong +/# (invented continuations); -0.5 if the text has fewer than 2 such verified
moves (min(2, solution length)), is under 40 words, or FINAL_MOVE is missing.
--reward truth3 (v2 made the model stop calculating: wrong continuations cost more than right ones earned): like truth2
but line credit doubled (+1.0 x correct prefix / max(line length, solution length)), invented-move penalty halved (-0.25),
and the floor counts DISTINCT verified moves (repeating one move twice no longer passes).

Starts from a full SFT checkpoint (ckpt/path_A, ckpt/path_B). The policy keeps fp32 master weights with bf16 autocast
(pure-bf16 weights would round away updates at lr ~1e-6). Samples come from vLLM running inside the trainer
("colocate"); TRL syncs the weights into vLLM every step. No KL term (beta 0) and the DAPO loss (TRL defaults).
Each optimizer step: --prompts-per-step fresh puzzles × --generations samples each; every puzzle is used once.

Run inside `trainenv`:
  docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trainenv python -u rl_grpo.py \
      --model ckpt/path_A --puzzles pool_collect1.jsonl --skip 40000 --out ckpt/rl_A --max-steps 100
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "/work/src")  # run inside the trainenv container (repo mounted at /work)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="full SFT checkpoint dir")
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N puzzles (used for SFT)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--micro", type=int, default=8, help="completions per forward/backward micro-batch")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--max-completion", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-vllm", action="store_true", help="generate with HF generate instead of colocated vLLM")
    ap.add_argument("--vllm-util", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0, help="also save <out>/step<N> (bf16) every N steps")
    ap.add_argument("--reward", default="move", choices=["move", "truth", "truth2", "truth3"])
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from claim_check import SAN_RE, check
    from run_pilot import build_prompt, grade, grade_line

    n = a.max_steps * a.prompts_per_step
    pz = []
    for i, line in enumerate(open(a.puzzles)):
        if i < a.skip:
            continue
        pz.append(json.loads(line))
        if len(pz) >= n:
            break
    by_id = {p["puzzle_id"]: p for p in pz}
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": build_prompt(p, "P1L")}],
                             "puzzle_id": p["puzzle_id"]} for p in pz])
    print(f"{len(pz)} RL puzzles (rows {a.skip + 1}-{a.skip + len(pz)} of {a.puzzles})", flush=True)

    stats = {"n": 0, "correct": 0, "parse_fail": 0, "claim_err": 0, "short": 0, "line_frac": 0.0, "move_flag": 0}

    def reward(prompts, completions, puzzle_id, **kw):
        out = []
        for comp, pid in zip(completions, puzzle_id):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            p = by_id[pid]
            status, _, _ = grade(p, text)
            stats["n"] += 1
            stats["correct"] += status == "correct"
            stats["parse_fail"] += status == "parse_fail"
            if a.reward == "move":
                out.append(1.0 if status == "correct" else (-0.1 if status == "parse_fail" else 0.0))
                continue
            body = text.split("FINAL_LINE")[0].strip()
            r = 0.0
            if a.reward in ("truth2", "truth3"):
                w_line, w_flag = (1.0, 0.25) if a.reward == "truth3" else (0.5, 0.5)
                g = grade_line(p, text)
                c = check(body, p)
                nflag = sum(1 for x in c["soft"] if x[0] == "move")
                if status == "correct":
                    frac = g["line_match_len"] / max(1, len(g["line_moves"] or []), g["solution_len"])
                    stats["line_frac"] += frac
                    r += 1.0 + w_line * frac
                if c["errors"]:
                    stats["claim_err"] += 1
                    r -= 0.5
                if nflag:
                    stats["move_flag"] += 1
                    r -= w_flag
                sans = SAN_RE.findall(body)
                n_ok = (len(set(sans)) - len({x[1] for x in c["soft"] if x[0] == "move"})) if a.reward == "truth3" \
                    else len(sans) - nflag
                if (n_ok < min(2, g["solution_len"]) or len(body.split()) < 40
                        or status == "parse_fail"):
                    stats["short"] += 1
                    r -= 0.5
                out.append(r)
                continue
            if status == "correct":
                g = grade_line(p, text)
                frac = min(1.0, g["line_match_len"] / max(1, g["solution_len"]))
                stats["line_frac"] += frac
                r += 1.0 + 0.5 * frac
            if check(body, p)["errors"]:
                stats["claim_err"] += 1
                r -= 0.5
            if len(body.split()) < 25 or status == "parse_fail":
                stats["short"] += 1
                r -= 0.5
            out.append(r)
        return out

    def save_bf16(model, path):
        sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
        model.save_pretrained(path, state_dict=sd)
        model.config.dtype = torch.bfloat16
        model.config.save_pretrained(path)
        tok.save_pretrained(path)

    class Progress(TrainerCallback):
        def __init__(self):
            self.t0 = time.time()

        def on_step_end(self, args, state, control, model=None, **kw):
            if a.save_every and state.global_step % a.save_every == 0 and state.global_step < state.max_steps:
                save_bf16(model, f"{a.out}/step{state.global_step}")
                print(f"\nsaved {a.out}/step{state.global_step} "
                      f"(sample accuracy so far {stats['correct'] / max(1, stats['n']):.3f})", flush=True)
            if state.global_step % 5 == 0 or state.global_step == state.max_steps:
                el = time.time() - self.t0
                print(f"\nrl step {state.global_step}/{state.max_steps} sample-accuracy so far "
                      f"{stats['correct'] / max(1, stats['n']):.3f} ({stats['n']} samples, {stats['parse_fail']} parse fails) "
                      + (f"claim errors {stats['claim_err'] / max(1, stats['n']):.3f}, short {stats['short']}, "
                         f"move flags {stats['move_flag'] / max(1, stats['n']):.3f}, "
                         f"mean line credit on right {stats['line_frac'] / max(1, stats['correct']):.3f} "
                         if a.reward != "move" else "")
                      + f"{el / 60:.1f} min, {el / max(1, state.global_step):.1f} s/step "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB peak", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="flash_attention_2")
    per_dev = a.micro
    accum = a.prompts_per_step * a.generations // per_dev
    cfg = GRPOConfig(output_dir=a.out, learning_rate=a.lr, beta=a.beta, num_generations=a.generations,
                     per_device_train_batch_size=per_dev, gradient_accumulation_steps=accum,
                     max_completion_length=a.max_completion, temperature=a.temperature, bf16=True,
                     logging_steps=1, save_strategy="no", max_steps=a.max_steps, report_to=[], seed=a.seed,
                     chat_template_kwargs={"enable_thinking": False}, lr_scheduler_type="constant",
                     warmup_steps=0, use_vllm=not a.no_vllm, vllm_mode="colocate",
                     vllm_gpu_memory_utilization=a.vllm_util, shuffle_dataset=False)
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[reward], args=cfg, train_dataset=ds,
                          callbacks=[Progress()])
    t0 = time.time()
    trainer.train()
    save_bf16(trainer.model, a.out)  # bf16 like the SFT checkpoints
    json.dump({**vars(a), "puzzles_used": len(pz), "minutes": round((time.time() - t0) / 60, 1),
               "samples": stats["n"], "sample_accuracy": stats["correct"] / max(1, stats["n"]),
               "parse_fails": stats["parse_fail"], "claim_errors": stats["claim_err"], "short": stats["short"],
               "move_flags": stats["move_flag"]}, open(f"{a.out}/rl_meta.json", "w"), indent=1)
    print(f"saved {a.out}: {stats['n']} samples, sample accuracy {stats['correct'] / max(1, stats['n']):.3f}, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
