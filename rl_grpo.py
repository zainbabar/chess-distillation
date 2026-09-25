"""RL after SFT (GRPO, binary reward from python-chess) — DRAFT for the student stage (written 2026-09-25 night,
smoke-test before real use).

Master Distillation found RL on top of SFT added ~7 points and that a *binary* reward (right move = 1) beat partial
credit. Reward here: 1.0 if FINAL_MOVE is the Lichess answer (any mate accepted on mate-in-1 puzzles), else 0;
-0.1 extra if there is no parsable FINAL_MOVE (keeps the format). The prompt is the same one the student was
SFT-trained on (P1L or compact).

Run inside `trainenv` (TRL 1.14 installed), starting from an SFT checkpoint (a merged model dir or base + --adapter):
  docker exec trainenv python -u rl_grpo.py --model Qwen/Qwen3-1.7B --adapter ckpt/night2_scale \
      --puzzles pool_collect2.jsonl --skip 100000 --n 4000 --out ckpt/rl_test --max-steps 20
"""
import argparse
import json
import sys

sys.path.insert(0, "/work")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None, help="LoRA SFT adapter to merge first")
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N puzzles (used for SFT)")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--prompt", default="p1l", choices=["p1l", "compact"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.001)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from make_answer_data import compact_prompt
    from run_pilot import build_prompt, grade

    pz = []
    for i, line in enumerate(open(a.puzzles)):
        if i < a.skip:
            continue
        pz.append(json.loads(line))
        if len(pz) >= a.n:
            break
    by_id = {p["puzzle_id"]: p for p in pz}
    mk = (lambda p: build_prompt(p, "P1L")) if a.prompt == "p1l" else compact_prompt
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": mk(p)}], "puzzle_id": p["puzzle_id"]} for p in pz])

    def reward(prompts, completions, puzzle_id, **kw):
        out = []
        for comp, pid in zip(completions, puzzle_id):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            status, _, _ = grade(by_id[pid], text)
            out.append(1.0 if status == "correct" else (-0.1 if status == "parse_fail" else 0.0))
        return out

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    cfg = GRPOConfig(output_dir=a.out, learning_rate=a.lr, beta=a.beta, num_generations=a.generations,
                     per_device_train_batch_size=a.generations, gradient_accumulation_steps=4,
                     max_completion_length=256, temperature=1.0, bf16=True, logging_steps=5,
                     save_steps=200, max_steps=a.max_steps, report_to=[],
                     chat_template_kwargs={"enable_thinking": False})
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[reward], args=cfg, train_dataset=ds)
    trainer.train()
    trainer.save_model(a.out)


if __name__ == "__main__":
    main()
