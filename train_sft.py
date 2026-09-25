"""Minimal supervised fine-tuning (LoRA or full) of a small student on (prompt, target) pairs.

Runs inside the `trainenv` container (vLLM image + peft):
  docker exec trainenv python train_sft.py --data results/sft_codeA_2k.jsonl --out ckpt/codeA_2k \
      [--model Qwen/Qwen3-1.7B] [--lora-r 64] [--epochs 3] [--lr 2e-4] [--full]
Data: jsonl with "prompt" (the user message, same as the eval prompt) and "target" (the assistant reply).
Loss only on the target tokens. The prompt is rendered with the model's chat template in non-thinking mode,
exactly as vLLM renders it at evaluation time (enable_thinking=False).
"""
import argparse
import json
import math
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def render(tok, prompt, target):
    head = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    end = tok.eos_token if tok.eos_token else ""
    h = tok(head, add_special_tokens=False)["input_ids"]
    t = tok(target.strip() + end, add_special_tokens=False)["input_ids"]
    return h, t


def _parts(model):
    """(decoder, lm_head) for a plain or peft-wrapped causal LM."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.lm_head


def target_loss(model, ids, att, lab):
    """Cross-entropy on target tokens only; the vocab projection is applied only where labels exist
    (a full [batch, seq, vocab] logits tensor would cost several GB)."""
    decoder, head = _parts(model)
    hidden = decoder(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = lab[:, 1:]
    mask = tgt != -100
    h = hidden[:, :-1][mask]
    logits = head(h).float()
    return torch.nn.functional.cross_entropy(logits, tgt[mask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--full", action="store_true", help="full fine-tuning instead of LoRA")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-tokens", type=int, default=16384, help="tokens per micro-batch (padded)")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2", help="flash_attention_2 | sdpa | eager")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (speed tests)")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="faster, more memory (only with the teacher stopped)")
    args = ap.parse_args()
    lr = args.lr or (1e-5 if args.full else 2e-4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.data)]
    data = []
    for r in rows:
        h, t = render(tok, r["prompt"], r["target"])
        ids = (h + t)[: args.max_len]
        labels = ([-100] * len(h) + t)[: args.max_len]
        data.append((ids, labels))
    n_tok = sum(len(d[0]) for d in data)
    n_tgt = sum(sum(1 for x in d[1] if x != -100) for d in data)
    print(f"{len(data)} examples, {n_tok:,} tokens ({n_tgt:,} target tokens), lr {lr}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation=args.attn)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if not args.full:
        from peft import LoraConfig, get_peft_model
        model.enable_input_require_grads()
        cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))

    def batches():
        order = sorted(range(len(data)), key=lambda i: len(data[i][0]))  # length-bucketed
        chunks, cur, cur_max = [], [], 0
        for i in order:
            L = len(data[i][0])
            if cur and max(cur_max, L) * (len(cur) + 1) > args.batch_tokens:
                chunks.append(cur)
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, L)
        if cur:
            chunks.append(cur)
        random.shuffle(chunks)
        return chunks

    n_micro = len(batches())
    total_steps = math.ceil(n_micro * args.epochs / args.accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warm = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, total_steps - warm)))))
    print(f"{n_micro} micro-batches/epoch, {total_steps} optimizer steps", flush=True)

    model.train()
    step, t0, seen, losses = 0, time.time(), 0, []
    micro = 0  # micro-batches seen across epochs (a per-epoch counter never stepped when accum > batches/epoch)
    ep = 0
    while step < total_steps:
        for mi, chunk in enumerate(batches()):
            L = max(len(data[i][0]) for i in chunk)
            ids = torch.full((len(chunk), L), tok.pad_token_id or 0, dtype=torch.long)
            lab = torch.full((len(chunk), L), -100, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, i in enumerate(chunk):
                a, b = data[i]
                ids[j, :len(a)] = torch.tensor(a)
                lab[j, :len(b)] = torch.tensor(b)
                att[j, :len(a)] = 1
            loss = target_loss(model, ids.cuda(), att.cuda(), lab.cuda())
            (loss / args.accum).backward()
            losses.append(loss.item())
            seen += int(att.sum())
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == total_steps:
                    el = time.time() - t0
                    print(f"step {step}/{total_steps} ep {ep} loss {sum(losses[-40:]) / len(losses[-40:]):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {seen / el:,.0f} tok/s {el / 60:.1f} min", flush=True)
                if step >= total_steps:
                    break
        ep += 1
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    json.dump({**vars(args), "lr": lr, "examples": len(data), "tokens": n_tok, "target_tokens": n_tgt,
               "minutes": round((time.time() - t0) / 60, 1), "final_loss": sum(losses[-40:]) / len(losses[-40:])},
              open(os.path.join(args.out, "train_meta.json"), "w"), indent=1)
    print(f"saved {args.out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
