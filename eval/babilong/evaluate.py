"""
BABILong evaluation for Atlas/Titans models.

Evaluates trained MAC transformer checkpoints on the BABILong benchmark,
testing long-context recall at increasing context lengths.

Usage:
    # Evaluate on specific tasks and lengths:
    python eval/babilong/evaluate.py \
        --checkpoint runs/170m-atlas-mac/step-1000 \
        --model 170m --variant atlas-mac \
        --lengths 0k 4k 16k 64k \
        --tasks qa1 qa2 qa3 qa4 qa5 \
        --output results/atlas-mac-babilong.json

    # Quick smoke test:
    python eval/babilong/evaluate.py \
        --checkpoint runs/170m-atlas-mac/step-1000 \
        --model 170m --variant atlas-mac \
        --lengths 0k 4k --tasks qa1 --max-examples 10

    # BSC (via SLURM):
    sbatch eval/babilong/evaluate_slurm.sh

Scoring protocol (fixed 2026-09-02): context and candidate are tokenized
SEPARATELY with no special tokens and concatenated, and exactly the
candidate's token positions are scored. The previous implementation derived
the prompt length from `tokenizer.encode(prompt)`, which for T5 includes an
appended `</s>`, so the scored positions skipped the first candidate token
and included the trailing `</s>`; every qa1-qa4 answer is a single T5 token,
so the scored set was `['</s>']` for every candidate — every BABILong number
produced before this fix was an end-of-text prior, not an answer likelihood.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from titans_pytorch import MemoryAsContextTransformer
from experiments.configs import get_config
from eval.babilong.prompts import TASK_LABELS, build_prompt


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_dir, model_size, variant, ablation=None, device="cuda", vanilla=False):
    """Load model from accelerate checkpoint."""
    config = get_config(model_size, variant, ablation)
    if vanilla:
        # Memory-free baseline (Ablation 4): match --vanilla training
        # (neural_memory_layers=()), else the checkpoint's absent NeuralMemory
        # weights mismatch a full-Titans model on load.
        config["model"]["neural_memory_layers"] = ()
    model = MemoryAsContextTransformer(**config["model"])

    # accelerate saves model weights in a subdirectory
    weights_file = None
    for name in ["model.safetensors", "pytorch_model.bin"]:
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            weights_file = path
            break

    if weights_file is None:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_dir}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )

    if weights_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weights_file, device="cpu")
    else:
        state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)

    # Strip 'module.' prefix if saved with DDP
    cleaned = {}
    for k, v in state_dict.items():
        key = k.removeprefix("module.")
        cleaned[key] = v

    # Strict load, deliberately: a checkpoint whose parameter set does not
    # match the current config is not evaluable — its store semantics differ
    # (omega window fix), it lacks value_conv, or it carries the axial
    # positional embedding that MAC_DEFAULTS now disables. Loading it with
    # strict=False would silently run a randomly-initialized value conv or
    # drop the positional embedding the model was trained with.
    try:
        model.load_state_dict(cleaned)
    except RuntimeError as err:
        raise RuntimeError(
            f"Checkpoint {weights_file} does not match the current model config "
            f"({model_size} / {variant}" + (f" / {ablation}" if ablation else "") + "). "
            f"Likely cause: the checkpoint predates one of — the omega window fix "
            f"(different store semantics), the value_conv addition, or "
            f"use_axial_pos_emb=False in MAC_DEFAULTS (checkpoint carries "
            f"axial_pos_emb.* weights). Such checkpoints are not evaluable on this "
            f"code; retrain. Original error:\n{err}"
        ) from err
    model = model.to(device).eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {param_count / 1e6:.1f}M params from {weights_file}")

    return model, config


def load_tokenizer(tokenizer_dir):
    """Load tokenizer from directory."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_dir)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_example(model, tokenizer, prompt_text, candidates, device="cuda"):
    """Pick the candidate with the highest log-likelihood under the model.

    Closed-answer-set protocol for a pretrained-only (non-instruction-tuned)
    base LM: for each candidate, score log P(candidate tokens | prompt) with
    one forward over prompt + candidate, pick the argmax of the
    length-normalized log-probability.

    Tokenization: the prompt and the candidate are tokenized SEPARATELY with
    `add_special_tokens=False` and concatenated (candidate encoded as
    " <cand>" so it carries its leading-space piece). This is the standard
    guard against the joint-tokenization trap — tokenizing prompt+candidate
    together lets the boundary re-segment differently per candidate, and a
    tokenizer that appends specials (T5 appends `</s>`) shifts the scored
    positions. Exactly the candidate's token positions are scored and their
    ids are asserted to equal the candidate's own tokenization.

    History (2026-09-02): the previous implementation computed the prompt
    length from `tokenizer.encode(prompt_text)`, which for T5 includes the
    appended `</s>`, so `range(prompt_len, full_len)` skipped the first
    candidate token and scored the trailing `</s>`. Every qa1-qa4 answer is a
    single T5 token, so the scored set was `['</s>']` for every candidate:
    every BABILong number produced before this fix measured
    P(end-of-text | prompt + word) — an end-of-text prior with no dependence
    on the context — and sat at chance by construction.

    Returns: (chosen_candidate, per_candidate_norm_log_probs, prompt_num_tokens,
              per_candidate_details) where details[cand] = {"sum": raw summed
              log-prob, "num_tokens": candidate token count} so post-hoc
              analyses have the un-normalized vectors.
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    log_probs = {}
    details = {}
    for cand in candidates:
        cand_ids = tokenizer.encode(" " + cand, add_special_tokens=False)
        if len(cand_ids) == 0:
            log_probs[cand] = float("-inf")
            details[cand] = {"sum": float("-inf"), "num_tokens": 0}
            continue

        full_ids = prompt_ids + cand_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

        logits = model.forward(input_ids, disable_flex_attn=True)
        # logits[:, i] predicts the token at position i+1, so candidate token
        # at position pos (pos in [prompt_len, len(full_ids))) is scored by
        # logits[:, pos - 1].
        log_softmax = F.log_softmax(logits[0].float(), dim=-1)

        scored_ids = full_ids[prompt_len:]
        assert scored_ids == cand_ids, (
            f"scored token ids {scored_ids} != candidate ids {cand_ids} for {cand!r}"
        )

        total_lp = 0.0
        for pos, target_token in zip(range(prompt_len, len(full_ids)), scored_ids):
            total_lp += log_softmax[pos - 1, target_token].item()

        # Length-normalize so multi-token candidates aren't penalized (the
        # decision statistic); keep the raw sum + token count alongside.
        log_probs[cand] = total_lp / len(cand_ids)
        details[cand] = {"sum": total_lp, "num_tokens": len(cand_ids)}

    chosen = max(log_probs, key=log_probs.get)
    return chosen, log_probs, prompt_len, details


def _target_matches(target, chosen):
    """qa8 target can be a comma-separated list (e.g. 'apple,milk'). Everything
    else is a single label. Compare case-insensitively."""
    target_parts = {t.strip().lower() for t in target.split(",")}
    chosen_parts = {c.strip().lower() for c in chosen.split(",")}
    return target_parts == chosen_parts


def evaluate_task(model, tokenizer, task, length, max_examples=None, device="cuda"):
    """Evaluate model on a single task at a single context length via
    log-likelihood scoring over the closed candidate set (TASK_LABELS[task])."""
    from datasets import load_dataset

    dataset = load_dataset("RMT-team/babilong", length, split=task)

    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    candidates = sorted(TASK_LABELS[task])
    correct = 0
    total = 0
    results = []

    for example in tqdm(dataset, desc=f"{task}@{length}", leave=False):
        context = example["input"]
        question = example["question"]
        target = example["target"]

        prompt_text = build_prompt(task=task, context=context, question=question)

        t0 = time.time()
        chosen, log_probs, num_tokens, details = score_example(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            candidates=candidates,
            device=device,
        )
        elapsed = time.time() - t0

        is_correct = _target_matches(target=target, chosen=chosen)
        correct += int(is_correct)
        total += 1

        results.append({
            "question": question.strip(),
            "target": target,
            "chosen": chosen,
            "correct": is_correct,
            "log_probs": {k: round(v, 3) for k, v in log_probs.items()},
            "log_prob_sums": {k: round(d["sum"], 3) for k, d in details.items()},
            "cand_num_tokens": {k: d["num_tokens"] for k, d in details.items()},
            "num_tokens": num_tokens,
            "time_s": round(elapsed, 2),
        })

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TASKS = ["qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10"]
ALL_LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k"]


def parse_args():
    p = argparse.ArgumentParser(description="BABILong evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to accelerate checkpoint dir")
    p.add_argument("--model", required=True, choices=["170m", "340m", "760m", "1.3b"])
    p.add_argument("--variant", required=True,
                    choices=["titans-mac", "titans-mag", "atlas-mac", "atlas-mag"])
    p.add_argument("--ablation", default=None)
    p.add_argument("--vanilla", action="store_true",
                   help="Memory-free baseline: build with neural_memory_layers=() to match --vanilla training.")
    p.add_argument("--tokenizer-dir", default=None,
                    help="Path to tokenizer (default: checkpoint/../tokenizer or google-t5/t5-base)")
    p.add_argument("--tasks", nargs="+", default=["qa1", "qa2", "qa3", "qa4", "qa5"],
                    help="Tasks to evaluate (default: qa1-qa5)")
    p.add_argument("--lengths", nargs="+", default=["0k", "4k", "16k", "64k"],
                    help="Context lengths to evaluate")
    p.add_argument("--max-examples", type=int, default=None,
                    help="Max examples per task (for quick testing)")
    p.add_argument("--output", default=None, help="Output JSON path")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    # Load model
    model, config = load_model(
        checkpoint_dir=args.checkpoint,
        model_size=args.model,
        variant=args.variant,
        ablation=args.ablation,
        device=args.device,
        vanilla=args.vanilla,
    )

    # Load tokenizer
    tokenizer_dir = args.tokenizer_dir
    if tokenizer_dir is None:
        # Try to find tokenizer relative to checkpoint
        for candidate in [
            os.path.join(os.path.dirname(args.checkpoint), "tokenizer"),
            os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), "tokenizer"),
        ]:
            if os.path.exists(candidate):
                tokenizer_dir = candidate
                break
    tokenizer = load_tokenizer(tokenizer_dir or "google-t5/t5-base")
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    # Run evaluation
    all_results = {
        "model": args.model,
        "variant": args.variant,
        "ablation": args.ablation,
        "checkpoint": args.checkpoint,
        "tasks": {},
    }

    for length in args.lengths:
        print(f"\n=== Context length: {length} ===")
        all_results["tasks"][length] = {}

        for task in args.tasks:
            # Check if task exists at this length
            # qa6-qa10 only at 0k; qa1-qa5 at all lengths
            if length != "0k" and task not in ["qa1", "qa2", "qa3", "qa4", "qa5",
                                                 "qa6", "qa7", "qa8", "qa9", "qa10"]:
                continue

            result = evaluate_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                length=length,
                max_examples=args.max_examples,
                device=args.device,
            )

            all_results["tasks"][length][task] = {
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
                "examples": result["results"],
            }

            print(f"  {task}: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")

    # Summary table
    print("\n=== Summary ===")
    header = f"{'Task':<6}" + "".join(f"{l:>8}" for l in args.lengths)
    print(header)
    print("-" * len(header))
    for task in args.tasks:
        row = f"{task:<6}"
        for length in args.lengths:
            acc = all_results["tasks"].get(length, {}).get(task, {}).get("accuracy", None)
            row += f"{acc:>7.1%} " if acc is not None else f"{'—':>8}"
        print(row)

    # Save results
    output_path = args.output
    if output_path is None:
        output_path = f"results/{args.model}-{args.variant}-babilong.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {output_path}")


if __name__ == "__main__":
    main()
