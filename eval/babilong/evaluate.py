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

Memory ceiling (measured 2026-09-02): with per-token retrieve the model's
whole-sequence forward materializes every token's memory-weight state for the
final retrieve — 0.59 MB/token in bf16 at 170M with two memory layers (1.2 MB
in fp32): 4.8 GB @4K, 19 GB @16K, 39 GB @32K, 78 GB @64K, 1.2 TB @1M. This
script estimates that footprint per context length and refuses to run past
the device's memory unless --force is given; ≥128K needs a chunked-inference
forward that does not exist yet. --bf16 halves the footprint.
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

def normalize_model_config(model_config):
    """JSON round-trip so tuples become lists and non-JSON values (dtypes,
    modules) become strings — the form written next to a checkpoint by
    train.save_checkpoint and compared on load."""
    return json.loads(json.dumps(model_config, sort_keys=True, default=str))


def check_model_config_drift(checkpoint_dir, model_config):
    """Compare the config being built against the checkpoint's recorded
    `model_config.json`. Strict state-dict loading only catches PARAMETER-set
    drift; a non-parameter field (neural_memory_batch_size, omega_context,
    per_token_retrieve, ...) changes eval semantics without changing any
    shape and would load clean. Raise on any differing key; warn when the
    checkpoint predates the recording."""
    path = os.path.join(checkpoint_dir, "model_config.json")
    if not os.path.exists(path):
        print(
            f"WARNING: {path} missing (checkpoint predates model_config.json "
            f"recording) — cannot verify the checkpoint was trained with this "
            f"model config; only parameter-set drift is checked (strict load)."
        )
        return
    with open(path) as f:
        saved = f.read()
    saved = json.loads(saved)
    current = normalize_model_config(model_config)
    differing = sorted(
        key for key in set(saved) | set(current) if saved.get(key) != current.get(key)
    )
    if differing:
        details = "\n".join(
            f"  {key}: checkpoint={saved.get(key)!r} current={current.get(key)!r}"
            for key in differing
        )
        raise ValueError(
            f"model config drift between {path} and the config being built — "
            f"evaluating would run different semantics than training:\n{details}"
        )


def load_model(checkpoint_dir, model_size, variant, ablation=None, device="cuda", vanilla=False, dtype=None):
    """Load model from accelerate checkpoint.

    `vanilla` mirrors train.py --vanilla (neural_memory_layers=()): a
    memory-free baseline checkpoint has no NeuralMemory weights, so it can
    only load into a memory-free model.
    """
    config = get_config(model_size, variant, ablation)
    if vanilla:
        config["model"]["neural_memory_layers"] = ()
    check_model_config_drift(checkpoint_dir=checkpoint_dir, model_config=config["model"])
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
    if dtype is not None:
        model = model.to(dtype)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {param_count / 1e6:.1f}M params from {weights_file}")

    return model, config


def load_tokenizer(tokenizer_dir):
    """Load tokenizer from directory."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_dir)


# ---------------------------------------------------------------------------
# Memory ceiling
# ---------------------------------------------------------------------------

def memory_layers(model):
    """The NeuralMemory modules of a MemoryAsContextTransformer (layer slot 4)."""
    return [layer[4] for layer in model.layers if layer[4] is not None]


def estimate_retrieve_state_bytes(model, num_tokens):
    """Bytes of memory-weight state the whole-sequence forward holds for the
    final retrieve at `num_tokens` input tokens.

    Every memory layer's decay scan emits one weight state per token when the
    omega rule is on (per-token store granularity; the per-token retrieve
    reads them all at once), or one per store chunk otherwise. The states for
    ALL tokens of the sequence are concatenated before the retrieve, so the
    footprint is linear in the sequence length — the ceiling that makes
    single-GPU evaluation past ~32-64K impossible at 170M without a
    chunked-inference forward. The interleaved longterm-mem tokens count too:
    the memory sees seq_len_with_longterm_mem(num_tokens) positions.
    """
    positions = model.seq_len_with_longterm_mem(num_tokens) if num_tokens > 0 else 0
    total = 0
    for mem in memory_layers(model):
        state_values = sum(p.numel() for p in mem.memory_model_parameters)
        elem_bytes = next(iter(mem.memory_model_parameters)).element_size()
        per_position = state_values * elem_bytes
        if mem.omega_context <= 1:
            per_position /= mem.store_chunk_size
        total += per_position * positions
    return int(total)


def length_label_to_tokens(length):
    """BABILong split label ('0k', '4k', '64k', ...) -> approximate context
    tokens. '0k' is the bare bAbI facts (a few hundred tokens)."""
    k = int(length.rstrip("k"))
    return k * 1024 if k > 0 else 512


def memory_ceiling_message(model, length, device, force):
    """Return None when the estimated retrieve-state footprint fits the device,
    else a message explaining the refusal (unless `force`, which downgrades
    it to a warning printed by the caller)."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    num_tokens = length_label_to_tokens(length)
    estimate = estimate_retrieve_state_bytes(model=model, num_tokens=num_tokens)
    total = torch.cuda.get_device_properties(torch.device(device)).total_memory
    if estimate <= 0.8 * total:
        return None
    return (
        f"context {length} (~{num_tokens:,} tokens): the whole-sequence forward would hold "
        f"~{estimate / 1e9:.1f} GB of per-token memory-weight state for the final retrieve "
        f"(device total {total / 1e9:.1f} GB). Single-GPU evaluation past ~32-64K needs a "
        f"chunked-inference forward that does not exist yet; --bf16 halves the footprint. "
        + ("Running anyway (--force)." if force else "Skipping (pass --force to try).")
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def encode_prompt_and_candidates(tokenizer, prompt_text, candidates):
    """Tokenize the prompt and each candidate separately (no special tokens)
    and verify the two properties the scorer depends on:

      1. joint == separate: `encode(prompt + " " + cand)` must equal
         `prompt_ids + cand_ids`, otherwise the candidate's tokens would not be
         the ones the model sees in context (boundary re-segmentation);
      2. no special token (T5's `</s>`) may reach the model — that is exactly
         how the pre-fix scorer ended up scoring `</s>` instead of the answer.

    Raises ValueError (not assert — asserts are stripped under -O) on either.
    Returns (prompt_ids, {cand: cand_ids}).
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    specials = {t for t in (tokenizer.eos_token_id, tokenizer.pad_token_id) if t is not None}
    if any(t in specials for t in prompt_ids):
        raise ValueError("special token found in the prompt ids — encode with add_special_tokens=False")

    cand_ids = {}
    for cand in candidates:
        ids = tokenizer.encode(" " + cand, add_special_tokens=False)
        if len(ids) == 0:
            raise ValueError(f"candidate {cand!r} tokenizes to nothing")
        if any(t in specials for t in ids):
            raise ValueError(f"special token found in candidate ids for {cand!r}")
        joint = tokenizer.encode(prompt_text + " " + cand, add_special_tokens=False)
        if joint != prompt_ids + ids:
            raise ValueError(
                f"joint tokenization of prompt + {cand!r} differs from prompt_ids + cand_ids "
                f"at the boundary — the scored tokens would not be the in-context tokens"
            )
        cand_ids[cand] = ids
    return prompt_ids, cand_ids


def _candidate_rows_log_softmax(model, input_ids, rows, disable_flex_attn):
    """Log-softmax over the vocab for the given positions only. The forward
    returns pre-logit hidden states; only the needed rows are projected, so
    the [L, vocab] logits tensor is never materialized (it was 8.4 GB fp32 at
    64K on its own)."""
    hidden = model.forward(input_ids, disable_flex_attn=disable_flex_attn, return_hidden=True)
    logits = model.to_logits(hidden[:, rows])
    return F.log_softmax(logits[0].float(), dim=-1)


@torch.no_grad()
def score_example(model, tokenizer, prompt_text, candidates, device="cuda", disable_flex_attn=True):
    """Pick the candidate with the highest log-likelihood under the model.

    Closed-answer-set protocol for a pretrained-only (non-instruction-tuned)
    base LM: score log P(candidate tokens | prompt) for every candidate and
    pick the argmax of the length-normalized log-probability.

    Tokenization: the prompt and the candidate are tokenized SEPARATELY with
    `add_special_tokens=False` and concatenated (candidate encoded as
    " <cand>" so it carries its leading-space piece). This is the standard
    guard against the joint-tokenization trap — tokenizing prompt+candidate
    together lets the boundary re-segment differently per candidate, and a
    tokenizer that appends specials (T5 appends `</s>`) shifts the scored
    positions. `encode_prompt_and_candidates` verifies joint == separate and
    that no special token reaches the model; exactly the candidate's token
    positions are scored.

    Forward passes: when every candidate is a single token, ONE forward over
    the prompt scores all of them from the last row. Multi-token candidate
    sets fall back to one forward per candidate over prompt + candidate.
    Only the candidate rows are projected to logits (return_hidden).

    Why one forward is valid — and why "the last row depends only on the
    prompt" is NOT the reason: this model is not length-invariant. With the
    omega rule + per-token retrieve, positions inside an incomplete final
    store chunk read the state after the last COMPLETE chunk (the remainder
    is cached, never stored in a whole-sequence forward), so the logits at a
    fixed position change when an appended token completes the chunk
    (measured 0.8-2.1 nats at random init, only at length % chunk == chunk-1;
    zero for the memory-free trunk — see
    test_per_token_retrieve_tail_reads_last_complete_chunk_state). Scoring
    all candidates from ONE forward keeps them consistent with each other;
    scoring each with the candidate appended would also be consistent, at a
    different chunk phase. Roughly 1/8 of prompts fall in the fresh-state
    regime under either protocol — the same tail quirk training saw on the
    last 4 of every 1084 positions.

    Normalization: mean log-prob per candidate token. Residual asymmetry:
    where a candidate set mixes token counts (qa7's " ten" is 2 T5 tokens
    while the other numerals are 1), the per-token mean is not perfectly
    comparable; the raw sums and token counts are persisted in `details` so
    post-hoc analyses can re-normalize. qa5's names are single tokens only in
    their capitalized in-context form — see prompts.TASK_LABELS.

    Known, unquantified conditioning shift: training documents end with
    `</s>`; scoring inputs contain none. Shared by every candidate, so it
    cannot create a per-candidate asymmetry.

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
              log-prob, "num_tokens": candidate token count}.
    """
    prompt_ids, cand_ids = encode_prompt_and_candidates(
        tokenizer=tokenizer, prompt_text=prompt_text, candidates=candidates
    )
    prompt_len = len(prompt_ids)

    log_probs = {}
    details = {}

    if all(len(ids) == 1 for ids in cand_ids.values()):
        # one forward over the prompt; the last row predicts the first
        # (and only) candidate token for every candidate
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        rows = slice(prompt_len - 1, prompt_len)
        log_softmax = _candidate_rows_log_softmax(
            model=model, input_ids=input_ids, rows=rows, disable_flex_attn=disable_flex_attn
        )
        for cand, ids in cand_ids.items():
            lp = log_softmax[0, ids[0]].item()
            log_probs[cand] = lp
            details[cand] = {"sum": lp, "num_tokens": 1}
    else:
        for cand, ids in cand_ids.items():
            full_ids = prompt_ids + ids
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            # logits[:, i] predicts the token at position i+1: candidate token
            # at position prompt_len + j is scored by row prompt_len - 1 + j
            rows = slice(prompt_len - 1, prompt_len - 1 + len(ids))
            log_softmax = _candidate_rows_log_softmax(
                model=model, input_ids=input_ids, rows=rows, disable_flex_attn=disable_flex_attn
            )
            total_lp = sum(log_softmax[j, target].item() for j, target in enumerate(ids))
            log_probs[cand] = total_lp / len(ids)
            details[cand] = {"sum": total_lp, "num_tokens": len(ids)}

    chosen = max(log_probs, key=log_probs.get)
    return chosen, log_probs, prompt_len, details


def _target_matches(target, chosen):
    """qa8 target can be a comma-separated list (e.g. 'apple,milk'). Everything
    else is a single label. Compare case-insensitively."""
    target_parts = {t.strip().lower() for t in target.split(",")}
    chosen_parts = {c.strip().lower() for c in chosen.split(",")}
    return target_parts == chosen_parts


def target_in_candidates(target, candidates):
    """Whether a target is reachable by argmax over the candidate set (case-
    insensitive). qa8's two-object targets ('apple,football', ~6% of examples)
    are not — no single label can match them — so they are excluded and
    counted rather than scored as guaranteed failures."""
    lowered = {c.lower() for c in candidates}
    return target.strip().lower() in lowered


def evaluate_task(model, tokenizer, task, length, max_examples=None, device="cuda", disable_flex_attn=True):
    """Evaluate model on a single task at a single context length via
    log-likelihood scoring over the closed candidate set (TASK_LABELS[task])."""
    from datasets import load_dataset

    dataset = load_dataset("RMT-team/babilong", length, split=task)

    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    candidates = sorted(TASK_LABELS[task])
    correct = 0
    total = 0
    n_excluded = 0
    results = []

    for example in tqdm(dataset, desc=f"{task}@{length}", leave=False):
        context = example["input"]
        question = example["question"]
        target = example["target"]

        if not target_in_candidates(target=target, candidates=candidates):
            n_excluded += 1
            continue

        prompt_text = build_prompt(task=task, context=context, question=question)

        t0 = time.time()
        chosen, log_probs, num_tokens, details = score_example(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            candidates=candidates,
            device=device,
            disable_flex_attn=disable_flex_attn,
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

    if n_excluded:
        print(f"  {task}@{length}: {n_excluded} example(s) excluded — target not in the candidate set")

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "n_excluded": n_excluded,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TASKS = ["qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10"]
ALL_LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k"]


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="BABILong evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to accelerate checkpoint dir")
    p.add_argument("--model", required=True, choices=["170m", "340m", "760m", "1.3b"])
    p.add_argument("--variant", required=True,
                    choices=["titans-mac", "titans-mag", "atlas-mac", "atlas-mag"])
    p.add_argument("--ablation", default=None)
    p.add_argument("--vanilla", action="store_true",
                   help="Memory-free baseline: build with neural_memory_layers=() to match train.py --vanilla.")
    p.add_argument("--tokenizer-dir", default=None,
                    help="Path to tokenizer (default: checkpoint/../tokenizer or google-t5/t5-base)")
    p.add_argument("--tasks", nargs="+", default=["qa1", "qa2", "qa3", "qa4", "qa5"], choices=ALL_TASKS,
                    help="Tasks to evaluate (default: qa1-qa5)")
    p.add_argument("--lengths", nargs="+", default=["0k", "4k", "16k", "64k"], choices=ALL_LENGTHS,
                    help="Context lengths to evaluate")
    p.add_argument("--max-examples", type=int, default=None,
                    help="Max examples per task (for quick testing)")
    p.add_argument("--output", default=None, help="Output JSON path (written after every task)")
    p.add_argument("--device", default=default_device())
    p.add_argument("--bf16", action="store_true",
                   help="Run the model in bfloat16 (training dtype); halves the per-token retrieve-state footprint")
    p.add_argument("--use-flex-attn", action="store_true",
                   help="Use flex attention (CUDA only); default disables it")
    p.add_argument("--force", action="store_true",
                   help="Run a context length even when the estimated retrieve-state footprint exceeds device memory")
    return p.parse_args()


def write_results(all_results, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)


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
        dtype=torch.bfloat16 if args.bf16 else None,
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
    # T5 reports 32100 (no sentinel ids); the model's embedding has 32128 rows
    # (padded to a multiple of 128) — both are correct, they are different
    # quantities.
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size} | model num_tokens={config['model']['num_tokens']}")

    output_path = args.output or f"results/{args.model}-{args.variant}-babilong.json"

    # Run evaluation — results are written after every (length, task) so a
    # mid-run failure at a long context keeps everything finished before it
    all_results = {
        "model": args.model,
        "variant": args.variant,
        "ablation": args.ablation,
        "checkpoint": args.checkpoint,
        "dtype": "bfloat16" if args.bf16 else "float32",
        "tasks": {},
        "skipped_lengths": {},
    }

    for length in args.lengths:
        print(f"\n=== Context length: {length} ===")
        ceiling = memory_ceiling_message(model=model, length=length, device=args.device, force=args.force)
        if ceiling is not None:
            print(f"  {ceiling}")
            if not args.force:
                all_results["skipped_lengths"][length] = ceiling
                write_results(all_results=all_results, output_path=output_path)
                continue

        all_results["tasks"][length] = {}

        for task in args.tasks:
            result = evaluate_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                length=length,
                max_examples=args.max_examples,
                device=args.device,
                disable_flex_attn=not args.use_flex_attn,
            )

            all_results["tasks"][length][task] = {
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
                "n_excluded": result["n_excluded"],
                "examples": result["results"],
            }
            write_results(all_results=all_results, output_path=output_path)

            print(f"  {task}: {result['accuracy']:.1%} ({result['correct']}/{result['total']}"
                  + (f", {result['n_excluded']} excluded" if result["n_excluded"] else "") + ")")

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

    write_results(all_results=all_results, output_path=output_path)
    print(f"\nResults saved → {output_path}")


if __name__ == "__main__":
    main()
