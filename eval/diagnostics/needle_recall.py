"""Does anything cross the attention window through the neural memory?

BABILong cannot answer that at this scale: the memory-free trunk matches the
memory arms at every context length >= 2k (2026-09-07), so its plateau is a
prior induced by the filler text, not retrieval. This probe removes both the
reasoning and the guessing.

A trial is: [needle] [filler of `distance` tokens] [needle's first `cue_len`
tokens] and the score is the mean NLL of the needle's REMAINING tokens at the
second occurrence. The needle is a uniformly random token sequence, so a model
with no access to the first occurrence pays ~log(vocab) nats per token
whatever the filler says; there is no prior to exploit and no candidate set to
guess from. Any drop below that baseline is information that travelled from
the first occurrence to the second.

The attention window is `segment_len` tokens (64 in the shipped config), so a
distance beyond it can only be bridged by the memory. Two controls make the
reading unambiguous:

  - the memory-free trunk (--vanilla): must stay at the baseline past the
    window, or the probe is measuring something other than memory;
  - `--shuffle-control`: the same trial with a DIFFERENT random needle at the
    cue, which must score at the baseline for every model — it detects a probe
    that leaks the answer through the filler or the position.

Reported per distance: mean NLL of the recalled tokens, the no-context
baseline measured on the same needles, and the recall margin (baseline minus
NLL) in nats. Margin ~0 means nothing crossed; margin -> log(vocab) means
verbatim recall.

    python needle_recall.py --checkpoint DIR --model 170m --variant atlas-mac \
        --distances 32 128 512 1024 2048 4096 --trials 32 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.babilong.evaluate import load_model  # noqa: E402


def build_trial(generator, needle_lo, needle_hi, needle_len, cue_len, distance, filler_lo, filler_hi):
    """[needle] [filler] [cue] -> (input ids, the needle, the total length).

    The scored positions are the needle's tail at its second occurrence, so the
    model must produce them from the first occurrence: the cue alone is a
    random prefix that determines nothing.

    The needle and the filler are drawn from DISJOINT token ranges, so no
    needle token can appear in the filler by chance. Without that, at 4096
    filler tokens each of the needle's tokens has a ~13% chance of occurring in
    the filler, and a partial match there is an alternative source for the
    scored tail — the probe would no longer isolate the first occurrence.

    The needle's own tokens are distinct (sampled without replacement): a
    repeated token would give the cue two possible continuations in the first
    occurrence, weakening the strongest available signal for no reason.
    """
    span = needle_hi - needle_lo
    if needle_len > span:
        raise ValueError(f"needle_len ({needle_len}) exceeds the needle range ({span} tokens)")
    needle = needle_lo + torch.randperm(span, generator=generator)[:needle_len]
    filler = torch.randint(filler_lo, filler_hi, (distance,), generator=generator)
    ids = torch.cat((needle, filler, needle[:cue_len]))
    return ids, needle, ids.numel()


def score_tail(model, ids, tail, device, chunk_len=None):
    """Mean NLL (nats/token) the model assigns to `tail` continuing `ids`."""
    full = torch.cat((ids, tail)).unsqueeze(0).to(device)
    with torch.no_grad():
        if chunk_len is not None and hasattr(model, "forward_chunked"):
            hidden = model.forward_chunked(full[:, :-1], chunk_len=chunk_len, return_hidden=True)
            logits = model.to_logits(hidden)
        else:
            logits = model(full[:, :-1])
    scored = logits[0, -tail.numel():].float().log_softmax(dim=-1)
    return float(-scored.gather(-1, tail.to(device).unsqueeze(-1)).mean())


def run(model, args, device):
    generator = torch.Generator().manual_seed(args.seed)
    results = {}
    for distance in args.distances:
        recalled, baseline, shuffled = [], [], []
        for _ in range(args.trials):
            ids, needle, _ = build_trial(
                generator=generator, needle_lo=args.needle_lo, needle_hi=args.needle_hi,
                needle_len=args.needle_len, cue_len=args.cue_len, distance=distance,
                filler_lo=args.filler_lo, filler_hi=args.filler_hi,
            )
            tail = needle[args.cue_len:]
            recalled.append(score_tail(model=model, ids=ids, tail=tail, device=device, chunk_len=args.chunk_len))

            # baseline: the same cue and tail with NO first occurrence — filler
            # then cue, so the tail is unpredictable by construction
            no_needle = ids[args.needle_len:]
            baseline.append(score_tail(model=model, ids=no_needle, tail=tail, device=device, chunk_len=args.chunk_len))

            if args.shuffle_control:
                # a DIFFERENT needle at the cue: must stay at the baseline
                other = torch.randint(args.needle_lo, args.needle_hi, (args.needle_len,), generator=generator)
                mixed = torch.cat((other, ids[args.needle_len:]))
                shuffled.append(score_tail(model=model, ids=mixed, tail=tail, device=device, chunk_len=args.chunk_len))

        entry = dict(
            distance=distance,
            nll_recall=sum(recalled) / len(recalled),
            nll_baseline=sum(baseline) / len(baseline),
            trials=args.trials,
        )
        entry["margin"] = entry["nll_baseline"] - entry["nll_recall"]
        if shuffled:
            entry["nll_shuffle_control"] = sum(shuffled) / len(shuffled)
            entry["margin_shuffle_control"] = entry["nll_baseline"] - entry["nll_shuffle_control"]
        results[str(distance)] = entry
        line = (f"  distance {distance:>6}: recall NLL {entry['nll_recall']:.4f}  "
                f"baseline {entry['nll_baseline']:.4f}  margin {entry['margin']:+.4f} nats")
        if shuffled:
            line += f"  (shuffle control margin {entry['margin_shuffle_control']:+.4f})"
        print(line, flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", required=True, choices=["170m", "340m", "760m", "1.3b"])
    p.add_argument("--variant", required=True, choices=["titans-mac", "titans-mag", "atlas-mac", "atlas-mag"])
    p.add_argument("--ablation", default=None)
    p.add_argument("--vanilla", action="store_true", help="memory-free control (must stay at the baseline past the window)")
    p.add_argument("--memory-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--distances", type=int, nargs="+", default=[32, 128, 512, 1024, 2048, 4096],
                   help="tokens between the needle and its cue (the attention window is segment_len, 64 in the shipped config)")
    p.add_argument("--needle-len", type=int, default=12)
    p.add_argument("--cue-len", type=int, default=4, help="tokens of the needle repeated as the cue; the rest is scored")
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--vocab-size", type=int, default=32100)
    # disjoint halves: no needle token can occur in the filler
    p.add_argument("--needle-lo", type=int, default=0)
    p.add_argument("--needle-hi", type=int, default=16050)
    p.add_argument("--filler-lo", type=int, default=16050)
    p.add_argument("--filler-hi", type=int, default=32100)
    p.add_argument("--shuffle-control", action="store_true", help="also score trials whose cue follows a DIFFERENT needle")
    p.add_argument("--chunk-len", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="results/needle_recall.json")
    args = p.parse_args()

    if args.cue_len >= args.needle_len:
        raise ValueError(f"cue_len ({args.cue_len}) must be < needle_len ({args.needle_len}) or nothing is scored")
    if max(args.needle_lo, args.filler_lo) < min(args.needle_hi, args.filler_hi):
        raise ValueError(
            f"the needle range [{args.needle_lo}, {args.needle_hi}) and the filler range "
            f"[{args.filler_lo}, {args.filler_hi}) must be disjoint, or the filler can contain needle tokens"
        )

    from experiments.train import apply_memory_kwargs, parse_memory_kwargs  # noqa: F401  (mirrors the other diagnostics)

    model, _ = load_model(
        checkpoint_dir=args.checkpoint, model_size=args.model, variant=args.variant,
        ablation=args.ablation, device=args.device, vanilla=args.vanilla,
        memory_kwargs=parse_memory_kwargs(items=args.memory_kwarg),
    )
    model.eval()

    print(f"needle {args.needle_len} tokens, cue {args.cue_len}, scoring {args.needle_len - args.cue_len} tokens, "
          f"{args.trials} trials/distance, vocab {args.vocab_size} (uniform baseline ~{torch.log(torch.tensor(float(args.vocab_size))):.3f} nats)",
          flush=True)
    results = run(model=model, args=args, device=args.device)

    payload = dict(checkpoint=args.checkpoint, model=args.model, variant=args.variant, ablation=args.ablation,
                   vanilla=args.vanilla, needle_len=args.needle_len, cue_len=args.cue_len, trials=args.trials,
                   vocab_size=args.vocab_size, seed=args.seed, chunk_len=args.chunk_len, distances=results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
