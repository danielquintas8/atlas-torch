"""
Per-position validation NLL at contexts beyond the training length.

The "evaluate beyond the trained length" instrument: a model trained at 1K
whose NLL climbs as position grows past ~2x its training length has a
positional / extrapolation problem, independent of any QA benchmark. Two
intended uses:

  1. On a checkpoint trained WITH the absolute axial positional embedding
     (use_axial_pos_emb=True): confirm the out-of-distribution finding — the
     embedding feeds raw integer segment indices into a SiLU MLP, so beyond
     the training length its norm grows roughly linearly with position and
     NLL should rise with position.
  2. On a post-fix checkpoint (use_axial_pos_emb=False): verify NLL stays
     flat-ish with position (rotary inside the windowed attention carries
     within-window position; the neural memory is position-free).

Reads val.bin (uint16 memmap, same chunking convention as
experiments.train.MemmapTokenDataset: consecutive non-overlapping chunks of
seq_len + 1 tokens), runs the full parallel forward on the first
--max-chunks chunks for each --seq-lens value, computes per-token NLL (fp32
cross-entropy over inputs x[:, :-1] / labels x[:, 1:]) and reports the mean
NLL per --bucket positions plus the overall mean. Writes JSON to --output.

Usage:
    python eval/diagnostics/nll_by_position.py \
        --checkpoint runs/170m-atlas-mac/step-4000 --model 170m --variant atlas-mac \
        --val-bin /gpfs/projects/.../fineweb-t5/val.bin \
        --seq-lens 1024 2048 4096 8192 --max-chunks 32 --bucket 256 \
        --device cuda --output results/nll_by_position.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.babilong.evaluate import load_model


def iter_val_chunks(val_bin, seq_len, max_chunks):
    """Yield the first `max_chunks` consecutive (seq_len + 1)-token chunks."""
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    chunk_len = seq_len + 1
    n_chunks = len(data) // chunk_len
    for idx in range(min(n_chunks, max_chunks)):
        start = idx * chunk_len
        yield torch.from_numpy(data[start : start + chunk_len].astype(np.int64))


@torch.no_grad()
def nll_by_position(model, val_bin, seq_len, max_chunks, bucket, device, disable_flex_attn):
    """Mean per-bucket NLL over the first `max_chunks` val chunks at `seq_len`."""
    num_buckets = (seq_len + bucket - 1) // bucket
    bucket_sums = torch.zeros(num_buckets, dtype=torch.float64)
    bucket_counts = torch.zeros(num_buckets, dtype=torch.float64)
    bucket_index = torch.arange(seq_len) // bucket
    chunks_seen = 0

    for chunk in iter_val_chunks(val_bin=val_bin, seq_len=seq_len, max_chunks=max_chunks):
        x = chunk[None].to(device)
        inputs, labels = x[:, :-1], x[:, 1:]
        logits = model(inputs, disable_flex_attn=disable_flex_attn)
        token_nll = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).cpu().double()
        bucket_sums.index_add_(0, bucket_index, token_nll)
        bucket_counts.index_add_(0, bucket_index, torch.ones_like(token_nll))
        chunks_seen += 1

    if chunks_seen == 0:
        raise ValueError(f"{val_bin} holds no complete chunk of {seq_len + 1} tokens")

    buckets = []
    for b in range(num_buckets):
        buckets.append(dict(
            start=b * bucket,
            end=min((b + 1) * bucket, seq_len),
            nll=(bucket_sums[b] / bucket_counts[b]).item(),
        ))

    return dict(
        seq_len=seq_len,
        chunks=chunks_seen,
        overall=(bucket_sums.sum() / bucket_counts.sum()).item(),
        buckets=buckets,
    )


def print_table(results, bucket):
    seq_lens = [r["seq_len"] for r in results]
    max_len = max(seq_lens)
    header = f"{'position':>14}" + "".join(f"{f'L={L}':>12}" for L in seq_lens)
    print(header)
    print("-" * len(header))
    for b in range((max_len + bucket - 1) // bucket):
        start, end = b * bucket, min((b + 1) * bucket, max_len)
        row = f"{f'{start}-{end}':>14}"
        for r in results:
            cell = r["buckets"][b]["nll"] if b < len(r["buckets"]) else None
            row += f"{cell:>12.4f}" if cell is not None else f"{'—':>12}"
        print(row)
    print("-" * len(header))
    print(f"{'overall':>14}" + "".join(f"{r['overall']:>12.4f}" for r in results))
    print(f"{'chunks':>14}" + "".join(f"{r['chunks']:>12d}" for r in results))


def parse_args():
    p = argparse.ArgumentParser(description="Per-position validation NLL beyond the training length")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint dir (model.safetensors)")
    p.add_argument("--model", required=True, choices=["170m", "340m", "760m", "1.3b"])
    p.add_argument("--variant", required=True,
                   choices=["titans-mac", "titans-mag", "atlas-mac", "atlas-mag"])
    p.add_argument("--ablation", default=None)
    p.add_argument("--vanilla", action="store_true",
                   help="Memory-free baseline: build with neural_memory_layers=()")
    p.add_argument("--val-bin", required=True, help="Path to val.bin (uint16 memmap)")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--max-chunks", type=int, default=32, help="Chunks per seq-len")
    p.add_argument("--bucket", type=int, default=256, help="Positions per reported bucket")
    p.add_argument("--device", default="cpu")
    p.add_argument("--use-flex-attn", action="store_true",
                   help="Use flex attention (CUDA only); default disables it, matching the BABILong harness")
    p.add_argument("--output", default="results/nll_by_position.json")
    return p.parse_args()


def main():
    args = parse_args()

    model, _ = load_model(
        checkpoint_dir=args.checkpoint,
        model_size=args.model,
        variant=args.variant,
        ablation=args.ablation,
        device=args.device,
        vanilla=args.vanilla,
    )

    results = []
    for seq_len in args.seq_lens:
        print(f"\n=== seq_len {seq_len} ===")
        result = nll_by_position(
            model=model,
            val_bin=args.val_bin,
            seq_len=seq_len,
            max_chunks=args.max_chunks,
            bucket=args.bucket,
            device=args.device,
            disable_flex_attn=not args.use_flex_attn,
        )
        print(f"  overall NLL {result['overall']:.4f} over {result['chunks']} chunks")
        results.append(result)

    print()
    print_table(results=results, bucket=args.bucket)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dict(
            checkpoint=args.checkpoint,
            model=args.model,
            variant=args.variant,
            ablation=args.ablation,
            bucket=args.bucket,
            results=results,
        ), f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
