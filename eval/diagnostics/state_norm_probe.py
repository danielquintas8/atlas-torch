"""Why does the chunk-wise memory diverge past its training length?

Runs one long validation sequence through a checkpoint and records, per memory
layer and per position bucket:
  - the norm of the memory weight state the retrieve reads (‖M_t‖ / ‖M_0‖)
  - the norm of the retrieved values entering the residual stream
  - the mean decay gate and adaptive step, if the layer exposes them

Both per-token arms are flat to 8x their training length while the chunk-wise
Titans arm explodes past 2x (2026-09-07). Measured on an 8K sequence, the
chunk-wise arm's deeper memory layer runs 82 -> 2.4e7 -> 1.1e9 across the first
three buckets and injects 5.8e8 into the residual stream; the per-token arms'
states decay to ~0.2 and stay there. The two configs differ in five things at
once, so the norms alone cannot name the cause.

`--spectral-norm on|off` is the single-variable intervention: it flips
Newton-Schulz on the LOADED model without touching the weights, so the same
trained network runs with and without the update-magnitude bound. Newton-Schulz
is the prime suspect because it orthogonalizes each update (bounding its norm
by construction) and is off in the chunk-wise config. Run it both ways:

  - `--spectral-norm on` on the chunk-wise checkpoint: if the state stops
    diverging, Newton-Schulz is sufficient to bound it;
  - `--spectral-norm off` on a per-token checkpoint: if the state starts
    diverging, Newton-Schulz is necessary for the bound.

Both together identify the mechanism; either alone leaves the other half open.
What neither can show is whether a chunk-wise model TRAINED with Newton-Schulz
would extrapolate — the intervention runs weights that never saw it, so the
model is out of distribution and only the dynamics of the state are being read,
not the quality of its predictions.

    python state_norm_probe.py --checkpoint DIR --variant titans-mac \
        --val-bin val.bin --seq-len 8192 [--ablation no-omega] [--vanilla] \
        [--spectral-norm on|off]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/gpfs/projects/eporaif01/atlas-torch")

from eval.babilong.evaluate import load_model  # noqa: E402
from titans_pytorch.neural_memory import NeuralMemory  # noqa: E402


def bucket_stats(values, positions, bucket):
    """Mean of `values` (1-D, per position) inside each `bucket`-sized range."""
    out = {}
    for start in range(0, len(values), bucket):
        chunk = values[start:start + bucket]
        if len(chunk):
            out[f"{start}-{start + len(chunk)}"] = float(np.mean(chunk))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="170m")
    p.add_argument("--variant", required=True)
    p.add_argument("--ablation", default=None)
    p.add_argument("--vanilla", action="store_true")
    p.add_argument("--val-bin", required=True)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--bucket", type=int, default=512)
    p.add_argument("--spectral-norm", choices=("keep", "on", "off"), default="keep",
                   help="flip Newton-Schulz on the loaded model (an inference-time intervention on "
                        "trained weights; see the module docstring for what it can and cannot show)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    model, _ = load_model(checkpoint_dir=args.checkpoint, model_size=args.model, variant=args.variant,
                          ablation=args.ablation, device=args.device, vanilla=args.vanilla)
    model.eval()

    tokens = np.memmap(args.val_bin, dtype=np.uint16, mode="r")[: args.seq_len + 1]
    ids = torch.from_numpy(tokens[:-1].astype(np.int64)).unsqueeze(0).to(args.device)

    records = {}

    def hook(layer_index, mem):
        original = mem.retrieve_memories

        def spy(seq, weights, *rest, **kwargs):
            result = original(seq, weights, *rest, **kwargs)
            values = result[0] if isinstance(result, tuple) else result
            state = None
            for tensor in (weights.values() if hasattr(weights, "values") else []):
                # (b n ...) per-token states, or (b ...) one state
                flat = tensor.detach().float()
                norms = flat.flatten(2).norm(dim=-1).mean(0) if flat.ndim > 2 else flat.flatten(1).norm(dim=-1)
                state = norms if state is None else state + norms
            rec = records.setdefault(layer_index, {"state": [], "retrieved": []})
            if state is not None:
                rec["state"].append(state.cpu().numpy())
            rec["retrieved"].append(values.detach().float().flatten(0, -2).norm(dim=-1).cpu().numpy()
                                    if values.ndim > 1 else values.detach().float().cpu().numpy())
            return result

        mem.retrieve_memories = spy

    layer_index = 0
    for layer in model.layers:
        mem = layer[4]
        if isinstance(mem, NeuralMemory):
            if args.spectral_norm != "keep":
                mem.spectral_norm_surprises = args.spectral_norm == "on"
            hook(layer_index, mem)
            layer_index += 1
    if layer_index == 0 and args.spectral_norm != "keep":
        raise ValueError("--spectral-norm has no effect on a model with no memory layers")
    if args.spectral_norm != "keep":
        print(f"intervention: spectral_norm_surprises = {args.spectral_norm == 'on'} on {layer_index} memory layers", flush=True)

    with torch.no_grad():
        model(ids, return_hidden=True)

    summary = {"checkpoint": args.checkpoint, "variant": args.variant, "ablation": args.ablation,
               "seq_len": args.seq_len, "bucket": args.bucket, "spectral_norm": args.spectral_norm, "layers": {}}
    for index, rec in records.items():
        entry = {}
        for key, arrays in rec.items():
            if not arrays:
                continue
            joined = np.concatenate([a.reshape(-1) for a in arrays])
            entry[key] = dict(
                first=float(joined[0]), last=float(joined[-1]),
                ratio_last_first=float(joined[-1] / joined[0]) if joined[0] else float("nan"),
                max=float(np.nanmax(joined)), n=int(joined.size),
                by_bucket=bucket_stats(joined, None, args.bucket),
            )
        summary["layers"][str(index)] = entry

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    for index, entry in summary["layers"].items():
        for key, stats in entry.items():
            buckets = list(stats["by_bucket"].items())
            head = "  ".join(f"{k}:{v:.3g}" for k, v in buckets[:3])
            tail = "  ".join(f"{k}:{v:.3g}" for k, v in buckets[-3:])
            print(f"layer {index} {key:>9}: first {stats['first']:.4g} last {stats['last']:.4g} "
                  f"max {stats['max']:.4g} ratio {stats['ratio_last_first']:.3g}")
            print(f"           buckets: {head}  ...  {tail}")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
