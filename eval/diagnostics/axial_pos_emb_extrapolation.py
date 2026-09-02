"""
Axial positional-embedding extrapolation check — confirm the OOD finding on a
real checkpoint (or reproduce it at random init).

The MAC's ContinuousAxialPositionalEmbedding feeds raw integer segment indices
(arange(ceil(seq_len / neural_memory_segment_len))) into a SiLU MLP with no
normalization. A model trained at 1K context saw outer-axis inputs 0-135; a 4K
eval feeds 0-512 and a 1M eval 0-125K. At random init (3 seeds) the embedding
norm at the tail is 7.6x the trained-range mean at 4K, 30x at 16K, 243x at
128K and ~1950x at 1M — and 6.7x the token-embedding norm even in-range
(2026-09-02). Those ratios are at raw token positions; the model feeds the
interleaved longterm-mem positions (seq_len_with_longterm_mem), where the 4K
tail is position 4347 -> ~8.0x, 16K -> ~32x, 128K -> ~257x, 1M -> ~2060x — the
tool reports both. Nothing in training pushes an unbounded integer->MLP map toward
saturation, but the trained weights are what matter: run this against a
checkpoint trained with use_axial_pos_emb=True to confirm. The experiment
config now disables the embedding (MAC_DEFAULTS use_axial_pos_emb=False).

Usage:
    # real checkpoint (must have been trained with the embedding on):
    python eval/diagnostics/axial_pos_emb_extrapolation.py \
        --model 170m --variant atlas-mac --checkpoint runs/170m-atlas-mac/step-4000

    # random init, several seeds:
    python eval/diagnostics/axial_pos_emb_extrapolation.py \
        --model 170m --variant atlas-mac --random-init --seeds 3
"""

import argparse
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axial_positional_embedding import ContinuousAxialPositionalEmbedding

from experiments.configs import get_config
from experiments.train import apply_memory_kwargs, parse_memory_kwargs

AXIAL_PREFIX = "axial_pos_emb."


def load_state_dict(checkpoint_dir):
    for name in ["model.safetensors", "pytorch_model.bin"]:
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            if path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(path, device="cpu")
            else:
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
            return {k.removeprefix("module."): v for k, v in state_dict.items()}
    raise FileNotFoundError(f"No model.safetensors / pytorch_model.bin in {checkpoint_dir}")


@torch.no_grad()
def norms_at(pos_emb, positions, stride):
    pos = torch.tensor(positions, dtype=torch.long)
    return pos_emb.forward_with_pos(pos, (stride,)).norm(dim=-1)


@torch.no_grad()
def report(pos_emb, token_emb_norm, train_seq_len, positions, stride, label):
    train_mean = norms_at(pos_emb=pos_emb, positions=list(range(train_seq_len)), stride=stride).mean().item()
    tail = norms_at(pos_emb=pos_emb, positions=positions, stride=stride)
    print(f"{label}: trained-range (0..{train_seq_len - 1}) mean pos-emb norm = {train_mean:.2f}"
          + (f"  ({train_mean / token_emb_norm:.2f}x the token-embedding row norm {token_emb_norm:.2f})"
             if token_emb_norm is not None else ""))
    print(f"{'position':>12}{'norm':>12}{'x trained':>12}")
    for position, norm in zip(positions, tail.tolist()):
        print(f"{position:>12}{norm:>12.2f}{norm / train_mean:>12.1f}")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Axial positional-embedding extrapolation check")
    p.add_argument("--model", required=True, choices=["170m", "340m", "760m", "1.3b"])
    p.add_argument("--variant", required=True,
                   choices=["titans-mac", "titans-mag", "atlas-mac", "atlas-mag"])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Checkpoint dir trained with use_axial_pos_emb=True")
    src.add_argument("--random-init", action="store_true", help="Reproduce the finding at random init")
    p.add_argument("--memory-kwarg", action="append", default=[], metavar="KEY=VALUE",
                   help="Mirror of train.py --memory-kwarg for a checkpoint trained with overrides.")
    p.add_argument("--seeds", type=int, default=3, help="Seeds for --random-init")
    p.add_argument("--train-seq-len", type=int, default=1084,
                   help="Positions the model actually saw in training (1024 tokens + longterm mem tokens = 1084)")
    p.add_argument("--positions", type=int, nargs="+", default=[4095, 16383, 131071, 1048575],
                   help="TOKEN positions (0-indexed) to probe. They are converted to the positions the "
                        "model actually feeds the embedding — the interleaved longterm-mem tokens push "
                        "every token later (token 4095 is fed at position 4347 in the 170m config, "
                        "~8.0x rather than the 7.6x quoted for the raw position); both are reported.")
    p.add_argument("--stride", type=int, default=None,
                   help="Axial stride (default: config neural_memory_segment_len)")
    return p.parse_args()


def fed_position(token_position, segment_len, num_longterm_mem_tokens):
    """Position at which token `token_position` reaches the embedding once the
    longterm-mem tokens are interleaved (mirrors
    MemoryAsContextTransformer.seq_len_with_longterm_mem for a length of
    token_position + 1, minus one)."""
    seq_len = token_position + 1
    return ((seq_len - 1) // segment_len) * num_longterm_mem_tokens + seq_len - 1


def main():
    args = parse_args()
    config = get_config(model_size=args.model, variant=args.variant)
    model_cfg = apply_memory_kwargs(config=config, overrides=parse_memory_kwargs(items=args.memory_kwarg))["model"]
    dim = model_cfg["dim"]
    stride = args.stride or model_cfg["neural_memory_segment_len"]
    fed = [
        fed_position(
            token_position=pos,
            segment_len=model_cfg["segment_len"],
            num_longterm_mem_tokens=model_cfg["num_longterm_mem_tokens"],
        )
        for pos in args.positions
    ]
    print("token position -> fed position (interleaved longterm-mem tokens included): "
          + ", ".join(f"{t} -> {f}" for t, f in zip(args.positions, fed)))
    print()

    if args.checkpoint:
        state_dict = load_state_dict(checkpoint_dir=args.checkpoint)
        axial = {k[len(AXIAL_PREFIX):]: v for k, v in state_dict.items() if k.startswith(AXIAL_PREFIX)}
        if not axial:
            raise SystemExit(
                f"{args.checkpoint} carries no {AXIAL_PREFIX}* weights — trained with "
                f"use_axial_pos_emb=False; nothing to confirm."
            )
        pos_emb = ContinuousAxialPositionalEmbedding(dim=dim, num_axial_dims=2)
        pos_emb.load_state_dict(axial)
        token_emb_norm = None
        if "token_emb.weight" in state_dict:
            token_emb_norm = state_dict["token_emb.weight"].float().norm(dim=-1).mean().item()
        report(pos_emb=pos_emb, token_emb_norm=token_emb_norm, train_seq_len=args.train_seq_len,
               positions=fed, stride=stride, label=f"checkpoint {args.checkpoint}")
        return

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        pos_emb = ContinuousAxialPositionalEmbedding(dim=dim, num_axial_dims=2)
        token_emb_norm = nn.Embedding(model_cfg["num_tokens"], dim).weight.norm(dim=-1).mean().item()
        report(pos_emb=pos_emb, token_emb_norm=token_emb_norm, train_seq_len=args.train_seq_len,
               positions=fed, stride=stride, label=f"random init seed {seed}")


if __name__ == "__main__":
    main()
