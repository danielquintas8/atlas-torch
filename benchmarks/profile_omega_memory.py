"""Profile the GPU memory footprint of the Omega Rule path in NeuralMemory.

This is the standalone version of the `ATLAS_LOG_OMEGA_MEM` instrumentation that
used to live inside `NeuralMemory.store_memories`. It was pulled out of the
reference module to keep the hot path clean, but the measurement itself is worth
keeping: the per-NeuralMemory memory cost of the omega-windowed gradient
accumulation is the number any future FSDP / model-parallel implementation has to
beat, and it is what made 4K-context training infeasible on a single GPU.

Unlike the original inline probe (which measured only the windowed-accumulation
delta during the forward), this reports the full omega-path footprint, including
the autograd graph held for backward, which is the dominant and decision-relevant
cost.

Run on a CUDA machine, for example:

    python benchmarks/profile_omega_memory.py --seq-len 1024 --omega-context 8
    python benchmarks/profile_omega_memory.py --seq-len 4096 --chunk-size 8

This is a development and benchmarking tool, not part of the library.
"""

from __future__ import annotations

import argparse

import torch

from titans_pytorch import NeuralMemory

GB = 1e9


def profile_omega_memory(
    dim: int,
    heads: int,
    seq_len: int,
    omega_context: int,
    chunk_size: int,
    batch: int,
) -> dict[str, float]:
    """Run one omega-path forward and backward and report the memory it costs.

    Returns a dict of GB measurements: allocated before, allocated after, and the
    peak across the forward and backward (the number that has to fit on the GPU).
    """
    if not torch.cuda.is_available():
        raise SystemExit("This profiler requires a CUDA device.")

    if omega_context > chunk_size:
        raise SystemExit(
            f"omega_context ({omega_context}) must be <= chunk_size ({chunk_size}); "
            f"pass --chunk-size {omega_context} or larger."
        )

    if dim % heads != 0:
        raise SystemExit(
            f"dim ({dim}) must be divisible by heads ({heads}) so dim_head = dim // heads is exact."
        )

    device = "cuda"

    # Match the multi-head split used by the training configs (dim_head = dim / heads),
    # then enable the omega window at the requested length.
    config = NeuralMemory.atlas_config()
    config["omega_context"] = omega_context

    memory = NeuralMemory(
        dim=dim,
        dim_head=dim // heads,
        heads=heads,
        chunk_size=chunk_size,
        **config,
    ).to(device)

    seq = torch.randn(batch, seq_len, dim, device=device)

    torch.cuda.reset_peak_memory_stats()
    before_gb = torch.cuda.memory_allocated() / GB

    retrieved, _ = memory(seq)            # exercises the omega store + per-token retrieve
    loss = retrieved.float().pow(2).mean()
    loss.backward()                       # holds the windowed-gradient graph

    after_gb = torch.cuda.memory_allocated() / GB
    peak_gb = torch.cuda.max_memory_allocated() / GB

    return {"before_gb": before_gb, "after_gb": after_gb, "peak_gb": peak_gb}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--omega-context", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = profile_omega_memory(
        dim=args.dim,
        heads=args.heads,
        seq_len=args.seq_len,
        omega_context=args.omega_context,
        chunk_size=args.chunk_size,
        batch=args.batch,
    )
    print(
        f"OMEGA_MEM dim={args.dim} heads={args.heads} seq_len={args.seq_len} "
        f"omega_context={args.omega_context} chunk_size={args.chunk_size} batch={args.batch} | "
        f"before={result['before_gb']:.2f}GB after={result['after_gb']:.2f}GB "
        f"peak={result['peak_gb']:.2f}GB"
    )


if __name__ == "__main__":
    main()
