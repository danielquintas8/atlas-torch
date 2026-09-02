"""
Atlas experiment configurations.

Model architectures from Table 7, training recipe from Appendix E.
Paper: Atlas — Learning to Memorize at Test Time (arXiv:2505.23735)

Usage:
    from experiments.configs import get_config

    config = get_config("170m", "atlas-mac")
    model = MemoryAsContextTransformer(**config["model"])
"""

from titans_pytorch.neural_memory import NeuralMemory

# ---------------------------------------------------------------------------
# Model architecture (Table 7)
# ---------------------------------------------------------------------------

MODELS = {
    "170m": dict(dim=768,  depth=12, heads=16),   # dim_head=48
    "340m": dict(dim=1024, depth=24, heads=16),   # dim_head=64
    "760m": dict(dim=1536, depth=24, heads=16),   # dim_head=96
    "1.3b": dict(dim=2048, depth=18, heads=8),    # dim_head=256
}

# ---------------------------------------------------------------------------
# Per-size training hyperparams (Appendix E + Table 7)
# ---------------------------------------------------------------------------
# NOTE: peak LRs follow Atlas Table 7 for BOTH variants. The Titans paper's
# own recipe is AdamW 4e-4 (verified against its arXiv source: "learning
# rate of 4e-4", Experiments section) — running the titans baseline at 3e-3
# is a deliberate same-LR-comparison choice, not the Titans paper's recipe.
# Any headline Titans-vs-Atlas claim should either sweep LR per condition
# or state this choice.
# Second documented baseline-recipe deviation: the Titans paper trained on
# FineWeb-Edu; this project uses FineWeb (per the Atlas paper, Appendix E)
# for BOTH variants.

TRAINING = {
    "170m": dict(peak_lr=3e-3,    total_tokens=15_000_000_000),
    "340m": dict(peak_lr=1.5e-3,  total_tokens=15_000_000_000),
    "760m": dict(peak_lr=1.25e-3, total_tokens=30_000_000_000),
    "1.3b": dict(peak_lr=7e-4,    total_tokens=100_000_000_000),
}

TRAINING_DEFAULTS = dict(
    optimizer="adamw",
    weight_decay=0.1,
    lr_schedule="cosine",
    warmup_steps=2000,
    batch_tokens=500_000,           # 0.5M tokens per batch
    seq_len=4096,                   # 4K training context
    tokenizer="google-t5/t5-base",  # T5 tokenizer, 32K vocab
    vocab_size=32128,               # T5 vocab size (derived from tokenizer)
    dataset="HuggingFaceFW/fineweb",
    grad_clip=1.0,
    bf16=True,
)

# ---------------------------------------------------------------------------
# MAC transformer defaults (from Titans architecture)
# ---------------------------------------------------------------------------

MAC_DEFAULTS = dict(
    segment_len=64,                 # local attention window
    num_persist_mem_tokens=4,       # persistent memory tokens
    num_longterm_mem_tokens=4,      # retrieved memory tokens per segment
    num_residual_streams=4,         # hyper-connection streams
    ff_mult=4,
    sliding_window_attn=True,
    use_flex_attn=True,
    neural_memory_batch_size=1024,  # segment length at which the memory's base weights advance
                                    # (the paper's chunk-size-b analog, Section 3.3) — so it also
                                    # shapes eval-time semantics at long contexts. NOTE: with
                                    # detach_segment_memory the retained store graph is the LAST
                                    # segment, so larger batch_size = MORE retained memory, not less.
    use_axial_pos_emb=False,        # absolute axial positional embedding OFF (2026-09-02). Rotary
                                    # inside the sliding-window attention already carries within-
                                    # window position, the neural memory is position-free, and the
                                    # paper's MAC specifies no absolute embedding. lucidrains' axial
                                    # embedding feeds raw integer segment indices into a SiLU MLP with
                                    # no normalization, so every eval beyond the training length was
                                    # out of distribution: trained at 1K (indices 0-135), the tail
                                    # embedding norm at random init is 7.6x the trained range at 4K,
                                    # 30x at 16K, 243x at 128K, ~1950x at 1M (and 6.7x the token
                                    # embedding norm even in-range). Any model trained with it ON is
                                    # incompatible with this setting.
)

# ---------------------------------------------------------------------------
# Memory module configs
# ---------------------------------------------------------------------------

_ATLAS_DEFAULTS = NeuralMemory.atlas_config()

MEMORY_CONFIGS = {
    # lucidrains' chunk-wise Titans memory: one gradient per store chunk (per-token losses
    # summed at the chunk-start weights), chunk-pooled gates, per-chunk retrieve reading the
    # previous chunk's state (Titans' pre-update read). The BASELINE, not an ablation — the
    # everything-else-equal ablation of the window is ABLATIONS["no-omega"] below.
    "titans": dict(
        momentum=True,
        momentum_order=1,
        qk_rmsnorm=True,
        attn_pool_chunks=True,
        default_step_transform_max_lr=1e-1,
        per_parameter_lr_modulation=True,
        spectral_norm_surprises=False,
        use_accelerated_scan=False,
    ),
    "atlas": dict(
        **_ATLAS_DEFAULTS,
        momentum_order=1,
        attn_pool_chunks=False,     # the per-token path uses per-token params, no chunk pooling
        default_step_transform_max_lr=1e-1,
        per_parameter_lr_modulation=True,
        use_accelerated_scan=False,
        detach_segment_memory=False,  # MUST stay False for training (fixed 2026-09-01): with
                                      # train ctx 1024 the interleaved sequence (1084 tokens)
                                      # splits at neural_memory_batch_size=1024 into [1024, 60]
                                      # and detach cut segment 0's store graph — the learned
                                      # memory init got ZERO outer-loop gradient (frozen at
                                      # random init) and store-side params trained on only the
                                      # last ~60 tokens. that justification is the ctx-1024
                                      # geometry actually trained (slurm/train.sh --seq-len
                                      # 1024); at TRAINING_DEFAULTS seq_len=4096 the interleaved
                                      # sequence splits into 5 segments and the retained
                                      # no-detach graph is substantially larger (directional
                                      # CPU-RSS measurement: ~+50%). full backprop retains the
                                      # full store graph: run a BSC memory smoke at the geometry
                                      # actually launched before the next training run —
                                      # benchmarks/profile_omega_memory.py with --batch-size set
                                      # (without it the profiler builds one segment and cannot
                                      # see this trade-off at all). the flag remains available
                                      # for seq_len >> batch_size experiments where truncated
                                      # BPTT is intended.
        use_sequential_scan=True,  # O(1) memory scan instead of O(n log n) parallel scan
    ),
}

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS = {
    "titans-mac": dict(memory="titans", gate_attn_output=False),
    "titans-mag": dict(memory="titans", gate_attn_output=True),
    "atlas-mac":  dict(memory="atlas",  gate_attn_output=False),
    "atlas-mag":  dict(memory="atlas",  gate_attn_output=True),
}

# ---------------------------------------------------------------------------
# Ablations (applied on top of atlas memory config)
# ---------------------------------------------------------------------------

ABLATIONS = {
    "no-poly":  dict(polynomial_degree=None),
    # atlas minus the window and its gamma gates, nothing else: per_token_updates=True
    # (from atlas_config) keeps the per-token store path — per-token gradients at
    # segment-start weights, per-token gates, eta outside Newton-Schulz, Muon, poly,
    # conv, per-token retrieve. Before 2026-09-02 omega_context=1 silently switched
    # to lucidrains' chunk-wise path (per-chunk gradients and retrieve, pooled gates,
    # eta washed out by Muon), so the ablation changed five things at once. Costs an
    # Atlas run (per-token gradients), not a Titans one — that is the price of the
    # attribution being real. Guarded by test_no_omega_ablation_is_atlas_minus_the_window.
    "no-omega": dict(omega_context=1),
    "no-muon":  dict(spectral_norm_surprises=False),
}


def get_memory_layers(depth: int) -> tuple[int, ...]:
    """Memory on 2 layers — at 1/3 and end of network. Each omega memory layer
    uses ~12 GB in autograd (vmap/grad outputs + scan I/O), and
    torch.utils.checkpoint is incompatible with torch.func.grad, so all
    layers' intermediates are held simultaneously. 2 layers fits on 1×H100
    64GB with room for batch>1.
    """
    first = max(1, depth // 3)
    return (first, depth)


def get_config(
    model_size: str,
    variant: str,
    ablation: str | None = None,
    **overrides,
) -> dict:
    """Build complete config from model size + variant + optional ablation.

    Args:
        model_size: One of "170m", "340m", "760m", "1.3b"
        variant: One of "titans-mac", "titans-mag", "atlas-mac", "atlas-mag"
        ablation: Optional. One of "no-poly", "no-omega", "no-muon"
        **overrides: Override any model/memory/training param.
            Prefix with "model." or "memory." or "training." for scoping,
            or pass flat keys to override MAC_DEFAULTS / memory kwargs.

    Returns:
        dict with keys:
            model: kwargs for MemoryAsContextTransformer
            training: training hyperparams
    """
    assert model_size in MODELS, f"Unknown model: {model_size}. Options: {list(MODELS)}"
    assert variant in VARIANTS, f"Unknown variant: {variant}. Options: {list(VARIANTS)}"

    model_arch = MODELS[model_size]
    variant_def = VARIANTS[variant]
    training = {**TRAINING_DEFAULTS, **TRAINING[model_size]}

    # Memory config
    memory_kwargs = {**MEMORY_CONFIGS[variant_def["memory"]]}
    if ablation:
        assert ablation in ABLATIONS, f"Unknown ablation: {ablation}. Options: {list(ABLATIONS)}"
        memory_kwargs.update(ABLATIONS[ablation])

    # Derived params
    depth = model_arch["depth"]
    dim_head = model_arch["dim"] // model_arch["heads"]
    omega_context = memory_kwargs.get("omega_context", 1)
    neural_memory_segment_len = max(8, omega_context)

    memory_kwargs["dim_head"] = dim_head
    memory_kwargs["heads"] = model_arch["heads"]  # Table 7: same heads for attention and memory

    # Model config (kwargs for MemoryAsContextTransformer)
    model_config = dict(
        num_tokens=training["vocab_size"],
        dim=model_arch["dim"],
        depth=depth,
        dim_head=dim_head,
        heads=model_arch["heads"],
        neural_memory_layers=get_memory_layers(depth),
        neural_memory_segment_len=neural_memory_segment_len,
        neural_mem_gate_attn_output=variant_def["gate_attn_output"],
        neural_memory_kwargs=memory_kwargs,
        **MAC_DEFAULTS,
    )

    # Apply overrides
    for key, value in overrides.items():
        if key.startswith("model."):
            model_config[key[6:]] = value
        elif key.startswith("memory."):
            memory_kwargs[key[7:]] = value
        elif key.startswith("training."):
            training[key[9:]] = value

    return dict(model=model_config, training=training)


# ---------------------------------------------------------------------------
# Convenience: print config summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    size = sys.argv[1] if len(sys.argv) > 1 else "170m"
    variant = sys.argv[2] if len(sys.argv) > 2 else "atlas-mac"
    ablation = sys.argv[3] if len(sys.argv) > 3 else None

    config = get_config(size, variant, ablation)

    # Compute derived info
    model = config["model"]
    training = config["training"]
    mem = model["neural_memory_kwargs"]
    dim_head = model["dim_head"]
    n_mem_layers = len(model["neural_memory_layers"])

    print(f"=== {size} / {variant}" + (f" / {ablation}" if ablation else "") + " ===")
    print(f"Model:    dim={model['dim']}, depth={model['depth']}, heads={model['heads']}, dim_head={dim_head}")
    print(f"Memory:   {n_mem_layers} layers, heads={mem['heads']}, omega={mem.get('omega_context', 1)}, poly={mem.get('polynomial_degree', 'none')}")
    print(f"Training: lr={training['peak_lr']}, tokens={training['total_tokens']/1e9:.0f}B, seq_len={training['seq_len']}")
    print(f"Batch:    {training['batch_tokens']/1e6:.1f}M tokens/batch → {training['batch_tokens'] // training['seq_len']} seqs/batch")
