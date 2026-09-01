"""
Atlas training script.

Usage:
    # Single GPU:
    python experiments/train.py --model 170m --variant atlas-mac

    # Multi-GPU (local):
    accelerate launch --num_processes 4 --mixed_precision bf16 \
        experiments/train.py --model 170m --variant atlas-mac --wandb

    # BSC — see experiments/slurm/ for job scripts
"""

import argparse
import math
import os
import re
import shutil
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset

# accelerate is imported lazily inside main() (same pattern as datasets /
# transformers in load_data) so unit tests can import MemmapTokenDataset and
# cosine_with_warmup from this module in environments without accelerate.

# allow running from repo root: `python experiments/train.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from titans_pytorch import MemoryAsContextTransformer
from experiments.configs import get_config


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class MemmapTokenDataset(IterableDataset):
    """Pre-tokenized binary data (from experiments/data/prepare.py).

    Reads uint16 memmap, yields (seq_len + 1,) int64 chunks.

    Deliberately does NOT self-shard by rank. Multi-GPU distribution is owned
    by Accelerate's dataloader dispatch: a single reader on the main process
    iterates this dataset and slices batches across ranks. Rank striding here
    combined with that dispatch double-sharded the data — only the main
    process's 1/world_size shard was ever read, cutting effective epochs to
    a quarter on 4 GPUs (found 2026-09-01). Do not reintroduce it.

    The per-epoch permutation is a pure function of `seed + epoch`, drawn
    from a private numpy Generator — NEVER the process-global `random` state.
    That is what makes resume possible at all: accelerator.load_state
    restores the global RNGs to their checkpoint-time (post-shuffle) state,
    so a global-random shuffle after resume draws the NEXT permutation, not
    the one the checkpointed run was mid-way through (probe-proven
    2026-09-01: 9/9 post-resume batches mismatched; a seed-derived
    permutation matches 9/9). The epoch is normally driven by Accelerate's
    dispatcher, which calls set_epoch(iteration) at the start of every pass;
    plain iteration falls back to an internal counter.

    `skip_chunks` seeks past the first N permutation entries on the FIRST
    yielded pass only (pure index arithmetic — no reads, no broadcasts),
    replacing accelerate's skip_first_batches, which fetched and broadcast
    every skipped batch. `limit_chunks` caps the dataset to its first N
    chunks (used to give validation a finite, fixed slice).
    """

    def __init__(
        self,
        bin_path,
        seq_len,
        shuffle=True,
        seed=0,
        epoch_base=0,
        skip_chunks=0,
        limit_chunks=None,
    ):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.chunk_len = seq_len + 1
        self.n_chunks = len(self.data) // self.chunk_len
        if limit_chunks is not None:
            self.n_chunks = min(self.n_chunks, limit_chunks)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch_base = epoch_base
        self._skip_chunks = skip_chunks
        self._dispatcher_epoch = None
        self._iter_count = 0

    def set_epoch(self, epoch):
        # called by Accelerate's DataLoaderDispatcher / DataLoaderShard at the
        # start of every pass (epoch = its internal iteration counter)
        self._dispatcher_epoch = epoch

    def __iter__(self):
        if self._dispatcher_epoch is not None:
            epoch = self.epoch_base + self._dispatcher_epoch
        else:
            epoch = self.epoch_base + self._iter_count
        self._iter_count += 1

        if self.shuffle:
            perm = np.random.default_rng(self.seed + epoch).permutation(self.n_chunks)
        else:
            perm = np.arange(self.n_chunks)

        start_at = self._skip_chunks
        self._skip_chunks = 0  # seek applies to the first yielded pass only

        for idx in perm[start_at:]:
            start = int(idx) * self.chunk_len
            chunk = self.data[start : start + self.chunk_len].astype(np.int64)
            yield torch.from_numpy(chunk)


class StreamingTokenDataset(IterableDataset):
    """Tokenize HuggingFace dataset on the fly (needs internet)."""

    def __init__(self, hf_dataset, tokenizer, seq_len):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.chunk_len = seq_len + 1

    def __iter__(self):
        buffer = []
        for example in self.hf_dataset:
            tokens = self.tokenizer.encode(example["text"])
            buffer.extend(tokens)
            while len(buffer) >= self.chunk_len:
                yield torch.tensor(buffer[: self.chunk_len], dtype=torch.long)
                buffer = buffer[self.chunk_len :]


def load_data(
    data_dir,
    seq_len,
    split="train",
    seed=0,
    epoch_base=0,
    skip_chunks=0,
    limit_chunks=None,
):
    """Load dataset — binary format (BSC) or streaming HF.

    Returns None if the requested split doesn't exist (or, for val, exists
    but holds less than one full chunk — a 0-byte val.bin would otherwise
    crash np.memmap at startup).
    """
    bin_path = os.path.join(data_dir, f"{split}.bin")
    if os.path.exists(bin_path):
        if os.path.getsize(bin_path) < (seq_len + 1) * 2:  # uint16 = 2 bytes
            if split == "train":
                raise ValueError(
                    f"{bin_path} holds less than one {seq_len + 1}-token chunk — "
                    f"broken or empty training data"
                )
            return None  # val: too small to score → validation disabled
        return MemmapTokenDataset(
            bin_path,
            seq_len,
            shuffle=(split == "train"),
            seed=seed,
            epoch_base=epoch_base,
            skip_chunks=skip_chunks,
            limit_chunks=limit_chunks,
        )

    if split != "train":
        return None  # no val split for streaming

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer_path = os.path.join(data_dir, "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

    hf_ds = load_dataset(data_dir, split="train", streaming=True)
    return StreamingTokenDataset(hf_ds, tokenizer, seq_len)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_schedule(train_cfg, args, num_gpus):
    """Derive the run's schedule shape from config + CLI + world size.

    Returns (grad_accum, batch_tokens, schedule_steps, max_steps,
    warmup_steps). Pure so it is unit-testable — in particular: warmup comes
    from config or the CLI override and is NEVER zeroed on resume. The
    scheduler's step counter is restored from the checkpoint, so the original
    schedule continues exactly and no re-warmup happens; the old
    warmup=0-on-resume rebuild did not prevent re-warmup (the restored
    counter already does) — it swapped in a different cosine, measured
    -1.6% to -12.2% LR vs the intended schedule over the run (2026-09-01).
    """
    seq_len = train_cfg["seq_len"]

    # gradient accumulation: match the 0.5M-token batch target
    tokens_per_micro = num_gpus * args.per_device_batch_size * seq_len
    grad_accum = args.grad_accum or max(1, train_cfg["batch_tokens"] // tokens_per_micro)
    batch_tokens = tokens_per_micro * grad_accum

    schedule_steps = int(train_cfg["total_tokens"] // batch_tokens)
    max_steps = args.max_steps or schedule_steps

    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = train_cfg["warmup_steps"]

    return grad_accum, batch_tokens, schedule_steps, max_steps, warmup_steps


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_param_groups(model, weight_decay):
    """Split parameters into decay / no-decay groups for AdamW.

    The rule, exactly: a parameter DECAYS iff ndim >= 2 AND its qualified
    name carries none of the no-decay tags AND it is not a norm gain
    (name ending in `.gamma` or containing `_norm`). Everything else gets
    wd=0.0. What that protects, and why:

    - ndim < 2: biases, gate scalars, the polynomial Taylor coefficients
      (uniform decay pulled them toward 0, not their 1/d! init).
    - embedding-like tables (token_emb, longterm_mems, persistent_memory,
      axial_pos_emb): weight decay on embeddings shrinks their norms and
      inflates early-layer gradients via the LayerNorm Jacobian (OLMo-2
      mechanism).
    - hyper-connection routing (static_alpha, dynamic_alpha_fn,
      stream_embed): static_alpha is initialized one-hot + eye — the
      residual-stream identity routing — and decoupled decay pulls it
      toward the ZERO matrix at lr*wd ~ 3e-4/step, dismantling the
      residual path.
    - norm gains with ndim >= 2 (q_norm/k_norm gammas are (heads, 1, dim)):
      (gamma + 1)-parameterized, so decay pulls toward identity scale —
      benign, but they are gains, not matmul weights.

    Known gap: the neural memory's per-head norm gain lives inside the
    memory_model_parameters ParameterList, whose names carry no `norm` tag —
    it stays in the decay group. Benign for the same (gamma + 1) reason.

    Field standard is decay-matrices-only; the previous uniform wd=0.1
    decayed all of the above (found 2026-09-01).
    """
    no_decay_tags = (
        # embedding-like tables
        "token_emb",
        "longterm_mems",
        "persistent_memory",
        "axial_pos_emb",
        # hyper-connection residual-stream routing
        "static_alpha",
        "dynamic_alpha_fn",
        "stream_embed",
    )

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        tagged = any(tag in name for tag in no_decay_tags)
        is_norm_gain = name.endswith(".gamma") or "_norm" in name
        if param.ndim >= 2 and not tagged and not is_norm_gain:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return [
        dict(params=decay_params, weight_decay=weight_decay),
        dict(params=no_decay_params, weight_decay=0.0),
    ]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(accelerator, step, output_dir, schedule_meta):
    """Save accelerate state + meta.pt marking the checkpoint complete.

    meta.pt is written by the main process AFTER a barrier, so its presence
    marks a fully-written checkpoint (all ranks done) — prune_checkpoints
    only counts dirs that have it. It carries the schedule-shape fields
    (warmup_steps, schedule_steps, batch_tokens, grad_accum, world_size) so
    resume can refuse a run whose shape silently changed.
    """
    ckpt_dir = os.path.join(output_dir, f"step-{step}")
    accelerator.save_state(ckpt_dir)
    accelerator.wait_for_everyone()  # all ranks finished writing state files
    if accelerator.is_main_process:
        meta = dict(step=step)
        meta.update(schedule_meta)
        torch.save(meta, os.path.join(ckpt_dir, "meta.pt"))
    accelerator.print(f"Checkpoint saved → {ckpt_dir}")


def read_checkpoint_meta(resume_dir):
    """Read meta.pt without touching accelerate (usable before Accelerator)."""
    return torch.load(
        os.path.join(resume_dir, "meta.pt"), map_location="cpu", weights_only=True
    )


def prune_checkpoints(accelerator, output_dir, keep):
    """Delete the oldest complete step-* checkpoint dirs beyond the newest `keep`.

    Opt-in via --keep-checkpoints: the eval workflow scores multiple
    historical checkpoints, so nothing is deleted by default. At
    --save-every 100 a full 15B run writes ~300 checkpoints x ~2-3 GB
    (600-900 GB of GPFS).

    Safety: only directories matching step-<digits> inside output_dir are
    ever touched, and only dirs containing meta.pt count toward the keep
    window — a partial save (killed mid-write, before the post-barrier
    meta.pt) can neither be kept as if good nor push a good checkpoint out.
    Accelerate's own rotation (ProjectConfiguration(total_limit=...)) is not
    used because it forces checkpoints/checkpoint_<i> naming; the eval
    workflow reads step-<N>.
    """
    if keep is None or keep < 1:
        return
    # every rank reaches this barrier (the guard above is rank-invariant);
    # deletion must not race non-main ranks still inside save_state
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    pattern = re.compile(r"^step-(\d+)$")
    step_dirs = []
    for entry in os.listdir(output_dir):
        match = pattern.match(entry)
        full = os.path.join(output_dir, entry)
        if match and os.path.isdir(full) and os.path.exists(os.path.join(full, "meta.pt")):
            step_dirs.append((int(match.group(1)), full))

    step_dirs.sort()  # numerically, oldest first
    for _, path in step_dirs[:-keep]:
        shutil.rmtree(path)
        accelerator.print(f"Pruned checkpoint {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Atlas training")
    p.add_argument("--model", required=True, choices=["170m", "340m", "760m", "1.3b"])
    p.add_argument(
        "--variant",
        required=True,
        choices=["titans-mac", "titans-mag", "atlas-mac", "atlas-mag"],
    )
    p.add_argument(
        "--ablation", default=None, choices=["no-poly", "no-omega", "no-muon"]
    )
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="Checkpoint dir to resume from")
    p.add_argument("--data-dir", default=None, help="Path to pre-tokenized data (from prepare.py) or HF dataset")
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="atlas-torch")
    p.add_argument("--validate-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=None, help="Override training sequence length")
    p.add_argument("--grad-accum", type=int, default=None, help="Override gradient accumulation steps")
    p.add_argument("--peak-lr", type=float, default=None, help="Override peak learning rate")
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override warmup steps (schedule shape). On resume the restored "
        "scheduler step counter continues the original schedule — do NOT zero "
        "this to 'skip' warmup, it rebuilds a different cosine instead.",
    )
    p.add_argument(
        "--find-unused-params",
        action="store_true",
        help="Enable DDP find_unused_parameters (escape hatch — only needed if "
        "some parameters receive no gradients, e.g. detach_segment_memory "
        "geometries that truncate the store graph)",
    )
    p.add_argument(
        "--keep-checkpoints",
        type=int,
        default=None,
        help="Keep only the newest N step-* checkpoints, deleting older ones "
        "after each save. Default None keeps all — the eval workflow scores "
        "multiple historical checkpoints, so rotation is opt-in.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.keep_checkpoints is not None and args.keep_checkpoints < 1:
        p.error("--keep-checkpoints must be >= 1 (omit the flag to keep all checkpoints)")
    return args


def main():
    # lazy import — see the note at the top of the module
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import set_seed

    args = parse_args()
    config = get_config(args.model, args.variant, args.ablation)
    train_cfg = config["training"]
    if args.seq_len:
        train_cfg["seq_len"] = args.seq_len
    if args.peak_lr:
        train_cfg["peak_lr"] = args.peak_lr
    seq_len = train_cfg["seq_len"]

    run_name = args.run_name or (
        f"{args.model}-{args.variant}"
        + (f"-{args.ablation}" if args.ablation else "")
    )
    output_dir = os.path.join(args.output_dir, run_name)

    # Schedule shape — pure function of config + CLI + world size (see
    # compute_schedule for the warmup-on-resume rationale). schedule_meta is
    # written into every checkpoint's meta.pt and validated on resume.
    num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum, batch_tokens, schedule_steps, max_steps, warmup_steps = compute_schedule(
        train_cfg=train_cfg, args=args, num_gpus=num_gpus
    )
    schedule_meta = dict(
        warmup_steps=warmup_steps,
        schedule_steps=schedule_steps,
        batch_tokens=batch_tokens,
        grad_accum=grad_accum,
        world_size=num_gpus,
    )

    # Accelerator
    # find_unused_parameters defaults False: all parameters receive gradients
    # now that detach_segment_memory is off (the detached first store segment
    # was what left params without gradients — the old comment blamed vmap's
    # functional interface, which was never the cause; probe-verified
    # 2026-09-01). The flag costs a full graph traversal per iteration. If
    # DDP errors with "expected to have finished reduction" on a future
    # config, rerun with --find-unused-params and find which param went unused.
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=args.find_unused_params
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        log_with="wandb" if args.wandb else None,
        project_dir=output_dir,
        mixed_precision="bf16" if train_cfg["bf16"] else "no",
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(args.seed)

    # WORLD_SIZE drives the schedule arithmetic above; a bare `python
    # train.py` under srun would silently compute a different batch_tokens
    assert num_gpus == accelerator.num_processes, (
        f"WORLD_SIZE={num_gpus} but accelerator sees "
        f"{accelerator.num_processes} processes — launch via `accelerate launch`"
    )

    # Resume: peek meta.pt early — start_step drives the data seek below, and
    # the schedule shape must match before any state is restored. Changing
    # GPU count / batch size / warmup mid-run rebuilds a different cosine
    # against the restored step counter — the same failure class as the old
    # warmup-zeroing.
    start_step = 0
    if args.resume:
        resume_meta = read_checkpoint_meta(resume_dir=args.resume)
        start_step = resume_meta["step"]
        missing = [key for key in schedule_meta if key not in resume_meta]
        if missing:
            accelerator.print(
                f"WARNING: checkpoint meta.pt lacks schedule fields {missing} "
                f"(pre-2026-09 format) — cannot validate the schedule shape"
            )
        else:
            for field, current in schedule_meta.items():
                saved = resume_meta[field]
                if saved != current:
                    raise ValueError(
                        f"resume schedule mismatch on '{field}': checkpoint has "
                        f"{saved}, this run computes {current}. Relaunch with "
                        f"the original geometry (GPU count, batch size, "
                        f"grad-accum, warmup)."
                    )

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    accelerator.print(f"=== {run_name} ===")
    accelerator.print(
        f"GPUs={num_gpus}  batch/dev={args.per_device_batch_size}  "
        f"grad_accum={grad_accum}  batch_tokens={batch_tokens/1e6:.2f}M"
    )
    accelerator.print(
        f"max_steps={max_steps:,}  schedule_steps={schedule_steps:,}  warmup={warmup_steps}  "
        f"lr={train_cfg['peak_lr']}  wd={train_cfg['weight_decay']}"
    )

    # Data — on resume the stream SEEKS instead of read-and-discarding:
    # permutations are a pure function of seed + epoch (MemmapTokenDataset),
    # so the resumed position is reconstructed from (a) epoch_base = full
    # epochs already consumed, (b) skip_chunks = chunks consumed within the
    # current epoch. The dispatcher consumes per_device_batch_size * world
    # chunks per per-rank yield. Approximation: a resume that falls inside an
    # epoch's trailing partial-batch region (drop_last=False padding) is off
    # by at most one batch.
    data_dir = args.data_dir or train_cfg["dataset"]
    epoch_base, skip_chunks = 0, 0
    train_bin = os.path.join(data_dir, "train.bin")
    if start_step > 0 and os.path.exists(train_bin):
        n_chunks = os.path.getsize(train_bin) // ((seq_len + 1) * 2)
        chunks_per_yield = args.per_device_batch_size * num_gpus
        yields_per_epoch = n_chunks // chunks_per_yield
        assert yields_per_epoch > 0, (
            f"train.bin holds {n_chunks} chunks — fewer than one "
            f"{chunks_per_yield}-chunk yield"
        )
        total_yields = start_step * grad_accum
        epoch_base, yields_within = divmod(total_yields, yields_per_epoch)
        assert yields_within < yields_per_epoch
        skip_chunks = yields_within * chunks_per_yield
        accelerator.print(
            f"Resume data seek: epoch {epoch_base}, skipping {skip_chunks} chunks "
            f"({yields_within}/{yields_per_epoch} yields into the epoch)"
        )
    elif start_step > 0:
        accelerator.print(
            "Streaming dataset: resume restarts the stream (no seek support)"
        )

    train_dataset = load_data(
        data_dir,
        seq_len,
        split="train",
        seed=args.seed,
        epoch_base=epoch_base,
        skip_chunks=skip_chunks,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_batch_size)

    # Model
    model = MemoryAsContextTransformer(**config["model"])
    param_count = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Optimizer + scheduler
    # betas set explicitly: neither the Titans nor the Atlas paper pins them;
    # every reference pretraining config (Llama 1-3, DeepSeek V1-V3,
    # SmolLM2/3, Salamandra) uses beta2=0.95. torch's default 0.999 adapts
    # the second moment too slowly at peak_lr 3e-3 and is spike-prone.
    # Weight decay applies to matmul weights only — see build_param_groups.
    # NOTE: optimizer state from pre-existing checkpoints will not load
    # (param-group count changed) — those checkpoints are already declared
    # incompatible on this branch (omega window fix).
    optimizer = AdamW(
        build_param_groups(model=model, weight_decay=train_cfg["weight_decay"]),
        lr=train_cfg["peak_lr"],
        betas=(0.9, 0.95),
    )

    # The scheduler is deliberately NOT passed through accelerator.prepare.
    # A prepared AcceleratedScheduler advances num_processes times per .step()
    # call, which required scaling warmup/total by num_processes to compensate
    # (the uncompensated form ramped LR 4x too fast and caused the loss spike
    # at step ~290 in job 40107853; the compensated form was verified exact on
    # accelerate 1.14 — but it couples the schedule to an Accelerate internal).
    # Keeping the scheduler unwrapped and stepping it exactly once per
    # completed optimization step is exact by construction on any version.
    # register_for_checkpointing carries its state through save_state /
    # load_state. Checkpoints from before this change stored the scheduler
    # under the prepared name; they are already incompatible with this branch
    # (omega window fix), so there is no migration shim.
    scheduler = cosine_with_warmup(
        optimizer=optimizer, warmup_steps=warmup_steps, total_steps=schedule_steps
    )

    # Prepare
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    accelerator.register_for_checkpointing(scheduler)

    # Validation loader — created and prepared ONCE; each validation pass
    # re-iterates it. (The old per-validation accelerator.prepare re-wrapped a
    # fresh loader every time, growing the accelerator's in-memory dataloader
    # registry — nothing dataloader-related is written to checkpoints.)
    # limit_chunks caps val to a fixed slice sized for exactly 50 per-rank
    # yields, so each pass EXHAUSTS naturally: no mid-iteration break, hence
    # no lingering GradientState references and no masked end-of-dataloader
    # sync from an unfinished dispatcher generator.
    val_yields = 50
    val_dataset = load_data(
        data_dir,
        seq_len,
        split="val",
        limit_chunks=val_yields * args.per_device_batch_size * num_gpus,
    )
    if val_dataset is not None and val_dataset.n_chunks > 0:
        val_loader = accelerator.prepare(
            DataLoader(val_dataset, batch_size=args.per_device_batch_size)
        )
    else:
        val_loader = None
        accelerator.print("No usable val.bin — validation disabled")

    # Resume — restore model/optimizer/scheduler/RNG state. The data stream
    # is deliberately NOT positioned from here: load_state restores the
    # global RNGs to their checkpoint-time (post-shuffle) state, which is
    # exactly why the data permutation must never depend on them — the seek
    # above already positioned the stream (see MemmapTokenDataset).
    if args.resume:
        accelerator.load_state(args.resume)
        accelerator.print(f"Resumed from step {start_step}")

    # Wandb
    if args.wandb:
        accelerator.init_trackers(
            args.wandb_project,
            config={
                "model_size": args.model,
                "variant": args.variant,
                "ablation": args.ablation,
                "max_steps": max_steps,
                "batch_tokens": batch_tokens,
                "peak_lr": train_cfg["peak_lr"],
                "params": param_count,
            },
            init_kwargs={"wandb": {"name": run_name}},
        )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    model.train()

    # the resume seek lives inside MemmapTokenDataset (seed⊕epoch permutation
    # + first-pass chunk skip) — no skip_first_batches wrapper: it fetched
    # and broadcast every skipped batch, re-skipped on every re-iteration,
    # and could not carry a skip across an epoch boundary
    train_iter = iter(train_loader)

    global_step = start_step
    tokens_this_run = 0
    # initialized before the loop: a requeued job resuming at/after max_steps
    # never runs the body, and the closing print still reads tokens_total
    tokens_total = start_step * batch_tokens
    running_loss = torch.zeros((), device=accelerator.device)
    loss_count = 0
    t0 = time.time()

    while global_step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(model):
            loss = model(batch, return_loss=True)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(), train_cfg["grad_clip"]
                )

            optimizer.step()
            optimizer.zero_grad()

        # step scheduler only on full optimization steps (not micro-batches)
        if accelerator.sync_gradients:
            scheduler.step()

        # accumulate on device — .item() here would force a host sync on every
        # micro-batch (~grad_accum syncs per optimization step)
        running_loss += loss.detach()
        loss_count += 1

        if not accelerator.sync_gradients:
            continue

        # Full optimization step completed
        global_step += 1
        tokens_this_run += batch_tokens
        tokens_total = start_step * batch_tokens + tokens_this_run

        # Log
        if global_step % args.log_every == 0:
            avg_loss = (running_loss / loss_count).item()
            running_loss.zero_()
            loss_count = 0

            elapsed = time.time() - t0
            tok_per_sec = tokens_this_run / elapsed if elapsed > 0 else 0
            lr = scheduler.get_last_lr()[0]

            accelerator.print(
                f"step {global_step:>7d} | loss {avg_loss:.4f} | "
                f"ppl {math.exp(min(avg_loss, 20)):.1f} | lr {lr:.2e} | "
                f"{tok_per_sec / 1e3:.1f}k tok/s | {tokens_total / 1e9:.3f}B"
            )

            if args.wandb:
                accelerator.log(
                    {
                        "train/loss": avg_loss,
                        "train/perplexity": math.exp(min(avg_loss, 20)),
                        "train/lr": lr,
                        "train/tokens_per_sec": tok_per_sec,
                        "train/tokens_seen": tokens_total,
                    },
                    step=global_step,
                )

        # Validate — re-iterates the once-prepared val_loader. The dataset is
        # capped via limit_chunks (unshuffled, same fixed slice every pass),
        # so the loop exhausts naturally — no mid-iteration break
        if global_step % args.validate_every == 0 and val_loader is not None:
            model.eval()

            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_loss = model(val_batch, return_loss=True)
                    val_losses.append(
                        accelerator.gather(val_loss).mean().item()
                    )

            avg_val = sum(val_losses) / max(1, len(val_losses))
            accelerator.print(
                f"step {global_step:>7d} | val_loss {avg_val:.4f} | "
                f"val_ppl {math.exp(min(avg_val, 20)):.1f}"
            )

            if args.wandb:
                accelerator.log(
                    {
                        "val/loss": avg_val,
                        "val/perplexity": math.exp(min(avg_val, 20)),
                    },
                    step=global_step,
                )

            model.train()

        # Checkpoint
        if global_step % args.save_every == 0:
            save_checkpoint(
                accelerator=accelerator,
                step=global_step,
                output_dir=output_dir,
                schedule_meta=schedule_meta,
            )
            prune_checkpoints(
                accelerator=accelerator,
                output_dir=output_dir,
                keep=args.keep_checkpoints,
            )

    # Final save
    save_checkpoint(
        accelerator=accelerator,
        step=global_step,
        output_dir=output_dir,
        schedule_meta=schedule_meta,
    )
    prune_checkpoints(
        accelerator=accelerator, output_dir=output_dir, keep=args.keep_checkpoints
    )

    # Peak GPU memory — used by Phase 0 smoke runs to verify the asymmetric
    # MLP path fits comfortably under the H100 budget before committing to a
    # full retrain. Logs from rank 0 only.
    #
    # If peak exceeds ATLAS_PEAK_MEM_FAIL_GB (default 60 on H100 64GB), exit
    # non-zero so SLURM marks the job FAILED with a clear message — catches
    # configurations that "barely" fit on the smoke but would OOM at scale,
    # before we commit days of compute to the full retrain.
    if torch.cuda.is_available() and accelerator.is_main_process:
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        accelerator.print(f"PEAK_GPU_MEM_GB: {peak_gb:.2f}")
        peak_fail_gb = float(os.environ.get("ATLAS_PEAK_MEM_FAIL_GB", "60"))
        if peak_gb > peak_fail_gb:
            accelerator.print(
                f"PEAK_GPU_MEM_FAIL: peak {peak_gb:.2f}GB exceeds threshold "
                f"{peak_fail_gb}GB — retrain at scale will OOM. Set "
                f"ATLAS_PEAK_MEM_FAIL_GB to override the threshold."
            )
            sys.exit(2)

    if args.wandb:
        accelerator.end_training()

    accelerator.print(
        f"Done. {tokens_total / 1e9:.2f}B tokens in {(time.time() - t0) / 3600:.1f}h"
    )


if __name__ == "__main__":
    main()
