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
    """

    def __init__(self, bin_path, seq_len, shuffle=True):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.chunk_len = seq_len + 1
        self.n_chunks = len(self.data) // self.chunk_len
        self.shuffle = shuffle

    def __iter__(self):
        # fresh permutation per epoch (per __iter__ call), driven by the
        # process-global `random` state seeded once via set_seed
        indices = list(range(self.n_chunks))
        if self.shuffle:
            import random
            random.shuffle(indices)

        for idx in indices:
            start = idx * self.chunk_len
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


def load_data(data_dir, seq_len, split="train"):
    """Load dataset — binary format (BSC) or streaming HF.

    Returns None if the requested split doesn't exist.
    """
    bin_path = os.path.join(data_dir, f"{split}.bin")
    if os.path.exists(bin_path):
        return MemmapTokenDataset(bin_path, seq_len, shuffle=(split == "train"))

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


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_param_groups(model, weight_decay):
    """Split parameters into decay / no-decay groups for AdamW.

    Decay: matmul-style weights (ndim >= 2), excluding embedding-like tables.
    No decay (wd=0.0): everything else — biases, norm gains, the
    hyper-connection split_fracs (the parameter the seq-4096 divergence
    localized to), the polynomial Taylor coefficients (uniform decay pulled
    them toward 0, not their 1/d! init), gate scalars — plus the
    embedding-like tables (token_emb, longterm_mems, persistent_memory,
    axial_pos_emb): weight decay on embeddings shrinks their norms and
    inflates early-layer gradients via the LayerNorm Jacobian (OLMo-2
    mechanism). Field standard is decay-matrices-only; the previous uniform
    wd=0.1 decayed all of the above at lr*wd ≈ 3e-4/step (found 2026-09-01).
    """
    embedding_like = ("token_emb", "longterm_mems", "persistent_memory", "axial_pos_emb")

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_embedding_like = any(tag in name for tag in embedding_like)
        if param.ndim >= 2 and not is_embedding_like:
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

def save_checkpoint(accelerator, step, output_dir):
    ckpt_dir = os.path.join(output_dir, f"step-{step}")
    accelerator.save_state(ckpt_dir)
    if accelerator.is_main_process:
        torch.save({"step": step}, os.path.join(ckpt_dir, "meta.pt"))
    accelerator.print(f"Checkpoint saved → {ckpt_dir}")


def load_checkpoint(accelerator, resume_dir):
    accelerator.load_state(resume_dir)
    meta = torch.load(
        os.path.join(resume_dir, "meta.pt"), map_location="cpu", weights_only=True
    )
    return meta["step"]


def prune_checkpoints(accelerator, output_dir, keep):
    """Delete the oldest step-* checkpoint dirs beyond the newest `keep`.

    Opt-in via --keep-checkpoints: the eval workflow scores multiple
    historical checkpoints, so nothing is deleted by default. At
    --save-every 100 a full 15B run writes ~300 checkpoints x ~2-3 GB
    (600-900 GB of GPFS). Only directories matching step-<digits> inside
    output_dir are ever touched.
    """
    if keep is None or keep < 1 or not accelerator.is_main_process:
        return

    pattern = re.compile(r"^step-(\d+)$")
    step_dirs = []
    for entry in os.listdir(output_dir):
        match = pattern.match(entry)
        full = os.path.join(output_dir, entry)
        if match and os.path.isdir(full):
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
    return p.parse_args()


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

    # Gradient accumulation: match 0.5M token batch target
    num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
    tokens_per_micro = num_gpus * args.per_device_batch_size * seq_len
    grad_accum = args.grad_accum or max(1, train_cfg["batch_tokens"] // tokens_per_micro)
    batch_tokens = tokens_per_micro * grad_accum

    schedule_steps = int(train_cfg["total_tokens"] // batch_tokens)
    max_steps = args.max_steps or schedule_steps
    # warmup stays as configured even on resume: the scheduler's step counter
    # is restored from the checkpoint (register_for_checkpointing below), so
    # the original schedule continues exactly and no re-warmup happens. The
    # old warmup=0-on-resume rebuild did not prevent re-warmup (the restored
    # counter already does) — it swapped in a different cosine, measured
    # -1.6% to -12.2% LR vs the intended schedule over the run (2026-09-01).
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = train_cfg["warmup_steps"]

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

    # Data
    data_dir = args.data_dir or train_cfg["dataset"]
    train_dataset = load_data(data_dir, seq_len, split="train")
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
    # fresh loader every time, accumulating dataloader state in the
    # accelerator and into every checkpoint.) Prepared before load_state so
    # the set of registered dataloaders is identical at save and load time.
    val_dataset = load_data(data_dir, seq_len, split="val")
    if val_dataset is not None:
        val_loader = accelerator.prepare(
            DataLoader(val_dataset, batch_size=args.per_device_batch_size)
        )
    else:
        val_loader = None
        accelerator.print("No val.bin found — validation disabled")

    # Resume
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(accelerator, args.resume)
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

    # Skip past already-seen data on resume — FIRST pass only. The wrapper
    # returned by skip_first_batches skips on every __iter__, so rebinding
    # train_loader to it (as this code used to) silently dropped — and
    # read-and-discarded — the first start_step * grad_accum batches of every
    # subsequent epoch as well (probe-proven 2026-09-01). Epoch restarts below
    # re-iterate the ORIGINAL loader.
    if start_step > 0:
        batches_to_skip = start_step * grad_accum
        accelerator.print(f"Skipping {batches_to_skip} batches for resume...")
        from accelerate.data_loader import skip_first_batches

        train_iter = iter(skip_first_batches(train_loader, num_batches=batches_to_skip))
    else:
        train_iter = iter(train_loader)

    global_step = start_step
    tokens_this_run = 0
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

        # Validate — re-iterates the once-prepared val_loader (val.bin is
        # unshuffled, so every pass scores the same first 50 batches)
        if global_step % args.validate_every == 0 and val_loader is not None:
            model.eval()

            val_losses = []
            with torch.no_grad():
                for i, val_batch in enumerate(val_loader):
                    if i >= 50:
                        break
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
            save_checkpoint(accelerator, global_step, output_dir)
            prune_checkpoints(
                accelerator=accelerator,
                output_dir=output_dir,
                keep=args.keep_checkpoints,
            )

    # Final save
    save_checkpoint(accelerator, global_step, output_dir)
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
