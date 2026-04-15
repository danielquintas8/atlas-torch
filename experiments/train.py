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
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

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
    Shards by rank for multi-GPU — each GPU sees a disjoint subset.
    """

    def __init__(self, bin_path, seq_len, shuffle=True):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.chunk_len = seq_len + 1
        self.n_chunks = len(self.data) // self.chunk_len
        self.shuffle = shuffle

    def __iter__(self):
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        indices = list(range(self.n_chunks))
        if self.shuffle:
            import random
            random.shuffle(indices)

        # shard by rank so each GPU gets unique data
        indices = indices[rank::world_size]

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
    p.add_argument("--warmup-steps", type=int, default=None, help="Override warmup steps (0 to skip warmup)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
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
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    elif args.resume:
        warmup_steps = 0
    else:
        warmup_steps = train_cfg["warmup_steps"]

    # Accelerator
    # find_unused_parameters=True needed because vmap(vmap(grad)) in the omega
    # rule uses a functional interface that bypasses DDP's autograd tracking
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
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
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["peak_lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = cosine_with_warmup(optimizer, warmup_steps, schedule_steps)

    # Prepare
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

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
    train_iter = iter(train_loader)

    # Skip past already-seen data on resume
    if start_step > 0:
        batches_to_skip = start_step * grad_accum
        accelerator.print(f"Skipping {batches_to_skip} batches for resume...")
        from accelerate.data_loader import skip_first_batches

        train_loader = skip_first_batches(train_loader, batches_to_skip)
        train_iter = iter(train_loader)

    global_step = start_step
    tokens_this_run = 0
    running_loss = 0.0
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

        running_loss += loss.detach().item()
        loss_count += 1

        if not accelerator.sync_gradients:
            continue

        # Full optimization step completed
        global_step += 1
        tokens_this_run += batch_tokens
        tokens_total = start_step * batch_tokens + tokens_this_run

        # Log
        if global_step % args.log_every == 0:
            avg_loss = running_loss / loss_count
            running_loss = 0.0
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

        # Validate
        if global_step % args.validate_every == 0:
            val_dataset = load_data(data_dir, seq_len, split="val")
            if val_dataset is None:
                if global_step == args.validate_every:
                    accelerator.print("Skipping validation (no val.bin)")
            else:
                model.eval()
                val_loader = DataLoader(
                    val_dataset, batch_size=args.per_device_batch_size
                )
                val_loader = accelerator.prepare(val_loader)

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

    # Final save
    save_checkpoint(accelerator, global_step, output_dir)

    if args.wandb:
        accelerator.end_training()

    accelerator.print(
        f"Done. {tokens_total / 1e9:.2f}B tokens in {(time.time() - t0) / 3600:.1f}h"
    )


if __name__ == "__main__":
    main()
