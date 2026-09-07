"""Profile one training step of the shipped model (forward + backward + optimizer)
at the launched geometry, to attribute wall time and peak memory to operators
instead of estimating them.

    python benchmarks/profile_train_step.py --variant atlas-mac --ablation no-omega \
        --seq-len 1024 --steps 3 --memory-kwarg use_sequential_scan=false --out runs/profile-no-omega

Writes <out>/summary.json (per-phase seconds, peak GB, top operators by CUDA
and CPU time), <out>/trace.json (Chrome trace of the last step, if
--trace) and, on CUDA, <out>/memory_snapshot.pickle (torch.cuda memory
history, also dumped when the step OOMs). CPU-only runs give timings only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.configs import get_config  # noqa: E402
from experiments.train import apply_memory_kwargs, apply_vanilla, parse_memory_kwargs  # noqa: E402
from titans_pytorch import MemoryAsContextTransformer  # noqa: E402


def build(args):
    config = get_config(model_size=args.model, variant=args.variant, ablation=args.ablation)
    config = apply_vanilla(config=config, vanilla=args.vanilla)
    config = apply_memory_kwargs(config=config, overrides=parse_memory_kwargs(items=args.memory_kwarg))
    for item in args.model_kwarg:
        key, raw = item.split("=", 1)
        config["model"][key] = parse_memory_kwargs(items=[item])[key]
    return config


def run_step(model, optimizer, batch, autocast_dtype, timings):
    device_type = batch.device.type
    sync = torch.cuda.synchronize if device_type == "cuda" else (lambda: None)
    sync(); t0 = time.perf_counter()
    with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        loss = model(batch, return_loss=True)
    sync(); t1 = time.perf_counter()
    loss.backward()
    sync(); t2 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sync(); t3 = time.perf_counter()
    timings.append(dict(forward=t1 - t0, backward=t2 - t1, optimizer=t3 - t2, loss=loss.detach().item()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="170m")
    p.add_argument("--variant", default="atlas-mac")
    p.add_argument("--ablation", default=None)
    p.add_argument("--vanilla", action="store_true")
    p.add_argument("--memory-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--model-kwarg", action="append", default=[], metavar="KEY=VALUE", help="override a MemoryAsContextTransformer kwarg, e.g. neural_memory_batch_size=512")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=3, help="timed steps after one warm-up step")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--trace", action="store_true", help="also write a Chrome trace of the last step")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    config = build(args)
    torch.manual_seed(0)
    model = MemoryAsContextTransformer(**config["model"]).to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = torch.randint(0, config["model"]["num_tokens"], (args.batch, args.seq_len + 1), device=args.device)
    autocast_dtype = None if (args.no_bf16 or args.device != "cuda") else torch.bfloat16
    cuda = args.device == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history(max_entries=200_000)

    summary = dict(config=dict(variant=args.variant, ablation=args.ablation, vanilla=args.vanilla, memory_kwarg=args.memory_kwarg,
                               model_kwarg=args.model_kwarg, seq_len=args.seq_len, batch=args.batch, bf16=autocast_dtype is not None),
                   params_m=sum(q.numel() for q in model.parameters()) / 1e6)
    timings = []
    try:
        run_step(model, optimizer, batch, autocast_dtype, timings)  # warm-up (kernels, allocator)
        summary["warmup"] = timings.pop()
        for _ in range(args.steps):
            run_step(model, optimizer, batch, autocast_dtype, timings)
        if args.trace or cuda:
            from torch.profiler import ProfilerActivity, profile
            activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
            with profile(activities=activities, record_shapes=False, profile_memory=cuda) as prof:
                run_step(model, optimizer, batch, autocast_dtype, timings)
            key = "device_time_total" if cuda else "cpu_time_total"
            rows = prof.key_averages()
            try:
                table = rows.table(sort_by=key, row_limit=30)
            except Exception:  # noqa: BLE001 — older torch names the column differently
                key = "cuda_time_total"
                table = rows.table(sort_by=key, row_limit=30)
            summary["top_ops"] = table
            if args.trace:
                prof.export_chrome_trace(os.path.join(args.out, "trace.json"))
    except torch.OutOfMemoryError as err:
        summary["oom"] = str(err)[:400]
    finally:
        if cuda:
            summary["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            summary["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
            try:
                torch.cuda.memory._dump_snapshot(os.path.join(args.out, "memory_snapshot.pickle"))
            except Exception as err:  # noqa: BLE001
                summary["snapshot_error"] = str(err)
    summary["steps"] = timings
    if timings:
        steady = timings[: args.steps]
        summary["mean_step_s"] = sum(t["forward"] + t["backward"] + t["optimizer"] for t in steady) / len(steady)
        summary["tokens_per_s"] = args.batch * args.seq_len / summary["mean_step_s"]
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "top_ops"}, indent=2, default=str))
    if "top_ops" in summary:
        print(summary["top_ops"])


if __name__ == "__main__":
    main()
