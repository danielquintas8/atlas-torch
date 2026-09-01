import copy
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.train import MemmapTokenDataset, cosine_with_warmup


def _write_bin(path, n_chunks, chunk_len):
    # chunk i is filled with the value i so a yielded chunk identifies itself
    data = np.concatenate(
        [np.full(chunk_len, i, dtype = np.uint16) for i in range(n_chunks)]
    )
    data.tofile(path)


def test_memmap_dataset_full_coverage_single_process(tmp_path):
    """The dataset must NOT self-shard by rank — Accelerate's dataloader
    dispatch owns multi-GPU distribution (a single reader slices batches
    across ranks). Rank striding here combined with that dispatch meant only
    the main process's 1/world_size shard was ever read, cutting effective
    epochs to a quarter on 4 GPUs (double-sharding regression, 2026-09-01).
    Distributed-style env vars are set on purpose: the pre-fix code would
    yield only a quarter of the chunks under them."""
    n_chunks, seq_len = 32, 16
    bin_path = str(tmp_path / "data.bin")
    _write_bin(path = bin_path, n_chunks = n_chunks, chunk_len = seq_len + 1)

    os.environ["RANK"] = "1"
    os.environ["WORLD_SIZE"] = "4"
    try:
        ds = MemmapTokenDataset(bin_path = bin_path, seq_len = seq_len, shuffle = False)
        ids = [int(chunk[0]) for chunk in ds]
        assert ids == list(range(n_chunks)), (
            f'expected all {n_chunks} chunks in order, got {len(ids)} — '
            f'the dataset must not shard itself by RANK/WORLD_SIZE'
        )

        random.seed(123)
        ds_shuffled = MemmapTokenDataset(bin_path = bin_path, seq_len = seq_len, shuffle = True)
        shuffled_ids = [int(chunk[0]) for chunk in ds_shuffled]
        assert sorted(shuffled_ids) == list(range(n_chunks)), 'shuffle must permute, not drop'
        assert shuffled_ids != list(range(n_chunks)), 'shuffle=True must actually permute'

        # a second pass reshuffles (per-epoch permutation semantics)
        second_pass = [int(chunk[0]) for chunk in ds_shuffled]
        assert sorted(second_pass) == list(range(n_chunks))
        assert second_pass != shuffled_ids, 'each __iter__ must draw a fresh permutation'
    finally:
        os.environ.pop("RANK", None)
        os.environ.pop("WORLD_SIZE", None)


def test_cosine_schedule_resume_continuity():
    """Resume must continue the ORIGINAL schedule. load_state_dict restores the
    scheduler's step counter, so warmup must stay as configured — no re-warmup
    happens anyway. The old resume path rebuilt the lambda with warmup=0,
    which does not prevent re-warmup (the restored counter already does); it
    swaps in a different cosine, measured -1.6% to -12.2% LR vs the intended
    schedule over a run (2026-09-01)."""
    warmup, total, resume_at = 20, 300, 57

    def make(warmup_steps):
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr = 1.0)
        sched = cosine_with_warmup(
            optimizer = opt, warmup_steps = warmup_steps, total_steps = total
        )
        return sched

    # original run: step to the checkpoint, snapshot state, keep stepping
    sched_orig = make(warmup_steps = warmup)
    for _ in range(resume_at):
        sched_orig.step()
    state = copy.deepcopy(sched_orig.state_dict())

    continued = []
    for _ in range(10):
        sched_orig.step()
        continued.append(sched_orig.get_last_lr()[0])

    # resumed run with the SAME warmup: must reproduce the original exactly
    sched_resumed = make(warmup_steps = warmup)
    sched_resumed.load_state_dict(copy.deepcopy(state))
    resumed = []
    for _ in range(10):
        sched_resumed.step()
        resumed.append(sched_resumed.get_last_lr()[0])

    assert resumed == continued, (
        'a resumed scheduler with identical warmup/total must continue the '
        'original LR sequence exactly'
    )

    # the old warmup=0 rebuild produces a DIFFERENT lr at the same step —
    # documents why warmup must not be zeroed on resume
    sched_zeroed = make(warmup_steps = 0)
    sched_zeroed.load_state_dict(copy.deepcopy(state))
    sched_zeroed.step()
    assert abs(sched_zeroed.get_last_lr()[0] - continued[0]) > 1e-6, (
        'warmup=0 on resume was expected to deform the schedule — if these '
        'now match, this guard needs re-derivation'
    )
