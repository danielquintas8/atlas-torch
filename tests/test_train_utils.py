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


def test_param_groups_split():
    """AdamW must decay matmul weights only. Uniform wd=0.1 was decaying norm
    gains, gate biases, the poly Taylor coefficients (toward 0, not their
    1/d! init), embedding-like tables, and the hyper-connection split_fracs —
    the parameter the seq-4096 divergence localized to (2026-09-01)."""
    from experiments.train import build_param_groups
    from titans_pytorch import MemoryAsContextTransformer
    from titans_pytorch.neural_memory import NeuralMemory

    mem_kwargs = NeuralMemory.atlas_config()
    mem_kwargs.update(dim_head = 8, heads = 4)

    model = MemoryAsContextTransformer(
        num_tokens = 64,
        dim = 32,
        depth = 2,
        segment_len = 16,
        num_persist_mem_tokens = 4,
        num_longterm_mem_tokens = 4,
        neural_memory_layers = (1,),
        neural_memory_segment_len = 4,
        use_flex_attn = False,
        neural_memory_kwargs = mem_kwargs,
    )

    wd = 0.1
    groups = build_param_groups(model = model, weight_decay = wd)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == wd
    assert groups[1]["weight_decay"] == 0.0

    decay_ids = {id(p) for p in groups[0]["params"]}
    no_decay_ids = {id(p) for p in groups[1]["params"]}

    # every trainable param exactly once
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert decay_ids.isdisjoint(no_decay_ids)
    assert len(decay_ids) + len(no_decay_ids) == len(trainable)
    assert {id(p) for _, p in trainable} == decay_ids | no_decay_ids

    embedding_like = ("token_emb", "longterm_mems", "persistent_memory", "axial_pos_emb")
    saw_poly_coefficients = False
    saw_embedding_like = False
    saw_decay_weight = False

    for name, param in trainable:
        if param.ndim < 2:
            assert id(param) in no_decay_ids, f"{name} (ndim<2) must not decay"
        if any(tag in name for tag in embedding_like):
            saw_embedding_like = True
            assert id(param) in no_decay_ids, f"{name} (embedding-like) must not decay"
        if "poly_features.coefficients" in name:
            saw_poly_coefficients = True
            assert id(param) in no_decay_ids, f"{name} (Taylor coefficients) must not decay"
        if name.endswith("to_qkv.weight") or name.endswith("to_logits.weight"):
            saw_decay_weight = True
            assert id(param) in decay_ids, f"{name} (matmul weight) must decay"

    # prove the discriminating cases actually existed in this model
    assert saw_poly_coefficients, "atlas config must expose poly coefficients"
    assert saw_embedding_like, "model must expose embedding-like tables"
    assert saw_decay_weight, "model must expose ordinary matmul weights"


def test_write_train_val_split_crossing_shards(tmp_path):
    """The train/val boundary crossing 2+ shards must lose nothing. The
    pre-fix loop sliced shard[split:] with a negative split after the
    crossing shard — synthetic case produced val=20 with 20 tokens silently
    lost while meta.json claimed 40 (2026-09-01)."""
    from experiments.data.prepare import write_train_val_split

    sizes = (100, 100, 30)
    starts = np.cumsum((0,) + sizes[:-1])
    shard_paths = []
    for i, (start, size) in enumerate(zip(starts, sizes)):
        path = str(tmp_path / f"shard_{i:04d}.bin")
        np.arange(start, start + size, dtype = np.uint16).tofile(path)
        shard_paths.append(path)

    train_path = str(tmp_path / "train.bin")
    val_path = str(tmp_path / "val.bin")
    total = sum(sizes)
    train_tokens = 190  # boundary inside shard 1; shard 2 is entirely val

    train_written, val_written = write_train_val_split(
        shard_paths = shard_paths,
        train_tokens = train_tokens,
        train_path = train_path,
        val_path = val_path,
    )

    train = np.fromfile(train_path, dtype = np.uint16)
    val = np.fromfile(val_path, dtype = np.uint16)

    assert train_written == train_tokens and len(train) == train_tokens
    assert val_written == total - train_tokens and len(val) == total - train_tokens
    assert np.array_equal(train, np.arange(0, train_tokens, dtype = np.uint16))
    assert np.array_equal(val, np.arange(train_tokens, total, dtype = np.uint16))
    # shards consumed
    assert not any(os.path.exists(sp) for sp in shard_paths)
