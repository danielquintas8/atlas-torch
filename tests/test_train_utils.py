import argparse
import copy
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.train import MemmapTokenDataset, compute_schedule, cosine_with_warmup


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

        ds_shuffled = MemmapTokenDataset(
            bin_path = bin_path, seq_len = seq_len, shuffle = True, seed = 123
        )
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


def test_memmap_dataset_resume_permutation(tmp_path):
    """The permutation must be a pure function of seed + epoch, immune to the
    process-global RNGs — accelerator.load_state restores those to their
    checkpoint-time (post-shuffle) state, so any global-random shuffle after
    resume draws the WRONG permutation (probe-proven 2026-09-01: 9/9
    post-resume batches mismatched). skip_chunks must seek within the first
    pass only; epoch_base must replay a later epoch's permutation exactly."""
    n_chunks, seq_len = 32, 16
    bin_path = str(tmp_path / "data.bin")
    _write_bin(path = bin_path, n_chunks = n_chunks, chunk_len = seq_len + 1)

    def ids(dataset):
        return [int(chunk[0]) for chunk in dataset]

    reference = MemmapTokenDataset(
        bin_path = bin_path, seq_len = seq_len, shuffle = True, seed = 11
    )
    epoch0 = ids(reference)
    epoch1 = ids(reference)
    assert sorted(epoch0) == sorted(epoch1) == list(range(n_chunks))
    assert epoch0 != epoch1

    # global RNG state must be irrelevant: scramble every global RNG and the
    # same constructor must reproduce the same epochs
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    replay = MemmapTokenDataset(
        bin_path = bin_path, seq_len = seq_len, shuffle = True, seed = 11
    )
    assert ids(replay) == epoch0, 'permutation must not depend on global RNG state'
    assert ids(replay) == epoch1

    # skip_chunks: first pass seeks, second pass is the full next epoch
    resumed_mid = MemmapTokenDataset(
        bin_path = bin_path, seq_len = seq_len, shuffle = True, seed = 11,
        skip_chunks = 5,
    )
    assert ids(resumed_mid) == epoch0[5:], 'skip_chunks must seek within the first pass'
    assert ids(resumed_mid) == epoch1, 'the pass after the seek must be the full next epoch'

    # epoch_base: replay a later epoch from a fresh dataset (past-epoch resume)
    resumed_late = MemmapTokenDataset(
        bin_path = bin_path, seq_len = seq_len, shuffle = True, seed = 11,
        epoch_base = 1, skip_chunks = 3,
    )
    assert ids(resumed_late) == epoch1[3:]

    # limit_chunks caps the dataset (validation's fixed slice)
    limited = MemmapTokenDataset(
        bin_path = bin_path, seq_len = seq_len, shuffle = False, limit_chunks = 8
    )
    assert ids(limited) == list(range(8))


def test_compute_schedule_resume_keeps_warmup():
    """compute_schedule must NOT consult args.resume: warmup comes from config
    (or the CLI override) regardless. The old `elif args.resume: warmup = 0`
    rebuilt a different cosine against the restored step counter, measured
    -1.6% to -12.2% LR vs the intended schedule (2026-09-01); re-adding it
    must fail this test."""
    train_cfg = dict(
        seq_len = 1024, batch_tokens = 500_000, total_tokens = 15_000_000_000,
        warmup_steps = 2000,
    )

    def make_args(resume, warmup_steps = None):
        return argparse.Namespace(
            per_device_batch_size = 4,
            grad_accum = None,
            max_steps = None,
            warmup_steps = warmup_steps,
            resume = resume,
        )

    fresh = compute_schedule(
        train_cfg = train_cfg, args = make_args(resume = None), num_gpus = 4
    )
    resumed = compute_schedule(
        train_cfg = train_cfg,
        args = make_args(resume = "runs/whatever/step-4000"),
        num_gpus = 4,
    )
    assert fresh == resumed, 'resume must not change the schedule shape'
    assert resumed[4] == train_cfg["warmup_steps"], 'warmup must come from config on resume'

    # CLI override still wins
    overridden = compute_schedule(
        train_cfg = train_cfg,
        args = make_args(resume = "runs/whatever/step-4000", warmup_steps = 77),
        num_gpus = 4,
    )
    assert overridden[4] == 77


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
    1/d! init), embedding-like tables, and the hyper-connection routing
    params — static_alpha is initialized one-hot + eye (the residual-stream
    identity routing), so decay pulls it toward the zero matrix
    (2026-09-01)."""
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
    hyper_conn_routing = ("static_alpha", "dynamic_alpha_fn", "stream_embed")
    saw_poly_coefficients = False
    saw_embedding_like = False
    saw_decay_weight = False
    saw_routing = {tag: False for tag in hyper_conn_routing}
    saw_norm_gain_2d = False

    for name, param in trainable:
        if param.ndim < 2:
            assert id(param) in no_decay_ids, f"{name} (ndim<2) must not decay"
        if any(tag in name for tag in embedding_like):
            saw_embedding_like = True
            assert id(param) in no_decay_ids, f"{name} (embedding-like) must not decay"
        for tag in hyper_conn_routing:
            if tag in name:
                saw_routing[tag] = True
                assert id(param) in no_decay_ids, (
                    f"{name} (hyper-connection routing — static_alpha is the "
                    f"identity residual routing at init) must not decay"
                )
        if name.endswith(".gamma") and param.ndim >= 2:
            saw_norm_gain_2d = True
            assert id(param) in no_decay_ids, (
                f"{name} (norm gain, (gamma+1)-parameterized) must not decay"
            )
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
    for tag, seen in saw_routing.items():
        assert seen, f"model must expose hyper-connection {tag} (default 4 residual streams)"
    assert saw_norm_gain_2d, "model must expose an ndim>=2 norm gain (qk_rmsnorm gammas)"


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


def test_resume_data_position_roundtrip(tmp_path):
    """The checkpoint records the data position (epochs_done /
    yields_this_epoch) COUNTED by the training loop; resume_data_position
    maps it back to (epoch_base, skip_chunks) whose stream continues the
    uninterrupted one exactly. Recording — never inverting from step
    counts — is the point: Accelerate's end-of-dataloader forced sync makes
    each epoch's final optimization step consume fewer than grad_accum
    yields, so `start_step * grad_accum` overcounted (delta review
    2026-09-02: 2 yields off at toy scale, up to 121 at production
    cadence). Covers mid-epoch, exactly-at-epoch-end (the clamp corner),
    and past-boundary positions, plus the missing-fields fallback."""
    from experiments.train import resume_data_position

    n_chunks, seq_len, chunks_per_yield = 64, 16, 4
    bin_path = str(tmp_path / "data.bin")
    _write_bin(path = bin_path, n_chunks = n_chunks, chunk_len = seq_len + 1)
    seed = 7

    def consume(n_yields):
        ds = MemmapTokenDataset(
            bin_path = bin_path, seq_len = seq_len, shuffle = True, seed = seed,
        )
        it = iter(ds)
        epochs_done, yields_this_epoch = 0, 0
        yields = []
        while len(yields) < n_yields:
            batch = []
            for _ in range(chunks_per_yield):
                try:
                    batch.append(int(next(it)[0]))
                except StopIteration:
                    epochs_done += 1
                    yields_this_epoch = 0
                    it = iter(ds)
                    batch.append(int(next(it)[0]))
            yields.append(list(batch))
            yields_this_epoch += 1
        return yields, epochs_done, yields_this_epoch

    total_positions = [
        9,        # mid-epoch 0
        16,       # exactly at epoch 0's end (clamp corner: 16*4 == n_chunks)
        22,       # past the boundary, mid-epoch 1
    ]
    horizon = 40
    reference, _, _ = consume(horizon)

    for stop_at in total_positions:
        _, epochs_done, yields_this_epoch = consume(stop_at)
        meta = dict(epochs_done = epochs_done, yields_this_epoch = yields_this_epoch)
        epoch_base, skip_chunks, exact = resume_data_position(
            resume_meta = meta, n_chunks = n_chunks, chunks_per_yield = chunks_per_yield,
        )
        assert exact

        # resumed stream must equal the uninterrupted continuation exactly
        ds = MemmapTokenDataset(
            bin_path = bin_path, seq_len = seq_len, shuffle = True,
            seed = seed, epoch_base = epoch_base, skip_chunks = skip_chunks,
        )
        it = iter(ds)
        resumed = []
        while len(resumed) < horizon - stop_at:
            batch = []
            for _ in range(chunks_per_yield):
                try:
                    batch.append(int(next(it)[0]))
                except StopIteration:
                    it = iter(ds)
                    batch.append(int(next(it)[0]))
            resumed.append(batch)

        assert resumed == reference[stop_at:horizon], (
            f"stop_at={stop_at}: resumed stream diverged "
            f"(epoch_base={epoch_base}, skip_chunks={skip_chunks})"
        )

    # liveness: a deliberately wrong position must NOT match
    _, epochs_done, yields_this_epoch = consume(9)
    wrong_meta = dict(epochs_done = epochs_done, yields_this_epoch = yields_this_epoch + 1)
    eb, sc, _ = resume_data_position(
        resume_meta = wrong_meta, n_chunks = n_chunks, chunks_per_yield = chunks_per_yield,
    )
    ds = MemmapTokenDataset(
        bin_path = bin_path, seq_len = seq_len, shuffle = True,
        seed = seed, epoch_base = eb, skip_chunks = sc,
    )
    first_wrong = [int(next(iter(ds))[0])]
    assert first_wrong != [reference[9][0]], "comparator cannot detect a wrong seek"

    # at-epoch-end positions roll over at construction — an empty first pass
    # must never reach the dispatcher (its main process would broadcast a
    # None batch and crash; probe-proven 2026-09-02)
    assert resume_data_position(
        resume_meta = dict(epochs_done = 0, yields_this_epoch = 16),
        n_chunks = n_chunks,
        chunks_per_yield = chunks_per_yield,
    ) == (1, 0, True)

    # missing recorded fields -> inexact epoch-0 restart
    assert resume_data_position(
        resume_meta = dict(step = 5), n_chunks = n_chunks, chunks_per_yield = chunks_per_yield,
    ) == (0, 0, False)
