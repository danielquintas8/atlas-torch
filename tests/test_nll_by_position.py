"""The per-position NLL diagnostic's chunked path must score exactly what
the whole-sequence path scores: same per-token NLL vector (chunk-to-label
alignment), same bucket means and overall mean end-to-end on a val.bin."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.diagnostics.nll_by_position import _token_nll, iter_val_chunks, nll_by_position
from titans_pytorch import MemoryAsContextTransformer
from titans_pytorch.neural_memory import NeuralMemory

SEQ_LEN = 256
CHUNK = 64   # one memory segment of the tiny MAC, in interleaved positions


def _tiny_mac():
    mem_kwargs = NeuralMemory.atlas_config()
    mem_kwargs.update(dim_head = 8, heads = 4, use_sequential_scan = True)
    torch.manual_seed(0)
    return MemoryAsContextTransformer(
        num_tokens = 256, dim = 32, depth = 3, segment_len = 16,
        num_persist_mem_tokens = 4, num_longterm_mem_tokens = 4,
        neural_memory_layers = (1, 3), neural_memory_segment_len = 8,
        neural_memory_batch_size = CHUNK, use_flex_attn = False,
        sliding_window_attn = True, neural_memory_kwargs = mem_kwargs,
        use_axial_pos_emb = False,
    ).double().eval()


def test_token_nll_chunked_matches_whole():
    model = _tiny_mac()
    torch.manual_seed(1)
    x = torch.randint(0, 256, (1, SEQ_LEN + 1))
    inputs, labels = x[:, :-1], x[:, 1:]

    whole = _token_nll(model = model, inputs = inputs, labels = labels, disable_flex_attn = True, chunk_len = None)
    chunked = _token_nll(model = model, inputs = inputs, labels = labels, disable_flex_attn = True, chunk_len = CHUNK)

    assert whole.shape == chunked.shape == (SEQ_LEN,)
    assert torch.allclose(whole, chunked, atol = 1e-10, rtol = 0), (whole - chunked).abs().max()
    # liveness: the NLL is not a constant, and a one-position label shift is visible
    assert whole.std() > 1e-3
    shifted = _token_nll(
        model = model, inputs = inputs, labels = torch.roll(labels, 1, dims = 1), disable_flex_attn = True, chunk_len = CHUNK,
    )
    assert not torch.allclose(whole, shifted, atol = 1e-6, rtol = 0)


def test_nll_by_position_chunked_matches_whole(tmp_path):
    val_bin = tmp_path / "val.bin"
    rng = np.random.default_rng(0)
    rng.integers(0, 256, size = 3 * (SEQ_LEN + 1) + 17, dtype = np.uint16).tofile(val_bin)
    assert len(list(iter_val_chunks(val_bin = str(val_bin), seq_len = SEQ_LEN, max_chunks = 10))) == 3

    model = _tiny_mac()
    common = dict(model = model, val_bin = str(val_bin), seq_len = SEQ_LEN, max_chunks = 3,
                  bucket = 64, device = "cpu", disable_flex_attn = True)
    whole = nll_by_position(**common)
    chunked = nll_by_position(**common, chunk_len = CHUNK)

    assert whole["chunks"] == chunked["chunks"] == 3
    assert whole["overall"] == pytest.approx(chunked["overall"], abs = 1e-10)
    assert [b["nll"] for b in whole["buckets"]] == pytest.approx([b["nll"] for b in chunked["buckets"]], abs = 1e-10)
    assert [(b["start"], b["end"]) for b in chunked["buckets"]] == [(0, 64), (64, 128), (128, 192), (192, 256)]
