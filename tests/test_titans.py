import pytest

import torch
from titans_pytorch import NeuralMemory

def test_titans():
    mem = NeuralMemory(
        dim = 384,
        chunk_size = 64,
    )

    seq = torch.randn(2, 1024, 384)
    retrieved = mem(seq)

    assert seq.shape == retrieved.shape

def test_titans_attn_memory():
    from titans_pytorch import MemoryAttention

    mem = NeuralMemory(
        dim = 384,
        model = MemoryAttention(
            dim = 384
        ),
        chunk_size = 64,
    )

    seq = torch.randn(2, 1024, 384)
    retrieved = mem(seq)

    assert seq.shape == retrieved.shape
