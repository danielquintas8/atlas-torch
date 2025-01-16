import torch
import pytest
from titans_pytorch import NeuralMemory

@pytest.mark.parametrize('seq_len', (32, 1024, 77))
@pytest.mark.parametrize('max_grad_norm', (None, 2.))
def test_titans(
    seq_len,
    max_grad_norm
):
    mem = NeuralMemory(
        dim = 384,
        chunk_size = 64,
        max_grad_norm = max_grad_norm
    )

    seq = torch.randn(2, seq_len, 384)
    retrieved = mem(seq)

    assert seq.shape == retrieved.shape

def test_titans_attn_memory():
    from titans_pytorch.titans import MemoryAttention

    mem = NeuralMemory(
        dim = 384,
        chunk_size = 64,
        model = MemoryAttention(
            dim = 384
        )
    )

    seq = torch.randn(2, 1024, 384)
    retrieved = mem(seq)

    assert seq.shape == retrieved.shape

@pytest.mark.parametrize('num_persist_mem_tokens', (0, 16))
@pytest.mark.parametrize('num_longterm_mem_tokens', (0, 16))
def test_mac(
    num_persist_mem_tokens,
    num_longterm_mem_tokens
):
    from titans_pytorch.mac_transformer import MemoryAsContextTransformer

    transformer = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 256,
        depth = 2,
        num_persist_mem_tokens = num_persist_mem_tokens,
        num_longterm_mem_tokens = num_longterm_mem_tokens,
        segment_len = 128,
    )

    x = torch.randint(0, 256, (1, 1023))

    logits = transformer(x)
    assert logits.shape == (1, 1023, 256)
