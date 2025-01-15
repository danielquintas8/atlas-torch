import torch
import pytest

@pytest.mark.parametrize('seq_len', (32, 1024, 77))
@pytest.mark.parametrize('max_grad_norm', (None, 2.))
def test_titans(
    seq_len,
    max_grad_norm
):

    from titans_pytorch import NeuralMemory

    mem = NeuralMemory(
        dim = 384,
        chunk_size = 64,
        max_grad_norm = max_grad_norm
    )

    seq = torch.randn(2, seq_len, 384)
    retrieved = mem(seq)

    assert seq.shape == retrieved.shape

def test_titans_attn_memory():
    from titans_pytorch.titans_attn_memory import NeuralMemory

    mem = NeuralMemory(
        dim = 384,
        chunk_size = 64,
    )

    seq = torch.randn(2, 1024, 384)
    retrieved = mem(seq)

    assert seq.shape == retrieved.shape

def test_mac():
    from titans_pytorch.mac_transformer import MemoryAsContextTransformer

    transformer = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 256,
        depth = 2,
        num_persist_mem_tokens = 16,
        segment_len = 128,
    )

    x = torch.randint(0, 256, (1, 1023))

    logits = transformer(x)
    assert logits.shape == (1, 1023, 256)
