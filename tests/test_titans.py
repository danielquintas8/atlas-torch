import torch
from torch import nn

import pytest
from titans_pytorch import NeuralMemory
from titans_pytorch.mac_transformer import flex_attention, SegmentedAttention

def exists(v):
    return v is not None

@pytest.mark.parametrize('seq_len', (32, 1024, 77))
@pytest.mark.parametrize('silu', (False, True))
@pytest.mark.parametrize('learned_mem_model_weights', (False, True))
@pytest.mark.parametrize('max_grad_norm', (None, 2.))
@pytest.mark.parametrize('per_parameter_lr_modulation', (False, True))
def test_titans(
    seq_len,
    silu,
    learned_mem_model_weights,
    max_grad_norm,
    per_parameter_lr_modulation
):
    mem = NeuralMemory(
        dim = 384,
        chunk_size = 64,
        activation = nn.SiLU() if silu else None,
        max_grad_norm = max_grad_norm,
        per_parameter_lr_modulation = per_parameter_lr_modulation,
        learned_mem_model_weights = learned_mem_model_weights
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
@pytest.mark.parametrize('neural_mem_gate_attn_output', (False, True))
def test_mac(
    num_persist_mem_tokens,
    num_longterm_mem_tokens,
    neural_mem_gate_attn_output
):
    from titans_pytorch.mac_transformer import MemoryAsContextTransformer

    transformer = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 256,
        depth = 2,
        num_persist_mem_tokens = num_persist_mem_tokens,
        num_longterm_mem_tokens = num_longterm_mem_tokens,
        segment_len = 128,
        neural_mem_gate_attn_output = neural_mem_gate_attn_output
    )

    x = torch.randint(0, 256, (1, 1023))

    logits = transformer(x)
    assert logits.shape == (1, 1023, 256)

@pytest.mark.parametrize('sliding', (True, False))
def test_flex(
    sliding
):
    if not (torch.cuda.is_available() and exists(flex_attention)):
        pytest.skip()

    attn = SegmentedAttention(
        dim = 512,
        segment_len = 32,
        num_persist_mem_tokens = 1,
        num_longterm_mem_tokens = 1,
        use_flex_attn = True,
        sliding = sliding
    ).cuda()

    seq = torch.randn(1, 1019, 512).cuda()

    out_flex, _ = attn(seq)
    out_non_flex, _ = attn(seq, disable_flex_attn = True)

    assert torch.allclose(out_flex, out_non_flex, atol = 1e-5)
