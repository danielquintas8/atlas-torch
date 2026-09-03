from contextlib import contextmanager

import torch
from torch import nn

import pytest
from titans_pytorch import NeuralMemory
from titans_pytorch.mac_transformer import flex_attention, SegmentedAttention, MemoryAsContextTransformer

# functions

def exists(v):
    return v is not None

def diff(x, y):
    return (x - y).abs().amax()

@contextmanager
def torch_default_dtype(dtype):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)

# main test

@pytest.mark.parametrize('seq_len', (32, 512, 77))
@pytest.mark.parametrize('silu', (False, True))
@pytest.mark.parametrize('chunk_size, attn_pool_chunks', ((64, True), (64, False), (1, False)))
@pytest.mark.parametrize('momentum', (False, True))
@pytest.mark.parametrize('qk_rmsnorm', (False, True))
@pytest.mark.parametrize('heads', (1, 4))
@pytest.mark.parametrize('max_grad_norm', (None, 2.))
@pytest.mark.parametrize('num_kv_per_token', (1, 2))
@pytest.mark.parametrize('per_parameter_lr_modulation', (False, True))
@pytest.mark.parametrize('per_head_learned_parameters', (False, True))
@pytest.mark.parametrize('test_store_mask', (False, True))
@pytest.mark.parametrize('store_with_lookahead_value', (False, True))
def test_titans(
    seq_len,
    silu,
    attn_pool_chunks,
    chunk_size,
    momentum,
    qk_rmsnorm,
    heads,
    max_grad_norm,
    num_kv_per_token,
    per_parameter_lr_modulation,
    per_head_learned_parameters,
    test_store_mask,
    store_with_lookahead_value
):
    mem = NeuralMemory(
        dim = 16,
        chunk_size = chunk_size,
        activation = nn.SiLU() if silu else None,
        attn_pool_chunks = attn_pool_chunks,
        max_grad_norm = max_grad_norm,
        num_kv_per_token = num_kv_per_token,
        momentum = momentum,
        qk_rmsnorm = qk_rmsnorm,
        heads = heads,
        per_parameter_lr_modulation = per_parameter_lr_modulation,
        per_head_learned_parameters = per_head_learned_parameters,
        store_with_lookahead_value = store_with_lookahead_value
    )

    seq = torch.randn(2, seq_len, 16)

    store_mask = None

    if test_store_mask:
        store_mask = torch.randint(0, 2, (2, seq_len)).bool()

    retrieved, _ = mem(seq, store_mask = store_mask)

    assert seq.shape == retrieved.shape

def test_return_surprises():

    mem = NeuralMemory(
        dim = 384,
        chunk_size = 2,
        dim_head = 64,
        heads = 4,
    )

    seq = torch.randn(4, 64, 384)

    _, _, (surprises, adaptive_lr) = mem(seq, return_surprises = True)

    assert all([t.shape == (4, 4, 64) for t in (surprises, adaptive_lr)])

@pytest.mark.parametrize('learned_momentum_combine', (False, True))
@pytest.mark.parametrize('learned_combine_include_zeroth', (False, True))
def test_titans_second_order_momentum(
    learned_momentum_combine,
    learned_combine_include_zeroth
):

    mem  = NeuralMemory(
        dim = 384,
        dim_head = 64,
        heads = 2,
        chunk_size = 1,
        batch_size = 2,
        momentum_order = 2,
        learned_momentum_combine = learned_momentum_combine,
        learned_combine_include_zeroth = learned_combine_include_zeroth
    )

    seq = torch.randn(2, 5, 384)

    parallel_retrieved, state = mem(seq)
    assert seq.shape == parallel_retrieved.shape

def test_titans_attn_memory():
    from titans_pytorch.memory_models import MemoryAttention

    mem = NeuralMemory(
        dim = 16,
        chunk_size = 64,
        model = MemoryAttention(
            dim = 16
        )
    )

    seq = torch.randn(2, 1024, 16)
    retrieved, _ = mem(seq)

    assert seq.shape == retrieved.shape

def test_swiglu_ff_memory():
    from titans_pytorch.memory_models import MemorySwiGluMLP

    mem = NeuralMemory(
        dim = 16,
        chunk_size = 2,
        mem_model_norm_add_residual = False,
        model = MemorySwiGluMLP(
            dim = 16,
            depth = 2
        )
    )

    seq = torch.randn(2, 64, 16)
    retrieved, _ = mem(seq)

    assert seq.shape == retrieved.shape

@pytest.mark.parametrize('gated_transition', (True, False))
def test_neural_mem_chaining_chunks(
    gated_transition
):
    mem  = NeuralMemory(
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 16,
        gated_transition = gated_transition
    )

    seq = torch.randn(2, 48, 16)

    parallel_retrieved, state = mem(seq)

    seq_first, seq_second, seq_third = seq.split(16, dim = 1)

    first_retrieved, state = mem(seq_first)
    second_retrieved, state = mem(seq_second, state = state)
    third_retrieved, state = mem(seq_third, state = state)

    assert torch.allclose(parallel_retrieved, torch.cat((first_retrieved, second_retrieved, third_retrieved), dim = 1), atol = 1e-5)

def test_neural_mem_chaining_with_weight_residual():
    mem  = NeuralMemory(
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 64
    )

    mem2 = NeuralMemory(
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 64,
        accept_weight_residual = True
    )

    seq = torch.randn(2, 256, 16)

    seq, state = mem(seq)

    parallel_retrieved, _ = mem2(seq, prev_weights = state.updates)

    seq_first, seq_second = seq[:, :128], seq[:, 128:]

    first_retrieved, state1 = mem2(seq_first, prev_weights = state.updates)
    second_retrieved, state2 = mem2(seq_second, state = state1, prev_weights = state.updates)

    assert torch.allclose(parallel_retrieved, torch.cat((first_retrieved, second_retrieved), dim = 1), atol = 1e-5)

def test_neural_mem_chaining_with_batch_size():
    mem  = NeuralMemory(
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 16,
        batch_size = 64
    )

    seq = torch.randn(2, 112, 16)

    parallel_retrieved, state = mem(seq)

    seq_first, seq_second, seq_third = seq[:, :16], seq[:, 16:64], seq[:, 64:]

    first_retrieved, state = mem(seq_first)
    second_retrieved, state = mem(seq_second, state = state)
    third_retrieved, state = mem(seq_third, state = state)

    parallel_part_retrieved = torch.cat((first_retrieved, second_retrieved, third_retrieved), dim = 1)

    assert torch.allclose(parallel_retrieved, parallel_part_retrieved, atol = 1e-5)

@pytest.mark.parametrize('seq_len', (1023, 17))
@pytest.mark.parametrize('num_persist_mem_tokens', (0, 16))
@pytest.mark.parametrize('num_longterm_mem_tokens', (0, 16))
@pytest.mark.parametrize('neural_mem_gate_attn_output', (False, True))
@pytest.mark.parametrize('neural_mem_segment_len', (8, 16))
@pytest.mark.parametrize('neural_mem_weight_residual', (False, True))
@pytest.mark.parametrize('neural_mem_batch_size', (None, 64))
@pytest.mark.parametrize('neural_mem_qkv_receives_diff_views', (False, True))
@pytest.mark.parametrize('neural_mem_momentum', (False, True))
def test_mac(
    seq_len,
    num_persist_mem_tokens,
    num_longterm_mem_tokens,
    neural_mem_gate_attn_output,
    neural_mem_segment_len,
    neural_mem_weight_residual,
    neural_mem_batch_size,
    neural_mem_qkv_receives_diff_views,
    neural_mem_momentum
):
    transformer = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 16,
        depth = 2,
        num_persist_mem_tokens = num_persist_mem_tokens,
        num_longterm_mem_tokens = num_longterm_mem_tokens,
        segment_len = 128,
        neural_mem_gate_attn_output = neural_mem_gate_attn_output,
        neural_memory_segment_len = neural_mem_segment_len,
        neural_memory_batch_size = neural_mem_batch_size,
        neural_memory_qkv_receives_diff_views = neural_mem_qkv_receives_diff_views,
        neural_mem_weight_residual = neural_mem_weight_residual,
        neural_memory_kwargs = dict(
            momentum = neural_mem_momentum
        )
    )

    x = torch.randint(0, 256, (1, seq_len))

    logits = transformer(x)
    assert logits.shape == (1, seq_len, 256)

@pytest.mark.parametrize('sliding', (False, True))
@pytest.mark.parametrize('mem_layers', ((), None))
@pytest.mark.parametrize('longterm_mems', (0, 4, 16))
@pytest.mark.parametrize('prompt_len', (4, 16))
@torch_default_dtype(torch.float64)
def test_mac_sampling(
    sliding,
    mem_layers,
    longterm_mems,
    prompt_len
):
    transformer = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 16,
        depth = 4,
        segment_len = 32,
        num_persist_mem_tokens = 4,
        num_longterm_mem_tokens = longterm_mems,
        sliding_window_attn = sliding,
        neural_memory_layers = mem_layers,
        neural_mem_gate_attn_output = False
    )

    ids = torch.randint(0, 256, (1, 1023))

    # after much training

    prompt = ids[:, :prompt_len]

    sampled = transformer.sample(prompt, 53, use_cache = False, temperature = 0.)
    sampled_with_cache = transformer.sample(prompt, 53, use_cache = True, temperature = 0.)

    assert torch.allclose(sampled, sampled_with_cache)

@pytest.mark.parametrize('prompt_len', (4, 16, 33, 65))
def test_mac_sampling_with_weight_residual(prompt_len):
    # regression for a prev_weights out-of-bounds slice when sampling with segment
    # length > 1 and weight residual on (upstream lucidrains 1d40c44 / issue #61)
    transformer = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 16,
        depth = 2,
        segment_len = 32,
        num_persist_mem_tokens = 4,
        num_longterm_mem_tokens = 4,
        neural_mem_weight_residual = True,
        neural_mem_gate_attn_output = False,
    )

    prompt = torch.randint(0, 256, (1, prompt_len))

    sampled = transformer.sample(prompt, prompt_len + 65, use_cache = True, temperature = 0., show_progress = False)

    assert sampled.shape == (1, 65)

@pytest.mark.parametrize('seq_len', (2, 64, 256))
@pytest.mark.parametrize('prompt_len', (0, 65))
@pytest.mark.parametrize('mem_chunk_size', (2, 32, 64))
@pytest.mark.parametrize('gated_transition', (False, True))
@torch_default_dtype(torch.float64)
def test_neural_mem_inference(
    seq_len,
    prompt_len,
    mem_chunk_size,
    gated_transition
):

    mem = NeuralMemory(
        dim = 16,
        chunk_size = mem_chunk_size,
        gated_transition = gated_transition
    )

    seq = torch.randn(2, seq_len, 16)
    parallel_retrieved, _ = mem(seq)

    assert seq.shape == parallel_retrieved.shape

    state = None
    sequential_retrieved = []

    # test initial parallel prompt

    test_parallel_prompt = prompt_len > 0 and prompt_len < seq_len

    if test_parallel_prompt:
        prompt, seq = seq[:, :prompt_len], seq[:, prompt_len:]
        retrieved_prompt, state = mem(prompt)
        sequential_retrieved.append(retrieved_prompt)

    # sequential inference

    for token in seq.unbind(dim = 1):

        one_retrieved, state = mem.forward(
            token,
            state = state,
        )

        sequential_retrieved.append(one_retrieved)

    sequential_retrieved = torch.cat(sequential_retrieved, dim = -2)

    assert torch.allclose(parallel_retrieved, sequential_retrieved, atol = 1e-6)

@pytest.mark.parametrize('seq_len', (1023, 17))
@pytest.mark.parametrize('sliding', (True, False))
def test_flex(
    seq_len,
    sliding
):
    if not (torch.cuda.is_available() and exists(flex_attention)):
        pytest.skip()

    attn = SegmentedAttention(
        dim = 16,
        segment_len = 32,
        num_persist_mem_tokens = 1,
        num_longterm_mem_tokens = 1,
        use_flex_attn = True,
        sliding = sliding
    ).cuda()

    seq = torch.randn(1, seq_len, 16).cuda()

    out_flex, _ = attn(seq)
    out_non_flex, _ = attn(seq, disable_flex_attn = True)

    assert torch.allclose(out_flex, out_non_flex, atol = 1e-5)

@pytest.mark.parametrize('use_accelerated', (True, False))
def test_assoc_scan(
    use_accelerated
):
    from titans_pytorch.neural_memory import AssocScan

    if use_accelerated and not torch.cuda.is_available():
        pytest.skip()

    scan = AssocScan(use_accelerated = use_accelerated)

    seq_len = 128
    mid_point = seq_len // 2

    gates = torch.randn(2, seq_len, 16).sigmoid()
    inputs = torch.randn(2, seq_len, 16)

    if use_accelerated:
        gates = gates.cuda()
        inputs = inputs.cuda()

    output = scan(gates, inputs)

    gates1, gates2 = gates[:, :mid_point], gates[:, mid_point:]
    inputs1, inputs2 = inputs[:, :mid_point], inputs[:, mid_point:]

    first_half = scan(gates1, inputs1)

    second_half = scan(gates2, inputs2, prev = first_half[:, -1])
    assert second_half.shape == inputs2.shape

    assert torch.allclose(output[:, -1], second_half[:, -1], atol = 1e-5)

def test_mem_state_detach():
    from titans_pytorch.neural_memory import mem_state_detach

    mem = NeuralMemory(
        dim = 384,
        chunk_size = 2,
        qk_rmsnorm = True,
        dim_head = 64,
        heads = 4,
    )

    seq = torch.randn(4, 64, 384)

    state = None

    for _ in range(2):
        parallel_retrieved, state = mem(seq, state = state)
        state = mem_state_detach(state)
        parallel_retrieved.sum().backward()

# atlas extensions

def test_muon_custom_steps():
    mem = NeuralMemory(
        dim = 16,
        chunk_size = 4,
        spectral_norm_surprises = True,
        muon_ns_steps = 3,
        muon_ns_eps = 1e-6,
    )

    seq = torch.randn(2, 32, 16)
    retrieved, _ = mem(seq)
    assert seq.shape == retrieved.shape
    retrieved.sum().backward()

def test_polynomial_features():
    mem = NeuralMemory(
        dim = 16,
        chunk_size = 4,
        polynomial_degree = 2,
    )

    seq = torch.randn(2, 32, 16)
    retrieved, _ = mem(seq)
    assert seq.shape == retrieved.shape
    retrieved.sum().backward()

def test_polynomial_features_multihead():
    mem = NeuralMemory(
        dim = 16,
        chunk_size = 4,
        dim_head = 8,
        heads = 2,
        polynomial_degree = 2,
    )

    seq = torch.randn(2, 32, 16)
    retrieved, _ = mem(seq)
    assert seq.shape == retrieved.shape
    retrieved.sum().backward()

def test_polynomial_features_degree_3():
    mem = NeuralMemory(
        dim = 16,
        chunk_size = 4,
        polynomial_degree = 3,
    )

    seq = torch.randn(2, 32, 16)
    retrieved, _ = mem(seq)
    assert seq.shape == retrieved.shape
    retrieved.sum().backward()

def test_omega_context_1_equals_baseline():
    """omega_context=1 must produce identical results to default (no omega rule)"""
    torch.manual_seed(42)
    mem = NeuralMemory(dim = 16, chunk_size = 4, omega_context = 1)
    seq = torch.randn(2, 32, 16)
    out1, _ = mem(seq)

    torch.manual_seed(42)
    mem2 = NeuralMemory(dim = 16, chunk_size = 4)
    out2, _ = mem2(seq)

    assert torch.allclose(out1, out2, atol = 1e-5)

def test_omega_context_changes_output():
    """omega_context > 1 must produce different results than per-chunk updates"""
    torch.manual_seed(42)
    mem1 = NeuralMemory(dim = 16, chunk_size = 4, omega_context = 1)
    seq = torch.randn(2, 64, 16)
    out1, _ = mem1(seq)

    torch.manual_seed(42)
    mem2 = NeuralMemory(dim = 16, chunk_size = 4, omega_context = 4)
    out2, _ = mem2(seq)

    assert out1.shape == out2.shape
    assert not torch.allclose(out1, out2, atol = 1e-5), 'omega_context > 1 should produce different output'

def test_omega_context_partial_window():
    """omega_context between 1 and chunk_size should work"""
    mem = NeuralMemory(dim = 16, chunk_size = 8, omega_context = 4)
    seq = torch.randn(2, 64, 16)
    retrieved, _ = mem(seq)
    assert seq.shape == retrieved.shape
    retrieved.sum().backward()

def _omega_window_reference(g, gates, c):
    """Brute-force reference for the omega window (paper Section 3.2, Eq 9):
    G_i = sum_{k=0}^{c-1} gamma_k^(i) * grad[i - (c-1-k)], zero outside the segment.
    g: (B, T, *weight_shape), gates: (B, T, c). Loop implementation on purpose —
    any vectorization shortcut could share a bug with the code under test."""
    out = torch.zeros_like(g)
    seq_len = g.shape[1]
    for i in range(seq_len):
        for k in range(c):
            j = i - (c - 1 - k)
            if j < 0:
                continue
            gamma = gates[:, i, k].reshape(-1, *([1] * (g.ndim - 2)))
            out[:, i] += g[:, j] * gamma
    return out

def test_omega_window_matches_reference():
    """apply_omega_window must equal the brute-force paper equation exactly.
    Value-level regression for the non-sliding-window bug (found 2026-09-01):
    the pre-fix slice direction returned every gradient to its original index,
    reducing the omega rule to a per-position gate-sum LR multiplier with zero
    cross-token mixing. Shape tests cannot catch this class of bug."""
    from tensordict import TensorDict
    from titans_pytorch.neural_memory import apply_omega_window

    torch.manual_seed(0)
    c = 8
    g = torch.randn(3, 20, 4, 5, dtype = torch.float64)
    gates = torch.rand(3, 20, c, dtype = torch.float64)

    out = apply_omega_window(grads = TensorDict({'w': g}), context_gates = gates, omega_context = c)['w']
    ref = _omega_window_reference(g = g, gates = gates, c = c)

    assert torch.allclose(out, ref, atol = 1e-12)

def test_omega_window_impulse_slides():
    """An impulse gradient at token 0 with all-ones gates must appear in the
    window of every one of the next c positions — the defining property of a
    sliding window. The pre-fix code produced [1, 0, ..., 0]."""
    from tensordict import TensorDict
    from titans_pytorch.neural_memory import apply_omega_window

    c = 8
    g = torch.zeros(1, c, 1, 1)
    g[0, 0] = 1.
    gates = torch.ones(1, c, c)

    out = apply_omega_window(grads = TensorDict({'w': g}), context_gates = gates, omega_context = c)['w']

    assert torch.allclose(out.flatten(), torch.ones(c)), (
        f'impulse at token 0 must reach all {c} window positions, got {out.flatten().tolist()}'
    )

def test_omega_window_crosses_chunk_boundary():
    """The window must mix gradients across vmap-chunk boundaries — all chunks in
    a store segment share the same segment-start base weights (exactness holds
    because accept_weight_residual, which would give chunks different base
    points, is asserted off for omega). Also asserts store_memories hands
    apply_omega_window the full segment token axis, not per-chunk blocks."""
    from tensordict import TensorDict
    import titans_pytorch.neural_memory as nm

    # unit level: impulse at position 3 with c=4 must propagate into positions
    # 4, 5, 6 (the old chunk_size=4 boundary sat between 3 and 4)
    c = 4
    g = torch.zeros(1, 8, 1, 1)
    g[0, 3] = 1.
    gates = torch.ones(1, 8, c)
    out = nm.apply_omega_window(grads = TensorDict({'w': g}), context_gates = gates, omega_context = c)['w'].flatten()
    assert torch.allclose(out[3:7], torch.ones(4)), 'impulse must appear in the next c-1 positions across the chunk boundary'
    assert torch.allclose(out[[0, 1, 2, 7]], torch.zeros(4))

    # integration level: grads reach the window with the full token axis
    seen_token_dims = []
    orig = nm.apply_omega_window

    def spy(grads, context_gates, omega_context):
        seen_token_dims.append(next(iter(grads.values())).shape[1])
        return orig(grads = grads, context_gates = context_gates, omega_context = omega_context)

    nm.apply_omega_window = spy
    try:
        mem = NeuralMemory(dim = 16, chunk_size = 4, omega_context = 4)
        seq = torch.randn(2, 32, 16)
        retrieved, _ = mem(seq)
    finally:
        nm.apply_omega_window = orig

    assert seen_token_dims == [32], (
        f'expected the window to see the full 32-token segment axis, got {seen_token_dims}'
    )

def test_omega_context_exceeds_chunk_size():
    """omega_context may exceed the vmap chunk size — the window lives on the
    segment token axis, not the chunk axis. Previously asserted out."""
    mem = NeuralMemory(dim = 16, chunk_size = 4, omega_context = 8)
    seq = torch.randn(2, 32, 16)
    retrieved, _ = mem(seq)
    assert seq.shape == retrieved.shape
    retrieved.sum().backward()

def test_omega_window_context_longer_than_segment():
    """Window size c larger than the segment token count: taps reaching before
    the segment start contribute zeros (the offset >= num_tokens guard)."""
    from tensordict import TensorDict
    from titans_pytorch.neural_memory import apply_omega_window

    torch.manual_seed(1)
    c, seq_len = 16, 8
    g = torch.randn(2, seq_len, 3, dtype = torch.float64)
    gates = torch.rand(2, seq_len, c, dtype = torch.float64)

    out = apply_omega_window(grads = TensorDict({'w': g}), context_gates = gates, omega_context = c)['w']
    ref = _omega_window_reference(g = g, gates = gates, c = c)

    assert torch.allclose(out, ref, atol = 1e-12)

def test_omega_with_momentum_backward():
    """omega rule + momentum must support gradient flow"""
    mem = NeuralMemory(dim = 16, chunk_size = 4, omega_context = 4, momentum = True)
    seq = torch.randn(2, 64, 16)
    retrieved, _ = mem(seq)
    retrieved.sum().backward()

    for p in mem.parameters():
        if p.requires_grad:
            assert p.grad is not None

def test_atlas_config():
    """all three Atlas extensions combined via atlas_config()"""
    config = NeuralMemory.atlas_config()
    mem = NeuralMemory(dim = 16, chunk_size = 8, **config)

    seq = torch.randn(2, 64, 16)
    retrieved, _ = mem(seq)
    assert seq.shape == retrieved.shape
    retrieved.sum().backward()

    for p in mem.parameters():
        if p.requires_grad:
            assert p.grad is not None

def test_atlas_config_overrides():
    """atlas_config() accepts overrides"""
    config = NeuralMemory.atlas_config(omega_context = 4, polynomial_degree = 3)
    assert config['omega_context'] == 4
    assert config['polynomial_degree'] == 3
    assert config['momentum'] == True

def test_per_head_learned_parameters_own_storage():
    """Per-head memory parameters must have independent storage, not a stride-0
    broadcast view. Shared storage causes load_state_dict to fail and makes all
    heads stay bit-identical under AdamW updates (paper-fidelity regression)."""
    mem = NeuralMemory(
        dim = 16,
        chunk_size = 4,
        dim_head = 8,
        heads = 16,
        per_head_learned_parameters = True,
    )
    for name, p in mem.memory_model_parameters.named_parameters():
        assert p.stride(0) != 0, (
            f'memory_model_parameters.{name} has stride 0 on the head dim '
            f'— heads share storage. Expected independent per-head slices.'
        )

def test_polynomial_features_includes_constant_term():
    """φ(x) must include the degree-0 (constant) Taylor term — without it the
    Taylor approximation of softmax (Section 3.1) is missing the leading 1.
    expanded_dim must be 1 + Sigma_{d=1..p} C(d+dim-1, d)."""
    from titans_pytorch.neural_memory import PolynomialFeatures
    poly = PolynomialFeatures(dim = 16, degree = 2, project_back = False)
    # degree-0: 1, degree-1: 16, degree-2: C(17,2) = 136 -> total 1 + 16 + 136 = 153
    assert poly.expanded_dim == 153, f'expected 1 + 16 + 136 = 153, got {poly.expanded_dim}'
    assert poly.coefficients.shape == (3,), f'expected coefficients length degree+1=3, got {poly.coefficients.shape}'
    # init values: 1/0!, 1/1!, 1/2!
    assert torch.allclose(poly.coefficients, torch.tensor([1.0, 1.0, 0.5]))

def test_polynomial_features_constant_in_forward_output():
    """Forward output's first element along the feature dim must be the
    degree-0 constant feature (broadcast from coefficients[0] init=1)."""
    from titans_pytorch.neural_memory import PolynomialFeatures
    poly = PolynomialFeatures(dim = 16, degree = 2, project_back = False)
    x = torch.randn(2, 5, 16)
    out = poly(x)
    assert out.shape == (2, 5, 153)
    # The first feature-dim slot is the degree-0 constant: 1 * coefficients[0] = 1.0 at init.
    assert torch.allclose(out[..., 0], torch.full((2, 5), 1.0))

def test_atlas_config_poly_project_back_default():
    """atlas_config() defaults poly_project_back=True as the documented
    production tradeoff. The strict Eq (56) reading would have the MLP
    consume phi(k) directly (project_back=False), but Phase 0 OOM evidence
    (job 40049757) showed the asymmetric path saturates H100 at the
    omega-windowed gradient accumulation. project_back=True keeps the
    polynomial features (Taylor-init coefficients, including the degree-0
    constant) but compresses phi(k) -> dim_head before the MLP, capping
    capacity at O(dim_hidden). Tracked as a Phase 3+ scaling question in
    GitHub issue #17."""
    config = NeuralMemory.atlas_config()
    assert config.get('poly_project_back') is True, (
        'atlas_config() defaults to poly_project_back=True per the documented '
        'production tradeoff. Override with atlas_config(poly_project_back=False) '
        'when running on FSDP / model-parallel infrastructure that can absorb '
        'the asymmetric MLP memory cost.'
    )

def test_atlas_config_project_back_path_runs():
    """End-to-end: atlas_config() must produce a NeuralMemory whose
    poly_features projects expanded_dim back to dim_head (the production
    path), and forward + backward run without dimension errors."""
    config = NeuralMemory.atlas_config()
    mem = NeuralMemory(dim = 16, chunk_size = 8, **config)
    # poly_features should be present and project_back enabled.
    assert mem.poly_features is not None, 'poly_features must be constructed'
    assert mem.poly_features.projection is not None, (
        'poly_features.projection must be present (poly_project_back=True). '
        'See atlas_config docstring for why this is the documented default.'
    )
    # Forward + backward succeed
    seq = torch.randn(2, 64, 16)
    retrieved, _ = mem(seq)
    assert retrieved.shape == seq.shape
    retrieved.sum().backward()

def test_memory_mlp_asymmetric_dim_in_dim_out():
    """MemoryMLP must accept distinct dim_in and dim_out for the asymmetric
    Atlas path (input = poly.expanded_dim, output = dim_head)."""
    from titans_pytorch.memory_models import MemoryMLP
    mlp = MemoryMLP(dim = 16, depth = 2, dim_in = 153, dim_out = 16)
    # weights[0]: (153, hidden=32), weights[1]: (32, 16)
    assert mlp.weights[0].shape == (153, 32), f'first weight shape {tuple(mlp.weights[0].shape)}'
    assert mlp.weights[1].shape == (32, 16), f'last weight shape {tuple(mlp.weights[1].shape)}'
    x = torch.randn(2, 5, 153)
    out = mlp(x)
    assert out.shape == (2, 5, 16)
    out.sum().backward()

def test_atlas_muon_actually_applies_to_matrix_surprises():
    """Regression guard: an adversarial review (2026-04-28) raised the concern
    that newtonschulz5's `if ndim <= 3: return t` early-return might silently
    skip Muon on per-head Atlas surprises. Empirically the surprise tensor in
    the per_head_learned_parameters path is 4D — (batch*heads, num_tokens,
    in, out) for matrix params after vmap(grad) — so Muon DOES fire.
    This test asserts at least one matrix surprise is transformed by
    newtonschulz5 during a real Atlas forward+backward, so any future refactor
    that drops a dim (and silently disables Muon) trips immediately.
    Vector params (e.g. norm.gamma → 3D surprise) correctly skip Muon per
    the standard Muon convention (Muon is for matrices, not vectors)."""
    import titans_pytorch.neural_memory as nm
    orig_ns5 = nm.newtonschulz5
    matrix_call_count = 0
    matrix_transform_count = 0
    try:
        def spy_ns5(t, **kwargs):
            nonlocal matrix_call_count, matrix_transform_count
            out = orig_ns5(t, **kwargs)
            if t.ndim >= 4:  # matrix params, the ones Muon should hit
                matrix_call_count += 1
                if not torch.allclose(t, out):
                    matrix_transform_count += 1
            return out
        nm.newtonschulz5 = spy_ns5

        config = NeuralMemory.atlas_config()
        mem = NeuralMemory(dim = 16, chunk_size = 8, **config)
        seq = torch.randn(2, 64, 16)
        retrieved, _ = mem(seq)
        retrieved.sum().backward()
    finally:
        nm.newtonschulz5 = orig_ns5

    assert matrix_call_count > 0, (
        'newtonschulz5 was never called with a 4D matrix surprise tensor — '
        'Muon path may have been refactored away. Atlas requires Muon (Section 5, Eq (32)).'
    )
    assert matrix_transform_count == matrix_call_count, (
        f'newtonschulz5 returned its input unchanged on {matrix_call_count - matrix_transform_count} '
        f'of {matrix_call_count} matrix calls. Muon is silently no-op-ing on Atlas matrix surprises. '
        f'Check the early-return condition and the surprise tensor shape.'
    )

def test_atlas_config_enables_per_token_retrieve():
    """atlas_config() must default to per_token_retrieve=True. The constructor
    default is False (paper-deviating per-chunk approximation); atlas_config()
    is the only place we promise paper-faithful behavior, so this default
    matters for every Atlas run that goes through it."""
    config = NeuralMemory.atlas_config()
    assert config['per_token_retrieve'] is True, (
        'atlas_config() must enable per_token_retrieve — Eq (41) gives the per-token '
        'weight state and Eq (42) the read (y_t = M_t(q_t)), retrieval at every token. Falling back to '
        'per-chunk retrieve silently re-runs Titans on the retrieve side.'
    )

def test_atlas_config_produces_per_token_retrieve_chunk_size():
    """When atlas_config() is applied to NeuralMemory, the constructor must
    actually set retrieve_chunk_size=1 (the wire form of per-token retrieve)."""
    config = NeuralMemory.atlas_config()
    mem = NeuralMemory(dim = 16, chunk_size = 8, **config)
    assert mem.retrieve_chunk_size == 1, (
        f'expected retrieve_chunk_size=1 with atlas_config + omega_context>1, '
        f'got {mem.retrieve_chunk_size}'
    )

def test_atlas_config_per_token_retrieve_forward_backward():
    """End-to-end: atlas_config defaults must produce a working forward +
    backward through the per-token retrieve path."""
    config = NeuralMemory.atlas_config()
    mem = NeuralMemory(dim = 16, chunk_size = 8, **config)
    seq = torch.randn(2, 64, 16)
    retrieved, _ = mem(seq)
    assert seq.shape == retrieved.shape
    retrieved.sum().backward()
    for p in mem.parameters():
        if p.requires_grad:
            assert p.grad is not None

def test_detach_segment_memory_truncates_outer_loop_grad():
    """detach_segment_memory=True must actually truncate the autograd graph
    across segments. Verify by comparing gradient norms on store-side params
    (to_keys, to_values, to_adaptive_step, etc.) between the detach=True and
    detach=False paths under identical seeds and inputs.

    With detach=True, gradients on store-side params should be substantially
    smaller because they only receive contributions from each segment's direct
    forward — not from the cross-segment chain through the per-segment memory
    state. If detach is silently a no-op (e.g., a future refactor moves the
    detach call out of the loop), the two grad-norm dicts would be identical.

    Same runtime-spy methodology as test_atlas_muon_actually_applies — the
    code can't tell us the detach is firing, but the gradient norms can.
    """
    def grad_norms(detach):
        torch.manual_seed(42)
        mem = NeuralMemory(
            dim = 16, dim_head = 8, heads = 2,
            chunk_size = 4, batch_size = 16,
            detach_segment_memory = detach,
        )
        torch.manual_seed(0)
        seq = torch.randn(2, 64, 16)  # 64 tokens / batch_size=16 = 4 segments
        out, _ = mem(seq)
        out.sum().backward()
        return {
            n: p.grad.norm().item()
            for n, p in mem.named_parameters()
            if p.requires_grad and p.grad is not None
        }

    norms_detach = grad_norms(detach = True)
    norms_full = grad_norms(detach = False)

    store_side_params = ['to_keys.weight', 'to_values.weight', 'to_adaptive_step.0.weight']
    for param_name in store_side_params:
        d, f = norms_detach[param_name], norms_full[param_name]
        # detach=True should produce STRICTLY SMALLER gradient (cross-segment
        # path is cut). A 2× ratio is the looser regression bound; in practice
        # we observe ~30-50× shrinkage for these params.
        assert d < f * 0.5, (
            f'{param_name}: detach_segment_memory=True did not measurably reduce '
            f'gradient norm (detach={d:.4f}, full={f:.4f}, ratio={d/f:.3f}). '
            f'Detach may be silently a no-op.'
        )

def test_atlas_adaptive_lr_affects_muon_update_magnitude():
    """Regression: the adaptive learning rate η must affect the magnitude of the
    Atlas (Muon) memory update. Paper Section 5, Eq (32) applies η OUTSIDE Newton-Schulz
    (M_t = α M_{t-1} − η_t·NS-5(S_t), raw gradient inside S_t). Because newtonschulz5
    normalizes its input by norm and is scale-invariant (NS5(c·S) = NS5(S)), folding η
    into the surprise — as the code originally did, via the grad loss weight — silently
    cancels it, leaving the learned adaptive step dead on the store side. Verified against
    the paper on 2026-06-18.

    We scale η globally via default_step_transform_max_lr (η = sigmoid(logit)·max_lr) and
    assert the retrieved output changes. With the bug, the two outputs are bit-identical
    because η is washed out by NS-5; with the fix (η applied after NS-5) they differ.
    Same runtime-difference methodology as test_detach_segment_memory_truncates."""
    config = NeuralMemory.atlas_config()

    torch.manual_seed(42)
    mem_small_lr = NeuralMemory(dim = 16, chunk_size = 8, default_step_transform_max_lr = 0.1, **config)
    seq = torch.randn(2, 64, 16)
    out_small, _ = mem_small_lr(seq)

    torch.manual_seed(42)
    mem_large_lr = NeuralMemory(dim = 16, chunk_size = 8, default_step_transform_max_lr = 1.0, **config)
    out_large, _ = mem_large_lr(seq)

    # this discriminates only because omega forces the other adaptive_lr-dependent store
    # terms off (no lookahead, num_kv_per_token=1, no per-parameter lr modulation), so max_lr
    # reaches the output solely through the post-NS η multiply. guard against a NaN regression
    # trivially satisfying `not allclose` (NaN != NaN).
    assert out_small.isfinite().all() and out_large.isfinite().all(), 'non-finite memory output'

    assert not torch.allclose(out_small, out_large, atol = 1e-6), (
        'Adaptive learning rate η has no effect on the Atlas/Muon update — it is being '
        'cancelled by the scale-invariant Newton-Schulz normalization. η must be applied '
        'OUTSIDE NS-5 (paper Section 5, Eq (32)), not folded into the surprise as the grad loss weight.'
    )

def test_value_conv_in_atlas_config():
    """Paper Section 5 architectural backbone: keys, VALUES, and queries all get a
    short causal conv (size 4) after their projections. The repo previously conv'd
    only keys and queries — values went straight from projection to storage
    (2026-09-01 audit)."""
    config = NeuralMemory.atlas_config()
    mem = NeuralMemory(dim = 16, chunk_size = 8, **config)
    assert mem.value_conv is not None, 'atlas_config (short_conv_size=4) must construct a value conv'

    mem_no_conv = NeuralMemory(dim = 16, chunk_size = 8, **NeuralMemory.atlas_config(short_conv_size = 0))
    assert mem_no_conv.value_conv is None

    seq = torch.randn(2, 64, 16)
    retrieved, _ = mem(seq)
    assert retrieved.shape == seq.shape
    retrieved.sum().backward()
    assert mem.value_conv.conv.weight.grad is not None, 'value conv must participate in the store path'

def test_store_path_receives_outer_loop_grads_in_mac_geometry():
    """N1 regression guard (2026-09-01 audit): in the shipped 170M geometry the
    interleaved sequence (train ctx + longterm mem tokens) exceeded
    neural_memory_batch_size, splitting storage into two segments, and
    detach_segment_memory=True detached the first — so the learned memory init
    (W0) received ZERO outer-loop gradient for the whole run (frozen at random
    init; DDP find_unused_parameters=True masked the symptom) and store-side
    params trained on only the trailing ~5% of tokens. The atlas config now
    ships detach_segment_memory=False. This test reproduces the trigger
    geometry at reduced size (interleaved 268 > batch_size 256 -> segments
    [256, 12]) and pins both halves of the behavior."""
    from titans_pytorch.mac_transformer import MemoryAsContextTransformer

    def build_and_backward(detach):
        torch.manual_seed(42)
        mem_kwargs = NeuralMemory.atlas_config()
        mem_kwargs.update(
            dim_head = 8,
            heads = 4,
            use_sequential_scan = True,
            default_step_transform_max_lr = 1e-1,
            detach_segment_memory = detach,
        )
        model = MemoryAsContextTransformer(
            num_tokens = 256,
            dim = 32,
            depth = 2,
            segment_len = 64,
            num_persist_mem_tokens = 4,
            num_longterm_mem_tokens = 4,
            neural_memory_layers = (1,),
            neural_memory_segment_len = 8,
            neural_memory_batch_size = 256,
            use_flex_attn = False,
            sliding_window_attn = True,
            neural_memory_kwargs = mem_kwargs,
        )
        mem = next(layer[4] for layer in model.layers if layer[4] is not None)
        x = torch.randint(0, 256, (1, 257))
        loss = model(x, return_loss = True)
        loss.backward()
        return mem

    # shipped config (detach off): the learned init and store-side params train

    mem = build_and_backward(detach = False)
    w0_grad = mem.memory_model_parameters[0].grad
    assert w0_grad is not None, (
        'learned memory init (W0) must receive outer-loop gradient with '
        'detach_segment_memory=False — if this fails, the store path is starved again'
    )
    assert w0_grad.abs().sum() > 0, 'W0 gradient exists but is identically zero — starved by another route'
    to_keys_weight = mem.to_keys.weight if isinstance(mem.to_keys, nn.Linear) else mem.to_keys[0].weight
    assert to_keys_weight.grad is not None
    assert to_keys_weight.grad.abs().sum() > 0

    # characterization: detach=True in this geometry silently freezes W0 —
    # the reason the atlas training config must not re-enable it

    mem_detached = build_and_backward(detach = True)
    assert mem_detached.memory_model_parameters[0].grad is None, (
        'expected detach_segment_memory=True to cut all outer-loop gradient to the '
        'learned memory init in the two-segment geometry — if it now receives '
        'gradient, the detach semantics changed and this guard needs re-derivation'
    )

def test_no_muon_omega_eta_affects_output():
    """Single-variable ablation guard (2026-09-01 review round): the adaptive lr
    eta is applied per target position OUTSIDE the momentum for ALL omega paths,
    with or without Newton-Schulz — so the no-muon ablation differs from atlas by
    exactly Newton-Schulz, not by eta placement too. Before the window fix the two
    placements were equivalent (no cross-token mixing); after it they diverge, and
    leaving eta as the grad loss weight in the no-muon path would have made the
    ablation change two variables at once."""
    config = NeuralMemory.atlas_config(spectral_norm_surprises = False)

    torch.manual_seed(42)
    mem_small_lr = NeuralMemory(dim = 16, chunk_size = 8, default_step_transform_max_lr = 0.1, **config)
    seq = torch.randn(2, 64, 16)

    # placement probe: the grad fn must receive a RAW loss weight (all ones — the
    # store mask), not eta. eta folded in as the loss weight is the pre-fix
    # two-variable-ablation regression: without allclose(ones) here, eta rides
    # inside the windowed gradients instead of scaling per target position.
    seen_loss_weights = []
    orig_grad_fn = mem_small_lr.per_token_grad_fn

    def spy(params, keys, loss_weights, values):
        seen_loss_weights.append(loss_weights.detach().clone())
        return orig_grad_fn(params, keys, loss_weights, values)

    mem_small_lr.per_token_grad_fn = spy
    out_small, _ = mem_small_lr(seq)
    mem_small_lr.per_token_grad_fn = orig_grad_fn

    assert len(seen_loss_weights) > 0, 'spy never fired — instrument dead'
    assert torch.allclose(seen_loss_weights[0], torch.ones_like(seen_loss_weights[0])), (
        'the no-muon omega path fed eta to the grad fn as the loss weight — eta '
        'placement has regressed inside the windowed gradient, making the no-muon '
        'ablation a two-variable change (Newton-Schulz AND eta placement)'
    )

    torch.manual_seed(42)
    mem_large_lr = NeuralMemory(dim = 16, chunk_size = 8, default_step_transform_max_lr = 1.0, **config)
    out_large, _ = mem_large_lr(seq)

    assert out_small.isfinite().all() and out_large.isfinite().all(), 'non-finite memory output'
    assert not torch.allclose(out_small, out_large, atol = 1e-6), (
        'adaptive lr has no effect on the no-muon omega path — eta is being dropped entirely'
    )

def test_shipped_atlas_memory_config_pins():
    """Pin the shipped atlas memory config. Mutation testing (2026-09-01 review
    round) showed nothing guarded it: flipping detach_segment_memory back to True
    left the entire 9,839-test suite green, because the mechanism tests build
    their own kwargs. These values are the decision surface of the 2026-09-01
    audit — change them deliberately or not at all."""
    from experiments.configs import MEMORY_CONFIGS

    atlas = MEMORY_CONFIGS['atlas']
    assert atlas['detach_segment_memory'] is False, 'detach starves the store path in the trained geometry'
    assert atlas['short_conv_size'] == 4, 'paper Section 5 backbone: conv on keys/values/queries'
    assert atlas['omega_context'] == 8
    assert atlas['per_token_retrieve'] is True
    assert atlas['spectral_norm_surprises'] is True
    assert atlas['polynomial_degree'] == 2
    assert atlas['per_token_updates'] is True, 'the per-token store path must not depend on the window size (2026-09-02 ablation confound)'


def _perturbation_reach(mem, seq, store, position):
    """First retrieve position whose output changes when the STORE input at
    `position` is perturbed with the queries fixed. `position` itself is the
    per-token post-update read M_t(q_t); a later position means the retrieve
    reads a coarser (per-chunk) state."""
    with torch.no_grad():
        base, _ = mem(seq, store_seq = store)
        perturbed = store.clone()
        perturbed[0, position] += 1.0
        out, _ = mem(seq, store_seq = perturbed)
    changed = ((out - base).abs().amax(dim = (0, 2)) > 1e-12).nonzero().flatten()
    return int(changed[0]) if changed.numel() else None


def _no_omega_test_memory(**overrides):
    torch.manual_seed(0)
    kwargs = NeuralMemory.atlas_config(dim_head = 8, heads = 4, use_sequential_scan = True)
    kwargs.update(overrides)
    return NeuralMemory(dim = 32, chunk_size = 8, batch_size = 64, **kwargs).double().eval()


def test_no_omega_ablation_is_atlas_minus_the_window():
    """ABLATIONS['no-omega'] must differ from the atlas memory by the omega window
    and its gamma gates only. Before 2026-09-02, omega_context=1 silently left the
    per-token path: one summed gradient per store chunk (at the same segment-start
    weights), chunk-pooled gates, eta folded into the gradient (its magnitude then
    normalized away by Muon on matrix parameters), a per-chunk scan and per-chunk
    retrieve — five changes for one ablation. Behavioural probe: perturb the store input at one
    position with the queries fixed; the per-token post-update read reaches the
    retrieve at that position, the chunk-wise read only at the end of its chunk."""
    from experiments.configs import MEMORY_CONFIGS, ABLATIONS

    assert ABLATIONS['no-omega'] == dict(omega_context = 1), 'no-omega must touch the window only'

    torch.manual_seed(1)
    seq = torch.randn(1, 64, 32, dtype = torch.float64)
    store = torch.randn(1, 64, 32, dtype = torch.float64)

    atlas = _no_omega_test_memory(short_conv_size = 0)
    no_omega = _no_omega_test_memory(short_conv_size = 0, **ABLATIONS['no-omega'])
    # the control differs from no_omega by the store path alone
    chunk_wise = _no_omega_test_memory(
        short_conv_size = 0, omega_context = 1, per_token_updates = False, per_token_retrieve = False,
    )

    assert no_omega.per_token_updates and no_omega.retrieve_chunk_size == 1
    assert not hasattr(no_omega, 'to_context_gates'), 'the gamma gates belong to the window'
    assert hasattr(atlas, 'to_context_gates')

    for position in (3, 10, 21):
        assert _perturbation_reach(atlas, seq, store, position) == position
        assert _perturbation_reach(no_omega, seq, store, position) == position
    # instrument: the chunk-wise path reads a coarser state — the change lands at the
    # end of the store chunk (7 / 15 / 23), never at the position itself
    assert [_perturbation_reach(chunk_wise, seq, store, p) for p in (3, 10, 21)] == [7, 15, 23]

    # the shipped resolution (memory config + ablation) builds the per-token path
    resolved = {**MEMORY_CONFIGS['atlas'], **ABLATIONS['no-omega'], 'dim_head': 8, 'heads': 4}
    mem = NeuralMemory(dim = 32, chunk_size = 8, batch_size = 64, **resolved)
    assert mem.per_token_updates and mem.retrieve_chunk_size == 1 and not hasattr(mem, 'to_context_gates')


def test_per_token_path_at_c1_equals_window_with_only_newest_tap():
    """The no-omega ablation and atlas share every parameter and code path except
    the window: with the gamma gates forced to select only the newest tap (older
    taps -> sigmoid(-60), newest -> sigmoid(60)) the omega_context=8 memory
    reproduces the omega_context=1 per-token memory to fp64 precision. A path
    difference anywhere else (gates, eta placement, scan, retrieve) breaks the
    equality; the pre-fix chunk-wise fallback differed by O(1). 61 tokens, not
    a multiple of the store chunk: the retrieve's per-token branch then pads
    the cached remainder, and a retrieve keyed on the window instead of the
    path crashes there (mutation, adversarial review 2026-09-02) — every
    other c=1 input in this file is chunk-aligned and never enters it."""
    no_omega, atlas = _no_omega_test_memory(omega_context = 1), _no_omega_test_memory(omega_context = 8)
    missing, unexpected = atlas.load_state_dict(no_omega.state_dict(), strict = False)
    assert not unexpected
    assert missing and all(key.startswith('to_context_gates') for key in missing), missing

    heads, window = 4, 8
    gate = atlas.to_context_gates[0]
    with torch.no_grad():
        gate.weight.zero_()
        bias = torch.full((heads, window), -60.0, dtype = gate.bias.dtype)
        bias[:, -1] = 60.0
        gate.bias.copy_(bias.reshape(-1))

    torch.manual_seed(1)
    seq = torch.randn(2, 61, 32, dtype = torch.float64)
    with torch.no_grad():
        out_no_omega, _ = no_omega(seq)
        out_atlas, _ = atlas(seq)
    max_dev = (out_no_omega - out_atlas).abs().max().item()
    assert torch.allclose(out_no_omega, out_atlas, atol = 1e-12, rtol = 0), f'max |dev| {max_dev:.3e}'

    # instrument: opening every tap makes the window mix and the outputs diverge
    with torch.no_grad():
        gate.bias.fill_(60.0)
        out_mixed, _ = atlas(seq)
    assert (out_mixed - out_no_omega).abs().max().item() > 1e-3


def test_no_omega_eta_stays_outside_newton_schulz_at_c1(monkeypatch):
    """eta placement is part of what no-omega must share with atlas: on the
    per-token path eta is applied per target position AFTER Newton-Schulz at
    omega_context=1 exactly as at 8 (same code). Placement instrument: spy the
    tensors handed to newtonschulz5 while scaling eta 10x through
    default_step_transform_max_lr — on the per-token path they are identical
    (eta has not been applied yet) while the retrieved output changes (eta is
    live); on the chunk-wise path, the memory the old no-omega silently ran,
    the same tensors scale 10x because eta is folded into the gradient as a
    loss weight before NS-5 (whose skipped vector parameters and per-token
    relative weights are then all that keeps eta alive there)."""
    import titans_pytorch.neural_memory as neural_memory_module

    original = neural_memory_module.newtonschulz5
    torch.manual_seed(1)
    seq = torch.randn(2, 64, 32, dtype = torch.float64)

    def ns_inputs_and_output(max_lr, **overrides):
        seen = []

        def spy(update, **kwargs):
            seen.append(update.detach().clone())
            return original(update, **kwargs)

        monkeypatch.setattr(neural_memory_module, 'newtonschulz5', spy)
        with torch.no_grad():
            out, _ = _no_omega_test_memory(default_step_transform_max_lr = max_lr, **overrides)(seq)
        monkeypatch.setattr(neural_memory_module, 'newtonschulz5', original)
        assert seen, 'instrument: Newton-Schulz was never entered'
        return seen, out

    small_inputs, small_out = ns_inputs_and_output(0.1, omega_context = 1)
    large_inputs, large_out = ns_inputs_and_output(1.0, omega_context = 1)
    assert len(small_inputs) == len(large_inputs)
    for a, b in zip(small_inputs, large_inputs):
        assert torch.equal(a, b), 'eta reached Newton-Schulz on the per-token path'
    assert not torch.allclose(small_out, large_out, atol = 1e-8), 'eta has no effect on the no-omega path'

    chunk_wise = dict(omega_context = 1, per_token_updates = False, per_token_retrieve = False)
    small_inputs, _ = ns_inputs_and_output(0.1, **chunk_wise)
    large_inputs, _ = ns_inputs_and_output(1.0, **chunk_wise)
    for a, b in zip(small_inputs, large_inputs):
        assert torch.allclose(10 * a, b, atol = 1e-9, rtol = 0), 'instrument: the chunk-wise path must fold eta into the gradient'


def test_per_token_path_flags_refuse_contradictions():
    """The store path is explicit: the two contradictory combinations fail loudly
    instead of silently picking a path (the 2026-09-02 confound), and the library
    default still follows the window."""
    with pytest.raises(ValueError, match = 'per_token_updates'):
        NeuralMemory(dim = 16, chunk_size = 8, omega_context = 1, per_token_retrieve = True)
    with pytest.raises(ValueError, match = 'per_token_updates'):
        NeuralMemory(dim = 16, chunk_size = 8, omega_context = 8, per_token_updates = False)
    assert NeuralMemory(dim = 16, chunk_size = 8).per_token_updates is False
    assert NeuralMemory(dim = 16, chunk_size = 8, omega_context = 4).per_token_updates is True


def test_omega_weight_residual_asserts():
    """omega + accept_weight_residual must refuse to construct: prev_weights are
    sliced per chunk, giving chunks different base weights, and the omega window
    would mix gradients taken at different base points (the paper's chunked form
    evaluates all window gradients at the same chunk-start state)."""
    with pytest.raises(AssertionError):
        NeuralMemory(dim = 16, chunk_size = 8, omega_context = 8, accept_weight_residual = True)

def test_omega_context_exceeding_batch_size_warns():
    """omega_context > neural memory batch_size is valid (windows truncate at
    segment boundaries) but wasteful — taps beyond the segment length can never
    fire and their to_context_gates parameters are dead. Construction should
    warn, not fail."""
    with pytest.warns(UserWarning, match = 'omega_context'):
        NeuralMemory(dim = 16, chunk_size = 8, batch_size = 8, omega_context = 16)

def test_mac_without_axial_pos_emb():
    """The absolute axial positional embedding is optional (off in the experiment
    config, 2026-09-02): it feeds raw integer segment indices into an MLP, so any
    eval beyond the training length ran on out-of-distribution positions. With
    the flag off, forward (loss) and cached sampling must both run, and no
    axial params may remain in the state dict. The default (on) is kept for
    library users — asserted as the liveness control."""
    def build(use_axial_pos_emb):
        return MemoryAsContextTransformer(
            num_tokens = 256,
            dim = 16,
            depth = 2,
            segment_len = 32,
            num_persist_mem_tokens = 4,
            num_longterm_mem_tokens = 4,
            neural_mem_gate_attn_output = False,
            use_axial_pos_emb = use_axial_pos_emb,
        )

    model = build(use_axial_pos_emb = False)
    assert model.axial_pos_emb is None
    assert not any('axial_pos_emb' in name for name in model.state_dict()), 'axial params must not exist when disabled'

    ids = torch.randint(0, 256, (1, 129))
    loss = model(ids, return_loss = True)
    assert loss.isfinite()
    loss.backward()

    sampled = model.sample(ids[:, :16], 16 + 40, use_cache = True, temperature = 0., show_progress = False)
    assert sampled.shape == (1, 40)

    # liveness: the default still builds the embedding
    default_model = build(use_axial_pos_emb = True)
    assert default_model.axial_pos_emb is not None
    assert any('axial_pos_emb' in name for name in default_model.state_dict())

def test_mac_return_hidden_matches_logits():
    """return_hidden hands back the final-normed hidden states; projecting them
    with to_logits must reproduce the logits exactly. The BABILong scorer uses
    this to project only the candidate rows instead of the full [L, vocab]
    logits tensor (its memory ceiling at long contexts)."""
    torch.manual_seed(0)
    model = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 16,
        depth = 2,
        segment_len = 32,
        num_persist_mem_tokens = 4,
        num_longterm_mem_tokens = 4,
        neural_mem_gate_attn_output = False,
        use_axial_pos_emb = False,
    )
    model.eval()
    ids = torch.randint(0, 256, (1, 70))
    with torch.no_grad():
        logits = model(ids)
        hidden = model(ids, return_hidden = True)
        assert hidden.shape == (1, 70, 16)
        assert torch.allclose(model.to_logits(hidden), logits, atol = 1e-6)
        # projecting a slice equals slicing the projection
        assert torch.allclose(model.to_logits(hidden[:, 10:13]), logits[:, 10:13], atol = 1e-6)
    with pytest.raises(ValueError):
        model(ids, return_hidden = True, return_loss = True)

def test_per_token_retrieve_tail_reads_last_complete_chunk_state():
    """Modeling quirk pinned for the eval harness (found 2026-09-02): with the
    omega rule + per-token retrieve, positions inside an INCOMPLETE final store
    chunk read the state after the last complete chunk (the remainder is
    cached, never stored in a whole-sequence forward), so the logits at a fixed
    position depend on the input length modulo store_chunk_size — appending
    one token that completes the chunk changes what the previous positions
    retrieve. A pure transformer (memory-free trunk) has no such dependence.
    Consequence for the BABILong scorer: 'the last prompt row depends only on
    the prompt' is FALSE here; scoring is consistent across candidates only
    because every candidate is scored from the same forward (or the same
    length). Never 'optimize' the scorer on the causal-invariance assumption."""

    def build(mem_layers):
        torch.manual_seed(0)
        mem_kwargs = NeuralMemory.atlas_config()
        mem_kwargs.update(dim_head = 8, heads = 4, use_sequential_scan = True)
        return MemoryAsContextTransformer(
            num_tokens = 256,
            dim = 32,
            depth = 2,
            segment_len = 64,
            num_persist_mem_tokens = 4,
            num_longterm_mem_tokens = 4,
            neural_memory_layers = mem_layers,
            neural_memory_segment_len = 8,
            neural_memory_batch_size = 1024,
            use_flex_attn = False,
            sliding_window_attn = True,
            neural_memory_kwargs = mem_kwargs,
            use_axial_pos_emb = False,
        ).eval()

    torch.manual_seed(1)
    ids = torch.randint(0, 256, (1, 60))

    def last_row_dev(model, length):
        with torch.no_grad():
            a = model(ids[:, :length])[0, length - 1]
            b = model(ids[:, :length + 1])[0, length - 1]
        return (a - b).abs().max().item()

    vanilla = build(())
    atlas = build((1,))

    # control: the memory-free trunk is strictly causal at every length
    assert all(last_row_dev(vanilla, length) < 1e-6 for length in range(40, 56))

    # atlas: dependence appears exactly when the appended token completes a
    # store chunk (length % 8 == 7), and nowhere else
    deviations = {length: last_row_dev(atlas, length) for length in range(40, 56)}
    completing = [length for length in deviations if length % 8 == 7]
    others = [length for length in deviations if length % 8 != 7]
    assert all(deviations[length] > 1e-3 for length in completing), deviations
    assert all(deviations[length] < 1e-5 for length in others), deviations

# chunked inference

def _chunked_test_mac(
    mem_layers,
    mem_kwargs,
    use_axial_pos_emb = False,
    sliding = True,
    seed = 0,
    neural_memory_segment_len = 8,
    neural_memory_batch_size = 64,
    neural_mem_weight_residual = False,
):
    torch.manual_seed(seed)
    return MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 32,
        depth = 3,
        segment_len = 16,
        num_persist_mem_tokens = 4,
        num_longterm_mem_tokens = 4,
        neural_memory_layers = mem_layers,
        neural_memory_segment_len = neural_memory_segment_len,
        neural_memory_batch_size = neural_memory_batch_size,
        neural_mem_weight_residual = neural_mem_weight_residual,
        use_flex_attn = False,
        sliding_window_attn = sliding,
        neural_memory_kwargs = mem_kwargs,
        use_axial_pos_emb = use_axial_pos_emb,
    ).eval()


def _atlas_mem_kwargs(**overrides):
    kwargs = NeuralMemory.atlas_config()
    kwargs.update(dim_head = 8, heads = 4, use_sequential_scan = True)
    kwargs.update(overrides)
    return kwargs


def _titans_mem_kwargs():
    return dict(dim_head = 8, heads = 4, momentum = True, qk_rmsnorm = True, attn_pool_chunks = True)


CHUNKED_CASES = {
    'atlas per-token retrieve': dict(mem_layers = (1, 3), mem_kwargs = _atlas_mem_kwargs()),
    'titans (omega=1, per-chunk)': dict(mem_layers = (1, 3), mem_kwargs = _titans_mem_kwargs()),
    'memory-free trunk': dict(mem_layers = (), mem_kwargs = dict()),
    'atlas + axial pos emb ON': dict(mem_layers = (1, 3), mem_kwargs = _atlas_mem_kwargs(), use_axial_pos_emb = True),
    'atlas, block attention (sliding off)': dict(mem_layers = (1, 3), mem_kwargs = _atlas_mem_kwargs(), sliding = False),
    'atlas no-omega (c=1, per-token path)': dict(mem_layers = (1, 3), mem_kwargs = _atlas_mem_kwargs(omega_context = 1)),
}


@pytest.mark.parametrize('chunk_len', (64, 128))
@pytest.mark.parametrize('case', list(CHUNKED_CASES))
def test_chunked_forward_matches_parallel(case, chunk_len):
    """iter_chunked_hidden / forward_chunked must reproduce the whole-sequence
    forward exactly: same longterm-mem interleave, same axial embedding at
    global interleaved positions (when enabled), same memory segmentation
    (chunk boundaries coincide with neural_memory_batch_size boundaries on
    the interleaved axis — the parallel forward's own segments), and the
    attention's exact segment fold — sliding (previous window carried as the
    K/V cache) and block (no previous window; only the head of the segment a
    chunk starts in) — on global interleaved positions. 200 tokens -> 248
    interleaved positions, so the last chunk is partial for both chunk
    lengths, and chunk starts (64, 128, 192) sit mid-segment (segment 20).
    Chunk starts ON a segment boundary — the only place a sliding cache one
    row short of `need` is visible — are covered by
    test_chunked_forward_matches_parallel_at_window_boundary_starts."""
    model = _chunked_test_mac(**CHUNKED_CASES[case])
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (2, 200))

    # fp32: the two paths accumulate in a different order and the memory
    # (Newton-Schulz, decay scans) and the axial SiLU MLP amplify rounding to
    # ~1e-3 on hidden states (measured 8e-5 titans, 1e-3 atlas, 3e-3 axial);
    # a real chunking bug (the pre-fix query-conv context loss) was 3.4.
    with torch.no_grad():
        reference = model(ids, return_hidden = True)
        chunked = model.forward_chunked(ids, chunk_len = chunk_len, return_hidden = True)
    max_dev_fp32 = (reference - chunked).abs().max().item()
    assert max_dev_fp32 < 1e-2, f'{case} chunk {chunk_len}: fp32 hidden max |dev| {max_dev_fp32:.3e}'

    # fp64: the semantic guarantee — the chunked forward IS the whole-sequence
    # forward (measured exactly 0.0 for every case and chunk length). Built
    # fresh (same seed -> same init): a model that already ran in fp32 shows
    # ~7e-7 rounding after .double() — the rotary embedding fills its
    # non-persistent `cached_freqs` buffer on the first call, and .double()
    # widens the buffer without recomputing the fp32-rounded values.
    #
    # Exact 0.0 is GEOMETRY-DEPENDENT, not a law: with num_residual_streams=1
    # the adversarial review (2026-09-02) measured 1.6e-4 fp64 drift at 900
    # tokens that is purely numerical — batched-kernel rounding differs at the
    # ULP level between 129 and 65 weight sets in retrieve_memories (4.4e-16),
    # and the deviation grows exponentially with position at random init (a
    # 1e-15 embedding perturbation reproduces the slope). At streams=4 the
    # hyper-connection residual absorbs ULP-level differences, which is why
    # these cases sit at 0.0. A real semantic bug shows deviations of 3+
    # (mutation evidence: dropping the query-conv cache gives 3.4). Keep the
    # tight tolerance here — on this geometry it is the right instrument —
    # but a new geometry that drifts at 1e-4 is a numerics observation, not
    # a chunking defect, until the deviation is orders of magnitude larger.
    model = _chunked_test_mac(**CHUNKED_CASES[case]).double()
    with torch.no_grad():
        reference = model(ids, return_hidden = True)
        chunked = model.forward_chunked(ids, chunk_len = chunk_len, return_hidden = True)
        reference_logits = model(ids)
        chunked_logits = model.forward_chunked(ids, chunk_len = chunk_len)
    max_dev = (reference - chunked).abs().max().item()
    assert torch.allclose(reference, chunked, atol = 1e-10, rtol = 0), f'{case} chunk {chunk_len}: fp64 hidden max |dev| {max_dev:.3e}'
    assert torch.allclose(reference_logits, chunked_logits, atol = 1e-10, rtol = 0)

    # the generator tiles exactly the original token axis, in order

    starts, lengths = [], []
    with torch.no_grad():
        for start, hidden in model.iter_chunked_hidden(ids, chunk_len = chunk_len):
            starts.append(start)
            lengths.append(hidden.shape[1])

    assert starts[0] == 0
    assert all(s == sum(lengths[:i]) for i, s in enumerate(starts))
    assert sum(lengths) == 200


def test_chunked_forward_rejects_misaligned_chunk_lengths():
    """Chunk boundaries must sit on the memory's segment boundaries (multiples
    of neural_memory_batch_size on the interleaved axis) and on store-chunk
    multiples; without a batch size the parallel forward is one segment and
    no chunk boundary can match it."""
    model = _chunked_test_mac(**CHUNKED_CASES['atlas per-token retrieve'])
    ids = torch.randint(0, 256, (1, 100))

    with pytest.raises(ValueError):
        model.forward_chunked(ids, chunk_len = 48)   # multiple of 8, not of 64

    with pytest.raises(ValueError):
        model.forward_chunked(ids, chunk_len = 68)   # not a multiple of the store chunk

    torch.manual_seed(0)
    no_batch = MemoryAsContextTransformer(
        num_tokens = 256, dim = 32, depth = 2, segment_len = 16,
        num_persist_mem_tokens = 4, num_longterm_mem_tokens = 4,
        neural_memory_layers = (1,), neural_memory_segment_len = 8,
        neural_memory_batch_size = None, use_flex_attn = False,
        sliding_window_attn = True, neural_memory_kwargs = _atlas_mem_kwargs(),
        use_axial_pos_emb = False,
    ).eval()

    with pytest.raises(ValueError):
        no_batch.forward_chunked(ids, chunk_len = 64)

    # attention-only model: no memory segmentation to align with, so any
    # positive chunk length is accepted and still reproduces the forward
    # (chunk 50 starts mid-segment: 50 = 2 x 20 + 10)
    trunk = _chunked_test_mac(**CHUNKED_CASES['memory-free trunk']).double()
    trunk.chunked_inference_alignment(50)
    with torch.no_grad():
        reference = trunk(ids, return_hidden = True)
        chunked = trunk.forward_chunked(ids, chunk_len = 50, return_hidden = True)
    assert torch.allclose(reference, chunked, atol = 1e-10, rtol = 0)

    with pytest.raises(ValueError):
        trunk.forward_chunked(ids, chunk_len = 0)


def test_chunked_forward_carried_kv_cache_is_bounded():
    """The keys / values handed to the attention with each chunk (`prev_kv`)
    reach back exactly to the previous segment boundary — the head of the
    segment the chunk starts in plus the previous window — so they never
    exceed 2 x attn_window_size - 1 positions: the O(1) attention state that
    makes chunked inference O(chunk). Liveness: chunk 64 starts mid-segment
    (64 = 3 x 20 + 4), so some call must carry MORE than one window."""
    model = _chunked_test_mac(**CHUNKED_CASES['atlas per-token retrieve'])
    ids = torch.randint(0, 256, (1, 300))
    seen = []
    original = SegmentedAttention.forward

    def spy(self, seq, *args, prev_kv = None, **kwargs):
        if prev_kv is not None:
            seen.append(prev_kv[0].shape[-2])
        return original(self, seq, *args, prev_kv = prev_kv, **kwargs)

    SegmentedAttention.forward = spy
    try:
        with torch.no_grad():
            model.forward_chunked(ids, chunk_len = 64)
    finally:
        SegmentedAttention.forward = original

    window = model.attn_window_size
    num_positions = model.seq_len_with_longterm_mem(300)
    assert seen, 'no chunk after the first carried a cache'
    assert max(seen) <= 2 * window - 1, seen
    assert max(seen) > window, seen
    # per chunk start: the head of its segment plus the previous window
    # (starts 64, 128, 192, 256, 320 -> heads 4, 8, 12, 16, 0 at window 20)
    expected = {start % window + window for start in range(64, num_positions, 64)}
    assert set(seen) == expected, (sorted(set(seen)), sorted(expected))


def test_chunked_forward_peak_memory_is_bounded_by_chunk():
    """The point of the mode: at 2048 tokens the chunked forward's live
    storage high-water mark is a fraction of the whole-sequence forward's
    (measured with the allocator-independent tracker; chunk = one memory
    segment)."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_babilong_scorer import LiveStorageTracker

    model = _chunked_test_mac(**CHUNKED_CASES['atlas per-token retrieve'])
    ids = torch.randint(0, 256, (1, 2048))
    exclude = list(model.parameters()) + list(model.buffers()) + [ids]

    tracker = LiveStorageTracker(exclude_tensors = exclude)
    with torch.no_grad(), tracker:
        model(ids, return_hidden = True)
    whole = tracker.peak

    tracker = LiveStorageTracker(exclude_tensors = exclude)
    with torch.no_grad(), tracker:
        for _ in model.iter_chunked_hidden(ids, chunk_len = 64):
            pass
    chunked = tracker.peak

    assert chunked < 0.4 * whole, f'chunked peak {chunked / 1e6:.1f} MB vs whole {whole / 1e6:.1f} MB'


def test_chunked_forward_attention_memory_is_linear_in_chunk():
    """The chunked attention folds [cache ∥ chunk] into segments like the
    whole-sequence forward, so its live storage is linear in the chunk
    length. Measured on the memory-free trunk (nothing but the attention /
    residual path) at chunks 1024 / 2048 / 4096 over 8192 tokens: doubling
    the chunk must at most ~double the peak. The previous dense
    chunk x (cache + chunk) mask and scores scaled quadratically (x14 from
    1024 to 4096 at this geometry against ~x4 here)."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_babilong_scorer import LiveStorageTracker

    model = _chunked_test_mac(**CHUNKED_CASES['memory-free trunk'])
    ids = torch.randint(0, 256, (1, 8192))
    exclude = list(model.parameters()) + list(model.buffers()) + [ids]

    peaks = {}
    for chunk_len in (1024, 2048, 4096):
        tracker = LiveStorageTracker(exclude_tensors = exclude)
        with torch.no_grad(), tracker:
            for _ in model.iter_chunked_hidden(ids, chunk_len = chunk_len):
                pass
        peaks[chunk_len] = tracker.peak

    assert peaks[2048] <= 2.3 * peaks[1024], peaks
    assert peaks[4096] <= 2.3 * peaks[2048], peaks
    # liveness: the peak does scale with the chunk (the tracker sees the chunk-sized tensors)
    assert peaks[4096] >= 2.5 * peaks[1024], peaks


def test_neural_mem_chaining_atlas_config_carries_query_conv_context():
    """Chained calls of the memory at a segment boundary must equal one call
    over the whole sequence for the ATLAS config too. The store-side key /
    value convs run per memory segment on both paths, but the query conv
    (retrieve side) spans the whole call — before the retrieve_conv_cache
    field the first short_conv_size - 1 positions of every continued call
    were convolved over zero padding instead of the previous call's rows
    (measured deviation ~1.0 at exactly those 3 positions, 0 elsewhere).
    fp64 so the check is exactness, not tolerance."""
    torch.manual_seed(0)
    seq = torch.randn(2, 128, 32, dtype = torch.float64)
    cfg = NeuralMemory.atlas_config()
    torch.manual_seed(1)
    mem = NeuralMemory(dim = 32, chunk_size = 8, dim_head = 8, heads = 4, batch_size = 64,
                       use_sequential_scan = True, **cfg).double().eval()
    assert exists(mem.query_conv) and mem.query_conv.pad == 3

    with torch.no_grad():
        whole, _ = mem(seq)
        first, state = mem(seq[:, :64])
        second, next_state = mem(seq[:, 64:], state = state)

    assert state.retrieve_conv_cache.shape == (2, 3, 32)
    assert torch.allclose(whole[:, :64], first, atol = 1e-10, rtol = 0)
    assert torch.allclose(whole[:, 64:], second, atol = 1e-10, rtol = 0)

    # liveness: without the carried context the same call diverges at
    # exactly the first kernel_size - 1 positions
    with torch.no_grad():
        uncarried, _ = mem(seq[:, 64:], state = state._replace(retrieve_conv_cache = None))
    bad = ((whole[:, 64:] - uncarried).abs().amax(dim = (0, 2)) > 1e-6).nonzero().flatten().tolist()
    assert bad == [0, 1, 2]


def test_neural_mem_chaining_kernel_one_query_conv_carries_nothing():
    """short_conv_size = 1: the query conv has no left context (pad 0). The
    cache must stay None — `rows[:, -0:]` is the WHOLE tensor, so an
    unguarded slice would carry every query row of every call — and chained
    calls must still equal the whole-sequence forward exactly (fp64)."""
    torch.manual_seed(0)
    seq = torch.randn(2, 128, 32, dtype = torch.float64)
    cfg = NeuralMemory.atlas_config()
    cfg.update(short_conv_size = 1)
    torch.manual_seed(1)
    mem = NeuralMemory(dim = 32, chunk_size = 8, dim_head = 8, heads = 4, batch_size = 64,
                       use_sequential_scan = True, **cfg).double().eval()
    assert exists(mem.query_conv) and mem.query_conv.pad == 0

    with torch.no_grad():
        whole, _ = mem(seq)
        first, state = mem(seq[:, :64])
        second, next_state = mem(seq[:, 64:], state = state)

    assert state.retrieve_conv_cache is None
    assert next_state.retrieve_conv_cache is None
    assert torch.allclose(whole[:, :64], first, atol = 1e-10, rtol = 0)
    assert torch.allclose(whole[:, 64:], second, atol = 1e-10, rtol = 0)

    # the conv itself ignores a passed context at kernel 1
    rows = torch.randn(2, 5, mem.query_conv.conv.in_channels, dtype = torch.float64)
    with torch.no_grad():
        assert torch.equal(mem.query_conv(rows, prev = rows[:, :2]), mem.query_conv(rows))

def test_chunked_forward_rejects_gated_transition():
    """gated_transition is refused by chunked_inference_alignment: the
    whole-sequence forward's segment concatenation drops each non-final
    segment's last entry and substitutes the next segment's first entry — the
    GATED lerp(weights, last_update, sigmoid(transition_gate)) state — so the
    last token of every memory segment retrieves with the gated state, while
    a chunk boundary there would retrieve with the un-gated one (adversarial
    review 2026-09-02: 2-3 max deviation from the first boundary onward with
    the guard silent). Replicating the boundary semantics chunkwise is out of
    scope; the guard must refuse instead of silently diverging."""
    kwargs = dict(CHUNKED_CASES['atlas per-token retrieve'])
    kwargs['mem_kwargs'] = {**kwargs['mem_kwargs'], 'gated_transition': True}
    model = _chunked_test_mac(**kwargs)
    ids = torch.randint(0, 256, (1, 100))

    with pytest.raises(ValueError, match = 'gated_transition'):
        model.forward_chunked(ids, chunk_len = 64)

    # liveness: the same geometry without the gate is accepted
    model = _chunked_test_mac(**CHUNKED_CASES['atlas per-token retrieve'])
    with torch.no_grad():
        model.forward_chunked(ids, chunk_len = 64)


def test_chunked_forward_matches_parallel_at_window_boundary_starts(monkeypatch):
    """Sliding attention is fold-invariant whenever at least one full window
    precedes a chunk, and the parametrized parity geometry never starts a
    chunk on an attention segment boundary (64 % 20 = 4, 128 % 20 = 8,
    192 % 20 = 12) — so a carried K/V cache one row short of `need` passes
    every sliding parity case and is caught only by the block cases and the
    shape test (adversarial review 2026-09-02). A chunk that starts ON a
    segment boundary is where the short cache is visible: its first query
    needs the key at exactly distance W. Geometry: W = 16 + 4 = 20, store
    chunk 4, memory batch 20, chunk 40 = 2W (150 tokens -> 186 interleaved
    positions; chunk starts 0/40/80/120/160, all on segment boundaries).
    Parity is exactly 0.0 here, and dropping the oldest cached row deviates
    by ~3 from the first position of the second chunk — the instrument
    liveness the other sliding cases cannot provide."""
    model = _chunked_test_mac(
        mem_layers = (1, 3),
        mem_kwargs = _atlas_mem_kwargs(),
        neural_memory_segment_len = 4,
        neural_memory_batch_size = 20,
    ).double()
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (2, 150))
    chunk_len = 40
    window = model.layers[0][5].total_segment_len
    assert window == 20 and chunk_len % window == 0

    with torch.no_grad():
        reference = model(ids, return_hidden = True)
        chunked = model.forward_chunked(ids, chunk_len = chunk_len, return_hidden = True)
    max_dev = (reference - chunked).abs().max().item()
    assert torch.allclose(reference, chunked, atol = 1e-10, rtol = 0), f'boundary-start fp64 hidden max |dev| {max_dev:.3e}'

    # instrument liveness: a cache one row short must be caught here, from
    # exactly the first token of the second chunk (token 32 is interleaved
    # position 40: two 16-token segments plus 2 x 4 mem tokens)
    original_forward = SegmentedAttention.forward

    def one_row_short(self, seq, *args, prev_kv = None, **kwargs):
        if prev_kv is not None:
            prev_kv = tuple(t[..., 1:, :] for t in prev_kv)
        return original_forward(self, seq, *args, prev_kv = prev_kv, **kwargs)

    monkeypatch.setattr(SegmentedAttention, 'forward', one_row_short)
    with torch.no_grad():
        mutated = model.forward_chunked(ids, chunk_len = chunk_len, return_hidden = True)
    per_token = (mutated - reference).abs().amax(dim = (0, 2))
    deviating = (per_token > 1e-10).nonzero().flatten()
    assert per_token.max().item() > 1e-3, 'the one-row-short cache mutant went undetected'
    assert deviating.numel() > 0 and deviating[0].item() == 32


def test_chunked_forward_rejects_weight_residual():
    """neural_mem_weight_residual adds the previous memory layer's per-chunk
    weight updates to a later layer's store; that slicing was never validated
    across chunk boundaries, so chunked_inference_alignment refuses the config
    instead of trusting it. The guard existed but nothing exercised it
    (adversarial review 2026-09-02). The titans case is used because omega
    refuses the residual at construction."""
    kwargs = CHUNKED_CASES['titans (omega=1, per-chunk)']
    model = _chunked_test_mac(**kwargs, neural_mem_weight_residual = True)
    ids = torch.randint(0, 256, (1, 100))

    with pytest.raises(ValueError, match = 'weight_residual'):
        model.forward_chunked(ids, chunk_len = 64)

    # liveness: the same geometry without the residual is accepted
    model = _chunked_test_mac(**kwargs)
    with torch.no_grad():
        model.forward_chunked(ids, chunk_len = 64)


def test_carried_memory_state_owns_its_storage():
    """The carried state (last_update, last_momentum) must be state-sized
    copies, not views of the full per-token scan outputs: a view pins the
    previous chunk's O(chunk) scan tensors across chunked-inference calls
    (adversarial review 2026-09-02 measured the chunk-256 peak dropping 81 ->
    73 MB once cloned). Asserts each carried tensor's storage is exactly its
    own bytes."""
    mem = NeuralMemory(dim = 16, chunk_size = 8, dim_head = 8, heads = 2, batch_size = 64, **NeuralMemory.atlas_config())
    seq = torch.randn(1, 64, 16)

    with torch.no_grad():
        _, state = mem(seq)

    last_update, last_momentum = state.states
    for name, t in list(last_update.items()) + list(last_momentum.items()):
        assert t.untyped_storage().nbytes() == t.numel() * t.element_size(), (
            f'carried state {name} shares storage with a larger tensor '
            f'({t.untyped_storage().nbytes()} bytes for {t.numel() * t.element_size()} own bytes)'
        )
