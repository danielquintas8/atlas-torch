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
    a store segment share the same segment-start base weights, so this mixing is
    exact, not an approximation. Also asserts store_memories hands
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
    assert mem.memory_model_parameters[0].grad is not None, (
        'learned memory init (W0) must receive outer-loop gradient with '
        'detach_segment_memory=False — if this fails, the store path is starved again'
    )
    to_keys_weight = mem.to_keys.weight if isinstance(mem.to_keys, nn.Linear) else mem.to_keys[0].weight
    assert to_keys_weight.grad is not None

    # characterization: detach=True in this geometry silently freezes W0 —
    # the reason the atlas training config must not re-enable it

    mem_detached = build_and_backward(detach = True)
    assert mem_detached.memory_model_parameters[0].grad is None, (
        'expected detach_segment_memory=True to cut all outer-loop gradient to the '
        'learned memory init in the two-segment geometry — if it now receives '
        'gradient, the detach semantics changed and this guard needs re-derivation'
    )
