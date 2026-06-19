from __future__ import annotations
from typing import Callable

import math
from functools import partial
from itertools import zip_longest, combinations_with_replacement
from collections import namedtuple

import torch
from torch import nn, stack, cat, is_tensor, tensor, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, Parameter, ParameterList, ParameterDict
from torch.func import functional_call, vmap, grad
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

from tensordict import TensorDict

from assoc_scan import AssocScan

from titans_pytorch.memory_models import(
    MemoryMLP,
    ResidualNorm
)

import einx
from einops import einsum, rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

"""
ein notation:
b - batch
h - heads
bh - batch and heads
n - sequence
d - feature dimension
c - intra-chunk
w - num memory network weight parameters
o - momentum orders
u - key / value updates - allowing a token to emit multiple key / values
"""

LinearNoBias = partial(Linear, bias = False)

# neural mem state related

NeuralMemState = namedtuple('NeuralMemState', [
    'seq_index',
    'weights',
    'cache_store_segment',
    'states',
    'updates',
])

def mem_state_detach(
    state: NeuralMemState
):
    assert isinstance(state, NeuralMemState)
    state = tree_map(lambda t: t.detach() if is_tensor(t) else t, tuple(state))
    return NeuralMemState(*state)

# functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def identity(t):
    return t

def xnor(x, y):
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def safe_cat(inputs, dim = -2):
    inputs = tuple(filter(exists, inputs))

    if len(inputs) == 0:
        return None
    elif len(inputs) == 1:
        return inputs[0]

    return cat(inputs, dim = dim)

def is_empty_tensor(t):
    return t.numel() == 0

def dict_get_value_shapes(td):
    return [v.shape for k, v in td.items()]

def rearrange_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: rearrange(t, pattern, **kwargs))

def repeat_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: repeat(t, pattern, **kwargs))

def pair(v):
    return (v, v) if not isinstance(v, tuple) else v

def round_down_multiple(seq, mult):
    return seq // mult * mult

def round_up_multiple(seq, mult):
    return math.ceil(seq / mult) * mult

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def sequential_scan(gates, inputs, prev = None, remove_prev = True):
    """Sequential associative scan: state[t] = gates[t] * state[t-1] + inputs[t].

    Drop-in replacement for AssocScan that uses O(1) forward memory instead of
    O(n log n). Slower (sequential vs parallel) but allows standard autograd
    backward without TorchScript intermediate retention.

    Gates shape (b, n, ...) is broadcast to match inputs shape (b, n, *weight_shape).
    """

    seq_len = inputs.shape[1]
    if seq_len == 0:
        if not remove_prev and exists(prev):
            return prev.unsqueeze(1)
        return inputs[:, :0]
    state = prev if exists(prev) else torch.zeros_like(inputs[:, 0])
    outputs = []

    if not remove_prev:
        outputs.append(state)

    # broadcast gates to match input weight dimensions
    gate_expand = gates.ndim < inputs.ndim
    extra_dims = inputs.ndim - gates.ndim

    for i in range(seq_len):
        g = gates[:, i]
        if gate_expand:
            g = g.reshape(g.shape + (1,) * extra_dims)
        state = g * state + inputs[:, i]
        outputs.append(state)

    return stack(outputs, dim = 1)

def pack_one_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse

def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    if len(modules) == 1:
        return modules[0]

    return nn.Sequential(*modules)

# softclamping gradients

def softclamp_max(t, max_value):
    half_max_value = max_value / 2
    return ((t / half_max_value).tanh() * half_max_value) + half_max_value

def softclamp_grad_norm(t, max_value):
    if is_empty_tensor(t):
        return t

    t, inverse = pack_one_with_inverse(t, 'bn *')

    norm = t.norm(dim = -1, keepdim = True)
    clamped_norm = softclamp_max(norm, max_value)

    t = t * (clamped_norm / norm)
    return inverse(t)

# spectral norming the surprise update w/ newton schulz matrix iter
# Keller Jordan et al. from OSS w/ nanogpt, now being used for two works, Atlas and 'TTT done right'

def newtonschulz5(
    t,
    steps = 5,
    eps = 1e-7,
    coefs = (3.4445, -4.7750, 2.0315)
):
    if t.ndim <= 3:
        return t

    shape = t.shape
    should_transpose = shape[-2] > shape[-1]

    if should_transpose:
        t = t.transpose(-1, -2)

    t, inv_pack = pack_one_with_inverse(t, '* i j')
    t = t / t.norm(dim = (-1, -2), keepdim = True).clamp(min = eps)

    a, b, c = coefs

    for _ in range(steps):
        A = t @ t.transpose(-1, -2)
        B = b * A + c * A @ A
        t = a * t + B @ t

    if should_transpose:
        t = t.transpose(-1, -2)

    return inv_pack(t)


def apply_omega_window(
    grads: TensorDict,
    context_gates: Tensor,
    omega_context: int
) -> TensorDict:
    """Apply the Atlas Omega Rule's gamma-weighted sliding window to per-token grads.

    For each position i the windowed gradient is a causal, gamma-weighted sum over
    the last `omega_context` (= c) per-token gradients (paper Section 3.2, Eq 9):

        G_i = Σ_{k=0}^{c-1} γ_k^(i) · grad[i - c + 1 + k]

    where the γ gates are input-dependent (c learned weights per position).

    Args:
        grads: per-token gradients; each value has shape (bhn, chunk_size, *weight_shape).
        context_gates: gamma gates, shape (bhn, chunk_size, omega_context).
        omega_context: sliding window size c.

    Returns:
        TensorDict of windowed gradients, with the same keys and shapes as `grads`.
    """
    windowed_grads = TensorDict()

    for name, g in grads.items():
        # g: (bhn, chunk_size, *weight_shape)
        windowed = torch.zeros_like(g)

        for k in range(omega_context):
            offset = omega_context - 1 - k  # k=0 → oldest (largest shift), k=c-1 → newest (no shift)
            gamma_k = context_gates[..., k]  # (bhn, chunk_size)

            if offset == 0:
                shifted = g
            else:
                # shift right by `offset` along the chunk (time) axis, zero-pad the start
                shifted = F.pad(
                    g[:, offset:],
                    (0,) * (2 * (g.ndim - 2)) + (offset, 0)
                )

            # broadcast gamma over all trailing weight dimensions
            gamma_expanded = gamma_k
            for _ in range(g.ndim - 2):
                gamma_expanded = gamma_expanded.unsqueeze(-1)

            windowed = windowed + shifted * gamma_expanded

        windowed_grads[name] = windowed

    return windowed_grads


# short causal depthwise convolution on keys/queries (paper p.13)
# provides local mixing before memory read/write

class CausalDepthwiseConv1d(Module):
    def __init__(self, channels, kernel_size = 4):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups = channels, bias = False)

    def forward(self, x):
        if x.shape[1] == 0:
            return x
        x = x.transpose(1, 2)
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        return x.transpose(1, 2)

# polynomial feature mapping for Atlas (Section 3.1)
# increases memory capacity from O(d_k) to O(d_k^p) by expanding keys/queries
# with interaction terms before they enter the memory MLP

class PolynomialFeatures(Module):
    """Taylor approximation of softmax (Atlas Section 3.1).

    φ_p(x) ≈ Σ_{d=0..p} a_d · x^⊗d, where the coefficients are initialized to
    1/d! to match exp(⟨q,k⟩) = Σ ⟨q,k⟩^d/d!. The d=0 (constant) term is the
    leading-order kernel piece and must be included for the Taylor-softmax
    motivation to hold.

    For each degree d ≥ 1 the feature is the multiset of distinct monomials
    x_{i_1} … x_{i_d} (combinations_with_replacement) — symmetric monomials,
    not all d-tuples, since e.g. x_i x_j == x_j x_i.
    """

    def __init__(
        self,
        dim,
        degree = 2,
        project_back = True,
    ):
        super().__init__()
        assert degree >= 2, 'polynomial degree must be at least 2'

        self.dim = dim
        self.degree = degree

        # degree-0 constant feature contributes 1 dimension (the leading 1 in
        # the Taylor expansion), independent of `dim`.
        expanded_dim = 1
        index_groups = [1]

        for d in range(1, degree + 1):
            combos = list(combinations_with_replacement(range(dim), d))
            indices = torch.tensor(combos, dtype = torch.long).T
            self.register_buffer(f'indices_{d}', indices, persistent = False)
            index_groups.append(len(combos))
            expanded_dim += len(combos)

        self.expanded_dim = expanded_dim
        self.index_group_sizes = index_groups

        if expanded_dim > 100_000:
            import warnings
            warnings.warn(
                f'PolynomialFeatures: expanded_dim={expanded_dim} with dim={dim}, degree={degree}. '
                f'This will create large intermediate tensors and a projection with {expanded_dim * dim} parameters. '
                f'Consider reducing degree or dim_head.'
            )

        # Learnable Taylor coefficients init at 1/d! for d in 0..degree.
        coefficients = [1.0 / math.factorial(d) for d in range(0, degree + 1)]
        self.coefficients = Parameter(torch.tensor(coefficients))

        # project_back=True: compresses the full O(d^p)-dim φ(k) back to `dim` via a learned
        #   linear, keeping the memory MLP architecture unchanged. PAPER-DEVIATING — collapses
        #   Proposition 2's capacity argument since the MLP only ever sees a `dim`-rank
        #   compression of φ(k) instead of φ(k) itself.
        # project_back=False: feeds the full expanded_dim directly into the memory MLP. Paper-
        #   faithful per Eq 56-57 (M(φ(k))). Requires the surrounding NeuralMemory to construct
        #   a memory_model with input dim = self.expanded_dim.

        self.projection = LinearNoBias(self.expanded_dim, dim) if project_back else None

    @property
    def output_dim(self):
        return self.dim if exists(self.projection) else self.expanded_dim

    def forward(self, x):
        # degree-0 constant: shape (..., 1) — broadcast a learnable scalar.
        const_feature = self.coefficients[0].expand(*x.shape[:-1], 1)
        features = [const_feature]

        for d in range(1, self.degree + 1):
            indices = getattr(self, f'indices_{d}')
            mono = x[..., indices[0]]
            for i in range(1, d):
                mono = mono * x[..., indices[i]]
            features.append(self.coefficients[d] * mono)

        out = cat(features, dim = -1)

        if exists(self.projection):
            out = self.projection(out)

        return out

# multi head rmsnorm

class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.rmsnorm = nn.RMSNorm(dim, elementwise_affine = False)
        self.gamma = Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.)

# chunk pooling

class AveragePool(Module):
    def __init__(
        self,
        chunk_size
    ):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(
        self,
        x,
        chunk_size = None
    ):
        chunk_size = default(chunk_size, self.chunk_size)
        return reduce(x, 'b (n c) d -> b n d', 'mean', c = chunk_size)

class AttentionPool(Module):
    def __init__(
        self,
        dim,
        chunk_size
    ):
        """
        taken from Enformer https://www.nature.com/articles/s41592-021-01252-x , in turn taken from somewhere else
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.to_attn_logits = nn.Linear(dim, dim)

        # default to average pool

        nn.init.zeros_(self.to_attn_logits.weight)
        nn.init.zeros_(self.to_attn_logits.bias)

    def forward(
        self,
        x,
        chunk_size = None
    ):
        chunk_size = default(chunk_size, self.chunk_size)

        x = rearrange(x, 'b (n c) d -> b n c d', c = chunk_size)

        attn_logits = self.to_attn_logits(x)

        attn = attn_logits.softmax(dim = -2)

        return reduce(x * attn, 'b n c d -> b n d', 'sum')

# main neural memory

def default_adaptive_step_transform(adaptive_step, max_lr = 1e-2):
    return adaptive_step.sigmoid() * max_lr

def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim = -1)

class NeuralMemory(Module):

    @classmethod
    def atlas_config(cls, **overrides):
        """Returns kwargs for Atlas-style configuration.
        Momentum and Muon confirmed in paper Table 1, Eq. 57-58.
        omega_context=8 based on Figure 5 (best for OmegaNet).
        polynomial_degree=2 — paper does not specify exact degree.
        poly_project_back=True is the documented production tradeoff. Strict
        Eq 56-57 reading is `M(φ(k))` with the MLP consuming `expanded_dim`
        directly (project_back=False), but Phase 0 OOM evidence (job
        40049757) showed the omega-windowed gradient accumulation saturates
        a single H100 at 64 GB on the 170M / seq_len=1024 / chunk_size=8
        config — the asymmetric MLP path is not viable at this scale without
        FSDP/model parallelism. With project_back=True the polynomial
        features still feed φ(k) → learned linear → dim_head into the MLP,
        retaining the Taylor-init learnable coefficients but capping
        effective capacity at O(dim_hidden). Treat the project_back=False
        path as a Phase 3+ scaling question (see GitHub issue #17).
        per_token_retrieve=True is paper-faithful: Eq 41 / Section 3.3 /
        Appendix D.4 specify y_t = M_t(q_t), retrieval at every token using
        the per-token memory state. Disabling this falls back to a per-chunk
        retrieve approximation that is structurally Titans-grade, not Atlas.
        """
        defaults = dict(
            momentum = True,
            spectral_norm_surprises = True,
            polynomial_degree = 2,
            poly_project_back = True,
            omega_context = 8,
            per_token_retrieve = True,
            short_conv_size = 4,
            qk_rmsnorm = True,
        )
        defaults.update(overrides)
        return defaults
    def __init__(
        self,
        dim,
        chunk_size: int | tuple[int, int] = 1,
        batch_size = None,
        dim_head = None,
        heads = 1,
        model: Module | None = None,
        store_memory_loss_fn: Callable = default_loss_fn,
        adaptive_step_transform: Callable | None = None,
        default_step_transform_max_lr = 1.,
        per_parameter_lr_modulation = False, # allow outer network to control learning rate per weight matrix of memory network
        max_mem_layer_modulation = 1., # max of 10.
        per_head_learned_parameters = True,
        attn_pool_chunks = False,
        momentum = True,
        momentum_order = 1,
        learned_momentum_combine = False,
        learned_combine_include_zeroth = False,
        num_kv_per_token = 1, # whether a single token can do multiple updates to the memory model
        qkv_receives_diff_views = False, # to address an issue raised by a phd student (who will be credited if experiments are green). basically the issue raised is that the memory MLP is only learning Wk @ Wv linear mapping and that may not be expressive enough. we will use hyper connections to allow the network to choose different previous layer inputs as keys / values and see if that does anything
        pre_rmsnorm = True,
        post_rmsnorm = False,
        qk_rmsnorm = False,
        max_grad_norm: float | None = None,
        use_accelerated_scan = False,
        activation: Module | None = None,
        init_adaptive_step_bias = None,
        init_momentum_bias = None,
        init_decay_bias = None,
        accept_weight_residual = False,
        spectral_norm_surprises = False,
        muon_ns_steps = 5,
        muon_ns_eps = 1e-7,
        polynomial_degree: int | None = None,
        poly_project_back = True,
        omega_context: int = 1,  # sliding window size c (paper Section 3.2). 1 = Titans. must be <= store_chunk_size.
        per_token_retrieve: bool = False,  # per-token retrieve: each y_t = M_t(q_t) using per-token weights from Eq 41. requires ~chunk_size× more memory. default False uses per-chunk approximation. enable when hardware allows (multi-GPU or smaller model).
        short_conv_size: int = 0,  # causal depthwise conv on keys/queries (paper p.13, kernel size). 0 = disabled.
        detach_segment_memory: bool = False,  # detach intermediate segment updates to reduce autograd memory from O(segments) to O(1). outer-loop gradients for store-side params come only from last segment. enable when GPU memory is constrained.
        use_sequential_scan: bool = False,  # use sequential scan for momentum/decay instead of parallel associative scan. O(1) forward memory vs O(n log n). slower but drastically reduces GPU memory for large sequences.
        gated_transition = False,
        mem_model_norm_add_residual = True,  # by default, layernorm output and add residual as proposed in TTT paper, but could be removed
        store_with_lookahead_value = False,  # Tianyu Zhao and Llion Jones - https://arxiv.org/abs/2601.00671 - they use the values from the next timestep for the gradients for storing, showing much better performance
        default_model_kwargs: dict = dict(
            depth = 2,
            expansion_factor = 4.
        )
    ):
        super().__init__()
        dim_head = default(dim_head, dim)
        assert not (heads == 1 and dim_head != dim)

        self.retrieve_chunk_size, self.store_chunk_size = pair(chunk_size)

        # omega rule incompatibilities
        if omega_context > 1:
            assert num_kv_per_token == 1, 'omega rule requires num_kv_per_token == 1 (context gates assume chunk_size tokens, not chunk_size * num_kv_per_token)'
            assert not store_with_lookahead_value, 'omega rule is incompatible with store_with_lookahead_value (lookahead trims keys to chunk_size - 1, mismatching context gate dimensions)'

        # omega rule produces per-token weight updates (Eq 41: M_t = M_{t-1} + S'_t).
        # per-token retrieve (retrieve_chunk_size=1) is correct per the paper — each y_t = M_t(q_t).
        # however, it requires ~chunk_size× more autograd memory (one weight set per token in the
        # computation graph, held for backward across all layers). defaults to per-chunk approximation
        # (subsample at chunk boundaries, same as Titans) which works on single GPU.
        # enable per_token_retrieve=True when hardware supports it (multi-GPU, model parallelism).
        if omega_context > 1 and per_token_retrieve:
            self.retrieve_chunk_size = 1

        # batch size

        if exists(batch_size):
            assert divisible_by(batch_size, self.store_chunk_size)

        self.batch_size = batch_size

        # associative scan

        self.assoc_scan = AssocScan(use_accelerated = use_accelerated_scan)

        # key values receiving different views

        self.qkv_receives_diff_views = qkv_receives_diff_views

        # norms

        self.retrieve_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.store_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        self.multihead_rmsnorm = MultiheadRMSNorm(dim_head, heads) if post_rmsnorm else nn.Identity()

        self.q_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()
        self.k_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()

        # maybe multi-headed

        dim_inner = dim_head * heads

        self.heads = heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.split_kv_heads = Rearrange('b n (h u d) -> b h (n u) d', h = heads, u = num_kv_per_token)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = LinearNoBias(dim_inner, dim) if heads > 1 else nn.Identity()

        self.retrieve_gate = Sequential(
            LinearNoBias(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if heads > 1 else None

        # polynomial feature mapping (Atlas Section 3.1) — constructed early so its
        # expanded_dim can size the default memory model's input dim when
        # poly_project_back=False (paper-faithful path: the MLP consumes φ(k) directly).

        self.poly_features = None

        if exists(polynomial_degree) and polynomial_degree >= 2:
            self.poly_features = PolynomialFeatures(
                dim = dim_head,
                degree = polynomial_degree,
                project_back = poly_project_back,
            )

        # memory model — when poly_features feeds the MLP directly (no project_back),
        # the MLP must accept input dim = poly.expanded_dim and emit dim_head.

        feeds_poly_features_directly = exists(self.poly_features) and not exists(self.poly_features.projection)
        mem_model_input_dim = self.poly_features.expanded_dim if feeds_poly_features_directly else dim_head

        if not exists(model):
            if feeds_poly_features_directly:
                # asymmetric MLP: in=expanded_dim, hidden=dim_head*expansion, out=dim_head
                model = MemoryMLP(dim_head, dim_in = mem_model_input_dim, dim_out = dim_head, **default_model_kwargs)
            else:
                model = MemoryMLP(dim_head, **default_model_kwargs)

        # validate memory model — input shape depends on whether poly_features feeds it.

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        test_input_shape = (3, 2, mem_model_input_dim)
        test_output_shape = (3, 2, dim_head)

        with torch.no_grad():
            try:
                test_input = torch.randn(test_input_shape)
                mem_model_output = model(test_input)
            except:
                raise RuntimeError(f'memory model unable to accept a tensor of shape {test_input_shape}')

            assert mem_model_output.shape == test_output_shape, (
                f'memory model must emit shape {test_output_shape}, got {tuple(mem_model_output.shape)}. '
                f'When poly_features feeds the MLP directly (poly_project_back=False), input dim must be '
                f'{mem_model_input_dim} (poly_features.expanded_dim).'
            )

        # the memory is the weights of the model

        # ResidualNorm (TTT-paper convention) does `norm(model(x)) + x`, which requires
        # input_dim == output_dim. Auto-disable for the asymmetric Atlas path where the
        # MLP consumes φ(k) (expanded_dim) and emits dim_head — the residual makes no
        # sense there and the paper's M(φ(k)) is the direct output anyway.
        if mem_model_norm_add_residual and not feeds_poly_features_directly:
            model = ResidualNorm(dim = dim_head, model = model)

        self.memory_model = model

        mem_model_params = dict(model.named_parameters())

        self.num_memory_parameter_tensors = len(mem_model_params)

        self.memory_model_parameter_names = [*mem_model_params.keys()]

        memory_model_parameters = [*mem_model_params.values()]

        if per_head_learned_parameters:
            # NOTE: .clone() materializes storage so each head owns its slice.
            # Without it, repeat() returns an expanded view; the resulting Parameter
            # shares storage across heads (stride 0 on the head dim). Consequences:
            #   - load_state_dict() fails: "more than one element of the written-to
            #     tensor refers to a single memory location"
            #   - AdamW in-place updates broadcast identically across heads, so all
            #     H heads stay bit-identical across training — a paper-fidelity
            #     regression since Section 3 requires independent per-head weights.
            memory_model_parameters = [repeat(p, '... -> h ...', h = heads).clone() for p in memory_model_parameters]

            # Strip the originals from self.memory_model so they don't appear as
            # orphan params in .parameters() — functional_call will provide the
            # active per-head versions from self.memory_model_parameters at every
            # call site, and the originals are never read after this point.
            # Without this, the originals show up in .parameters() but never
            # receive gradients (gradient flows only through the per-head copies).
            for name in self.memory_model_parameter_names:
                parts = name.split('.')
                parent = self.memory_model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                del parent._parameters[parts[-1]]

        self.init_weight_shape = [p.shape for p in memory_model_parameters]

        self.memory_model_parameters = ParameterList(memory_model_parameters)
        self.per_head_learned_parameters = per_head_learned_parameters

        # the chunk size within the paper where adaptive step, momentum, weight decay are shared

        self.chunk_size = chunk_size

        # prepare function for per sample gradients from model above, using torch.func

        def forward_and_loss(params, inputs, loss_weights, target):
            pred = functional_call(self.memory_model, params, inputs)
            loss = self.store_memory_loss_fn(pred, target) # simple mse loss in paper - eq (12) - |M(k) - v|²
            weighted_loss = loss * loss_weights
            return weighted_loss.sum(), loss

        # two functions

        grad_fn = grad(forward_and_loss, has_aux = True)

        self.per_sample_grad_fn = vmap(grad_fn, in_dims = (0, 0, 0, 0))

        # per-token gradient function for Omega Rule (Atlas Section 3.3)
        # computes one gradient per token within each chunk, all w.r.t. same chunk-start weights

        if omega_context > 1:
            def single_token_forward_and_loss(params, single_input, single_lr, single_target):
                pred = functional_call(self.memory_model, params, single_input.unsqueeze(0))
                loss = self.store_memory_loss_fn(pred, single_target.unsqueeze(0))
                weighted_loss = loss * single_lr
                return weighted_loss.sum(), loss.sum()

            single_grad_fn = grad(single_token_forward_and_loss, has_aux = True)

            # inner vmap: over tokens within a chunk. params shared (in_dims=None) — same M_{t'}
            per_token_in_chunk_grad_fn = vmap(single_grad_fn, in_dims = (None, 0, 0, 0))

            # outer vmap: over chunks. params vary per chunk (in_dims=0)
            self.per_token_grad_fn = vmap(per_token_in_chunk_grad_fn, in_dims = (0, 0, 0, 0))

        # queries for retrieving from the model

        self.to_queries = Sequential(LinearNoBias(dim, dim_inner), activation)

        # keys and values for storing to the model

        assert num_kv_per_token > 0

        self.to_keys = Sequential(
            LinearNoBias(dim, dim_inner * num_kv_per_token),
            activation,
        )

        self.to_values = Sequential(
            LinearNoBias(dim, dim_inner * num_kv_per_token),
            activation,
        )

        # short causal convolution on keys/queries (paper p.13)

        self.key_conv = CausalDepthwiseConv1d(dim_inner * num_kv_per_token, short_conv_size) if short_conv_size > 0 else None
        self.query_conv = CausalDepthwiseConv1d(dim_inner, short_conv_size) if short_conv_size > 0 else None

        self.store_with_lookahead_value = store_with_lookahead_value
        self.detach_segment_memory = detach_segment_memory
        self.use_sequential_scan = use_sequential_scan

        self.store_memory_loss_fn = store_memory_loss_fn

        self.num_kv_per_token = num_kv_per_token

        # `chunk_size` refers to chunk size used for storing to memory model weights

        chunk_size = self.store_chunk_size

        # whether to use averaging of chunks, or attention pooling

        assert not (attn_pool_chunks and chunk_size == 1), '`attn_pool_chunks` cannot be set to True if `chunk_size` is set to 1'

        if not attn_pool_chunks:
            self.reduce_to_chunk_rep = AveragePool(chunk_size = chunk_size)
        else:
            self.reduce_to_chunk_rep = AttentionPool(dim, chunk_size = chunk_size)

        # learned adaptive learning rate

        self.to_adaptive_step = Sequential(
            nn.Linear(dim, heads * num_kv_per_token),
            Rearrange('b n (h u) -> (b h) (n u)', u = num_kv_per_token)
        )

        if not exists(adaptive_step_transform):
            adaptive_step_transform = partial(default_adaptive_step_transform, max_lr = default_step_transform_max_lr)

        self.adaptive_step_transform = adaptive_step_transform

        # momentum related

        self.to_momentum = Sequential(
            nn.Linear(dim, heads * momentum_order),
            Rearrange('b n (h o) -> o (b h) n 1', o = momentum_order)
        ) if momentum else None

        self.momentum_order = momentum_order
        self.to_learned_momentum_combine = None

        if learned_momentum_combine:
            assert momentum
            assert momentum_order > 1, 'only second order momentum allowed for now, but may allow learned combination of zeroth'

            if learned_combine_include_zeroth:
                momentum_order += 1

            self.to_learned_momentum_combine = Sequential(
                nn.Linear(dim, heads * momentum_order),
                Rearrange('b n (h o) -> o (b h) n', h = heads),
                nn.Softmax(dim = 0),
            )

            self.learned_combine_include_zeroth = learned_combine_include_zeroth

        # per layer learning rate modulation

        self.to_layer_modulation = Sequential(
            nn.Linear(dim, heads * self.num_memory_parameter_tensors),
            Rearrange('b n (h w) -> w (b h) n', h = heads),
            nn.Sigmoid()
        ) if per_parameter_lr_modulation else None

        self.max_mem_layer_modulation = max_mem_layer_modulation

        # learned weight residual

        self.to_learned_weight_residual_mix = Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n'),
            nn.Sigmoid()
        ) if accept_weight_residual else None

        # allow for softclamp the gradient norms for storing memories

        self.max_grad_norm = max_grad_norm

        # spectral norming the surprises before update, a la Muon from Jordan et al.

        self.spectral_norm_surprises = spectral_norm_surprises
        self.muon_ns_steps = muon_ns_steps
        self.muon_ns_eps = muon_ns_eps

        # poly_features was constructed earlier (before the memory model) so its
        # expanded_dim can size an asymmetric MLP for the paper-faithful path.

        # omega rule — sliding window context (Atlas Section 3.2)

        self.omega_context = omega_context

        if omega_context > 1:
            assert omega_context <= self.store_chunk_size, \
                f'omega_context ({omega_context}) must be <= chunk_size ({self.store_chunk_size})'

            # NOTE: the paper's M_s mask (Section 3.3) is a banded lower-triangular matrix
            # that enforces which tokens fall within each position's window. In our implementation,
            # this hard boundary is implicitly enforced by zero-padding in the shift-based gamma
            # gate computation — tokens outside the window produce zero-padded shifted gradients,
            # so no explicit M_s buffer is needed.

            # learned per-window-position context gates γ_i^(t) (Section 3.2, page 8)
            # for each position t, produces c gates ∈ [0,1] weighting each token in the window
            # effective weight for token i in position t's window = adaptive_lr_i × γ_k^(t)

            self.to_context_gates = Sequential(
                nn.Linear(dim, heads * omega_context),
                Rearrange('b n (h c) -> (b h) n c', h = heads, c = omega_context),
            )

        # weight decay factor

        self.to_decay_factor = Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> (b h) n 1')
        )

        # learned transition, as seeing instability when decreasing neural mem batch size
        # perhaps it can slowly learn to adjust from early residual to fully transitioning to new weights every batch size

        self.transition_gate = nn.Parameter(tensor(-5.)) if gated_transition else None

        # inits

        if exists(init_adaptive_step_bias):
            linear = self.to_adaptive_step[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_adaptive_step_bias)

        if exists(init_momentum_bias):
            linear = self.to_momentum[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_momentum_bias)

        if exists(init_decay_bias):
            linear = self.to_decay_factor[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_decay_bias)

        # maybe use accelerated scan

        self.use_accelerated_scan = use_accelerated_scan

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def memory_model_parameter_dict(self):
        return TensorDict(dict(zip(self.memory_model_parameter_names, self.memory_model_parameters)))

    def init_weights(
        self,
        batch,
    ):
        if self.per_head_learned_parameters:
            weights = repeat_dict_values(self.memory_model_parameter_dict, 'h ... -> (b h) ...', b = batch)
        else:
            weights = repeat_dict_values(self.memory_model_parameter_dict, '... -> bh ...', bh = batch * self.heads)

        return weights

    def init_momentum(
        self,
        batch,
    ):
        zeros = self.memory_model_parameter_dict.clone().zero_()

        if self.per_head_learned_parameters:
            zeros = repeat_dict_values(zeros, 'h ... -> o (b h) ...', b = batch, o = self.momentum_order)
        else:
            zeros = repeat_dict_values(zeros, '... -> o bh ...', bh = batch * self.heads, o = self.momentum_order)

        return zeros

    def store_memories(
        self,
        seq,
        weights: dict[str, Tensor] | None = None,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        seq_index = 0,
        prev_weights = None,
        mask: Tensor | None = None,
        return_surprises = True
    ):
        if self.qkv_receives_diff_views:
            _, batch, seq_len = seq.shape[:3]
        else:
            batch, seq_len = seq.shape[:2]

        # shapes and variables

        heads, chunk_size, num_updates = self.heads, self.store_chunk_size, self.num_kv_per_token

        # curtail sequence by multiple of the chunk size
        # only a complete chunk of the sequence provides the memory for the next chunk

        round_down_seq_len = round_down_multiple(seq_len, chunk_size)
        num_chunks = round_down_seq_len // chunk_size

        seq, remainder = seq[..., :round_down_seq_len, :], seq[..., round_down_seq_len:, :]

        next_seq_len_index = seq_index + round_down_seq_len

        # init weights if needed
        # weights of the memory network

        if not exists(weights):
            weights = self.init_weights(batch)

        weights = TensorDict(weights)

        # allow for neural memory of a previous layer to influence surprise of current layer

        weights_for_surprise = repeat_dict_values(weights, 'b ... -> b n ...', n = num_chunks)

        # initial norm

        seq = self.store_norm(seq)

        # handle keys and values coming from different sequences from hyper connection

        values_seq = seq

        if self.qkv_receives_diff_views:
            seq, values_seq = seq

        # derive learned hparams for optimization of memory network

        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)

        use_omega = self.omega_context > 1

        # derive per-chunk or per-token representations for momentum/decay/lr params
        # omega rule requires per-token granularity (Section 5.1, Eqs 37-39)

        if use_omega:
            chunked_seq = seq  # per-token: no pooling
        else:
            chunked_seq = self.reduce_to_chunk_rep(seq, chunk_size = chunk_size)

        decay_factor = self.to_decay_factor(chunked_seq).sigmoid()

        need_layer_lr_mod = exists(self.to_layer_modulation) and num_chunks > 0
        has_momentum = exists(self.to_momentum)

        adaptive_momentum = None
        combine_momentums = None

        if has_momentum:
            adaptive_momentum = self.to_momentum(chunked_seq).sigmoid()

            learned_combine = exists(self.to_learned_momentum_combine)

            if learned_combine:
                combine_momentums = self.to_learned_momentum_combine(chunked_seq)

        if need_layer_lr_mod:
            layer_lr_mod = self.to_layer_modulation(chunked_seq) * self.max_mem_layer_modulation

        # keys and values

        keys = self.to_keys(seq)
        values = self.to_values(values_seq)

        # maybe short causal conv

        if exists(self.key_conv):
            keys = self.key_conv(keys)

        # maybe multi head

        keys, values = map(self.split_kv_heads, (keys, values))

        # maybe keys rmsnorm

        keys = self.k_norm(keys)

        # maybe polynomial feature expansion (Atlas Section 3.1)

        if exists(self.poly_features):
            keys = self.poly_features(keys)

        # take care of chunking

        keys, values = tuple(rearrange(t, 'b h (n c u) d -> (b h n) (c u) d', c = chunk_size, u = num_updates) for t in (keys, values))

        # adaptive lr

        adaptive_lr = rearrange(adaptive_lr, 'b (n c u) -> (b n) (c u)', c = chunk_size, u = num_updates)

        # optionally a storing memories mask can be passed in. if False, will set the learning rate to 0. for those positions

        if exists(mask):
            mask = mask[..., :round_down_seq_len]
            mask = repeat(mask, 'b (n c) -> (b h n) (c u)', h = heads, u = num_updates, c = chunk_size)

            adaptive_lr = torch.where(mask, adaptive_lr, 0.)

        # maybe add previous layer weight

        assert xnor(exists(self.to_learned_weight_residual_mix), exists(prev_weights))

        if exists(prev_weights):

            start_index = math.ceil(seq_index / chunk_size)
            end_index = start_index + num_chunks

            prev_weights = prev_weights.apply(lambda t: t[:, start_index:end_index])

            if exists(self.to_learned_weight_residual_mix) and num_chunks > 0:
                # weight residual operates at chunk granularity (prev_weights is per-chunk)
                chunked_seq_for_mix = self.reduce_to_chunk_rep(seq, chunk_size = chunk_size) if use_omega else chunked_seq
                mix = self.to_learned_weight_residual_mix(chunked_seq_for_mix)
                mix = rearrange(mix, 'b h n -> (b h) n')
                prev_weights = prev_weights.apply(lambda t: einx.multiply('bh n, bh n ... -> bh n ...', mix, t))

            weights_for_surprise = weights_for_surprise + prev_weights

        # flatten batch and time if surprise depends on previous layer memory model

        weights_for_surprise = rearrange_dict_values(weights_for_surprise, 'b n ... -> (b n) ...')

        # maybe lookahead values

        if self.store_with_lookahead_value:
            adaptive_lr = adaptive_lr[..., :-1]
            keys = keys[..., :-1, :]
            values = values[..., 1:, :]

        # The adaptive learning rate η is applied OUTSIDE Newton-Schulz for the Atlas
        # (omega + Muon) path, matching paper Table 1: M_t = α M_{t-1} − η_t·NS-5(S_t) with
        # the raw gradient inside S_t. newtonschulz5 normalizes its input by norm and is
        # therefore scale-invariant, so folding η into the surprise (as the grad loss weight)
        # would cancel it. We feed the grad fn the store mask only (raw, masked gradient) and
        # re-apply η per output token after NS-5. Other paths keep η as the loss weight —
        # correct there, since without NS-5 there is nothing to cancel it.
        #
        # NOTE: a non-omega + Muon config would still wash η out, but no shipped config hits
        # it (Muon always travels with omega here; the no-muon ablation turns Muon off, not
        # omega) and a per-chunk η has no clean paper form, so we do not apply η-outside there.
        # Only test_muon_custom_steps exercises that combination, for NS-step plumbing.

        apply_eta_outside = use_omega and self.spectral_norm_surprises
        eta_for_update = None

        if apply_eta_outside:
            store_mask_weight = mask.to(adaptive_lr.dtype) if exists(mask) else torch.ones_like(adaptive_lr)
            eta_for_update = rearrange(adaptive_lr, '(b n) c -> b (n c)', b = batch * heads)

        # get grads and extra auxiliary loss

        if use_omega:
            # omega rule: per-token gradients within each chunk, all w.r.t. same M_{t'}
            # then apply sliding window mask M_s (Section 3.3)

            grads, unweighted_mem_model_loss = self.per_token_grad_fn(dict(weights_for_surprise), keys, store_mask_weight if apply_eta_outside else adaptive_lr, values)

            grads = TensorDict(grads)
            # grads shape: (batch*heads*num_chunks, chunk_size, *weight_shape)

            # gamma gates: c learned weights per position, input-dependent
            omega_c = self.omega_context
            context_gates = self.to_context_gates(seq).sigmoid()  # (bh, num_tokens, c)
            context_gates = rearrange(context_gates, 'bh (n b) c -> (bh n) b c', b = chunk_size)
            # context_gates: (bh*num_chunks, chunk_size, omega_context)

            # apply the gamma-weighted sliding window (Section 3.2, Eq 9)
            grads = apply_omega_window(grads, context_gates, omega_c)

            # reshape from (bh*num_chunks, chunk_size, ...) to (bh, num_tokens, ...)
            # so momentum/decay scan runs per-position (Eqs 37-39)
            grads = rearrange_dict_values(grads, '(b n) c ... -> b (n c) ...', b = batch * heads)

            # surprises
            adaptive_lr = rearrange(adaptive_lr, '(b h n) c -> b h (n c)', b = batch, h = heads)
            unweighted_mem_model_loss = rearrange(unweighted_mem_model_loss, '(b h n) c -> b h (n c)', b = batch, h = heads)

        else:
            # standard Titans path: one gradient per chunk
            grads, unweighted_mem_model_loss = self.per_sample_grad_fn(dict(weights_for_surprise), keys, adaptive_lr, values)

            grads = TensorDict(grads)

            # surprises
            adaptive_lr = rearrange(adaptive_lr, '(b h n) c -> b h (n c)', b = batch, h = heads)
            unweighted_mem_model_loss = rearrange(unweighted_mem_model_loss, '(b h n) c -> b h (n c)', b = batch, h = heads)

            # restore batch and sequence dimension
            grads = rearrange_dict_values(grads, '(b n) ... -> b n ...', b = batch * heads)

        # maybe softclamp grad norm

        if exists(self.max_grad_norm):
            grads = grads.apply(lambda t: softclamp_grad_norm(t, self.max_grad_norm))

        # maybe per layer modulation

        if need_layer_lr_mod:
            grads = TensorDict({name: einx.multiply('b h, b h ... -> b h ...', layer_lr_mod, t) for layer_lr_mod, (name, t) in zip(layer_lr_mod, grads.items())})

        # negative gradients, adaptive lr already applied as loss weight

        surprises = grads.mul(-1)

        # past states

        if not exists(past_state):
            # minibatch_init_weight corresponds to W0 in figure 7 of TTT paper

            minibatch_init_weight = weights
            init_momentum = self.init_momentum(batch)

            past_state = (minibatch_init_weight, init_momentum)

        past_last_update, past_last_momentum = past_state

        # early return if sequence length less than chunk size

        if num_chunks == 0:
            updates = rearrange_dict_values(weights, 'bh ... -> bh 1 ...')
            next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, past_state, updates)

            output = (updates, next_store_state)

            if not return_surprises:
                return output

            return (*output, (unweighted_mem_model_loss, adaptive_lr))

        # momentum + weight decay - momentum is the new contribution, as most linear RNNs have learned forgetting gates

        scan_fn = sequential_scan if self.use_sequential_scan else self.assoc_scan

        updates = TensorDict()

        next_last_update = TensorDict()
        next_last_momentum = TensorDict()

        for (param_name, surprise), (_, last_update) in zip(surprises.items(), past_last_update.items()):

            update = surprise

            # derive momentum with associative scan - eq (10)

            if has_momentum:
                momentum = surprise

                momentums = []

                last_momentum = past_last_momentum[param_name]

                # go from first order momentum all the way to the Nth

                for one_adaptive_momentum, one_last_momentum in zip_longest(adaptive_momentum, last_momentum):
                    momentum = scan_fn(one_adaptive_momentum, momentum, prev = one_last_momentum)

                    momentums.append(momentum)

                momentums = stack(momentums)

                next_last_momentum[param_name] = momentums[:, :, -1]

                if learned_combine and self.learned_combine_include_zeroth:
                    momentums = cat((rearrange(surprise, '... -> 1 ...'), momentums), dim = 0)

                if not learned_combine:
                    update = momentums[-1]
                else:
                    update = einsum(combine_momentums, momentums, 'o b n, o b n ... -> b n ...')

            # maybe spectral norm surprises

            if self.spectral_norm_surprises:
                update = newtonschulz5(update, steps = self.muon_ns_steps, eps = self.muon_ns_eps)

                if apply_eta_outside:
                    # paper Table 1: η_t scales the spectrally-normalized surprise (outside NS-5).
                    # NS-5 is scale-invariant, so this is where the adaptive lr actually takes effect.
                    update = einx.multiply('bh m, bh m ... -> bh m ...', eta_for_update, update)

            # use associative scan again for learned forgetting (weight decay) - eq (13)

            update = scan_fn(1. - decay_factor, update, prev = last_update, remove_prev = False)

            updates[param_name] = update
            next_last_update[param_name] = update[:, -1]

        # determine next state for the storing of memories

        next_state = (next_last_update, next_last_momentum)

        next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, next_state, updates)

        # return updates to neural memory at all chunked timesteps + neural mem cache / state to be fed back

        if not return_surprises:
            return updates, next_store_state

        return updates, next_store_state, (unweighted_mem_model_loss, adaptive_lr)

    def retrieve_memories(
        self,
        seq,
        weights: dict[str, Tensor],
    ):
        chunk_size = self.retrieve_chunk_size

        weights_have_expanded_shape = dict_get_value_shapes(weights) != self.init_weight_shape

        batch, seq_len = seq.shape[:2]

        # auto infer single token decoding, if there are only 1 set of weights and 1 token

        is_one_token = seq_len == 1
        is_one_weight = (not weights_have_expanded_shape) or next(iter(weights.values())).shape[1] == 1

        is_single_token_decode = is_one_token and is_one_weight

        if is_single_token_decode:
            chunk_size = 1

        # padding related, for chunked processing

        need_pad = chunk_size > 1 or not is_one_weight

        if need_pad:
            seq = pad_at_dim(seq, (1, 0), dim = 1)

        seq_len_plus_one = seq.shape[-2]

        next_seq_len = round_up_multiple(seq_len_plus_one, chunk_size)

        padding = next_seq_len - seq_len_plus_one
        seq = pad_at_dim(seq, (0, padding), dim = 1)

        # the parameters of the memory model stores the memories of the key / values
        # when the MLP has only 1 weight matrix, it is equivalent to `kv` fast weight memories from linear attention literature (recall fetching of memories is q @ (kv)) / schmidhuber's paper

        weights = TensorDict(weights)

        # pre norm

        seq = self.retrieve_norm(seq)

        # sequence Float['b n d'] to queries

        queries = self.to_queries(seq)

        # maybe short causal conv

        if exists(self.query_conv):
            queries = self.query_conv(queries)

        # maybe multihead

        queries = self.split_heads(queries)

        # maybe qk rmsnorm

        queries = self.q_norm(queries)

        # maybe polynomial feature expansion (Atlas Section 3.1)

        if exists(self.poly_features):
            queries = self.poly_features(queries)

        # fetch values from memory model

        if weights_have_expanded_shape:
            if self.omega_context > 1:
                if chunk_size == 1:
                    # per-token retrieve: use full per-token weights, no subsampling.
                    # pad for remainder tokens (those not processed by store_memories due
                    # to round_down_multiple). the last M_t is the correct state.
                    n_weights = next(iter(weights.values())).shape[1]
                    n_needed = seq_len_plus_one
                    if n_weights < n_needed:
                        diff = n_needed - n_weights
                        weights = weights.apply(
                            lambda t: cat((t, t[:, -1:].expand(-1, diff, *t.shape[2:])), dim = 1)
                        )
                else:
                    # per-chunk retrieve approximation: subsample per-token weights at chunk
                    # boundaries. uses last token's M_t within each chunk. same granularity
                    # as non-omega Titans retrieve. costs less autograd memory than per-token.
                    init_w = weights.apply(lambda t: t[:, :1])
                    token_w = weights.apply(lambda t: t[:, 1:])
                    subsampled = token_w.apply(lambda t: t[:, self.store_chunk_size - 1::self.store_chunk_size])
                    weights = TensorDict({k: cat((init_w[k], subsampled[k]), dim = 1) for k in weights.keys()})

            weights = rearrange_dict_values(weights, 'b n ... -> (b n) ...')

        queries = rearrange(queries, 'b h (n c) d -> (b h n) c d', c = chunk_size)

        # forward functional call

        values = functional_call(self.memory_model, dict(weights), queries)

        # reconstitute batch dimension

        values = rearrange(values, '(b h n) c d -> b h (n c) d', b = batch, h = self.heads)

        values = self.multihead_rmsnorm(values)

        # maybe gate

        if exists(self.retrieve_gate):
            values = values * self.retrieve_gate(seq)

        # maybe merge heads and combine

        values = self.merge_heads(values)

        values = self.combine_heads(values)

        # restore, pad with empty memory embed

        if need_pad:
            values = values[:, 1:]

        return values[:, :seq_len]

    def forward(
        self,
        seq,
        store_seq = None,
        state: NeuralMemState | None = None,
        detach_mem_state = False,
        prev_weights = None,
        store_mask: Tensor | None = None,
        return_surprises = False,
        ttt_batch_size: int | None = None
    ):
        is_multi_input = self.qkv_receives_diff_views

        # handle single token

        if seq.ndim == 2 or (is_multi_input and seq.ndim == 3):
            seq = rearrange(seq, '... b d -> ... b 1 d')

        is_single_token = seq.shape[-2] == 1

        # if different views for qkv, then

        if is_multi_input:
            retrieve_seq, seq = seq[0], seq[1:]
        else:
            retrieve_seq = seq

        # handle previous state init

        if not exists(state):
            state = (0, None, None, None, None)

        seq_index, weights, cache_store_seq, past_state, updates = state

        # store

        store_seq = default(store_seq, seq)

        # take care of cache

        if exists(cache_store_seq):
            store_seq = safe_cat((cache_store_seq, store_seq))

        # compute split sizes of sequence
        # for now manually update weights to last update at the correct boundaries

        store_seq_len, chunk_size, batch_size = store_seq.shape[-2], self.chunk_size, default(ttt_batch_size, self.batch_size)

        need_update_weights = exists(batch_size)

        # determine split sizes and when to update

        if need_update_weights:
            update_after_final_store = divisible_by(seq_index + store_seq_len, batch_size)

            seq_range = torch.arange(store_seq_len) + seq_index + 1
            batch_boundary = divisible_by(seq_range, batch_size)

            indices = seq_range[batch_boundary] - seq_index

            indices = F.pad(indices, (1, 0), value = 0)

            if indices[-1] != store_seq_len:
                indices = F.pad(indices, (0, 1), value = store_seq_len)

            split_sizes = (indices[1:] - indices[:-1]).tolist()

            assert sum(split_sizes) == store_seq_len
        else:
            split_sizes = (store_seq_len,)
            update_after_final_store = False

        # collect segment updates — concatenated once at the end to avoid O(n²)
        # autograd memory from incremental cat (each intermediate concat is retained
        # for backward, holding all prior segments simultaneously)

        all_segment_updates = []

        # loop through chunks of store sequences

        store_seqs = store_seq.split(split_sizes, dim = -2)

        if exists(store_mask):
            store_masks = store_mask.split(split_sizes, dim = -1)
        else:
            store_masks = (None,) * len(split_sizes)

        # whether to allow network to slowly adjust from initial weight throughout (residual path) to fully updating weights every batch

        surprises = (None, None)
        gate = None

        if exists(self.transition_gate):
            gate = self.transition_gate.sigmoid()

        for ind, (store_seq_chunk, maybe_store_mask) in enumerate(zip(store_seqs, store_masks)):
            is_last = ind == (len(store_seqs) - 1)

            # store

            next_updates, next_neural_mem_state, chunk_surprises = self.store_memories(
                store_seq_chunk,
                weights = weights,
                past_state = past_state,
                seq_index = seq_index,
                prev_weights = prev_weights,
                mask = maybe_store_mask,
                return_surprises = True,
            )

            seq_index = next_neural_mem_state.seq_index

            # optionally detach intermediate segments to reduce autograd memory from
            # O(segments) to O(1). trade-off: outer-loop (LM loss) gradients for store-side
            # parameters (to_keys, adaptive step, momentum, decay) come only from the last
            # segment. inner-loop surprise gradients are unaffected. this is a standard
            # approximation in TTT literature (see TTT-E2E for the non-truncated alternative).
            # enable via detach_segment_memory=True when memory is constrained.
            if self.detach_segment_memory and not is_last:
                next_updates = next_updates.apply(lambda t: t.detach())
                weights = next_neural_mem_state.weights.apply(lambda t: t.detach())
                past_state = tree_map(lambda t: t.detach() if is_tensor(t) else t, next_neural_mem_state.states)
            else:
                weights = next_neural_mem_state.weights
                past_state = next_neural_mem_state.states

            all_segment_updates.append(next_updates)

            surprises = tuple(safe_cat(args, dim = -1) for args in zip(surprises, chunk_surprises))

            if is_last and not update_after_final_store:
                continue

            # update weights once batch size is fulfilled

            last_update, last_momentum = past_state

            if exists(gate):
                last_update = TensorDict({param_name: one_weight.lerp(one_last_update, gate) for (param_name, one_weight), (_, one_last_update) in zip(weights.items(), last_update.items())})

            past_state = (last_update, last_momentum)

            # set weights to the last updated weights for the last minibatch

            weights = last_update

            next_neural_mem_state = next_neural_mem_state._replace(
                weights = weights,
                states = past_state,
            )

        # single cat of all segment updates — drop last entry of each segment except
        # the final one (boundary overlap: next segment's first entry is the gated
        # version of the previous segment's last entry, matching original accum_updates)

        if len(all_segment_updates) == 1:
            updates = all_segment_updates[0]
        else:
            trimmed = [
                s.apply(lambda t: t[:, :-1]) for s in all_segment_updates[:-1]
            ] + [all_segment_updates[-1]]
            updates = TensorDict({
                name: cat([s[name] for s in trimmed], dim=1)
                for name in all_segment_updates[0].keys()
            })

        next_neural_mem_state = next_neural_mem_state._replace(updates = updates)

        # retrieve

        if is_single_token:
            last_update, _ = next_neural_mem_state.states
            updates = rearrange_dict_values(last_update, 'b ... -> b 1 ...')

        retrieved = self.retrieve_memories(
            retrieve_seq,
            updates
        )

        # maybe detach

        if detach_mem_state:
            next_neural_mem_state = mem_state_detach(next_neural_mem_state)

        # returning

        if not return_surprises:
            return retrieved, next_neural_mem_state

        return retrieved, next_neural_mem_state, surprises
