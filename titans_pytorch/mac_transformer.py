from __future__ import annotations
from typing import Callable

from math import ceil
from copy import deepcopy
from functools import partial
from collections import namedtuple

import tqdm

import torch
from torch import nn, stack, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

# flex attention
# https://pytorch.org/blog/flexattention/

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

def create_mac_block_mask(seq_len, window_size, persist_mem_len, sliding = False):

    def create_mac_mask(_, __, q_idx, kv_idx):
        is_persist_mem = kv_idx < persist_mem_len
        kv_without_mem = kv_idx - persist_mem_len
        causal_mask = q_idx >= kv_without_mem

        if not sliding:
            block_diagonal = (q_idx // window_size) == (kv_without_mem // window_size)
            causal_mask = causal_mask & block_diagonal
        else:
            sliding_mask = (q_idx - kv_without_mem) <= window_size
            causal_mask = causal_mask & sliding_mask

        return is_persist_mem | (~is_persist_mem & causal_mask)

    block_mask = create_block_mask(create_mac_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len + persist_mem_len, _compile = True)
    return block_mask

# einstein notation related

from einops import repeat, rearrange, pack, unpack, einsum
from einops.layers.torch import Rearrange

# b - batch
# n - sequence
# h - heads
# d - feature dimension

# absolute and relative positions

from axial_positional_embedding import ContinuousAxialPositionalEmbedding
from rotary_embedding_torch import RotaryEmbedding

# hyper connections / attend from x-transformers, which handles different queries and key lengths better

from x_transformers.attend import Attend

from hyper_connections import mc_get_init_and_expand_reduce_stream_functions

# proposed neural memory

from titans_pytorch.neural_memory import NeuralMemory

# constants

LinearNoBias = partial(Linear, bias = False)

AttnIntermediates = namedtuple('AttnIntermediates', ('value_residual', 'cached_key_values'))

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def round_up_multiple(seq, mult):
    return ceil(seq / mult) * mult

def round_down_multiple(seq, mult):
    return seq // mult * mult

def pack_with_inverse(t, pattern):
    packed, packed_shape = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        return unpack(out, packed_shape, default(inv_pattern, pattern))

    return packed, inverse

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pad_and_segment_with_inverse(
    seq,
    segment_len,
    fold_into_batch = True,
    inverse_remove_pad = True
):
    batch, seq_len = seq.shape[:2]
    next_seq_len_mult = round_up_multiple(seq_len, segment_len)

    padding = next_seq_len_mult - seq_len
    needs_pad = padding > 0

    if needs_pad:
        seq = F.pad(seq, (0, 0, 0, padding))

    if fold_into_batch:
        seq = rearrange(seq, 'b (w n) d -> (b w) n d', n = segment_len)

    def inverse(out):

        if fold_into_batch:
            out = rearrange(out, '(b w) ... n d -> b ... (w n) d', b = batch)

        if needs_pad and inverse_remove_pad:
            out = out[..., :-padding, :]

        return out

    return seq, inverse

# sampling related

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    if temperature > 0.:
        t = t / temperature + gumbel_noise(t)
    return t.argmax(dim = -1, keepdim = True)

# min_p
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# feedforward and attention

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.silu(gate) * x

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )

class SegmentedAttention(Module):
    def __init__(
        self,
        dim,
        segment_len,
        num_persist_mem_tokens = 0,
        num_longterm_mem_tokens = 0,
        dim_head = 64,
        heads = 8,
        sliding = False,
        accept_value_residual = False,
        attend_kwargs: dict = dict(),
        use_flex_attn = False
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

        dim_inner = dim_head * heads

        self.rotary_emb = RotaryEmbedding(dim_head)

        self.attend = Attend(causal = True, **attend_kwargs)

        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_learned_v_mix = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if accept_value_residual else None

        self.segment_len = segment_len
        self.num_longterm_mem_tokens = num_longterm_mem_tokens

        total_segment_len = segment_len + num_longterm_mem_tokens
        self.total_segment_len = total_segment_len

        self.sliding = sliding # sliding window attn - doubt their non-sliding results being the best. local attention with overlapping windows is very strong

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.persistent_memory = nn.Parameter(torch.zeros(2, heads, num_persist_mem_tokens, dim_head))

        # flex attn related

        assert not (use_flex_attn and not exists(flex_attention)), 'you need to be on the latest pytorch with a cuda device available'
        self.use_flex_attn = use_flex_attn

        self.segment_len = segment_len
        self.num_persist_mem_tokens = num_persist_mem_tokens

    def project_qkv(self, seq, value_residual = None):
        """Shared prologue of every path: pre-norm, q / k / v projections split
        into heads, and the learned value-residual mix. Returns (q, k, v,
        orig_v) with orig_v the pre-mix values (the residual handed to the
        next layers)."""
        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        seq = self.norm(seq)

        q, k, v = self.to_qkv(seq).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        orig_v = v

        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        return q, k, v, orig_v

    def prepend_persistent_memory(self, k, v):
        """Persistent memory keys / values in front of `k` / `v` (cast to their
        dtype: the Parameter stays fp32 under bf16 autocast)."""
        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b = k.shape[0])

        k = cat((pmk.to(k.dtype), k), dim = -2)
        v = cat((pmv.to(v.dtype), v), dim = -2)

        return k, v

    def project_out(self, out, output_gating = None):
        """Shared epilogue: merge heads, output projection, optional gating by
        the retrieved memories."""
        out = self.merge_heads(out)

        out = self.to_out(out)

        if exists(output_gating):
            out = out * output_gating

        return out

    def forward_inference(
        self,
        token,
        cache,
        value_residual = None,
        output_gating = None,
    ):
        q, k, v, orig_v = self.project_qkv(token, value_residual = value_residual)

        # caching

        ck, cv = cache
        k = cat((ck, k), dim = -2)
        v = cat((cv, v), dim = -2)

        next_cache = (k, v)

        # relative positions

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # persistent memory

        k, v = self.prepend_persistent_memory(k, v)

        # attention

        out, _ = self.attend(q, k, v)

        out = self.project_out(out, output_gating = output_gating)

        return out, AttnIntermediates(orig_v, next_cache)

    def forward_flex(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        output_gating = None,
        cache = None
    ):

        seq_len = seq.shape[1]

        q, k, v, orig_v = self.project_qkv(seq, value_residual = value_residual)

        # caching

        next_cache = (k, v)

        # relative positions

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # persistent memory

        k, v = self.prepend_persistent_memory(k, v)

        # prep flex attention

        if not exists(flex_attn_fn):
            block_mask = create_mac_block_mask(seq_len, self.total_segment_len, self.num_persist_mem_tokens, self.sliding)

            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        # attention

        out = flex_attn_fn(q, k, v)

        out = self.project_out(out, output_gating = output_gating)

        return out, AttnIntermediates(orig_v, next_cache)

    def forward(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        disable_flex_attn = False,
        output_gating = None,
        cache = None,
        prev_kv = None,
    ):
        """`prev_kv` (chunked inference, MemoryAsContextTransformer.
        iter_chunked_hidden): the un-rotated keys / values, each (b, h, m, d),
        of the m interleaved positions immediately before `seq`, sliced from
        the cache this method returned for the previous chunk. The joint
        sequence [prev ∥ seq] must start on an attention segment boundary
        (the caller slices the cache so that it does); it is then folded into
        segments exactly as the whole-sequence forward folds the full
        sequence, so `seq`'s rows see the same keys under the same mask — the
        previous segment plus the head of their own (sliding) or their own
        segment only (block). The prev rows get zero queries and their
        outputs are dropped; memory is O(len(seq)). Returns the outputs of
        `seq`'s rows and, as the next cache, the un-rotated keys / values of
        [prev ∥ seq] (the caller keeps the last 2 x total_segment_len).
        Non-flex path only.
        """
        is_inferencing = exists(cache)

        if is_inferencing:
            assert seq.shape[-2] == 1
            return self.forward_inference(seq, cache, value_residual, output_gating = output_gating)

        if seq.is_cuda and self.use_flex_attn and not disable_flex_attn:
            assert not exists(prev_kv), 'prev_kv (chunked inference) needs the non-flex path: pass disable_flex_attn = True'
            return self.forward_flex(seq, value_residual, flex_attn_fn, output_gating = output_gating, cache = cache)

        total_segment_len = self.total_segment_len

        batch, seq_len = seq.shape[:2]

        q, k, v, orig_v = self.project_qkv(seq, value_residual = value_residual)

        # chunked inference: the carried keys / values of the positions right
        # before this chunk join with zero queries (rows computed, then dropped)

        prev_len = 0

        if exists(prev_kv):
            prev_k, prev_v = prev_kv
            prev_len = prev_k.shape[-2]
            k = cat((prev_k, k), dim = -2)
            v = cat((prev_v, v), dim = -2)
            q = pad_at_dim(q, (prev_len, 0), dim = -2)

        # caching — un-rotated, without the segment padding below

        next_cache = (k, v)

        # pad to a multiple of the segment length. Zero rows after the
        # projections equal projecting zero-padded input rows (RMSNorm(0) = 0
        # and the projections have no bias), and let the cached rows join
        # without being re-projected.

        total_len = q.shape[-2]
        padded_len = round_up_multiple(total_len, total_segment_len)

        if padded_len > total_len:
            q, k, v = tuple(pad_at_dim(t, (0, padded_len - total_len), dim = -2) for t in (q, k, v))

        # relative positions — offsets from the start of [prev ∥ seq]; only
        # the differences matter and they equal the whole-sequence forward's

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # fold

        q, k, v = tuple(rearrange(t, 'b h (w n) d -> (b w) h n d', n = total_segment_len) for t in (q, k, v))

        # maybe sliding for cpu

        attend_kwargs = dict()

        if self.sliding:
            k, v = tuple(rearrange(t, '(b w) ... -> b w ...', b = batch) for t in (k, v))
            k, v = tuple(pad_at_dim(t, (1, 0), value = 0., dim = 1) for t in (k, v))
            k = cat((k[:, :-1], k[:, 1:]), dim = -2)
            v = cat((v[:, :-1], v[:, 1:]), dim = -2)
            k, v = tuple(rearrange(t, 'b w ... -> (b w) ...') for t in (k, v))

            # take care of masking

            idx = torch.arange(padded_len, device = seq.device)
            q_idx = rearrange(idx, '(w n) -> w n', n = total_segment_len)
            k_idx = pad_at_dim(q_idx, (1, 0), dim = 0, value = -1e4)
            k_idx = cat((k_idx[:-1], k_idx[1:]), dim = -1)

            q_idx = rearrange(q_idx, 'w i -> w i 1')
            k_idx = rearrange(k_idx, 'w j -> w 1 j')

            sliding_mask = (q_idx - k_idx) <= total_segment_len
            sliding_mask = F.pad(sliding_mask, (self.num_persist_mem_tokens, 0), value = True)

            sliding_mask = repeat(sliding_mask, 'w i j -> (b w) 1 i j', b = batch)
            attend_kwargs.update(mask = sliding_mask)

        # persistent memory (per window)

        k, v = self.prepend_persistent_memory(k, v)

        # attention

        out, _ = self.attend(q, k, v, **attend_kwargs)

        out = self.merge_heads(out)

        out = self.to_out(out)

        out = rearrange(out, '(b w) n d -> b (w n) d', b = batch)

        # drop the prev rows and the segment padding

        out = out[:, prev_len:prev_len + seq_len]

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

# MAC transformer

class MemoryAsContextTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        segment_len,
        neural_memory_segment_len = None,
        neural_mem_gate_attn_output = False,
        neural_memory_add_value_residual = False,
        num_longterm_mem_tokens = 0,
        num_persist_mem_tokens = 0,
        neural_memory_batch_size = None,
        neural_memory_qkv_receives_diff_views = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        num_residual_streams = 4,
        neural_memory_model: Module | None = None,
        neural_memory_kwargs: dict = dict(),
        neural_memory_layers: tuple[int, ...] | None = None,
        use_flex_attn = False,
        sliding_window_attn = False,
        neural_mem_weight_residual = False,
        token_emb: Module | None = None,
        use_axial_pos_emb = True,  # lucidrains default kept for library users; the experiment config turns it off (see experiments/configs.py)
    ):
        super().__init__()

        if not exists(token_emb):
            token_emb = nn.Embedding(num_tokens, dim)

        self.token_emb = token_emb

        # absolute positions — optional. the continuous axial embedding feeds raw
        # integer segment indices (arange(ceil(seq_len / neural_mem_segment_len)))
        # into a SiLU MLP with no normalization, so at any sequence length beyond
        # the training length the outer-axis inputs are out of distribution and
        # the embedding norm grows roughly linearly with position (measured at
        # random init, 3 seeds: 7.6x the trained range at 4K for a 1K-trained
        # model, 30x at 16K, 243x at 128K, ~1950x at 1M; 2026-09-02). rotary
        # inside the windowed attention already carries within-window position
        # and the neural memory is position-free, so the experiment config
        # disables it.

        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(dim = dim, num_axial_dims = 2) if use_axial_pos_emb else None

        # long term mem tokens

        self.segment_len = segment_len

        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        has_longterm_mems = num_longterm_mem_tokens > 0

        self.longterm_mems = nn.Parameter(torch.randn(num_longterm_mem_tokens, dim) * 0.02)

        # maybe sliding window attn

        self.sliding_window_attn = sliding_window_attn
        self.attn_window_size = segment_len + num_longterm_mem_tokens

        # hyper connection

        self.num_residual_streams = num_residual_streams

        init_hyper_conn, self.expand_streams, self.reduce_streams = mc_get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim, add_stream_embed = True, disable = num_residual_streams == 1)

        self.layers = ModuleList([])

        self.neural_memory_segment_len = default(neural_memory_segment_len, num_longterm_mem_tokens + segment_len)

        # kept for chunked inference: chunk boundaries must coincide with the
        # neural memory's batch (segment) boundaries — see iter_chunked_hidden
        self.neural_memory_batch_size = neural_memory_batch_size

        layers = tuple(range(1, depth + 1))

        neural_memory_layers = default(neural_memory_layers, layers)

        # weight residual related

        self.neural_mem_weight_residual = neural_mem_weight_residual
        is_first_neural_mem = True

        # mem, attn, and feedforward layers

        for layer in layers:
            is_first = layer == 1

            # attention and feedforward

            attn = SegmentedAttention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                segment_len = segment_len,
                use_flex_attn = use_flex_attn,
                accept_value_residual = not is_first,
                num_longterm_mem_tokens = num_longterm_mem_tokens,
                num_persist_mem_tokens = num_persist_mem_tokens,
                sliding = sliding_window_attn
            )

            mem = None
            mem_qkv_layer_selector = None
            mem_hyper_conn = None

            if layer in neural_memory_layers:
                mem_hyper_conn = init_hyper_conn(add_branch_out_to_residual = not neural_mem_gate_attn_output)

                if not is_first and neural_memory_qkv_receives_diff_views:
                    num_layer_choices = (layer - 1) * 4 + 1 # for each layer, have memory input select from attn inp, attn out, ff inp, and ff out - plus one for the current point in the residual stream (memory input)

                    mem_qkv_layer_selector = nn.Sequential(
                        nn.RMSNorm(dim),
                        nn.Linear(dim, 3 * num_layer_choices),
                        Rearrange('... (views layers) -> views ... layers', views = 3),
                        nn.Softmax(dim = -1)
                    )

                mem = NeuralMemory(
                    dim = dim,
                    chunk_size = self.neural_memory_segment_len,
                    batch_size = neural_memory_batch_size,
                    model = deepcopy(neural_memory_model),
                    qkv_receives_diff_views = True,
                    accept_weight_residual = neural_mem_weight_residual and not is_first_neural_mem,
                    **neural_memory_kwargs
                )

                is_first_neural_mem = False

            ff = FeedForward(dim = dim, mult = ff_mult)

            self.layers.append(ModuleList([
                mem_hyper_conn,
                init_hyper_conn(),
                init_hyper_conn(),
                mem_qkv_layer_selector,
                mem,
                attn,
                ff,
            ]))

        # layer slot 4 is the neural memory (None on attention-only layers)

        self.has_neural_memory = any(exists(layer[4]) for layer in self.layers)

        self.norm = nn.RMSNorm(dim)

        self.to_logits = LinearNoBias(dim, num_tokens)

        # whether to gate the attention output with the retrieved memories

        self.gate_attn_output = neural_mem_gate_attn_output

        # zero for maybe aux loss + device

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # flex attn related

        assert not (use_flex_attn and not exists(flex_attention)), 'you need to be on the latest pytorch with a cuda device available'
        self.use_flex_attn = use_flex_attn

        self.num_persist_mem_tokens = num_persist_mem_tokens

    def seq_index_is_longterm(
        self,
        seq_index
    ):
        total_segment_len, segment_len = self.attn_window_size, self.segment_len
        return ((seq_index % total_segment_len + 1) - segment_len) > 0

    def seq_len_with_longterm_mem(
        self,
        seq_len
    ):
        assert seq_len > 0

        segment_len, num_mem = self.segment_len, self.num_longterm_mem_tokens
        return ((seq_len - 1) // segment_len) * num_mem + seq_len

    @torch.no_grad()
    def sample(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.5,
        filter_fn: Callable = min_p_filter,
        filter_kwargs: dict = dict(
            min_p = 0.1,
        ),
        show_progress = True,
        use_cache = False
    ):
        was_training = self.training
        self.eval()

        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        # cache for axial pos, attention, and neural memory

        cache = None
        factorized_pos_emb = None

        # precompute factorized pos emb (only when the absolute embedding is enabled)

        if use_cache and exists(self.axial_pos_emb):
            seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)

            axial_dims = self.axial_pos_emb.maybe_derive_outer_dim(seq_len_with_mem, (self.neural_memory_segment_len,))

            factorized_pos_emb = self.axial_pos_emb(axial_dims, return_factorized = True)

        # sample

        with tqdm.tqdm(total = sample_num_times, disable = not show_progress) as pbar:

            while out.shape[-1] < seq_len:

                logits, next_cache = self.forward(
                    out,
                    disable_flex_attn = True,
                    cache = cache,
                    return_cache = True,
                    factorized_pos_emb = factorized_pos_emb
                )

                if use_cache:
                    cache = next_cache

                if not exists(logits):
                    continue

                logits = logits[:, -1]

                logits = filter_fn(logits, **filter_kwargs)
                sample = gumbel_sample(logits, temperature = temperature)

                out = torch.cat((out, sample), dim = -1)
                pbar.update(1)

        self.train(was_training)

        return out[..., prompt_seq_len:]

    def chunked_inference_alignment(self, chunk_len):
        """Validate a chunk length for iter_chunked_hidden; raise ValueError with
        the reason otherwise. `chunk_len` is measured in INTERLEAVED positions
        (tokens plus the longterm-mem tokens inserted after every segment) —
        the neural memory's own axis, on which its batch (segment) boundaries
        lie. Chunk boundaries must coincide with those boundaries: a boundary
        inside a memory segment would add an omega-window truncation the
        parallel forward does not have, and a chunk that is not a multiple of
        the store chunk would leave remainder tokens reading a stale state
        mid-sequence. Original tokens per chunk therefore vary (e.g. 1024
        interleaved positions = 964 tokens + 60 mem tokens at segment_len 64
        with 4 mem tokens); the generator reports the token offsets."""
        if chunk_len <= 0:
            raise ValueError(f'chunk_len must be positive, got {chunk_len}')

        # attention-only models: any positive chunk length aligns (the
        # attention fold is aligned by the carried cache, not by the chunk)

        if self.has_neural_memory:
            if not divisible_by(chunk_len, self.neural_memory_segment_len):
                raise ValueError(
                    f'chunk_len ({chunk_len}) must be a multiple of the neural memory store chunk '
                    f'({self.neural_memory_segment_len}) — remainder positions would read a stale memory state'
                )

            if not exists(self.neural_memory_batch_size):
                raise ValueError(
                    'chunked inference needs neural_memory_batch_size: without it the parallel forward '
                    'treats the whole sequence as one memory segment, so any chunk boundary would add an '
                    'omega-window truncation and the chunked output could not match it'
                )

            if not divisible_by(chunk_len, self.neural_memory_batch_size):
                raise ValueError(
                    f'chunk_len ({chunk_len} interleaved positions) must be a multiple of '
                    f'neural_memory_batch_size ({self.neural_memory_batch_size}) so chunk boundaries '
                    f'coincide with the memory segment boundaries of the parallel forward'
                )

        if self.neural_mem_weight_residual:
            raise ValueError('chunked inference does not support neural_mem_weight_residual')

        # gated_transition: the whole-sequence forward's segment concatenation
        # drops each non-final segment's last entry and substitutes the next
        # segment's first entry — the GATED lerp(weights, last_update, gate)
        # state — so the last token of every segment retrieves with the gated
        # state, while a chunk boundary there retrieves with the un-gated one
        # (measured 2-3 max deviation from the first boundary onward,
        # 2026-09-02). Replicating that boundary semantics chunkwise is
        # possible but out of scope; refuse the config instead.

        for layer in self.layers:
            mem = layer[4]
            if exists(mem) and exists(mem.transition_gate):
                raise ValueError(
                    'chunked inference does not support gated_transition: the parallel forward '
                    'retrieves the last token of each memory segment with the gated state, which '
                    'a chunk boundary cannot reproduce'
                )

    @torch.no_grad()
    def iter_chunked_hidden(
        self,
        x,
        chunk_len,
    ):
        """Chunked inference: process token ids `x` (batch, seq_len) in chunks
        of `chunk_len` INTERLEAVED positions, carrying only state across chunk
        boundaries — the neural memory state (weights, momentum, store
        remainder, query-conv context) and each attention layer's un-rotated
        keys/values for the last two attention segments — so memory is
        O(chunk) instead of O(L).

        Yields (start, hidden): final-normed hidden states for the ORIGINAL
        token positions [start, start + hidden.shape[1]) covered by the chunk,
        longterm-mem tokens excised. Equal to the whole-sequence forward's
        hidden states (parity-tested: same interleave, same axial positional
        embedding at global positions when enabled, same memory segmentation
        because chunk boundaries coincide with memory segment boundaries —
        see chunked_inference_alignment — and the same segment fold in the
        attention: a chunk may start mid-segment, so each layer is handed
        the cached keys/values back to the previous segment boundary and
        folds [cache ∥ chunk] exactly as the whole-sequence forward folds
        the full sequence, see SegmentedAttention.forward's `prev_kv`).

        The last chunk may be shorter. Not a decoding path: it is the
        parallel forward split along the sequence, for likelihood-style
        evaluation past the whole-sequence memory ceiling.
        """
        self.chunked_inference_alignment(chunk_len)

        batch, seq_len = x.shape
        segment_len, total_segment_len = self.segment_len, self.attn_window_size
        device = x.device

        num_positions = self.seq_len_with_longterm_mem(seq_len)

        # global interleaved position -> (segment, offset): offsets below
        # segment_len are tokens, the rest are that segment's mem tokens

        positions = torch.arange(num_positions, device = device)
        segment_index, offset = positions // total_segment_len, positions % total_segment_len
        is_token = offset < segment_len
        token_index = segment_index * segment_len + offset
        mem_index = offset - segment_len

        kv_caches = [None] * len(self.layers)
        mem_states = [None] * len(self.layers)

        for pos_start in range(0, num_positions, chunk_len):
            pos_end = min(pos_start + chunk_len, num_positions)

            chunk_is_token = is_token[pos_start:pos_end]
            chunk_token_index = token_index[pos_start:pos_end][chunk_is_token]

            # one method call per chunk, so its transients (activations, the
            # per-token memory states) die on return — before the next chunk
            # allocates its own

            h = self._chunk_hidden(
                x = x,
                chunk_positions = positions[pos_start:pos_end],
                chunk_is_token = chunk_is_token,
                chunk_token_index = chunk_token_index,
                chunk_mem_index = mem_index[pos_start:pos_end][~chunk_is_token],
                kv_caches = kv_caches,
                mem_states = mem_states,
            )

            if h.shape[1] == 0:
                continue

            yield int(chunk_token_index[0]), h

    def _chunk_hidden(
        self,
        x,
        chunk_positions,
        chunk_is_token,
        chunk_token_index,
        chunk_mem_index,
        kv_caches,
        mem_states,
    ):
        """One chunk of iter_chunked_hidden: every layer over the interleaved
        positions `chunk_positions`, advancing the carried attention caches
        and memory states IN PLACE (lists indexed by layer), returning the
        final-normed hidden states of the chunk's original tokens."""
        batch = x.shape[0]
        total_segment_len = self.attn_window_size
        pos_start = int(chunk_positions[0])
        # the chunk's interleaved embeddings — the same layout the parallel
        # forward builds with pad_and_segment + pack, assembled directly.
        # dtype taken from the embedded tokens themselves (token_emb may be
        # any module, not necessarily an nn.Embedding with a `.weight`)

        token_embeds = self.token_emb(x[:, chunk_token_index])
        emb_dtype = token_embeds.dtype

        emb = torch.empty((batch, chunk_positions.shape[0], self.longterm_mems.shape[-1]), device = x.device, dtype = emb_dtype)
        emb[:, chunk_is_token] = token_embeds
        emb[:, ~chunk_is_token] = self.longterm_mems[chunk_mem_index].to(emb_dtype)

        if exists(self.axial_pos_emb):
            emb = emb + self.axial_pos_emb.forward_with_pos(chunk_positions, (self.neural_memory_segment_len,))

        h = self.expand_streams(emb)

        value_residual = None
        mem_input_layers = []

        for layer_index, (mem_hyper_conn, attn_hyper_conn, ff_hyper_conn, mem_qkv_layer_selector, mem, attn, ff) in enumerate(self.layers):

            attn_out_gates = None

            if exists(mem):
                mem_input, add_residual = mem_hyper_conn(h)

                if not exists(mem_qkv_layer_selector):
                    qkv_mem_input = stack((mem_input, mem_input, mem_input))
                else:
                    layers_to_choose_from = stack((mem_input, *mem_input_layers))
                    selected = mem_qkv_layer_selector(mem_input)
                    qkv_mem_input = einsum(layers_to_choose_from, selected, 'l b n d, v b n l -> v b n d')

                retrieved, next_mem_state = mem.forward(
                    seq = qkv_mem_input,
                    state = mem_states[layer_index],
                )

                # carry the compact state only: the per-token weight states
                # (`updates`, O(chunk)) are this chunk's retrieve inputs and
                # are never read by the next call — drop the local too, or
                # they would stay alive through the rest of this chunk
                mem_states[layer_index] = next_mem_state._replace(updates = None)
                del next_mem_state

                if self.gate_attn_output:
                    attn_out_gates = retrieved.sigmoid()
                else:
                    h = add_residual(retrieved)

            attn_in, add_residual = attn_hyper_conn(h)

            mem_input_layers.append(attn_in)

            # the cached rows the attention fold needs so that [prev ∥ chunk]
            # starts on a segment boundary: the head of the segment this
            # chunk starts in, plus the previous whole segment for sliding
            # windows (the cache holds everything since position 0 while
            # that is shorter)

            need = pos_start % total_segment_len + (total_segment_len if attn.sliding else 0)
            prev_kv = kv_caches[layer_index]

            if exists(prev_kv):
                prev_kv = tuple(t[..., -need:, :] for t in prev_kv) if need > 0 else None

            attn_out, (values, next_kv_cache) = attn(
                attn_in,
                value_residual = value_residual,
                disable_flex_attn = True,
                output_gating = attn_out_gates,
                prev_kv = prev_kv,
            )

            # a view would keep the whole chunk's keys / values alive: copy the
            # last two segments
            kv_caches[layer_index] = tuple(t[..., -2 * total_segment_len:, :].clone() for t in next_kv_cache)
            del next_kv_cache

            mem_input_layers.append(attn_out)

            value_residual = default(value_residual, values)

            h = add_residual(attn_out)

            ff_in, add_ff_residual = ff_hyper_conn(h)

            mem_input_layers.append(ff_in)

            ff_out = ff(ff_in)

            mem_input_layers.append(ff_out)

            h = add_ff_residual(ff_out)

        h = self.reduce_streams(h)

        # excise the mem tokens, final norm

        return self.norm(h[:, chunk_is_token])

    def forward_chunked(
        self,
        x,
        chunk_len,
        return_hidden = False,
    ):
        """Whole output of iter_chunked_hidden concatenated: logits (batch,
        seq_len, num_tokens) by default, or the hidden states with
        return_hidden=True. Peak memory is O(chunk_len), but the returned
        tensor is O(seq_len) — for a few positions, consume the generator."""
        pieces = [h for _, h in self.iter_chunked_hidden(x, chunk_len = chunk_len)]

        if len(pieces) == 0:
            raise ValueError(f'forward_chunked needs at least one token, got input shape {tuple(x.shape)}')

        hidden = cat(pieces, dim = 1)

        if hidden.shape[1] != x.shape[1]:
            raise RuntimeError(f'chunked forward yielded {hidden.shape[1]} rows for {x.shape[1]} tokens')

        if return_hidden:
            return hidden

        return self.to_logits(hidden)

    def forward(
        self,
        x,
        return_loss = False,
        return_loss_breakdown = False,
        disable_flex_attn = False,
        cache = None,
        return_cache = False,
        factorized_pos_emb = None,
        return_hidden = False   # return the final-normed hidden states instead of logits — callers that need a few positions project them with `to_logits` themselves, avoiding the [L, vocab] logits tensor (the BABILong scorer's memory ceiling at long contexts)
    ):

        if return_hidden and return_loss:
            raise ValueError('return_hidden returns pre-logit hidden states; it cannot be combined with return_loss')

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # math

        batch, seq_len, neural_mem_segment_len, segment_len, num_longterm_mem_tokens, attn_window_size = *x.shape, self.neural_memory_segment_len, self.segment_len, self.num_longterm_mem_tokens, self.attn_window_size

        seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)

        # token embedding

        x = self.token_emb(x)

        # intersperse longterm memory

        x, inverse_segment = pad_and_segment_with_inverse(x, segment_len, inverse_remove_pad = False)

        mems = repeat(self.longterm_mems, 'n d -> b n d', b = x.shape[0])
        x, inverse_pack_mems = pack_with_inverse((x, mems), 'b * d')

        x = inverse_segment(x)

        # splice out unneeded tokens from padding for longterm mems

        x = x[:, :seq_len_with_mem]

        # maybe apply axial positional embedding
        # so intra and inter segment can be more easily discerned by the network
        # (disabled in the experiment config — see the constructor note)

        if exists(self.axial_pos_emb):
            pos_emb = self.axial_pos_emb.forward_with_seq_len(seq_len_with_mem, (neural_mem_segment_len,), factorized = factorized_pos_emb)

            x = x + pos_emb

        # prep flex attention

        use_flex_attn = x.is_cuda and self.use_flex_attn and not disable_flex_attn

        flex_attn_fn = None

        if use_flex_attn:
            block_mask = create_mac_block_mask(seq_len_with_mem, self.attn_window_size, self.num_persist_mem_tokens, self.sliding_window_attn)
            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        # kv caching

        is_inferencing = exists(cache)

        if not exists(cache):
            cache = (seq_len_with_mem - 1, None, None)

        inference_seq_index, kv_caches, neural_mem_caches = cache

        kv_caches = iter(default(kv_caches, []))
        neural_mem_caches = iter(default(neural_mem_caches, []))

        next_kv_caches = []
        next_neural_mem_caches = []

        # value residual

        value_residual = None

        # neural mem weight residual

        mem_weight_residual = None

        # layers for the neural mem to select the qkv inputs from

        mem_input_layers = []

        # when inferencing, only do one token at a time

        if is_inferencing:
            ind = inference_seq_index
            x = x[:, ind:(ind + 1)]

        # expand and reduce streams for hyper connections

        x = self.expand_streams(x)

        for mem_hyper_conn, attn_hyper_conn, ff_hyper_conn, mem_qkv_layer_selector, mem, attn, ff in self.layers:

            retrieved = None
            attn_out_gates = None
            next_neural_mem_cache = None

            # maybe neural memory

            if exists(mem):

                mem_input, add_residual = mem_hyper_conn(x)

                if not exists(mem_qkv_layer_selector):
                    qkv_mem_input = stack((mem_input, mem_input, mem_input))
                else:
                    layers_to_choose_from = stack((mem_input, *mem_input_layers))

                    # let the current `mem_input` select the 3 layers for qkv

                    selected = mem_qkv_layer_selector(mem_input)

                    qkv_mem_input = einsum(layers_to_choose_from, selected, 'l b n d, v b n l -> v b n d')

                retrieved, next_neural_mem_cache = mem.forward(
                    qkv_mem_input,
                    state = next(neural_mem_caches, None),
                    prev_weights = mem_weight_residual
                )

                if self.neural_mem_weight_residual:
                    mem_weight_residual = next_neural_mem_cache.updates

                if self.gate_attn_output:
                    attn_out_gates = retrieved.sigmoid()
                else:
                    x = add_residual(retrieved)

            # attention

            attn_in, add_residual = attn_hyper_conn(x)

            mem_input_layers.append(attn_in)

            attn_out, (values, next_kv_cache) = attn(
                attn_in,
                value_residual = value_residual,
                disable_flex_attn = disable_flex_attn,
                flex_attn_fn = flex_attn_fn,
                output_gating = attn_out_gates,
                cache = next(kv_caches, None)
            )

            mem_input_layers.append(attn_out)

            value_residual = default(value_residual, values)

            x = add_residual(attn_out)

            # caches

            next_kv_caches.append(next_kv_cache)

            if exists(mem):
                next_neural_mem_caches.append(next_neural_mem_cache)

            # feedforward

            ff_in, add_ff_residual = ff_hyper_conn(x)

            mem_input_layers.append(ff_in)

            ff_out = ff(ff_in)

            mem_input_layers.append(ff_out)

            x = add_ff_residual(ff_out)

        # taking care of cache first
        # for early return when processing long term mem tokens during inference

        if return_cache:
            next_kv_caches = stack([stack(kv_cache) for kv_cache in next_kv_caches])

            # handle kv cache length depending on local attention type

            next_kv_caches = next_kv_caches[..., -attn_window_size:, :]

            kv_cache_length = next_kv_caches.shape[-2]

            if not self.sliding_window_attn and divisible_by(kv_cache_length, attn_window_size):
                next_kv_caches = next_kv_caches[..., 0:0, :]

            next_cache = (
                inference_seq_index + 1,
                next_kv_caches,
                next_neural_mem_caches
            )

            is_longterm_mem = self.seq_index_is_longterm(inference_seq_index)

            if is_inferencing and is_longterm_mem:
                return None, next_cache

        # hyper connection reducing of streams

        x = self.reduce_streams(x)

        # excise out the memories

        if not is_inferencing:

            x, inverse_segment = pad_and_segment_with_inverse(x, attn_window_size, inverse_remove_pad = False)

            x, _ = inverse_pack_mems(x)

            x = inverse_segment(x)

            x = x[:, :seq_len]

        # to logits

        x = self.norm(x)

        if return_hidden:
            if not return_cache:
                return x

            return x, next_cache

        logits = self.to_logits(x)

        if not return_loss:
            if not return_cache:
                return logits

            return logits, next_cache

        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
