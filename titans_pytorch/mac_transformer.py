from __future__ import annotations
from typing import Callable
from math import ceil
from functools import partial

import tqdm

import torch
from torch import nn, cat
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

from einops import repeat, rearrange, pack, unpack
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
from hyper_connections import get_init_and_expand_reduce_stream_functions

# proposed neural memory

from titans_pytorch.titans import NeuralMemory

# constants

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def round_up_multiple(seq, mult):
    return ceil(seq / mult) * mult

def pack_with_inverse(t, pattern):
    packed, packed_shape = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        return unpack(out, packed_shape, default(inv_pattern, pattern))

    return packed, inverse

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pad_and_segment_with_inverse(seq, segment_len, fold_into_batch = True):
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
            out = rearrange(out, '(b w) n d -> b (w n) d', b = batch)

        if needs_pad:
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

    def forward_flex(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None
    ):

        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        batch, seq_len = seq.shape[:2]

        # attention

        seq = self.norm(seq)

        q, k, v = self.to_qkv(seq).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        # value residual

        orig_v = v

        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, 'kv h n d -> kv b h n d', b = batch)

        # relative positions

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # persistent memory

        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)

        # prep flex attention

        if not exists(flex_attn_fn):
            block_mask = create_mac_block_mask(seq_len, self.total_segment_len, self.num_persist_mem_tokens, self.sliding)

            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        # attention

        out = flex_attn_fn(q, k, v)

        out = self.merge_heads(out)

        out = self.to_out(out)

        return out, orig_v

    def forward(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        disable_flex_attn = False
    ):
        if seq.is_cuda and self.use_flex_attn and not disable_flex_attn:
            return self.forward_flex(seq, value_residual, flex_attn_fn)

        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        segment_len, num_longterm_mem_tokens = self.segment_len, self.num_longterm_mem_tokens
        total_segment_len = segment_len + num_longterm_mem_tokens

        batch, seq_len = seq.shape[:2]

        # auto pad to multiple

        seq, inverse_segment = pad_and_segment_with_inverse(seq, total_segment_len, fold_into_batch = False)

        # attention

        seq = self.norm(seq)

        q, k, v = self.to_qkv(seq).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        # value residual

        orig_v = v

        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        # relative positions

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

            idx = torch.arange(seq.shape[-2], device = seq.device)
            q_idx = rearrange(idx, '(w n) -> w n', n = total_segment_len)
            k_idx = pad_at_dim(q_idx, (1, 0), dim = 0, value = -1e4)
            k_idx = cat((k_idx[:-1], k_idx[1:]), dim = -1)

            q_idx = rearrange(q_idx, 'w i -> w i 1')
            k_idx = rearrange(k_idx, 'w j -> w 1 j')

            sliding_mask = (q_idx - k_idx) <= total_segment_len
            sliding_mask = F.pad(sliding_mask, (self.num_persist_mem_tokens, 0), value = True)

            sliding_mask = repeat(sliding_mask, 'w i j -> (b w) 1 i j', b = batch)
            attend_kwargs.update(mask = sliding_mask)

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b = k.shape[0])

        # persistent memory

        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)

        # attention

        out, _ = self.attend(q, k, v, **attend_kwargs)

        out = self.merge_heads(out)

        out = self.to_out(out)

        out = rearrange(out, '(b w) n d -> b (w n) d', b = batch)

        out = inverse_segment(out)

        return out, orig_v

# Attention + Neural Memory gating configuration, as depicted in Figure 2

class NeuralMemoryGatingWrapper(Module):
    def __init__(
        self,
        dim,
        attn: SegmentedAttention,
        neural_mem: NeuralMemory | None = None,
        gate_attn_output = True
    ):
        super().__init__()
        self.attn = attn
        self.neural_mem = neural_mem
        self.gate_attn_output = gate_attn_output

    def forward(
        self,
        seq,
        *args,
        **kwargs
    ):
        batch, seq_len = seq.shape[:2]
        mem = self.neural_mem

        if not exists(mem):
            return self.attn(seq, *args, **kwargs), 0.

        # initial retrieve, still should store first, it doesn't make sense not to, unless if all layers share the same neural memory

        retrieved, kv_aux_loss = mem(seq, return_aux_kv_loss = True)

        if not self.gate_attn_output:
            seq = seq + retrieved

        # attention

        attn_out, values = self.attn(seq, *args, **kwargs)

        if self.gate_attn_output:
            attn_out = attn_out * retrieved.sigmoid()

        return (attn_out, values), kv_aux_loss

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
        neural_mem_gate_attn_output = True,
        num_longterm_mem_tokens = 0,
        num_persist_mem_tokens = 0,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        num_residual_streams = 4,
        neural_memory_kwargs: dict = dict(),
        neural_memory_layers: tuple[int, ...] | None = None,
        aux_kv_recon_loss_weight = 0.,
        use_flex_attn = False,
        sliding_window_attn = False
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(dim = dim, num_axial_dims = 2)

        # long term mem tokens

        self.segment_len = segment_len

        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        has_longterm_mems = num_longterm_mem_tokens > 0

        self.longterm_mems = nn.Parameter(torch.randn(num_longterm_mem_tokens, dim) * 0.02)

        # maybe sliding window attn

        self.sliding_window_attn = sliding_window_attn

        # hyper conection

        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        self.layers = ModuleList([])

        self.neural_memory_segment_len = default(neural_memory_segment_len, num_longterm_mem_tokens + segment_len)

        layers = tuple(range(1, depth + 1))

        if not exists(neural_memory_layers):
            neural_memory_layers = layers if has_longterm_mems else ()

        assert not (num_longterm_mem_tokens > 0 and len(neural_memory_layers) == 0), 'empty `neural_memory_layers` when longterm memory tokens are present'

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

            if layer in neural_memory_layers:
                assert has_longterm_mems, '`num_longterm_mem_tokens` must be greater than 0'

                mem = NeuralMemory(
                    dim = dim,
                    chunk_size = self.neural_memory_segment_len,
                    **neural_memory_kwargs
                )

            attn = NeuralMemoryGatingWrapper(
                dim,
                attn = attn,
                neural_mem = mem,
                gate_attn_output = neural_mem_gate_attn_output
            )

            ff = FeedForward(dim = dim, mult = ff_mult)

            self.layers.append(ModuleList([
                init_hyper_conn(dim = dim, branch = attn),
                init_hyper_conn(dim = dim, branch = ff)
            ]))

        self.norm = nn.RMSNorm(dim)

        self.to_logits = LinearNoBias(dim, num_tokens)

        # auxiliary loss on kv recon

        self.has_aux_kv_recon_loss = aux_kv_recon_loss_weight > 0.
        self.aux_kv_recon_loss_weight = aux_kv_recon_loss_weight

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # flex attn related

        assert not (use_flex_attn and not exists(flex_attention)), 'you need to be on the latest pytorch with a cuda device available'
        self.use_flex_attn = use_flex_attn

        self.segment_len = segment_len
        self.num_persist_mem_tokens = num_persist_mem_tokens

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
        show_progress = True
    ):
        was_training = self.training
        self.eval()

        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        iter_wrap = tqdm.tqdm if show_progress else identity

        for _ in iter_wrap(range(sample_num_times)):
            logits = self.forward(out, disable_flex_attn = True)
            logits = logits[:, -1]

            logits = filter_fn(logits, **filter_kwargs)
            sample = gumbel_sample(logits, temperature = temperature)

            out = torch.cat((out, sample), dim = -1)

        self.train(was_training)

        return out[..., prompt_seq_len:]

    def forward(
        self,
        x,
        return_loss = False,
        return_loss_breakdown = False,
        disable_flex_attn = False
    ):

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # math

        batch, seq_len, neural_mem_segment_len, segment_len, num_longterm_mem_tokens = *x.shape, self.neural_memory_segment_len, self.segment_len, self.num_longterm_mem_tokens

        # token embedding

        x = self.token_emb(x)

        # intersperse longterm memory

        x, inverse_segment = pad_and_segment_with_inverse(x, segment_len)

        mems = repeat(self.longterm_mems, 'n d -> b n d', b = x.shape[0])
        x, inverse_pack_mems = pack_with_inverse((x, mems), 'b * d')

        x = inverse_segment(x)

        seq_len_with_mem = x.shape[-2]

        # apply axial positional embedding
        # so intra and inter segment can be more easily discerned by the network

        pos_emb = self.axial_pos_emb.forward_with_seq_len(seq_len_with_mem, (neural_mem_segment_len,))

        x = x + pos_emb

        # prep flex attention

        use_flex_attn = x.is_cuda and self.use_flex_attn and not disable_flex_attn

        flex_attn_fn = None

        if use_flex_attn:
            block_mask = create_mac_block_mask(seq_len_with_mem, segment_len + num_longterm_mem_tokens, self.num_persist_mem_tokens, self.sliding_window_attn)
            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        # value residual

        value_residual = None

        # aux losses

        kv_recon_losses = self.zero

        # expand and reduce streams for hyper connections

        x = self.expand_streams(x)

        for attn, ff in self.layers:

            (x, values), maybe_mem_kv_aux_loss = attn(
                x,
                value_residual = value_residual,
                disable_flex_attn = disable_flex_attn,
                flex_attn_fn = flex_attn_fn
            )

            kv_recon_losses = kv_recon_losses + maybe_mem_kv_aux_loss

            value_residual = default(value_residual, values)

            x = ff(x)

        x = self.reduce_streams(x)

        # excise out the memories

        x, inverse_segment = pad_and_segment_with_inverse(x, segment_len + num_longterm_mem_tokens)

        x, _ = inverse_pack_mems(x)

        x = inverse_segment(x)

        # to logits

        x = self.norm(x)

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        ar_loss = F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)

        losses = ar_loss

        if self.has_aux_kv_recon_loss:
            losses = losses + kv_recon_losses * self.aux_kv_recon_loss_weight

        if not return_loss_breakdown:
            return losses

        return losses, (ar_loss, kv_recon_losses)
