from __future__ import annotations
import math
from functools import partial

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

from einops import repeat
from einops.layers.torch import Rearrange

from hyper_connections import get_init_and_expand_reduce_stream_functions

# constants

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def round_up_multiple(seq, mult):
    return math.ceil(seq / mult) * mult

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
        num_persist_mem_tokens,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

        dim_inner = dim_head * heads

        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.segment_len = segment_len

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.segment_seq = Rearrange('b (n w) d -> (b n) w d', n = segment_len)
        self.merge_seq_back = Rearrange('(b n) w d -> b (n w) d', n = segment_len)

        self.persistent_memory = nn.Parameter(torch.zeros(2, heads, num_persist_mem_tokens, dim_head))

    def forward(self, seq):
        batch, seq_len = seq.shape[:2]

        # auto pad to multiple
        # todo - get rid of logic with flex attention

        need_segment = seq_len >= self.segment_len

        if need_segment:
            next_seq_len = round_up_multiple(seq_len, self.segment_len)
            padding = next_seq_len - seq_len

            if padding > 0:
                seq = F.pad(seq, (0, 0, 0, padding))

            seq = self.segment_seq(seq)

        # attention

        seq = self.norm(seq)

        q, k, v = self.to_qkv(seq).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        # take care of persistent memory key / values

        pmk, pmv = tuple(repeat(t, 'h n d -> b h n d', b = seq.shape[0]) for t in self.persistent_memory)

        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)

        # sdpa

        out = F.scaled_dot_product_attention(q, k, v, is_causal = True)

        out = self.merge_heads(out)

        out = self.to_out(out)

        if need_segment:
            out = self.merge_seq_back(out)

        return out[:, :seq_len]

# MAC transformer

class MemoryAsContextTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        segment_len,
        num_persist_mem_tokens,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        num_residual_streams = 4
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)

        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        self.layers = ModuleList([])

        for _ in range(depth):
            attn = SegmentedAttention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                segment_len = segment_len,
                num_persist_mem_tokens = num_persist_mem_tokens
            )

            ff = FeedForward(dim = dim, mult = ff_mult)

            self.layers.append(ModuleList([
                init_hyper_conn(dim = dim, branch = attn),
                init_hyper_conn(dim = dim, branch = ff)
            ]))

        self.norm = nn.RMSNorm(dim)

        self.to_logits = LinearNoBias(dim, num_tokens)

    def forward(self, x):

        x = self.token_emb(x)

        x = self.expand_streams(x)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        x = self.reduce_streams(x)

        x = self.norm(x)

        return self.to_logits(x)

# main

if __name__ == '__main__':
    transformer = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 256,
        depth = 2,
        num_persist_mem_tokens = 16,
        segment_len = 128,
    )

    x = torch.randint(0, 256, (1, 1023))

    logits = transformer(x)
