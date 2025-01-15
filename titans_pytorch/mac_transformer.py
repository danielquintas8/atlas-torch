from __future__ import annotations
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops.layers.torch import Rearrange

from hyper_connections import get_init_and_expand_reduce_stream_functions

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

        dim_inner = dim_head * heads

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(self, x):
        batch, seq_len = x.shape[:2]

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v, is_causal = True)

        out = self.merge_heads(out)

        return self.to_out(out)

# MAC transformer

class MemoryAsContextTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        segment_len,
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
            attn = SegmentedAttention(dim = dim, dim_head = dim_head, heads = heads, segment_len = segment_len)
            ff = FeedForward(dim = dim, mult = ff_mult)

            self.layers.append(ModuleList([
                init_hyper_conn(dim = dim, branch = attn),
                init_hyper_conn(dim = dim, branch = ff)
            ]))

        self.norm = nn.RMSNorm(dim)

        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

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
        segment_len = 128,
    )

    x = torch.randint(0, 256, (1, 1023))

    logits = transformer(x)
