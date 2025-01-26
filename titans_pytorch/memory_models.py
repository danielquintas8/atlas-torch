import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter, ParameterList

from einops import rearrange

# functions

def l2norm(t):
    return F.normalize(t, dim = -1)

# norms

class LayerNorm(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.gamma = Parameter(torch.zeros(dim))

    def forward(self, x):
        gamma = self.gamma

        if gamma.ndim == 2:
            gamma = rearrange(gamma, 'b d -> b 1 d')

        return self.ln(x) * (gamma + 1.)

# memory mlp proposed in TTT

class MemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth
    ):
        super().__init__()
        self.weights = ParameterList([Parameter(torch.randn(dim, dim)) for _ in range(depth)])

        self.ln = LayerNorm(dim)

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(
        self,
        x
    ):
        residual = x

        for ind, weight in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x)

            x = x @ weight

        return self.ln(x) + residual

# memory mlp, but with gated residual + final projection

class GatedResidualMemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        expansion_factor = 4.
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)

        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, dim_hidden)),
                Parameter(torch.randn(dim_hidden, dim)),
                Parameter(torch.randn(dim * 2, dim)),
            ]) for _ in range(depth)
        ])

        self.final_proj = Parameter(torch.randn(dim, dim))

        self.ln = LayerNorm(dim)

        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(
        self,
        x
    ):
        residual = x

        for weight1, weight2, to_gates in self.weights:
            res = x

            hidden = x @ weight1
            hidden = F.gelu(hidden)
            branch_out = hidden @ weight2

            # gated residual

            gates = cat((branch_out, res), dim = -1) @ to_gates
            x = res.lerp(branch_out, gates.sigmoid())

        out = x @ self.final_proj

        return self.ln(out) + residual

# memory mlp with factorized weights
# so can tradeoff capacity for smaller chunk sizes

class FactorizedMemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        k = 32
    ):
        super().__init__()
        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, k)),
                Parameter(torch.randn(k, dim)),
            ]) for _ in range(depth)
        ])

        self.ln = LayerNorm(dim)

        for weight1, weight2 in self.weights:
            nn.init.xavier_uniform_(weight1)
            nn.init.xavier_uniform_(weight2)

    def forward(
        self,
        x
    ):
        residual = x

        for ind, (weight1, weight2) in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x)

            x = x @ weight1 @ weight2

        return self.ln(x) + residual

# improvised attention as memory module

class MemoryAttention(Module):
    def __init__(
        self,
        dim,
        scale = 8.,
        expansion_factor = 2.
    ):
        super().__init__()
        self.scale = scale
        dim_ff_hidden = int(dim * expansion_factor)

        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim)), # queries
            nn.Parameter(torch.randn(dim, dim)), # keys
            nn.Parameter(torch.randn(dim, dim)), # values
            nn.Parameter(torch.randn(dim, dim_ff_hidden)), # ff w1
            nn.Parameter(torch.randn(dim_ff_hidden, dim)), # ff w2
        ])

        self.ln = LayerNorm(dim)

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x):
        residual = x

        wq, wk, wv, ffw1, ffw2 = self.weights

        q = l2norm(x @ wq)
        k = l2norm(x @ wk)
        v = x @ wv

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            scale = self.scale,
            is_causal = True
        )

        # parallel attention + feedforward block
        # as in PaLM + Gpt-J

        h = F.gelu(x @ ffw1)
        ff_out = h @ ffw2

        return self.ln(attn_out + ff_out) + residual
