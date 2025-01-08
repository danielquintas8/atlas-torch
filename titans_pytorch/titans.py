from __future__ import annotations
from functools import partial

import torch
from torch import nn
from torch.nn import Linear, Module
from torch.func import functional_call, grad

from einops import rearrange

# constants

LinearNoBias = partial(Linear, bias = False)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

def MLP(dim, depth):
    layers = []

    for i in range(depth):
        is_first = i == 0

        if not is_first:
            layers.append(nn.SiLU())

        layers.append(LinearNoBias(dim, dim))

    return nn.Sequential(*layers)

class NeuralMemory(Module):
    def __init__(
        self,
        dim,
        model: Module | None = None
    ):
        super().__init__()

        if not exists(model):
            model = MLP(dim, depth = 4)

        self.memory_model = model

        self.to_queries = LinearNoBias(dim, dim)
        self.to_keys_values = LinearNoBias(dim, dim * 2)

    def forward(
        self,
        seq
    ):
        queries = self.to_queries(seq)

        values = self.memory_model(queries)

        return values
