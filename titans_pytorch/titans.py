from __future__ import annotations
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import Linear, Module
from torch.func import functional_call, vmap, grad

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

# main neural memory

class NeuralMemory(Module):
    def __init__(
        self,
        dim,
        model: Module | None = None
    ):
        super().__init__()

        if not exists(model):
            model = MLP(dim, depth = 4)

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        self.memory = model

        self.to_queries = LinearNoBias(dim, dim)
        self.to_keys_values = LinearNoBias(dim, dim * 2)

    def init_memories(self):
        init_memories = {param_name: param.clone().zero_() for param_name, param in self.memory.named_parameters()}
        return init_memories

    def retrieve_memories(
        self,
        seq,
        past_memories: dict[str, Tensor] | None = None
    ):
        queries = self.to_queries(seq)

        # the parameters of the memory model stores the memories of the key / values
        # when the MLP has only 1 weight matrix, it is equivalent to `kv` fast weight memories from linear attention literature / schmidhuber's paper

        curr_memories = dict(self.memory.named_parameters())

        if exists(past_memories):
            assert past_memories.keys() == curr_memories.keys()

            curr_memories = {param_name: (curr_memory + past_memory) for (param_name, curr_memory), (_, past_memory) in zip(curr_memories.items(), past_memories.items())}

        # fetch values from memory model

        values = functional_call(self.memory, curr_memories, queries)

        return values
