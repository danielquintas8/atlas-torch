from __future__ import annotations
from functools import partial

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch.func import functional_call, vmap, grad_and_value

from einops import rearrange
from einops.layers.torch import Rearrange

# constants

LinearNoBias = partial(Linear, bias = False)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def round_down_multiple(seq, mult):
    return seq // mult * mult

# classes

class MLP(Module):
    def __init__(
        self,
        dim,
        depth
    ):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(dim, dim)) for _ in range(depth)])

    def forward(
        self,
        x
    ):

        for ind, weight in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.silu(x)

            x = x @ weight

        return x

# main neural memory

def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim = -1).sum()

class NeuralMemory(Module):
    def __init__(
        self,
        dim,
        model: Module | None = None,
        store_memory_loss_fn: Callable = default_loss_fn
    ):
        super().__init__()

        if not exists(model):
            model = MLP(dim, depth = 4)

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        # the memory is the weights of the model

        self.memory = model

        # prepare function for per sample gradients from model above, using torch.func

        def forward_and_loss(params, inputs, target):
            pred = functional_call(self.memory, params, inputs)
            loss = self.store_memory_loss_fn(pred, target) # simple mse loss in paper - eq (12) - |M(k) == v|²
            return loss

        self.per_sample_grad_and_value_fn = vmap(grad_and_value(forward_and_loss), in_dims = (None, 0, 0))

        # queries for retrieving from the model

        self.to_queries = LinearNoBias(dim, dim)

        # keys and values for storing to the model

        self.to_keys_values = LinearNoBias(dim, dim * 2)
        self.store_memory_loss_fn = store_memory_loss_fn

        # learned adaptive learning rate and momentum
        # todo - explore mlp layerwise learned lr / momentum

        self.to_adaptive_step = nn.Sequential(LinearNoBias(dim, 1), Rearrange('... 1 -> ...'))
        self.to_momentum = nn.Sequential(LinearNoBias(dim, 1), Rearrange('... 1 -> ...'))

    def init_memories(self):
        init_memories = {param_name: param.clone().zero_() for param_name, param in self.memory.named_parameters()}
        return init_memories

    def store_memories(
        self,
        seq,
        past_memories: dict[str, Tensor] | None = None
    ):

        curr_memories = dict(self.memory.named_parameters())

        if exists(past_memories):
            assert past_memories.keys() == curr_memories.keys()

            curr_memories = {param_name: (curr_memory + past_memory) for (param_name, curr_memory), (_, past_memory) in zip(curr_memories.items(), past_memories.items())}

        # pack batch and sequence dimension

        batch = seq.shape[0]

        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_momentum = self.to_momentum(seq)

        # keys and values

        seq = rearrange(seq, 'b n d -> (b n) d')
        keys, values = self.to_keys_values(seq).chunk(2, dim = -1)

        # get grads and extra auxiliary loss (for backwarding through qkv projection in base neural memory module)

        grads, aux_store_loss = self.per_sample_grad_and_value_fn(curr_memories, keys, values)

        # restore batch and sequence dimension

        grads = {name: rearrange(grad, '(b n) ... -> b n ...', b = batch) for name, grad in grads.items()}

        # accumulate gradients across time, without momentum and weight decay for starters

        grads = {name: grad.cumsum(dim = 1) for name, grad in grads.items()}

        # compute the next weight per batch

        next_memories = {name: param + grad[:, -1] for (name, param), (_, grad) in zip(curr_memories.items(), grads.items())}

        return grads, next_memories, aux_store_loss.sum()

    def retrieve_memories(
        self,
        seq,
        past_memories: dict[str, Tensor] | None = None
    ):
        # the parameters of the memory model stores the memories of the key / values
        # when the MLP has only 1 weight matrix, it is equivalent to `kv` fast weight memories from linear attention literature (recall fetching of memories is q @ (kv)) / schmidhuber's paper

        curr_memories = dict(self.memory.named_parameters())

        if exists(past_memories):
            assert past_memories.keys() == curr_memories.keys()

            curr_memories = {param_name: (curr_memory + past_memory) for (param_name, curr_memory), (_, past_memory) in zip(curr_memories.items(), past_memories.items())}

        # sequence Float['b n d'] to queries

        queries = self.to_queries(seq)

        # fetch values from memory model

        values = functional_call(self.memory, curr_memories, queries)

        return values
