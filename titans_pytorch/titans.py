import torch
from torch.nn import Module
from einops import rearrange

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class NeuralMemory(Module):
    def __init__(self, dim):
        super().__init__()

    def forward(
        self,
        seq
    ):
        return seq
