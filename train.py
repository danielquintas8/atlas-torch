import random
import tqdm
import gzip
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from local_attention import LocalTransformer

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from titans_pytorch.titans import NeuralMemory

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 512

PROJECT_NAME = 'titans-neural-memory'
WANDB_ONLINE = True # turn this on to pipe experiment to cloud
GLOBAL_LAYERS = (4, 5)
USE_TITANS_MEMORY = True
NEURAL_MEMORY_DEPTH = 2
WINDOW_SIZE = 64
RUN_NAME = 'neural memory'

# wandb experiment tracker

import wandb
wandb.init(project = PROJECT_NAME, mode = 'offline' if not WANDB_ONLINE else 'online')
wandb.run.name = RUN_NAME
wandb.run.save()

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model

titans_neural_memory = NeuralMemory(
    dim = 384,
    chunk_size = WINDOW_SIZE,
    pre_rmsnorm = True,
    post_rmsnorm = True,
    default_mlp_kwargs = dict(
        depth = NEURAL_MEMORY_DEPTH
    )
)

linear_attn = TaylorSeriesLinearAttn(
    dim = 384,
    dim_head = 16,
    heads = 16,
    causal = True,
    prenorm = True
)

model = LocalTransformer(
    num_tokens = 256,
    dim = 384,
    depth = 8,
    causal = True,
    local_attn_window_size = WINDOW_SIZE,
    max_seq_len = SEQ_LEN,
    global_attn_layer = linear_attn if not USE_TITANS_MEMORY else titans_neural_memory,
    layers_insert_global_attn = GLOBAL_LAYERS
).cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
    data_train, data_val = np.split(data, [int(90e6)])
    data_train, data_val = map(torch.from_numpy, (data_train, data_val))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# optimizer

optim = Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader), return_loss = True)
        loss.backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()
    wandb.log(dict(loss = loss.item()))

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader), return_loss = True)
            print(f'validation loss: {loss.item()}')

    if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(inp[None, ...], GENERATE_LENGTH, use_kv_cache = False)
        output_str = decode_tokens(sample[0])
        print(output_str)
