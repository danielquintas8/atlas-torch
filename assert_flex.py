import torch
from titans_pytorch.mac_transformer import SegmentedAttention

attn = SegmentedAttention(
    dim = 512,
    segment_len = 32,
    num_persist_mem_tokens = 1,
    num_longterm_mem_tokens = 1,
    use_flex_attn = True
).cuda()

seq = torch.randn(1, 1019, 512).cuda()

out_flex, _ = attn(seq)
out_non_flex, _ = attn(seq, disable_flex_attn = True)

assert torch.allclose(out_flex, out_non_flex, atol = 1e-5)
