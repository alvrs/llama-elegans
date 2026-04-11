# LLaMA elegans

# Embed
# RMSnorm
#
# Decoder layer
#   RMSnorm
#
#   Attention
#     Q, K, V, O
#     RoPE (Q, K)
#     Split into attention heads
#       attn = Q @ K
#       causal mask
#       softmax
#       attn @ V
#     Concat heads
#     attn_v @ O 
#
#   Residual add
#   RMSnorm
#
#   MLP
#     Up
#     Gate
#     SiLU
#     Gate @ Up
#     Down
#   Residual add
#
# RMSnorm
# lm_head

import torch as t
from torch import nn
from dataclasses import dataclass

@dataclass
class Config:
    hidden_size: int = 256
    heads: int = 4
    kv_heads: int = 1
    intermediate: int = 1024
    head_dim = 64

class RMSnorm(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gamma = nn.Parameter(t.ones(config.hidden_size))

    def forward(self, x):
        assert len(x.shape) == 3 # batch, seq, hidden
        s = x ** 2
        ms = s.mean(dim=-1, keepdim=True)
        rms = ms ** 0.5
        return (x / rms) * self.gamma

class LlamaElegans(nn.Module):

    def forward(self, x):
        return x
