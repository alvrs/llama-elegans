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

from torch import nn
from dataclasses import dataclass

@dataclass
class Config:
    hidden_size: int = 256
    heads: int = 4
    kv_heads: int = 1
    intermediate: int = 1024
    head_dim = 64

class LlamaElegans(nn.Module):

    def forward(self, x):
        return x
