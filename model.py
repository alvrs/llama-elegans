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

import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class Config:
    hidden_size: int = 256
    heads: int = 4
    kv_heads: int = 1
    intermediate: int = 1024
    head_dim: int = 64
    vocab_size: int = 4096

class RMSnorm(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, x):
        assert len(x.shape) == 3 # batch, seq, hidden
        s = x ** 2
        ms = s.mean(dim=-1, keepdim=True)
        rms = ms ** 0.5
        return (x / rms) * self.gamma

class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
    
    def forward(self, x):
        # TODO: implement
        return x

class LlamaElegans(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.layer = Decoder(config)
        self.norm = RMSnorm(config)
        self.lm_head = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size, bias=False)

    def forward(self, x):
        assert len(x.shape) == 2 # batch, seq 
        x = self.embed(x)
        x = self.layer(x)
        x = self.norm(x)
        return self.lm_head(x)
