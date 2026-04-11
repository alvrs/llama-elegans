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
import torch.nn.functional as F
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

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate, bias=False)
        self.down_proj = nn.Linear(config.intermediate, config.hidden_size, bias=False)

    def forward(self, x):
        assert len(x.shape) == 3 # batch, seq, hidden
        up = self.up_proj(x)
        gate = F.silu(self.gate_proj(x))
        intermediate = gate * up
        return self.down_proj(intermediate)

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
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x):
        assert len(x.shape) == 2 # batch, seq 
        x = self.embed(x)
        x = self.layer(x)
        x = self.norm(x)
        return self.lm_head(x)
