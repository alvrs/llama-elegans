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
import einops

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
    max_seq_len: int = 2048

def get_rope_angles(config: Config):
    n_pairs = config.head_dim // 2
    pairs = torch.arange(n_pairs)
    freq = 1 / (10_000 ** (pairs / n_pairs))
    pos = torch.arange(config.max_seq_len)
    angles = torch.outer(pos, freq)
    return angles

def apply_rope(x: torch.Tensor, cos_table: torch.Tensor, sin_table: torch.Tensor):
    assert len(x.shape) == 4 # (batch, heads, seq, head_dim)
    seq = x.size(2)
    cos = cos_table[:seq, :][None, None, :, :]
    sin = sin_table[:seq, :][None, None, :, :]
    pairs = einops.rearrange(x, "b h s (pairs i) -> b h s pairs i", i=2)
    x_new = pairs[..., 0] * cos - pairs[..., 1] * sin
    y_new = pairs[..., 0] * sin + pairs[..., 1] * cos 
    result = torch.stack([x_new, y_new], dim=-1)
    return einops.rearrange(result, "b h s pairs i -> b h s (pairs i)")

class RMSnorm(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, x):
        assert len(x.shape) == 3 # (batch, seq, hidden)
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
        assert len(x.shape) == 3 # (batch, seq, hidden)
        up = self.up_proj(x)
        gate = F.silu(self.gate_proj(x))
        intermediate = gate * up
        return self.down_proj(intermediate)

class Attention(nn.Module):
    causal_mask: torch.Tensor

    def __init__(self, config: Config):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.heads * config.head_dim, config.hidden_size, bias=False)
        self.head_dim = config.head_dim
        self.heads = config.heads
        self.kv_heads = config.kv_heads
        causal_mask = torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1) * -torch.inf
        self.register_buffer('causal_mask', causal_mask)
    
    def forward(self, x, cos, sin):
        assert len(x.shape) == 3 # (batch, seq, hidden)
        assert len(cos.shape) == 2 # (seq, pairs)
        assert len(sin.shape) == 2 # (seq, pairs)
        assert cos.size(0) == x.size(1)
        assert sin.size(0) == x.size(1)

        batch, seq, hidden = x.shape

        Q = self.q_proj(x) # (batch, seq, heads * head_dim)
        K = self.k_proj(x) # (batch, seq, kv_heads * head_dim)
        V = self.v_proj(x) # (batch, seq, kv_heads * head_dim)

        q_heads = einops.rearrange(Q, "b s (h d) -> b h s d", h=self.heads) # (batch, heads, seq, head_dim)
        k_heads = einops.rearrange(K, "b s (k d) -> b k s d", k=self.kv_heads) # (batch, kv_heads, seq, head_dim)
        v_heads = einops.rearrange(V, "b s (v d) -> b v s d", v=self.kv_heads) # (batch, kv_heads, seq, head_dim)

        kv_repeat = self.heads // self.kv_heads
        k_heads = einops.repeat(k_heads, "b k s d -> b (r k) s d", r=kv_repeat) # (batch, heads, seq, head_dim)
        v_heads = einops.repeat(v_heads, "b v s d -> b (r v) s d", r=kv_repeat) # (batch, heads, seq, head_dim)

        q_heads = apply_rope(q_heads, cos_table=cos, sin_table=sin) # (batch, heads, seq, head_dim) 
        k_heads = apply_rope(k_heads, cos_table=cos, sin_table=sin) # (batch, heads, seq, head_dim)

        scale = 1 / (self.head_dim ** 0.5)
        qk = q_heads @ k_heads.transpose(-2, -1) * scale # (batch, heads, seq, seq)

        qk = qk + self.causal_mask[:seq, :seq] # (batch, heads, seq, seq)

        attn = F.softmax(qk, dim=-1)

        attn_v = attn @ v_heads # (batch, heads, seq, head_dim)
        attn_v = einops.rearrange(attn_v, "b h s d -> b s (h d)") # (batch, seq, heads * head_dim)

        out = self.o_proj(attn_v) # (batch, seq, hidden)
        assert out.shape == (batch, seq, hidden)

        return out

class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.pre_attn_norm = RMSnorm(config)
        self.attention = Attention(config)
        self.post_attn_norm = RMSnorm(config)
        self.mlp = MLP(config)
    
    def forward(self, x, cos, sin):
        residual = x
        x = self.pre_attn_norm(x)
        x = self.attention(x, cos, sin)
        x = x + residual

        residual = x
        x = self.post_attn_norm(x)
        x = self.mlp(x)
        x = x + residual
        return x 

class LlamaElegans(nn.Module):
    cos_table: torch.Tensor
    sin_table: torch.Tensor

    def __init__(self, config: Config):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.layer = Decoder(config)
        self.norm = RMSnorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        angles = get_rope_angles(config)
        self.register_buffer('cos_table', torch.cos(angles)) 
        self.register_buffer('sin_table', torch.sin(angles))

    def forward(self, x):
        assert len(x.shape) == 2 # (batch, seq)
        seq = x.size(1)

        x = self.embed(x)

        cos = self.cos_table[:seq]
        sin = self.sin_table[:seq]
        x = self.layer(x, cos, sin)

        x = self.norm(x)
        return self.lm_head(x)
