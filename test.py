import torch
from model import Config, RMSnorm

config = Config()
norm = RMSnorm(config)
x = torch.randn(2, 8, config.hidden_size) * 50 + 20
out = norm(x)

print(out.shape) # should be (2, 8, 256)
print(x.mean(), x.std()) # original stats
print(out.mean(), out.std()) # should be roughly normalized 

rms = (out ** 2).mean(dim=-1).sqrt() # should be close to 1 for each pos
print(rms)
