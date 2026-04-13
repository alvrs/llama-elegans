import constants
import numpy as np
import einops
import torch
from torch.nn import functional as F
from dataclasses import dataclass
from tqdm import tqdm
from model import LlamaElegans, Config as ModelConfig

@dataclass
class Config:
  batch_size: int = 16 
  seq_len: int = 512
  steps: int = 1001

config = Config() 
train_dataset = torch.from_numpy(np.memmap(constants.train_dataset_path, dtype=np.uint16))
val_dataset = torch.from_numpy(np.memmap(constants.val_dataset_path, dtype=np.uint16))

def get_batch(dataset: torch.Tensor):
    indices = torch.randint(
        low=0,
        high=dataset.shape[0] - config.seq_len - 1,
        size=[config.batch_size]
    )
    offsets = torch.arange(config.seq_len+1)
    indices = indices[:, None] + offsets[None, :]
    tokens = dataset[indices]
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    return (x,y)


model_config = ModelConfig()
model = LlamaElegans(model_config)
optimizer = torch.optim.AdamW(model.parameters())

for i in tqdm(range(config.steps)):
    x, y = get_batch(train_dataset)
    logits = model(x.long())
    loss = F.cross_entropy(
       einops.rearrange(logits, "batch seq vocab -> (batch seq) vocab"),
       einops.rearrange(y.long(), "batch seq -> (batch seq)")
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    if i % 100 == 0:
       print(f"train {loss:4f}")
    
    if i % 1000 == 0:
        with torch.no_grad():
            losses = []
            for _ in range(20):
                x, y = get_batch(val_dataset)
                logits = model(x.long())
                loss = F.cross_entropy(
                    einops.rearrange(logits, "batch seq vocab -> (batch seq) vocab"),
                    einops.rearrange(y.long(), "batch seq -> (batch seq)")
                )
                losses.append(loss.item())
            loss = sum(losses)/len(losses)
            print(f"val {loss:4f}")
