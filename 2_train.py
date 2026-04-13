import constants
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class Config:
  batch_size: int = 16 
  seq_len: int = 16


config = Config() 
train_dataset = torch.from_numpy(np.memmap(constants.train_dataset_path, dtype=np.uint16))

def get_batch():
    indices = torch.randint(
        low=0,
        high=train_dataset.shape[0] - config.seq_len - 1,
        size=[config.batch_size]
    )
    offsets = torch.arange(config.seq_len+1)
    indices = indices[:, None] + offsets[None, :]
    tokens = train_dataset[indices]
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    return (x,y)

x,y = get_batch()
print(x.shape, y.shape)
