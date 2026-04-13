import constants
import numpy as np
import einops
import torch
import argparse
from torch.nn import functional as F
from dataclasses import dataclass
from tqdm import tqdm
from model import LlamaElegans, Config as ModelConfig

torch.manual_seed(42)

parser = argparse.ArgumentParser(description="Train a LlamaElegans model")
parser.add_argument("--out", type=str, default="out/model.pt", help="Final model output file")
args = parser.parse_args()
print(f"{args.out=}")

@dataclass
class Config:
    batch_size: int = 64
    seq_len: int = 1024
    steps: int = 1001

config = Config() 
train_dataset = torch.from_numpy(np.memmap(constants.train_dataset_path, dtype=np.uint16))
val_dataset = torch.from_numpy(np.memmap(constants.val_dataset_path, dtype=np.uint16))
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(dataset: torch.Tensor):
    indices = torch.randint(
        low=0,
        high=dataset.shape[0] - config.seq_len - 1,
        size=[config.batch_size]
    )
    offsets = torch.arange(config.seq_len+1)
    indices = indices[:, None] + offsets[None, :]
    tokens = dataset[indices]
    x = tokens[:, :-1].to(device)
    y = tokens[:, 1:].to(device)
    return (x,y)


model_config = ModelConfig()
model = LlamaElegans(model_config).to(device)
optimizer = torch.optim.AdamW(model.parameters())

for i in tqdm(range(config.steps)):
    x, y = get_batch(train_dataset)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
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
       print(f"step: {i:>5d} | train {loss.item():.4f}")
    
    if i % 1000 == 0:
        model.eval()
        with torch.no_grad():
            losses = []
            for _ in range(20):
                x, y = get_batch(val_dataset)
                logits = model(x.long())
                val_loss = F.cross_entropy(
                    einops.rearrange(logits, "batch seq vocab -> (batch seq) vocab"),
                    einops.rearrange(y.long(), "batch seq -> (batch seq)")
                )
                losses.append(val_loss.item())
            val_loss = sum(losses)/len(losses)
            print(f"val {val_loss:.4f}")
        model.train()

torch.save({ "model": model.state_dict(), "config": model_config }, args.out)
