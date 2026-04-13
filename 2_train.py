import constants
import numpy as np
import einops
import torch
import argparse
import math
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from dataclasses import dataclass
from tqdm import tqdm
from model import LlamaElegans, Config as ModelConfig

torch.manual_seed(42)

parser = argparse.ArgumentParser(description="Train a LlamaElegans model")
parser.add_argument("--out", type=str, default="out/model.pt", help="Final model output file")
parser.add_argument("--batch_size", type=int, default=64, help="Sequences per batch")
parser.add_argument("--seq_len", type=int, default=1024, help="Length of each sequence in the batch")
parser.add_argument("--steps", type=int, default=50_000, help="Number of steps to train for")
args = parser.parse_args()
print(f"{args.out=}")
print(f"{args.batch_size=}")
print(f"{args.seq_len=}")
print(f"{args.steps=}")

@dataclass
class Config:
    batch_size: int
    seq_len: int
    steps: int

config = Config(
    batch_size=args.batch_size,
    seq_len=args.seq_len,
    steps=args.steps
) 
train_dataset = torch.from_numpy(np.memmap(constants.train_dataset_path, dtype=np.uint16))
val_dataset = torch.from_numpy(np.memmap(constants.val_dataset_path, dtype=np.uint16))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device=}")

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

warmup_steps = config.steps // 100
min_ratio = 0.1

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (config.steps - warmup_steps)
    return min_ratio + (1-min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = LambdaLR(optimizer, lr_lambda)

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
    scheduler.step()
    optimizer.zero_grad()

    if i % 100 == 0:
       tqdm.write(f"step: {i:>5d} | train {loss.item():.4f}")
    
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
            tqdm.write(f"step: {i:>5d} | val {val_loss:.4f}")
        model.train()
    
    if i % 5000 == 0 and i > 0:
        torch.save({ "model": model.state_dict(), "config": model_config }, f"out/chkt_{i}.pt")


torch.save({ "model": model.state_dict(), "config": model_config }, args.out)
