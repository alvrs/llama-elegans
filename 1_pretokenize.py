"""
Pre-tokenize a dataset for higher efficiency during training
"""
import argparse
import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_dataset
from tokenizer import HuggingFaceTokenizer

parser = argparse.ArgumentParser(description='Pre-tokenize a dataset')
parser.add_argument('--max-docs', type=int, default=None, help='Maximum docs to pretokenize (default: all)')
args = parser.parse_args()
print(f"max_docs: {args.max_docs}")

tokenizer = HuggingFaceTokenizer.from_directory("./out/tokenizer")
ds = load_dataset("roneneldan/TinyStories", split="train")
assert isinstance(ds, Dataset), f"Expected a Dataset, got {type(ds).__name__}"
if args.max_docs is not None:
  ds = ds.select(range(min(args.max_docs, len(ds))))

# Tokenize docs
ids: list[list[int]] = []
for doc in tqdm(ds):
  assert isinstance(doc, dict)
  text = doc["text"]
  assert isinstance(text, str)
  encoded = tokenizer.encode(text, prepend="<|bos|>")
  assert max(encoded) < 2**16
  ids.append(encoded)

n_val = 1000 if len(ids) > 1000 else 1
print(f"train docs: {len(ids)-n_val:,}, val docs: {n_val:,}")
train_ids = np.concatenate([np.array(doc, dtype=np.uint16) for doc in ids[:-n_val]])
val_ids = np.concatenate([np.array(doc, dtype=np.uint16) for doc in ids[-n_val:]])

print(f"train tokens: {train_ids.size:,}, val tokens: {val_ids.size:,}")
print(f"train bytes: {train_ids.nbytes:,}, val size: {val_ids.nbytes:,}")

train_ids.tofile("./out/tiny_stories_train.bin")
val_ids.tofile("./out/tiny_stories_val.bin")
