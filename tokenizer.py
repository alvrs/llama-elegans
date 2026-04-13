"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference

Adapted from https://github.com/karpathy/nanochat/blob/master/nanochat/tokenizer.py
"""

import os

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from typing import Iterable, overload

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer: HFTokenizer) -> None:
        self.tokenizer: HFTokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path: str) -> "HuggingFaceTokenizer":
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "HuggingFaceTokenizer":
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator: Iterable[str], vocab_size: int) -> "HuggingFaceTokenizer":
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self) -> list[str]:
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id: int) -> str | None:
        return self.tokenizer.id_to_token(id)

    def _encode_one(
        self,
        text: str,
        prepend: str | int | None = None,
        append: str | int | None = None,
        num_threads: int | None = None,
    ) -> list[int]:
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        # num_threads is ignored (only used by the nanochat Tokenizer for parallel encoding)
        assert isinstance(text, str)
        ids: list[int] = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            assert prepend_id is not None, f"Unknown special token: {prepend!r}"
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            assert append_id is not None, f"Unknown special token: {append!r}"
            ids.append(append_id)
        return ids

    def encode_special(self, text: str) -> int | None:
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self) -> int:
        # Different HuggingFace models use different BOS tokens and there is little consistency
        # 1) attempt to find a <|bos|> token
        bos = self.encode_special("<|bos|>")
        # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) if these fail, it's better to crash than to silently return None
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    @overload
    def encode(
        self,
        text: str,
        prepend: str | int | None = ...,
        append: str | int | None = ...,
        num_threads: int | None = ...,
    ) -> list[int]: ...
    @overload
    def encode(
        self,
        text: list[str],
        prepend: str | int | None = ...,
        append: str | int | None = ...,
        num_threads: int | None = ...,
    ) -> list[list[int]]: ...
    def encode(
        self,
        text: str | list[str],
        prepend: str | int | None = None,
        append: str | int | None = None,
        num_threads: int | None = None,
    ) -> list[int] | list[list[int]]:
        if isinstance(text, str):
            return self._encode_one(text, prepend=prepend, append=append, num_threads=num_threads)
        elif isinstance(text, list):
            return [
                self._encode_one(t, prepend=prepend, append=append, num_threads=num_threads)
                for t in text
            ]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, text: str | list[str]) -> list[int] | list[list[int]]:
        return self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir: str) -> None:
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")
