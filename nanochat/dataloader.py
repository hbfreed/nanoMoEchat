from collections import deque

import torch

from nanochat.common import get_dist_info
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda"):
    """Stream pretraining text from parquet files, tokenize, yield training batches."""
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left

    # infinite iterator over document batches
    def document_batches():
        while True:
            # batch will iterate in group size of the parquet files, usually e.g. 1024 rows
            for batch in parquets_iter_batched(split=split, start=ddp_rank, step=ddp_world_size):
                # for the tokenizer we might want to go in usually smaller batches, e.g. 128 rows
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
    batches = document_batches()

    batch_index = 0
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
            batch_index += 1
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for faster transfers between CPU and GPU:
        scratch = torch.tensor(tokens, dtype=torch.int64, pin_memory=(device == "cuda"))
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(B, T).to(device=device, dtype=torch.int64, non_blocking=True)
        yield inputs, targets


class TokenStream:
    """The pre-tokenized shards of a split (scripts/pretokenize.py) as one contiguous,
    memory-mapped token stream. Shards are taken in parquet order up to the first
    one not yet tokenized, so a partially tokenized dataset is a usable prefix."""

    def __init__(self, split, tokenizer_name=None):
        import os
        import numpy as np
        from nanochat.common import get_base_dir
        from nanochat.dataset import list_parquet_files
        from nanochat.tokenizer import get_tokenizer, get_tokenizer_name
        name = tokenizer_name or get_tokenizer_name()
        token_dir = os.path.join(get_base_dir(), "tokens", name)
        paths = list_parquet_files()
        paths = paths[:-1] if split == "train" else paths[-1:]
        dtype = np.uint16 if get_tokenizer(name).get_vocab_size() <= 2**16 else np.uint32
        self.shards = []
        for path in paths:
            stem = os.path.join(token_dir, os.path.basename(path).removesuffix(".parquet"))
            if not os.path.exists(stem + ".idx"):
                break
            self.shards.append(np.memmap(stem + ".bin", dtype=dtype, mode="r"))
        assert self.shards, f"no pre-tokenized {split} shards in {token_dir}; run scripts/pretokenize.py"
        self.ends = np.cumsum([len(s) for s in self.shards])
        self.num_tokens = int(self.ends[-1])

    def read(self, start, n):
        """n tokens from stream position start (wrapping around at the end)."""
        import numpy as np
        out, filled = np.empty(n, dtype=np.int64), 0
        while filled < n:
            pos = (start + filled) % self.num_tokens
            i = int(np.searchsorted(self.ends, pos, side="right"))
            lo = pos - (int(self.ends[i - 1]) if i else 0)
            take = min(n - filled, len(self.shards[i]) - lo)
            out[filled : filled + take] = self.shards[i][lo : lo + take]
            filled += take
        return out


def pretokenized_data_loader(B, T, split, device="cuda", start_micro_step=0, tokenizer_name=None):
    """Like tokenizing_distributed_data_loader, from the pre-tokenized stream. Micro-step m
    on rank r reads the B*T(+1 target) tokens at ((m * world_size + r) * B*T), so the data
    order is a pure function of the step: resuming starts at start_micro_step directly."""
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    stream = TokenStream(split, tokenizer_name)
    n = B * T
    m = start_micro_step
    while True:
        tokens = torch.from_numpy(stream.read((m * ddp_world_size + ddp_rank) * n, n + 1))
        if device == "cuda":
            tokens = tokens.pin_memory()
        inputs = tokens[:-1].view(B, T).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = tokens[1:].view(B, T).to(device=device, dtype=torch.int64, non_blocking=True)
        yield inputs, targets
        m += 1
