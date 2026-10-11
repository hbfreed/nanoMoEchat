"""
Pre-tokenize the pretraining parquet shards into flat token files, one per shard:

    <base_dir>/tokens/<tokenizer>/<shard>.bin   uint16 tokens (uint32 if vocab > 65536);
                                                each document is <bos> + its tokens
    <base_dir>/tokens/<tokenizer>/<shard>.idx   int64 document start offsets, plus the end

Shards keep the parquet order (nanochat.dataset.list_parquet_files), so the last
one is val as before. Training reads them as one contiguous memory-mapped stream
(nanochat.dataloader.pretokenized_data_loader): every token's position is fixed,
so runs resume exactly and teacher logits can be precomputed against it.
Finished shards are skipped, so rerunning resumes.

Run with:
python -m scripts.pretokenize --tokenizer=tokenizer_qwen49152 [--max-shards=N] [--workers=24]
"""

import argparse
import os
import time
from multiprocessing import Pool

parser = argparse.ArgumentParser(description="Pre-tokenize the pretraining shards")
parser.add_argument("--tokenizer", type=str, required=True, help="tokenizer dir under the base dir")
parser.add_argument("--max-shards", type=int, default=-1, help="train shards to do (-1: all); val is always done")
parser.add_argument("--workers", type=int, default=24)
args = parser.parse_args()
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # parallelism comes from the worker processes

import numpy as np
import pyarrow.parquet as pq

from nanochat.common import get_base_dir
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

out_dir = os.path.join(get_base_dir(), "tokens", args.tokenizer)
_tok = None


def tokenize_shard(path):
    global _tok
    if _tok is None:
        _tok = get_tokenizer(args.tokenizer)
    stem = os.path.join(out_dir, os.path.basename(path).removesuffix(".parquet"))
    if os.path.exists(stem + ".idx"):  # .idx is written last
        return 0, 0, True
    dtype = np.uint16 if _tok.get_vocab_size() <= 2**16 else np.uint32
    bos = _tok.get_bos_token_id()
    pf = pq.ParquetFile(path)
    chunks, starts, n = [], [], 0
    for rg in range(pf.num_row_groups):
        texts = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
        for ids in _tok.encode(texts, prepend=bos):
            starts.append(n)
            chunks.append(np.asarray(ids, dtype=dtype))
            n += len(ids)
    starts.append(n)
    for suffix, arr in ((".bin", np.concatenate(chunks)), (".idx", np.asarray(starts, dtype=np.int64))):
        with open(stem + suffix + ".tmp", "wb") as f:
            arr.tofile(f)
        os.rename(stem + suffix + ".tmp", stem + suffix)
    return n, len(starts) - 1, False


if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    paths = list_parquet_files()
    train, val = paths[:-1], paths[-1:]
    if args.max_shards >= 0:
        train = train[: args.max_shards]
    todo = val + train
    print(f"tokenizing {len(todo)} shards ({len(train)} train + val) with {args.tokenizer} into {out_dir}")
    t0, total, docs, skipped = time.time(), 0, 0, 0
    with Pool(args.workers) as pool:
        for i, (n, d, skip) in enumerate(pool.imap_unordered(tokenize_shard, todo), 1):
            total, docs, skipped = total + n, docs + d, skipped + skip
            if i % 25 == 0 or i == len(todo):
                dt = time.time() - t0
                print(f"  {i}/{len(todo)} shards ({skipped} already done) | {total/1e9:.2f}B tokens, "
                      f"{docs:,} docs | {total/dt/1e6:.1f}M tok/s | {dt/60:.1f} min", flush=True)
