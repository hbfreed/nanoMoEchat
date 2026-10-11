"""
Is the Qwen tokenizer (248k vocab) worth its output-layer cost, and can it be trimmed?

Reports, on our pretraining data:
1. compression: bytes per token for our tokenizer vs Qwen's
2. coverage: Qwen token frequencies counted on train shards; for each kept-vocab
   size K (the K most frequent ids + all special tokens), the fraction of val
   tokens inside the kept set
3. teacher mass: a Qwen teacher's probability mass outside each kept set on val
   text, and the distillation signal lost by renormalizing over it,
   E[-log mass_in] = KL(teacher restricted+renormalized || teacher)

Run with:
python -m dev.qwen_vocab_stats --teacher=Qwen/Qwen3.5-0.8B-Base
"""

import argparse
import json
import math
import time

import numpy as np
import torch

from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

parser = argparse.ArgumentParser(description="Qwen vocab compression / coverage / teacher-mass stats")
parser.add_argument("--teacher", type=str, default="Qwen/Qwen3.5-0.8B-Base")
parser.add_argument("--count-bytes", type=float, default=3e8, help="train text bytes for token frequencies")
parser.add_argument("--val-bytes", type=float, default=2e7, help="val text bytes for compression / coverage")
parser.add_argument("--keep-sizes", type=str, default="32768,49152,65536,98304,131072")
parser.add_argument("--teacher-seqs", type=int, default=64)
parser.add_argument("--seq-len", type=int, default=2048)
parser.add_argument("--json", action="store_true", help="also print one JSON line of results")
args = parser.parse_args()

from transformers import AutoTokenizer
import pyarrow.parquet as pq

qtok = AutoTokenizer.from_pretrained(args.teacher)
V = len(qtok)
special_ids = sorted(set(qtok.all_special_ids) | set(qtok.added_tokens_decoder))
doc_sep = qtok.convert_tokens_to_ids("<|endoftext|>")
keep_sizes = [int(k) for k in args.keep_sizes.split(",")]
print(f"Qwen tokenizer: {V:,} ids, {len(special_ids)} special/added | doc separator id {doc_sep}")


def read_texts(paths, budget):
    """Documents from the given shards, row group by row group, until `budget` bytes."""
    texts, nbytes = [], 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            for t in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                texts.append(t)
                nbytes += len(t.encode("utf-8"))
                if nbytes >= budget:
                    return texts, nbytes
    return texts, nbytes


def qwen_encode(texts):
    return qtok(texts, add_special_tokens=False)["input_ids"]


paths = list_parquet_files()
train_paths, val_path = paths[:-1], paths[-1]
results = {"teacher": args.teacher, "vocab": V}

# 1. compression
t0 = time.time()
val_texts, val_bytes = read_texts([val_path], args.val_bytes)
val_ids = qwen_encode(val_texts)
n_qwen = sum(len(x) for x in val_ids)
n_ours = sum(len(x) for x in get_tokenizer().encode(val_texts))
results["bytes_per_token"] = {"ours": val_bytes / n_ours, "qwen": val_bytes / n_qwen}
print(f"\n[compression] {len(val_texts):,} val docs, {val_bytes/1e6:.0f} MB ({time.time()-t0:.0f}s)")
print(f"  ours (65,536): {val_bytes/n_ours:.3f} bytes/token | qwen: {val_bytes/n_qwen:.3f} bytes/token "
      f"| qwen uses {100*(n_qwen/n_ours-1):+.1f}% tokens")

# 2. coverage
t0 = time.time()
# spread the count over many shards so no single source dominates the ranking
counts = np.zeros(V, dtype=np.int64)
per_shard = args.count_bytes / 32
counted = 0
for path in train_paths[:: max(1, len(train_paths) // 32)][:32]:
    texts, nbytes = read_texts([path], per_shard)
    for ids in qwen_encode(texts):
        counts += np.bincount(ids, minlength=V)
    counted += nbytes
order = np.argsort(-counts, kind="stable")
val_counts = np.zeros(V, dtype=np.int64)
for ids in val_ids:
    val_counts += np.bincount(ids, minlength=V)
print(f"\n[coverage] frequencies from {counted/1e6:.0f} MB of train text ({time.time()-t0:.0f}s); "
      f"{(counts > 0).sum():,} ids ever seen")
keep_masks = {}
results["coverage"] = {}
for k in keep_sizes:
    mask = np.zeros(V, dtype=bool)
    mask[order[:k]] = True
    mask[special_ids] = True
    keep_masks[k] = mask
    cov = val_counts[mask].sum() / val_counts.sum()
    results["coverage"][k] = cov
    print(f"  keep {k:>7,}: val tokens outside kept set {100*(1-cov):.4f}%")

# 3. teacher probability mass outside the kept set
from transformers import AutoModelForCausalLM

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).cuda().eval()
stream = []
for ids in val_ids:
    stream.append(doc_sep)
    stream.extend(ids)
    if len(stream) >= args.teacher_seqs * args.seq_len:
        break
seqs = torch.tensor(stream[: args.teacher_seqs * args.seq_len]).view(args.teacher_seqs, args.seq_len)
# the output layer is padded past the tokenizer's ids; padding ids are never kept
n_logits = model.get_output_embeddings().weight.size(0)
masks = {k: torch.from_numpy(np.pad(m, (0, n_logits - V))).cuda() for k, m in keep_masks.items()}
lost = {k: [] for k in keep_sizes}  # per position -log(mass in kept set)
nll_nats, n_pred, n_bytes = 0.0, 0, 0
with torch.no_grad():
    for seq in seqs:
        seq = seq.cuda()
        logp = model(seq[None]).logits[0, :-1].float().log_softmax(-1)
        tgt = seq[1:]
        nll_nats += -logp.gather(1, tgt[:, None]).sum().item()
        n_pred += tgt.numel()
        n_bytes += len(qtok.decode(tgt.tolist()).encode("utf-8"))
        for k, m in masks.items():
            lost[k].append(-torch.logsumexp(logp[:, m], dim=-1))
        del logp
print(f"\n[teacher] {args.teacher}: {args.teacher_seqs}x{args.seq_len} val tokens, "
      f"bpb {nll_nats/math.log(2)/n_bytes:.4f} ({time.time()-t0:.0f}s)")
results["teacher_bpb"] = nll_nats / math.log(2) / n_bytes
results["teacher_lost"] = {}
for k in keep_sizes:
    x = torch.cat(lost[k])
    mass_out = 1 - torch.exp(-x)
    stats = {"mean_mass_out": mass_out.mean().item(),
             "p99_mass_out": mass_out.quantile(0.99).item(),
             "kl_nats_per_token": x.mean().item()}
    results["teacher_lost"][k] = stats
    print(f"  keep {k:>7,}: mass outside mean {stats['mean_mass_out']:.2e}, "
          f"p99 {stats['p99_mass_out']:.2e} | KL lost {stats['kl_nats_per_token']:.2e} nats/token")
if args.json:
    print("JSON " + json.dumps(results))
