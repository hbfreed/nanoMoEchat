"""Interleave the cookbook SPT mix into the pre-tokenized ClimbMix stream at document level.

build_cookbook_mix.py + pretokenize.py place each corpus copy as one contiguous mix shard
between two ClimbMix shards, and the loader reads the stream in order. Each copy therefore
arrives as a ~10M-token burst (~75 steps at a 131k batch) of pure cooking + Dolci. This script
rewrites the token stream so every mix document sits at its own random document boundary
inside the ClimbMix shards, giving a steady ~3% density instead of bursts.

It works directly on the .bin/.idx token files (numpy copies, no re-tokenizing):
  - the documents come from the existing tokenized mix shards (shard_*_mix*.bin), one copy
    each, with every copy keeping its own Dolci slice;
  - copy k is spread over the k-th of K equal strata of the first --span-tokens ClimbMix
    tokens, so repeats of the same document stay ~span/K apart rather than clumping;
  - inside its stratum each document lands at a uniformly random position, snapped forward
    to the next document boundary, so no ClimbMix document is split;
  - mix documents longer than --max-doc-tokens (whole cookbooks run ~150k tokens, more than a
    131k-token step) are cut into pieces that are placed independently, each later piece
    getting its own BOS. Training sees 1024-token windows, so no usable context is lost, and
    no single step ends up dominated by one book.

The output is a separate base dir (the source is never modified, so runs reading it are
unaffected):
    <out>/base_data_climbmix/   symlinks to the ClimbMix parquet names (no mix shards);
                                the loader only lists names, and the last one stays val
    <out>/tokens/<tokenizer>/   rewritten train shards; untouched ones and val are symlinks
    <out>/<tokenizer>           symlink to the tokenizer
Point a run at it with NANOCHAT_BASE_DIR=<out> (or symlink those three into a run dir).

Run with:
python -m scripts.interleave_mix --out=/media/henry/MoreFiles/interleaved
"""

import argparse
import os

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--src", default="/media/henry/MoreFiles", help="source base dir")
ap.add_argument("--out", required=True, help="output base dir (created)")
ap.add_argument("--tokenizer", default="tokenizer_qwen49152")
ap.add_argument("--span-tokens", type=float, default=0,
                help="spread the mix over the first N ClimbMix train tokens (0: all of them). "
                     "Match the run's budget so every copy is seen.")
ap.add_argument("--start-tokens", type=float, default=0,
                help="no mix before this many ClimbMix tokens (e.g. skip warmup)")
ap.add_argument("--max-doc-tokens", type=int, default=4096,
                help="cut longer mix documents into pieces of at most this many tokens (0: never)")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--dry-run", action="store_true", help="plan and report, write nothing")
args = ap.parse_args()

src_tok = os.path.join(args.src, "tokens", args.tokenizer)
src_parquet = os.path.join(args.src, "base_data_climbmix")
out_tok = os.path.join(args.out, "tokens", args.tokenizer)
out_parquet = os.path.join(args.out, "base_data_climbmix")
assert os.path.realpath(args.out) != os.path.realpath(args.src), "--out must differ from --src"


def load_idx(stem):
    return np.fromfile(os.path.join(src_tok, stem + ".idx"), dtype=np.int64)


# The stream layout the loader would see without the mix shards: parquet names sorted, the
# last is val, train is the tokenized prefix of the rest (nanochat.dataloader.TokenStream).
names = sorted(f.removesuffix(".parquet") for f in os.listdir(src_parquet)
               if f.endswith(".parquet") and "_mix" not in f)
val_stem, train_names = names[-1], names[:-1]
train = []
for stem in train_names:
    if not os.path.exists(os.path.join(src_tok, stem + ".idx")):
        break
    train.append(stem)
mix = sorted(f.removesuffix(".idx") for f in os.listdir(src_tok)
             if "_mix" in f and f.endswith(".idx"))
assert train and mix, f"nothing to do: {len(train)} ClimbMix shards, {len(mix)} mix shards"

idxs = [load_idx(s) for s in train]
lens = np.array([int(i[-1]) for i in idxs], dtype=np.int64)
ends = np.cumsum(lens)
total = int(ends[-1])
span = int(args.span_tokens) or total
start = int(args.start_tokens)
assert 0 <= start < span <= total, f"need 0 <= start < span <= {total:,} ClimbMix tokens"
bos = int(np.fromfile(os.path.join(src_tok, train[0] + ".bin"), dtype=np.uint16, count=1)[0])
dtype = np.uint16 if os.path.getsize(os.path.join(src_tok, train[0] + ".bin")) == 2 * lens[0] else np.uint32

def pieces(midx):
    """(start, end, add_bos) slices of a mix shard: whole documents, long ones cut up."""
    out, cap = [], args.max_doc_tokens
    for a, z in zip(midx[:-1].tolist(), midx[1:].tolist()):
        if not cap or z - a <= cap:
            out.append((a, z, False))
            continue
        out.append((a, a + cap, False))  # the first piece keeps the document's own BOS
        out += [(p, min(p + cap - 1, z), True) for p in range(a + cap, z, cap - 1)]
    return out


# Plan: inserts[i] lists (boundary doc index within shard i, mix shard k, piece j of it).
rng = np.random.default_rng(args.seed)
K = len(mix)
inserts = [[] for _ in train]
mix_pieces = []
mix_tokens = 0
for k, stem in enumerate(mix):
    mix_pieces.append(pieces(load_idx(stem)))
    n = len(mix_pieces[k])
    mix_tokens += sum(z - a + bos_ for a, z, bos_ in mix_pieces[k])
    lo = start + (span - start) * k // K
    hi = start + (span - start) * (k + 1) // K
    pos = rng.integers(lo, hi, size=n)
    shard = np.searchsorted(ends, pos, side="right")
    local = pos - (ends[shard] - lens[shard])
    for j in rng.permutation(n):  # random order among documents that share a boundary
        i = int(shard[j])
        b = int(np.searchsorted(idxs[i], local[j], side="left"))  # next boundary at or after
        inserts[i].append((b, k, int(j)))

touched = sum(1 for x in inserts if x)
print(f"ClimbMix train: {len(train)} shards, {total/1e9:.3f}B tokens; val {val_stem}")
print(f"mix: {K} copies, {sum(map(len, mix_pieces)):,} pieces, {mix_tokens/1e6:.0f}M tokens, spread over ClimbMix tokens "
      f"{start/1e9:.2f}B-{span/1e9:.2f}B ({touched} shards rewritten)")
density = mix_tokens / (span - start + mix_tokens)
print(f"density inside the span: {density*100:.2f}% of the stream")
if args.dry_run:
    raise SystemExit

# Layout: parquet-name symlinks (train prefix + val), tokenizer symlink, token dir.
os.makedirs(out_parquet, exist_ok=True)
os.makedirs(out_tok, exist_ok=True)


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


for stem in train + [val_stem]:
    link(os.path.join(src_parquet, stem + ".parquet"), os.path.join(out_parquet, stem + ".parquet"))
link(os.path.join(args.src, args.tokenizer), os.path.join(args.out, args.tokenizer))
for suffix in (".bin", ".idx"):
    link(os.path.join(src_tok, val_stem + suffix), os.path.join(out_tok, val_stem + suffix))

mix_bins = [np.memmap(os.path.join(src_tok, s + ".bin"), dtype=dtype, mode="r") for s in mix]

written = 0
for i, stem in enumerate(train):
    dst = os.path.join(out_tok, stem)
    if not inserts[i]:
        for suffix in (".bin", ".idx"):
            link(os.path.join(src_tok, stem + suffix), dst + suffix)
        written += int(lens[i])
        continue
    for suffix in (".bin", ".idx"):  # a rerun replaces earlier output, symlink or file
        if os.path.lexists(dst + suffix):
            os.remove(dst + suffix)
    idx = idxs[i]
    src_bin = np.memmap(os.path.join(src_tok, stem + ".bin"), dtype=dtype, mode="r")
    ins = sorted(inserts[i], key=lambda t: t[0])  # stable: keeps the random order per boundary
    doc_lens = np.diff(idx)
    ins_b = np.array([b for b, _, _ in ins])
    ins_len = np.array([z - a + add for a, z, add in (mix_pieces[k][j] for _, k, j in ins)])
    new_lens = np.insert(doc_lens, ins_b, ins_len)
    new_idx = np.concatenate([[0], np.cumsum(new_lens)]).astype(np.int64)
    with open(dst + ".bin.tmp", "wb") as f:
        prev = 0
        for b, k, j in ins:
            src_bin[idx[prev]:idx[b]].tofile(f)
            a, z, add = mix_pieces[k][j]
            if add:
                np.array([bos], dtype=dtype).tofile(f)
            mix_bins[k][a:z].tofile(f)
            prev = b
        src_bin[idx[prev]:idx[-1]].tofile(f)
    assert os.path.getsize(dst + ".bin.tmp") == new_idx[-1] * np.dtype(dtype).itemsize
    new_idx.tofile(dst + ".idx.tmp")
    os.rename(dst + ".bin.tmp", dst + ".bin")
    os.rename(dst + ".idx.tmp", dst + ".idx")  # .idx last: it marks the shard done
    out_bin = np.memmap(dst + ".bin", dtype=dtype, mode="r")
    assert (out_bin[new_idx[:-1]] == bos).all(), f"{stem}: a document does not start with BOS"
    written += int(new_idx[-1])
    if (i + 1) % 20 == 0 or i == len(train) - 1:
        print(f"  {i+1}/{len(train)} shards, {written/1e9:.2f}B tokens", flush=True)

assert written == total + mix_tokens, (written, total, mix_tokens)
print(f"done: {written/1e9:.3f}B train tokens in {out_tok}")
