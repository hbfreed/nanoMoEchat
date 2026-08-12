"""Build the cookbook + Dolci SPT mix and interleave it into the ClimbMix stream.

Specialized pretraining a la DatologyAI's "Finetuner's Fallacy": mix the
domain corpus (Henry's 54 cookbooks + Cooking Issues transcripts) and a slice
of the eventual SFT distribution (Dolci-Instruct) into pretraining as a small
fraction of tokens, repeating the domain corpus diffusely (their Figure 1:
10-50x over the course of training). Diffuse repetition regularizes;
concentrated repetition (consecutive finetuning epochs) memorizes.

Mechanics: the pretraining dataloader walks parquet files in sorted filename
order with no shuffling, so WHERE a file's name sorts IS the mixing schedule.
This script writes `shard_00007_mix00.parquet`-style files: '.' < '_' in
ASCII, so `shard_00007.parquet` < `shard_00007_mix00.parquet` <
`shard_00008.parquet` -- each mix shard lands between two ClimbMix shards.
Repeats are spread evenly over `--span` gaps so the mix stays uniform across
however much of the stream the run consumes; keep span at or under the number
of shards your token budget will consume (a ClimbMix shard is ~60M tokens).

Each mix shard holds one full copy of the cookbook corpus (document order
reshuffled per repeat) plus a fresh, non-repeating slice of Dolci rendered as
plain-text dialogue (the chat special-token template is SFT's job; pretraining
just needs the distribution). Holdout books are excluded entirely and listed
at the end for the never-seen-cookbook eval.

Usage:
    python -m scripts.build_cookbook_mix                  # writes with defaults
    python -m scripts.build_cookbook_mix --repeats 33 --span 40
    python -m scripts.build_cookbook_mix --dry-run        # report only
"""

import argparse
import os
import random
import re

import pyarrow as pa
import pyarrow.parquet as pq

CORPUS_DIR = "/home/henry/Documents/PythonProjects/Cooking-Corpus/data"
TRANSCRIPTS_DIR = "/home/henry/Documents/cooking-issues-transcripts/labeled"
DATA_DIR = "/media/henry/MoreFiles/base_data_climbmix"

# Single-author books spanning distinct styles, held out entirely so a
# never-seen-cookbook quiz measures domain learning rather than memorization.
HOLDOUT_BOOKS = [
    "Amrikan",
    "The Noma Guide to Fermentation",
    "Guerrilla Tacos",
]

_XML_HEADER = re.compile(r"^<\?xml[^>]*\?>\s*$", re.MULTILINE)
_IMAGE_REF = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BLANK_RUNS = re.compile(r"\n{3,}")


def clean_chunk(text):
    text = _XML_HEADER.sub("", text)
    text = _IMAGE_REF.sub("", text)
    return text


def load_books():
    """Reassemble each book from its retrieval chunks, in filename order.

    Chunk filenames encode order (NNN-part_split_chunk) and consecutive
    splits are clean continuations (verified: no overlap), so sorted
    concatenation reconstructs the book text.
    """
    books = {}
    for book in sorted(os.listdir(CORPUS_DIR)):
        book_dir = os.path.join(CORPUS_DIR, book)
        if not os.path.isdir(book_dir):
            continue
        parts = []
        for f in sorted(os.listdir(book_dir)):
            if f.endswith(".md"):
                with open(os.path.join(book_dir, f), encoding="utf-8") as fh:
                    parts.append(clean_chunk(fh.read()))
        text = _BLANK_RUNS.sub("\n\n", "\n\n".join(parts)).strip()
        if text:
            books[book] = text
    return books


def load_transcripts():
    docs = []
    for f in sorted(os.listdir(TRANSCRIPTS_DIR)):
        if f.endswith(".txt"):
            with open(os.path.join(TRANSCRIPTS_DIR, f), encoding="utf-8") as fh:
                docs.append(fh.read().strip())
    return docs


def render_dialogue(messages):
    """Plain-text rendering for the pretraining slice. The chat special-token
    template is deliberately NOT used here: SPT mixes the distribution into
    pretraining; SFT later teaches the exact template."""
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        speaker = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
            role, role.capitalize()
        )
        lines.append(f"{speaker}: {content}")
    return "\n\n".join(lines)


def dolci_docs(num_docs, seed):
    """Stream a deterministic, non-repeating sample of Dolci-Instruct-SFT."""
    from datasets import load_dataset

    ds = load_dataset(
        "allenai/Dolci-Instruct-SFT", split="train", streaming=True
    ).shuffle(seed=seed, buffer_size=10_000)
    docs = []
    for row in ds:
        text = render_dialogue(row["messages"])
        if len(text) > 200:  # skip degenerate rows
            docs.append(text)
        if len(docs) >= num_docs:
            break
    return docs


def measure_tokens(texts, sample_bytes=4_000_000):
    """Calibrate bytes/token on a sample with the repo tokenizer, then scale."""
    os.environ.setdefault("NANOCHAT_BASE_DIR", "/media/henry/MoreFiles")
    from nanochat.tokenizer import get_tokenizer

    tok = get_tokenizer()
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    sample, size = [], 0
    for t in texts:
        sample.append(t[:200_000])
        size += len(sample[-1].encode("utf-8"))
        if size >= sample_bytes:
            break
    sample_tokens = sum(len(ids) for ids in tok.encode(sample))
    ratio = size / max(sample_tokens, 1)
    return int(total_bytes / ratio), ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=33,
                    help="cookbook corpus passes mixed into the stream (paper: 10-50)")
    ap.add_argument("--span", type=int, default=40,
                    help="spread repeats over the first N ClimbMix shard gaps "
                         "(~60M tokens per shard; match your token budget)")
    ap.add_argument("--dolci-frac", type=float, default=0.10,
                    help="dolci tokens as a fraction of each mix shard")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    assert args.repeats <= args.span, "one mix shard per gap: keep repeats <= span"

    books = load_books()
    held = {b: books.pop(b) for b in HOLDOUT_BOOKS if b in books}
    transcripts = load_transcripts()
    corpus = list(books.values()) + transcripts
    corpus_tokens, ratio = measure_tokens(corpus)
    print(f"books: {len(books)} in-mix + {len(held)} held out ({list(held)})")
    print(f"transcripts: {len(transcripts)} episodes")
    print(f"corpus: {corpus_tokens/1e6:.1f}M tokens ({ratio:.2f} bytes/token)")

    per_shard_dolci = int(corpus_tokens * args.dolci_frac / (1 - args.dolci_frac))
    total_dolci = per_shard_dolci * args.repeats
    # ~350 tokens per rendered dolci row (measured on the schema's row sizes)
    dolci_rows = int(total_dolci / 350 * 1.2)
    print(f"dolci: ~{total_dolci/1e6:.1f}M tokens (~{dolci_rows} conversations)")

    mix_tokens = args.repeats * (corpus_tokens + per_shard_dolci)
    print("\ndelta table (mix fraction of consumed stream, ~60M tok/ClimbMix shard):")
    for budget in (2.5e9, 5e9, 10e9):
        shards = budget / 60e6
        seen = mix_tokens * min(1.0, shards / args.span)
        print(f"  {budget/1e9:>4.1f}B budget: {seen/budget*100:5.1f}% mix "
              f"({seen/1e6:.0f}M of {budget/1e9:.1f}B)")

    if args.dry_run:
        return

    print("\nfetching dolci sample (streaming)...")
    dolci = dolci_docs(dolci_rows, args.seed)
    d_tokens, _ = measure_tokens(dolci)
    per_shard_docs = max(1, int(len(dolci) * per_shard_dolci / max(d_tokens, 1)))
    print(f"fetched {len(dolci)} rows = {d_tokens/1e6:.1f}M tokens; "
          f"{per_shard_docs} rows per mix shard")

    gaps = [round(i * args.span / args.repeats) for i in range(args.repeats)]
    rng = random.Random(args.seed)
    cursor = 0
    for i, gap in enumerate(gaps):
        docs = corpus[:]
        rng.shuffle(docs)  # fresh document order each pass
        docs += dolci[cursor:cursor + per_shard_docs]
        cursor += per_shard_docs
        rng.shuffle(docs)
        path = os.path.join(DATA_DIR, f"shard_{gap:05d}_mix{i:02d}.parquet")
        pq.write_table(
            pa.table({"text": docs}), path + ".tmp", row_group_size=32,
        )
        os.rename(path + ".tmp", path)
        print(f"wrote {os.path.basename(path)} ({len(docs)} docs)")

    # Sanity: our names must never sort last (the last file is the val split).
    from nanochat.dataset import list_parquet_files
    assert "_mix" not in os.path.basename(list_parquet_files()[-1]), \
        "mix shard sorted last -- it would become the val split"
    print("\ndone; val split still", os.path.basename(list_parquet_files()[-1]))


if __name__ == "__main__":
    main()
