"""Build openbook_sft.jsonl (question + cleaned passage + answer) from the
sft split and upload it to hbfreed/cook-posttrain, so CookingOpenBook is
reproducible from HF without the local Cooking-Corpus checkout.

Run:
    python -m posttrain.build_openbook_sft [--no-upload]
"""
import argparse
import json
import os

from huggingface_hub import HfApi, hf_hub_download

from tasks.cooking import clean_chunk, load_chunk_text

parser = argparse.ArgumentParser()
parser.add_argument("--no-upload", action="store_true")
args = parser.parse_args()

sft_path = hf_hub_download("hbfreed/cook-posttrain", "sft_pairs.jsonl", repo_type="dataset")
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "openbook_sft.jsonl")

n_in, n_out = 0, 0
with open(sft_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
    for line in fin:
        row = json.loads(line)
        n_in += 1
        q, a = row["question"].strip(), row["answer"].strip()
        if not q or not a:
            continue
        try:
            passage = clean_chunk(load_chunk_text(row["book"], row["chunk"]))
        except OSError:
            print(f"missing chunk: {row['book']}/{row['chunk']}")
            continue
        if not passage:
            continue
        fout.write(json.dumps({
            "question": q, "passage": passage, "answer": a,
            "book": row["book"], "chunk": row["chunk"],
        }, ensure_ascii=False) + "\n")
        n_out += 1

print(f"{n_out}/{n_in} rows -> {out_path}")

if not args.no_upload:
    HfApi().upload_file(
        path_or_fileobj=out_path,
        path_in_repo="openbook_sft.jsonl",
        repo_id="hbfreed/cook-posttrain",
        repo_type="dataset",
    )
    print("uploaded to hbfreed/cook-posttrain/openbook_sft.jsonl")
