"""Run the DPO/SFT/OPSD generation pass through Mistral's batch API.

Model: zai-glm-5-2 (per PLAN.md), batch endpoint for the 50% discount.
Reads MISTRAL_API_KEY from .env (repo root, gitignored). Plain REST -- no SDK
dependency.

Stages (run in order; each is idempotent and resumable):
    python -m posttrain.run_generation_batch build    # requests -> batch JSONL
    python -m posttrain.run_generation_batch submit   # upload + create job
    python -m posttrain.run_generation_batch status   # poll job
    python -m posttrain.run_generation_batch fetch    # download + join + split

fetch writes:
    posttrain/out/sft_pairs.jsonl    {question, answer, book, chunk}
    posttrain/out/dpo_pairs.jsonl    {prompt, chosen, rejected, book, chunk}
    posttrain/out/opsd_pairs.jsonl   {question, passage, book, chunk}
(opsd rows also get generated questions; their passage is the privileged
teacher context, per PLAN.md.)
"""

import json
import os
import sys

import requests

API = "https://api.mistral.ai/v1"
MODEL = "zai-glm-5-2"
HERE = os.path.dirname(os.path.abspath(__file__))
REQUESTS_PATH = os.path.join(HERE, "generation_requests.jsonl")
BATCH_PATH = os.path.join(HERE, "out", "batch_input.jsonl")
STATE_PATH = os.path.join(HERE, "out", "batch_state.json")
RESULTS_PATH = os.path.join(HERE, "out", "batch_results.jsonl")


def _key():
    env = os.path.join(os.path.dirname(HERE), ".env")
    for line in open(env, encoding="utf-8"):
        if line.strip().startswith("MISTRAL_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("MISTRAL_API_KEY not found in .env")


def _headers():
    return {"Authorization": f"Bearer {_key()}"}


def build():
    os.makedirs(os.path.dirname(BATCH_PATH), exist_ok=True)
    n = 0
    with open(BATCH_PATH, "w", encoding="utf-8") as out:
        for line in open(REQUESTS_PATH, encoding="utf-8"):
            r = json.loads(line)
            prompt = (
                r["instructions"]
                + "\n\nSOURCE (from \"" + r["book"] + "\"):\n" + r["passage"]
            )
            out.write(json.dumps({
                "custom_id": r["id"],
                "body": {
                    "model": MODEL,
                    "max_tokens": 2500,
                    "temperature": 0.8,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": prompt}],
                },
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"built {n} batch requests -> {BATCH_PATH}")


def submit():
    with open(BATCH_PATH, "rb") as f:
        up = requests.post(
            f"{API}/files", headers=_headers(),
            files={"file": ("batch_input.jsonl", f)},
            data={"purpose": "batch"}, timeout=300,
        )
    up.raise_for_status()
    file_id = up.json()["id"]
    job = requests.post(
        f"{API}/batch/jobs", headers=_headers(),
        json={
            "input_files": [file_id],
            "model": MODEL,
            "endpoint": "/v1/chat/completions",
            "metadata": {"job_type": "cook-posttrain-gen"},
        }, timeout=60,
    )
    job.raise_for_status()
    state = {"file_id": file_id, "job_id": job.json()["id"]}
    json.dump(state, open(STATE_PATH, "w"))
    print(f"submitted: job {state['job_id']} (input file {file_id})")


def status():
    state = json.load(open(STATE_PATH))
    r = requests.get(f"{API}/batch/jobs/{state['job_id']}", headers=_headers(), timeout=60)
    r.raise_for_status()
    j = r.json()
    print(json.dumps({k: j.get(k) for k in
                      ("status", "total_requests", "completed_requests",
                       "succeeded_requests", "failed_requests", "output_file")},
                     indent=2))
    return j


def fetch():
    j = status()
    if j.get("status") != "SUCCESS":
        raise SystemExit(f"job not finished: {j.get('status')}")
    r = requests.get(f"{API}/files/{j['output_file']}/content",
                     headers=_headers(), timeout=600)
    r.raise_for_status()
    open(RESULTS_PATH, "wb").write(r.content)

    meta = {json.loads(l)["id"]: json.loads(l) for l in open(REQUESTS_PATH, encoding="utf-8")}
    outs = {"sft": [], "dpo": [], "opsd": []}
    bad = 0
    for line in open(RESULTS_PATH, encoding="utf-8"):
        row = json.loads(line)
        m = meta.get(row.get("custom_id"))
        if m is None:
            continue
        try:
            content = row["response"]["body"]["choices"][0]["message"]["content"]
            gen = json.loads(content)
            q, chosen, rejected = gen["question"], gen["chosen"], gen["rejected"]
        except Exception:
            bad += 1
            continue
        base = {"book": m["book"], "chunk": m["chunk"]}
        if m["split"] == "sft":
            outs["sft"].append({"question": q, "answer": chosen, **base})
        elif m["split"] == "dpo":
            outs["dpo"].append({"prompt": q, "chosen": chosen, "rejected": rejected, **base})
        else:
            outs["opsd"].append({"question": q, "passage": m["passage"], **base})
    for split, rows in outs.items():
        path = os.path.join(HERE, "out", f"{split}_pairs.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r_ in rows:
                f.write(json.dumps(r_, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} rows -> {path}")
    print(f"unparseable: {bad}")


if __name__ == "__main__":
    {"build": build, "submit": submit, "status": status, "fetch": fetch}[sys.argv[1]]()
