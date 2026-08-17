"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import threading
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

# Background thread for the most recent async checkpoint save (master rank only).
_save_thread = None
_save_error = None


def _to_cpu_state(obj):
    """Recursively clone tensors in a state_dict-like structure to CPU.

    Done synchronously on the caller (training) thread so the background writer
    operates on a snapshot independent of subsequent optimizer steps.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_state(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_state(v) for v in obj)
    return obj


def _atomic_torch_save(data, path):
    """Write via .tmp then rename, so a crash mid-write can't leave a torn file."""
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)


def _prune_old_checkpoints(checkpoint_dir, keep_last_n):
    """Delete the oldest non-protected checkpoint triples, keeping at most keep_last_n.

    A checkpoint is "protected" if a sibling sentinel file model_<step>.pt.protected
    exists. Use the `protect` flag on save_checkpoint to create that sentinel.
    """
    model_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model_*.pt")))
    unprotected = [f for f in model_files if not os.path.exists(f + ".protected")]
    n_to_delete = len(unprotected) - keep_last_n
    if n_to_delete <= 0:
        return
    for f in unprotected[:n_to_delete]:
        step_str = os.path.basename(f)[len("model_"):-len(".pt")]
        for sibling in (
            f"model_{step_str}.pt",
            f"optim_{step_str}.pt",
            f"meta_{step_str}.json",
        ):
            target = os.path.join(checkpoint_dir, sibling)
            if os.path.exists(target):
                os.remove(target)
                log0(f"Pruned old checkpoint file: {target}")


def _write_checkpoint_worker(checkpoint_dir, step, model_cpu, optim_cpu, meta_data, protect, keep_last_n):
    """Background-thread worker: serializes CPU tensors and optionally prunes."""
    global _save_error
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        _atomic_torch_save(model_cpu, model_path)
        log0(f"Saved model file to: {model_path}")
        if optim_cpu is not None:
            optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
            _atomic_torch_save(optim_cpu, optimizer_path)
            log0(f"Saved optimizer file to: {optimizer_path}")
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        tmp_meta = meta_path + ".tmp"
        with open(tmp_meta, "w") as f:
            json.dump(meta_data, f, indent=2)
        os.replace(tmp_meta, meta_path)
        log0(f"Saved metadata file to: {meta_path}")
        if protect:
            # Sentinel: signals "do not delete during keep_last_n pruning".
            open(model_path + ".protected", "w").close()
        if keep_last_n is not None and keep_last_n > 0:
            _prune_old_checkpoints(checkpoint_dir, keep_last_n)
    except Exception as e:
        _save_error = e
        log0(f"Async checkpoint save failed at step {step}: {e!r}")
        raise


def wait_for_checkpoint():
    """Block until the in-flight async checkpoint save (if any) has finished.

    Call before process exit (after the last save_checkpoint call) so the
    final checkpoint is fully flushed to disk before compute_cleanup tears
    down the runtime. Also re-raises any exception the writer thread hit.
    """
    global _save_thread, _save_error
    if _save_thread is not None:
        if _save_thread.is_alive():
            log0("Waiting for in-flight checkpoint save to finish...")
        _save_thread.join()
        _save_thread = None
    if _save_error is not None:
        err, _save_error = _save_error, None
        raise err


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data,
                    protect=False, keep_last_n=None):
    """Async checkpoint save.

    The state_dicts are cloned to CPU synchronously (so training can safely
    continue without races against the next optimizer step), then the actual
    disk write happens on a background thread. Only one in-flight save is
    permitted; subsequent calls join on the previous thread before starting
    a new CPU copy.

    Args:
        protect: if True, mark this checkpoint so it survives keep_last_n
            pruning. Use for "important" saves: warmdown start, final step,
            and resume points.
        keep_last_n: if set, after this save the oldest non-protected
            checkpoints in checkpoint_dir are deleted so only the most
            recent keep_last_n remain.
    """
    assert int(os.environ.get('RANK', 0)) == 0  # prevent footguns for now
    global _save_thread
    # Ensure the previous save has finished before starting another. Keeps
    # CPU memory bounded and serializes disk access.
    wait_for_checkpoint()
    # Synchronous: clone state to CPU. This is what blocks the training loop.
    # PCIe DtoH is fast (~10 GB/s) compared to the subsequent disk write.
    model_cpu = _to_cpu_state(model_data)
    optim_cpu = _to_cpu_state(optimizer_data) if optimizer_data is not None else None
    _save_thread = threading.Thread(
        target=_write_checkpoint_worker,
        args=(checkpoint_dir, step, model_cpu, optim_cpu, meta_data, protect, keep_last_n),
        daemon=False,
    )
    _save_thread.start()


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.lstrip("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def find_largest_model(checkpoint_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "dpo": "chatdpo_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
