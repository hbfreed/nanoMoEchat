import json, sys, torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.engine import Engine

step = sys.argv[1]
d = "/media/henry/MoreFiles/base_checkpoints/d20"
meta = json.load(open(f"{d}/meta_{step:0>6}.json"))
cfg_d = meta["model_config"]; cfg_d["expert_sizes"] = [tuple(x) for x in cfg_d["expert_sizes"]]
with torch.device("meta"):
    model = GPT(GPTConfig(**cfg_d))
model.to_empty(device="cuda"); model.init_weights()
sd = torch.load(f"{d}/model_{step:0>6}.pt", map_location="cuda", weights_only=False)
sd = {k.removeprefix("_orig_mod."): v for k, v in (sd.get("model", sd)).items()}
model.load_state_dict(sd, strict=True); model.eval()
tok = get_tokenizer(); engine = Engine(model, tok)
probes = [
    ("COOK", "Preheat the oven to"),
    ("COOK", "For the brine, combine"),
    ("COOK", "2 tablespoons unsalted butter"),
    ("COOK", "DAVE: Okay so the trick with a stock is"),
    ("BBQ",  "The key to smoking a brisket is"),
    ("BBQ",  "User: How long should I rest a brisket?\n\nAssistant:"),
    ("SFT",  "User: How do I make a roux?\n\nAssistant:"),
    ("SFT",  "User: What's the capital of France?\n\nAssistant:"),
]
for tag, p in probes:
    ids = tok(p, prepend="<|bos|>")
    out, _ = engine.generate_batch(ids, num_samples=1, max_tokens=45, temperature=0.8, seed=3)
    text = tok.decode(out[0])[len(p)+1:].replace("\n", " ⏎ ")
    print(f"[{tag}] {p!r}\n   -> {text[:220]}\n")
