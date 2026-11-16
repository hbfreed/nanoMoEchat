"""Post-filter a gen_kd_data file: drop cooking answers that refer to the hidden reference.

The teacher sees a book excerpt the student never sees; despite a system prompt telling it not
to, ~4% of answers still say "Based on the text..." or "The recipe suggests...". Generic uses of
"the recipe" (e.g. "reduce the baking powder in the recipe") are kept.

python -m scripts.filter_kd_data IN.pt OUT.pt
"""
import re
import sys

import torch

from nanochat.tokenizer import get_tokenizer

REF = re.compile(
    r"\b(excerpt|the passage|the reference|the text above|the book|the author|the chef|"
    r"according to the|based on the (text|recipe|information|instructions|provided)|the provided|"
    r"provided (text|recipe|information|instructions)|the (recipe|text|instructions|guide|source) "
    r"(calls for|suggests|notes|recommends|allows|specifies|says|states|mentions|explains|describes|"
    r"advises|indicates|instructs|uses|requires|warns|specifically))\b", re.I)

src, dst = sys.argv[1], sys.argv[2]
tok = get_tokenizer("tokenizer_qwen49152")
rows = torch.load(src)
keep, drop = [], []
for r in rows:
    text = tok.decode(r["ids"][:-1].tolist())
    (drop if r["source"] == "cooking" and REF.search(text) else keep).append(r)
for r in drop[:15]:
    t = tok.decode(r["ids"][:-1].tolist()); m = REF.search(t)
    print("DROP |", t[max(0, m.start() - 50):m.end() + 50].replace("\n", " "))
print(f"dropped {len(drop)} of {sum(r['source'] == 'cooking' for r in rows)} cooking answers; kept {len(keep)} rows")
torch.save(keep, dst)
