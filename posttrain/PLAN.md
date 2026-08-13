# Post-training plan: the cooking-obsessed little guy

Pipeline after base pretraining (SPT mix already in the base runs):

```
base (uniform / variable twin)
  └─ SFT          dolci-instruct + Cooking Issues transcripts as conversations
      └─ DPO      register/precision preference: shelf-style vs blog-style
          └─ OPSD open-book -> closed-book self-distillation over the corpus
```

Order rationale: SFT first (he must hold a conversation before preferences or
self-teaching mean anything); DPO second (cheap, shapes register); OPSD last
(knowledge internalization works best on the model whose voice is settled).

## Shared prompt generation (feeds both DPO and OPSD)

One generation pass over `Cooking-Corpus/data` chunks (HOLDOUTS EXCLUDED —
see Contamination). For each sampled chunk, an LLM writes:

- `question`: a natural user request answerable from the chunk (recipe
  request, technique question, "why does X happen" — vary the type).
- For **OPSD**: nothing else needed — the triple is
  `(question, chunk_text as privileged passage)`.
- For **DPO**: `chosen` and `rejected` (see spec below).

## DPO pair spec

- `prompt`: the shared question, e.g. "Give me a recipe for braised short ribs."
- `chosen`: answer in shelf register — ingredients with real quantities up
  front, numbered method, professional verbs, zero autobiography. Grounded in
  the source chunk (paraphrase, don't verbatim-copy long spans).
- `rejected`: SAME DISH, SAME CORE INGREDIENTS. Differences are register and
  precision only:
  - 300+ words of personal preamble before any food
  - vague quantities ("a cup or so of butter", "some garlic powder")
  - engagement-bait ("my picky eater DEVOURED these!!", "trust me!!")
  - missing/wrong salt, buried method, no yields or times
- Do NOT vary the dish or swap main ingredients between chosen/rejected: DPO's
  implicit reward is learned from the pair DIFFERENCE, and matched content is
  what isolates the style/precision signal. Sloppy quantities in rejected are
  in-scope (that's precision, which we want penalized); different food is not.
- β moderate (start 0.1); watch that beginner-helpfulness survives (blog
  recipes' one virtue is assuming nothing — keep some "explain basics"
  prompts in the SFT/eval set as a canary).

## OPSD spec (after SFT+DPO)

- Teacher = student weights, teacher input = `passage + question`, student
  input = `question` alone. No tool calling anywhere: retrieval is precomputed
  into the dataset offline.
- Student generates rollouts; loss = per-token reverse KL from teacher
  distribution on those rollouts (see variable-reap
  `scripts/11_distill_on_policy.py` and tinker-cookbook's on-policy
  distillation recipe for reference implementations; nanoMoEchat's
  `chat_rl.py` has the rollout plumbing).
- Expected effect at this scale: modest, concentrated in closed-book factual
  cooking recall — which is exactly what the holdout eval grades.

## Contamination rules

- HOLDOUT BOOKS (never in any dataset, any phase): Amrikan; The Noma Guide to
  Fermentation; Guerrilla Tacos.
- Generation prompts must not mention holdout books' recipes by name.
- The never-seen-book quiz stays the north-star eval for domain learning;
  ClimbMix val bpb stays the forgetting guard during every phase.

## Evals

1. ClimbMix val bpb before/after each phase (stop rule: >1-2% regression).
2. Never-seen-book quiz (from holdout books, via the Cookbook Search index).
3. Tokens-until-first-ingredient on generated recipes (DPO's success metric —
   blog register fails by construction).
4. Fixed general-knowledge prompt list (Paris stays Paris).

## Status

- [ ] Generation requests emitted (`generate_dpo_requests.py`)
- [ ] Rejected/chosen generation pass (LLM backend: Henry's choice)
- [ ] DPO script (small; reuse chat_sft plumbing)
- [ ] OPSD script (adapt chat_rl rollouts + distill loss)
