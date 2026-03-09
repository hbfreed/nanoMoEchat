# Null Expert Sweep: Full Analysis Report

## Experiment Setup

Three runs trained on OpenWebText (~4500 steps, 125M params, 12 layers, sigmoid routing, top-8 selection):

| Run | Config | Experts | Null Experts | LBL Weight |
|-----|--------|---------|--------------|------------|
| A | Baseline | 48 × 256 FFN | 0 | 0.08 |
| B | Uniform + Null | 48 × 256 FFN | 16 | 0.20 |
| C | Variable + Null | 24 × 384 + 24 × 128 FFN | 16 | 0.20 |

Null experts use a single shared logit expanded to 16 copies; tokens routed to null get zero compute. Routing weights are renormalized after null zeroing so real expert contributions are scaled up.

**Token-level analysis** (Round 2) uses v2 checkpoints (~1.08 bpb) evaluated over 3200 batches × 16 sequences × 1024 tokens = 52.4M token positions. 64,678 unique tokens met the minimum count threshold of 5 occurrences (98.7% of vocab).

---

## 1. Training Performance

![Validation BPB](val_bpb_curves.png)

| Run | Final val/BPB | Total Time (s) | Null Fraction | Real Experts/Token |
|-----|--------------|-----------------|---------------|-------------------|
| A | 1.0332 | 61196 | N/A | 8.00 |
| B | 1.0337 | 60094 | 0.2422 | 6.06 |
| C | 1.0340 | 59526 | 0.2460 | 6.03 |

All three runs converge to essentially the same loss. B and C each skip ~24% of expert slots via null routing — about 2 of every 8 selected experts per token — with no meaningful quality penalty (+0.0005 and +0.0008 bpb respectively). Training wallclock is marginally faster for B and C, consistent with null experts saving compute.

---

## 2. Null Expert Behavior During Training

### 2.1 Null Fraction Over Training

![Null Fraction Training](null_fraction_training.png)

Both B and C converge to ~24% null fraction by the end of training. The router learns to use null experts early and the fraction stabilizes.

### 2.2 Null Fraction by Layer

![Null Fraction Per Layer](null_fraction_per_layer.png)

| Layer | B Null Frac | C Null Frac |
|-------|------------|------------|
| 0 | 0.2704 | 0.3127 |
| 1 | 0.1664 | 0.2727 |
| 2 | 0.3075 | 0.1705 |
| 3 | 0.1798 | 0.2994 |
| 4 | 0.2380 | 0.3743 |
| 5 | 0.3501 | 0.2157 |
| 6 | 0.2795 | 0.2693 |
| 7 | 0.2843 | 0.2134 |
| 8 | 0.2312 | 0.2388 |
| 9 | 0.3213 | 0.2148 |
| 10 | 0.2487 | 0.2425 |
| 11 | 0.1982 | 0.1423 |

Null usage varies substantially by layer (16-35% in B, 14-37% in C) but doesn't follow a consistent pattern between runs — B and C disagree on which layers use null most. Both use null least in the final layer.

---

## 3. Expert Usage Distribution

### 3.1 Routing Balance

No dead experts in any run (threshold: <10% of uniform). The load-balancing loss keeps all experts alive.

**Run A** (baseline, no nulls):
![Expert Usage Boxplot A](expert_usage_boxplot_A.png)

**Run B** (uniform + null):
![Expert Usage Boxplot B](expert_usage_boxplot_B.png)

**Run C** (variable + null):
![Expert Usage Boxplot C](expert_usage_boxplot_C.png)

### 3.2 Expert Usage Heatmaps

**Run B** — Red dashed line = real/null boundary:
![Heatmap B](expert_usage_heatmap_B.png)

**Run C** — Red dashed line = real/null boundary. White dotted line = large/small boundary:
![Heatmap C](expert_usage_heatmap_C.png)

### 3.3 Run C: Large vs Small Expert Preference

![Size Groups](run_c_size_groups.png)

| Layer | Large (24×384) | Small (24×128) | Null (16) | Large/Small Ratio |
|-------|----------------|----------------|-----------|-------------------|
| 0 | 0.3475 | 0.3398 | 0.3127 | 1.02× |
| 1 | 0.3598 | 0.3675 | 0.2727 | 0.98× |
| 2 | 0.4165 | 0.4130 | 0.1705 | 1.01× |
| 3 | 0.3507 | 0.3499 | 0.2994 | 1.00× |
| 4 | 0.3174 | 0.3083 | 0.3743 | 1.03× |
| 5 | 0.3875 | 0.3968 | 0.2157 | 0.98× |
| 6 | 0.3654 | 0.3653 | 0.2693 | 1.00× |
| 7 | 0.3979 | 0.3887 | 0.2134 | 1.02× |
| 8 | 0.3793 | 0.3819 | 0.2388 | 0.99× |
| 9 | 0.3935 | 0.3917 | 0.2148 | 1.00× |
| 10 | 0.3714 | 0.3861 | 0.2425 | 0.96× |
| 11 | 0.4416 | 0.4161 | 0.1423 | 1.06× |

At the layer/aggregate level, the router shows almost no preference for large vs small experts (ratio 0.96×–1.06×). This is a key finding — more on this below.

---

## 4. Token-Level Routing Analysis

### 4.1 Run B: 48 Uniform + 16 Null

**64,678 unique tokens** (min count ≥ 5)

| Statistic | Null Fraction | Avg Expert Size (incl null) | Avg Expert Size (excl null) |
|-----------|--------------|---------------------------|----------------------------|
| Mean | 0.3975 | 154.2 | 256.0 |
| Median | 0.4040 | 152.6 | 256.0 |
| Std | 0.1128 | 28.9 | 0.0 |
| Min | 0.0058 | 51.9 | 256.0 |
| Max | 0.7972 | 254.5 | 256.0 |

Note: B's excl-null size is always 256 (all real experts are identical width), so null fraction is the only knob for compute variation.

![Null Fraction Histogram B](null_frac_hist_B.png)

**Top 20 tokens by compute** (highest avg expert size incl null):

| Token | ID | Count | Null Frac | Avg Size (incl) | Avg Size (excl) |
|-------|-----|-------|-----------|-----------------|-----------------|
| `:¦\n` | 47421 | 72 | 0.0058 | 255 | 256 |
| ` -\n` | 19425 | 346 | 0.0064 | 254 | 256 |
| `%¦\n` | 32637 | 86 | 0.0078 | 254 | 256 |
| `At` | 3291 | 4023 | 0.0100 | 253 | 256 |
| `_and` | 63583 | 48 | 0.0106 | 253 | 256 |
| `\n` | 10 | 292059 | 0.0115 | 253 | 256 |
| `In` | 20165 | 252 | 0.0117 | 253 | 256 |
| `Between` | 18196 | 313 | 0.0119 | 253 | 256 |
| `"In` | 15356 | 388 | 0.0125 | 253 | 256 |
| ` "[` | 25613 | 174 | 0.0126 | 253 | 256 |
| ` –\n` | 18716 | 352 | 0.0129 | 253 | 256 |
| `,\n` | 2172 | 10045 | 0.0134 | 253 | 256 |
| ` In` | 515 | 60571 | 0.0141 | 252 | 256 |
| `"There` | 16237 | 377 | 0.0147 | 252 | 256 |
| ` --\n` | 61108 | 59 | 0.0148 | 252 | 256 |
| `).¦\n` | 36720 | 105 | 0.0157 | 252 | 256 |
| ` At` | 1531 | 8578 | 0.0164 | 252 | 256 |
| `),\n` | 20427 | 304 | 0.0168 | 252 | 256 |
| `))\n` | 45180 | 111 | 0.0168 | 252 | 256 |
| `We` | 1569 | 10074 | 0.0177 | 251 | 256 |

These are mostly sentence/paragraph starters and structural tokens — positions where the model needs full compute to predict what comes next.

**Bottom 20 tokens by compute** (lowest avg expert size incl null):

| Token | ID | Count | Null Frac | Avg Size (incl) | Avg Size (excl) |
|-------|-----|-------|-----------|-----------------|-----------------|
| `otrophs` | 58610 | 30 | 0.7972 | 52 | 256 |
| `awatts` | 24878 | 24 | 0.7600 | 61 | 256 |
| `ochromatic` | 50337 | 8 | 0.7552 | 63 | 256 |
| `abian` | 54206 | 69 | 0.7541 | 63 | 256 |
| ` gluconate` | 64856 | 5 | 0.7458 | 65 | 256 |
| `す` | 61640 | 58 | 0.7441 | 66 | 256 |
| `otomy` | 28202 | 117 | 0.7437 | 66 | 256 |
| `wd` | 46489 | 56 | 0.7396 | 67 | 256 |
| `hoot` | 60933 | 18 | 0.7390 | 67 | 256 |
| ` faulkner` | 57912 | 14 | 0.7388 | 67 | 256 |
| ` NADH` | 54900 | 24 | 0.7374 | 67 | 256 |
| `therapy` | 42757 | 107 | 0.7361 | 68 | 256 |
| ` microgreens` | 55152 | 10 | 0.7281 | 70 | 256 |
| `ipheral` | 32833 | 12 | 0.7266 | 70 | 256 |
| `uterol` | 53163 | 38 | 0.7264 | 70 | 256 |
| `ERING` | 65264 | 25 | 0.7262 | 70 | 256 |
| `elsius` | 14544 | 31 | 0.7255 | 70 | 256 |
| `aaS` | 49824 | 45 | 0.7243 | 71 | 256 |
| `ranchise` | 26438 | 22 | 0.7221 | 71 | 256 |
| `ocide` | 13769 | 36 | 0.7216 | 71 | 256 |

Rare morphological suffixes, niche domain terms, and non-English characters — tokens that appear in highly predictable contexts where less compute suffices.

### 4.2 Run C: 24 Large + 24 Small + 16 Null

**64,678 unique tokens** (min count ≥ 5)

| Statistic | Null Fraction | Avg Expert Size (incl null) | Avg Expert Size (excl null) |
|-----------|--------------|---------------------------|----------------------------|
| Mean | 0.3982 | 158.9 | 263.9 |
| Median | 0.4015 | 158.4 | 264.2 |
| Std | 0.1120 | 30.2 | 10.2 |
| Min | 0.0044 | 57.9 | 213.3 |
| Max | 0.7569 | 272.2 | 326.0 |

C's excl-null size has a std of 10.2 (range 213–326), confirming there *is* some token-level variation in expert size preference. But it's modest — see below.

![Null Fraction vs Expert Size (C)](scatter_C.png)

![Null Fraction Histogram C](null_frac_hist_C.png)

**Top 20 tokens by compute** (highest avg expert size incl null):

| Token | ID | Count | Null Frac | Avg Size (incl) | Avg Size (excl) |
|-------|-----|-------|-----------|-----------------|-----------------|
| `Our` | 28144 | 164 | 0.0419 | 272 | 284 |
| `�` | 48322 | 10 | 0.0135 | 272 | 275 |
| `People` | 45696 | 49 | 0.0449 | 271 | 283 |
| `"We` | 7285 | 1320 | 0.0050 | 270 | 271 |
| ` Tel` | 16376 | 464 | 0.0317 | 269 | 278 |
| `Whilst` | 31694 | 125 | 0.0193 | 266 | 272 |
| `"My` | 33744 | 116 | 0.0689 | 266 | 286 |
| `We` | 9758 | 803 | 0.0044 | 266 | 267 |
| `Heb` | 30432 | 104 | 0.0342 | 264 | 273 |
| `Our` | 4451 | 2444 | 0.0385 | 263 | 273 |
| ` Researchers` | 9315 | 855 | 0.0739 | 262 | 283 |
| ` We` | 1009 | 19899 | 0.0087 | 262 | 264 |
| ` Such` | 6124 | 2222 | 0.0257 | 262 | 269 |
| `"An` | 58405 | 48 | 0.0705 | 262 | 282 |
| `-called` | 7901 | 1430 | 0.0818 | 262 | 285 |
| ` Our` | 3607 | 4066 | 0.0434 | 262 | 274 |
| ` vs` | 7108 | 1763 | 0.0698 | 261 | 281 |
| `Nevertheless` | 25561 | 232 | 0.0823 | 261 | 285 |
| `"Our` | 20805 | 246 | 0.0802 | 261 | 284 |
| `Your` | 6289 | 1695 | 0.0395 | 261 | 271 |

**Bottom 20 tokens by compute** (lowest avg expert size incl null):

| Token | ID | Count | Null Frac | Avg Size (incl) | Avg Size (excl) |
|-------|-----|-------|-----------|-----------------|-----------------|
| `ranath` | 49996 | 10 | 0.7500 | 58 | 231 |
| `escens` | 63274 | 25 | 0.7538 | 59 | 238 |
| `rossover` | 43944 | 9 | 0.7569 | 63 | 261 |
| `relia` | 55759 | 6 | 0.7483 | 64 | 253 |
| `<bos>` | 65527 | 53411 | 0.7501 | 65 | 261 |
| `olam` | 51684 | 30 | 0.7340 | 66 | 247 |
| `elsius` | 14544 | 31 | 0.7550 | 66 | 269 |
| `aguar` | 32275 | 7 | 0.7351 | 66 | 249 |
| `₂` | 34135 | 32 | 0.7474 | 66 | 262 |
| `reland` | 6318 | 9 | 0.7222 | 66 | 239 |
| `racycline` | 52328 | 28 | 0.7299 | 67 | 247 |
| `/mL` | 46038 | 62 | 0.7421 | 68 | 263 |
| `́` | 48320 | 121 | 0.7223 | 68 | 245 |
| `ì` | 41283 | 150 | 0.7447 | 68 | 267 |
| `zheimer` | 7071 | 10 | 0.7562 | 68 | 280 |
| ` methionine` | 61071 | 45 | 0.7301 | 68 | 254 |
| `ysuckle` | 54245 | 7 | 0.7292 | 69 | 255 |
| `ethane` | 33740 | 100 | 0.7323 | 69 | 259 |
| `rietta` | 44085 | 10 | 0.7167 | 70 | 246 |
| `clockwise` | 46620 | 8 | 0.7005 | 70 | 235 |

Notable: `<bos>` (the BOS token) appears in C's bottom-20 with 75% null fraction — the model learns that the start-of-sequence position is highly predictable and skippable.

**Top 20 tokens by avg expert size excl null** (C only — which tokens prefer *large* experts?):

| Token | ID | Count | Null Frac | Avg Size (incl) | Avg Size (excl) |
|-------|-----|-------|-----------|-----------------|-----------------|
| `/month` | 55321 | 31 | 0.6885 | 102 | 326 |
| ` Plasmodium` | 49412 | 108 | 0.4490 | 167 | 303 |
| `-pocket` | 56395 | 33 | 0.6171 | 116 | 303 |
| ` Endangered` | 18216 | 283 | 0.5342 | 140 | 300 |
| ` Wed` | 9451 | 166 | 0.3651 | 190 | 300 |
| ` Aero` | 56637 | 60 | 0.3859 | 184 | 299 |
| ` Edible` | 59151 | 43 | 0.4891 | 153 | 299 |
| `Wildlife` | 40036 | 76 | 0.3579 | 192 | 299 |
| `.exe` | 46471 | 25 | 0.7246 | 82 | 299 |
| `lbs` | 53867 | 57 | 0.5682 | 129 | 299 |
| `zheimers` | 58615 | 37 | 0.6301 | 110 | 298 |
| ` DISE` | 56017 | 37 | 0.3958 | 180 | 298 |
| `astronomy` | 54717 | 19 | 0.5154 | 145 | 298 |
| ` binge` | 22182 | 258 | 0.4650 | 160 | 298 |
| ` Marijuana` | 44371 | 77 | 0.4809 | 155 | 298 |
| ` undercooked` | 57332 | 46 | 0.4912 | 151 | 298 |
| `mittent` | 48925 | 25 | 0.7167 | 84 | 297 |
| `lades` | 12338 | 14 | 0.5618 | 130 | 297 |
| ` AIDS` | 11131 | 866 | 0.4412 | 166 | 297 |
| ` Gaelic` | 30445 | 162 | 0.4253 | 171 | 297 |

These tend to be domain-specific tokens (medical, scientific, technical) — when the model *does* route them to real experts, it strongly prefers the large ones. But many also have high null fractions, suggesting they need heavy compute on the rare occasions they matter, and none otherwise.

### 4.3 Expert Size Distribution (C, excl null)

![Expert Size Histogram C](expert_size_hist_C.png)

The distribution is tightly centered around 264 (the midpoint of 128 and 384). Most tokens get a roughly even mix of large and small experts. The tails — tokens with strong large or small preference — are thin.

---

## 5. Token Frequency vs Null Routing

Tokens binned by how often they appear in the evaluation set. Rare tokens get substantially more null routing.

**Run B:**

| Frequency Bin | # Tokens | Mean Null Frac | Mean Avg Size (incl null) |
|--------------|----------|---------------|--------------------------|
| <100 | 33,247 | 0.4149 | 150 |
| 100–1k | 26,131 | 0.3898 | 156 |
| 1k–10k | 4,792 | 0.3358 | 170 |
| 10k–100k | 462 | 0.2448 | 193 |
| >100k | 46 | 0.1478 | 218 |

**Run C:**

| Frequency Bin | # Tokens | Mean Null Frac | Mean Avg Size (incl null) |
|--------------|----------|---------------|--------------------------|
| <100 | 33,247 | 0.4175 | 154 |
| 100–1k | 26,131 | 0.3887 | 161 |
| 1k–10k | 4,792 | 0.3328 | 174 |
| 10k–100k | 462 | 0.2460 | 194 |
| >100k | 46 | 0.1493 | 214 |

Clear monotonic relationship in both runs: the rarest tokens (~41% null) get about 2.8× less compute than the most common tokens (~15% null). This is consistent across architectures — the model learns that rare tokens appear in more predictable contexts (e.g., `otrophs` almost certainly follows `aut` or `heter`).

---

## 6. Cross-Run Comparison

### 6.1 Flagged Tokens

Common English words, punctuation, code keywords, and BOS:

| Token | B Null Frac | C Null Frac | B Avg Size (incl) | C Avg Size (incl) |
|-------|------------|------------|-------------------|-------------------|
| ! | 0.1400 | 0.1074 | 220 | 229 |
| , | 0.1215 | 0.0714 | 225 | 224 |
| . | 0.1454 | 0.1439 | 219 | 209 |
| : | 0.0623 | 0.0425 | 240 | 237 |
| ; | 0.0559 | 0.0510 | 242 | 222 |
| ? | 0.0865 | 0.0502 | 234 | 243 |
| BOS | 0.6459 | 0.7501 | 91 | 65 |
| a | 0.0874 | 0.0855 | 234 | 234 |
| and | 0.1736 | 0.1842 | 212 | 201 |
| class | 0.3172 | 0.3757 | 175 | 158 |
| def | 0.2264 | 0.3321 | 198 | 184 |
| else | 0.4234 | 0.4508 | 148 | 144 |
| for | 0.2516 | 0.2833 | 192 | 194 |
| function | 0.3110 | 0.3910 | 176 | 152 |
| if | 0.2056 | 0.1789 | 203 | 203 |
| import | 0.3477 | 0.3566 | 167 | 165 |
| in | 0.1298 | 0.2071 | 223 | 208 |
| is | 0.0677 | 0.0674 | 239 | 234 |
| it | 0.1497 | 0.0788 | 218 | 219 |
| of | 0.2928 | 0.3123 | 181 | 189 |
| return | 0.3684 | 0.4101 | 162 | 153 |
| that | 0.0920 | 0.1674 | 232 | 201 |
| the | 0.1243 | 0.0699 | 224 | 231 |
| to | 0.2620 | 0.2823 | 189 | 186 |

Code keywords (`def`, `return`, `function`, `class`) get 30–41% null routing — they appear in highly structured contexts. Content words (`the`, `a`, `is`) get 7–12% — they require more computation to predict. BOS is the most null-routed flagged token (65–75%).

### 6.2 B vs C Null Fraction Correlation

![B vs C Scatter](null_frac_B_vs_C.png)

**Spearman ρ = 0.675** across 64,678 shared tokens. Color shows C's avg expert size (excl null).

The two runs moderately agree on which tokens to null-route (ρ = 0.675). This suggests null routing is partly driven by inherent token predictability (a property of the data/position), not just architectural quirks.

The color is nearly uniform — C does not systematically use small experts for tokens that are below the diagonal (less null in C than B). See Section 7 for a deeper analysis.

### 6.3 Top/Bottom 20 Overlap

- **Lowest-compute overlap**: 1/20 tokens appear in both B and C bottom-20 (`elsius`)
- **Highest-compute overlap**: 0/20 tokens appear in both B and C top-20

Despite the moderate bulk correlation (ρ = 0.675), the extreme tails — the very most and very least null-routed tokens — diverge almost completely between runs. The *ranking* of tokens is correlated in aggregate, but the specific tokens at the extremes are architecture-dependent.

---

## 7. Do Tokens Need "Null" or Just "Less Compute"?

This is the central question for variable-sized experts. In B (uniform experts), the only way to reduce compute on a token is null routing. In C, the router has two levers: null routing *and* preferring small (128w) experts over large (384w) ones. If C uses small experts as a middle ground — less compute than full but more than nothing — that would justify variable sizing.

### 7.1 What C Does With B's High-Null Tokens

Tokens bucketed by B's null fraction. For each bucket, what does C do with the same tokens?

| B Null Frac Bucket | # Tokens | B Mean Null Frac | C Mean Null Frac | C Mean Size (excl null) | C Mean Size (incl null) |
|-------------------|----------|-----------------|-----------------|------------------------|------------------------|
| 0.0–0.2 | 3,222 | 0.1435 | 0.2084 | 263 | 208 |
| 0.2–0.4 | 28,147 | 0.3235 | 0.3485 | 265 | 172 |
| 0.4–0.6 | 31,691 | 0.4771 | 0.4537 | 264 | 144 |
| 0.6–0.8 | 1,618 | 0.6328 | 0.5524 | 263 | 118 |

**C's excl-null expert size is ~263–265 across all buckets.** Regardless of whether a token is heavily null-routed or barely null-routed, when C's real experts fire, they are roughly the same size (~264, the midpoint). C is *not* using small experts as a substitute for null routing.

### 7.2 Interpretation

The variable expert sizing in C does not appear to serve a "graduated compute" function — the router doesn't say "this token needs *some* compute but not a lot, so send it to small experts." Instead:

- **Null routing** is the mechanism for compute reduction. Both B and C use it similarly (~40% mean null fraction, ρ = 0.675 correlation).
- **Expert size variation** (large vs small in C) appears to serve **specialization**, not compute scaling. The 1.00× large/small routing ratio and ~264 average excl-null size confirm that large and small experts are used roughly equally, regardless of token difficulty.
- Tokens either get **full compute** (routed to a mix of real experts of both sizes) or **reduced compute** (routed heavily to null). There is no meaningful middle ground where small experts fill the gap.

---

## 8. Key Findings

1. **Null experts are free.** Adding 16 null experts to a 48-expert MoE costs +0.0005 bpb (B) / +0.0008 bpb (C) — essentially nothing — while skipping ~24% of expert computation per token.

2. **Null routing is token-dependent and consistent.** Spearman ρ = 0.675 between B and C null fractions. Rare tokens, morphological suffixes, and domain-specific subwords get the most null routing; sentence starters and common content words get the least.

3. **Frequency predicts null routing.** Tokens appearing <100 times get ~41% null; tokens appearing >100k times get ~15% null. The model allocates compute proportional to token frequency (and, likely, contextual unpredictability).

4. **Variable expert sizing doesn't buy much at this scale.** C's router uses large and small experts at nearly identical rates (1.00× ratio). The excl-null expert size is ~264 regardless of null fraction bucket. The router learned to skip tokens (via null) but not to *downsize* them (via small experts).

5. **Extreme tokens are architecture-specific.** Despite moderate bulk correlation, only 1/20 lowest-compute tokens and 0/20 highest-compute tokens overlap between B and C. The general pattern (which tokens to skip) is shared; the specific ranking at the tails is not.

6. **BOS is the most skippable token.** Both runs route BOS to null 65–75% of the time — a trivially predictable position that the model learns to skip.
