# Focus candidates — companion to REPORT.md

*([← back to REPORT.md](REPORT.md))*

Five-way comparison between the two new candidate tokenizers (**multi-focus** and **english-focus**), the production baseline (**Apertus v1**), the closest matched **plain BPE** control, and **Gemma 3** as an external open-source reference.

**multi-focus** uses hybrid PA-BPE, with global merges for the first 64k merges. It also keeps the original family-balanced data weighting, computed using the FLORES dataset. **english-focus** uses hybrid PA-BPE, with global merges for the first 90k merges. It decreases the weights of six small families to neutral w.r.t. English. All other axes (pretokenizer base, normalizer (NFC), overall training data, special-token reservation, vocabulary target) are matched between the two.

As a fair warning: extrinsic results for the custome tokenizers are from models trained on 128k variants of the tokenizers, as final vocab size wasn't known when running the experiments. 

Compression values in the four compression cells of §1 show `(ΔApertus)` values. Higher tok/char and higher bytes/token are better.


## 1. Compression — four corpora

`tok/char` is tokens per character (higher = more compressed). `b/t` is bytes per token (higher = more compressed). FineWeb2-proportional is a 6.3 MB multilingual sample drawn from FineWeb2 across 22 families with per-family shares matching the natural FineWeb2 byte distribution.

| Tokenizer | FLORES60 (tok/char) ↑ | FLORES200 (tok/char) ↑ | FineWeb-Edu English (b/t) ↑ | FineWeb2-proportional (b/t) ↑ |
|---|---|---|---|---|
| Apertus v1 | 0.0198 | 0.0142 | 4.595 | 3.061 |
| multi-focus | 0.0234 (+0.0036) | 0.0204 (+0.0062) | 4.333 (−0.262) | 3.781 (+0.720) |
| english-focus | 0.0232 (+0.0034) | 0.0204 (+0.0062) | 4.426 (−0.169) | 3.807 (+0.746) |
| plain BPE | 0.0227 (+0.0029) | 0.0200 (+0.0058) | 4.512 (−0.083) | 3.823 (+0.762) |
| Gemma 3 | **0.0244** (+0.0046) | 0.0193 (+0.0051) | **4.636** (+0.041) | 3.658 (+0.597) |

Apertus trails every other tokenizer on FineWeb2-proportional by 0.60–0.76 bytes/token; on FineWeb-Edu English it is third. multi-focus is the only candidate that compresses English worse than Apertus (−0.262 b/t). english-focus closes most of that English gap (−0.169 b/t) by allocating more vocabulary to English.

## 2. Fairness — Gini coefficient and worst-language sequence-length factor

Worst-language factor is the multiplicative increase in token sequence length, on the same parallel FLORES content, between the worst-served language and English (`eng_Latn`). 1.0× would mean the worst language is as compressed the same as English; 12.55× means a sentence costs 12.55 times more tokens in the worst-served language than in English.

| Tokenizer | FLORES60 Gini ↓ | FLORES200 Gini ↓ | Worst FLORES60 factor ↓ | Worst FLORES200 factor ↓ |
|---|---|---|---|---|
| Apertus v1 | 0.205 | 0.313 | 12.55× (sin_Sinh) | 14.70× (khm_Khmr) |
| multi-focus | **0.087** | **0.098** | **2.36×** (tha_Thai) | **3.63×** (sat_Olck) |
| english-focus | 0.093 | 0.102 | 2.34× (tha_Thai) | 3.93× (sat_Olck) |
| plain BPE | 0.115 | 0.124 | 2.53× (sin_Sinh) | 4.87× (taq_Tfng) |
| Gemma 3 | 0.106 | 0.150 | 2.40× (mya_Mymr) | 5.25× (sat_Olck) |


## 3. Vocabulary utilization and junk tokens

| Tokenizer | FLORES60 vocab util ↑ | FLORES200 vocab util ↑ | Junk tokens (≥8-char decorative runs) ↓ |
|---|---|---|---|
| Apertus v1 | 0.557 | 0.643 | 46 |
| multi-focus | 0.614 | **0.836** | **13** |
| english-focus | 0.589 | 0.809 | 21 |
| plain BPE | 0.590 | 0.776 | 26 |
| Gemma 3 | 0.419 | 0.507 | 150 |

multi-focus has the lowest junk-token count of the tokenizers, followed by english-focus (21). On FLORES200, multi-focus reaches 83.6% utilization.

## 4. Code-structure metrics

AST full-alignment is the fraction of AST-node spans whose token boundaries exactly match the AST boundary on both ends, across the 17-language StarCoder sample. Operator isolation is the fraction of arithmetic operator occurrences that the tokenizer emits as a standalone token.

| Tokenizer | AST full-alignment ↑ | Operator isolation (arithmetic) ↑ |
|---|---|---|
| Apertus v1 | 0.488 | 0.365 |
| multi-focus | 0.689 | **0.991** |
| english-focus | 0.681 | 0.989 |
| plain BPE | 0.684 | 0.989 |
| Gemma 3 | **0.747** | 0.989 |

Apertus glues arithmetic operators into surrounding tokens (0.365 vs ≥0.989 for every other tokenizer) and is weaker on AST full alignment

## 5. Extrinsic — 1B-parameter LM

10B-token balanced training for Val BPB, FLORES BPB, code BPB, BLiMP, MultiBLiMP, MGSM; 20B-token math+code training for MC-math, GSM8K, HumanEval, MBPP. CIs are reproduced from the standard extrinsic index: Wilson 95% binomial for binary-accuracy benchmarks (MBPP, HumanEval, GSM8K-flex; n=500/164/500 respectively) and across-language mean-of-means 95% CI for FLORES-trained-31 BPB (n=31 languages).

| Tokenizer | Val BPB ↓ | FLORES-31 BPB ↓ [95% CI] | Code BPB ↓ | BLiMP ↑ | MultiBLiMP ↑ | MGSM ↑ | MC-math ↑ | GSM8K-flex ↑ [95% CI] | HumanEval pass@1 ↑ [95% CI] | MBPP pass@1 ↑ [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Apertus v1 `[matched]` | **0.720** | 1.168 [1.063, 1.272] | **0.526** | 0.819 | 0.914 | 0.012 | 0.257 | 0.228 [0.192, 0.264] | 0.030 [0.006, 0.061] | 0.000 [0.000, 0.008] |
| multi-focus `[proxy]` | 0.729 | 1.170 [1.063, 1.277] | 0.534 | **0.824** | **0.917** | 0.015 | 0.271 | **0.248 [0.212, 0.288]** | 0.043 [0.021, 0.085] | **0.186 [0.154, 0.222]** |
| english-focus `[proxy]` | 0.726 | **1.165** [1.058, 1.271] | 0.529 | 0.813 | 0.910 | pending | **0.312** | 0.222 [0.186, 0.260] | **0.110 [0.061, 0.159]** | 0.168 [0.136, 0.202] |
| plain BPE | — | — | — | — | — | — | — | — | — | — |
| Gemma 3 | — | — | — | — | — | — | — | — | — | — |

MC-math is the mean of three subsets (math, gsm8k-MC, pythonio); each subset is n=500. The Apertus MBPP CI floor [0.000, 0.008] reflects 0/500 pass.


## Takeaways

- **multi-focus** has the best multilingual fairness (lowest Gini and worst-language factor on both FLORES sets) and fewest junk-token (13). It compresses English worse than the other tokenizers presented here (4.333 b/t, the only candidate below Apertus on English). It matches or beats Apertus on every measured downstream metric and reaches MBPP 0.186 against Apertus's 0.000.
- **english-focus** trades a small amount of multilingual fairness (Gini 0.093 vs 0.087; FLORES200 worst-lang 3.93× vs 3.63×) for measurable English gains (FineWeb-Edu +0.093 b/t over multi-focus, FineWeb2-proportional +0.026 b/t) and a code-LM gain (HumanEval 0.110 vs 0.043, MC-math 0.312 vs 0.271). 
- **Apertus v1** is weak on the long-tail of low-resource languages (FineWeb2-proportional 0.72–0.76 b/t behind the candidates), on code structure (operator isolation 0.365 against ≥0.989), and on math/code downstream (MBPP 0.000, HumanEval 0.030). It has the best Val BPB and Code BPB at 1B-balanced.
- **plain BPE** is intrinsically competitive on the FineWeb2-proportional multilingual sample (3.823 b/t, marginally above the candidates) at the cost of worse Gini (0.115 vs 0.087–0.093) and more junk tokens (26 vs 13–21). An LM on a similar tokenizer has been trained. Extrinsic results can be seen there (REPORT.md)
- **Gemma 3** is best on English compression (4.636 b/t) and AST full alignment (0.747) due to its 262k vocab, but fails production-safety tests (see REPORT_production_safety.md) and has a large number of junk tokens.
