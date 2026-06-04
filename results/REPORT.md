# Tokenizer comparison for Apertus

Purpose: choose a tokenizer for Apertus. The decision axes are (i) distributional quality (enabling better downstream language modeling); (ii) **multilingual efficiency & fairness**; (iii) **production safety** (it must not silently corrupt text or emit unknown tokens). This is my recommendation and the evidence it's based off of.

**How to read:** ↑ = higher is better, ↓ = lower is better. Sanity verdicts are **pass** (no issue) / **warn** (advisory, not disqualifying) / **fail** (disqualifying defect) / **n/a** (check doesn't apply, neutral). All evaluated tokenizers are GPT-2-style **byte-level** (so a 'byte coverage' or 'byte encoding' column would be constant and is omitted). Metric definitions, datasets, and the design matrix are in *Methods and metrics*; exhaustive tables in the appendix.

Disclaimer: Claude helped compile this report from all of my analyses. All numbers are computed and inserted programmatically, but if something seems off, please flag it.

## Contents

- [Recommendation](#recommendation)
- [Metric guide](#metric-guide)
- [Candidate comparison](#candidate-comparison)
  - [Candidates and references across FLORES sets](#candidates-and-references-across-flores-sets)
- [Trade-offs](#trade-offs)
- [Missing evidence](#missing-evidence)
- [Terminology](#terminology)
- [Methods and metrics](#methods-and-metrics)
  - [Metrics legend](#metrics-legend)
  - [Tokenizer design matrix](#tokenizer-design-matrix)
- [Appendix — extrinsic (downstream LM) details](#appendix--extrinsic-downstream-lm-details)
- [Related documents](#related-documents)

## Recommendation

I recommend PA-Clean-capped as the default. Against the current Apertus tokenizer it is fairer across languages (Gini, the inequality of per-language encoding cost, 0.081 against 0.205; lower is fairer) and encodes the multilingual text more compactly (about 0.023 against 0.0198 FLORES sentences per token), and it matches Apertus on downstream modeling quality over the training languages (FLORES bits-per-byte, trained languages, 1.169 against 1.168). It has the highest candidate math score (MC-math 0.295) and the highest candidate Python code-generation score (MBPP pass-rate 0.190); its clean-multi pretokenizer keeps math operators as separate tokens (operator-isolation 0.99).
PA-Apertus-capped is the better fit where multilingual modeling matters more than code. It has the same fairness (Gini 0.081), but its pretokenizer merges operators with operands (operator-isolation 0.50), which lowers code generation (MBPP 0.058, below the clean-multi candidates at p_BH<0.001).
SuperBPE-clean-cap-hw encodes English most compactly (5.01 bytes per token, against 4.24 for PA-Clean-capped), and its downstream quality is now measured: Val BPB 0.732 (against 0.729 for PA-Clean-capped), FLORES bits-per-byte over the trained languages 1.161, and MBPP pass-rate 0.196. It is less fair (Gini 0.106 against 0.081), which is the main reason it is not the default.
For some use cases, the cost is a little English compression: the parity-aware candidates encode English at 4.24–4.34 bytes per token, against Apertus's 4.60. On math and code the candidates are ahead of Apertus: PA-Clean-capped has MC-math 0.295 (against Apertus's 0.257) and MBPP pass-rate 0.190 (against 0.000).
Disqualified by a production-safety fail: Gemma 3, EuroLLM (see *Production-safety gates*).

## Metric guide

Full definitions in *Methods and metrics*.
- **sent/tok** — FLORES sentences (lines) per token; higher = more multilingual compression.
- **Gini** — cross-language inequality of per-language token cost; lower = more equal.
- **vocab-util CoV** — cross-language variation in vocabulary use; lower = more even.
- **Avg langs/token** — for each learned merge token used at least once, the number of languages it appears in, averaged across used merge tokens; higher = more cross-language sharing. Range [1, n_languages].
- **Eng B/tok** — FineWeb-Edu English bytes per token; higher = more compression.
- **AST align** — fraction of tree-sitter AST nodes (identifier / keyword / operator / etc., pooled) whose start and end byte offsets both fall on a tokenizer boundary on StarCoder snippets across 19 programming languages; higher = the tokenizer respects code syntax more often. Computed on the core pipeline only (the metric uses its own code corpus and is independent of the natural-language subset).
- **Val BPB / FLORES BPB** — downstream LM bits per byte; lower = better. The headline FLORES BPB is the macro-mean over the 31 training languages; the full 214-language macro is in the appendix. The FLORES BPB cell shows `mean [lo, hi]` with an across-language 95% CI (mean ± 1.96·stdev/√n); the FLORES BPB σ column reports the across-language stdev (how much BPB varies across languages). Val BPB has no per-language decomposition in the candidates' meta files, so no stdev is shown for it.
- **MC-math / MBPP** — downstream math and code scores; higher = better.
- **gate** — production-safety verdict (pass/warn/fail); fail disqualifies.

## Candidate comparison

The recommended tokenizers and the current Apertus production baseline, on the decision metrics. The intrinsic columns are computed on the broad FLORES set. Val BPB, FLORES BPB, MC-math, and MBPP come from the downstream language models. FLORES BPB here is the macro-mean over the 31 FLORES languages in the LM training set; the full 214-language macro is in the appendix. `pending` means the run is mapped but not yet measured, and `—` means not run.

| Tokenizer | Role | Multiling. sent/tok ↑ | Gini ↓ | Vocab-util CoV ↓ | Avg langs/token ↑ | Eng B/tok ↑ | Vocab util ↑ | AST align ↑ | Val BPB ↓ | FLORES BPB (trained) [95% CI] ↓ | FLORES BPB σ (trained) ↓ | MC-math ↑ | MBPP ↑ [95% CI] | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CleanV1-pretok + PA-BPE | default | 0.0232 | 0.081 | 0.4138 | 2.79 | 4.24 | 0.605 | 0.686 | 0.729 | 1.169 [1.061, 1.277] | 0.306 | 0.295 | 0.190 [0.156, 0.226] | warn |
| CleanV2-pretok + PA-BPE | provisional | 0.0233 | 0.081 | 0.4132 | 2.79 | 4.26 | 0.607 | 0.686 | 0.729 | 1.171 [1.063, 1.278] | 0.307 | 0.311 | 0.200 [0.166, 0.236] | warn |
| CleanV3-pretok + PA-BPE (rebalanced data) | provisional | 0.0233 | 0.087 | 0.4212 | 2.74 | 4.26 | 0.625 | 0.688 | 0.729 | pending | pending | — | — | warn |
| CleanV3-pretok + PA-BPE (base parity, rebalanced data) | provisional | 0.0217 | 0.095 | 0.4352 | 2.79 | 3.18 | 0.559 | 0.728 | — | — | — | — | — | warn |
| Apertus-pretok + PA-BPE | conditional | 0.0233 | 0.081 | 0.4130 | 2.79 | 4.34 | 0.606 | 0.551 | 0.729 | 1.170 [1.064, 1.277] | 0.303 | 0.270 | 0.058 [0.038, 0.080] | warn |
| CleanV1-pretok + PA-BPE + SuperBPE | conditional | 0.0227 | 0.106 | 0.4892 | 3.02 | 5.01 | 0.550 | 0.641 | 0.732 | 1.161 [1.056, 1.266] | 0.299 | 0.268 | 0.196 [0.162, 0.230] | warn |
| Apertus v1 (production) | baseline | 0.0198 | 0.205 | 0.5133 | 2.86 | 4.60 | 0.561 | 0.488 | 0.720 | 1.168 [1.063, 1.272] | 0.297 | 0.257 | 0.000 [0.000, 0.000] | warn |

`warn` is advisory: for NFC tokenizers exact-match below 1.0 is canonical re-spelling, not loss. MBPP has a paired-bootstrap 95% CI; MC-math is a single run.

Across the 10 tokenizers with both numbers, Spearman ρ(AST align, MBPP) = +0.721 (p = 0.019). AST alignment is on StarCoder snippets (multi-language); MBPP is Python pass-rate at 1B tokens, so the relationship is indicative, not a guarantee.

### Candidates and references across FLORES sets

This table shows multilingual compression (sent/tok), fairness (Gini), and vocabulary utilization for every candidate and reference at all three FLORES sets (core, broad, full). The full intrinsic tables, with every column, are in the appendix.

| Tokenizer | sent/tok ↑ (core) | sent/tok ↑ (broad) | sent/tok ↑ (full) | Gini ↓ (core) | Gini ↓ (broad) | Gini ↓ (full) | Vocab util ↑ (core) | Vocab util ↑ (broad) | Vocab util ↑ (full) |
|---|---|---|---|---|---|---|---|---|---|
| Apertus-pretok + PA-BPE | 0.0252 | 0.0233 | 0.0203 | 0.068 | 0.081 | 0.093 | 0.252 | 0.606 | 0.851 |
| CleanV1-pretok + PA-BPE | 0.0251 | 0.0232 | 0.0201 | 0.067 | 0.081 | 0.098 | 0.252 | 0.605 | 0.853 |
| CleanV2-pretok + PA-BPE | 0.0252 | 0.0233 | 0.0203 | 0.066 | 0.081 | 0.093 | 0.252 | 0.607 | 0.854 |
| CleanV3-pretok + PA-BPE (rebalanced data) | 0.0255 | 0.0233 | 0.0204 | 0.066 | 0.087 | 0.098 | 0.263 | 0.625 | 0.849 |
| CleanV3-pretok + PA-BPE (base parity, rebalanced data) | 0.0235 | 0.0217 | 0.0186 | 0.092 | 0.095 | 0.107 | 0.240 | 0.559 | 0.776 |
| Apertus-pretok + PA-BPE + SuperBPE | 0.0259 | 0.0230 | 0.0212 | 0.085 | 0.110 | 0.102 | 0.266 | 0.544 | 0.757 |
| CleanV1-pretok + PA-BPE + SuperBPE | 0.0255 | 0.0227 | 0.0208 | 0.080 | 0.106 | 0.103 | 0.266 | 0.550 | 0.776 |
| Apertus v1 (production) | 0.0275 | 0.0198 | 0.0142 | 0.071 | 0.205 | 0.313 | 0.344 | 0.561 | 0.648 |
| Gemma 3 | 0.0302 | 0.0244 | 0.0193 | 0.055 | 0.106 | 0.150 | 0.222 | 0.430 | 0.520 |
| GLM | 0.0225 | 0.0126 | 0.0116 | 0.206 | 0.379 | 0.354 | 0.251 | 0.347 | 0.405 |
| Kimi | 0.0217 | 0.0163 | 0.0144 | 0.153 | 0.199 | 0.213 | 0.173 | 0.225 | 0.275 |
| Qwen 3 | 0.0223 | 0.0136 | 0.0131 | 0.181 | 0.320 | 0.280 | 0.228 | 0.314 | 0.373 |
| Qwen 3.5 | 0.0295 | 0.0211 | 0.0160 | 0.099 | 0.180 | 0.242 | 0.234 | 0.379 | 0.445 |
| EuroLLM | 0.0276 | 0.0121 | 0.0116 | 0.066 | 0.459 | 0.402 | 0.363 | 0.665 | 0.758 |
| Llama 4 | 0.0302 | 0.0228 | 0.0172 | 0.071 | 0.153 | 0.221 | 0.273 | 0.480 | 0.559 |
| OLMo 2 | 0.0183 | 0.0114 | 0.0109 | 0.215 | 0.353 | 0.339 | 0.206 | 0.277 | 0.342 |
| K2 Think | 0.0223 | 0.0136 | 0.0131 | 0.181 | 0.320 | 0.280 | 0.228 | 0.314 | 0.373 |

## Trade-offs

- **Parity-aware vs plain BPE.** Parity-aware BPE has a lower Gini than plain BPE (0.081 against 0.114 at matched settings), at a small Eng B/tok cost (4.24 against 4.43). Use parity-aware BPE.
- **PA-BPE vs SuperBPE.** The SuperBPE stage raises Eng B/tok by 18–25%, but it also raises Gini (0.081 to 0.106) and lowers vocab utilization (0.605 to 0.550); on the apertus base it lowers MBPP (0.058 to 0.004). The added tokens are used mostly for space-delimited languages and rarely for CJK, Indic, or Thai.
- **clean-multi vs apertus.** clean-multi has a higher operator-isolation (0.99 against 0.50) and a higher MBPP (0.190 against 0.058). Use clean-multi unless multilingual FLORES BPB matters more than code generation.
- **Capped vs uncapped.** The capped regex produces fewer junk tokens (28 against 64) and has no dead-vocab warning, with no change to Eng B/tok or Val BPB. Keep capping enabled.
- **Hybrid-window vs base parity.** Hybrid-window has a higher Eng B/tok (4.24 against 3.13) with no fairness gain for base parity. Use hybrid-window.

## Missing evidence

- **SuperBPE-gpt4-fw2full-hw has no standard run yet**, so its Val and FLORES BPB are pending. Its math+code run is in (MC-math 0.265, MBPP 0.070).
- **Extrinsic coverage is uneven.** Not every tokenizer has both a standard and a math+code run, and MultiBLiMP and MGSM are missing for some; where a matched run is absent, the ablation rests on the 1B proxy.
- **Extrinsic numbers are single runs without seed variance.** MBPP is the exception (it has a paired-bootstrap CI). The MC-math difference between candidates (0.295 for PA-Clean-capped against 0.270 for PA-Apertus-capped) has no CI and is small.
- **The LMs are small proxies.** Whether the parity-aware BPE fairness difference or the SuperBPE Eng B/tok difference is larger at the target model scale is not measured here.
- **Vocabulary size is not swept.** Every candidate is near 128–131k, and size varies only alongside the SuperBPE transition rows, so it is confounded with the superword stage. A 64k/128k/256k sweep on one fixed design would separate the two.
- **NFC against no-NFC is not isolated** (the single noNFC tokenizer also differs in base and algorithm), and the references are not vocab-size-matched to the candidates, so the reference compression differences partly reflect vocabulary size. Newer algorithms are deferred because their code is not yet production-grade (see Methods).
## Terminology

Tokenizer-design jargon used throughout this report. Full definitions of the metrics and gates are in *Methods and metrics* later in the document.

**Pretokenization and algorithm**
- **Pretok / pretokenizer** — the regex that splits raw text into pre-tokens before BPE merges run. Two families appear in this report: `Apertus-pretok` (GPT-4-class) and `CleanVN-pretok` (the multilingual-safe "clean-multi" family, V1 baseline → V2 with apostrophe forward-attach → V3 with apostrophe forward + trailing attach).
- **NFC** — Unicode Normalization Form C. Rewrites visually-identical characters into their canonical composed form (e.g. `é` always stored as one codepoint, never as `e` + combining-acute). All candidates apply NFC before pretokenization; Apertus v1 production does not.
- **BPE** — byte-pair encoding; the standard merging algorithm.
- **PA-BPE** — parity-aware BPE. Merges biased toward low-resource languages so they get proportionally more vocab capacity than under plain frequency-driven BPE.
- **Hybrid+window (HW)** — PA-BPE training mode with a global-merge warmup phase under a moving window before parity-driven merging takes over. Production target for the candidates in this report.
- **SuperBPE** — a stage-2 extension on top of any BPE base. Drops word boundaries and learns cross-word merges (e.g. `def main` as one token).

**Evaluation corpora**
- **FLORES60** — 60-language slice of FLORES used as the broad headline set.
- **FLORES205** (devtest) — full 205-language FLORES+ set used for long-tail analysis.
- **FineWeb-Edu** — English web corpus used for English-compression measurement.
- **StarCoder** — multi-language code corpus (19 languages) used for AST boundary alignment analysis.

**Intrinsic metrics**
- **sent/tok** — FLORES sentences per token. Higher = denser encoding.
- **Gini** — inequality in per-language encoding cost. 0 = every language costs the same per byte; 1 = one language monopolises all the merges. Lower is fairer.
- **Vocab-util / vocab utilization** — fraction of the 128k vocab actually emitted by at least one language. Higher = more vocab slots doing productive work.
- **Vocab-util CoV** — coefficient of variation in per-language vocab usage. Lower = vocab is exercised more evenly across languages.
- **Avg langs/token** — for each merged-vocab token that gets used, how many languages it appears in (averaged). Range [1, n_languages].
- **AST align** — fraction of tree-sitter AST nodes (identifiers, keywords, operators, etc.) whose start and end byte offsets both fall on tokenizer boundaries. Higher = the tokenizer respects code syntax more often.

**Extrinsic (downstream LM) metrics**
- **BPB** — bits per byte. Lower = better per-byte fit.
- **Val BPB** — bits per byte on the LM training-mix validation set.
- **FLORES BPB** — bits per byte on FLORES sentences; reported here as a macro-mean over the 31 trained languages with an across-language 95% CI.
- **MBPP** — Mostly Basic Python Problems; 500-problem Python code-generation eval scored as pass@1. CIs are paired bootstraps over the 500 problems.
- **MC-math** — multiple-choice math eval; aggregate over GSM8K, MATH, and Python-IO (1500 problems total). CIs are Wilson 95% binomial.
- **HumanEval** — 164-problem Python code-generation eval; pass@1.

**Tokenizer-design data configs** (per-family `ratio` weighting in the training corpus)
- **tuned** — hand-tuned per-family weighting (European family ×1.2, drop two data-quality failures, regroup script-mismatched languages into `semitic`). The deck baseline.
- **rebalanced** — principled per-family weighting derived from a `max(data volume, speaker count)` formula. Drops the European boost.
- **untuned** — baseline weighting from FLORES line lengths; no adjustments.

---

## Methods and metrics

This section gives the datasets, the full metric definitions, the safety-gate definitions, and the tokenizer design matrix. The decision sections above use the short metric guide; this section is the complete reference.

**FLORES evaluation sets** (three multilingual FLORES/FLORES+ corpora of increasing breadth):
- **core** — 13 high-resource languages (FLORES dev split, 997 sentences/language).
- **broad** — 60 languages spanning high-to-mid resource levels (FLORES dev split, 997 sent/lang). The main study set: the headline numbers and the ablations are computed on it.
- **full** — all available FLORES+ languages (devtest split, 1012 sent/lang). The widest multilingual view.

### Metrics legend

**Efficiency / fairness (from the analysis runs):**
- **Eng comp (B/tok) ↑** — FineWeb-Edu *English* compression, **bytes per token** (more bytes/token = more compression). Measured standalone on a FineWeb-Edu English snippet; English-only.
- **Multiling. sent/tok ↑** — average **FLORES parallel sentences (lines) encoded per token** (more = more multilingual compression). This is the library's native `compression_rate` for the line-measured FLORES run, reported as-is (so it points the same way as the English column: higher = better). Values are small (~0.02–0.05; the reciprocal is tokens/sentence). Computed on the run's language set; **not** comparable to the English bytes/token column (different unit & corpus) — compare within the column.
- **Special toks** — count of tokens the tokenizer adds outside its learned vocabulary: declared special tokens (`<bos>`, `<eos>`, `<unk>`, `<pad>`, chat markers) plus reserved/control tokens (`<unused123>`, `[multimodal]`). Read from the tokenizer's own metadata (`added_tokens` / `all_special_ids`), not guessed from surface form. These are excluded from the *Vocab util*, *Junk*, and *Scaffold/Unseen* statistics.
- **Vocab util ↑** — fraction of the **learned** vocabulary (special/reserved tokens excluded from the denominator) that appears when encoding the corpus (corpus-dependent; differs between runs, as expected).
- **Vocab-util CoV ↓** — coefficient of variation of per-language vocab utilization (lower = each language gets a similarly sized share of the vocabulary).
- **Avg langs/token ↑** — cross-language token-sharing metric. For each learned merge token used at least once on the multilingual corpus, count the distinct languages it is emitted in (threshold `K=1`, any occurrence); average across used merge tokens. Single-character base tokens (the byte-level 256-byte alphabet) and declared special/reserved tokens are excluded, so the metric reflects *learned* cross-language sharing rather than structural byte coverage. Range `[1, n_languages]`; higher = more sharing. Reported on each FLORES set independently (the corpus determines the language set).
- **Gini ↓** — cross-language fairness of byte-normalized token cost: 0 = every language equally cheap to encode, 1 = maximally unfair.
- **CER ↓** — character error rate of encode→decode round-trip (0 = perfect). Severity companion to *Lossless* below (which measures how *often*, not how *much*).
- **Boundary-cross ↓** — fraction of tokens that fuse bytes across a UTF-8 character boundary (unrecoverable merges). Concentrates in multi-byte scripts (CJK/Indic/Arabic/emoji). The global average is mostly ASCII, so it sits near 0; see the per-language faceted plots.
- **Operator-isol ↑** — fraction of math operators tokenized standalone (vs attached to operands); near 1.0 = clean operator separation (helps arithmetic).
- **Enc ms/seq ↓** — mean wall-clock encoding time per sequence (line), milliseconds, from the analysis run (main table only). **Hardware/run-dependent** — it shifts with machine load between runs, so read it as a rough relative indicator within one table, not an absolute benchmark.

**Production safety gates (sanity check):**
- **Lossless ↑** — exact-match round-trip rate. For **NFC** tokenizers <1.0 is *expected* (NFC canonical-composition rewrites, not corruption; CER stays ~0); no-normalizer tokenizers reach 1.0.
- **UNK ↓** — global rate of unknown tokens (0 across all here = good).
- **Byte coverage** — all 256 byte values round-trip (pass/fail). This is the round-trip test: a byte counts as covered if `decode(encode(b))` reproduces it, even if the encoder reaches it through a multi-token fallback. See *Byte-alphabet missing* for the stricter vocab-presence check.
- **Byte-alphabet missing ↓** — count of byte values that are not present as their own standalone single-token vocab entry. Round-trip can still succeed via multi-token fallback (and *Byte coverage* will say `pass`), but missing valid UTF-8 lead bytes (0xC2–0xF4) fragment tokenization for characters in Supplementary Unicode planes (rare CJK extensions, Linear B, Cuneiform, etc.) and leave the LM without a learned embedding for each byte. WARN above zero.
- **Determinism** — encoding is stable and reproducible (the same input produces the same tokens).
- **Whitespace** — whitespace survives round-trip (advisory/warn-only: WordPiece/SentencePiece are intentionally whitespace-lossy).
- **Per-script UNK** — flags any script with >1% UNK; *n/a* = tokenizer has no UNK token, so the check doesn't apply.
- **Dead vocab ↓** — count of vocabulary entries that can *never* be emitted under the tokenizer's own faithful pipeline, for either of two reasons: the **normalizer** rewrites the surface so the entry is unreachable, or the **pretokenizer** always splits the entry's surface into ≥2 pre-tokens so within-pretoken merges can never build it. (The pretokenizer case is skipped for SuperBPE-style tokenizers that merge across pretoken boundaries by design.) Either way the slot is permanently unreachable. Reported as a warning: the slot wastes vocabulary capacity but does not corrupt text or emit UNK.
- **Byte-frag (benign)** — count of sub-character byte-fragment tokens. **Normal and expected for byte-level BPE; NOT a defect**; informational, no direction-of-better.
- **Long toks (>64)** — count of vocabulary tokens longer than 64 chars (advisory/warn-only; examples in the appendix).
- **Junk toks (≥8) ↓** — count of vocabulary tokens that are runs of ≥8 punctuation/symbol/whitespace chars with no letters or digits (decorative separators / whitespace runs; low-value, wasted vocabulary; examples in the appendix).

### Tokenizer design matrix

This section explains the tokenizer settings, and for the ablations, why that design choice was worth testing. 

**Design dimensions:**

- **Algorithm (plain BPE vs parity-aware BPE vs SuperBPE vs Unigram LM)** — parity-aware BPE (PA-BPE), via the merge selection criteria, equalizes per-language encoding cost instead of following raw frequency. Ablated to test whether that fairness objective actually beats plain frequency-driven BPE on multilingual balance. **SuperBPE** is a distinct algorithmic axis: a two-stage scheme that runs a normal subword stage and then learns 'superword' merges spanning whitespace (its base and transition point are dimensions of their own, below). **Unigram LM** (a likelihood-pruned piece inventory rather than agglomerative merges) is carried only as a single-point ablation, not a full sweep.
- **Parity mode (hybrid-window vs base)** — base PA-BPE optimizes the single worst-off language at each step; the *hybrid-window* variant adds a global phase that prevents always selecting the same language. Ablated because the base variant allocates ~40–45% fewer merges to English and European; the question is whether hybrid-window corrects that while still improving multilingual equity.
- **Punctuation/whitespace capping (capped vs uncapped)** — *capped* bounds runs of punctuation/symbols/whitespace to ≤16 chars during pretokenization. Ablated because *uncapped* BPE merges long decorative runs (`----`, `====`, space runs) into single junk vocabulary tokens that waste slots; capping should remove that failure mode with little effect on real text.
- **Pretokenization family** — the regex that splits raw text into pre-tokens before BPE even runs (glossary below; full design writeup: [pretokenization design](../apertus_tokenizer_design.md)). Ablated because it dictates digit grouping, apostrophe/contraction handling, and CamelCase/script behavior. Each of these shifts multilingual fairness and arithmetic friendliness.
- **Training-data composition** — 30-language-*balanced* vs natural *FineWeb2-full* vs *tuned* (glossary below). Ablated because the corpus the tokenizer is *trained* on decides which languages get allocated vocabulary.
- **Parity tuning — European up-weighting (×1.2 vs ×1.1)** — how much the tuned config weights the European families up. The trainer selects the group/language with the minimum `compression_rate / ratio`, so a higher ratio gives more merges and more compression. ×1.2 weights English and European up (the base config allocates ~40–45% fewer merges to them); the ×1.1 variant uses a smaller weight. (See the parity-tuning ablation in the appendix.)
- **NFC normalization** — Unicode canonical composition applied before tokenizing. Most candidates use it; reference Apertus and the `noNFC` SuperBPE variant do not (see the *Lossless* caveat in the legend; NFC makes exact-match <1.0 *by design*, not corruption).
- **SuperBPE base & transition point** — SuperBPE is a two-stage 'superword' tokenizer; we record the **base** it was started from (PA-BPE vs plain BPE) and the stage-1→stage-2 *transition* vocab size (64k/90k). Ablated to see whether superwords help and whether the PA-BPE base keeps its fairness after the SuperBPE stage.

**Algorithms not evaluated this round.** Several newer tokenization algorithms look promising but are excluded here because their implementations are not yet production-grade: correctness, determinism, and serialization to a standard `tokenizer.json` are not all in place. We defer them to a later round rather than draw production conclusions from prototype code; this round covers BPE, parity-aware BPE, SuperBPE, and Unigram LM.

**Training-data compositions** (what the tokenizer was trained on; distinct from the FLORES/FineWeb-Edu corpora it is *evaluated* on):

- **balanced** — the 10 GB tokenizer-training mixture in `tokenizer-lm/configs/data/balanced.json`: 3.5 GB English (FineWeb-Edu), 3.0 GB multilingual (30 FineWeb2 languages), 1.5 GB math (FineMath-4+), 1.5 GB code (StarCoder). The 30 multilingual languages are sized in proportion to how much text each has, so most of the 3.0 GB goes to the high-resource ones (rus_Cyrl ~1.0 GB, tam_Taml ~0.004 GB). "Balanced" refers to the fixed split across domains (English is 35% of the total), not to an equal split across languages. Plain BPE, Unigram, and SuperBPE use this mixture as-is; the PA-BPE variants use the same mixture with a parity config (below).
- **FineWeb2-full** — the temperature sampled (t = 3) FineWeb2 multilingual distribution (most of the text is high-resource languages), with parity-aware *family* grouping but no hand-tuning.
- **FineWeb2-full (tuned)** — FineWeb2-full plus three targeted fixes from the intrinsic-analysis diagnosis: (1) European family ratios ×1.2 to weight English/European up; (2) drop two data-quality failures (`kas_Deva`, script purity 0.59; `lij_Latn`, 68% duplicate lines); (3) regroup script-mismatched languages (`ydd_Hebr` Hebrew-script; `kas/knc/uzs_Arab` Arabic-script) into the *semitic* group so they share script-appropriate merges. The **EU×1.1** ablation differs only in change (1).
- **balanced; transition Nk** (SuperBPE) — trained on the balanced mixture; *transition Nk* is the stage-1→stage-2 vocab size at which superword merges begin.

**Parity-aware BPE configs (how PA-BPE training is set up).** PA-BPE either treats training languages individually or puts them into linguistic groups (language families, here). Each group/language has a `quota_bytes` (how much of its data to read) and a `ratio` (its weight). At each step the trainer scores every group/language by `adjusted = compression_rate / ratio` and the base variant advances the group/language with the lowest `adjusted`. A higher `ratio` therefore gets a group/language selected more often, which gives it more merges and better compression. The presets set group vs. language and `ratio` differently:

- **balanced**: per-language. Ratios from FLORES+ bytes-per-line, targeting equal cost per language; as is standard, ratios are normalized w.r.t. English..
- **FineWeb2-full**: All FineWeb2 languages with more than 1000 samples (after quality-filtering) grouped into 25 linguistic-families. Ratio is determined using FLORES+ bytes-per-line from the portion of those languages with FLORES+ entries. Specifically, bytes-per-line for all the Flores+-available languages are computed and averaged, and normalized relative to English.
- **tuned**: FineWeb2-full with the three fixes above (European families ×1.2, two quality removals, semitic regroup); EU×1.1 changes only the ×1.2.

In every preset the math and code groups are heuristically fixed at `ratio` 1.0, since they have no parallel FLORES+ data to derive one from.

`hybrid-window` adds a global phase and a window so the trainer does not keep selecting the same language; `base` is the plain lowest-`adjusted` rule.

**Pretokenization families** (the regex that splits text into pre-tokens before BPE; it bounds which merges are possible). One line each below; the full rationale and exact stage-1/stage-2 regexes, including why the **current direction is clean-multi**, are in the [pretokenization design writeup](../apertus_tokenizer_design.md).

- **gpt2** — GPT-2 regex: English contractions, no digit-run cap, no script-awareness.
- **gpt4 / gpt4o** — CamelCase splitting, digits capped `{1,3}`; gpt4o is the multilingual o200k-style variant.
- **apertus** — Mistral-Nemo scheme (verified from Apertus-70B-2509): single-digit splitting (arithmetic-friendly for *numbers*), CamelCase, no contraction handling. Note this is separate from operator handling: apertus has low operator-isolation (operators tokenized together with operands), which lowers MBPP (see the *Pretokenizer family* ablation).
- **clean-multi** *(current direction)* — apertus word arms but a **space-only word prefix** (apostrophes/punctuation don't attach forward: `don't` → `don | ' | t`) and **no trailing-char fusion**, with a matching reduced SuperBPE stage-2 (words removed, single digits and single punctuation kept isolated).
- **right-aligned digits** — digits grouped right-to-left (Singh & Strouse 2024).
- **capped (suffix on any family)** — punctuation/symbol and whitespace runs bounded to `{1,16}`, so BPE can't build long decorative-junk tokens; byte-identical on normal text/code/math.

**Reference matrix** — all tokenizers in one table (columns map to the dimensions above):

| Tokenizer | Type | Algorithm | Base / parity-mode | Pretok | NFC | Capping | Training data |
|---|---|---|---|---|---|---|---|
| Apertus-pretok + PA-BPE | Candidate | Parity-aware BPE | hybrid-window | apertus | NFC | capped | FineWeb2-full (tuned) |
| CleanV1-pretok + PA-BPE | Candidate | Parity-aware BPE | hybrid-window | clean-multi | NFC | capped | FineWeb2-full (tuned) |
| CleanV2-pretok + PA-BPE | Candidate | Parity-aware BPE | hybrid-window | clean-multi-plus2 (apostrophe/right-curly attach + tsek-attach) | NFC | capped | FineWeb2-full (tuned) |
| CleanV3-pretok + PA-BPE (rebalanced data) | Candidate | Parity-aware BPE | hybrid-window | clean-multi-plus3 (plus2 + trailing-apostrophe attach, guarded) | NFC | capped | FineWeb2-full (tuned consv2: v6 reweighting, D_REF=10GB, S_REF=50M, taikadai_cap=2.0) |
| CleanV3-pretok + PA-BPE (base parity, rebalanced data) | Candidate | Parity-aware BPE | base (no window) | clean-multi-plus3 (plus2 + trailing-apostrophe attach, guarded) | NFC | capped | FineWeb2-full (tuned consv2: v6 reweighting, D_REF=10GB, S_REF=50M, taikadai_cap=2.0) |
| Apertus-pretok + PA-BPE + SuperBPE | Candidate | SuperBPE | PA-BPE base (apertus-capped, hw) | apertus-capped | NFC | capped | FineWeb2-full (tuned); transition 90k |
| CleanV1-pretok + PA-BPE + SuperBPE | Candidate | SuperBPE | PA-BPE base (clean-capped, hw) | clean-multi-capped | NFC | capped | FineWeb2-full (tuned); transition 90k, vocab 128k |
| Apertus v1 (production) | Reference | production: swiss-ai/Apertus-70B-2509 | — | — | none | — | — |
| Gemma 3 | Reference | production: google/gemma-3-1b-it | — | — | — | — | — |
| GLM | Reference | production: THUDM/glm-4-9b-chat | — | — | — | — | — |
| Kimi | Reference | production: moonshotai/Kimi-K2-Instruct-0905 | — | — | — | — | — |
| Qwen 3 | Reference | production: Qwen/Qwen3-8B | — | — | — | — | — |
| Qwen 3.5 | Reference | production: Qwen/Qwen3.5-35B-A3B | — | — | — | — | — |
| EuroLLM | Reference | production: utter-project/EuroLLM-1.7B-Instruct (same tokenizer as 9B/22B) | — | — | — | — | — |
| Llama 4 | Reference | production: meta-llama/Llama-4-Scout-17B-16E-Instruct (via open mirror unsloth/...) | — | — | — | — | — |
| OLMo 2 | Reference | production: allenai/OLMo-2-1124-7B (OLMo-3 not yet on HF) | — | — | — | — | — |
| K2 Think | Reference | production: LLM360/K2-Think | — | — | — | — | — |
| SuperBPE(PA-base)·gpt4o·t90k | Ablation | SuperBPE | PA-BPE base (gpt4) | gpt4o + gpt4o-reduced | NFC | — | balanced; transition 90k |
| SuperBPE(PA-base)·clean-c3·t90k | Ablation | SuperBPE | PA-BPE base (clean-multi) | clean-multi C3 | NFC | — | balanced; transition 90k |
| PA-Clean-uncapped | Ablation | Parity-aware BPE | hybrid-window | clean-multi | NFC | uncapped | FineWeb2-full |
| BPE-Clean-capped | Ablation | Plain BPE | — | clean-multi | NFC | capped | FineWeb2-full (tuned) |
| BPE-Clean-uncapped | Ablation | Plain BPE | — | clean-multi | NFC | uncapped | balanced |
| PA-Clean-capped-base | Ablation | Parity-aware BPE | base (no window) | clean-multi | NFC | capped | tuned |
| PA-gpt4-balanced | Ablation | Parity-aware BPE | hybrid-window | gpt4 | NFC | uncapped | balanced |
| PA-gpt4-fineweb2full | Ablation | Parity-aware BPE | hybrid-window | gpt4 | NFC | uncapped | FineWeb2-full |
| Apertus-pretok + PA-BPE (European ×1.1) | Ablation | Parity-aware BPE | hybrid-window | apertus | NFC | capped | FineWeb2-full (tuned, EU×1.1) |
| Apertus-pretok + PA-BPE (untuned data) | Ablation | Parity-aware BPE | hybrid-window | apertus | NFC | capped | FineWeb2-full (original/untuned, EU×1.0) |
| Apertus-pretok + PA-BPE (no semitic regroup) | Ablation | Parity-aware BPE | hybrid-window | apertus | NFC | capped | FineWeb2-full (tuned, no semitic regroup) |
| SuperBPE(PA-base)·gpt4o·t64k | Ablation | SuperBPE | PA-BPE base (gpt4) | gpt4o | NFC | — | balanced; transition 64k |
| SuperBPE(PA-base)·clean-c2·t90k | Ablation | SuperBPE | PA-BPE base (clean-multi) | clean-multi C2 | NFC | — | balanced; transition 90k |
| SuperBPE(plain-base)·gpt4o·noNFC | Ablation | SuperBPE | plain-BPE base (gpt4o) | gpt4o | none | — | balanced; transition 90k |
| Unigram-gpt4o | Ablation | Unigram LM | — | gpt4o | — | — | balanced |
| BPE-rightalign | Ablation | Plain BPE | — | right-aligned digits | — | — | balanced |
| BPE-gpt2 | Ablation | Plain BPE | — | gpt2-style | — | — | balanced |
| SuperBPE·clean-cap·hw·fw2full·t110k/130k | Ablation | SuperBPE | PA-BPE base (clean-capped, hw) | clean-multi-capped | NFC | capped | FineWeb2-full (tuned); transition 110k, vocab 130k |
| SuperBPE·clean-cap·base·fw2full·t110k/130k | Ablation | SuperBPE | PA-BPE base (clean-capped, base) | clean-multi-capped | NFC | capped | FineWeb2-full (tuned); transition 110k, vocab 130k |
| SuperBPE·apertus-cap·base·fw2full | Ablation | SuperBPE | PA-BPE base (apertus-capped, base) | apertus-capped | NFC | capped | FineWeb2-full (tuned); transition 90k |
| SuperBPE·clean-cap·base·fw2full | Ablation | SuperBPE | PA-BPE base (clean-capped, base) | clean-multi-capped | NFC | capped | FineWeb2-full (tuned); transition 90k |
| SuperBPE·gpt4·hw·fw2full | Ablation | SuperBPE | PA-BPE base (gpt4, hw) | gpt4o | NFC | — | FineWeb2-full; transition 90k |
| SuperBPE·gpt4·base·fw2full | Ablation | SuperBPE | PA-BPE base (gpt4, base) | gpt4o | NFC | — | FineWeb2-full; transition 90k |

## Appendix — extrinsic (downstream LM) details

Small transformers trained from scratch on each tokenizer (companion `tokenizer-lm` project), then evaluated; the ablations above show the relevant rows.

**Training setup.**
- **Models:** nanochat-based transformers; every comparison is within a single vocabulary size, so transformer-matrix parameters and the token budget match across the pair. Token budget = 10.5 × (transformer matrices + lm_head), the Kaplan Chinchilla variant ablated by nanochat (includes embedding params, unlike the 20× rule; similar in practice). µP for LR transfer; fixed batch sizes; matrix LR 0.02 (5-point sweep).
- **Data mixture** (shared by tokenizer + LM training): 35% FineWeb-Edu (English), 30% filtered FineWeb2 (30 languages, top-33% quality), 15% FineMath-4+, 15% StarCoderData (top tier).
- **Budgets:** distributional/linguistic metrics from the **10B** balanced run (`full-128k-<slug>`, step ~8800); math+code from the **20B** math+code-from-scratch run (`-mathcode-scratch`, step 19073). The 10B-vs-20B table below justifies reading BPB/MC rankings off 10B for the bulk of the panel.
- **Metric notes:** BPB (bits-per-byte) is tokenizer-independent and zero-variance. Generative tasks (GSM8K/HumanEval/MBPP) are noisy single-run point estimates. **MBPP** separates the candidates (apertus≪clean, paired-bootstrap p_BH<0.001); **GSM8K-flexible and HumanEval do not separate them** even at 20B; HumanEval additionally sits on a greedy-decoding repetition floor (~50–65% of generations degenerate across all runs). Treat generative numbers as directional.

**Full per-tokenizer results** (point estimates; `[matched]`/`[proxy]`/`pending`/`—` as in the ablations; MBPP/GSM8K/HumanEval 95% CIs in `bootstrap_mathcode_significance.json`, MBPP CI shown with the ablations). **BLiMP is Option-B (BOS / empty-context) scoring for all rows.** The main eval files mix Option-A and Option-B, which are not comparable, so only Option-B is reported; `optA-only` flags a run that has no Option-B eval (its Option-A value is omitted, not substituted):
| Tokenizer | Val BPB ↓ | FLORES all [95% CI] ↓ | FLORES all σ ↓ | FLORES tr [95% CI] ↓ | FLORES tr σ ↓ | Code BPB ↓ | BLiMP ↑ | MultiBLiMP ↑ | MGSM ↑ | MC-math ↑ | GSM8K ↑ | HumanEval ↑ | MBPP ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Apertus-pretok + PA-BPE [matched] | 0.729 | 2.943 [2.793, 3.093] | 1.119 | 1.170 [1.064, 1.277] | 0.303 | 0.531 | 0.819 | 0.916 | 0.015 | 0.270 | 0.240 | 0.049 | 0.058 |
| CleanV1-pretok + PA-BPE [matched] | 0.729 | 2.965 [2.812, 3.118] | 1.141 | 1.169 [1.061, 1.277] | 0.306 | 0.533 | 0.816 | 0.919 | 0.013 | 0.295 | 0.232 | 0.055 | 0.190 |
| CleanV2-pretok + PA-BPE [matched] | 0.729 | 2.953 [2.801, 3.104] | 1.129 | 1.171 [1.063, 1.278] | 0.307 | 0.534 | 0.819 | 0.920 | 0.012 | 0.311 | 0.226 | 0.043 | 0.200 |
| CleanV3-pretok + PA-BPE (rebalanced data) [matched] | 0.729 | pending | pending | pending | pending | pending | pending | pending | pending | — | — | — | — |
| Apertus-pretok + PA-BPE + SuperBPE [matched] | 0.733 | 3.081 [2.920, 3.241] | 1.198 | 1.176 [1.069, 1.283] | 0.304 | 0.541 | 0.815 | 0.912 | 0.011 | 0.269 | 0.198 | 0.055 | 0.004 |
| CleanV1-pretok + PA-BPE + SuperBPE [matched] | 0.732 | 3.038 [2.879, 3.197] | 1.188 | 1.161 [1.056, 1.266] | 0.299 | 0.536 | 0.814 | 0.920 | 0.010 | 0.268 | 0.222 | 0.073 | 0.196 |
| Apertus v1 (production) [matched] | 0.720 | 2.768 [2.632, 2.904] | 1.015 | 1.168 [1.063, 1.272] | 0.297 | 0.526 | 0.819 | 0.914 | 0.012 | 0.257 | 0.228 | 0.030 | 0.000 |
| BPE-Clean-uncapped [matched] | 0.716 | 2.654 [2.517, 2.791] | 1.022 | 1.157 [1.052, 1.261] | 0.296 | 0.523 | 0.821 | 0.910 | 0.012 | 0.270 | 0.216 | 0.018 | 0.148 |
| BPE-Punct [matched] | 0.717 | 2.647 [2.511, 2.784] | 1.016 | 1.161 [1.059, 1.263] | 0.290 | 0.527 | 0.814 | 0.911 | 0.007 | — | — | — | — |
| BPE-gpt2 [matched] | 0.713 | 2.640 [2.505, 2.776] | 1.014 | 1.157 [1.056, 1.258] | 0.287 | 0.515 | 0.816 | 0.909 | 0.012 | — | — | — | — |
| BPE-gpt4o-balanced [matched] | 0.711 | 2.639 [2.503, 2.775] | 1.015 | 1.157 [1.054, 1.260] | 0.293 | 0.518 | 0.817 | 0.917 | 0.009 | — | — | — | — |
| BPE-gpt4o-balanced-NFC [matched] | 0.711 | 2.637 [2.500, 2.774] | 1.022 | 1.154 [1.049, 1.259] | 0.297 | 0.519 | 0.813 | 0.916 | 0.011 | — | — | — | — |
| BPE-rightalign [matched] | 0.712 | 2.644 [2.508, 2.780] | 1.016 | 1.160 [1.057, 1.264] | 0.293 | 0.519 | 0.816 | 0.912 | 0.012 | 0.295 | 0.252 | 0.061 | 0.062 |
| PA-Clean-balanced-hw [matched] | pending | pending | pending | pending | pending | pending | pending | pending | pending | — | — | — | — |
| PA-Clean-uncapped [matched] | 0.728 | 2.966 [2.812, 3.119] | 1.146 | 1.167 [1.061, 1.274] | 0.303 | 0.529 | 0.818 | 0.917 | 0.009 | — | — | — | — |
| PA-gpt4-balanced [matched] | 0.719 | 2.662 [2.527, 2.797] | 1.007 | 1.177 [1.071, 1.282] | 0.300 | 0.524 | 0.816 | 0.914 | 0.011 | — | — | — | — |
| PA-gpt4-fineweb2full [matched] | 0.728 | 2.962 [2.808, 3.116] | 1.147 | 1.169 [1.062, 1.275] | 0.303 | 0.531 | 0.827 | 0.914 | 0.012 | — | — | — | — |
| SuperBPE(PA-base)·clean-c2·t90k [matched] | 0.729 | 2.656 [2.521, 2.792] | 1.012 | 1.169 [1.066, 1.272] | 0.294 | 0.526 | 0.811 | 0.911 | 0.007 | — | — | — | — |
| SuperBPE(PA-base)·clean-c3·t90k [matched] | 0.730 | 2.650 [2.515, 2.784] | 1.006 | 1.173 [1.069, 1.277] | 0.295 | 0.531 | 0.803 | 0.919 | 0.007 | — | — | — | — |
| SuperBPE·clean-cap·hw·fw2full·t110k/130k [matched] | 0.732 | 2.993 [2.836, 3.150] | 1.171 | 1.161 [1.055, 1.266] | 0.300 | 0.534 | 0.821 | 0.912 | 0.008 | 0.288 | 0.236 | 0.104 | 0.202 |
| SuperBPE·gpt4·hw·fw2full [matched] | pending | pending | pending | pending | pending | pending | pending | pending | pending | 0.265 | 0.198 | 0.085 | 0.070 |
| SuperBPE(PA-base)·gpt4o·t64k [matched] | 0.729 | 2.658 [2.523, 2.794] | 1.012 | 1.180 [1.076, 1.284] | 0.295 | 0.530 | 0.792 | 0.920 | 0.006 | — | — | — | — |
| SuperBPE(PA-base)·gpt4o·t90k [matched] | 0.729 | 2.659 [2.523, 2.794] | 1.008 | 1.181 [1.077, 1.284] | 0.294 | 0.528 | 0.801 | 0.916 | 0.006 | — | — | — | — |
| SuperBPE(plain-base)·gpt4o·noNFC [matched] | 0.724 | 2.645 [2.510, 2.781] | 1.012 | 1.173 [1.069, 1.278] | 0.297 | 0.525 | 0.804 | 0.909 | 0.004 | — | — | — | — |
| Unigram-gpt4o [matched] | 0.731 | 2.686 [2.551, 2.821] | 1.008 | 1.190 [1.084, 1.297] | 0.303 | 0.554 | 0.833 | 0.911 | 0.015 | — | — | — | — |

**10B vs 20B stability (balanced mixture).** Five tokenizers continued from their 10B checkpoint for +10B on the same data. BPB ↓ better; BLiMP/GSM8K/HumanEval/MBPP/MGSM ↑ better. This is the justification for reporting most runs at 10B: **BPB/code-BPB rankings are budget-stable, generative-task rankings are not** (single runs, no CIs). BLiMP is Option-B (BOS) scoring; the 20B *-continue* runs have no Option-B eval, shown `—`.
| Tokenizer | Budget | Val BPB ↓ | FLORES ↓ | BLiMP ↑ | Code BPB ↓ | GSM8K ↑ | HEval ↑ | MBPP ↑ | MGSM ↑ |
|---|---|---|---|---|---|---|---|---|---|
| gpt4o-balanced | 10B | 0.711 | 2.64 | 0.817 | 0.518 | 0.046 | 0.006 | 0.030 | 0.009 |
| gpt4o-balanced | 20B | 0.698 | 2.62 | 0.813 | 0.507 | 0.056 | 0.024 | 0.052 | 0.017 |
| rightalign-balanced | 10B | 0.712 | 2.64 | 0.816 | 0.519 | 0.041 | 0.024 | 0.052 | 0.012 |
| rightalign-balanced | 20B | 0.698 | 2.63 | 0.824 | 0.508 | 0.065 | 0.037 | 0.060 | 0.014 |
| claude-balanced-nfc | 10B | 0.714 | 2.64 | 0.823 | 0.518 | 0.037 | 0.043 | 0.056 | 0.012 |
| claude-balanced-nfc | 20B | 0.701 | 2.65 | 0.820 | 0.509 | 0.056 | 0.012 | 0.070 | 0.006 |
| llama3 | 10B | 0.718 | 2.66 | 0.820 | 0.548 | 0.038 | 0.006 | 0.042 | 0.008 |
| llama3 | 20B | 0.704 | 2.67 | 0.820 | 0.560 | 0.055 | 0.006 | 0.060 | 0.005 |
| gpt4o-code | 10B | 0.724 | 2.63 | 0.821 | 0.543 | 0.046 | 0.018 | 0.020 | 0.011 |
| gpt4o-code | 20B | 0.709 | 2.60 | 0.827 | 0.545 | 0.064 | 0.024 | 0.028 | 0.011 |

## Related documents

- **[Design-choice ablations](REPORT_ablations.md)** — capping, parity-aware vs plain BPE, SuperBPE-on-PA-base, pretokenizer family, hybrid-window vs base, transition point, parity tuning, plus the SuperBPE-vs-its-base comparison and the family-faceted plots.
- **[Production-safety gates and vocabulary inspection](REPORT_production_safety.md)** — gate verdicts, round-trip fidelity, vocabulary-usage breakdown, long-token / junk-token / dead-vocab examples.
- **[Appendix: full intrinsic tables](REPORT_appendix_intrinsic.md)** — per-tokenizer × per-language numbers across the broad / core / full FLORES sets, with per-language plots.

---

