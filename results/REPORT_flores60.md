# Tokenizer comparison for multilingual-LLM selection

Purpose: choose a tokenizer for a production multilingual LLM. The decision axes are (1) **multilingual efficiency & fairness**, (2) **production safety** (it must not silently corrupt text or emit unknown tokens), and (3) distributional quality.

**How to read:** ↑ = higher is better, ↓ = lower is better. Sanity verdicts are **pass** (no issue) / **warn** (advisory, not disqualifying) / **fail** (disqualifying defect) / **n/a** (check doesn't apply — neutral). All evaluated tokenizers are GPT-2-style **byte-level** (so a 'byte coverage' or 'byte encoding' column would be constant and is omitted).

## Metrics legend

**Efficiency / fairness (from the analysis runs):**
- **Eng comp (B/tok) ↑** — FineWeb-Edu *English* compression, **bytes per token** (more bytes/token = denser English encoding). Measured standalone on a FineWeb-Edu English snippet; English-only.
- **Multiling. tok/sent ↓** — average **tokens per FLORES parallel sentence** (fewer = denser multilingual encoding). Computed on the run's language set; **not** comparable to the English bytes/token column (different units & corpus).
- **Vocab util ↑** — fraction of the vocabulary that appears when encoding the corpus (corpus-dependent; differs between the 60-lang and core runs, as expected).
- **Vocab-util CoV ↓** — coefficient of variation of per-language vocab utilization (lower = each language gets a similarly sized share of the vocabulary).
- **Gini ↓** — cross-language fairness of byte-normalized token cost: 0 = every language equally cheap to encode, 1 = maximally unfair.
- **CER ↓** — character error rate of encode→decode round-trip (0 = perfect). Severity companion to *Lossless* below (which measures how *often*, not how *much*).
- **Boundary-cross ↓** — fraction of tokens that fuse bytes across a UTF-8 character boundary (unrecoverable merges). Concentrates in multi-byte scripts (CJK/Indic/Arabic/emoji); the global value is diluted by ASCII — see the per-language faceted plots.
- **Operator-isol ↑** — fraction of math operators tokenized standalone (vs glued to operands); near 1.0 = clean operator separation (helps arithmetic).

**Production safety gates (sanity check):**
- **Lossless ↑** — exact-match round-trip rate. For **NFC** tokenizers <1.0 is *expected* (NFC canonical-composition rewrites, not corruption — CER stays ~0); no-normalizer tokenizers reach 1.0.
- **UNK ↓** — global rate of unknown tokens (0 across all here = good).
- **Byte coverage** — all 256 byte values round-trip (pass/fail).
- **Determinism** — encoding is stable/reproducible (same input → same tokens).
- **Whitespace** — whitespace survives round-trip (advisory/warn-only: WordPiece/SentencePiece are intentionally whitespace-lossy).
- **Per-script UNK** — flags any script with >1% UNK; *n/a* = tokenizer has no UNK token, so the check doesn't apply.
- **Dead vocab ↓** — count of vocabulary entries the tokenizer's own normalizer can *never* emit (permanently unreachable); drives FAIL verdicts.
- **Byte-frag (benign)** — count of sub-character byte-fragment tokens. **Normal and expected for byte-level BPE — NOT a defect**; informational, no direction-of-better.
- **Long toks (>64)** — count of vocabulary tokens longer than 64 chars (advisory/warn-only; examples in the appendix).
- **Junk toks (≥8) ↓** — count of vocabulary tokens that are runs of ≥8 punctuation/symbol/whitespace chars with no letters or digits (decorative separators / whitespace runs — low-value, wasted vocabulary).

## Tokenizer design matrix

What each tokenizer is. *Parity-aware BPE* = BPE trained to balance per-language encoding cost; *hybrid-window* vs *base* = parity variant; *capped* = punctuation/whitespace run-length caps (prevents decorative-junk tokens); *SuperBPE* = two-stage 'superword' tokenizer (note the **base** it was built on — PA-BPE vs plain BPE); *NFC* = Unicode canonical-composition normalization applied before tokenizing; pretok families (apertus/clean-multi/gpt4/gpt4o/gpt2) differ in their pre-tokenization regex.

| Tokenizer | Type | Algorithm | Base / parity-mode | Pretok | NFC | Capping | Training data |
|---|---|---|---|---|---|---|---|
| PA-Apertus-capped | Candidate | Parity-aware BPE | hybrid-window | apertus | NFC | capped | FineWeb2-full (tuned) |
| PA-Clean-capped | Candidate | Parity-aware BPE | hybrid-window | clean-multi | NFC | capped | FineWeb2-full (tuned) |
| SuperBPE(PA-base)·gpt4o·t90k | Candidate | SuperBPE | PA-BPE base (gpt4) | gpt4o + gpt4o-reduced | NFC | — | balanced; transition 90k |
| SuperBPE(PA-base)·clean-c3·t90k | Candidate | SuperBPE | PA-BPE base (clean-multi) | clean-multi C3 | NFC | — | balanced; transition 90k |
| Apertus | Reference | production: swiss-ai/Apertus-70B-2509 | — | — | none | — | — |
| Gemma 3 | Reference | production: google/gemma-3-1b-it | — | — | — | — | — |
| GLM | Reference | production: THUDM/glm-4-9b-chat | — | — | — | — | — |
| Kimi | Reference | production: moonshotai/Kimi-K2-Instruct-0905 | — | — | — | — | — |
| Qwen 3 | Reference | production: Qwen/Qwen3-8B | — | — | — | — | — |
| Qwen 3.5 | Reference | production: Qwen/Qwen3.5-35B-A3B | — | — | — | — | — |
| PA-Clean-uncapped | Ablation | Parity-aware BPE | hybrid-window | clean-multi | NFC | uncapped | FineWeb2-full |
| BPE-Clean-capped | Ablation | Plain BPE | — | clean-multi | NFC | capped | FineWeb2-full (tuned) |
| BPE-Clean-uncapped | Ablation | Plain BPE | — | clean-multi | NFC | uncapped | balanced |
| PA-Clean-capped-base | Ablation | Parity-aware BPE | base (no window) | clean-multi | NFC | capped | tuned |
| PA-gpt4-balanced | Ablation | Parity-aware BPE | hybrid-window | gpt4 | NFC | uncapped | balanced |
| PA-gpt4-fineweb2full | Ablation | Parity-aware BPE | hybrid-window | gpt4 | NFC | uncapped | FineWeb2-full |
| SuperBPE(PA-base)·gpt4o·t64k | Ablation | SuperBPE | PA-BPE base (gpt4) | gpt4o | NFC | — | balanced; transition 64k |
| SuperBPE(PA-base)·clean-c2·t90k | Ablation | SuperBPE | PA-BPE base (clean-multi) | clean-multi C2 | NFC | — | balanced; transition 90k |
| SuperBPE(plain-base)·gpt4o·noNFC | Ablation | SuperBPE | plain-BPE base (gpt4o) | gpt4o | none | — | balanced; transition 90k |
| Unigram-gpt4o | Ablation | Unigram LM | — | gpt4o | — | — | balanced |
| BPE-rightalign | Ablation | Plain BPE | — | right-aligned digits | — | — | balanced |
| BPE-gpt2 | Ablation | Plain BPE | — | gpt2-style | — | — | balanced |

## 1. Main comparison — candidates vs open-source references


### flores60 — 60-language FLORES set (60 languages, 59820 parallel sentences/tokenizer)

**Candidates:**

| Tokenizer | Vocab size | Eng comp (B/tok) ↑ | Multiling. tok/sent ↓ | Vocab util ↑ | Vocab-util CoV ↓ | Gini ↓ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ |
|---|---|---|---|---|---|---|---|---|---|
| PA-Apertus-capped | 127,835 | 4.336 | 43.0 | 0.606 | 0.4130 | 0.081 | 0.00043 | 0.02208 | 0.502 |
| PA-Clean-capped | 127,835 | 4.238 | 43.1 | 0.605 | 0.4138 | 0.081 | 0.00043 | 0.02198 | 0.987 |
| SuperBPE(PA-base)·gpt4o·t90k | 128,000 | 5.620 | 73.2 | 0.662 | 0.4906 | 0.428 | 0.00043 | 0.01127 | 0.509 |
| SuperBPE(PA-base)·clean-c3·t90k | 128,000 | 5.598 | 73.3 | 0.651 | 0.4978 | 0.429 | 0.00043 | 0.01030 | 0.627 |

**Open-source references:**

| Tokenizer | Vocab size | Eng comp (B/tok) ↑ | Multiling. tok/sent ↓ | Vocab util ↑ | Vocab-util CoV ↓ | Gini ↓ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ |
|---|---|---|---|---|---|---|---|---|---|
| Apertus | 131,072 | 4.595 | 50.5 | 0.556 | 0.5133 | 0.205 | 0.00000 | 0.02010 | 0.486 |
| Gemma 3 | 262,145 | 4.636 | 41.0 | 0.419 | 0.3919 | 0.106 | 0.00000 | 0.03414 | 0.929 |
| GLM | 151,343 | 4.726 | 79.3 | 0.347 | 0.6230 | 0.379 | 0.00000 | 0.06151 | 0.576 |
| Kimi | 163,601 | 4.726 | 61.5 | 0.225 | 0.6648 | 0.199 | 0.00000 | 0.03995 | 0.533 |
| Qwen 3 | 151,669 | 4.623 | 73.7 | 0.314 | 0.6222 | 0.320 | 0.00043 | 0.06152 | 0.577 |
| Qwen 3.5 | 248,077 | 4.573 | 47.4 | 0.379 | 0.5427 | 0.180 | 0.00043 | 0.00361 | 0.576 |

## 2. Production-safety gates

Any **fail** should disqualify before ranking. *Lossless* and *UNK* are from the analysis runs; the rest from the standalone sanity check. *Byte-frag* is benign (see legend).

| Tokenizer | Overall | Lossless ↑ | UNK ↓ | Byte coverage | Determinism | Whitespace | Per-script UNK | Dead vocab ↓ | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PA-Apertus-capped | warn | 0.9867 | 0.0000 | pass | pass | pass | pass | 0 | 5592 | 8 | 27 |
| PA-Clean-capped | warn | 0.9867 | 0.0000 | pass | pass | pass | pass | 0 | 5596 | 8 | 28 |
| SuperBPE(PA-base)·gpt4o·t90k | warn | 0.9867 | 0.0000 | pass | pass | pass | pass | 0 | 2445 | 4 | 50 |
| SuperBPE(PA-base)·clean-c3·t90k | warn | 0.9867 | 0.0000 | pass | pass | pass | pass | 0 | 2357 | 5 | 55 |
| Apertus | warn | 1.0000 | 0.0000 | pass | pass | pass | pass | 0 | 1435 | 8 | 46 |
| Gemma 3 | fail | 1.0000 | 0.0000 | pass | pass | pass | pass | 5 | 9571 | 0 | 150 |
| GLM | warn | 1.0000 | 0.0000 | pass | pass | pass | n/a | 0 | 1077 | 119 | 334 |
| Kimi | warn | 1.0000 | 0.0000 | pass | pass | pass | pass | 0 | 1172 | 90 | 273 |
| Qwen 3 | fail | 0.9867 | 0.0000 | pass | pass | pass | n/a | 248 | 1448 | 116 | 337 |
| Qwen 3.5 | warn | 0.9867 | 0.0000 | pass | pass | pass | n/a | 0 | 944 | 80 | 245 |
| BPE-Clean-capped | warn | 0.9867 | 0.0000 | pass | pass | pass | pass | 0 | 2642 | 0 | 46 |
| BPE-Clean-uncapped | warn | 0.9867 | 0.0000 | pass | pass | pass | pass | 0 | 1325 | 17 | 135 |
| BPE-gpt2 | warn | 1.0000 | 0.0000 | pass | pass | pass | pass | 0 | 1249 | 0 | 117 |
| BPE-rightalign | warn | 1.0000 | 0.0000 | pass | pass | pass | pass | 0 | 1290 | 0 | 116 |
| PA-Clean-capped-base | warn | 0.9867 | 0.0000 | pass | pass | pass | pass | 0 | 5188 | 3 | 14 |
| PA-Clean-uncapped | warn | 0.9867 | 0.0000 | pass | pass | pass | pass | 0 | 5691 | 14 | 64 |
| PA-gpt4-balanced | warn | 0.9867 | 0.0000 | pass | pass | pass | pass | 0 | 2837 | 4 | 59 |
| PA-gpt4-fineweb2full | warn | 0.9867 | 0.0000 | pass | pass | pass | pass | 0 | 5673 | 8 | 33 |
| SuperBPE(PA-base)·clean-c2·t90k | warn | 0.9867 | 0.0000 | pass | pass | pass | pass | 0 | 2359 | 5 | 63 |
| SuperBPE(PA-base)·gpt4o·t64k | warn | 0.9867 | 0.0000 | pass | pass | pass | pass | 0 | 2103 | 6 | 72 |
| SuperBPE(plain-base)·gpt4o·noNFC | warn | 1.0000 | 0.0000 | pass | pass | pass | pass | 0 | 1156 | 6 | 92 |
| Unigram-gpt4o | warn | 1.0000 | 0.0000 | pass | pass | pass | pass | 0 | 9932 | 0 | 304 |

> **FAIL (disqualified):** Gemma 3, Qwen 3 — see *Dead vocab* column (tokens their own normalizer can never emit).

## 3. Design-choice ablations

Intrinsic metrics from the **flores60** run. Production-safety gate columns are appended **only where they differ** across the tokenizers in that ablation (identical gates are omitted to keep the focus on what the design choice changes).


### Punctuation/whitespace capping (capped vs uncapped)

*Safety gates that differ here: Byte-frag (benign), Long toks (>64), Junk toks (≥8) ↓.*

| Tokenizer | Vocab size | Eng comp (B/tok) ↑ | Multiling. tok/sent ↓ | Vocab util ↑ | Vocab-util CoV ↓ | Gini ↓ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PA-Clean-capped | 127,835 | 4.238 | 43.1 | 0.605 | 0.4138 | 0.081 | 0.00043 | 0.02198 | 0.987 | 5596 | 8 | 28 |
| PA-Clean-uncapped | 127,835 | 4.242 | 43.2 | 0.586 | 0.3863 | 0.074 | 0.00043 | 0.02184 | 0.987 | 5691 | 14 | 64 |

### Parity-aware vs plain BPE

*Safety gates that differ here: Byte-frag (benign), Long toks (>64), Junk toks (≥8) ↓.*

| Tokenizer | Vocab size | Eng comp (B/tok) ↑ | Multiling. tok/sent ↓ | Vocab util ↑ | Vocab-util CoV ↓ | Gini ↓ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PA-Clean-capped | 127,835 | 4.238 | 43.1 | 0.605 | 0.4138 | 0.081 | 0.00043 | 0.02198 | 0.987 | 5596 | 8 | 28 |
| BPE-Clean-capped | 128,000 | 4.428 | 43.8 | 0.615 | 0.4913 | 0.114 | 0.00043 | 0.02860 | 0.987 | 2642 | 0 | 46 |
| BPE-Clean-uncapped | 128,004 | 4.559 | 71.4 | 0.535 | 0.6167 | 0.375 | 0.00043 | 0.02832 | 0.986 | 1325 | 17 | 135 |

### Hybrid-window vs base parity

*Safety gates that differ here: Byte-frag (benign), Long toks (>64), Junk toks (≥8) ↓.*

| Tokenizer | Vocab size | Eng comp (B/tok) ↑ | Multiling. tok/sent ↓ | Vocab util ↑ | Vocab-util CoV ↓ | Gini ↓ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PA-Clean-capped | 127,835 | 4.238 | 43.1 | 0.605 | 0.4138 | 0.081 | 0.00043 | 0.02198 | 0.987 | 5596 | 8 | 28 |
| PA-Clean-capped-base | 127,835 | 3.133 | 46.7 | 0.527 | 0.4258 | 0.087 | 0.00043 | 0.02238 | 0.986 | 5188 | 3 | 14 |

### Training data (language-balanced vs FineWeb2-full)

*Safety gates that differ here: Byte-frag (benign), Long toks (>64), Junk toks (≥8) ↓.*

| Tokenizer | Vocab size | Eng comp (B/tok) ↑ | Multiling. tok/sent ↓ | Vocab util ↑ | Vocab-util CoV ↓ | Gini ↓ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PA-gpt4-balanced | 127,826 | 4.610 | 72.5 | 0.689 | 0.4619 | 0.415 | 0.00043 | 0.01205 | 0.472 | 2837 | 4 | 59 |
| PA-gpt4-fineweb2full | 127,825 | 4.433 | 42.6 | 0.590 | 0.3755 | 0.076 | 0.00043 | 0.02226 | 0.505 | 5673 | 8 | 33 |

### Pretokenizer family (apertus vs clean-multi vs gpt4)

*Safety gates that differ here: Byte-frag (benign), Junk toks (≥8) ↓.*

| Tokenizer | Vocab size | Eng comp (B/tok) ↑ | Multiling. tok/sent ↓ | Vocab util ↑ | Vocab-util CoV ↓ | Gini ↓ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Byte-frag (benign) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PA-Apertus-capped | 127,835 | 4.336 | 43.0 | 0.606 | 0.4130 | 0.081 | 0.00043 | 0.02208 | 0.502 | 5592 | 27 |
| PA-Clean-capped | 127,835 | 4.238 | 43.1 | 0.605 | 0.4138 | 0.081 | 0.00043 | 0.02198 | 0.987 | 5596 | 28 |
| PA-gpt4-fineweb2full | 127,825 | 4.433 | 42.6 | 0.590 | 0.3755 | 0.076 | 0.00043 | 0.02226 | 0.505 | 5673 | 33 |

### SuperBPE base, transition point & stage-2 preset

*Safety gates that differ here: Lossless ↑, Byte-frag (benign), Long toks (>64), Junk toks (≥8) ↓.*

| Tokenizer | Vocab size | Eng comp (B/tok) ↑ | Multiling. tok/sent ↓ | Vocab util ↑ | Vocab-util CoV ↓ | Gini ↓ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Lossless ↑ | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SuperBPE(PA-base)·gpt4o·t90k | 128,000 | 5.620 | 73.2 | 0.662 | 0.4906 | 0.428 | 0.00043 | 0.01127 | 0.509 | 0.9867 | 2445 | 4 | 50 |
| SuperBPE(PA-base)·gpt4o·t64k | 128,000 | 5.869 | 70.1 | 0.602 | 0.5094 | 0.400 | 0.00043 | 0.01336 | 0.493 | 0.9867 | 2103 | 6 | 72 |
| SuperBPE(PA-base)·clean-c2·t90k | 128,000 | 5.148 | 70.9 | 0.652 | 0.5124 | 0.397 | 0.00043 | 0.01087 | 0.987 | 0.9867 | 2359 | 5 | 63 |
| SuperBPE(PA-base)·clean-c3·t90k | 128,000 | 5.598 | 73.3 | 0.651 | 0.4978 | 0.429 | 0.00043 | 0.01030 | 0.627 | 0.9867 | 2357 | 5 | 55 |
| SuperBPE(plain-base)·gpt4o·noNFC | 128,000 | 6.159 | 72.2 | 0.484 | 0.6230 | 0.387 | 0.00000 | 0.02663 | 0.452 | 1.0000 | 1156 | 6 | 92 |

### Algorithm / pretok (plain BPE vs Unigram, right-align digits, gpt2-style)

*Safety gates that differ here: Byte-frag (benign), Junk toks (≥8) ↓.*

| Tokenizer | Vocab size | Eng comp (B/tok) ↑ | Multiling. tok/sent ↓ | Vocab util ↑ | Vocab-util CoV ↓ | Gini ↓ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Byte-frag (benign) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BPE-gpt2 | 128,256 | 4.761 | 74.5 | 0.507 | 0.5843 | 0.389 | 0.00000 | 0.02119 | 0.987 | 1249 | 117 |
| BPE-rightalign | 128,256 | 4.796 | 72.9 | 0.500 | 0.5809 | 0.384 | 0.00000 | 0.02668 | 0.478 | 1290 | 116 |
| Unigram-gpt4o | 128,256 | 3.093 | 77.2 | 0.583 | 0.5201 | 0.306 | 0.00000 | 0.08215 | 0.887 | 9932 | 304 |

## 4. Per-language plots (small multiples, one panel per language)


**flores60** (`results/report_flores60/faceted_plots`):
- [bigram_entropy_faceted.svg](results/report_flores60/faceted_plots/bigram_entropy_faceted.svg)
- [compression_rate_faceted.svg](results/report_flores60/faceted_plots/compression_rate_faceted.svg)
- [fertility_faceted.svg](results/report_flores60/faceted_plots/fertility_faceted.svg)
- [vocabulary_utilization_faceted.svg](results/report_flores60/faceted_plots/vocabulary_utilization_faceted.svg)

## Appendix — long-token (>64 char) examples

Examples truncated to 40 chars; entries that look blank are long runs of spaces. These flag decorative-junk tokens (e.g. `----`, `====`, space runs) vs legitimate long multibyte-script words.

- **PA-Apertus-capped** (8): `ဝႃးသျိၼ်းဢၼ်ၽိမ်းဢွၵ်ႇလႆႈ`, `ိူဝ်းသျိၼ်းဢၼ်ဢိတ်ႇဢွၵ်ႇလႆႈ`, `ဢဝ်ၼႃႈလိၵ်ႈသၢင်ႇထုၵ်ႇဝႃႈ`, ` ဢၼ်လွတ်ႈလႅဝ်းထၢင်ႇႁၢင်ႈ`, `ລາຍການກະຈາຍສຽງຂອງວີໂອເອ`, `လွင်ႈလႅၵ်ႈလၢႆႈမႂ်ႇမႂ်ႇ`
- **PA-Clean-capped** (8): `ၵၢၼ်ႁဵတ်းသၢင်ႈယၢမ်းလဵဝ်`, `ဝိူဝ်းသျိၼ်းဢၼ်ဢိတ်ႇဢွၵ်ႇလႆႈ`, `ိူဝ်းသျိၼ်းဢၼ်ဢိတ်ႇဢွၵ်ႇလႆႈ`, `ဢဝ်ၼႃႈလိၵ်ႈသၢင်ႇထုၵ်ႇဝႃႈ`, ` ဢၼ်လွတ်ႈလႅဝ်းထၢင်ႇႁၢင်ႈ`, `လွင်ႈလႅၵ်ႈလၢႆႈမႂ်ႇမႂ်ႇ`
- **SuperBPE(PA-base)·gpt4o·t90k** (4): ` =======================================`, `สัตว์เลี้ยงลูกด้วยน้ำนม`, ` ---------------------------------------`, ` ---------------------------------------`
- **SuperBPE(PA-base)·clean-c3·t90k** (5): `----------------------------------------`, ` ---------------------------------------`, ` ---------------------------------------`, `########################################`, ` =======================================`
- **Apertus** (8): `                                        `, `                                        `, `----------------------------------------`, `                                        `, `----------------------------------------`, ` ***************************************`
- **GLM** (119): `                                        `, `/***************************************`, `########################################`, ` *--------------------------------------`, ` =======================================`, ` =======================================`
- **Kimi** (90): ` =======================================`, ` ---------------------------------------`, `----------------------------------------`, ` ---------------------------------------`, ` ***************************************`, `//**************************************`
- **Qwen 3** (116): `                                        `, `                                        `, `                                        `, `                                        `, ` =======================================`, `//======================================`
- **Qwen 3.5** (80): `                                        `, ` //-------------------------------------`, `                                        `, `/***************************************`, `                                        `, `                                        `
- **BPE-Clean-uncapped** (17): `                                        `, ` =======================================`, `                                        `, `----------------------------------------`, ` #######################################`, ` =======================================`
- **PA-Clean-capped-base** (3): `လွင်ႈလႅၵ်ႈလၢႆႈမႂ်ႇမႂ်ႇ`, `ဢဝ်ၼႃႈလိၵ်ႈသၢင်ႇထုၵ်ႇဝႃႈ`, `ၵၢၼ်ႁဵတ်းသၢင်ႈယၢမ်းလဵဝ်`
- **PA-Clean-uncapped** (14): `ဝိူဝ်းသျိၼ်းဢၼ်ဢိတ်ႇဢွၵ်ႇလႆႈ`, `ၵၢၼ်ႁဵတ်းသၢင်ႈယၢမ်းလဵဝ်`, `****************************************`, `ລາຍການກະຈາຍສຽງຂອງວີໂອເອ`, `လွင်ႈလႅၵ်ႈလၢႆႈမႂ်ႇမႂ်ႇ`, `----------------------------------------`
- **PA-gpt4-balanced** (4): ` =======================================`, `#---------------------------------------`, `สัตว์เลี้ยงลูกด้วยน้ำนม`, `########################################`
- **PA-gpt4-fineweb2full** (8): ` ဢၼ်လွတ်ႈလႅဝ်းထၢင်ႇႁၢင်ႈ`, `လွင်ႈလႅၵ်ႈလၢႆႈမႂ်ႇမႂ်ႇ`, `ဝႃးသျိၼ်းဢၼ်ၽိမ်းဢွၵ်ႇလႆႈ`, `ိူဝ်းသျိၼ်းဢၼ်ဢိတ်ႇဢွၵ်ႇလႆႈ`, `ລາຍການກະຈາຍສຽງຂອງວີໂອເອ`, `ၵၢၼ်ႁဵတ်းသၢင်ႈယၢမ်းလဵဝ်`
- **SuperBPE(PA-base)·clean-c2·t90k** (5): ` ---------------------------------------`, ` ---------------------------------------`, ` =======================================`, `########################################`, `----------------------------------------`
- **SuperBPE(PA-base)·gpt4o·t64k** (6): ` ---------------------------------------`, ` ---------------------------------------`, ` =======================================`, `----------------------------------------`, `########################################`, `########################################`
- **SuperBPE(plain-base)·gpt4o·noNFC** (6): `########################################`, ` =======================================`, `########################################`, `****************************************`, ` =======================================`, `########################################`

### Dead / unreachable vocabulary examples (tokens the normalizer can never emit)

- **Gemma 3** (5): ` yyyy`, ` YYYY`, ` `, ` ::::::::`, ` diffformul`
- **Qwen 3** (248): `龍`, `煉`, `留`, `林`, `暴`, `禮`, `女`, `練`
