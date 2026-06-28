# Production-safety gates and vocabulary inspection — companion to REPORT.md

*([← back to REPORT.md](REPORT.md))*

Production-safety verdicts (pass / warn / fail), round-trip exact-match and CER, vocabulary-usage breakdown (Active / Rare / Uncommon / Unseen / Scaffold), and examples of long, junk, and dead-vocab tokens for each tokenizer.


## Production-safety gates

A **fail** disqualifies before ranking. Dead vocab (normalizer- or pretokenizer-unreachable slots) is a **warning**, not a fail: the slots waste vocabulary capacity but do not corrupt text or emit UNK. *Lossless* and *UNK* are from the analysis runs; the rest from the standalone sanity check. *Byte-frag* is benign (see legend).

| Tokenizer | Overall | Lossless ↑ | UNK ↓ | Byte coverage | Byte-alphabet missing ↓ | Determinism | Whitespace | Per-script UNK | Dead vocab ↓ | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CleanV2-pretok + PA-BPE | warn | 0.9867 | 0.0000 | pass | 0 | pass | pass | pass | 0 | 5617 | 8 | 28 |
| CleanV3-pretok + PA-BPE (rebalanced data) | warn | 0.9867 | 0.0000 | pass | 0 | pass | pass | pass | 0 | 2800 | 0 | 34 |
| Apertus-pretok + PA-BPE | warn | 0.9867 | 0.0000 | pass | 0 | pass | pass | pass | 0 | 5592 | 8 | 27 |
| Apertus-pretok + PA-BPE (untuned data) | warn | 0.9867 | 0.0000 | pass | 0 | pass | pass | pass | 0 | 5679 | 8 | 27 |
| CleanV1-pretok + PA-BPE + SuperBPE | warn | 0.9867 | 0.0000 | pass | 15 | pass | pass | n/a | 0 | 3435 | 0 | 77 |
| CleanV3-pretok + plain BPE | warn | 0.9867 | 0.0000 | pass | 0 | pass | pass | pass | 0 | 2660 | 0 | 46 |
| Apertus v1 (production) | warn | 1.0000 | 0.0000 | pass | 0 | pass | pass | pass | 0 | 1435 | 8 | 46 |
| Gemma 3 | fail | 1.0000 | 0.0000 | pass | 3 | pass | pass | pass | 5 | 9571 | 0 | 150 |
| GLM | warn | 1.0000 | 0.0000 | pass | 0 | pass | pass | n/a | 0 | 1077 | 119 | 334 |
| Kimi | warn | 1.0000 | 0.0000 | pass | 0 | pass | pass | pass | 0 | 1172 | 90 | 273 |
| Qwen 3 | warn | 0.9867 | 0.0000 | pass | 0 | pass | pass | n/a | 248 | 1448 | 116 | 337 |
| Qwen 3.5 | warn | 0.9867 | 0.0000 | pass | 0 | pass | pass | n/a | 0 | 944 | 80 | 245 |
| EuroLLM | fail | 1.0000 | 0.0000 | pass | 26 | pass | pass | pass | 5370 | 12297 | 0 | 14 |
| Llama 4 | warn | 1.0000 | 0.0000 | pass | 0 | pass | pass | n/a | 0 | 1828 | 68 | 293 |
| OLMo 2 | warn | 1.0000 | 0.0000 | pass | 0 | pass | pass | pass | 0 | 773 | 116 | 334 |
| K2 Think | warn | 0.9867 | 0.0000 | pass | 0 | pass | pass | n/a | 248 | 1448 | 116 | 337 |

> **Fail (disqualified):** Gemma 3, EuroLLM.

> **Unreachable-vocab warning:** Gemma 3, Qwen 3, EuroLLM, K2 Think each have at least one normalizer- or pretokenizer-unreachable vocab token (the *Dead vocab* column). These slots are unreachable but do not affect correctness.

## Round-trip fidelity — where reconstruction differs

Measured on the **full** corpus. *Round-trip* = `decode(encode(text)) == text`. A difference is only a defect if it loses information (an UNK, or a byte that cannot be recovered). Every tokenizer here is byte-level with full 256-byte coverage (the *Byte coverage* gate above), so none can emit UNK or drop bytes. **NFC** normalization, however, deliberately rewrites text to canonical composed form, so for NFC tokenizers `decode(encode(x))` returns the *canonical* form of `x`. The exact-match rate is below 1.0 by reversible re-spelling, not loss (CER stays near zero). Table shows the 7 candidates, the Apertus baseline, and 5 open-source references; all other ablations follow one of these patterns and are omitted.
| Tokenizer | Exact-match ↑ | Mean CER ↓ |
|---|---|---|
| CleanV1-pretok + PA-BPE | 0.9673 | 0.00133 |
| CleanV2-pretok + PA-BPE | 0.9673 | 0.00133 |
| CleanV3-pretok + PA-BPE (rebalanced data) | 0.9673 | 0.00133 |
| CleanV3-pretok + PA-BPE (base parity, rebalanced data) | 0.9673 | 0.00133 |
| Apertus-pretok + PA-BPE | 0.9673 | 0.00133 |
| CleanV1-pretok + PA-BPE + SuperBPE | 0.9673 | 0.00133 |
| Apertus-pretok + PA-BPE + SuperBPE | 0.9673 | 0.00133 |
| Apertus v1 (production) | 1.0000 | 0.00000 |
| Gemma 3 | 1.0000 | 0.00000 |
| Llama 4 | 1.0000 | 0.00000 |
| OLMo 2 | 1.0000 | 0.00000 |
| Qwen 3 | 0.9673 | 0.00133 |
| K2 Think | 0.9673 | 0.00133 |

In the table above, the tokenizers reach exact-match 1.0 (Apertus v1 (production), Gemma 3, Llama 4, OLMo 2); these tokenizers do not apply NFC and so reproduce input byte-for-byte, while the rest sit at ~0.967 (CleanV1-pretok + PA-BPE, CleanV2-pretok + PA-BPE, CleanV3-pretok + PA-BPE (rebalanced data), CleanV3-pretok + PA-BPE (base parity, rebalanced data), Apertus-pretok + PA-BPE, CleanV1-pretok + PA-BPE + SuperBPE, Apertus-pretok + PA-BPE + SuperBPE, Qwen 3, K2 Think); these apply NFC, so the difference is reversible canonical re-spelling, not loss.

**Where the rewrites concentrate** (representative NFC tokenizer *CleanV1-pretok + PA-BPE*; all NFC byte-level tokenizers share this profile because round-trip is governed by the normalizer, not the vocabulary). Scripts with exact-match < 1.0, worst first:
| Script | Exact-match ↑ | Mean CER ↓ | # langs |
|---|---|---|---|
| Mtei | 0.1868 | 0.03219 | 1 |
| Beng | 0.2899 | 0.02656 | 3 |
| Guru | 0.4338 | 0.01729 | 1 |
| Orya | 0.7569 | 0.00460 | 1 |
| Deva | 0.8589 | 0.00539 | 10 |
| Mymr | 0.9758 | 0.00034 | 2 |
| Arab | 0.9841 | 0.00033 | 18 |
| Latn | 0.9909 | 0.00060 | 130 |
| Knda | 0.9931 | 0.00018 | 1 |
| Mlym | 0.9960 | 0.00006 | 1 |
| Tibt | 0.9975 | 0.00004 | 2 |
| Taml | 0.9980 | 0.00002 | 1 |
These are Brahmic/Indic and other scripts with many canonically-decomposable sequences (combining vowel signs, nuktas), where NFC composition changes the code points. CER stays near zero (most differences are single-codepoint canonical swaps) and UNK is zero, so no text is lost. Non-NFC tokenizers (e.g. Apertus, the `noNFC` SuperBPE) round-trip exactly (exact-match 1.0) everywhere.

## Vocabulary usage — Active / Rare / Uncommon / Unseen, and Scaffold

How each merge-created vocabulary token is actually used, from encoding a fixed corpus: **FLORES-200 (211 langs) + FineMath-4+ + StarCoder (python+javascript)**. For a merge token *t* (the base byte-alphabet is excluded), with `final(t)` = times emitted as a standalone token and `stepping(t)` = times built as an internal step inside a longer emitted token, define `formed(t) = final(t) + stepping(t)` and two corpus-invariant rates:
- `standalone_rate(t) = final(t) / Σ_t final(t)` &nbsp;&nbsp; `survival(t) = final(t) / formed(t)`

Every merge token (byte-fragments included) is classified by the **same** rule (no special-casing). Four buckets partition the merge vocabulary by standalone rate (sum to 100%):
- **Active** — `formed>0` and `standalone_rate ≥ 5/million`: appears on its own a normal amount.
- **Rare** — `formed>0` and `1/million ≤ standalone_rate < 5/million`: appears on its own, but seldom.
- **Uncommon** — `formed>0` and `standalone_rate < 1/million`: very seldom appears on its own.
- **Unseen** — `formed == 0`: never produced in **any** role on this corpus — neither a final token nor a merge step (a defined merge the corpus simply never exercised).

**Scaffold** is an overlay on Rare ∪ Uncommon (not a separate partition bucket): a token is Scaffold when `standalone_rate < 5/million` **and** `survival < 0.1`: it rarely appears on its own **and** is emitted as a final token < 10% of the times it is built, so it acts mostly as a stepping stone toward longer tokens. Scaffold is **rarely-exercised embedding capacity, not removable waste** (these tokens are structurally required to build the tokens that do surface), and is distinct from the absolute *Dead vocab* (normalizer- or pretokenizer-unreachable) and *Junk* gates.

Thresholds (all corpus-invariant): rate `1` and `5 per million`, survival `0.1`. Numbers are corpus-relative: cross-tokenizer differences partly reflect how well this corpus matches each tokenizer's training data, and **Unseen** is largely the web-text the eval corpus lacks (named entities, casual/spam register, the long tail of each language). References omitted (no merge tree). All bucket percentages use the merge-token denominator; *Vocab util* (fraction of the full vocab emitted ≥1×) uses the full-vocab denominator.

| Tokenizer | Vocab util ↑ | Active % | Rare % | Uncommon % | Unseen % | Scaffold % |
|---|---|---|---|---|---|---|
| Apertus-pretok + PA-BPE | 0.914 | 17.2 | 29.6 | 46.0 | 7.2 | 3.14 |
| CleanV1-pretok + PA-BPE | 0.913 | 16.4 | 29.9 | 46.4 | 7.3 | 3.25 |
| CleanV2-pretok + PA-BPE | 0.913 | 16.6 | 29.9 | 46.3 | 7.2 | 3.24 |
| CleanV3-pretok + PA-BPE (rebalanced data) | 0.913 | 16.8 | 30.6 | 45.2 | 7.4 | 3.02 |
| CleanV3-pretok + PA-BPE (base parity, rebalanced data) | 0.788 | 12.3 | 17.5 | 51.7 | 18.5 | 4.35 |
| Apertus-pretok + PA-BPE + SuperBPE | 0.928 | 22.0 | 39.0 | 32.5 | 6.5 | 2.27 |
| CleanV1-pretok + PA-BPE + SuperBPE | 0.932 | 19.3 | 38.4 | 36.3 | 6.0 | 2.48 |
| preliminary_mul_200k (CleanV2-pretok + PA-BPE, 200k) | 0.808 | 10.6 | 20.9 | 51.5 | 17.0 | 3.56 |
| preliminary_mul (CleanV3-pretok + PA-BPE, rebalanced) | 0.922 | 15.7 | 29.3 | 48.5 | 6.5 | 3.06 |
| preliminary_enh (CleanV2-pretok + PA-BPE, English-boosted) | 0.888 | 16.5 | 28.2 | 45.5 | 9.8 | 2.89 |
| preliminary_euh (CleanV2-pretok + PA-BPE, Fr/De-boosted) | 0.866 | 16.3 | 25.2 | 46.7 | 11.8 | 2.93 |
| SuperBPE(PA-base)·gpt4o·t90k | 0.889 | 12.2 | 24.8 | 54.0 | 9.0 | 4.01 |
| SuperBPE(PA-base)·clean-c3·t90k | 0.894 | 12.2 | 25.4 | 53.9 | 8.5 | 4.04 |
| PA-Clean-uncapped | 0.914 | 16.9 | 32.2 | 43.7 | 7.2 | 3.36 |
| BPE-Clean-capped | 0.927 | 17.2 | 32.3 | 44.2 | 6.3 | 2.48 |
| BPE-Clean-uncapped | 0.828 | 9.9 | 18.3 | 56.5 | 15.3 | 3.73 |
| PA-Clean-capped-base | 0.778 | 11.7 | 17.3 | 51.4 | 19.6 | 4.25 |
| PA-gpt4-balanced | 0.880 | 11.0 | 21.6 | 57.5 | 9.9 | 3.79 |
| PA-gpt4-fineweb2full | 0.915 | 17.8 | 32.9 | 42.1 | 7.2 | 3.17 |
| Apertus-pretok + PA-BPE (European ×1.1) | 0.913 | 17.2 | 30.6 | 44.8 | 7.3 | 3.21 |
| Apertus-pretok + PA-BPE (untuned data) | 0.914 | 17.4 | 31.4 | 43.9 | 7.2 | 3.19 |
| Apertus-pretok + PA-BPE (no semitic regroup) | 0.913 | 17.2 | 29.8 | 45.8 | 7.3 | 3.14 |
| SuperBPE(PA-base)·gpt4o·t64k | 0.911 | 12.2 | 26.4 | 54.1 | 7.3 | 3.47 |
| SuperBPE(PA-base)·clean-c2·t90k | 0.886 | 10.3 | 22.6 | 57.9 | 9.2 | 4.21 |
| SuperBPE(plain-base)·gpt4o·noNFC | 0.882 | 12.0 | 26.3 | 51.3 | 10.5 | 2.89 |
| BPE-rightalign | 0.823 | 11.0 | 21.1 | 51.8 | 16.2 | 3.20 |
| BPE-gpt2 | 0.808 | 10.1 | 18.7 | 53.8 | 17.4 | 3.52 |
| SuperBPE·clean-cap·hw·fw2full·t110k/130k | 0.926 | 18.4 | 34.4 | 41.0 | 6.3 | 2.83 |
| SuperBPE·clean-cap·base·fw2full·t110k/130k | 0.830 | 17.3 | 22.9 | 45.0 | 14.9 | 3.93 |
| SuperBPE·apertus-cap·base·fw2full | 0.881 | 20.9 | 29.2 | 39.7 | 10.2 | 3.27 |
| SuperBPE·clean-cap·base·fw2full | 0.872 | 18.5 | 28.9 | 41.4 | 11.2 | 3.50 |
| SuperBPE·gpt4·hw·fw2full | 0.929 | 22.7 | 39.7 | 31.1 | 6.4 | 2.22 |
| SuperBPE·gpt4·base·fw2full | 0.855 | 21.3 | 27.6 | 38.5 | 12.6 | 3.58 |
| BPE-gpt4o-balanced | 0.823 | 10.9 | 21.0 | 51.8 | 16.2 | 3.20 |
| BPE-gpt4o-balanced-NFC | 0.823 | 11.0 | 21.0 | 51.8 | 16.2 | 3.20 |
| PA-Clean-balanced-hw | 0.876 | 10.0 | 19.6 | 60.1 | 10.2 | 4.06 |
| BPE-Punct | 0.800 | 9.4 | 17.8 | 54.7 | 18.1 | 3.59 |
| SuperBPE on CleanV3-pretok (t110k/v130k) | 0.922 | 20.8 | 34.7 | 37.7 | 6.8 | 2.60 |
| CleanV3-pretok + plain BPE | 0.928 | 17.3 | 32.4 | 44.1 | 6.3 | 2.48 |
| SuperBPE-plus2-cv2-t110k | 0.924 | 20.7 | 34.9 | 37.8 | 6.6 | 2.59 |
| SuperBPE-plus2v2-cv2-t110k | 0.926 | 18.8 | 35.8 | 39.0 | 6.4 | 2.71 |
| PA-Clean-plus3-A8 | 0.920 | 16.8 | 32.0 | 44.4 | 6.9 | 2.87 |
| PA-Clean-plus3-A7 | 0.918 | 16.7 | 31.7 | 44.6 | 7.0 | 2.97 |
| PA-Clean-plus3-repcap8fr-cv2 | 0.911 | 16.4 | 30.1 | 45.9 | 7.6 | 3.04 |
| PA-Clean-plus3-repcap8fr-A8 | 0.920 | 16.8 | 31.9 | 44.4 | 6.9 | 2.88 |
| BPE-plus3-repcap8 | 0.913 | 16.9 | 31.5 | 44.0 | 7.6 | 2.55 |

*Composition note: of Scaffold, the byte-fragment (incomplete-UTF-8 sub-character) share is 0.24–0.79 pp of vocab across tokenizers; the rest are subword stepping-stones. Byte-fragments are not special-cased; they fall in Scaffold only when they behave like merge steps.*

*Scaffold examples (subword stepping-stones), CleanV1-pretok + PA-BPE + SuperBPE:* `্`→`্ৰ` (built 25312×, final 22×); `ction`→` function` (built 16977×, final 71×); `ould`→` should` (built 6864×, final 15×); `่`→`�ี่` (built 5757×, final 65×); `----`→`-------` (built 5711×, final 46×)

*Scaffold examples (byte-fragments), CleanV1-pretok + PA-BPE:* `�`→`।` (built 250725×, final 0×); `�`→`ပ` (built 239058×, final 52×); ` �`→` ह` (built 236822×, final 25×); ` �`→` �` (built 225274×, final 15×); `�`→`པ` (built 195181×, final 12×)

## Appendix — long-token (>64 char) examples

Examples truncated to 40 chars; entries that look blank are long runs of spaces. These flag decorative-junk tokens (e.g. `----`, `====`, space runs) vs legitimate long multibyte-script words.

- **CleanV2-pretok + PA-BPE** (8): `ລາຍການກະຈາຍສຽງຂອງວີໂອເອ`, `ၵၢၼ်ႁဵတ်းသၢင်ႈယၢမ်းလဵဝ်`, `ိူဝ်းသျိၼ်းဢၼ်ဢိတ်ႇဢွၵ်ႇလႆႈ`, `ဢဝ်ၼႃႈလိၵ်ႈသၢင်ႇထုၵ်ႇဝႃႈ`, ` ဢၼ်လွတ်ႈလႅဝ်းထၢင်ႇႁၢင်ႈ`, `ဝိူဝ်းသျိၼ်းဢၼ်ဢိတ်ႇဢွၵ်ႇလႆႈ`
- **Apertus-pretok + PA-BPE** (8): `ၵၢၼ်ႁဵတ်းသၢင်ႈယၢမ်းလဵဝ်`, `ဝႃးသျိၼ်းဢၼ်ၽိမ်းဢွၵ်ႇလႆႈ`, `ိူဝ်းသျိၼ်းဢၼ်ဢိတ်ႇဢွၵ်ႇလႆႈ`, `ລາຍການກະຈາຍສຽງຂອງວີໂອເອ`, ` ဢၼ်လွတ်ႈလႅဝ်းထၢင်ႇႁၢင်ႈ`, `လွင်ႈလႅၵ်ႈလၢႆႈမႂ်ႇမႂ်ႇ`
- **Apertus-pretok + PA-BPE (untuned data)** (8): `ဝႃးသျိၼ်းဢၼ်ၽိမ်းဢွၵ်ႇလႆႈ`, `ລາຍການກະຈາຍສຽງຂອງວີໂອເອ`, `ဢဝ်ၼႃႈလိၵ်ႈသၢင်ႇထုၵ်ႇဝႃႈ`, `ၵၢၼ်ႁဵတ်းသၢင်ႈယၢမ်းလဵဝ်`, `လွင်ႈလႅၵ်ႈလၢႆႈမႂ်ႇမႂ်ႇ`, `ဝိူဝ်းသျိၼ်းဢၼ်ဢိတ်ႇဢွၵ်ႇလႆႈ`
- **Apertus v1 (production)** (8): ` ***************************************`, `                                        `, `----------------------------------------`, `                                        `, `                                        `, `****************************************`
- **GLM** (119): `****************************************`, ` ---------------------------------------`, `                                        `, `//--------------------------------------`, `                                        `, ` *--------------------------------------`
- **Kimi** (90): ` //-------------------------------------`, `////////////////////////////////////////`, `                                        `, `/*--------------------------------------`, `//======================================`, `########################################`
- **Qwen 3** (116): `                                        `, `//--------------------------------------`, `########################################`, ` /**************************************`, `#---------------------------------------`, ` =======================================`
- **Qwen 3.5** (80): `                                        `, `########################################`, `//======================================`, `                                        `, `----------------------------------------`, ` ***************************************`
- **Llama 4** (68): `****************************************`, `****************************************`, `………………………………………………………………`, ` #######################################`, ` //-------------------------------------`, `........................................`
- **OLMo 2** (116): `/***************************************`, `////////////////////////////////////////`, ` =======================================`, `########################################`, ` |--------------------------------------`, ` ///////////////////////////////////////`
- **K2 Think** (116): `////////////////////////////////////////`, `                                        `, `                                        `, `//======================================`, `                                        `, ` ---------------------------------------`

### Junk-token examples

These are low-value vocab tokens that waste slots, in three categories (each token is counted once, in priority order punctuation, then web, then gibberish; examples truncated to 40 chars):
- **Punctuation** — runs of ≥8 punctuation/symbol/whitespace chars with no letters or digits (the *Junk toks* gate: decorative separators / whitespace runs).
- **Web/markup** — URL / HTML scrape residue: `://`, `www.…`, `.tld/path`, HTML entities (`&nbsp;`), self-closing/attributed tags (`/>`, `<a href=`, `class="…`). Strong markers only; bare `http`, `https`, `www`, `.com` and special/sentinel tokens (`<bos>`, `</s>`) are **not** flagged.
- **Gibberish** — hash / random-alphanumeric IDs: ASCII, ≥12 chars, a long hex run or many letter↔digit transitions (normal identifiers like `utf8`, `base64encoded`, `covid19` are **not** flagged).


**Punctuation runs:**
- **CleanV2-pretok + PA-BPE** (28): `;;;;;;;;`, `--------`, `================`, `////////`, `****************`, `-------------`
- **CleanV3-pretok + PA-BPE (rebalanced data)** (34): `##############`, `--------------`, `===============`, `________`, `-------------`, `-----------`
- **Apertus-pretok + PA-BPE** (27): `................`, `________________`, `********`, `----------------`, `------------`, `----------------`
- **Apertus-pretok + PA-BPE (untuned data)** (27): `////////////////`, `================`, `--------`, `________________`, `---------------`, `****************`
- **CleanV1-pretok + PA-BPE + SuperBPE** (77): `////////`, `***************`, `##############`, `))))))))`, `........`, `*********`
- **CleanV3-pretok + plain BPE** (46): `________`, `----------------`, `;;;;;;;;;;;;;;;;`, `================`, `////////////////`, `~~~~~~~~`
- **Apertus v1 (production)** (46): `============`, `****************************************`, `****************************************`, `================================`, `--------------------`, `}\\))\\({}_{`
- **Gemma 3** (150): `!!!!!!!!`, `.............`, `---------------`, `--------`, `~~~~~~~~~~~~~~~~`, `..............`
- **GLM** (334): `//--------------------------------------`, `~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`, `****************`, `********************`, `########################################`, `/***************************************`
- **Kimi** (273): `----------------------------------------`, `+---------------------------------------`, `----------------------------------------`, `========================================`, `;;;;;;;;;;;;;;;;`, `////////`
- **Qwen 3** (337): `+#+#+#+#+#+`, `/***************************************`, `+-+-+-+-+-+-+-+-`, `/***************************************`, `........`, `================`
- **Qwen 3.5** (245): `"../../../../`, `////////`, `/*======================================`, `'../../../../../`, `........................`, `................................`
- **EuroLLM** (14): `--------`, `////////`, `----------------`, `................`, `────────`, `________`
- **Llama 4** (293): `'../../../`, `"../../../`, `.________`, `-------------\n`, `******/\n`, `^^^^^^^^`
- **OLMo 2** (334): `========\n`, `---------------\n`, `//======================================`, `/*======================================`, `*******/\n`, `######################################`
- **K2 Think** (337): `/***************************************`, `****************************************`, `........................................`, `=======\n`, `________________________________`, `_______,`

**Web / markup residue:**
- **CleanV2-pretok + PA-BPE** (4): `/>`, `"/>`, `/>`, `://`
- **CleanV3-pretok + PA-BPE (rebalanced data)** (4): `"/>`, `://`, `/>`, `/>`
- **Apertus-pretok + PA-BPE** (4): `/>\n`, `"/>\n`, `://`, `/>\n`
- **Apertus-pretok + PA-BPE (untuned data)** (4): `/>\n`, `://`, `"/>\n`, `/>\n`
- **CleanV1-pretok + PA-BPE + SuperBPE** (10): `://`, `}"/>`, `/>`, `/></`, `'"/>`, `"/>`
- **CleanV3-pretok + plain BPE** (4): `/>`, `://`, `"/>`, `/>`
- **Apertus v1 (production)** (20): `/>;\n`, `/>`, `"/></`, `"/>\n\n`, `/><`, `://`
- **Gemma 3** (30): `/></`, `/>}`, `://$`, `)}/>`, `/>`, `://"`
- **GLM** (50): `/>\\`, `/>;\n`, `'/>\n`, `/>\r\n`, `/>\r\n`, `/>`
- **Kimi** (33): `</>\n`, `/>\n\n`, `://${`, `/>\r\n`, `/><`, `/>`
- **Qwen 3** (50): `/>\\`, `/>';\n`, `/></`, `/>.`, `"/>`, `/>\\`
- **Qwen 3.5** (25): `/>`, `/>`, `://"`, `/>\\`, `://`, `:///`
- **EuroLLM** (2): `://`, `/>`
- **Llama 4** (35): `"/>\n`, `/>}`, `/>);\n`, `/>`, `/>\n\n`, `/><`
- **OLMo 2** (50): `/>\n`, `/>`, `}/>\n`, `/>);\n`, `/>";\n`, `/>}`
- **K2 Think** (50): `://'`, `"/></`, `/>,`, `/>.\n`, `:///`, `/>\n`

**Hash / random-alphanumeric gibberish:**
- (none across the evaluated tokenizers)

### Dead / unreachable vocabulary examples (tokens that can never be emitted)

Dead vocab means entries that are unreachable under the faithful pipeline, either because the **normalizer** rewrites their surface or because the **pretokenizer** always splits their surface into ≥2 pre-tokens (within-pretoken merges can never build them). The pretokenizer case is skipped for SuperBPE-style tokenizers that merge across pretoken boundaries by design. Count shown is *normalizer-dead + pretokenizer-dead*.

- **Gemma 3** (5 normalizer): ` diffformul`, ` ::::::::`, ` yyyy`, ` YYYY`, ` `
- **Qwen 3** (248 normalizer): `טּ`, `露`, `劉`, `度`, `更`, `列`, `य़`, `לּ`
- **EuroLLM** (5370 normalizer): `akko`, `arms`, `ających`, `clusion`, `tedy`, `ab`, `andom`, `davo`
- **K2 Think** (248 normalizer): `量`, `鍊`, `〈`, `糖`, `殺`, `麟`, `שׂ`, ` của`
