# Design-choice ablations — companion to REPORT.md

*([← back to REPORT.md](REPORT.md))*

All ablation tables computed from the broad FLORES (60-language) analysis unless otherwise noted. Each ablation isolates one design axis while holding the others fixed.


## Design-choice ablations

Each ablation compares tokenizers that differ in one design choice, measured on the broad FLORES set. In each table, the columns most affected by that design choice are placed first, and a production-safety gate column is included only when its value differs across the tokenizers being compared. Where downstream-LM results exist, an *Extrinsic (downstream LM)* block follows the table. In that block, `[matched]` marks a tokenizer for which the report's own tokenizer was trained from scratch (Val, FLORES, and code BPB at 10B balanced; MC-math and MBPP at 20B math+code), and `[proxy]` marks a sibling tokenizer-lm run on a different tokenizer, which should be read as directional. `pending` means the run is mapped but the eval is not yet measured, and `—` means the eval was not run. The full per-tokenizer extrinsic table and the training setup are in the appendix.


### Punctuation/whitespace capping (capped vs uncapped)

This ablation compares capping runs of punctuation, symbols, and whitespace at 16 characters during pretokenization against leaving them uncapped.

Tokenizers using the capped regex produce 28 junk tokens, against 64 for the uncapped regex, and 8 long (>64 char) tokens against 14. The uncapped tokenizer also has one pretokenizer-unreachable vocab token, which the gate reports as a warning. Eng B/tok (4.24 against 4.24) and Val BPB (0.729 against 0.728) are unchanged; the uncapped tokenizer has a slightly lower Gini (0.074 against 0.081). Keep the capped regex.

| Tokenizer | Junk toks (≥8) ↓ | Long toks (>64) | Vocab util ↑ | Vocab size | Eng comp (B/tok) ↑ | Multiling. sent/tok ↑ | Vocab-util CoV ↓ | Avg langs/token ↑ | Gini ↓ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Dead vocab ↓ | Byte-frag (benign) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CleanV1-pretok + PA-BPE | **28** | 8 | **0.605** | 127,835 | 4.238 | 0.0232 | 0.4138 | 2.79 | 0.081 | 0.00043 | 0.02198 | 0.987 | **0** | 5596 |
| PA-Clean-uncapped | 64 | 14 | 0.586 | 127,835 | **4.242** | 0.0232 | **0.3863** | **2.85** | **0.074** | 0.00043 | **0.02184** | 0.987 | 1 | 5691 |

*Faceted per-language vocabulary utilization, one pane per tokenizer:*

![Punctuation/whitespace capping (capped vs uncapped): per-language vocabulary utilization](report_flores60/ablation_plots/punctuation-whitespace-capping-capped-vs-uncapped/vocabulary_utilization_faceted.svg)

*Extrinsic (downstream LM):*
| Tokenizer | Val BPB ↓ | FLORES BPB (tr.) ↓ | Code BPB ↓ | MC-math ↑ | MBPP ↑ [95% CI] |
|---|---|---|---|---|---|
| CleanV1-pretok + PA-BPE [matched] | 0.729 | 1.169 | 0.533 | 0.295 | 0.190 [0.156, 0.226] |
| PA-Clean-uncapped [matched] | 0.728 | 1.167 | 0.529 | — | — |

### Parity-aware vs plain BPE

This ablation compares parity-aware BPE, which equalizes per-language encoding cost through its merge-selection rule, against plain frequency-driven BPE.

At matched capped settings, parity-aware BPE has a Gini of 0.081 against 0.114 for plain BPE, and a vocab-util CoV of 0.414 against 0.491. Multilingual compression is similar (sent/tok 0.0232 against 0.0228). Plain BPE has a higher Eng B/tok (4.43 against 4.24). On the 1B proxy, parity-aware BPE has a Val BPB 0.008 higher than plain BPE. Use parity-aware BPE for a multilingual tokenizer.

| Tokenizer | Gini ↓ | Vocab-util CoV ↓ | Multiling. sent/tok ↑ | Vocab size | Eng comp (B/tok) ↑ | Vocab util ↑ | Avg langs/token ↑ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Dead vocab ↓ | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CleanV1-pretok + PA-BPE | **0.081** | **0.4138** | **0.0232** | 127,835 | 4.238 | 0.605 | 2.79 | 0.00043 | **0.02198** | **0.987** | **0** | 5596 | 8 | **28** |
| BPE-Clean-capped | 0.114 | 0.4913 | 0.0228 | 128,000 | 4.428 | **0.615** | 2.83 | 0.00043 | 0.02860 | **0.987** | **0** | 2642 | 0 | 46 |
| BPE-Clean-uncapped | 0.375 | 0.6167 | 0.0140 | 128,004 | **4.559** | 0.535 | **2.98** | 0.00043 | 0.02832 | 0.986 | 3 | 1325 | 17 | 135 |

*Faceted per-language vocabulary utilization, one pane per tokenizer:*

![Parity-aware vs plain BPE: per-language vocabulary utilization](report_flores60/ablation_plots/parity-aware-vs-plain-bpe/vocabulary_utilization_faceted.svg)

*Extrinsic (downstream LM):*
| Tokenizer | Val BPB ↓ | FLORES BPB (tr.) ↓ | Code BPB ↓ | MC-math ↑ | MBPP ↑ [95% CI] |
|---|---|---|---|---|---|
| CleanV1-pretok + PA-BPE [matched] | 0.729 | 1.169 | 0.533 | 0.295 | 0.190 [0.156, 0.226] |
| BPE-Clean-uncapped [matched] | 0.716 | 1.157 | 0.523 | 0.270 | 0.148 [0.118, 0.180] |

*[proxy] tokenizer-lm 1B-balanced Δ, factor: Trainer (BPE vs PA-BPE)* (Δ = B−A; BPB Δ<0 means B better; **bold** = p_adj<0.05):
| A | B | ΔVal | ΔFLORES (tr.) | ΔFLORES (all) | ΔBLiMP | ΔCode |
|---|---|---|---|---|---|---|
| BPE clean | PA-BPE clean | +0.0076 | -- | -- | -- | -- |

### SuperBPE on the PA-BPE candidate base (does SuperBPE help, matched data)

This ablation tests whether adding a SuperBPE superword stage on top of the PA-BPE candidate base helps, on matched base and training data.

Adding the SuperBPE stage raises Eng B/tok by 18–25% (clean: 4.24 to 5.01; apertus: 4.34 to 5.40) and raises Gini (clean: 0.081 to 0.106; apertus: 0.081 to 0.110); vocab utilization drops from 0.605 to 0.550. On the apertus base, FLORES BPB rises from 2.943 to 3.081 and MBPP drops from 0.058 to 0.004. On the clean base the downstream BPB is close to the PA-BPE base (Val BPB 0.732 against 0.729, FLORES bits-per-byte trained 1.161 against 1.169) and MBPP is higher (0.196 against 0.190); the cost is multilingual fairness.

| Tokenizer | Eng comp (B/tok) ↑ | Multiling. sent/tok ↑ | Gini ↓ | Vocab size | Vocab util ↑ | Vocab-util CoV ↓ | Avg langs/token ↑ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Byte-alphabet missing ↓ | Per-script UNK | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Apertus-pretok + PA-BPE | 4.336 | **0.0233** | **0.081** | 127,835 | **0.606** | **0.4130** | 2.79 | 0.00043 | 0.02208 | 0.502 | **0** | pass | 5592 | 8 | **27** |
| Apertus-pretok + PA-BPE + SuperBPE | **5.402** | 0.0230 | 0.110 | 128,000 | 0.544 | 0.4992 | **3.14** | 0.00043 | 0.02686 | 0.466 | 15 | n/a | 3441 | 1 | 76 |
| CleanV1-pretok + PA-BPE | 4.238 | 0.0232 | **0.081** | 127,835 | 0.605 | 0.4138 | 2.79 | 0.00043 | **0.02198** | **0.987** | **0** | pass | 5596 | 8 | 28 |
| CleanV1-pretok + PA-BPE + SuperBPE | 5.013 | 0.0227 | 0.106 | 128,000 | 0.550 | 0.4892 | 3.02 | 0.00043 | 0.02629 | **0.987** | 15 | n/a | 3435 | 0 | 77 |

*Faceted per-language compression (sentences/token), one pane per tokenizer:*

![SuperBPE on the PA-BPE candidate base (does SuperBPE help, matched data): per-language compression (sentences/token)](report_flores60/ablation_plots/superbpe-on-the-pa-bpe-candidate-base-does-superbpe-help-mat/compression_rate_faceted.svg)

*Extrinsic (downstream LM):*
| Tokenizer | Val BPB ↓ | FLORES BPB (tr.) ↓ | Code BPB ↓ | MC-math ↑ | MBPP ↑ [95% CI] |
|---|---|---|---|---|---|
| Apertus-pretok + PA-BPE [matched] | 0.729 | 1.170 | 0.531 | 0.270 | 0.058 [0.038, 0.080] |
| Apertus-pretok + PA-BPE + SuperBPE [matched] | 0.733 | 1.176 | 0.541 | 0.269 | 0.004 [0.000, 0.010] |
| CleanV1-pretok + PA-BPE [matched] | 0.729 | 1.169 | 0.533 | 0.295 | 0.190 [0.156, 0.226] |
| CleanV1-pretok + PA-BPE + SuperBPE [matched] | 0.732 | 1.161 | 0.536 | 0.268 | 0.196 [0.162, 0.232] |

### Pretokenizer family (apertus vs clean-multi vs gpt4)

This ablation compares the three pretokenizer families (apertus, clean-multi, gpt4), which differ in digit grouping, apostrophe and contraction handling, and operator handling.

clean-multi and apertus are close on multilingual compression and Gini; they differ on code. apertus and gpt4 have an operator-isolation near 0.50 (operators tokenized together with operands), against 0.99 for clean-multi, and apertus has an MBPP of 0.058 against 0.190 for clean-multi (p_BH<0.001). gpt4 has 3 pretokenizer-unreachable vocab tokens, which the gate reports as a warning. Use clean-multi unless multilingual FLORES BPB matters more than code generation.

| Tokenizer | Operator-isol ↑ | Eng comp (B/tok) ↑ | Multiling. sent/tok ↑ | Vocab size | Vocab util ↑ | Vocab-util CoV ↓ | Avg langs/token ↑ | Gini ↓ | CER ↓ | Boundary-cross ↓ | Dead vocab ↓ | Byte-frag (benign) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Apertus-pretok + PA-BPE | 0.502 | 4.336 | 0.0233 | 127,835 | **0.606** | 0.4130 | 2.79 | 0.081 | 0.00043 | 0.02208 | **0** | 5592 | **27** |
| CleanV1-pretok + PA-BPE | **0.987** | 4.238 | 0.0232 | 127,835 | 0.605 | 0.4138 | 2.79 | 0.081 | 0.00043 | **0.02198** | **0** | 5596 | 28 |
| PA-gpt4-fineweb2full | 0.505 | **4.433** | **0.0235** | 127,825 | 0.590 | **0.3755** | **2.93** | **0.076** | 0.00043 | 0.02226 | 3 | 5673 | 33 |

*Faceted per-language compression (sentences/token), one pane per tokenizer:*

![Pretokenizer family (apertus vs clean-multi vs gpt4): per-language compression (sentences/token)](report_flores60/ablation_plots/pretokenizer-family-apertus-vs-clean-multi-vs-gpt4/compression_rate_faceted.svg)

*Extrinsic (downstream LM):*
| Tokenizer | Val BPB ↓ | FLORES BPB (tr.) ↓ | Code BPB ↓ | MC-math ↑ | MBPP ↑ [95% CI] |
|---|---|---|---|---|---|
| Apertus-pretok + PA-BPE [matched] | 0.729 | 1.170 | 0.531 | 0.270 | 0.058 [0.038, 0.080] |
| CleanV1-pretok + PA-BPE [matched] | 0.729 | 1.169 | 0.533 | 0.295 | 0.190 [0.156, 0.226] |
| PA-gpt4-fineweb2full [matched] | 0.728 | 1.169 | 0.531 | — | — |

*[proxy] tokenizer-lm 1B-balanced Δ, factor: Pretokenizer* (Δ = B−A; BPB Δ<0 means B better; **bold** = p_adj<0.05):
| A | B | ΔVal | ΔFLORES (tr.) | ΔFLORES (all) | ΔBLiMP | ΔCode |
|---|---|---|---|---|---|---|
| GPT-4o | Claude | **+0.0019** | +0.0001 | **+0.0035** | +0.0245 | +0.0001 |
| GPT-4o | Punct | **+0.0074** | +0.0038 | **+0.0087** | +0.0179 | +0.0089 |
| GPT-4o | RightAlign | **+0.0021** | **+0.0028** | **+0.0052** | +0.0176 | +0.0012 |
| GPT-4o | Whitespace | +0.0094 | **+0.0072** | **-0.0116** | +0.0066 | +0.0086 |
| GPT-4o | GPT-2 | +0.0014 | +0.0001 | +0.0018 | +0.0167 | -0.0029 |

### Hybrid-window vs base parity

This ablation compares the hybrid-window parity rule, which adds a global phase so the trainer does not keep selecting the same language, against the base lowest-cost rule.

Base parity gives an Eng B/tok of 3.13, against 4.24 for hybrid-window, with a lower sent/tok (0.0214 against 0.0232) and lower vocab utilization (0.527 against 0.605), and no lower Gini (0.087 against 0.081). On the 1B proxy, hybrid-window has a lower Val BPB and FLORES BPB. Use hybrid-window.

| Tokenizer | Eng comp (B/tok) ↑ | Multiling. sent/tok ↑ | Gini ↓ | Vocab-util CoV ↓ | Vocab size | Vocab util ↑ | Avg langs/token ↑ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CleanV1-pretok + PA-BPE | **4.238** | **0.0232** | **0.081** | **0.4138** | 127,835 | **0.605** | 2.79 | 0.00043 | **0.02198** | **0.987** | 5596 | 8 | 28 |
| PA-Clean-capped-base | 3.133 | 0.0214 | 0.087 | 0.4258 | 127,835 | 0.527 | **2.84** | 0.00043 | 0.02238 | 0.986 | 5188 | 3 | **14** |

*Faceted per-language compression (sentences/token), one pane per tokenizer:*

![Hybrid-window vs base parity: per-language compression (sentences/token)](report_flores60/ablation_plots/hybrid-window-vs-base-parity/compression_rate_faceted.svg)

*Extrinsic (downstream LM):*
| Tokenizer | Val BPB ↓ | FLORES BPB (tr.) ↓ | Code BPB ↓ | MC-math ↑ | MBPP ↑ [95% CI] |
|---|---|---|---|---|---|
| CleanV1-pretok + PA-BPE [matched] | 0.729 | 1.169 | 0.533 | 0.295 | 0.190 [0.156, 0.226] |

*[proxy] tokenizer-lm 1B-balanced Δ, factor: PA-BPE family* (Δ = B−A; BPB Δ<0 means B better; **bold** = p_adj<0.05):
| A | B | ΔVal | ΔFLORES (tr.) | ΔFLORES (all) | ΔBLiMP | ΔCode |
|---|---|---|---|---|---|---|
| GPT-4 pretok | clean pretok | -0.0020 | **-0.0079** | **-0.0089** | +0.0073 | -0.0060 |
| Base | Hybrid-window | -0.0061 | **-0.0090** | **-0.0062** | +0.0441 | -0.0095 |

### SuperBPE transition point & vocab size (t90k/128k vs t110k/130k, clean fw2full)

This ablation compares two SuperBPE settings that change together: the stage-1 to stage-2 transition vocab size (90k against 110k) and the final vocab size (128k against 130k).

The transition (90k to 110k) and the final vocab (128k to 130k) change together, so this is not a single-variable comparison. The later transition gives a higher sent/tok (0.0227 to 0.0232), a lower Gini (0.106 to 0.092), and a higher vocab utilization, at a lower Eng B/tok (5.01 to 4.87). Both now have standard-budget BPB (Val BPB 0.732 and FLORES bits-per-byte trained 1.161 for each); t110k has higher MC-math (0.288 against 0.268) and MBPP (0.202 against 0.196). Lean t110k/130k.

| Tokenizer | Eng comp (B/tok) ↑ | Multiling. sent/tok ↑ | Vocab util ↑ | Vocab-util CoV ↓ | Vocab size | Avg langs/token ↑ | Gini ↓ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CleanV1-pretok + PA-BPE + SuperBPE | **5.013** | 0.0227 | 0.550 | 0.4892 | 128,000 | **3.02** | 0.106 | 0.00043 | 0.02629 | **0.987** | 3435 | 0 | 77 |
| SuperBPE·clean-cap·hw·fw2full·t110k/130k | 4.869 | **0.0232** | **0.577** | 0.4544 | 130,000 | 2.88 | **0.092** | 0.00043 | 0.02358 | **0.987** | 4597 | 3 | 61 |
| SuperBPE·clean-cap·base·fw2full | 4.693 | 0.0220 | 0.543 | 0.4613 | 128,000 | 2.98 | 0.100 | 0.00043 | 0.02473 | 0.985 | 4217 | 1 | 53 |
| SuperBPE·clean-cap·base·fw2full·t110k/130k | 4.438 | 0.0219 | 0.539 | **0.4458** | 130,000 | 2.91 | 0.094 | 0.00043 | **0.02356** | 0.985 | 4756 | 1 | **42** |

*Faceted per-language compression (sentences/token), one pane per tokenizer:*

![SuperBPE transition point & vocab size (t90k/128k vs t110k/130k, clean fw2full): per-language compression (sentences/token)](report_flores60/ablation_plots/superbpe-transition-point-vocab-size-t90k-128k-vs-t110k-130k/compression_rate_faceted.svg)

*Extrinsic (downstream LM):*
| Tokenizer | Val BPB ↓ | FLORES BPB (tr.) ↓ | Code BPB ↓ | MC-math ↑ | MBPP ↑ [95% CI] |
|---|---|---|---|---|---|
| CleanV1-pretok + PA-BPE + SuperBPE [matched] | 0.732 | 1.161 | 0.536 | 0.268 | 0.196 [0.162, 0.232] |
| SuperBPE·clean-cap·hw·fw2full·t110k/130k [matched] | 0.732 | 1.161 | 0.534 | 0.288 | 0.202 [0.168, 0.238] |

**Further ablations.** Additional design points, reported in full under *Appendix — additional ablations*:
- **PA-BPE training-data config (gpt4: balanced vs FineWeb2-full)** — training corpus on a fixed gpt4 pretok (balanced vs FineWeb2-full); FineWeb2-full is far fairer multilingually.
- **Parity tuning — European-family up-weighting (original ×1.0 → ×1.1 → ×1.2)** — European-family ratio strength; a higher ratio slightly raises English bytes per token at a small fairness cost.
- **Tuned config — semitic regroup of script-mismatched languages (with vs without)** — regrouping script-mismatched languages into the semitic group; effects are local to those scripts, not the global averages.
- **SuperBPE base, transition point & stage-2 preset** — balanced-data SuperBPE sweep over base, transition (64k/90k) and stage-2 preset.
- **SuperBPE training data (balanced vs FineWeb2-full)** — balanced vs FineWeb2-full under SuperBPE; FineWeb2-full restores the multilingual fairness lost on balanced data.
- **Hybrid-window vs base parity, under SuperBPE** — hybrid-window vs base parity across the SuperBPE pretok families.
- **Algorithm / pretok (plain BPE vs Unigram, right-align digits, gpt2-style)** — plain-BPE pretok variants and Unigram LM (the single non-merge algorithm point).

## Appendix — additional ablations

The design points referenced from the body's ablation section, in full. Same table and extrinsic conventions as the body ablations.


### PA-BPE training-data config (gpt4: balanced vs FineWeb2-full)

This ablation compares two training corpora for PA-BPE, the balanced mixture against FineWeb2-full, holding the pretokenizer, parity mode, and capping fixed.

The further FineWeb2-full to tuned refinements (European ratio up-weighting, two quality removals, semitic regroup) are isolated for apertus in the EU-weighting and semitic-regroup ablations. Punctuation and whitespace capping is a pretokenizer choice with its own ablation, not a data-config change.

| Tokenizer | Multiling. sent/tok ↑ | Gini ↓ | Vocab-util CoV ↓ | Vocab size | Eng comp (B/tok) ↑ | Vocab util ↑ | Avg langs/token ↑ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Dead vocab ↓ | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PA-gpt4-balanced | 0.0138 | 0.415 | 0.4619 | 127,826 | **4.610** | **0.689** | 2.62 | 0.00043 | **0.01205** | 0.472 | 4 | 2837 | 4 | 59 |
| PA-gpt4-fineweb2full | **0.0235** | **0.076** | **0.3755** | 127,825 | 4.433 | 0.590 | **2.93** | 0.00043 | 0.02226 | **0.505** | **3** | 5673 | 8 | **33** |

*Faceted per-language compression (sentences/token), one pane per tokenizer:*

![PA-BPE training-data config (gpt4: balanced vs FineWeb2-full): per-language compression (sentences/token)](report_flores60/ablation_plots/pa-bpe-training-data-config-gpt4-balanced-vs-fineweb2-full/compression_rate_faceted.svg)

*Extrinsic (downstream LM):*
| Tokenizer | Val BPB ↓ | FLORES BPB (tr.) ↓ | Code BPB ↓ | MC-math ↑ | MBPP ↑ [95% CI] |
|---|---|---|---|---|---|
| PA-gpt4-balanced [matched] | 0.719 | 1.177 | 0.524 | — | — |
| PA-gpt4-fineweb2full [matched] | 0.728 | 1.169 | 0.531 | — | — |

*[proxy] tokenizer-lm 1B-balanced Δ, factor: Training data* (Δ = B−A; BPB Δ<0 means B better; **bold** = p_adj<0.05):
| A | B | ΔVal | ΔFLORES (tr.) | ΔFLORES (all) | ΔBLiMP | ΔCode |
|---|---|---|---|---|---|---|
| Balanced | English | **+0.0511** | **+0.0333** | **+0.0082** | +0.0124 | +0.0401 |
| Balanced | Code | **+0.0363** | **+0.0188** | **-0.0133** | +0.0259 | +0.0248 |
| Claude bal | Claude eng | **+0.0475** | **+0.0281** | +0.0003 | -0.0141 | +0.0270 |
| Punct bal | Punct eng | **+0.0419** | **+0.0208** | **-0.0096** | +0.0190 | +0.0310 |
| Balanced | High-res | +0.0013 | **+0.0250** | **-0.0153** | +0.0212 | +0.0034 |
| Balanced | High-mid | +0.0023 | **+0.0164** | **+0.0044** | +0.0119 | +0.0031 |

### Parity tuning — European-family up-weighting (original ×1.0 → ×1.1 → ×1.2)

This ablation compares three European-family up-weighting strengths in the parity config: ×1.0 (original), ×1.1, and ×1.2.

×1.0 is the original (untuned) config. ×1.1 and ×1.2 are the tuned config (European ratio up-weighting, two quality removals, semitic regroup) at two up-weighting strengths, so original to ×1.1 bundles all the tuning changes and ×1.1 to ×1.2 isolates the European up-weighting strength. The trainer selects the group/language with the minimum `compression_rate / ratio`, so a higher European ratio gives more merges for English and European and more English compression.

| Tokenizer | Eng comp (B/tok) ↑ | Multiling. sent/tok ↑ | Gini ↓ | Vocab-util CoV ↓ | Vocab size | Vocab util ↑ | Avg langs/token ↑ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Byte-frag (benign) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Apertus-pretok + PA-BPE (untuned data) | 4.335 | 0.0233 | **0.075** | **0.3860** | 127,835 | 0.592 | **2.84** | 0.00043 | **0.02202** | **0.505** | 5679 |
| Apertus-pretok + PA-BPE (European ×1.1) | 4.335 | 0.0233 | 0.077 | 0.3976 | 127,835 | 0.601 | 2.80 | 0.00043 | 0.02205 | 0.499 | 5645 |
| Apertus-pretok + PA-BPE | **4.336** | 0.0233 | 0.081 | 0.4130 | 127,835 | **0.606** | 2.79 | 0.00043 | 0.02208 | 0.502 | 5592 |

*Faceted per-language compression (sentences/token), one pane per tokenizer:*

![Parity tuning — European-family up-weighting (original ×1.0 → ×1.1 → ×1.2): per-language compression (sentences/token)](report_flores60/ablation_plots/parity-tuning-european-family-up-weighting-original-1-0-1-1-/compression_rate_faceted.svg)

*Extrinsic (downstream LM):*
| Tokenizer | Val BPB ↓ | FLORES BPB (tr.) ↓ | Code BPB ↓ | MC-math ↑ | MBPP ↑ [95% CI] |
|---|---|---|---|---|---|
| Apertus-pretok + PA-BPE [matched] | 0.729 | 1.170 | 0.531 | 0.270 | 0.058 [0.038, 0.080] |

### Tuned config — semitic regroup of script-mismatched languages (with vs without)

This ablation tests regrouping script-mismatched languages (`ydd_Hebr`, Hebrew script; `kas/knc/uzs_Arab`, Arabic script) into the semitic group so they share script-appropriate merges.

This is one of the three tuned fixes, isolated at ×1.2. The effect is local to those scripts' per-language fairness and boundary-crossing, not the global averages.

| Tokenizer | Gini ↓ | Vocab-util CoV ↓ | Multiling. sent/tok ↑ | Boundary-cross ↓ | Vocab size | Eng comp (B/tok) ↑ | Vocab util ↑ | Avg langs/token ↑ | CER ↓ | Operator-isol ↑ | Byte-frag (benign) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Apertus-pretok + PA-BPE | 0.081 | 0.4130 | 0.0233 | 0.02208 | 127,835 | 4.336 | **0.606** | 2.79 | 0.00043 | **0.502** | 5592 |
| Apertus-pretok + PA-BPE (no semitic regroup) | 0.081 | **0.4109** | 0.0233 | 0.02208 | 127,835 | 4.336 | 0.605 | **2.80** | 0.00043 | 0.498 | 5601 |

*Faceted per-language vocabulary utilization, one pane per tokenizer:*

![Tuned config — semitic regroup of script-mismatched languages (with vs without): per-language vocabulary utilization](report_flores60/ablation_plots/tuned-config-semitic-regroup-of-script-mismatched-languages-/vocabulary_utilization_faceted.svg)

*Extrinsic (downstream LM):*
| Tokenizer | Val BPB ↓ | FLORES BPB (tr.) ↓ | Code BPB ↓ | MC-math ↑ | MBPP ↑ [95% CI] |
|---|---|---|---|---|---|
| Apertus-pretok + PA-BPE [matched] | 0.729 | 1.170 | 0.531 | 0.270 | 0.058 [0.038, 0.080] |

### SuperBPE base, transition point & stage-2 preset

This ablation sweeps the SuperBPE base, the stage-1 to stage-2 transition vocab size (64k and 90k), and the stage-2 preset on balanced training data.

| Tokenizer | Eng comp (B/tok) ↑ | Multiling. sent/tok ↑ | Vocab util ↑ | Vocab size | Vocab-util CoV ↓ | Avg langs/token ↑ | Gini ↓ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Lossless ↑ | Dead vocab ↓ | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SuperBPE(PA-base)·gpt4o·t90k | 5.620 | 0.0137 | **0.662** | 128,000 | **0.4906** | 2.73 | 0.428 | 0.00043 | 0.01127 | 0.509 | 0.9867 | 1 | 2445 | 4 | **50** |
| SuperBPE(PA-base)·gpt4o·t64k | 5.869 | **0.0143** | 0.602 | 128,000 | 0.5094 | 2.93 | 0.400 | 0.00043 | 0.01336 | 0.493 | 0.9867 | 1 | 2103 | 6 | 72 |
| SuperBPE(PA-base)·clean-c2·t90k | 5.148 | 0.0141 | 0.652 | 128,000 | 0.5124 | 2.62 | 0.397 | 0.00043 | 0.01087 | **0.987** | 0.9867 | **0** | 2359 | 5 | 63 |
| SuperBPE(PA-base)·clean-c3·t90k | 5.598 | 0.0136 | 0.651 | 128,000 | 0.4978 | 2.78 | 0.429 | 0.00043 | **0.01030** | 0.627 | 0.9867 | **0** | 2357 | 5 | 55 |
| SuperBPE(plain-base)·gpt4o·noNFC | **6.159** | 0.0139 | 0.484 | 128,000 | 0.6230 | **3.39** | **0.387** | **0.00000** | 0.02663 | 0.452 | 1.0000 | 8 | 1156 | 6 | 92 |

*Faceted per-language compression (sentences/token), one pane per tokenizer:*

![SuperBPE base, transition point & stage-2 preset: per-language compression (sentences/token)](report_flores60/ablation_plots/superbpe-base-transition-point-stage-2-preset/compression_rate_faceted.svg)

*Extrinsic (downstream LM):*
| Tokenizer | Val BPB ↓ | FLORES BPB (tr.) ↓ | Code BPB ↓ | MC-math ↑ | MBPP ↑ [95% CI] |
|---|---|---|---|---|---|
| SuperBPE(PA-base)·gpt4o·t90k [matched] | 0.729 | 1.181 | 0.528 | — | — |
| SuperBPE(PA-base)·gpt4o·t64k [matched] | 0.729 | 1.180 | 0.530 | — | — |
| SuperBPE(PA-base)·clean-c2·t90k [matched] | 0.729 | 1.169 | 0.526 | — | — |
| SuperBPE(PA-base)·clean-c3·t90k [matched] | 0.730 | 1.173 | 0.531 | — | — |
| SuperBPE(plain-base)·gpt4o·noNFC [matched] | 0.724 | 1.173 | 0.525 | — | — |

*[proxy] tokenizer-lm 1B-balanced Δ, factor: SuperBPE* (Δ = B−A; BPB Δ<0 means B better; **bold** = p_adj<0.05):
| A | B | ΔVal | ΔFLORES (tr.) | ΔFLORES (all) | ΔBLiMP | ΔCode |
|---|---|---|---|---|---|---|
| GPT-4o BPE | + SuperBPE | +0.0122 | **+0.0157** | **+0.0063** | +0.0045 | +0.0066 |
| PA-BPE bal | + SuperBPE | +0.0037 | -0.0049 | **-0.0093** | +0.0032 | -0.0059 |
| t90k | t64k | +0.0001 | -0.0005 | -0.0004 | +0.0036 | +0.0022 |
| C2 (bal) | C3 (bal) | +0.0011 | **+0.0042** | **-0.0064** | -0.0186 | +0.0047 |

*[proxy] tokenizer-lm 1B-balanced Δ, factor: NFC normalization* (Δ = B−A; BPB Δ<0 means B better; **bold** = p_adj<0.05):
| A | B | ΔVal | ΔFLORES (tr.) | ΔFLORES (all) | ΔBLiMP | ΔCode |
|---|---|---|---|---|---|---|
| GPT-4o | + NFC | +0.0004 | -0.0033 | -0.0017 | +0.0320 | +0.0003 |
| Claude | + NFC | **-0.0012** | -0.0045 | **-0.0052** | -0.0080 | -0.0006 |
| RightAlign | + NFC | **-0.0007** | **-0.0061** | **-0.0056** | -0.0031 | -0.0002 |

### SuperBPE training data (balanced vs FineWeb2-full)

This ablation compares two training corpora under SuperBPE, the balanced mixture against FineWeb2-full.

| Tokenizer | Multiling. sent/tok ↑ | Gini ↓ | Eng comp (B/tok) ↑ | Vocab size | Vocab util ↑ | Vocab-util CoV ↓ | Avg langs/token ↑ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Dead vocab ↓ | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SuperBPE(PA-base)·gpt4o·t90k | 0.0137 | 0.428 | **5.620** | 128,000 | **0.662** | 0.4906 | 2.73 | 0.00043 | **0.01127** | **0.509** | 1 | 2445 | 4 | **50** |
| SuperBPE·gpt4·base·fw2full | 0.0223 | **0.092** | 5.071 | 128,000 | 0.522 | **0.4411** | 3.22 | 0.00043 | 0.02378 | 0.506 | 1 | 4522 | 11 | 70 |
| SuperBPE·gpt4·hw·fw2full | **0.0232** | 0.109 | 5.560 | 128,000 | 0.537 | 0.4821 | **3.27** | 0.00043 | 0.02712 | 0.467 | **0** | 3444 | 19 | 104 |

*Faceted per-language compression (sentences/token), one pane per tokenizer:*

![SuperBPE training data (balanced vs FineWeb2-full): per-language compression (sentences/token)](report_flores60/ablation_plots/superbpe-training-data-balanced-vs-fineweb2-full/compression_rate_faceted.svg)

*Extrinsic (downstream LM):*
| Tokenizer | Val BPB ↓ | FLORES BPB (tr.) ↓ | Code BPB ↓ | MC-math ↑ | MBPP ↑ [95% CI] |
|---|---|---|---|---|---|
| SuperBPE(PA-base)·gpt4o·t90k [matched] | 0.729 | 1.181 | 0.528 | — | — |
| SuperBPE·gpt4·hw·fw2full [matched] | pending | pending | pending | 0.265 | 0.070 [0.048, 0.092] |

### Hybrid-window vs base parity, under SuperBPE

This ablation compares the hybrid-window parity rule against base parity across the SuperBPE pretokenizer families.

| Tokenizer | Eng comp (B/tok) ↑ | Multiling. sent/tok ↑ | Gini ↓ | Vocab-util CoV ↓ | Vocab size | Vocab util ↑ | Avg langs/token ↑ | CER ↓ | Boundary-cross ↓ | Operator-isol ↑ | Dead vocab ↓ | Byte-frag (benign) | Long toks (>64) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SuperBPE·apertus-cap·base·fw2full | 5.011 | 0.0225 | 0.100 | 0.4591 | 128,000 | **0.551** | 3.06 | 0.00043 | 0.02476 | 0.494 | **0** | 4126 | 1 | 60 |
| Apertus-pretok + PA-BPE + SuperBPE | 5.402 | 0.0230 | 0.110 | 0.4992 | 128,000 | 0.544 | 3.14 | 0.00043 | 0.02686 | 0.466 | **0** | 3441 | 1 | 76 |
| SuperBPE·clean-cap·base·fw2full | 4.693 | 0.0220 | 0.100 | 0.4613 | 128,000 | 0.543 | 2.98 | 0.00043 | 0.02473 | 0.985 | **0** | 4217 | 1 | **53** |
| CleanV1-pretok + PA-BPE + SuperBPE | 5.013 | 0.0227 | 0.106 | 0.4892 | 128,000 | 0.550 | 3.02 | 0.00043 | 0.02629 | **0.987** | **0** | 3435 | 0 | 77 |
| SuperBPE·gpt4·base·fw2full | 5.071 | 0.0223 | **0.092** | **0.4411** | 128,000 | 0.522 | 3.22 | 0.00043 | **0.02378** | 0.506 | 1 | 4522 | 11 | 70 |
| SuperBPE·gpt4·hw·fw2full | **5.560** | **0.0232** | 0.109 | 0.4821 | 128,000 | 0.537 | **3.27** | 0.00043 | 0.02712 | 0.467 | **0** | 3444 | 19 | 104 |

*Faceted per-language compression (sentences/token), one pane per tokenizer:*

![Hybrid-window vs base parity, under SuperBPE: per-language compression (sentences/token)](report_flores60/ablation_plots/hybrid-window-vs-base-parity-under-superbpe/compression_rate_faceted.svg)

*Extrinsic (downstream LM):*
| Tokenizer | Val BPB ↓ | FLORES BPB (tr.) ↓ | Code BPB ↓ | MC-math ↑ | MBPP ↑ [95% CI] |
|---|---|---|---|---|---|
| Apertus-pretok + PA-BPE + SuperBPE [matched] | 0.733 | 1.176 | 0.541 | 0.269 | 0.004 [0.000, 0.010] |
| CleanV1-pretok + PA-BPE + SuperBPE [matched] | 0.732 | 1.161 | 0.536 | 0.268 | 0.196 [0.162, 0.232] |
| SuperBPE·gpt4·hw·fw2full [matched] | pending | pending | pending | 0.265 | 0.070 [0.048, 0.092] |

### Algorithm / pretok (plain BPE vs Unigram, right-align digits, gpt2-style)

This ablation compares plain-BPE pretokenizer variants (gpt2-style, right-aligned digits) against Unigram LM, the single non-merge algorithm point.

| Tokenizer | Operator-isol ↑ | Eng comp (B/tok) ↑ | Multiling. sent/tok ↑ | Gini ↓ | Vocab size | Vocab util ↑ | Vocab-util CoV ↓ | Avg langs/token ↑ | CER ↓ | Boundary-cross ↓ | Dead vocab ↓ | Byte-frag (benign) | Junk toks (≥8) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BPE-gpt2 | **0.987** | 4.761 | 0.0134 | 0.389 | 128,256 | 0.507 | 0.5843 | **3.24** | 0.00000 | **0.02119** | **0** | 1249 | 117 |
| BPE-rightalign | 0.478 | **4.796** | **0.0137** | 0.384 | 128,256 | 0.500 | 0.5809 | **3.24** | 0.00000 | 0.02668 | 5 | 1290 | **116** |
| Unigram-gpt4o | 0.887 | 3.093 | 0.0130 | **0.306** | 128,256 | **0.583** | **0.5201** | 2.84 | 0.00000 | 0.08215 | 12 | 9932 | 304 |

*Faceted per-language compression (sentences/token), one pane per tokenizer:*

![Algorithm / pretok (plain BPE vs Unigram, right-align digits, gpt2-style): per-language compression (sentences/token)](report_flores60/ablation_plots/algorithm-pretok-plain-bpe-vs-unigram-right-align-digits-gpt/compression_rate_faceted.svg)

*Extrinsic (downstream LM):*
| Tokenizer | Val BPB ↓ | FLORES BPB (tr.) ↓ | Code BPB ↓ | MC-math ↑ | MBPP ↑ [95% CI] |
|---|---|---|---|---|---|
| BPE-gpt2 [matched] | 0.713 | 1.157 | 0.515 | — | — |
| BPE-rightalign [matched] | 0.712 | 1.160 | 0.519 | 0.295 | 0.062 [0.042, 0.084] |
| Unigram-gpt4o [matched] | 0.731 | 1.190 | 0.554 | — | — |

*[proxy] tokenizer-lm 1B-balanced Δ, factor: Algorithm* (Δ = B−A; BPB Δ<0 means B better; **bold** = p_adj<0.05):
| A | B | ΔVal | ΔFLORES (tr.) | ΔFLORES (all) | ΔBLiMP | ΔCode |
|---|---|---|---|---|---|---|
| BPE GPT-4o | Unigram | **+0.0331** | **+0.0329** | **+0.0472** | +0.0545 | +0.0355 |
| BPE Claude | Unigram | **+0.0304** | **+0.0308** | **+0.0408** | -0.0037 | +0.0355 |
| BPE RightAlign | Unigram | **+0.0300** | **+0.0312** | **+0.0452** | +0.0336 | +0.0343 |

*[proxy] tokenizer-lm 1B-balanced Δ, factor: Unigram tuning* (Δ = B−A; BPB Δ<0 means B better; **bold** = p_adj<0.05):
| A | B | ΔVal | ΔFLORES (tr.) | ΔFLORES (all) | ΔBLiMP | ΔCode |
|---|---|---|---|---|---|---|
| Untuned | Tuned | +0.0018 | **+0.0019** | **+0.0061** | -0.0846 | +0.0027 |

### SuperBPE vs. its PA-BPE base — what the superword stage changes

Each SuperBPE is compared to the PA-BPE subword base it was grown from (stage-1 = that base; stage-2 = superword merges). **Added** = tokens only in the SuperBPE; **sacrificed** = subword tokens only in the base; a **superword** has an internal space (spans ≥2 pretokenized words). Added-token usage is measured on FLORES devtest, code firing on a fixed code sample. Source: `results/report_superbpe_vs_base.json`.

| SuperBPE | Base | Pretok | Mode | Shared | Sacrificed | Added | Superword % | Eng added-share | Mean added-share | Code added % |
|---|---|---|---|---|---|---|---|---|---|---|
| CleanV1-pretok + PA-BPE + SuperBPE | CleanV1-pretok + PA-BPE | clean | hw | 93,506 | 34,329 | 34,494 | 38% | 0.184 | 0.063 | 5.4% |
| Apertus-pretok + PA-BPE + SuperBPE | Apertus-pretok + PA-BPE | apertus | hw | 92,467 | 35,368 | 35,533 | 41% | 0.214 | 0.077 | 18.9% |
| SuperBPE·gpt4·hw·fw2full | PA-gpt4-fineweb2full | gpt4 | hw | 92,934 | 34,891 | 35,066 | 42% | 0.219 | 0.068 | 20.9% |
| SuperBPE·clean-cap·base·fw2full | PA-Clean-capped-base | clean | base | 94,460 | 33,375 | 33,540 | 20% | 0.426 | 0.100 | 23.5% |
| SuperBPE·apertus-cap·base·fw2full | PA-Apertus-base | apertus | base | 93,928 | 33,907 | 34,072 | 23% | 0.464 | 0.107 | 38.8% |
| SuperBPE·gpt4·base·fw2full | PA-gpt4-fw2full-base | gpt4 | base | 94,513 | 33,312 | 33,487 | 21% | 0.457 | 0.111 | 42.6% |

**Sacrificed subwords.** The removed subwords are mostly low-resource, non-Latin-script fragments (for example ` zdravje`, ` хүүхд`, `nuti`, ` brifysgol`, `�້າງ`, ` ກົ`). The SuperBPE stage removes these and adds superwords.

**Added tokens.** Across all pairs the added tokens are used mostly by English and other space-delimited languages, and almost never by CJK, Indic, or Thai (near zero). Hybrid-window bases add more superwords (38–42% of added tokens) than base-parity bases (20–23%). Base-parity additions are the most English-concentrated (Eng added-share 0.43–0.46, against 0.18–0.22 for hybrid-window).

**Code-related superwords (hybrid-window pairs).**
- **clean**: 5.4% of code-sample tokens are SuperBPE-added, and 266 added superwords contain code syntax or keywords. In-sample superwords: ` compute the`, ` for i in`, ` if i`, ` not in`, ` or `, ` over a`
- **apertus**: 18.9% of code-sample tokens are SuperBPE-added, and 2831 added superwords contain code syntax or keywords. In-sample superwords: ` + `, ` - `, ` = `, ` compute the`, ` for i`, ` if i`, ` in range`, ` not in`, ` or `, ` over a`
- **gpt4**: 20.9% of code-sample tokens are SuperBPE-added, and 2866 added superwords contain code syntax or keywords. In-sample superwords: ` + `, ` - `, ` = `, ` compute the`, ` for i`, ` if i`, ` in range`, ` not in`, ` or `, ` over a`

The pretokenizer determines the code impact. clean-multi keeps operators as separate tokens (its space-only leading rule blocks operator+space merges), so its code superwords are mostly natural-language phrases inside comments and strings. apertus and gpt4 allow superwords that span operators and markup (for example ` = `, ` + `, `) * `, `] =`, `<div class`), which produces 3–4× as many disrupted code tokens. The clean-multi SuperBPE has the fewest disrupted code tokens of the SuperBPE set.

## Appendix — per-language plots faceted by PA-BPE training family

One panel per linguistic family from the PA-BPE *tuned* parity config (`parity_aware_config_grouped_fineweb2full_quota_tuned.json`); each panel plots a metric per language within the family, with one line per tokenizer. Restricted to the 4 candidates, the Apertus baseline, and 5 open-source references. Plots are over the full FLORES devtest set (205 languages), filtered to families with at least 2 languages in that set.


**Compression rate (sent/tok, higher = better)**:

![Compression rate (sent/tok, higher = better)](family_plots/compression_rate_by_family.svg)

**Vocabulary utilization (fraction of vocab used, higher = more reuse)**:

![Vocabulary utilization (fraction of vocab used, higher = more reuse)](family_plots/vocabulary_utilization_by_family.svg)
