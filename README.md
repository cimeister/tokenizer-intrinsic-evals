# TokEval
A toolkit for computing intrinsic quality metrics for tokenizers across natural language, code, and math.


## Quick Start

Get up and running in 30 seconds:

```bash
# Clone and install
git clone https://github.com/cimeister/tokenizer-intrinsic-evals.git
cd tokenizer-intrinsic-evals
uv sync

# Run demo analysis with built-in sample data
uv run tokenizer-analysis --use-sample-data

# View results
open results/fertility_individual.svg  # Basic metric comparison chart
```

This will analyze two sample tokenizers (BPE and Unigram) across 5 languages and generate comparison plots.

## Visualizing Tokenization

The `tokenizer-visualize` command renders token boundaries directly on source text, making it easy to inspect how different tokenizers split code, math, and multilingual content.

```bash
# Show built-in samples (Python code, LaTeX math, multilingual text)
uv run tokenizer-visualize \
    --tokenizer-config configs/sample_tokenizers.json

# Show only specific tokenizers
uv run tokenizer-visualize \
    --tokenizer-config configs/sample_tokenizers.json \
    --tokenizers "bpe" "unigramlm"

# Visualize all files in a directory
# Files can contain multiple samples separated by a line with only "---".
# Use --samples-per-file to control how many are read (default: 1).
uv run tokenizer-visualize \
    --tokenizer-config configs/sample_tokenizers.json \
    --input data/samples/ --samples-per-file 3

```

Each sample is shown with line-numbered source text followed by a colour-coded token-boundary view for every tokenizer, plus whitespace and indentation statistics.

## Sanity Checking a Tokenizer

The `tokenizer-sanity-check` command runs a single-tokenizer health report: byte coverage, whitespace and digit handling, special-token behaviour, determinism, Unicode normalization, vocabulary integrity, and vocabulary reachability. It flags each check as pass, warn, or fail and sets a non-zero exit code when a check fails, so it can gate a tokenizer before a full analysis.

```bash
# Check a single tokenizer (CLASS:PATH form)
uv run tokenizer-sanity-check huggingface:tokenizers/bpe.json

# Check every tokenizer listed in a config
uv run tokenizer-sanity-check --tokenizer-config configs/sample_tokenizers.json

# Restrict a config run to one tokenizer by name
uv run tokenizer-sanity-check --tokenizer-config configs/sample_tokenizers.json --only bpe

# Add multilingual breadth (requires a language config)
uv run tokenizer-sanity-check huggingface:tokenizers/bpe.json \
    --use-sample-data --language-config configs/core_lang_config.json
```

Use `--exit-zero` to always return exit code 0 (report without gating) and `--quiet` to collapse passing checks in the text report.

## Setup

### Requirements
- Python 3.10+
- Git (for submodules)

### Full Installation
```bash
git clone https://github.com/cimeister/tokenizer-intrinsic-evals.git
cd tokenizer-intrinsic-evals
uv sync

# Optional: MorphScore morphological analysis
git submodule update --init --recursive
uv pip install -e ./morphscore

# Optional: reading code corpora from parquet files
uv sync --extra parquet
```

**MorphScore note**: Only `<ISO 639-3>_<script>` language codes are automatically mapped. Data files must be downloaded separately (see [MorphScore README](morphscore/README.md)) and placed in `morphscore_data/`.

## Usage

### Common CLI Options

| Flag | Description |
|------|-------------|
| `--tokenizer-config FILE` | JSON file with tokenizer configurations |
| `--language-config FILE` | JSON file with languages and analysis groups |
| `--measurement-config FILE` | JSON file with text measurement method |
| `--use-sample-data` | Use built-in demo data |
| `--output-dir DIR` | Output directory (default: `results/`) |
| `--verbose` | Detailed console output |
| `--no-plots` | Skip plot generation |
| `--save-full-results` | Save detailed JSON results |
| `--run-grouped-analysis` | Group analysis by script families / resource levels |
| `--per-language-plots` | Per-language grouped bar charts |
| `--faceted-plots` | One subplot per tokenizer with shared y-axis |
| `--filter-script-family FAMILY` | Filter languages by script family |
| `--morphscore` | Enable MorphScore analysis |
| `--morphscore-config FILE` | Custom MorphScore configuration |
| `--code-ast-config FILE` | JSON mapping languages to code paths for AST analysis |
| `--no-code-ast` | Skip AST boundary analysis |
| `--no-digit-boundary` | Skip math metrics (digit boundaries, operators) |
| `--math-data FILE` | Math-rich text file (.txt/.json) for digit boundary metrics |
| `--no-utf8-integrity` | Skip UTF-8 character boundary metrics |
| `--no-reconstruction` | Skip reconstruction fidelity analysis (see [Performance](#performance)) |
| `--generate-latex-tables` | Generate LaTeX tables |
| `--update-results-md [PATH]` | Generate/update cumulative Markdown leaderboard |
| `--dataset NAME` | Dataset label for the results table |
| `--sort-results-by METRIC` | Sort results table by metric key |
| `--samples-per-lang N` | Text samples per language |
| `--save-tokenized-data` | Cache tokenized data for reuse |
| `--no-global-lines` | Hide global average lines in plots |

### Markdown Results Table

Generate a cumulative Markdown leaderboard that grows across successive runs. Each run merges new tokenizer rows into the existing table — previously evaluated tokenizers are preserved, re-evaluated ones are updated.

```bash
# Generate / update a local RESULTS.md
uv run tokenizer-analysis --use-sample-data --update-results-md --dataset flores

# Custom output path
uv run tokenizer-analysis --use-sample-data --update-results-md my_results.md
```

Each row is keyed by `tokenizer_name (user, dataset)` — different users or datasets produce separate rows, while re-running the same combination updates in place.

## Configuration Files

### Tokenizer Configuration

Specify tokenizers via `--tokenizer-config`:

```json
{
  "tokenizer1": {
    "class": "huggingface",
    "path": "bert-base-uncased"
  },
  "tokenizer2": {
    "class": "huggingface",
    "path": "/path/to/local/tokenizer"
  },
  "custom_bpe": {
    "class": "custom_bpe",
    "path": "/path/to/bpe/directory"
  }
}
```

Available classes: `"huggingface"` (aliases `"hf"`, `"transformers"`), `"sentencepiece"`, `"custom_bpe"` (requires `vocab.json` + `merges.txt`), `"unimixlm"`, and `"pretokenized"` (for pre-tokenized data).

### Data Configuration

Specify languages and analysis groupings via `--language-config`:

```json
{
  "languages": {
    "eng_Latn": {
      "name": "English",
      "iso_code": "en",
      "data_path": "/path/to/english/data"
    },
    "arb_Arab": {
      "name": "Arabic",
      "iso_code": "ar",
      "data_path": "/path/to/arabic/data"
    }
  },
  "analysis_groups": {
    "script_family": {
      "Latin": ["eng_Latn", "fra_Latn"],
      "Arabic": ["arb_Arab"]
    },
    "resource_level": {
      "high": ["eng_Latn"],
      "low": ["som_Latn"]
    }
  }
}
```

For simple setups, `"languages"` can map codes directly to file paths: `{"en": "/path/to/data"}`.

### Text Measurement Configuration

Control how text "length" is measured for metric normalization via `--measurement-config`:

| Method | `method` value | Counting key and options | Default for |
|--------|----------------|--------------------------|-------------|
| Bytes | `"bytes"` | `byte_counting`: `"utf8"`, `"hf_bytelevel"` | Compression metrics |
| Characters | `"characters"` | (none) | — |
| Lines | `"lines"` | `line_counting`: `"single"`, `"newline_split"`, `"custom_regex"` | Gini metrics |
| Words | `"words"` | `word_counting`: `"python_split"`, `"hf_whitespace"`, `"regex_whitespace"`, `"custom_regex"` | Fertility |

`include_empty_splits` (bool, default `false`) affects word and line counting. `custom_regex` (string) is required when a counting method is set to `"custom_regex"`. Unknown keys are rejected with an error.

Example:
```json
{
  "method": "lines",
  "line_counting": "newline_split",
  "include_empty_splits": false
}
```

### MorphScore Configuration

Specify via `--morphscore-config`:

```json
{
    "data_dir": "/path/to/morphscore/datasets",
    "by_split": false,
    "freq_scale": true,
    "exclude_single_tok": false
}
```

Requires languages in `<ISO 639-3>_<script>` format (e.g., `eng_Latn`). Override with `"language_subset"` in the config to bypass code mapping. Download datasets from [MorphScore README](morphscore/README.md).

### Code AST Configuration

Specify source code paths for AST boundary analysis via `--code-ast-config`:

```json
{
  "python": "/path/to/python/files/",
  "javascript": "/path/to/js/files.parquet",
  "java": "/path/to/java/dir/"
}
```

Supports 19 languages. Parquet files should have a `content` column; StarCoder metadata prefixes are stripped automatically. Without a config file, built-in synthetic code samples are used.

### Pre-tokenized Data

```bash
# Save tokenized data for reuse
uv run tokenizer-analysis --use-sample-data \
    --save-tokenized-data --tokenized-data-output-path my_data.pkl

# Reuse cached data (faster — no re-encoding)
uv run tokenizer-analysis \
    --tokenized-data-file my_data.pkl \
    --tokenized-data-config my_data_config.json
```

The save step auto-generates a config file and per-tokenizer vocabulary files. For manually prepared pre-tokenized data, provide a pickle/JSON dict mapping tokenizer names to lists of `TokenizedData` objects, a JSON config pointing to vocabulary files, and line-by-line vocabulary text files.

## Output Structure
```
results/
├── fertility_individual.svg         # Metric comparison charts
├── compression_rate_individual.svg
├── vocabulary_utilization_individual.svg
├── grouped_plots/                   # Cross-tokenizer comparisons
├── per-language/                    # Language-specific analysis
├── latex_tables/                    # Academic publication tables
├── RESULTS.md                       # Cumulative Markdown leaderboard
├── analysis_results.json            # Key metrics summary
├── analysis_results_full.json       # Detailed results (--save-full-results)
└── tokenized_data.pkl               # Cached data (--save-tokenized-data)
```

### JSON Results Schema

`analysis_results.json` is always written with a slimmed schema. Pass `--save-full-results` to also write `analysis_results_full.json` with all computed data. Both files follow the same per-metric layout:

```json
{
  "<metric_name>": {
    "per_tokenizer": {
      "<tokenizer_name>": {
        "global": {},
        "per_language": {"<lang_code>": {}}
      }
    },
    "per_language": {"<lang_code>": {"<tokenizer_name>": "<value>"}},
    "metadata": {}
  }
}
```

- **`global`**: Aggregate score for this tokenizer. Stats dicts contain `mean`, `std`, `median`, `count`; structured dicts vary by metric.
- **`per_language`** (inside `per_tokenizer`): Per-language breakdown for this tokenizer.
- **`per_language`** (top-level): Cross-tokenizer leaderboard keyed by language, where present in raw data.
- **`metadata`**: Metric configuration and data provenance, where present.

The slimmed file omits `pairwise_comparisons`, `summary`, `per_category` breakdowns, and derivable stat fields (`sum`, `std_err`, `min`, `max`). The full results file includes `per_category` for metrics that have category breakdowns (e.g. AST node types, operator types). Some metrics use additional keys (e.g. `by_digit_length`, `scaling` for digit metrics; `character_length`/`byte_length` for token length).

## Metrics

### Basic Tokenization Metrics
- **Compression Rate** (`compression_rate`): Ratio of total text units (bytes/chars/lines) to total tokens across the corpus — measures encoding efficiency
- **Fertility** (`fertility`): Tokens per word/character — measures tokenization granularity
- **Token Length**: Average token size in bytes/characters
- **Type-Token Ratio**: Unique tokens / total tokens — measures vocabulary usage diversity
- **Vocabulary Utilization** (`vocabulary_utilization`): Fraction of vocabulary actually used

### Information-Theoretic Metrics
- **Renyi Entropy**: Information content at different alpha values — generalizes Shannon entropy
- **Average Token Rank** (`avg_token_rank`): Typical position of tokens within the frequency-ordered vocabulary
- **Bigram Entropy** (`bigram_entropy`): For each token type, looks at what tokens follow it in the corpus and measures whether the followers are evenly spread or dominated by one or two tokens. A score of 1.0 means every token's followers are perfectly balanced; a score near 0 means most tokens are almost always followed by the same thing. Can interpret this as "how easy the tokenizer makes a very simple case of language modeling." Token types that appear too rarely (fewer than 3 times by default, configurable) are ignored to avoid noisy estimates. Bigrams do not cross document boundaries. Based on the Shannon efficiency metric (η) from [Poelman et al. 2025](https://aclanthology.org/2025.emnlp-main.369/), EMNLP.

### Morphological Metrics
- **MorphScore V2** (`morphscore_recall`, `morphscore_precision`): Morphological evaluation ([Arnett et al. 2025](https://arxiv.org/abs/2507.06378)). Enable with `--morphscore` or `--morphscore-config` (requires raw tokenization and the MorphScore submodule).

### Mathematical Content Metrics

Evaluates tokenizer handling of mathematical expressions. Based on Singh & Strouse (2024, [arXiv:2402.14903](https://arxiv.org/abs/2402.14903)), who showed that right-to-left tokenization of numbers improved arithmetic accuracy by >22 percentage points. Disable with `--no-digit-boundary`.

> **Data scope:** When `--math-data FILE` or `--use-builtin-math-data` is set — as in the recommended invocation above — these metrics are computed **only on the dedicated math texts**, *not* the general multilingual corpus (the math data replaces the corpus for this metric group). Without either flag they fall back to whatever numbers/operators appear in the main corpus. All per-language results in this group are therefore reported under the synthetic language `math` when dedicated math data is used.

#### Three-Digit Place-Value Boundary Alignment (`three_digit_boundary_f1`)

Measures whether numbers are tokenized with right-aligned 3-digit groupings that match place-value structure (units, thousands, millions).

For each number, compares actual token boundaries against ideal boundaries at positions L-3, L-6, L-9 from the left. Reports precision, recall, and F1. Short numbers (<=3 digits) that remain single tokens score F1 = 1.0; short numbers needlessly split score F1 = 0.

**Example:** The number `1234567` has ideal boundaries at positions 1, 4 — yielding `1|234|567` (millions, thousands, units). A tokenizer producing `1|234|567` scores F1 = 1.0. One producing `12|345|67` scores F1 = 0.0 — it has three boundaries but none at the right positions. A short number like `42` kept as a single token scores F1 = 1.0 (no boundaries needed, none placed). But `42` split into `4|2` scores F1 = 0.0 — a boundary was placed where none was needed.

**Why it matters:** Singh & Strouse (2024) showed that right-to-left digit grouping improves arithmetic accuracy by ensuring corresponding digit positions across operands occupy consistent token positions.

#### Digit Split Variability

For numbers of the same digit length, measures Shannon entropy of the distribution of boundary patterns. Low entropy means the tokenizer uses a consistent splitting scheme; high entropy means chaotic splitting.

Entropy is computed on patterns pooled across languages, not averaged per-language. Reports Shannon entropy (bits), dominant pattern, and dominant frequency per digit-length bucket.

**Example:** A corpus contains three 5-digit numbers. If all are split as `XX|XXX` (pattern `{2}`), the entropy for the 5-digit bucket is 0.0 bits — perfectly consistent. If instead one is split `XX|XXX`, one as `X|XXXX`, and one as `XXX|XX`, there are three distinct patterns with equal frequency, giving entropy of log2(3) ≈ 1.58 bits. The first tokenizer has a learnable (if wrong) scheme; the second forces the model to handle every number as a special case.

**Why it matters:** A tokenizer with moderate F1 but low entropy has a consistent-but-wrong scheme (potentially fixable by retraining). Moderate F1 with high entropy indicates a deeper structural problem.

#### Numeric Magnitude Consistency

Tracks fertility-per-digit (tokens per digit) across digit lengths. Reports Spearman correlation, coefficient of variation, and linear fit (slope, R-squared) between digit length and mean token count.

**Example:** A tokenizer has memorized `0`-`999` as single vocabulary entries, so 1-digit numbers cost 1 token (1.0 tokens/digit), 2-digit numbers cost 1 token (0.5 tokens/digit), and 3-digit numbers cost 1 token (0.33 tokens/digit). Then at 4 digits, it fragments: `1234` -> `12|34` (0.5 tokens/digit). At 7 digits: `1234567` -> `123|45|67` (0.43 tokens/digit). The discontinuity between 3 and 4 digits — where fertility-per-digit jumps from 0.33 to 0.5 — shows up as a break in the linear fit and a low R-squared value. A smooth tokenizer would instead show a near-constant ratio across all digit lengths.

**Why it matters:** Tokenizers trained on natural language often have dense vocabulary coverage for small numbers (0-999 as single tokens) but fragment larger numbers unpredictably, creating representational discontinuities.

#### Operator Isolation Rate (`operator_isolation`)

Fraction of mathematical operators (`+`, `-`, `*`, `=`, `<=`, etc.) tokenized as standalone tokens rather than merged with adjacent content. The hyphen-minus `-` is always treated as an operator, even when it appears as a unary negative sign (e.g., `-42`), since disambiguating unary minus from subtraction requires expression parsing. Includes a compound preservation sub-metric measuring whether multi-character operators (`**`, `<=`, `!=`) are kept as single tokens vs. split.

**Example:** In the expression `3+5>=8`, a good tokenizer produces `3` | `+` | `5` | `>=` | `8` — isolation rate 1.0 and compound preservation 1.0. A bad tokenizer produces `3+` | `5` | `>` | `=` | `8` — the `+` is merged with `3` (isolation fails), and `>=` is split into `>` and `=` (compound preservation fails). Isolation rate: 1/3 (only `=` might be isolated depending on boundaries). Compound preservation: 0/1.

**Why it matters:** Merging an operator with its operand (e.g., `+3` as one token) forces the model to disentangle operation from value within a single embedding.

### Reconstruction Fidelity Metrics

Measures how lossy the encode→decode round-trip is. Tokenizers can lose information through normalization, UNK substitution, whitespace mangling, and decode asymmetry. These metrics run on language text, code, and math data. Requires that the tokenizer supports decoding (most do); non-decodable tokenizers are silently skipped.

#### Round-trip Exact Match Rate (`exact_match_rate`)

Fraction of texts where `decode(encode(text)) == text`. A score of 1.0 means the tokenizer is perfectly lossless for the evaluated data.

**Example:** The text `"Hello, world!"` is encoded to `[15496, 11, 995, 0]` and decoded back to `"Hello, world!"` — exact match. The text `"café"` is encoded and decoded to `"cafe"` (accent stripped by normalization) — not an exact match.

#### Character Error Rate (`mean_cer`)

Levenshtein edit distance between the original text and the decoded text,
normalized by the length of the original. Measures the fraction of
single-character insertions, deletions, and substitutions needed to
transform the decoded text back into the original.

CER = 0 means a perfect round-trip. **Note:** CER can exceed 1.0 when the
decoded text is much longer than the original (e.g., a tokenizer that
expands byte-fallback tokens into multi-character escape sequences).

**Example:** Original `"hello"` decoded as `"helo"` → edit distance 1 / 5
characters = CER 0.2. Original `"a"` decoded as `"abcd"` → edit distance
3 / 1 character = CER 3.0.

#### UNK Token Rate (`unk_token_rate`)

Fraction of encoded tokens that are the tokenizer's UNK token ID. Measures how much of the input the tokenizer cannot represent. A rate of 0.0 means no unknown tokens were produced.

**Example:** Encoding `"𝕳𝖊𝖑𝖑𝖔"` produces `[UNK, UNK, UNK, UNK, UNK]` — UNK rate 1.0. Encoding `"Hello"` produces `[15496]` — UNK rate 0.0.

#### Whitespace Fidelity (`whitespace_fidelity`)

Fraction of whitespace characters (spaces, tabs, newlines) in the original text that are preserved through the encode-decode round-trip. Uses a greedy forward-scan alignment to pair characters.

**Example:** Original `"a b\tc"` decoded as `"a b c"` (tab replaced by space) has 1 out of 2 whitespace chars preserved = fidelity 0.5.

### UTF-8 Character Boundary Metrics

Evaluates how byte-level tokenizers handle multi-byte UTF-8 characters at token boundaries. Runs on any text data (no special config needed). Disable with `--no-utf8-integrity`.

#### Token UTF-8 Completeness Rate

Fraction of content tokens whose bytes form complete UTF-8 characters. A token like `<0xC3>` (a single byte from the two-byte sequence for `é`) is incomplete — it contains the start of a character but not the whole thing. This is a natural consequence of byte-level tokenization, not an error: byte-fallback tokens are working as designed. The completeness rate measures how often the tokenizer's vocabulary is expressive enough to represent whole characters rather than resorting to sub-character byte sequences.

**Example:** The character `é` (U+00E9) is encoded as bytes `C3 A9`. A tokenizer that keeps `café` as `caf` | `é` produces two tokens, both containing complete UTF-8 — completeness rate 1.0. A byte-fallback tokenizer that produces `caf` | `<0xC3>` | `<0xA9>` has 3 content tokens, of which 2 contain incomplete UTF-8 sequences — completeness rate 1/3.

#### Character Boundary Crossing Rate (`utf8_boundary_crossing`)

Fraction of content tokens that cross a UTF-8 character boundary — tokens containing bytes from more than one UTF-8 character where at least one of those characters is incomplete within the token. These tokens are the direct product of BPE merges that fused bytes across character boundaries, permanently preventing the affected characters from being represented as whole tokens.

This is distinct from simple byte-fallback tokens. A byte-fallback token like `<0xC3>` is incomplete but does not cross a boundary — it holds bytes from exactly one character. A boundary-crossing token like one containing `A9 E4` (the tail byte of `é` merged with the leading byte of a CJK character) spans two characters and completes neither.

**Example:** Consider bytes `C3 A9 E4 BD A0` (the characters `é你`). A BPE tokenizer that merges the last byte of `é` with the first byte of `你` might produce `C3` | `A9 E4` | `BD A0`. The middle token `A9 E4` crosses a character boundary — it contains the continuation byte of `é` and the leading byte of `你`, completing neither character. The crossing rate would be 1/3.

**Why it matters:** Boundary-crossing tokens are fundamentally unrecoverable. While a byte-fallback token can be recombined with its neighbors to reconstruct a character, a boundary-crossing token has fused bytes from different characters in a way that no amount of context can cleanly separate within a single embedding.

#### Character Boundary Split Count (`utf8_char_split`)

Counts how many multi-byte characters in the source text have their constituent bytes spread across multiple tokens. To decide this, each token's bytes are reconstructed and aligned to the source text. The split rate is `splits / aligned multi-byte characters`, and splits per 1k tokens is also reported.

**Example:** The Chinese text `你好` contains two 3-byte characters (`你` = `E4 BD A0`, `好` = `E5 A5 BD`). A tokenizer that keeps each character as a single token has 0 splits. A byte-fallback tokenizer that splits `你` into `<0xE4>` | `<0xBD>` | `<0xA0>` has 1 split (the character's bytes span 3 different tokens). The split rate would be 1/2 = 0.5 if `好` remains intact.

**Alignment reliability.** When a tokenizer does not reproduce the source bytes (for example an English-trained tokenizer on Cyrillic or CJK, where some characters are dropped or replaced by placeholders), alignment cannot map every source byte to a token. A multi-byte character with any unaligned byte cannot be classified as split-or-not, so it is excluded from both the numerator and the denominator and counted separately. Three fields report this per language and globally: `unaligned_multibyte_chars` (excluded count), `aligned_fraction` (the share of multi-byte characters that aligned), and `alignment_mismatches` (the raw count of unaligned source bytes). When no multi-byte character aligns, `split_rate`, `splits_per_1k_multibyte`, and the per-byte-width split rates are `null` rather than `0.0`, so "no data" is not read as "no splits". A low `aligned_fraction` means the split rate rests on few characters and should be treated with caution. For tokenizers that reproduce the source exactly (`alignment_mismatches == 0`), every multi-byte character aligns and the split rate is unaffected by this handling.

**Corpus resolution.** The denominator is the number of aligned multi-byte characters in the evaluated text. On ASCII-dominant languages measured on a small parallel corpus this is tiny (the English FLORES sample has 57 multi-byte characters), so `split_rate` takes only a few distinct values and carries little information for Latin-script languages. A corpus with more natural multi-byte content (for example FineWeb2) gives better resolution.

**Why it matters:** Split characters are the text-centric complement to the token-centric completeness metric. A tokenizer might have few incomplete tokens overall (high completeness rate) but still split most multi-byte characters because each split produces multiple incomplete tokens, so the split count reveals the actual character-level impact.

### Code Tokenization Metrics

Evaluates tokenizer handling of source code by parsing it with tree-sitter and measuring alignment between AST node boundaries and token boundaries. Tree-sitter support is installed by default. Supports 19 languages (Python, JavaScript, Java, C, C++, Go, Rust, TypeScript, PHP, Ruby, C#, Scala, Swift, Kotlin, Lua, R, Perl, Haskell, Bash). Configure with `--code-ast-config`; disable with `--no-code-ast`.

> **Data scope:** These metrics are **always** computed on dedicated source-code snippets (loaded via `--code-ast-config`, or small built-in synthetic samples as a fallback) — the general multilingual corpus passed to the analyzer is **never** used for this metric group, regardless of flags.

#### AST Leaf-Node Boundary Alignment (`ast_full_alignment`)

Parses source code with tree-sitter, extracts leaf-node spans, and measures the fraction whose boundaries coincide with token boundaries. Tracks five categories independently: identifiers, keywords, operators, literals, and delimiters.

Reports start-alignment rate, end-alignment rate, full-alignment rate, and cross-boundary rate, broken down by category and language.

**Example:** For the Python snippet `return total`, tree-sitter identifies `return` (keyword, bytes 0-6) and `total` (identifier, bytes 7-12). If the tokenizer produces `return` | ` total` — both AST nodes fully align with token boundaries: full alignment = 1.0. If it produces `ret` | `urn total` — the keyword `return` has start-aligned = True (token changes at position 0) but end-aligned = False (positions 5 and 6 share a token with position 7), so fully_aligned = False. The identifier `total` has start-aligned = False (it shares a token with `urn`), so it also fails. Full alignment rate: 0/2 = 0.0.

**Why it matters:** Code has deterministic grammar, so AST node boundaries are objectively derivable with no manual annotation. A tokenizer that splits `return` into `ret` + `urn` fragments a syntactically atomic unit.

#### Identifier Fragmentation Rate (`ident_fragmentation`)

Fraction of programmer-defined identifiers split into multiple tokens, plus average tokens per identifier. Computed occurrence-weighted from the same AST extraction pass.

**Example:** A Python file contains identifiers `self` (x10 occurrences), `i` (x5), `process_data` (x3), and `MyAuthenticationFactory` (x1). The tokenizer keeps `self`, `i` as single tokens but splits `process_data` -> `process` | `_` | `data` (3 tokens) and `MyAuthenticationFactory` -> `My` | `Auth` | `entication` | `Factory` (4 tokens). Fragmentation rate: 4 fragmented occurrences out of 19 total = 0.21. Average tokens per identifier: (10x1 + 5x1 + 3x3 + 1x4) / 19 = 1.47. Note that the 10 occurrences of `self` dominate the metric and mask the fragmentation of the rarer, semantically richer identifiers.

**Why it matters:** Identifiers carry domain-specific semantics. Fragmenting `getUserName` into arbitrary sub-pieces destroys meaningful structure, though the current implementation does not yet distinguish semantically-aligned splits (at camelCase/snake_case boundaries) from arbitrary ones.

#### Indentation Depth Proportionality Correlation (`indent_depth_corr`)

Measures whether the number of whitespace tokens a tokenizer produces for leading indentation grows proportionally with nesting depth. Computes the Spearman rank correlation (ρ) between logical nesting depth (from tree-sitter) and the count of whitespace-only tokens in the leading indentation of each line. Only evaluated on whitespace-significant languages (Python, YAML). Requires at least 3 distinct depth levels per language; languages with fewer are skipped.

**Example:** A Python file has lines at depths 1, 2, 3, and 4. A proportional tokenizer encodes depth-1 indentation as 1 whitespace token, depth-2 as 2, depth-3 as 3, and depth-4 as 4 — perfect rank correlation, ρ = 1.0. A tokenizer that merges all indentation into a single token regardless of depth (1, 1, 1, 1 whitespace tokens) produces ρ ≈ 0.0. A tokenizer that uses *more* tokens for shallow depths than deep ones gives ρ < 0.

**Why it matters:** If indentation depth maps monotonically to whitespace token count, the model receives a natural positional signal for nesting structure without needing to learn it from context.

### Multilingual Fairness
- **Tokenizer Gini Coefficient** (`tokenizer_fairness_gini`): Measures equitable treatment across languages, defined as:

* $`L = \{1, \dots, n\}`$ be the set of languages, each weighted equally.
* For every language $`\ell \in L`$, define the **token cost**
```math
  c_\ell \;=\;
  \frac{\text{number of tokens produced by the tokenizer on language }\ell}
       {\text{number of raw bytes (or lines for parallel ds) in the same text}}
```
  (lower $`c_\ell`$ means cheaper encoding, higher means more byte-hungry).

* Let the mean cost be
```math
  \mu \;=\; \frac{1}{n}\;\sum_{\ell=1}^{n} c_\ell.
```

Then the **Tokenizer Fairness Gini** with equal weights is

```math
\mathrm{TFG}
=\frac{\displaystyle\sum_{i=1}^{n}\sum_{j=1}^{n} \lvert c_i - c_j \rvert}
        {2\,n^2\,\mu}
```
* **Range:** $`0 \le \mathrm{TFG} \le 1`$
  * $`0`$: perfect parity (every language has identical byte-normalised token cost).
  * $`1`$: maximal unfairness.

- **Cross-Lingual Vocabulary-Utilization CoV** (`vocab_util_cross_lingual_cov`): Coefficient of variation (sample standard deviation ÷ mean, `ddof=1`) of the per-language vocabulary-utilization *ratio* across languages. Computed on the ratio (not the raw used-token count) so it is comparable across tokenizers with different vocabulary sizes. **Lower is better** — a low value means the tokenizer devotes a similarly-sized share of its vocabulary to each language (balanced cross-lingual utilization); a high value means utilization is concentrated in some languages. Complements the Gini coefficient above: Gini measures fairness of per-language *encoding cost*, while this measures balance of per-language *vocabulary coverage*. Requires ≥2 languages with mean utilization > 0; for single-language corpora it is reported as `---` (markdown) / omitted (plot) rather than a fabricated `0`. The JSON results also carry the underlying `per_language_mean` and `per_language_std`. Surfaced in the markdown results table and as an individual comparison plot (`vocab_util_cross_lingual_cov_individual.svg`).

## Data Format Requirements

The framework supports three input text formats:

- **Plain text** (`.txt`): One sentence per line recommended for parallel corpora
- **JSON**: Object with a `"texts"` array of strings
- **Parquet**: DataFrame with a `"text"` column

## Module Structure

```
tokenizer_analysis/
├── __init__.py                    # Main package exports
├── main.py                        # UnifiedTokenizerAnalyzer orchestration class
├── constants.py                   # Package-level constants
├── config/                        # Configuration modules
│   ├── language_metadata.py      # LanguageMetadata for grouping analysis
│   └── text_measurement.py       # Text measurement configuration
├── core/                          # Core data structures and providers
│   ├── input_providers.py        # InputProvider implementations
│   ├── input_types.py            # TokenizedData and core types
│   ├── input_utils.py            # Input loading and validation utilities
│   ├── tokenizer_wrapper.py      # Generic wrapper for tokenizer objects
│   └── validation.py             # Data validation functions
├── metrics/                       # Metrics computation modules
│   ├── base.py                   # BaseMetrics with common utilities
│   ├── basic.py                  # Basic tokenization metrics
│   ├── information_theoretic.py  # Information-theoretic metrics
│   ├── math.py                   # Mathematical content metrics (digit boundaries, operators)
│   ├── code_ast.py               # Code tokenization metrics (AST alignment, indentation)
│   ├── utf8_integrity.py         # UTF-8 character boundary metrics
│   ├── morphscore.py             # MorphScore morphological evaluation
│   └── gini.py                   # Multilingual fairness metrics
├── diagnostics/                   # Single-tokenizer sanity checks
│   ├── sanity_check.py           # Health checks (byte coverage, determinism, ...)
│   └── probe_corpus.py           # Probe corpus for reachability checks
├── cli/                           # Console-script entry points
│   ├── run_analysis.py           # tokenizer-analysis
│   ├── visualize_tokenization.py # tokenizer-visualize
│   └── sanity_check.py           # tokenizer-sanity-check
├── per_example.py                 # Per-document metric outputs
├── loaders/                       # Data loading modules
│   ├── constants.py              # Language code mappings (ISO639-1 to FLORES)
│   ├── code_data.py              # Code snippet loader for AST metrics
│   └── multilingual_data.py      # Multilingual text dataset loader
├── utils/                         # Utility functions
│   ├── text_utils.py             # Text processing utilities
│   └── tokenizer_utils.py        # Tokenizer loading utilities
└── visualization/                 # Plotting and visualization
    ├── plotter.py                # TokenizerVisualizer main class
    ├── plots.py                  # Core plotting functions
    ├── data_extraction.py        # Data extraction for plotting
    ├── latex_tables.py           # LaTeX table generation
    ├── markdown_tables.py        # Markdown table generation
    └── visualization_config.py   # Visualization configuration

scripts/
├── run_tokenizer_analysis.py     # Legacy CLI wrapper (use `uv run tokenizer-analysis` instead)
└── visualize_tokenization.py     # Token boundary visualization
```

## Performance

### Encoding (the main bottleneck)

Encoding is **single-threaded**: every combination of tokenizer, language, and sample is processed sequentially, so total encode calls scale as **O(N × L × M)** (tokenizers × languages × samples). With 10+ tokenizers, 13 languages, and 1000 samples per language, encoding alone takes roughly 80–165 s depending on tokenizer backend.

Knobs to reduce encoding time:

| Knob | Effect |
|------|--------|
| `--samples-per-lang N` | Fewer samples per language (default 2000) |
| `--save-tokenized-data` | Cache encoded data as a pickle file for reuse |
| `--tokenized-data-file PATH` | Load previously cached data instead of re-encoding |

### Reconstruction fidelity

Reconstruction metrics (`mean_cer`, `whitespace_fidelity`) decode every tokenized text back to a string and compare it to the original. Some `transformers` and tiktoken-backed tokenizers add significant per-call Python overhead, so this can dominate runtime on large runs. Pass `--no-reconstruction` to skip.

### Skipping expensive metrics

| Flag | What it skips |
|------|---------------|
| `--no-reconstruction` | Decode round-trip, CER, whitespace fidelity |
| `--no-digit-boundary` | Digit boundary alignment, digit split variability, numeric magnitude consistency, operator isolation |
| `--no-code-ast` | AST boundary alignment analysis (also skips synthetic code generation) |
| `--no-utf8-integrity` | UTF-8 character boundary integrity analysis |
| `--no-plots` | All matplotlib rendering |

### Pre-tokenized data cache

A two-step workflow lets you encode once and iterate on metrics/visualization without re-encoding:

```bash
# Step 1 — encode and save (slow, once)
uv run tokenizer-analysis \
  --tokenizer-config tokenizers.json --language-config languages.json \
  --save-tokenized-data --tokenized-data-output-path results/tokenized_data.pkl

# Step 2 — reuse cached data (fast, repeat as needed)
uv run tokenizer-analysis \
  --tokenized-data-file results/tokenized_data.pkl \
  --language-config languages.json
```

> **Note:** Code/math metrics that require raw `encode()` calls (AST boundary, MorphScore) are unavailable in pre-tokenized mode.

### Quick-iteration recipe

For fast development iterations (~10–20 s), minimize samples and disable expensive extras:

```bash
uv run tokenizer-analysis \
  --tokenizer-config tokenizers.json --language-config languages.json \
  --samples-per-lang 100 \
  --no-reconstruction --no-plots --no-code-ast --no-utf8-integrity --no-digit-boundary
```

## Troubleshooting

**`No module named 'morphscore'`** — Initialize submodules, then install MorphScore into the project environment: `git submodule update --init --recursive && uv pip install -e ./morphscore`

**`Unknown tokenizer class`** — Available classes: `"huggingface"`, `"custom_bpe"`, `"pretokenized"`, plus any custom classes you register at runtime with `register_tokenizer_class()` (see Contributing).

**`FileNotFoundError`** — Check that paths in config files are absolute or relative to the working directory.

**`_tkinter.TclError: no display name`** — Set `export MPLBACKEND=Agg` before running on headless servers.

## Contributing

### Adding New Tokenizers

Subclass `TokenizerWrapper` from `tokenizer_analysis.core.tokenizer_wrapper` and implement the required abstract methods. Then register it so the config system can instantiate it by name.

#### Required methods (abstract)

| Method | Purpose |
|--------|---------|
| `get_name() -> str` | Return the tokenizer's display name. |
| `get_vocab_size() -> int` | Return the total vocabulary size. |
| `get_vocab() -> Dict[str, int]` | Return `{token_string: id}` mapping. Used for vocabulary utilization metrics and as a fallback for `convert_ids_to_tokens`. Return `None` if unavailable (disables vocab-dependent metrics). |
| `can_encode() -> bool` | Return `True` if `encode()` works. Return `False` for pre-tokenized-only wrappers — this skips all encoding-dependent metrics (AST, math, UTF-8, indentation). |
| `encode(text: str) -> List[int]` | Encode text to token IDs. Only called when `can_encode()` is `True`. |
| `can_pretokenize() -> bool` | Whether `pretokenize()` is available. Return `False` if not applicable. |
| `pretokenize(text: str) -> List[str]` | Split text into subword pieces (strings). Only called when `can_pretokenize()` is `True`. |
| `from_config(cls, name, config) -> TokenizerWrapper` | Class method factory. Receives the tokenizer name and the config dict from the JSON file. |

#### Optional overrides

These have working defaults but can be overridden for better results:

| Method | Default behaviour | Why override |
|--------|------------------|--------------|
| `convert_ids_to_tokens(ids) -> List[str]` | Reverses `get_vocab()`. | Faster or more accurate when the underlying library has a direct lookup (e.g., `id_to_token`). |
| `encode_with_offsets(text) -> (List[int], Optional[List[Tuple[int,int]]])` | Returns `(self.encode(text), None)`. | Provide `(start_char, end_char)` offsets per token for exact source-to-token mapping. Without this, code metrics fall back to greedy character alignment, which can fail for tokenizers that strip whitespace from tokens (e.g., custom BPE with a `Whitespace` pre-tokenizer). HuggingFace `tokenizers` and SentencePiece both expose offsets natively. |
| `get_underlying_tokenizer()` | Returns `None`. | Expose the raw HuggingFace tokenizer object (if one exists) for specialized consumers like MorphScore (only compatible with HF tokenizers). |
| `get_unk_token_id() -> Optional[int]` | Returns `None`. | Enables UNK-related analysis. |

#### Minimal example

```python
from tokenizer_analysis.core.tokenizer_wrapper import TokenizerWrapper, register_tokenizer_class

class MyTokenizer(TokenizerWrapper):
    def __init__(self, name, tok):
        self._name, self._tok = name, tok

    def get_name(self): return self._name
    def get_vocab_size(self): return self._tok.vocab_size
    def get_vocab(self): return self._tok.get_vocab()
    def can_encode(self): return True
    def encode(self, text): return self._tok.encode(text)
    def can_pretokenize(self): return False
    def pretokenize(self, text): raise NotImplementedError

    @classmethod
    def from_config(cls, name, config):
        tok = load_my_tokenizer(config['path'])  # your loading logic
        return cls(name, tok)

register_tokenizer_class('my_class', MyTokenizer)
```

Then reference `"class": "my_class"` in your tokenizer config.

### Adding New Metrics
1. Inherit from `BaseMetrics` in `tokenizer_analysis/metrics/base.py`
2. Implement `compute()` method
3. Register in `main.py`

### Submitting Changes
1. Fork the repository
2. Create a feature branch
3. Ensure all existing tests pass
4. Submit a pull request

## Citation

```bibtex
@software{meister_tokenizer_analysis_2025,
  title = {TokEval: A Tokenizer Analysis Suite},
  author = {Meister, Clara},
  year = {2025},
  url = {https://github.com/cimeister/tokenizer-intrinsic-evals}
}
```
