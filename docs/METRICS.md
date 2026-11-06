# Metrics

Definitions for every metric TokEval computes, and the results key each is
written under. For which flag supplies the data a metric needs, see
[Metric families and the data they need](CONFIGURATION.md#metric-families-and-the-data-they-need).
For what the surrounding JSON looks like, see [OUTPUT.md](OUTPUT.md).

Section headings below give the top-level results key in backticks. Where a
value is deeper in the file, the path is written out.

## Numbers from tokenizer-analysis-suite are not comparable

Seven metrics changed what they compute between `tokenizer-analysis-suite` and
this package, not just where they are written. Each affected section below
includes a note. The full table with measured effects is in
[CHANGELOG.md](../CHANGELOG.md).

Separately, a field that reads `null` here and `0.0` in an older run reflects a
convention change rather than a different measurement: a value that could not
be computed is now `null`, because `0.0` is a legal value for most of these
metrics and a zero was indistinguishable from a measurement.

Some caveats below quote figures from a 37-tokenizer development run. That run
is not in this repository, so those specific numbers cannot be regenerated from
anything here: the largest run committed is the nine tokenizers of
`benchmarks/open_source`. They are quoted because the effect they describe is
the reason for the caveat, and the direction and rough size are what a reader
needs. Any figure attributed to `benchmarks/open_source` is reproducible with
`bash benchmarks/open_source/run.sh`.

## Metric names and results keys

Every metric, the top-level key it is written under in `analysis_results.json`,
and the path to its headline value. `<tok>` is a tokenizer name from your
config; `<lang>` is a language code.

The identifiers in this table are the results keys. The LaTeX table generator
uses its own row ids, and eight of them appear in no results file; see
[VISUALIZATION.md](VISUALIZATION.md#row-ids).

| Metric | Top-level key | Path to the headline value | Direction |
|---|---|---|---|
| Compression rate | `compression_rate` | `.per_tokenizer.<tok>.global.compression_rate` | higher |
| Tokens per line | (under `compression_rate`) | `.per_tokenizer.<tok>.tokens_per_line.global_avg` | lower |
| Fertility | `fertility` | `.per_tokenizer.<tok>.global.mean` | lower |
| Token length | `token_length` | `.per_tokenizer.<tok>.global.mean` | neither |
| Vocabulary utilization | `vocabulary_utilization` | `.per_tokenizer.<tok>.global.utilization` | higher |
| Type-token ratio | (under `vocabulary_utilization`) | `.per_tokenizer.<tok>.type_token_ratio.global_ttr` | neither |
| Cross-lingual vocabulary-utilization CoV | (under `vocabulary_utilization`) | `.per_tokenizer.<tok>.per_language_cov` | lower |
| Rényi efficiency | `renyi_efficiency` | `.per_tokenizer.<tok>.global.renyi_<alpha>`, one field per alpha in 1.0, 2.0, 2.5, 3.0 | higher |
| Average token rank | (under `renyi_efficiency`) | `.per_tokenizer.<tok>.unigram_distribution.global_avg_token_rank` | neither |
| Bigram entropy | `bigram_entropy` | `.per_tokenizer.<tok>.global.bigram_entropy` | higher |
| Trigram entropy | `trigram_entropy` | `.per_tokenizer.<tok>.global.trigram_entropy` | higher |
| MorphScore | `morphscore` | `.per_tokenizer.<tok>.global.avg_morphscore_recall` and `.avg_morphscore_precision` | higher |
| Three-digit boundary alignment | `three_digit_boundary_alignment` | `.per_tokenizer.<tok>.global.mean_f1` | higher |
| Digit split variability | (under `three_digit_boundary_alignment`) | `.per_tokenizer.<tok>.split_variability.by_digit_length.<n>.<lang>.entropy` | lower |
| Numeric magnitude consistency | `numeric_magnitude_consistency` | `.per_tokenizer.<tok>.global.mean_fertility` | lower |
| Operator isolation rate | `operator_isolation_rate` | `.per_tokenizer.<tok>.global.overall_isolation_rate` | higher |
| Compound operator preservation | (under `operator_isolation_rate`) | `.per_tokenizer.<tok>.global.overall_compound_preservation_rate` | higher |
| Round-trip exact match rate | `reconstruction_fidelity` | `.per_tokenizer.<tok>.global.exact_match_rate` | higher |
| Character error rate | (under `reconstruction_fidelity`) | `.per_tokenizer.<tok>.global.mean_cer` | lower |
| UNK token rate | (under `reconstruction_fidelity`) | `.per_tokenizer.<tok>.global.unk_token_rate` | lower |
| Whitespace fidelity | (under `reconstruction_fidelity`) | `.per_tokenizer.<tok>.global.whitespace_fidelity` | higher |
| Token UTF-8 completeness rate | `utf8_token_integrity` | `.per_tokenizer.<tok>.global.completeness_rate` | higher |
| Character boundary crossing rate | (under `utf8_token_integrity`) | `.per_tokenizer.<tok>.global.boundary_crossing_rate` | lower |
| Character boundary split rate | (under `utf8_token_integrity`) | `.per_tokenizer.<tok>.char_split.global.split_rate` | lower |
| Merge vocabulary usage | `merge_vocabulary_usage` | `.per_tokenizer.<tok>.global.pct_scaffold` and the other `pct_*` fields | neither |
| AST leaf-node boundary alignment | `ast_boundary_alignment` | `.per_tokenizer.<tok>.global.full_alignment_rate` | higher |
| Identifier fragmentation | `identifier_fragmentation` | `.per_tokenizer.<tok>.global.fragmentation_rate` | lower |
| Indentation depth correlation | `indentation_consistency` | `.per_tokenizer.<tok>.global.depth_proportionality_correlation` | higher |
| Tokenizer fairness Gini | `tokenizer_fairness_gini` | `.per_tokenizer.<tok>.global.gini_coefficient` | lower |
| Tokenizer fairness Gini, per line | (under `tokenizer_fairness_gini`) | `.per_tokenizer.<tok>.per_line_normalization.gini_coefficient` | lower |
| Lorenz curve | (under `tokenizer_fairness_gini`) | `.per_tokenizer.<tok>.lorenz_curve` | n/a, a curve |
| Encoding speed | `encoding_speed` | `.per_tokenizer.<tok>.global.mean_ms` | lower |

"Direction" gives the better direction for the quantity as defined.
"neither" marks a metric that describes a tokenizer without ranking it: a longer
mean token or a higher type-token ratio is not better or worse on its own. Plot
titles carry an arrow for a subset of these metrics and none for the rest; see
[VISUALIZATION.md](VISUALIZATION.md#direction-of-better-arrows).

Two more top-level keys are not metrics. `run_metadata` is described under
[Provenance](OUTPUT.md#provenance). `grouped_analysis` is written when
`--run-grouped-analysis` is passed, and holds the metrics recomputed within each
language group: `grouped_analysis.<group type>.<group name>.<metric>`. Both
output files carry it: `analysis_results.json` holds the slimmed per-group
blocks (the slimming step recurses into `grouped_analysis` with the same
per-metric rules), and `analysis_results_full.json` the full ones.

## Metrics reported under another metric

Six metrics restate a measurement another metric already publishes, four of them
as exact algebraic identities. They are written as a field of the metric that
owns the measurement rather than as separate top-level keys, so the file does not
present one number twice as if it were two pieces of evidence. Each primary
records the merge and its evidence under `metadata.merged_metrics`.

| Reported under | Field | Relationship |
|---|---|---|
| `compression_rate` | `tokens_per_line` | product is exactly 1 |
| `vocabulary_utilization` | `type_token_ratio` | TTR is utilization rescaled by vocab size over token count |
| `renyi_efficiency` | `unigram_distribution` | unigram entropy is the unnormalized numerator of `renyi_1.0` |
| `utf8_token_integrity` | `char_split` | the same events counted from the character side |
| `tokenizer_fairness_gini` | `lorenz_curve` | `1 - 2*area(lorenz)` is the Gini coefficient |
| `three_digit_boundary_alignment` | `split_variability` | Spearman -0.992 between pooled `entropy_short` and `avg_f1` |

## Basic tokenization metrics

- **Compression rate** (`compression_rate`): total text units (bytes, characters
  or lines) divided by total tokens across the corpus.
- **Fertility** (`fertility`): tokens per word or per character, as a mean of
  per-document ratios (`aggregation: mean_of_ratios`), so a long document counts
  the same as a short one. A document the tokenizer encodes to zero tokens is
  excluded and counted separately in `zero_token_documents`, published per
  tokenizer and per language: its ratio is a defined 0 that would drag the mean
  down with nothing in the output saying a document had vanished. `count` is
  therefore the documents that produced at least one token, and re-deriving the
  pooled weighting needs `count + zero_token_documents`. `token_length`,
  `avg_tokens_per_line` and `compression_rate` exclude the same documents and
  publish no such count.
- **Token length** (`token_length`): mean token size in bytes and in characters.
  This is a mean of per-document ratios, where `compression_rate` is a ratio of
  totals, so the two differ on short documents.
- **Type-token ratio**: unique tokens divided by total tokens. Written under
  `vocabulary_utilization.per_tokenizer.<tok>.type_token_ratio`, because it is
  that metric rescaled by vocabulary size over token count (see
  [Metrics reported under another metric](#metrics-reported-under-another-metric)).
- **Vocabulary utilization** (`vocabulary_utilization`): the fraction of the
  declared vocabulary that the corpus used.

**Compression rate and measurement method.** Under the `lines` measurement
method, `compression_rate` and `avg_tokens_per_line` (reported under
`compression_rate.per_tokenizer.<tok>.tokens_per_line`) are exact reciprocals:
their product is 1.000000 for every tokenizer. Under every measurement method the
two correlate at Spearman -1.000, because a higher token count always lowers
`compression_rate` and raises `avg_tokens_per_line`, whichever unit
`compression_rate` divides by.

Within a single run, `compression_rate` computed under the `characters`,
`bytes` and `lines` methods correlates at exactly 1.0000 between every pair.
The numerator (the character, byte or line count of the corpus) is fixed once
the corpus is fixed and does not depend on the tokenizer; only the token count
in the denominator varies by tokenizer. Changing `--measurement-config`
therefore rescales `compression_rate` but does not change the ranking of
tokenizers within a run.

**Fertility and whitespace word counting.** The default word counting splits
text on whitespace. On FLORES+, 96.2% of Japanese lines and 69.3% of Chinese
lines hold exactly one whitespace-delimited token, so for those languages
`fertility` measures tokens per sentence rather than tokens per word. Median
per-language fertility runs from 1.261 (eng) to 38.195 (jpn), and the pooled
value across the FLORES+ languages correlates with fertility over the other 11
languages at Spearman 0.401. A pooled `fertility` computed over a corpus
containing CJK text reflects the language mix as much as the tokenizer.

**Vocabulary utilization and unreachable tokens.** The denominator is the
declared vocabulary size, which includes special and added tokens that
encoding with `add_special_tokens=False` can never emit. The unreachable count
is tokenizer-dependent: 1000 of 131072 vocabulary entries for one tokenizer in
a 37-tokenizer run, 4 for most of the others. For a tokenizer with a large
unreachable count, the maximum `vocabulary_utilization` can reach is capped
below 1.0 by that margin.

### Encoding speed (`encoding_speed`)

Wall-clock time spent encoding, per tokenizer:
`encoding_speed.per_tokenizer.<tok>` holds `mean_ms` (mean milliseconds per
sample), `total_s` (total seconds) and `num_samples`, plus a `global` block that
repeats the same three fields. It measures the run, not the tokenizer's quality,
so it has no per-language breakdown. Use it to size a larger run from a small
one.

## Information-theoretic metrics

### Rényi efficiency (`renyi_efficiency`)

`H_alpha / log2(|V|)` following
[Zouhar et al. 2023](https://aclanthology.org/2023.acl-long.284/), with `|V|`
the declared vocabulary size. Computed at alphas 1.0, 2.0, 2.5 and 3.0. The
pre-1.0 normalization, which divided by the number of token types observed in
the corpus, is still written under the top-level
`renyi_efficiency.observed_normalization`; the two rank tokenizers at Spearman
0.678 over 37 tokenizers, so they are not interchangeable.

**Average token rank** is written under
`renyi_efficiency.per_tokenizer.<tok>.unigram_distribution.global_avg_token_rank`.
The token types observed in the corpus are ordered by descending frequency and
numbered from 1; the metric is the mean of those rank numbers over every token
occurrence. The ordering runs over the types this corpus produced, not over the
declared vocabulary.

> **Only compare tokenizers whose vocabularies are the same size.** The rank
> can never exceed the number of distinct types observed, and that count can
> never exceed the declared vocabulary size, so a small-vocabulary tokenizer is
> structurally prevented from reporting a large mean rank however unevenly it
> uses its vocabulary. On the nine tokenizers of `benchmarks/open_source` the
> mean rank runs from 475.9 for `bert-base` (vocabulary 30522, 6369 types
> observed) to 4521.8 for `gemma-2` (vocabulary 256000, 26490 types observed),
> correlating with the observed type count at Spearman 0.983 and with the
> vocabulary size at 0.917. It also correlates with total token count at
> Spearman -0.977 over a 37-tokenizer run, so it is not comparable across runs
> of different corpus size either.
>
> **TODO:** give this metric a scale-free definition. Dividing by the number of
> observed types, or reporting the mean rank quantile instead of the mean rank,
> would both remove the vocabulary-size ceiling. Until then it is a diagnostic
> for one vocabulary size at a time rather than a comparison across
> tokenizers.

The same block holds `global_unigram_entropy`. `renyi_1.0` multiplied by
`log2` of the declared vocabulary size reproduces `global_unigram_entropy` to
zero relative error: `global_unigram_entropy` is the unnormalized numerator of
`renyi_1.0`, not merely correlated with it. Checked on the nine tokenizers of
`benchmarks/open_source/analysis_results.json`, where the largest relative
error is 1.6e-16.

### Bigram entropy (`bigram_entropy`)

For each token type, the metric takes the distribution of tokens that follow it
in the corpus and measures how even that distribution is. A score of 1.0 means
every token type's successors are uniformly distributed; a score near 0 means
most token types are almost always followed by the same token. Token types
occurring fewer than 3 times (`min_bigram_occurrences`, configurable) are
excluded, because the estimate is noisy below that. Bigrams do not cross
document boundaries.

Based on the Shannon efficiency metric (η) of
[Poelman et al. 2025](https://aclanthology.org/2025.emnlp-main.369/), EMNLP.
This implementation deviates from the reference in four ways, listed in
`bigram_entropy.metadata.deviations_from_reference`. The most consequential is
the normalizer: this metric divides by the context's own successor count, the
reference divides by the corpus-wide accessor-domain size. The reference
normalizer and the reference unweighted aggregation are computed alongside and
written under `bigram_entropy.reference_definition`, both for the corpus at
`.per_tokenizer.<tok>.bigram_entropy` and per language at
`.per_tokenizer.<tok>.per_language.<lang>.bigram_entropy`.

A per-language entry treats that language as its own corpus: it normalizes by
the language's own accessor domain, published beside it as
`accessor_domain_size`. Over six FLORES languages with gpt2 that divisor ranges
from 104 to 1894, so two per-language values are not on a common scale. Compare
one language across tokenizers, not two languages under one tokenizer.

### Trigram entropy (`trigram_entropy`)

The same construction with a two-token context: for each bigram, the evenness of
the distribution of tokens that follow it. Poelman et al. define only the bigram
form, so there is no published value to compare this against. Token contexts
occurring fewer than 3 times (`min_trigram_occurrences`) are excluded.

The value is in `trigram_entropy.per_tokenizer.<tok>.global`, which holds
`trigram_entropy`, `total_trigrams`, `types_evaluated` and `types_excluded`.

The default threshold discards 89 to 90 percent of context types and a median 70
percent of occurrences, and the ranking moves with it: Spearman 0.728 between
`min_trigram_occurrences` 3 and 25 over 37 tokenizers, against 0.985 for bigram
entropy at the same settings. Treat a trigram ranking as threshold-dependent.

## Morphological metrics

### MorphScore (`morphscore`)

Whether a tokenizer's boundaries fall where a word's morpheme boundaries do,
following [Arnett et al. 2025](https://arxiv.org/abs/2507.06378). Enable with
`--morphscore` or `--morphscore-config`. Requires raw tokenization and the
MorphScore submodule.

`morphscore.per_tokenizer.<tok>.global` holds four averages over the languages
that had data, with `languages_evaluated` and `total_samples` beside them:

| Field | What it is |
|---|---|
| `avg_morphscore_recall` | of the morpheme boundaries a word has, the share the tokenizer also placed a token boundary at |
| `avg_morphscore_precision` | of the token boundaries the tokenizer placed inside a word, the share that fall on a morpheme boundary |
| `avg_micro_f1` | F1 pooled over samples, so a language with more words counts for more |
| `avg_macro_f1` | F1 averaged over languages, so every language counts the same |

All four are `null` when no language produced a usable result, with
`languages_evaluated` at 0 rather than four zeros that read as scores.

Three options change what is measured, and each is recorded in the metric's
`metadata`: `by_split` evaluates the train, validation and test splits
separately rather than together; `freq_scale` weights a word by its frequency,
and is on by default; `exclude_single_tok` drops words the tokenizer emitted as
a single token, which have no internal boundary to score.

The scoring itself is the submodule's, not this package's. It has not been
audited against the paper here, so treat these numbers as MorphScore's output
rather than as an independent reimplementation.

## Mathematical content metrics

Tokenizer handling of mathematical expressions, following Singh and Strouse
(2024, [arXiv:2402.14903](https://arxiv.org/abs/2402.14903)), who measured a
gain of more than 22 percentage points in arithmetic accuracy from right-to-left
tokenization of numbers. Disable with `--no-digit-boundary`.

> **Data scope:** the four metrics in this group do not all take their data from
> the same place, and the fallback applies to three of them only.
>
> The three digit metrics (`three_digit_boundary_alignment`, digit split
> variability, `numeric_magnitude_consistency`) use the dedicated math texts when
> `--math-data FILE` or `--use-builtin-math-data` is given, and write every
> per-language result in this group under the synthetic language key `math`.
> Without either flag they fall back to whatever numbers appear in the main
> corpus, and the run prints a warning naming all three. On the bundled
> five-language FLORES+ corpus at the default `--samples-per-lang`, the fallback
> gives 1797 digit spans: 385 of one digit, 600 of two, 355 of three, 451 of
> four, 4 of five and 2 of six. The place-value boundaries at positions L-6 and
> L-9 are therefore almost never exercised.
>
> `operator_isolation_rate` does not fall back. Its `math` domain always reads
> dedicated math texts, the bundled
> `tokenizer_analysis/sample_data/math_samples.json` when no flag names another
> file, and its `code` domain always reads code snippets, the bundled
> `sample_data/code_samples.json` when no `--code-ast-config` names a corpus.
> Its `prose` domain is the main corpus, and it is off by default. Pass
> `--operator-prose-domain` to score it. Each run logs the domains it will use
> on one line:
> `Operator isolation domains: math=..., code=...`.
>
> `--no-code-ast` drops the three AST metrics without affecting this metric's `code`
> domain running on the bundled samples. `--no-digit-boundary` drops
> `operator_isolation_rate` along with the three digit metrics.
>
> See [the fuller invocation](../README.md#settings-that-change-what-is-measured) for the invocation
> that supplies math data.

### Three-digit place-value boundary alignment (`three_digit_boundary_alignment`)

> **Not comparable with `tokenizer-analysis-suite`.** Source spans now map to
> tokens through the tokenizer's own character offsets. They were previously
> matched against a string rebuilt from token surfaces, which under-counted
> spans for any language with non-ASCII text.

Whether numbers are tokenized with right-aligned 3-digit groupings matching
place-value structure (units, thousands, millions).

For each number of L digits, the ideal internal boundaries are at positions
L-3, L-6, L-9 counting from the left. The metric compares the actual internal
boundaries against those and reports precision, recall and F1. A number of 3
digits or fewer has no ideal boundary: kept as one token it scores F1 = 1.0,
split it scores F1 = 0.

**Example:** `1234567` has 7 digits, so the ideal boundaries are at positions 1
and 4, giving `1|234|567`. A tokenizer producing `1|234|567` places boundaries
at 1 and 4 and scores F1 = 1.0. A tokenizer producing `12|345|67` places two
internal boundaries, at positions 2 and 5, neither of which is ideal, so
precision, recall and F1 are all 0.0. `42` kept as one token scores F1 = 1.0:
no boundary was needed and none was placed. `42` split into `4|2` scores
precision 0.0 and F1 = 0.0, because a boundary was placed where none was needed.

Singh and Strouse showed that right-to-left digit grouping improves arithmetic
accuracy because corresponding digit positions across two operands then occupy
the same token positions.

### Digit split variability

Written under
`three_digit_boundary_alignment.per_tokenizer.<tok>.split_variability`. For
numbers of the same digit length, the Shannon entropy of the distribution of
boundary patterns. Low entropy means one splitting scheme is used consistently;
high entropy means the scheme varies from number to number. Reports entropy in
bits, the dominant pattern and its frequency, keyed by digit-length bucket and
then by language (`by_digit_length.<n>.<lang>`) and by short/long bucket and
then by language (`by_bucket.<short|long>.<lang>`).

Every published entropy is per language. There is no cross-language pooled
entropy in either results file: the code computes one, but the merge into
`three_digit_boundary_alignment` keeps only the two per-language blocks, so the
pooled value is never written.

**Example:** a corpus holds three 5-digit numbers. If all three are split
`XX|XXX`, that bucket has one pattern and entropy 0.0 bits. If they are split
`XX|XXX`, `X|XXXX` and `XXX|XX`, that bucket has three patterns at equal
frequency and entropy log2(3) = 1.585 bits. The first case is a single
scheme, correct or not. In the second, numbers of the same length receive
different segmentations.

A tokenizer with moderate F1 and low entropy applies one scheme that does not
match place value, which retraining can change. Moderate F1 with high entropy
means no single scheme is being applied.

### Numeric magnitude consistency (`numeric_magnitude_consistency`)

How tokens-per-digit varies across digit lengths. Everything is under
`numeric_magnitude_consistency.per_tokenizer.<tok>.scaling`:

- `per_bucket.<digits>` holds `mean_fertility` (mean tokens per digit),
  `std_fertility`, `count` and `mean_digit_length` for that bucket.
- `spearman_rho` and `spearman_p`: rank correlation between bucket digit length
  and that bucket's **mean fertility per digit**.
- `cv_of_mean_fertility`: coefficient of variation of the same per-bucket
  **mean fertility per digit** values.
- `linear_fit`: `slope`, `intercept` and `r_squared` of a least-squares fit of
  each bucket's **mean token count** on that bucket's mean digit length. The fit
  is the only field computed on token counts rather than on fertility per digit.

The fit rests on at most 10 points (buckets `1` through `9` plus `10+`), so
`slope` and `r_squared` are coarse.

**Example:** a tokenizer has `0` to `999` as single vocabulary entries, so
1-digit numbers cost 1 token (1.0 tokens per digit), 2-digit numbers 1 token
(0.5), 3-digit numbers 1 token (0.333). At 4 digits it splits: `1234` becomes
`12|34`, 2 tokens (0.5). At 7 digits: `1234567` becomes `123|45|67`, 3 tokens
(0.429). Fertility per digit therefore rises from 0.333 to 0.5 between 3 and 4
digits after falling over 1 to 3 digits, so `spearman_rho` is -0.616 and
`cv_of_mean_fertility` is 0.420. The linear fit is computed on token counts
(1, 1, 1, 2, 3), which do rise with digit length, so it returns slope 0.368 and
R-squared 0.897. Read the two together: a high `r_squared` does not rule out the
discontinuity in tokens per digit that `cv_of_mean_fertility` reports.

Tokenizers trained on natural language often have dense vocabulary coverage for
small numbers and split larger numbers in ways that vary with the number, which
is what the per-bucket fertility values show.

### Operator isolation rate (`operator_isolation_rate`)

> **Not comparable with `tokenizer-analysis-suite`.** Two changes: spans map
> through encoder offsets rather than a rebuilt string, and a character
> claimed by two token ranges is assigned to the later token. The second
> alone moved XLM-RoBERTa's pooled isolation rate from 0.6948 to 0.6770 and
> its compound preservation from 0.7222 to 0.8443, over 454693 operators.

The fraction of mathematical operators tokenized as standalone tokens rather
than merged with adjacent content. Operators are located in the source text by
`_OPERATOR_SPAN` (`tokenizer_analysis/metrics/math.py`), which matches the
two-character forms `**`, `//`, `<<`, `>>`, `<=`, `>=`, `=>`, `==`, `!=`, `&&`,
`||`, `?:` before the single characters `+ - * / = < > ! & | ^ ~ %`. The
hyphen-minus `-` is always counted as an operator, including as a unary negative
sign such as `-42`, because telling the two apart requires parsing the
expression. A compound preservation sub-metric records whether a multi-character
operator (`**`, `<=`, `!=`) stayed in one token.

An operator counts as isolated when the tokens covering it hold no
non-whitespace character from outside the operator span. Leading or trailing
whitespace inside the same token does not break isolation, because a space is
not an operand.

**Example:** in `3+5>=8` the regex matches two operators, `+` and `>=`, so the
denominator is 2. A tokenizer producing `3` | `+` | `5` | `>=` | `8` isolates
both, isolation rate 2/2 = 1.0, and keeps `>=` in one token, compound
preservation 1/1 = 1.0. A tokenizer producing `3+` | `5` | `>` | `=` | `8`
merges `+` with the operand `3`, so `+` is not isolated, while `>=` is covered
only by tokens `>` and `=`, which hold nothing else, so `>=` is isolated:
isolation rate 1/2 = 0.5. Compound preservation is 0/1 = 0.0, because `>=` took
two tokens.

Merging an operator with its operand puts the operation and the value in one
embedding.

**Which domains are counted.** Code and math always run. Prose, meaning the
main corpus, runs only under `--operator-prose-domain`.

Prose is off by default because an operator is a code construct. The pattern
matches `-`, `/` and `!`, which are ordinary punctuation in prose, so a prose
corpus contributes occurrences that are hyphens and slashes rather than
operators. On the nine-tokenizer benchmark prose supplied 568 of 455558
occurrences, 0.12 percent, so it moved the pooled figure by almost nothing while
being the domain that cost the most to score: 78.5 percent of the documents of a
web corpus contain at least one character the pattern matches.

**How the global is computed:** `global` pools the domains that ran, weighted by
operator instances. On the bundled demo,

```bash
uv run tokenizer-analysis --use-sample-data
```

`bpe` scores 0.7250 pooled over 2229 instances, against 0.6832 for code over
1932 and 0.9966 for math over 297. Adding `--operator-prose-domain` gives 0.7938
over 3016, with prose at 0.9886 over 787.

The domain rates are far apart and the pooled figure is not close to any of
them, so quoting `global` alone reports a number that describes no domain. The
weights also move with the flags: `--code-ast-config` changes the code domain,
`--math-data` changes the math domain, and with prose on, `--samples-per-lang`
changes the prose domain.

`by_domain` holds each rate and each instance count separately, and is written
beside `global` for that reason. Read it before quoting the pooled number.

## Reconstruction fidelity metrics (`reconstruction_fidelity`)

How lossy the encode-then-decode round trip is. Information is lost through
normalization, UNK substitution, whitespace handling and decode asymmetry. These
metrics run on language text, code and math data. They require that the
tokenizer supports decoding; for one that does not, the run logs
`Reconstruction fidelity: skipping <name> (no decode support)` and the tokenizer
is absent from this metric's results.

All four values below are fields of
`reconstruction_fidelity.per_tokenizer.<tok>.global`, with the same fields
repeated per corpus under `per_domain` (this metric uses `per_domain`, not
`per_language`).

### Round-trip exact match rate (`exact_match_rate`)

The fraction of texts for which `decode(encode(text)) == text`. 1.0 means the
tokenizer is lossless on the evaluated data.

**Example:** `"Hello, world!"` encodes to `[15496, 11, 995, 0]` and decodes back
to `"Hello, world!"`, an exact match. `"café"` that decodes to `"cafe"`, the
accent removed by normalization, is not an exact match.

### Character error rate (`mean_cer`)

Levenshtein edit distance between the original text and the decoded text,
divided by the length of the original: the fraction of single-character
insertions, deletions and substitutions needed to turn the decoded text back
into the original.

CER = 0 means a perfect round trip. CER can exceed 1.0 when the decoded text is
much longer than the original, for example when a tokenizer expands
byte-fallback tokens into multi-character escape sequences.

**Example:** original `"hello"` decoded as `"helo"` is edit distance 1 over 5
characters, CER 0.2. Original `"a"` decoded as `"abcd"` is edit distance 3 over
1 character, CER 3.0.

CER is the most expensive field in this group, and `--cer-time-budget` caps it
per tokenizer. When the budget is exceeded, `mean_cer`, `whitespace_fidelity`
and the three structural sub-rates are all `null` for that tokenizer and the run
logs the projection that triggered the skip. `--cer-time-budget 0` disables the
cap.

**The budget covers more work than it used to.** The whitespace alignment runs
inside the same timed region as the edit distance and costs the same order, so
the projection roughly doubled at an unchanged budget value. A run that
previously completed CER on a lossy tokenizer may now skip it, and the skip
nulls all five fields. Raise `--cer-time-budget`, or pass 0 to remove the cap,
to get the previous coverage.

Which tokenizers report `mean_cer` therefore depends on how fast the machine
running the analysis is: a tokenizer skipped at a given `--cer-time-budget` on a
slow machine can complete at the same budget on a faster one.
`per_tokenizer.<tok>.cer_skipped` is `true` when the budget was exceeded for
that tokenizer, distinguishing a skipped value from a measured one.
`exact_match_rate` does not run the Levenshtein computation and is unaffected.

### UNK token rate (`unk_token_rate`)

The fraction of encoded tokens equal to the tokenizer's UNK token id: how much
of the input the tokenizer has no representation for. 0.0 means no unknown
tokens were produced. A tokenizer that declares no UNK token id, or a domain
that scored no tokens, has no denominator and publishes `null`.

**Example:** encoding `"𝕳𝖊𝖑𝖑𝖔"` to `[UNK, UNK, UNK, UNK, UNK]` gives UNK rate
1.0. Encoding `"Hello"` to `[15496]` gives 0.0.

### Whitespace fidelity (`whitespace_fidelity`)

> **Not comparable with `tokenizer-analysis-suite`.** Whitespace was widened
> from ASCII space, tab, newline and carriage return to those plus every
> Unicode `Zs` separator, so any corpus containing NBSP, thin space or
> ideographic space scores differently.

The fraction of whitespace characters (spaces, tabs, newlines, plus the Unicode
Zs category) in the original text preserved through the round trip. A character
counts as preserved when an alignment of the two texts matches it: specifically
a maximum-length common subsequence of the two, choosing among those of maximum
length the one that matches the most whitespace. A text holding no whitespace
has no denominator and publishes `null`, not 1.0.

Matching against the full text rather than against the whitespace alone is what
makes this a fidelity measure. `"hello world"` decoded as `"helloworld "` scores
0 of 1: the word boundary was deleted and a space appeared somewhere else.

**Example:** original `"a b\tc"` decoded as `"a b c"`, the tab replaced by a
space, preserves 1 of 2 whitespace characters, fidelity 0.5.

### Structural whitespace (`indentation_fidelity`, `newline_fidelity`, `tab_fidelity`)

`whitespace_fidelity` weights every whitespace character equally, which is not
what a reader vetting damage to code or tabular preprocessing needs. Indentation
is 42% of the whitespace in the bundled code corpus and 0% of FLORES prose,
where 95% is inter-word spaces that carry no structure. Destroying every indent
in the code corpus moves the roll-up only to 0.578, while collapsing inner spaces
harmlessly leaves it at 0.980; a Makefile tab replaced by four spaces scores 0.80
where trailing whitespace stripped scores 0.50.

Three sub-rates are published beside it, each `null` when its denominator is
zero:

- `indentation_fidelity`: of the original's non-blank lines that begin with
  whitespace, the fraction whose leading whitespace survives **exactly**. Run
  exact, because indentation depth is a property of the whole run: four spaces
  arriving as three is broken code, not 0.75 preserved. Indents are matched as a
  sequence, so a dropped line does not shift the rest.
- `newline_fidelity` and `tab_fidelity`: the newline and tab characters matched
  by the same alignment `whitespace_fidelity` uses, so their sum never exceeds
  it. How a tie is split between them is not defined: several alignments can
  match the same number of whitespace characters while attributing them to
  different kinds, and the implementation reports one of them. `"\n\t"`
  decoded as `"\t\n"` reports one tab preserved, and the same damage written
  `"\t\n"` to `"\n\t"` reports one newline. Read the pair as "one of these two
  survived", not as which one.

**Coverage limit.** No bundled corpus contains tabular data, and 470 of the
bundled code corpus's 471 tabs are indentation. A passing `tab_fidelity` is
evidence about code indentation, not about TSV or fixed-width data.

## UTF-8 character boundary metrics (`utf8_token_integrity`)

> **Not comparable with `tokenizer-analysis-suite`.** Byte-level detection no
> longer requires 50 of the 68 GPT-2 marker characters to be present in the
> vocabulary. A byte-level tokenizer below that threshold was misread as
> non-byte-level and reported a perfect 1.0: gpt4o-english-bpe read 1.0000,
> best of 37 tokenizers, where the true value is 0.6688, worst of 37.

How byte-level tokenizers handle multi-byte UTF-8 characters at token
boundaries. Runs on any text data with no extra configuration. Disable with
`--no-utf8-integrity`.

### Token UTF-8 completeness rate

`utf8_token_integrity.per_tokenizer.<tok>.global.completeness_rate`. The
fraction of content tokens whose bytes form complete UTF-8 characters. A token
such as `<0xC3>`, one byte of the two-byte sequence for `é`, is incomplete: it
holds the start of a character but not all of it. This is the designed behaviour
of byte-fallback tokenization, not an error. The rate measures how often the
vocabulary can represent a whole character rather than a sub-character byte
sequence.

**Example:** `é` (U+00E9) is the bytes `C3 A9`. A tokenizer that produces `caf` |
`é` for `café` emits two tokens, both holding complete UTF-8, completeness rate
1.0. A byte-fallback tokenizer that produces `caf` | `<0xC3>` | `<0xA9>` emits 3
content tokens, 2 of which hold incomplete UTF-8, completeness rate 1/3.

### Character boundary crossing rate

`utf8_token_integrity.per_tokenizer.<tok>.global.boundary_crossing_rate`. The
fraction of content tokens that hold bytes from more than one UTF-8 character
with at least one of those characters incomplete within the token. Such tokens
come from BPE merges that fused bytes across a character boundary.

This is distinct from a plain byte-fallback token. `<0xC3>` is incomplete but
does not cross a boundary: it holds bytes from exactly one character. A token
holding `A9 E4`, the tail byte of `é` merged with the leading byte of a CJK
character, spans two characters and completes neither.

**Example:** the bytes `C3 A9 E4 BD A0` are the characters `é你`. A tokenizer
that merges the last byte of `é` with the first byte of `你` produces `C3` |
`A9 E4` | `BD A0`. The middle token holds the continuation byte of `é` and the
leading byte of `你` and completes neither, so the crossing rate is 1/3.

A byte-fallback token can be recombined with its neighbours to reconstruct the
character. A boundary-crossing token cannot: its bytes belong to two characters,
and one embedding stands for both partial characters at once.

### Character boundary split count

`utf8_token_integrity.per_tokenizer.<tok>.char_split`. How many multi-byte
characters in the source text have their bytes spread across more than one
token. Each token's bytes are reconstructed and aligned to the source text to
determine this. `split_rate` is splits divided by aligned multi-byte characters;
`splits_per_1k_multibyte` and `splits_per_1k_tokens` are also reported, along
with a `per_byte_width` breakdown for 2-, 3- and 4-byte characters.

**Example:** `你好` holds two 3-byte characters (`你` = `E4 BD A0`, `好` =
`E5 A5 BD`). A tokenizer that keeps each character as one token has 0 splits. A
byte-fallback tokenizer that produces `<0xE4>` | `<0xBD>` | `<0xA0>` for `你` has
1 split, that character's bytes spanning 3 tokens. With `好` intact the split
rate is 1/2 = 0.5.

**Alignment reliability.** When a tokenizer does not reproduce the source bytes,
for example an English-trained tokenizer on Cyrillic or CJK where characters are
dropped or replaced by placeholders, alignment cannot map every source byte to a
token. A multi-byte character with any unaligned byte cannot be classified as
split or not split, so it is excluded from both the numerator and the
denominator and counted separately. Three fields report this per language and
globally: `unaligned_multibyte_chars` (the excluded count), `aligned_fraction`
(the share of multi-byte characters that aligned) and `alignment_mismatches`
(the raw count of unaligned source bytes). When no multi-byte character aligns,
`split_rate`, `splits_per_1k_multibyte` and the per-byte-width split rates are
`null` rather than `0.0`, so "no data" is not read as "no splits". A low
`aligned_fraction` means the split rate rests on few characters. When
`alignment_mismatches` is 0 every multi-byte character aligned and this handling
does not affect the split rate.

**Corpus resolution.** The denominator is the number of aligned multi-byte
characters in the evaluated text. On ASCII-dominant languages measured on a
small parallel corpus this is small (the English FLORES+ sample holds 57
multi-byte characters), so `split_rate` takes few distinct values and separates
Latin-script languages poorly. A corpus with more natural
multi-byte content, for example FineWeb2, gives better resolution.

The completeness rate counts tokens and the split count counts source
characters. A tokenizer can have few incomplete tokens overall and still split
most multi-byte characters, because one split character produces several
incomplete tokens.

## Merge vocabulary usage (`merge_vocabulary_usage`)

How the merge vocabulary of a BPE tokenizer is used on the corpus. A merge token
is a vocabulary entry that at least one merge produces. Declared special and
added tokens are not merge tokens. Runs on any text data with no extra
configuration. Disable with `--no-merge-vocabulary-usage`.

For each merge token, the metric counts two quantities on the corpus:

- `final`: the number of times the token is emitted as a final token.
- `formed`: `final` plus the number of times the token is built as an
  intermediate symbol inside a longer emitted token.

From these, with N the number of final tokens in the corpus (special and added
tokens excluded):

- standalone rate = `final / N`
- survival = `final / formed`

Each merge token is in exactly one of four categories, which together include
all merge tokens:

| Category | Condition |
|---|---|
| Active | `formed > 0` and standalone rate at least 5 per million |
| Rare | `formed > 0` and standalone rate from 1 up to 5 per million |
| Uncommon | `formed > 0` and standalone rate below 1 per million |
| Unseen | `formed = 0` |

A Scaffold token is a Rare or Uncommon token with survival below 0.1: it is
rarely emitted on the corpus, and in more than 90% of the cases where BPE
builds it, it is merged further. Scaffold tokens are not removable, because the
longer tokens that are emitted are built from them.

`merge_vocabulary_usage.per_tokenizer.<tok>.global` contains the percentage of
merge tokens in each category (`pct_active`, `pct_rare`, `pct_uncommon`,
`pct_unseen`, `pct_scaffold`), the same numbers as counts under `counts`, and
`pct_scaffold_byte_fragment`, the percentage of merge tokens that are Scaffold
tokens and whose bytes are not a complete UTF-8 sequence.

**Example.** A tokenizer has the merges `a b`, `ab c` and `x y`, `xy z`. On a
corpus where `abc` occurs 3 times and `xyz` once, `ab` is formed 3 times and
never emitted, so its survival is 0. `abc` is emitted 3 times and its survival
is 1.

**`ignore_merges`.** The counts are those of encoding with `ignore_merges`
off. With the flag on, a pre-token (a piece of text after pre-tokenization)
that is already a vocabulary entry is emitted directly, without the merges that build it, so the intermediate counts
would describe merges the encoder did not run. A tokenizer that ships with the
flag on is encoded a second time with it off, and its entry contains
`"encoding": "reencoded_ignore_merges_false"` and the number of texts whose ids
changed. Some merge tokens cannot be built at all with the flag off, because a
merge earlier in the merge list always combines one of their characters with a
different neighbour first. They count as
Unseen, and `n_unreachable_without_ignore_merges` gives their number. Llama 3
has 588 such tokens.

**Intermediate tokens.** The metric finds the intermediate tokens by running
BPE on the string of each distinct emitted token, in the order the `tokenizers`
library applies merges: the merge earliest in the merge list first, and the
leftmost occurrence first when the same pair occurs twice. Several merges can
produce the same token. For example, the Apertus merge list contains three
merges that produce `Ġthe`. In that case the merge that BPE applies
depends on the ranks of the pairs in the string, so the intermediate tokens
cannot be read from a single fixed tree.

**Corpus size.** The thresholds are rates, so on a corpus of fewer than one
million final tokens a token emitted once is already above 1 per million, and
Uncommon means never emitted on its own. The metric still runs and adds a
warning to the entry. `threshold_counts` gives both thresholds as numbers of
emissions for the corpus at hand. A corpus with no final tokens raises an error, because no
rate is defined.

**Scope.** The counts are pooled over every language of the main input, so the
language mix of the input changes the percentages. `--merge-usage-include-code-math` adds
the code and math corpora. Without `--code-ast-config` or `--math-data` these
are the bundled samples, and `metadata.code_math_sources` lists them. The
metric is not computed per language group.

**Not applicable.** The metric reads the merge list of the encoder. A tokenizer
that is not BPE, a Hugging Face tokenizer with no `tokenizers` backend, a BPE
with dropout or with a subword prefix or end-of-word suffix, and pre-tokenized
input get `"applicable": false` with the reason, and a warning in the log lists
them.

## Code tokenization metrics

Source code is parsed with tree-sitter and the alignment between AST node
boundaries and token boundaries is measured. Tree-sitter is a core dependency
and is installed by default. Configure the corpus with `--code-ast-config`;
disable with `--no-code-ast`.

Configured for 19 languages, of which 16 are measured. Swift, Kotlin and Perl
are excluded because the node types their grammars use for identifiers are not
classified: the identifier share of classified leaves is 0.073, 0.058 and 0.000
against 0.19 to 0.37 for the supported languages. They are skipped with a named
warning rather than scored on a fraction of their code. A grammar that crashes
its parser process is likewise reported as unmeasured and named in the log.

Each language is parsed in its own subprocess, because a corrupt parse can abort
the whole process. A grammar that hangs is reported as unmeasured and named in
the log rather than scored. The per-language timeout defaults to 120 seconds;
raise it with `TOKEVAL_PARSE_TIMEOUT_S` on a loaded machine, where a grammar can
exceed the default and be dropped from a run that would otherwise measure it.

> **Data scope:** these metrics are **always** computed on dedicated source-code
> snippets, loaded via `--code-ast-config` or, without it, from small built-in
> synthetic samples. The general multilingual corpus passed to the analyzer is
> **never** used for this metric group, whatever the flags.

### AST leaf-node boundary alignment (`ast_boundary_alignment`)

> **Not comparable with `tokenizer-analysis-suite`.** Source spans now map to
> tokens through the tokenizer's own character offsets, where they were
> previously matched against a string rebuilt from token surfaces. That
> reconstruction lost synchronisation after any multi-space token. Llama 3's
> `full_alignment_rate` moved from 0.127 to 0.519.

Source code is parsed with tree-sitter, leaf-node spans are extracted, and the
fraction whose boundaries coincide with token boundaries is measured. Five
categories are tracked separately: identifiers, keywords, operators, literals
and delimiters. Reports start-alignment rate, end-alignment rate,
full-alignment rate and cross-boundary rate, per language and per category. The
per-category breakdown is dropped from `analysis_results.json` and kept in
`analysis_results_full.json`.

A node is start-aligned when it begins at position 0 or the token changes
between the character before it and its first character. It is end-aligned when
it ends at the end of the text or the token changes between its last character
and the character after. Full alignment requires both.

The correspondence between source characters and tokens comes from the
tokenizer's own offsets, not from decoding the tokens and matching the result
back against the source. One word-start space is dropped from each token's
range first, because `trim_offsets` is a ByteLevel post-processor flag that
changes the reported offsets without changing the tokenization: GPT-2 sets it
false and GPT-NeoX true, and reading the raw offsets scored GPT-2 at 0.433
against GPT-NeoX at 0.770 on token ids that were byte-identical.

A span that no token covers is counted as `unmappable` and excluded, rather than
scored as a missed boundary. Measured on 1500 real source files, 198 spans per
tokenizer are unmappable, 0.03%, and every one is a literal whose source text is
a single space, which no token covers once the word-start space is dropped.

**Example:** for the Python snippet `return total`, tree-sitter identifies
`return` (keyword, characters 0 to 6) and `total` (identifier, characters 7 to
12).

- Tokenized `return` | ` total`, the keyword `return` is start-aligned (it
  begins at 0) and end-aligned (the token changes at the space), so it is fully
  aligned. The identifier `total` is not start-aligned: its first character
  shares a token with the preceding space. Full alignment is 1/2 = 0.5. A token
  that is a leading space plus a word is the ordinary output of a byte-level
  pre-tokenizer, so this is the common case, not a pathological one.
- Tokenized `return` | ` ` | `total`, the space is its own token, both nodes are
  fully aligned, and full alignment is 2/2 = 1.0.
- Tokenized `ret` | `urn total`, `return` is start-aligned but not end-aligned
  (its last character shares a token with the space that follows), and `total`
  is not start-aligned. Full alignment is 0/2 = 0.0.

Code has a deterministic grammar, so AST node boundaries are derivable with no
manual annotation. A tokenizer that splits `return` into `ret` and `urn` splits
a syntactically atomic unit.

### Identifier fragmentation rate (`identifier_fragmentation`)

> **Not comparable with `tokenizer-analysis-suite`.** The same move to encoder
> offsets, which took Llama 3's unmappable identifier count from 119560 to 0,
> and `avg_tokens_per_identifier` no longer counts unmappable spans with a
> `-1` sentinel, which had biased it low and made it negative for C#.
>
> `identifiers_unmappable` counts identifier spans no token covers. It is
> reported so a low fragmentation rate measured over few spans is visible as
> such.

The fraction of programmer-defined identifiers split into more than one token,
plus the average number of tokens per identifier. Computed occurrence-weighted
from the same AST extraction pass, so a frequent identifier counts once per
occurrence.

**Example:** a Python file holds `self` (10 occurrences), `i` (5),
`process_data` (3) and `MyAuthenticationFactory` (1). The tokenizer keeps `self`
and `i` as single tokens, splits `process_data` into `process` | `_` | `data`
(3 tokens) and `MyAuthenticationFactory` into `My` | `Auth` | `entication` |
`Factory` (4 tokens). Fragmentation rate is 4 fragmented occurrences out of 19
total = 0.211. Average tokens per identifier is (10x1 + 5x1 + 3x3 + 1x4) / 19 =
1.474. The 10 occurrences of `self` supply 10 of the 19 terms in that average,
so the metric mostly describes the most frequent identifier rather than the
rarer, longer ones.

Identifiers are the names a programmer chose. Splitting `getUserName` at
arbitrary positions produces pieces that do not correspond to those names. The
current implementation does not distinguish splits at camelCase or snake_case
boundaries from arbitrary ones.

### Indentation depth proportionality correlation (`indentation_consistency`)

Whether the number of whitespace tokens produced for leading indentation grows
monotonically with nesting depth. The Spearman rank correlation (ρ) is computed
between indentation depth and the count of whitespace tokens in each line's
leading indentation.

Spearman is rank-based, so ρ = 1 means the count increases monotonically with
depth, not in proportion to it. The key is named for proportionality and the
statistic measures monotonicity.

Only tokens whose surface is entirely whitespace are counted, because a
pre-tokenizer that groups a leading space with the following word emits a token
spanning both, and that token is code. Depth is the line's leading-whitespace
width divided by the indent unit, which is inferred per snippet as the GCD of
its non-zero indent widths; it does not come from the parse tree. Evaluated only
on whitespace-significant languages (Python and Haskell), and only where a
language has at least 3 distinct depth levels.

**Example:** a Python file has lines at depths 1, 2, 3 and 4. A tokenizer that
encodes those indentations as 1, 2, 3 and 4 whitespace tokens gives ρ = 1.0. A
tokenizer that merges all indentation into a single token regardless of depth
(1, 1, 1, 1) gives ρ near 0.0. A tokenizer that uses more tokens for shallow
depths than for deep ones gives ρ < 0.

Where indentation depth maps monotonically to whitespace token count, nesting
depth is recoverable from the token sequence alone.

**How the global is computed:** `global.depth_proportionality_correlation` is
one Spearman correlation over the (depth, whitespace-token count) pairs of every
whitespace-significant language pooled together, not the mean of the
per-language correlations. The two differ. On the demo run named above, `bpe`
scores 0.7598 pooled against a per-language mean of 0.8121 (Python 0.9159,
Haskell 0.7083). Indent conventions differ by language, so the pooled value
depends on how much of each language the code corpus holds. Read `per_language`
for each language separately.

## Multilingual fairness

### Tokenizer Gini coefficient (`tokenizer_fairness_gini`)

Equitability of encoding cost across languages.

* Let $`L = \{1, \dots, n\}`$ be the set of languages, each weighted equally.
* For every language $`\ell \in L`$, the **token cost** is

```math
  c_\ell \;=\;
  \frac{\text{number of tokens produced by the tokenizer on language }\ell}
       {\text{number of measurement units in the same text}}
```

  A lower $`c_\ell`$ means the language is encoded in fewer tokens per unit of
  text. The unit is whatever `--measurement-config` sets, UTF-8 bytes by
  default.

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
  * $`0`$: every language has the same token cost per unit of text.
  * $`1`$: maximal inequality. The attainable maximum is $`1 - 1/n`$, so on 13
    languages no tokenizer can exceed 0.923.

**This is a fair comparison only on a parallel corpus.** The cost is tokens per
unit of text, so if the texts in two languages say different things, the ratio
between their costs is partly the ratio of what they say. FLORES+ is parallel,
which is why the bundled configs use it.

**Even on a parallel corpus, the byte unit is not neutral across scripts.** UTF-8
uses one byte per Latin character, two for Cyrillic and Greek, three for most
CJK and Devanagari. A tokenizer can therefore look cheaper on Chinese than on
English purely because the denominator is larger, with no difference in how well
it segments either.

**So every run also publishes the coefficient normalized per line**, at
`tokenizer_fairness_gini.per_tokenizer.<tok>.per_line_normalization`, holding
`gini_coefficient`, `lines_per_language`, `mean_cost`, `cost_ratio`,
`language_costs` and `num_languages`. On a parallel corpus, line *i* of every
language is the same sentence, so tokens per line compares tokenizers on
identical content, which no other unit does. **On a parallel corpus this is the
coefficient to read.**

The two are far apart. Over the nine tokenizers of `benchmarks/open_source` on
13 FLORES+ languages they rank at Spearman 0.650 and disagree on which
tokenizer is the most equitable across languages:

| Tokenizer | per byte | per line |
|---|---:|---:|
| XLM-RoBERTa base | 0.0976 | **0.0494** |
| Mistral NeMo | 0.1097 | 0.0702 |
| Llama 3 | **0.0772** | 0.0926 |
| GPT-2 | 0.2000 | 0.2933 |

`per_line_normalization` is `null` unless every language has the same line
count. That condition is necessary for a corpus to be parallel and is not
sufficient: equal counts do not establish that line *i* in one language is a
translation of line *i* in another, which the library cannot check and the
caller has to know. When the counts differ, the block is `null` and the log
names the counts, rather than a number being published for a comparison that
does not hold.

Setting `--measurement-config` to a lines config instead moves the primary
coefficient onto lines, and also moves `compression_rate`. That is a different
choice from reading `per_line_normalization`, and the number under a lines
config is not comparable with the number under the byte default.

The underlying Lorenz curve is written under
`tokenizer_fairness_gini.per_tokenizer.<tok>.lorenz_curve`, from which
`1 - 2*area` is the Gini coefficient. With fewer than 2 languages the
coefficient is `null` and a sibling `warning` field records the reason.

### Cross-lingual vocabulary-utilization CoV

`vocabulary_utilization.per_tokenizer.<tok>.per_language_cov`. The coefficient
of variation (sample standard deviation divided by mean, `ddof=1`) of the
per-language vocabulary-utilization *ratio* across languages. Computed on the
ratio rather than on the raw used-token count, so it is comparable across
tokenizers with different vocabulary sizes.

Lower is better: a low value means each language uses a similarly sized share of
the vocabulary; a high value means the used vocabulary is concentrated in some
languages. This complements the Gini coefficient above, which measures the
equitability of per-language *encoding cost*, where this measures the balance of
per-language *vocabulary coverage*.

Requires at least 2 languages with mean utilization above 0. For a
single-language corpus it is `null` rather than a fabricated `0`, and it is
omitted from the plot. `per_language_mean` and `per_language_std` are beside
it, and are `null` under the same condition, since a dispersion over fewer than
two languages is undefined.

The comparison plot is written to `vocab_util_cross_lingual_cov_individual.svg`;
that string is a plot filename, not a results key.
