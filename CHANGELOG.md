# Changelog

This project supersedes `tokenizer-analysis-suite`. The section below is the
complete set of changes from that suite to the current release, ordered with
the breaking ones first. Per-release detail is in `git log`.

The install name is now `tokenizer-intrinsic-evals` (was `tokenizer-analysis`).
The import name is unchanged: `import tokenizer_analysis`. The prior suite,
including the `PA_BPE_tokenizers/` directory, is preserved on the
`legacy-suite` branch and the `legacy-suite-final` tag.

---

## Changes from tokenizer-analysis-suite

### Breaking: metric definitions changed, so old numbers are not comparable

These changed what a metric computes, not where it is written. A value from the
old suite cannot be compared with a value from this one for any field below.

| Field | What changed | Measured effect |
|---|---|---|
| `ast_boundary_alignment`, `identifier_fragmentation` | Source spans map to tokens through the tokenizer's own character offsets. They were previously matched against a string rebuilt from token surfaces, which lost synchronisation after any multi-space token. | Llama 3 `full_alignment_rate` 0.127 to 0.519; its `unmappable` count 119560 to 0 |
| `three_digit_boundary_alignment`, `numeric_magnitude_consistency`, `operator_isolation_rate` | Same change from reconstruction to offsets. | Far more spans measured for any language with non-ASCII text |
| `operator_isolation_rate` | A character claimed by two token ranges is assigned to the later token. | XLM-RoBERTa isolation 0.6948 to 0.6770, compound preservation 0.7222 to 0.8443, over 454693 operators |
| `operator_isolation_rate` | The main corpus is no longer scored as a `prose` domain unless `--operator-prose-domain` is passed, so `global` pools code and math only. An operator is a code construct, and the pattern matches a hyphen, a slash and an exclamation mark. | On the bundled demo `bpe` moves from 0.7938 over 3016 occurrences to 0.7250 over 2229. On the nine-tokenizer benchmark prose supplied 568 of 455558 occurrences, 0.12% |
| `identifier_fragmentation.avg_tokens_per_identifier` | Unmappable spans are excluded rather than counted with a `-1` sentinel. | Was biased low, and negative for C# |
| `utf8_token_integrity`, `utf8_char_split` | Byte-level detection no longer requires 50 of 68 GPT-2 marker characters in the vocabulary. | gpt4o-english-bpe `completeness_rate` 1.0000, best of 37 tokenizers, to 0.6688, worst of 37 |
| `reconstruction_fidelity.whitespace_fidelity` | Whitespace widened from ASCII space, tab, newline and carriage return to those plus every Unicode `Zs` separator. | Any corpus containing NBSP, thin space or ideographic space |
| `renyi_efficiency` | Normalizes by `log2` of the declared vocabulary size, following Zouhar et al. 2023. It previously divided by the number of token types observed in the corpus. | The two rank tokenizers at Spearman 0.678 over 37 tokenizers. The old normalization is still published under `renyi_efficiency.observed_normalization` |

A `null` where an old run had `0.0` reflects the convention change below, not a
change in what was measured.

### Breaking: output format

- Every metric publishes a per-tokenizer `global`. Five had none.
- Every metric's `metadata` includes `aggregation`, one of `micro_pooled`,
  `macro_languages`, `ratio_of_sums`, `set_union` or `mean_of_ratios`, naming
  which average its `global` is. Every `per_language` entry includes a `count`, so the other
  weighting can be re-derived.
- A value that could not be measured is `null`, not `0.0`. `0.0` is a legal
  value for most of these metrics, so a zero was indistinguishable from a
  measurement. `count` and `sum` stay numeric.
- The per-tokenizer compression key is `compression_rate`, was
  `compression_ratio`.
- `analysis_results.json` is organized as
  `{metric: {per_tokenizer: {tok: {global, per_language}}, metadata}}`. The old
  flat layout and its `summary` and `pairwise_comparisons` blocks are gone.
- `analysis_results.json` is a strict projection of `analysis_results_full.json`:
  every leaf is at the same key path with the same value in both.
- Six metrics are no longer top-level keys. Each is a field of the metric that
  owns the measurement, with the reason in that metric's
  `metadata.merged_metrics`.

  | Old top-level key | New location |
  |---|---|
  | `avg_tokens_per_line` | `compression_rate.per_tokenizer.<tok>.tokens_per_line` |
  | `type_token_ratio` | `vocabulary_utilization.per_tokenizer.<tok>.type_token_ratio` |
  | `unigram_distribution_metrics` | `renyi_efficiency.per_tokenizer.<tok>.unigram_distribution` |
  | `utf8_char_split` | `utf8_token_integrity.per_tokenizer.<tok>.char_split` |
  | `lorenz_curve_data` | `tokenizer_fairness_gini.per_tokenizer.<tok>.lorenz_curve` |
  | `digit_split_variability` | `three_digit_boundary_alignment.per_tokenizer.<tok>.split_variability` |

  Four are exact identities, so nothing is lost: `compression_rate` under the
  `lines` measurement times `avg_tokens_per_line` is 1; `1 - 2*area(lorenz)` is
  the Gini coefficient; `renyi_1.0` times `log2` of the declared vocabulary
  size is the unigram entropy; and the type-token ratio is vocabulary
  utilization rescaled by vocabulary size over token count. The other two rank
  tokenizers at Spearman -0.954 and -0.992 against the metric they were folded
  into.

### Breaking: Python API

- `TokenizerWrapper` subclasses must implement `get_special_token_strings()`.
  It is abstract, so a wrapper written against 0.x raises `TypeError` until it
  is added. Return the surface strings the tokenizer declares special, read
  from its own metadata; an empty set only if it genuinely has none, and `None`
  if it cannot report them, after which the library warns and falls back to
  `GENERIC_SPECIAL_TOKENS`. Do not pattern-match on token surfaces. That is
  what the removed `_SPECIAL_TOKEN` regex did, and it deleted ordinary content
  tokens such as `[0]` and `[...]` while missing `<s>` and `</s>`.
- `MorphologicalMetrics` and `MorphologicalDataLoader` are removed. Use
  MorphScore, through `MorphScoreMetrics` or the `--morphscore` flags.
- `MarkdownTableGenerator`, `results_filename` and
  `UnifiedTokenizerAnalyzer.generate_markdown_table()` are removed. The
  cumulative Markdown leaderboard they produced was built for one internal
  project. Read `analysis_results.json`, or use `generate_latex_tables()`.
- `UnifiedTokenizerAnalyzer(...)` no longer accepts `morphological_config`, and
  `run_analysis(...)` no longer accepts `include_morphological`.
- Constants moved from namespace classes to module-level names. Replace
  `from tokenizer_analysis.constants import DataProcessing` and
  `DataProcessing.DEFAULT_CHUNK_SIZE` with
  `from tokenizer_analysis.constants import DEFAULT_CHUNK_SIZE`.

Accessing a removed name raises with its replacement named, rather than a bare
`ImportError`.

### Breaking: command line

| Removed | Use instead |
|---|---|
| `--morphological-config` | `--morphscore`, or `--morphscore-config <file>` |
| `--latex-table-types morphological` | any of the remaining table types |
| `--update-results-md`, `--dataset`, `--sort-results-by` | read `analysis_results.json`, or `--generate-latex-tables` |

`--use-sample-data` no longer overrides `--tokenizer-config`,
`--language-config` or `--measurement-config`. It previously replaced all three
without a warning, producing a complete results file for tokenizers and a
corpus the caller did not ask for. The combination is now an error naming the
conflicting flags.

### Removed

- `avg_langs_per_token`, the cross-language token-sharing figure under
  `vocabulary_utilization`. It was published only in
  `analysis_results_full.json`, so no value in `analysis_results.json` changes.
  It was removed rather than documented because it is close to a monotone
  function of vocabulary size: over the nine tokenizers of
  `benchmarks/open_source` on 13 FLORES languages it ranges 1.239 to 2.504
  against a theoretical 1 to 13, with Spearman -0.950 against vocabulary size
  and -0.933 against the number of token types observed. Read as its own
  description invited, higher meaning more cross-language sharing, it ranked
  bert-base-uncased, which is English-only and strips accents, as the most
  multilingual of the nine, and put XLM-RoBERTa and Gemma 2 last.

### Added

- `--input PATH` for a single corpus, taking a file or a directory, with
  `--input-label` for the name it appears under. Omitting both `--input` and
  `--language-config` is an error naming both.
- A `run_metadata` block in every results file: package version, UTC timestamp,
  git commit and tree state, the config paths with hashes, a hash or Hub
  revision per tokenizer, a digest of each corpus file, the non-default
  arguments, and the code-corpus caps.
- `tokenizer-sanity-check`, which runs 16 checks against one tokenizer and
  exits non-zero on a failure.
- `benchmarks/open_source/`, nine widely used tokenizers measured on the full
  metric set, regenerated by one command.
- `--max-code-files-per-lang` and `--max-code-file-chars`, replacing silent
  caps of 100 files and 15000 characters.
- `--operator-prose-domain`, which restores the `prose` domain of
  `operator_isolation_rate`. See the breaking-change table above for why it is
  off by default.

### Fixed, where the fix changed a published number

- A missing corpus file, a missing MorphScore directory and a failed language
  load aborted rather than producing a complete-looking file with a language
  silently dropped or four zeros where nothing was evaluated.
- `tokenizer_fairness_gini` with fewer than two languages reports `null` rather
  than `0.0`, which read as perfect fairness. `cost_ratio` reports `null`
  rather than `float('inf')`, and `std_cost` uses `ddof=1`.
- The Gini coefficient is deterministic. The per-language cost vector was
  summed in a hash-dependent order, so one commit produced two values.
- Tree-sitter `ERROR` spans are excluded from `ast_boundary_alignment` and
  published as `parse_error_spans`. They were scored as AST leaves and scored
  above each tokenizer's average, so every rate was optimistic.
- `indentation_consistency.depth_proportionality_correlation` reports `null`
  when the whitespace-token count is constant across depths, where a
  correlation is undefined. It reported `0.0`, which is a real measurement
  meaning depth and whitespace-token count are unrelated.
- The results file is strict JSON on every corpus. It could contain `NaN`.
- A subword marker is stripped only from a tokenizer shown to use it. The
  WordPiece, CLIP-BPE and subword-nmt rules were applied to every tokenizer,
  truncating ordinary content such as Markdown headings and punctuation runs.
- Directory input is sorted, so the same corpus gives the same result on two
  machines.

---

## Releases

- **1.2.0** New metric, `merge_vocabulary_usage`. For each BPE merge token it
  counts how often the token is emitted and how often it is built as an
  intermediate step, and reports the percentage of merge tokens that are
  Active, Rare, Uncommon, Unseen and Scaffold. The definitions are in
  [docs/METRICS.md](docs/METRICS.md#merge-vocabulary-usage-merge_vocabulary_usage).
  On by default; `--no-merge-vocabulary-usage` turns it off, and
  `--merge-usage-include-code-math` adds the code and math corpora. The LaTeX
  output has a new `merge_usage` table type.

- **1.1.0** A metric constructed with the same corpus arguments as one
  built before it is accepted rather than refused. Metric constructors register
  the corpus they build, and the check that stops a later metric silently
  reading someone else's corpus asked only whether anything was registered, not
  whether it was the same thing. So passing identical `code_texts` to two
  classes failed from the second, each asking for the corpus the other had
  already put there. The check now compares what the arguments would build
  against what is registered, and refuses only when they differ, which is the
  case it exists for.

- **1.1.0** `reconstruction_fidelity`'s three structural whitespace
  measures appear in the LaTeX tables. They were published in the results file
  and rendered nowhere. All three are null on a corpus with no indentation, no
  tabs or no newlines, which is every prose-only run, so a table of a default
  run shows them empty.

- **1.1.0** `tokenizer_fairness_gini` publishes the same keys whether or
  not a group has enough languages to define a coefficient. The
  fewer-than-two-languages case omitted seven of them, directly below a comment
  saying every block carries the same keys so a consumer can index rather than
  test for existence. Both cases now come from one function, so they cannot
  drift apart. Only the coefficient and the standard deviation are null with a
  single language: the smallest and largest cost are that language's cost,
  their ratio is 1.0, and the most and least efficient language are both that
  one. **Values appear where keys were absent** in the full results file for
  single-language groups; the slim file is unchanged.

  The slim writer read those fields with `.get()`, so a key the metric never
  produced became a null and the slim file published two fields per
  single-language group that the full file did not have, twenty in a real run.
  It copies only what is present now. The check that the slim file is a subset
  of the full one was live and passing throughout, because its fixture ran no
  grouped analysis and so produced no group of one language. It does now.

- **1.1.0** Grouped analysis computes UTF-8 integrity metrics when the
  caller passed no base results, instead of omitting the block entirely. A
  whole metric was present or absent depending on how the function was called.
  It still reports none when the base run produced no UTF-8 metrics, which is
  what `--no-utf8-integrity` does: that missing key is the only signal the
  grouped path receives, so computing unconditionally would switch the metric
  back on for someone who asked for it off. Three cases now, matching the digit
  metrics beside it.

- **1.1.0** With `include_code_math` off, the digit metrics measure
  nothing instead of reading the whole maths corpus. That setting says a call
  covers one language group rather than the whole corpus, and the digit metrics
  ignored it, so every group got identical whole-corpus figures. Reachable
  through `run_grouped_analysis` without base results; the command-line tool
  always takes the filtered path, where the group's own languages are selected
  and the maths rows drop out, so no published number of ours was affected.
  Measuring the group's prose instead was the other option and is worse: that
  branch's own comment records that its numbers are not comparable with a run
  that has a maths corpus, so publishing them under the same field name would
  report one quantity under the name of another.

- **1.1.0** `reconstruction_fidelity` refuses a corpus whose prose language
  name collides with a code or maths domain, instead of adding the two together
  in one row. A corpus with a language named `math`, which `--input` produces
  from a directory holding `math.txt`, had its counts, token totals and rates
  summed with the maths corpus and nothing in the output said so. The same
  applied to a language named `code_<lang>` beside a code corpus of that
  language. `operator_isolation_rate` already refused this; the two checks stay
  separate because the operator metric only sees languages that produced
  operator characters, so one shared check would refuse runs that work today.

- **1.1.0** The keys naming a code or maths domain are built in one place.
  `metrics/basic.py` wrote them out by hand at three sites and recognised them
  by hand at two more, while `metrics/math.py` had a function whose docstring
  claimed to be the single definition. Six places, agreeing by coincidence.
  `published_language_key` and `is_corpus_domain_key` now live in
  `core/input_types.py`, which both files already import and which pulls in
  nothing beyond the standard library. No published value changes.

- **1.1.0** `benchmarks/open_source/` regenerated. The committed results
  were produced at package version 1.0.2, before the shared-corpora refactor and
  before every fix since, so the documentation described fields the file did not
  contain. Regenerated at 1.1.0 with a clean tree. Every difference is accounted
  for by a change already recorded above: 270 leaves each for
  `indentation_fidelity`, `newline_fidelity`, `tab_fidelity` and
  `decode_failures`; 126 for `zero_token_documents`; 144
  `operator_isolation_rate.per_language` keys renamed from the colon form to the
  underscore form; four aggregation labels and one `count_unit` corrected; and
  four `whitespace_fidelity` values, all Hindi text for `gpt-neox-20b` and
  `qwen-2.5`, moving from 0.9978 and 0.999996 to 1.0, because the old scan
  counted whitespace as lost where it had survived. `REPORT.md` changed only its
  provenance line, so no headline number moved.

  `docs/OUTPUT.md` named `decode_failures` at a path it does not occupy. The
  field is under `per_tokenizer.<tok>.global`. The documentation was wrong, not
  the code, and the test exemption added when the artifact lagged behind is
  removed: it could never have expired on its own, because the path it named
  was never going to resolve.

- **1.1.0** `operator_isolation_rate.per_language` keys code and math the
  way `reconstruction_fidelity.per_domain` does: `code_bash` and a bare `math`,
  where it published `code:bash` and `math:math`. One results file named the
  same thing two ways, and neither spelling was documented on the
  reconstruction side. **Key names change** in that block; no value moves. The
  colon was load-bearing, not cosmetic: `_filter_operator_results` selects a
  language group with `l in target_languages`, and a namespace no FLORES code
  can contain kept code and math rows out. Since `--input` names languages
  after files, a corpus holding `math.txt` now collides, so the collision is
  refused rather than relocated: two domains that would publish under one key
  abort naming it, instead of being summed into one number that a language
  group would then select. The abort covers `operator_isolation_rate` only, and
  only with `--operator-prose-domain`, which is off by default. The per-domain
  block of reconstruction fidelity had the same collision and pooled silently;
  that was recorded here when this entry was written and is fixed in the entry
  above about a prose language name colliding with a code or maths domain.

- **1.1.0** Three library-path defects. A `Corpus` label given a bare
  string was split into one text per character, so `"x = 1"` became five texts
  and every per-text metric scored single characters; it is refused with the
  list form named. `RawTokenizationProvider.get_vocab_size` returned 0 for a
  tokenizer that cannot report one, which flowed into
  `vocabulary_utilization` and `renyi_efficiency` as a zero denominator and
  published nothing measurable without saying why; it raises, as
  `PreTokenizedProvider` already did. `PreTokenizedProvider.get_languages`
  returned set order, so `per_language` key order was hash-dependent across
  runs of one dump; it is sorted.

- **1.1.0** `InputProvider._encode_corpus` aborts instead of falling back
  to a second encode path. A failing or malformed `encode_batch_with_offsets`
  re-encoded the corpus one text at a time under a warning asserting that the
  ids and offsets were the same either way; nothing verified that, and a
  wrapper whose two methods disagree makes it false, so the run would publish
  numbers measured through a path other than the one it reported. The batch
  path also gained the `ids is None` check the per-text path already had. The
  per-text `ids is None` branch is split rather than removed: with no
  `encode_with_offsets` at all it is the primary encode path, and only a
  `None` returned *by* that method is now an error. Library-reachable only; no
  shipped wrapper triggers any of it.

- **1.1.0** `PreTokenizedProvider` refuses a record whose
  `tokenizer_name` disagrees with the key it is filed under, naming both. It
  copied the record under the key's name at warning level, and every metric
  then scored one tokenizer's ids under another's name. Both name-agreement
  validators read the corrected output, so nothing downstream could catch it.
  Reachable through `--tokenized-data-file`; a dump that loaded with a warning
  now aborts.

- **1.1.0** `by_domain.<domain>.corpus` counts only texts the metrics
  scored. `corpus_size` applied no blank filter while every scored path filters
  on `text and text.strip()`, so a corpus holding a whitespace-only text
  published one text more than any metric read, in the block whose purpose is
  to say what a number was measured on. The AST metrics also refuse a corpus
  whose every label holds an empty list, which passed the previous
  empty-corpus guard because the dict itself was non-empty.

- **1.1.0** `reconstruction_fidelity` publishes `decode_failures` per
  domain, in `overall` and in `summary`, and logs one line per failure naming
  the tokenizer, the domain and the text. A text whose `decode()` returns
  `None` leaves every reconstruction denominator, so a domain with one failure
  in three published `exact_match_rate` 1.0 with nothing in the file saying it
  was the rate over the two that decoded. All four shipped wrappers return
  `None` from `decode` on an internal exception. `total_tokens` and
  `unk_tokens` deliberately still cover every encoded text: `count` is the
  texts that decoded, so `total_tokens / count` is inflated by the failures,
  and this is now stated in `docs/OUTPUT.md`.

- **1.1.0** A document a tokenizer encodes to zero tokens is logged, and
  each metric that reads one now states its position rather than differing by
  accident. `fertility` excludes it and publishes
  `zero_token_documents` per tokenizer and per language; `token_length`,
  `avg_tokens_per_line` and `compression_rate` already excluded it, two of them
  because tokens are their denominator. The Gini blocks and
  `reconstruction_fidelity` keep it deliberately: a zero-cost language is how a
  fairness metric reports that a tokenizer erased a language, and a total
  round-trip failure is what reconstruction fidelity measures. The recorded
  justification for dropping the `TokenizedData` emptiness check when the
  corpora were made shared was wrong, and is corrected: blank-to-`strip()` and empty-after-encoding are
  different properties, and a text of C0 control characters is the second
  without being the first, reachable through `--input`.

- **1.1.0** The bundled code samples honour `--max-code-files-per-lang`,
  so the two code metrics score the same texts. The cap reached
  `ASTBoundaryMetrics` through `get_code_snippets` and nothing else, so with
  `--max-code-files-per-lang 2` and no `--code-ast-config` the operator and
  digit metrics measured 57 texts over 19 languages while the AST metrics
  measured 2 per language, both published under one corpus name and source.
  **Values move on any run that sets the flag without a code config**; both
  caps default to 0, and every result file carrying `code_corpus_caps` recorded
  0, so no published number of ours is affected. `--max-code-file-chars` is now refused
  on this path with a warning rather than silently ignored: the 57 samples
  parse with zero tree-sitter errors at full length and 19 with a 400-character
  cut, so truncating them would move every AST alignment rate for a reason
  unrelated to tokenization, and the whole corpus is 48715 characters.

- **1.1.0** `fertility`, `token_length`, `three_digit_boundary_alignment`
  and `numeric_magnitude_consistency` declare `aggregation: mean_of_ratios`.
  All four averaged one ratio per document or per number while declaring
  `micro_pooled`, which `constants.py` defines as one ratio from summed counts.
  On the committed benchmark the two differ by up to 18.7% of the value
  (`token_length.byte_length.mean` 2.4154 for gpt2 against 2.0356 pooled), so a
  consumer following `docs/OUTPUT.md`'s re-derivation contract got a different
  number from the one published. `mean_of_ratios` is a fifth aggregation label.
  `token_length.metadata.count_unit` is corrected from `tokens` to `documents`:
  the count is 3250 documents on the benchmark against 109014 to 271337 actual
  tokens. `reconstruction_fidelity` gains
  `metadata.aggregation_exceptions`, because six of its seven rates are pooled
  counts and `mean_cer` is not. No computed value changes.

- **1.1.0** `reconstruction_fidelity.whitespace_fidelity` counts
  whitespace an alignment of the two texts can match, and three structural
  sub-rates are published beside it. The old greedy scan left its pointer
  behind on any character it could not match, so every whitespace after a
  substituted character was compared at the wrong index and counted as lost:
  `"El nino esta aqui"` decoded from the accented original reported 2 of 3, and
  a corpus with true fidelity 1.0 under an NFD-strip-lowercase decode published
  0.8667. **Values move**, in both directions and mostly upward. In the
  committed benchmark 4 leaves change (`gpt-neox-20b` and `qwen-2.5`); 10 files
  under `results/` hold a non-1.0 value, the lowest 0.066754. The rule matches
  the full text rather than the whitespace alone, so `"hello world"` decoded as
  `"helloworld "` scores 0 of 1 rather than crediting a deleted word boundary
  against a space appended elsewhere. New: `indentation_fidelity` (run exact
  per line, because a four-space indent arriving as three is broken code),
  `newline_fidelity` and `tab_fidelity`, all partitions of the same alignment.
  The roll-up alone could not separate harmful damage from harmless: destroying
  every indent in the bundled code corpus moves it to 0.578 while collapsing
  inner spaces harmlessly leaves it at 0.980. No bundled corpus holds tabular
  data, so `tab_fidelity` is evidence about code indentation only.

- **1.1.0** `reconstruction_fidelity` publishes `null` where a rate has no
  denominator, with no stand-in defaults. `exact_match_rate` and `mean_cer`
  were `0.0` on a domain where every decode failed, which read as a perfect
  round trip beside a `count` of 0; `unk_token_rate` was `0.0` for a tokenizer
  declaring no UNK id; `whitespace_fidelity` was `1.0` for a text holding no
  whitespace. Two conventions documented in `docs/METRICS.md` are removed in
  favour of the `docs/OUTPUT.md` rule that an uncomputable value is null.

- **1.1.0** A language group's digit-metric blocks report the group or
  nothing, never the whole corpus. `--run-grouped-analysis` copied the base
  run's `three_digit_boundary_alignment` and `numeric_magnitude_consistency`
  `summary` into every group byte-identically, and passed the magnitude
  `scaling` fit (Spearman rho, cv, linear fit, pooled over every language of
  the run) through unfiltered into the slim `analysis_results.json` of every
  group. On the grouped golden configuration (`--use-builtin-math-data`,
  where the digit metrics measure the math corpus, which belongs to no
  group), every group published the identical 627-number whole-corpus
  summary and fit next to an empty `per_language`. A group `summary` is now
  re-aggregated from the group's own per-language blocks (count-weighted
  means of per-language means), is empty when the group's languages hold no
  numbers, and leaves out the summary fields the per-language blocks it
  reads do not determine (`avg_uniform_chunk`, `single_token_frac`, and the
  scaling-derived `cv_of_mean_fertility`/`spearman_rho`/`linear_*`);
  `scaling` is dropped from filtered group blocks. The recompute path (no
  base results) computes the fit from whatever the digit metrics measure:
  the group's own data, unless a dedicated math corpus is configured, in
  which case the fit covers the whole math corpus in every group, a
  pre-existing behavior recorded as R12. A filtered `by_bucket` keeps
  every bucket key, holding `{}` where the group has no such numbers,
  instead of losing `long` when only short numbers survived the filter.
  Grouped `operator_isolation_rate` metadata now describes the filtered
  block (re-aggregated prose, no `by_domain`) instead of repeating the
  whole-corpus "Code and math always run" description beside an empty
  block; grouped `numeric_magnitude_consistency` metadata stops promising
  the dropped fit. Both descriptions are single-sourced in
  `metrics/math.py` (`operator_metadata`, `magnitude_metadata`). Grouped
  MorphScore blocks no longer carry an invented top-level `summary` whose
  `total_languages_evaluated` summed over tokenizers (9 tokenizers over 5
  languages reported 45). **Values under `grouped_analysis` change and are
  not comparable with an earlier grouped run**: on the grouped golden
  configuration, 1200 leaves removed (920 scaling, 280 summary), 20 empty
  summary leaves added, 20 metadata descriptions changed, all inside
  `grouped_analysis`. Nothing outside it moves: the ungrouped golden
  configurations are identical leaf for leaf, and the ungrouped
  `operator_isolation_rate` description is byte-identical.
  `benchmarks/open_source/analysis_results.json` contains no
  `grouped_analysis`, so the published benchmark is unaffected. All four
  copy defects predate this branch (RELEASE_AUDIT Q35.2 R1/R2).

- **1.1.0** `grouped_analysis.<grouping>.<group>.reconstruction_fidelity`
  reports the group's prose languages only. It reported the whole code and math
  corpus inside every group, because the group selects prose languages while the
  code and math loop ran unconditionally. On the bundled demo the Arabic script
  family reported 321 texts in its `global`, of which 6 were Arabic and 315 were
  the same code and math texts that appeared in every other group; it now
  reports 6. The Latin family goes from 333 to 18, and its `whitespace_fidelity`
  from 0.7641 to 0.9907. **Values under `grouped_analysis` change and are not
  comparable with an earlier run.** Nothing outside `grouped_analysis` moves:
  4883 leaves compared, none changed. `benchmarks/open_source/analysis_results.json`
  contains no `grouped_analysis`, so the published benchmark is unaffected.

- **1.1.0** The code and math corpora are read from disk once per run and
  encoded once per tokenizer. They were read twice and encoded three times, by
  `BasicTokenizationMetrics`, `DigitBoundaryMetrics` and `ASTBoundaryMetrics`
  separately. No metric value changes: `analysis_results_full.json` is identical
  leaf for leaf across four configurations, at 4794, 5914, 1338 and 14054
  leaves. With `--max-code-file-chars` set (it is unset by default) and a file
  whose truncated prefix is whitespace, three corpus-size leaves under
  `operator_isolation_rate.by_domain.code.corpus` move, because that snippet is
  now dropped rather than counted; see the truncation change below.

  `InputProvider` has two new concrete methods, `add_corpus(Corpus)` and
  `get_corpus_data(name)`, which the code and math corpora travel through.
  `get_tokenized_data()` keeps its 1.0.3 signature and still returns the
  provider's own prose texts, so an existing subclass needs no change. Both new
  methods refuse the name `prose`, which is served only by
  `get_tokenized_data()`. `Corpus` is exported from `tokenizer_analysis.core`.

  **Breaking**: `MixedInputProvider` is deleted, so
  `from tokenizer_analysis.core.input_providers import MixedInputProvider`
  raises `ImportError`. It was not exported from `tokenizer_analysis.core` and
  no CLI path constructed it.

  **Breaking**: building a second `UnifiedTokenizerAnalyzer` over an input
  provider that already has one raises `A corpus named 'code' is already
  registered`. The analyzer registers the code and math corpora on the provider,
  and a name is registered once so that two loaders cannot disagree about what
  a corpus holds. Build the second analyzer over its own provider. Constructing
  `DigitBoundaryMetrics` twice against one provider does not raise: the second
  finds the corpora the first registered, is passed no arguments, and reuses
  them.

  **Breaking**: `DigitBoundaryMetrics` raises `TypeError` against a provider
  that does not implement `add_corpus`, when it has to build a corpus itself,
  because it registers what it builds. `ASTBoundaryMetrics` does not raise:
  it never registers a corpus, so a corpus it builds itself stays in its own
  loader and no registry is needed. A duck-typed provider that satisfied 1.0.3
  therefore needs the corpus registry only for `DigitBoundaryMetrics`, which
  subclassing `InputProvider` supplies. This is the one case where the
  statement above, that an existing subclass needs no change, does not hold:
  it holds for a subclass, and not for an unrelated class that merely
  implemented the same method names.

  `BasicTokenizationMetrics`, `DigitBoundaryMetrics` and `ASTBoundaryMetrics`
  raise when given corpus arguments while a corpus of that name is registered
  on the input provider.

  The arguments are `code_texts`, `math_data_path` and `use_builtin_math_data`
  for the first two, and `code_config` for the AST metrics. The registered
  corpus previously took precedence and the argument was dropped without a
  warning, so a caller who named real code paths while the bundled samples were
  registered got AST numbers measured on synthetic code under the name of their
  own corpus. `max_snippets_per_lang` is still accepted alongside a registered
  corpus, because it bounds a corpus rather than selects one and
  `CodeDataLoader.get_code_snippets` applies it to whatever the loader holds.
  `max_snippet_chars` is refused, because nothing on that path can apply it: it
  takes effect inside `CodeDataLoader.load_all`, which a registered corpus does
  not go through. Whether an empty value counts as a request is decided per
  argument: `code_config={}` asks for the bundled samples and is refused, while
  `code_texts={}` and `max_snippet_chars=0` ask for nothing and are not.

  **Breaking**: `TokenizedData` no longer rejects an empty `tokens` list. A
  tokenizer that encodes a text to zero tokens has measured something, and
  refusing to construct the record turned that into a crash. A caller relying on
  the constructor to reject a pre-tokenized row with no ids now gets a scored
  record instead of an error: an exact-match miss with CER 1.0.

  **Breaking**: `encode_with_offsets` raising is no longer caught. It was logged
  at debug level and the text re-encoded with `encode()` and no offsets, which
  measured that one text through a different path from the rest of its corpus.
  `TokenizerWrapper.encode_with_offsets` returns `(ids, None)` when a tokenizer
  has no offsets, so raising is a defect in the wrapper. A run that used to
  complete with a degraded encoding now fails, naming the tokenizer, the corpus,
  the label and the original exception.

  A tokenizer that decodes but cannot encode raw text keeps its prose domains
  in reconstruction fidelity and loses only the code and math domains, with a
  logged warning, when the run has code or math texts to encode. It reached
  the encode call and raised out of the whole analysis. No wrapper in this
  package is both: `PreTokenizedDataTokenizer` reports `can_decode()` false
  and was already skipped one check earlier, so this affects a caller
  supplying a tokenizer object of their own. Its per-domain prose numbers
  are unchanged, but its `overall` and `global` figures, which pool every
  domain, then cover a different document set than tokenizers that also
  have code and math domains; the missing `code_*`/`math` keys under
  `per_domain` are the marker. Reconstruction fidelity also no longer skips a tokenizer whose
  loader raised an unrelated error; it caught every exception and reported
  the run as a success with that tokenizer absent.

  `--max-code-file-chars` drops a code snippet that truncation leaves
  whitespace-only, counted per language in
  `CodeDataLoader.dropped_whitespace_only_counts` and logged. A snippet that
  survives truncation is unchanged. Truncation is off by default.

- **1.0.3** `tokenizer_fairness_gini` publishes a second coefficient at
  `per_tokenizer.<tok>.per_line_normalization`, normalized by line count rather
  than by the configured unit, and `null` unless every language has the same
  line count. On a parallel corpus it is the one to read: over the nine
  tokenizers of `benchmarks/open_source/` the two rank at Spearman 0.650, and
  XLM-RoBERTa is fourth at 0.0976 per byte and first at 0.0494 per line.
  `HuggingFaceTokenizer.can_pretokenize()` read the pre-tokenizer off the
  transformers object rather than off `backend_tokenizer`, so it returned False
  for every tokenizer loaded from the Hub. Sanity check C10 was
  `not_applicable` for all nine benchmark tokenizers and is now a measurement,
  and C16 reported 0 pretokenizer-unreachable vocabulary tokens for
  bert-base-uncased where the count is 6823. `avg_langs_per_token` is removed:
  measured over those nine on 13 FLORES+ languages it ran 1.239 to 2.504 against
  a theoretical 1 to 13, at Spearman -0.950 against vocabulary size, and read as
  its own description suggested it ranked an English-only accent-stripping
  tokenizer as the most multilingual of the nine. It reached
  `analysis_results_full.json` only, so no published value changes. The prose
  corpus is checked for batch-length mismatch, which it was not before.
  Running `tokenizer-sanity-check` twice over the same tokenizers now writes the
  same file: `get_vocab()` returns its dict in an order seeded per process, and
  three checks drew their `examples` from it, so regenerating
  `benchmarks/open_source/sanity_results.json` changed 8 example lists, 7 of
  them to a different set of tokens. No severity, observed value or summary
  count changes. `SANITY_MAX_UNREPRESENTABLE_BYTES_WARN` is removed: no check
  read it, and it was published under `metadata.thresholds` as though it
  governed a result. `--per-language-plots` writes
  `tokenizer_fairness_gini_per_language.svg`, which it never wrote before, and
  the combined per-language figure gains its Gini panel; both read the Gini's
  per-language costs under the name the metric emits them under.
  `docs/OUTPUT.md` now lists which fields the slimmed file omits and where each
  omitted value is still readable. `docs/SANITY_CHECKS.md` and
  `docs/VISUALIZATION.md` are new. The README cites the paper,
  [arXiv:2608.18062](https://arxiv.org/abs/2608.18062), in place of the previous
  software entry. A mistake in a flag or a config file prints one line and exits
  1 instead of a traceback: eleven user errors raised a bare `ValueError`, which
  `main()` deliberately lets through with its stack, and a malformed config
  reported a line and a column without naming the file. `--quiet` logs warnings
  and errors to the console while `tokenizer_analysis.log` keeps everything, and
  that file is written at all for the first time: `setup_environment()`
  configured logging first, so the file handler was built and dropped on every
  run and the log was left empty. `reconstruction_fidelity` encodes code and
  math snippets with `encode()` rather than `encode_with_offsets()`, since it
  reads only the ids; measured at 2.16x the cost of encoding for the
  `script_bpe` wrappers, and free for the rest.
- **1.0.2** Fifteen fields that published a number where nothing was measured
  now publish `null`. `analysis_results.json` became a strict projection of the
  full file. `run_metadata` gained a timestamp, per-tokenizer Hub revisions and
  a corpus digest. Two sanity checks stopped reporting a passing score for
  checks that ran on nothing, and one could turn a FAIL into a PASS.
  `operator_isolation_rate` costs less to compute. The phase that holds it and
  the three digit metrics runs in 49.5 s against 454.8 s for the nine
  tokenizers of `benchmarks/open_source/`, and scoring one tokenizer's
  operators over 104.3 M characters of web text takes 16.3 s against 330.0 s.
  Its code and math domains are encoded in one batch per language rather than
  one call per text. Every published value is unchanged.
- **1.0.1** A LaTeX table had an empty column. Tree-sitter `ERROR` spans
  were scored as AST leaves. The verbose printers rendered a `null` as `0.000`
  or raised. A directory of `vocab.json` and `merges.txt` could not load.
- **1.0.0** First release of the consolidated repository.
