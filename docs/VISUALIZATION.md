# Visualization

Three separate tools live under `tokenizer_analysis/visualization/` and
`tokenizer_analysis/cli/`. They do not share an output format or an entry point,
so this page treats them separately.

| Tool | What it produces | How it is invoked |
|---|---|---|
| Metric plots | One SVG chart per metric from a completed analysis | Flags on `tokenizer-analysis` |
| `tokenizer-visualize` | Token boundaries drawn on source text, to a terminal | Its own command |
| LaTeX tables | `.tex` tables from a completed analysis | Flags on `tokenizer-analysis` |

See [../README.md](../README.md) for the commands themselves,
[CONFIGURATION.md](CONFIGURATION.md) for the full flag reference,
[OUTPUT.md](OUTPUT.md) for where files land, and
[SANITY_CHECKS.md](SANITY_CHECKS.md) for the health checker.

## Contents

- [Metric plots](#metric-plots)
- [tokenizer-visualize](#tokenizer-visualize)
- [LaTeX tables](#latex-tables)

## Metric plots

`tokenizer-analysis` writes plots unless `--no-plots` is passed. Everything goes
through one function, `generate_all_plots` at
`tokenizer_analysis/visualization/plots.py:608`, reached from
`main.py:363` by way of `visualization/plotter.py`. The other functions in
`plots.py` are importable but no command calls them directly.

### What is written

Nine files land in `--output-dir` itself:

| File | Metric |
|---|---|
| `fertility_individual.svg` | `fertility` |
| `vocabulary_utilization_individual.svg` | `vocabulary_utilization` |
| `vocab_util_cross_lingual_cov_individual.svg` | the cross-lingual CoV of that ratio |
| `compression_rate_individual.svg` | `compression_rate` |
| `bigram_entropy_individual.svg` | `bigram_entropy` |
| `tokenizer_fairness_gini_individual.svg` | `tokenizer_fairness_gini` |
| `lorenz_curves_individual.svg` | the Lorenz curve behind that coefficient |
| `morphscore_individual.svg` | `morphscore`, only when the results hold that block |
| `utf8_integrity.svg` | `utf8_token_integrity` |

`plot_morphscore` is called on every run and returns without writing when the
results carry no `morphscore` block, which is why that one file is conditional
on `--morphscore` or `--morphscore-config` having been passed.

`--per-language-plots` adds a `per-language/` subdirectory holding
`per_language_combined_subplots.svg` plus one file per metric:
`fertility_per_language.svg`, `compression_rate_per_language.svg`,
`vocabulary_utilization_per_language.svg`,
`tokenizer_fairness_gini_per_language.svg` and
`bigram_entropy_per_language.svg`.

`--faceted-plots` adds `faceted_plots/{metric}_faceted.svg`, one subplot per
tokenizer on a shared y-axis, for the four metrics `fertility`,
`compression_rate`, `vocabulary_utilization` and `bigram_entropy`, and only for
those present in the results.

`--run-grouped-analysis` adds `grouped_plots/{group_type}_{metric}_individual.png`
for the four metrics `fertility`, `vocabulary_utilization`, `compression_rate`
and `morphscore`. These are the only PNG files; everything else is SVG.

### Direction-of-better arrows

Plot titles carry an arrow saying which direction is better, from
`METRIC_BETTER_DIRECTION` at `plots.py:98-113`. Fourteen metrics are in it:

| Higher is better | Lower is better |
|---|---|
| `compression_rate`, `vocabulary_utilization`, `bigram_entropy`, `trigram_entropy`, `shannon_entropy`, `renyi_efficiency`, `morphscore_recall`, `morphscore_precision`, `utf8_token_integrity` | `fertility`, `vocab_util_cross_lingual_cov`, `tokenizer_fairness_gini`, `unk_percentage`, `utf8_char_split` |

A metric absent from that map gets a title with no arrow: `_arrow_suffix`
returns an empty string rather than guessing a direction. Adding a plot for a
new metric without adding it here produces a chart that renders correctly and
says nothing about which end of it is good.

### Flags

| Flag | Effect |
|---|---|
| `--no-plots` | Write no plots at all. `generate_all_plots` is not called |
| `--per-language-plots` | Add the `per-language/` subdirectory above |
| `--faceted-plots` | Add the `faceted_plots/` subdirectory above |
| `--no-global-lines` | Drop the dashed global-average reference line from the per-language and faceted charts |

`--faceted-plots` and `--per-language-plots` are independent: each adds its own
subdirectory and neither requires the other. Neither reaches grouped analysis.
`visualization/plotter.plot_grouped_analysis` passes both as `False`, and it
passes an empty results dict alongside the grouped results, so those two
branches would find nothing to draw even if the flags were forwarded.

## tokenizer-visualize

`tokenizer_analysis/cli/visualize_tokenization.py`. Prints the token boundaries
a tokenizer draws on a piece of text, which is how to see what a fertility
number is made of.

```bash
uv run tokenizer-visualize --tokenizer-config configs/sample_tokenizers.json
uv run tokenizer-visualize --tokenizer-config configs/sample_tokenizers.json \
    --input mycode.py --samples-per-file 3 --no-color > boundaries.txt
```

| Flag | Default | What it does |
|---|---|---|
| `--tokenizer-config FILE` | required | Tokenizer config, the same format `tokenizer-analysis` takes |
| `--tokenizers NAME [NAME ...]` | all of them | Restrict to a subset of the names in that config |
| `--input PATH` | none | A text file, or a directory read non-recursively in name order |
| `--code-file PATH` | none | Alias for `--input` with a single file, kept for compatibility |
| `--samples-per-file N` | 1 | How many samples to read from each file |
| `--no-color` | off | Turn off ANSI colour |

### Output

Everything goes to stdout; the command writes no files. Each sample's source
text is printed once for reference, then one block per tokenizer giving the
token count and the text with token boundaries marked.

Colour is on only when `--no-color` was not passed **and** stdout is a terminal
(`visualize_tokenization.py:616`), so redirecting to a file gives plain text
without the flag. Token backgrounds cycle through six colours so adjacent
tokens are distinguishable; a token that splits a character mid-byte is drawn in
a fixed red instead. Whitespace is printed visibly: a space as `·`, a tab as
`→`, a newline as `↵`, a carriage return as `␍`. The count of tokenizers loaded
goes to stderr rather than stdout, so it does not land in a redirected file.

### Input format

Within one file, samples are separated by a line containing only `---`.
`--samples-per-file` caps how many are taken from each file, and its default of
1 means only the text before the first separator.

With no `--input` and no `--code-file`, three built-in samples are used: a short
Python file, a page of Unicode mathematics (limits, Euler's identity, the Basel
problem, a Gaussian integral, set theory, logic), and one sentence in 15
languages and scripts.

### Exit codes

Exit 1 in four cases: the `--input` path does not exist; the path exists but
holds no text sample; a name given to `--tokenizers` is not in the config; every
requested tokenizer failed to load. That last one is deliberate, so that a wrong
config path does not print nothing and exit 0. Otherwise it exits 0.

## LaTeX tables

`tokenizer_analysis/visualization/latex_tables.py`, driven by four flags on
`tokenizer-analysis`. [CONFIGURATION.md](CONFIGURATION.md#cli-reference) lists
the flags and [OUTPUT.md](OUTPUT.md#output-structure) explains why
`--latex-output-dir` has two different defaults; this section covers what the
tables hold.

### Table types

`--latex-table-types` takes any of `basic`, `information`, `comprehensive` and
`merge_usage`, and defaults to `basic comprehensive merge_usage`.

- **`basic`**: eight columns, `compression_rate`, `type_token_ratio`,
  `vocabulary_utilization`, `fertility`, `renyi_2.5`, `tokenizer_fairness_gini`,
  `morphscore_precision`, `morphscore_recall`.
- **`information`**: two columns, `renyi_1.0` and `avg_token_rank`.
- **`comprehensive`**: every metric in the registry for which at least one
  tokenizer in the run has a value.
- **`merge_usage`**: five columns, `merge_active`, `merge_rare`,
  `merge_uncommon`, `merge_unseen` and `merge_scaffold`, the percentage of merge
  tokens in each category of `merge_vocabulary_usage`. These columns have no
  arrow and no bolded value, because none of the five percentages has a better
  direction.

A fourth type, `morphological`, was removed in 1.0.0. Passing it is rejected by
name with a message saying what to use instead, rather than with a generic
invalid-choice error.

### Row ids

The keys of `LaTeXTableGenerator.metric_configs` are the registry's own
identifiers. There are 32, and 16 of them match neither the metric they read nor
any field name in their own entry: `morphscore_recall`, `morphscore_precision`,
`avg_token_rank`, `avg_tokens_per_line`, `three_digit_boundary_f1`,
`operator_isolation`, `ast_full_alignment`, `ident_fragmentation`,
`indent_depth_corr`, `utf8_boundary_crossing`, `utf8_char_split`,
`merge_active`, `merge_rare`, `merge_uncommon`, `merge_unseen` and
`merge_scaffold`, at `latex_tables.py` lines 102, 111, 122, 142, 171, 180, 190,
199, 208, 218, 231, 242, 252, 262, 272 and 282. To find what one reads, follow `key_path` and `value_key` in its entry:
`operator_isolation` reads `overall_isolation_rate` from the
`operator_isolation_rate` block, and `utf8_char_split` reads `char_split` then
`global` from the `utf8_token_integrity` block.

A row id is what a `--custom-latex-config` table lists under `metrics`.
`--latex-table-types` takes `basic`, `information`, `comprehensive` or
`merge_usage`, not row ids. `vocab_util_cross_lingual_cov` is a third thing again, a plot filename.

Those `key_path` and `value_key` entries name the raw results the analyser holds
in memory, which is what `tokenizer-analysis` passes the generator. They are not
the key paths of `analysis_results.json`, which `normalize_results` renames and
pivots before writing. Building tables from a saved results file therefore
leaves 11 of the 24 rows as `---`, with only a log line from
`_warn_if_block_present_but_unresolved` to say why. No command does that; it is
a hazard for code that loads the published file and calls the generator itself.

`LaTeXTableGenerator.__init__` calls `_validate_registry` (line 286), which
raises if a registry entry is rooted at one of the six metrics
`merge_redundant_metrics` deletes from the top level of every results dict. Such
an entry would otherwise render `---` in every row of every table, which reads
as a metric that was not computed rather than as a stale registry.

### Custom tables

`--custom-latex-config` takes a JSON object mapping a table name to a table
definition:

```json
{
  "fairness": {
    "metrics": ["tokenizer_fairness_gini", "fertility", "compression_rate"],
    "caption": "Cross-language fairness and cost",
    "label": "tab:fairness"
  }
}
```

`metrics` is required and each entry has to be a row id from the registry above.
`caption` defaults to `Custom Table: <name>` and `label` to `tab:custom_<name>`.
Each table is written to `custom_<name>_table.tex`. A table whose definition is
not an object, or whose `metrics` list is empty, is skipped with a warning and
the rest are still written.
