# Changelog

All notable changes to this project are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-04

First release under the consolidated repository
`github.com/cimeister/tokenizer-intrinsic-evals`. This version supersedes the
older `tokenizer-analysis-suite`. The install (distribution) name is now
`tokenizer-intrinsic-evals` (was `tokenizer-analysis`); the import name is
still `import tokenizer_analysis`. See MIGRATION.md for a step-by-step upgrade
guide.

### Added
- New metric families: bigram and trigram successor entropy
  (`bigram_entropy`, `trigram_entropy`); math digit-boundary metrics
  (three-digit alignment, digit-split variability, magnitude consistency,
  operator isolation); code AST-boundary alignment and identifier
  fragmentation (tree-sitter, 19 languages); UTF-8 token integrity and
  character-split metrics; reconstruction fidelity (exact match, CER,
  whitespace fidelity); cross-lingual vocabulary-utilization CoV
  (`vocab_util_cross_lingual_cov`); `avg_langs_per_token`; `avg_tokens_per_line`.
- New console script `tokenizer-visualize`: colour-coded token-boundary views
  over source text.
- New console script `tokenizer-sanity-check`: single-tokenizer health report
  (byte coverage, whitespace/digits, special tokens, determinism, Unicode
  normalization, vocabulary integrity and reachability), with pass/warn/fail
  severities and non-zero exit codes.
- Per-document metric outputs (`tokenizer_analysis/per_example.py`).
- Reporting: faceted plots (one subplot per tokenizer), a cumulative Markdown
  leaderboard (`--update-results-md`), and expanded LaTeX tables with
  direction arrows.
- Packaging: `LICENSE` (MIT), `NOTICE` (FLORES+ attribution), this changelog,
  and `MIGRATION.md`.

### Changed
- Install (distribution) name renamed from `tokenizer-analysis` to
  `tokenizer-intrinsic-evals`, matching the repository. The import name is
  unchanged (`import tokenizer_analysis`), as are the console scripts
  (`tokenizer-analysis`, `tokenizer-visualize`, `tokenizer-sanity-check`).
  Because the distribution name changed, `pip install --upgrade
  tokenizer-analysis` will not find this release; install
  `tokenizer-intrinsic-evals` instead.
- Minimum Python raised from 3.8 to 3.10.
- Results JSON: the per-tokenizer compression key is `compression_rate`
  (previously `compression_ratio`); the slim `analysis_results.json` is now
  organized as `{per_tokenizer: {global, per_language}}`.
- Constants moved from namespace classes (`TextProcessing`, `DataProcessing`,
  `Statistics`, `Validation`, ...) to module-level names in
  `tokenizer_analysis/constants.py`.
- `text_measurement` configs now reject unknown keys with a clear error naming
  the offending key, instead of raising an opaque `TypeError`.
- Packaging moved to `pyproject.toml` + `uv.lock` (hatchling). tree-sitter
  support is a core dependency; parquet reading is the optional `parquet` extra.

### Removed
- The standalone morphological boundary metric, its module
  (`metrics/morphological.py`), its loader (`loaders/morphological.py`), the
  `MorphologicalMetrics` / `MorphologicalDataLoader` exports, the
  `--morphological-config` flag, and the `morphological` LaTeX table type. Use
  MorphScore (`--morphscore` / `--morphscore-config`) instead. The removed flag
  and table type now fail with a message pointing to MorphScore.
- The results-branch publishing workflow (`scripts/update_remote.py`).
- Bundled OpenAI tiktoken vocabulary JSONs
  (`tokenizers/gpt_4_hf.json`, `tokenizers/gpt_4o_hf.json`); load those via
  `tiktoken` at run time instead.
- Apertus-specific research artifacts (`apertus_tokenizer_design.md`, the
  `results/` reports and figures) and configs referencing untracked cluster
  data. The prior suite state is preserved on the `legacy-suite` branch and the
  `legacy-suite-final` tag.

### Fixed
- README documented `text_measurement` keys that did not match the code (for
  example `line_counting_method`); the documented example now loads.
- README quick-start pointed at `results/fertility.png`; individual plots are
  written as `results/fertility_individual.svg`.
