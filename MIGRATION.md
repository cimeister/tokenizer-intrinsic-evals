# Migrating from tokenizer-analysis-suite to 1.0.0

This project was consolidated into
`github.com/cimeister/tokenizer-intrinsic-evals` and released as 1.0.0. The
import name is unchanged (`import tokenizer_analysis`), but the install
(distribution) name changed. This guide lists the breaking changes and their
replacements.

## Repository and install

- The install name is now `tokenizer-intrinsic-evals` (was `tokenizer-analysis`).
  `pip install --upgrade tokenizer-analysis` will not find this release; install
  `tokenizer-intrinsic-evals` instead. The console scripts are unchanged
  (`tokenizer-analysis`, `tokenizer-visualize`, `tokenizer-sanity-check`), as is
  `import tokenizer_analysis`.
- The old repository was renamed, so existing clones keep working: `git pull`
  fast-forwards, and the old URL redirects for both web and git. To point a
  remote at the new name explicitly:
  `git remote set-url origin https://github.com/cimeister/tokenizer-intrinsic-evals.git`.
- Minimum Python is now 3.10 (was 3.8).
- tree-sitter (code AST metrics) is a core dependency. Parquet reading is now an
  optional extra: `uv sync --extra parquet`.

## Command-line changes

| Removed | Replacement |
|---------|-------------|
| `--morphological-config FILE` | `--morphscore` (defaults) or `--morphscore-config FILE` |
| `--latex-table-types morphological` | Use MorphScore; valid types are `basic`, `information`, `comprehensive` |

Both removed options now exit with a message pointing to MorphScore rather than
a generic argparse error.

## Results / output schema

- The per-tokenizer compression key is now `compression_rate` (was
  `compression_ratio`). Update any downstream parser that reads the old key.
- The slim `analysis_results.json` is now organized as
  `{per_tokenizer: {global, per_language}}`. If you consumed the old flat
  layout (or its `summary` / `pairwise_comparisons` blocks), read the new
  structure, or pass `--save-full-results` for the detailed output.

## Python API

- `MorphologicalMetrics` and `MorphologicalDataLoader` were removed from
  `tokenizer_analysis`. Use MorphScore instead.
- `UnifiedTokenizerAnalyzer(...)` no longer accepts `morphological_config`, and
  `run_analysis(...)` no longer accepts `include_morphological`.
- Constants moved from namespace classes to module-level names. Replace imports
  like `from tokenizer_analysis.constants import DataProcessing` (then
  `DataProcessing.DEFAULT_CHUNK_SIZE`) with the module-level constant
  (`from tokenizer_analysis.constants import DEFAULT_CHUNK_SIZE`).

## Configuration

- `text_measurement` configs now reject unknown keys. The correct keys are
  `method`, `byte_counting`, `word_counting`, `line_counting`, `custom_regex`,
  and `include_empty_splits`. Note earlier docs used names such as
  `line_counting_method` and `include_empty_lines`, which were never valid keys
  in the code; update those configs. See the README "Text Measurement
  Configuration" table for the valid values.

## Data and artifacts

- The bundled OpenAI tiktoken vocabulary JSONs were removed. Load GPT-4 /
  GPT-4o tokenizers via `tiktoken` (a core dependency) at run time instead.
- Apertus-specific reports and design docs were removed from the tracked tree.
  The complete prior suite state (including the `PA_BPE_tokenizers/` directory)
  is preserved on the `legacy-suite` branch and the `legacy-suite-final` tag.
