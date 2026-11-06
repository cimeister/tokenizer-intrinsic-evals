# TokEval

A toolkit for computing intrinsic quality metrics for tokenizers across natural
language, code and math.

The metrics, and the pretraining experiments that check which of them predict
downstream behaviour, are described in
[TokEval: A Tokenizer Evaluation Suite](https://arxiv.org/abs/2608.18062)
(COLM 2026).

Installing the package puts three commands on the path:
`tokenizer-analysis`, `tokenizer-visualize` and `tokenizer-sanity-check`.

This supersedes `tokenizer-analysis-suite`. The install name changed to
`tokenizer-intrinsic-evals`; the import name is still `tokenizer_analysis`. The
prior suite is preserved on the `legacy-suite` branch and the
`legacy-suite-final` tag. Several metrics changed what they compute, so numbers
from the old suite are not comparable with numbers from this one:
[CHANGELOG.md](CHANGELOG.md) lists every such change with its measured effect.

## Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [The metrics](#the-metrics)
- [Running on your own data](#running-on-your-own-data)
- [Reading the output](#reading-the-output)
- [`tokenizer-visualize` and `tokenizer-sanity-check`](#tokenizer-visualize-and-tokenizer-sanity-check)
- [Results for nine open-source tokenizers](#results-for-nine-open-source-tokenizers)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Documents](#documents)

Reference material is in [docs/](docs/): [METRICS.md](docs/METRICS.md) defines
every metric, [CONFIGURATION.md](docs/CONFIGURATION.md) is the flag and config
reference, [OUTPUT.md](docs/OUTPUT.md) describes the results file,
[SANITY_CHECKS.md](docs/SANITY_CHECKS.md) documents the 16 tokenizer health
checks, [VISUALIZATION.md](docs/VISUALIZATION.md) covers the plotting and LaTeX
tools, and [EXTENDING.md](docs/EXTENDING.md) covers the Python API and adding
your own tokenizers and metrics.

## Install

Python 3.10 or newer, tested on Linux. The commands below use
[uv](https://docs.astral.sh/uv/getting-started/installation/); install that
first if it is not already on your path.

```bash
git clone https://github.com/cimeister/tokenizer-intrinsic-evals.git
cd tokenizer-intrinsic-evals
uv sync
```

`uv run` puts the console scripts on the path; activate the venv instead if you
prefer, and drop the prefix.

Not published to PyPI. To use it as a dependency rather than working in the
checkout, `pip install git+https://github.com/cimeister/tokenizer-intrinsic-evals.git`.
The demo data and the example configs are not in the installable package.
Use a checkout for those.

MorphScore, parquet input and the FLORES+ fetcher are optional extras with an
install order that matters: see
[Optional extras](docs/CONFIGURATION.md#optional-extras).

## Quick Start

Compare two tokenizers on one corpus. `gpt2` and `xlm-roberta-base` download
from the Hugging Face Hub on first use; both are public.

```bash
cat > corpus.txt <<'CORPUS'
The quick brown fox jumps over the lazy dog.
नई दिल्ली भारत की राजधानी है।
def compute_total(items): return sum(i.price for i in items)
The invoice totalled 1234567 euros on 2024-03-15.
CORPUS

cat > my_tokenizers.json <<'TOKENIZERS'
{
  "gpt2":  {"class": "huggingface", "path": "gpt2"},
  "xlm-r": {"class": "huggingface", "path": "xlm-roberta-base"}
}
TOKENIZERS

uv run tokenizer-analysis --tokenizer-config my_tokenizers.json --input corpus.txt
```

Results are written to `results/analysis_results.json`, with plots beside it.

## The metrics

Nineteen metrics. One line each below; the definition, worked example and
caveats for each are in [docs/METRICS.md](docs/METRICS.md), and the exact key
path for each headline value is in
[Metric names and results keys](docs/METRICS.md#metric-names-and-results-keys).

| Metric | Measures | Better |
|---|---|---|
| `compression_rate` | text units per token | higher |
| `fertility` | tokens per word or character | lower |
| `token_length` | mean token size in bytes and characters | neither |
| `vocabulary_utilization` | share of the declared vocabulary the corpus used | higher |
| `renyi_efficiency` | how evenly token probability mass is spread | higher |
| `bigram_entropy` | how predictable each token's successor is | higher |
| `trigram_entropy` | the same over a two-token context | higher |
| `morphscore` | whether token boundaries fall on morpheme boundaries | higher |
| `three_digit_boundary_alignment` | whether numbers split on place-value boundaries | higher |
| `numeric_magnitude_consistency` | whether tokens per digit stay stable across magnitudes | lower |
| `operator_isolation_rate` | whether operators are their own tokens | higher |
| `reconstruction_fidelity` | whether decoding restores the input exactly | higher |
| `utf8_token_integrity` | whether tokens are complete UTF-8 sequences | higher |
| `merge_vocabulary_usage` | how often BPE merge tokens are emitted, and how often they are only built on the way to longer tokens | neither |
| `ast_boundary_alignment` | whether token boundaries fall on syntax-tree leaves | higher |
| `identifier_fragmentation` | how many tokens an identifier costs | lower |
| `indentation_consistency` | whether whitespace tokens scale with indent depth | higher |
| `tokenizer_fairness_gini` | how unevenly encoding cost falls across languages | lower |
| `encoding_speed` | wall-clock cost of the run, not a quality measure | lower |

"Better" gives the direction for the quantity as defined. "neither" marks a
metric that describes a tokenizer without ranking it.

## Running on your own data

`--input` takes one file (`.txt`, `.json`, `.jsonl`, `.parquet`) or a directory
of them. A `.txt` file is one document per line. For several corpora at once,
and for the cross-lingual metrics, use `--language-config` instead. Both are
described in [Data Configuration](docs/CONFIGURATION.md#data-configuration).

A tokenizer `path` is either a Hub model id or a local `tokenizer.json`. Ten
tokenizer classes are supported; see
[Tokenizer Configuration](docs/CONFIGURATION.md#tokenizer-configuration).

### Settings that change what is measured

The Quick Start command uses the default code corpus, the default math corpus
and the default character error rate budget. Each of the three changes what the
metrics are computed on, so numbers from that command are not comparable with
numbers from the command below.

```bash
uv run tokenizer-analysis \
    --tokenizer-config my_tokenizers.json \
    --language-config my_languages.json \
    --code-ast-config my_code.json \
    --use-builtin-math-data \
    --cer-time-budget 0
```

- **`--code-ast-config`** supplies real source code. Without it the code
  metrics run on built-in synthetic samples and the run warns. The two are not
  comparable.
- **`--use-builtin-math-data`**, or `--math-data FILE`, points the three digit
  metrics at a corpus with long numbers in it. Without either they run on
  whatever numbers the prose happens to contain, which on the bundled corpus is
  1797 digit spans of which 6 exceed 4 digits, so the place-value boundaries at
  L-6 and L-9 are almost never exercised.
- **`--cer-time-budget 0`** removes the cap on the character error rate, so
  `mean_cer` and `whitespace_fidelity` are measured rather than reported as
  `null` for slow tokenizers.

### The evaluation corpus

The configs in `configs/` and the `--use-sample-data` demo read
`parallel/<iso639-3>_<Script>.txt`. Those files are not in this repository.
They come from FLORES+, which is CC-BY-SA 4.0 and is fetched rather than
redistributed here. Cite NLLB Team et al., Nature 630 (2024) if you use it.

```bash
uv pip install datasets            # or: uv sync --extra flores
hf auth login                      # FLORES+ is gated, approval is automatic
uv run python scripts/fetch_flores.py                              # 13 languages
uv run python scripts/fetch_flores.py --all                        # every language
```

A run that names a missing file aborts with an error that prints the fetch
command above. It never proceeds on a smaller corpus than the config declares. Nothing in the library
requires FLORES+: `--input` and `--language-config` take your own corpus.

## Reading the output

```
results/
├── analysis_results.json       # every metric, per tokenizer and per language
├── analysis_results_full.json  # with --save-full-results
└── *.svg                       # one chart per metric, unless --no-plots
```

Every metric is written as `<metric>.per_tokenizer.<tok>.global` for the
headline value and `.per_language.<lang>` for the breakdown, with
`metadata.aggregation` naming which average the `global` is. A value that could
not be computed is `null`, never `0.0`.

[docs/OUTPUT.md](docs/OUTPUT.md) describes the schema, the aggregation labels,
the null convention and the `run_metadata` provenance block.

## `tokenizer-visualize` and `tokenizer-sanity-check`

**`tokenizer-visualize`** renders token boundaries on source text, for
inspecting how tokenizers split code, math and multilingual content.

```bash
uv run tokenizer-visualize --tokenizer-config configs/sample_tokenizers.json
```

**`tokenizer-sanity-check`** runs 16 checks against one tokenizer: byte
coverage, whitespace and digit handling, special tokens, determinism, Unicode
normalization, vocabulary integrity and reachability. A failing check sets a
non-zero exit code, so it can gate a tokenizer before a full run.

```bash
uv run tokenizer-sanity-check huggingface:tokenizers/bpe.json
```

Both bundled demo tokenizers fail this check on purpose: `bpe.json` and
`unigramlm.json` each cover 94 of 256 byte values. `unigramlm.json` also logs a
warning that it cannot report its declared special tokens, so the check falls
back to the 13 generic ones. That follows from the same file and is
expected.

[docs/SANITY_CHECKS.md](docs/SANITY_CHECKS.md) documents the 16 checks, the
five severities and the exit codes. [docs/VISUALIZATION.md](docs/VISUALIZATION.md)
covers `tokenizer-visualize`, the metric plots and the LaTeX table generator.

## Results for nine open-source tokenizers

[benchmarks/open_source/REPORT.md](benchmarks/open_source/REPORT.md) has the
full metric set for nine open-source tokenizers on 13 FLORES+ languages of
translated news, 1500 source files across 15 programming languages, and the 285
bundled math expressions. Three of its results:

Pooled over the 13 languages, bytes per token ranges from 2.036 for GPT-2 to
5.067 for XLM-RoBERTa base. On English alone the order reverses: GPT-2 is
highest at 4.805 and XLM-RoBERTa base lowest at 4.263. Vocabulary size is not
held constant across the nine: it correlates with bytes per token at Spearman
0.883.

The Gini coefficient over per-byte costs and the Gini coefficient over per-line
costs correlate at Spearman 0.650 over the nine, and the two orderings differ at
the top. Llama 3 has the lowest per-byte coefficient, 0.0772, and a per-line
coefficient of 0.0926. XLM-RoBERTa base has the lowest per-line coefficient,
0.0494, and a per-byte coefficient of 0.0976. The per-line coefficient is
published only when every language has the same line count, which is the case for
FLORES+.

An operator isolation rate of 1.000 does not mean compound operators are kept
whole. BERT base uncased has an isolation rate of 1.000 and a compound
preservation rate of 0.000 over 88,398 compound operators such as `==` and
`->`, where the other eight range from 0.844 to 0.997.

```bash
bash benchmarks/open_source/run.sh
```

That regenerates the report and the results file beside it. The results file
records the package version, the commit, and a hash of the configs and the
FLORES+ files.

## Troubleshooting

**`No module named 'morphscore'`**: initialize submodules, then install
MorphScore into the project environment:
`git submodule update --init --recursive && uv pip install -e ./morphscore`. If
it worked before and stopped, a later `uv sync` removed it; see
[Optional extras](docs/CONFIGURATION.md#optional-extras).

**`Unknown tokenizer class`**: the available classes are listed under
[Tokenizer Configuration](docs/CONFIGURATION.md#tokenizer-configuration), plus
any you register with `register_tokenizer_class()`.

**`FileNotFoundError`**: a relative `data_path` in a language config resolves
against the package root, while `--input` and a relative tokenizer `path`
resolve against your working directory. See
[How `data_path` is resolved](docs/CONFIGURATION.md#how-data_path-is-resolved).

**`Cannot read the parquet file`**: `uv sync --extra parquet`, naming any other
extras in the same command.

**`_tkinter.TclError: no display name`**: `export MPLBACKEND=Agg` before
running on a headless server.

**A code language is missing from the results**: three of the 19 configured
languages are excluded by design, and a grammar that crashes or exceeds
`TOKEVAL_PARSE_TIMEOUT_S` is reported as unmeasured and named in the log. See
[docs/METRICS.md](docs/METRICS.md#code-tokenization-metrics).

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Ensure the existing tests pass: `uv run pytest tokenizer_analysis/tests -q`.
4. Submit a pull request.

Adding a tokenizer class, adding a metric, and the repository layout are
covered in [docs/EXTENDING.md](docs/EXTENDING.md).

## Documents

| File | Contents |
|---|---|
| [docs/METRICS.md](docs/METRICS.md) | The definition, worked example and caveats for every metric |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Every CLI flag and every config file format |
| [docs/OUTPUT.md](docs/OUTPUT.md) | The results file: schema, aggregation, null convention, provenance |
| [docs/SANITY_CHECKS.md](docs/SANITY_CHECKS.md) | The 16 tokenizer health checks, the five severities, the exit codes |
| [docs/VISUALIZATION.md](docs/VISUALIZATION.md) | Metric plots, `tokenizer-visualize`, the LaTeX table generator |
| [docs/EXTENDING.md](docs/EXTENDING.md) | Python API, repository layout, adding tokenizers and metrics |
| [benchmarks/open_source/REPORT.md](benchmarks/open_source/REPORT.md) | Nine open-source tokenizers measured on the full metric set |
| [CHANGELOG.md](CHANGELOG.md) | Changes from `tokenizer-analysis-suite`, breaking ones first |
| [CITATION.cff](CITATION.cff) | Machine-readable citation metadata |

## License

MIT. See [LICENSE](LICENSE). The evaluation corpora and the tokenizers named in
the example configs are downloaded at run time under their own licenses and are
not redistributed here.

## Citation

```bibtex
@inproceedings{meister2026tokeval,
  title         = {TokEval: A Tokenizer Evaluation Suite},
  author        = {Meister, Clara},
  booktitle     = {Conference on Language Modeling},
  year          = {2026},
  eprint        = {2608.18062},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2608.18062}
}
```
