# TokEval

A framework for evaluating tokenizers across languages with comprehensive metrics and multilingual fairness analysis.

## Features

- **Multi-tokenizer comparison**: Compare any number of tokenizers simultaneously
- **Comprehensive metrics**: Basic tokenization, information-theoretic, morphological, and fairness metrics
- **Multilingual fairness**: Gini coefficient analysis for cross-language compression equity
- **MorphScore integration**: Carefully crafted morphological analysis ([Arnett et. al. 2025](https://arxiv.org/abs/2507.06378))
- **Flexible data inputs**: Raw texts, pre-tokenized data, JSON, and parquet files
- **Academic outputs**: LaTeX table generation and publication-ready plots
- **Performance optimized**: Efficient analysis with pre-tokenized data caching

## Quick Start

Get up and running in 30 seconds:

```bash
# Clone and install
git clone https://github.com/cimeister/tokenizer-analysis-suite.git
cd tokenizer-analysis-suite
pip install -e .

# Run demo analysis with built-in sample data
python scripts/run_tokenizer_analysis.py --use-sample-data

# View results 
open results/fertility.png  # Basic metric comparison chart
```

This will analyze two sample tokenizers (BPE and Unigram) across 5 languages and generate comparison plots.

## Setup

### Requirements
- Python 3.8+ 
- Git (for submodules)

### Full Installation (with submodules)
```bash
git clone https://github.com/cimeister/tokenizer-analysis-suite.git
cd tokenizer-analysis-suite
pip install -e .

# Install morphological analysis module
git submodule update --init --recursive
cd morphscore && pip install -e . && cd ..
```

**Note on MorphScore Integration**: Full integration with the MorphScore library is still a work in progress. Current limitations:
- Only `<ISO 639-3>_<script>` language codes are automatically mapped to data files
- MorphScore datasets must be downloaded separately (see [MorphScore README](morphscore/README.md) for links to dataset download)
- Data files should be placed in a `morphscore_data/` directory, or their location specified in a configuration file (see Configuration Files section)

## Usage Examples

### Basic Analysis
```bash
# Quick start with sample data
python scripts/run_tokenizer_analysis.py --use-sample-data

# Custom tokenizers and languages (see Configuration Files section below)
python scripts/run_tokenizer_analysis.py \
    --tokenizer-config configs/tokenizer_config.json \
    --language-config configs/language_config.json \
    --output-dir results/
```

### Advanced Analysis Options
```bash
# Grouped analysis by script families and resource levels
python scripts/run_tokenizer_analysis.py --use-sample-data --run-grouped-analysis

# Filter by script family (generates separate analysis for Latin scripts only)
python scripts/run_tokenizer_analysis.py --use-sample-data --filter-script-family Latin

# Include morphological analysis with MorphScore
python scripts/run_tokenizer_analysis.py --use-sample-data --morphscore

# Generate per-language plots (grouped bar charts with languages on x-axis)
python scripts/run_tokenizer_analysis.py --use-sample-data --per-language-plots

# Generate faceted plots (subplots with tied y-axes)
python scripts/run_tokenizer_analysis.py --use-sample-data --faceted-plots
```

**Note**: For custom configurations, replace `--use-sample-data` with your own `--tokenizer-config` and `--language-config` files (see Configuration Files section).

### Output Options
```bash
# Generate LaTeX tables for academic papers
python scripts/run_tokenizer_analysis.py --use-sample-data --generate-latex-tables

# Save both summary and detailed results in JSON format
python scripts/run_tokenizer_analysis.py --use-sample-data --save-full-results

# Skip plot generation (analysis only, results output by default to JSON results/analysis_results_full.json)
python scripts/run_tokenizer_analysis.py --use-sample-data --no-plots
```

## Configuration Files

The framework uses JSON configuration files to specify tokenizers, data files, and analysis settings. All configuration examples are provided below.

### Tokenizer Configuration

Specify tokenizers using a JSON file with the `--tokenizer-config` flag:

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
  },
  "pretokenized": {
    "class": "pretokenized",
    "vocab_size": 50000
  }
}
```

**Available tokenizer classes:**
- `"huggingface"`: HuggingFace transformers/tokenizers library tokenizers
- `"custom_bpe"`: Custom BPE tokenizers with vocab.json and merges.txt
- `"pretokenized"`: For pre-tokenized data analysis

### Data Configuration

Specify languages and analysis groupings using a JSON file with the `--language-config` flag:

#### Basic Data Configuration
```json
{
  "languages": {
    "en": "/path/to/english/data",
    "fr": "/path/to/french/file.txt", 
    "de": "/path/to/german/corpus.json"
  }
}
```

#### Extended Configuration with Analysis Groups
```json
{
  "languages": {
    "eng_Latn": {
      "name": "English",
      "iso_code": "en",
      "data_path": "/path/to/english/data"
    },
    "fra_Latn": {
      "name": "French",
      "iso_code": "fr", 
      "data_path": "/path/to/french/data"
    },
    "arb_Arab": {
      "name": "Arabic",
      "iso_code": "ar",
      "data_path": "/path/to/arabic/data"
    },
    "rus_Cyrl": {
      "name": "Russian",
      "iso_code": "ru",
      "data_path": "/path/to/russian/data"
    }
  },
  "analysis_groups": {
    "script_family": {
      "Latin": ["eng_Latn", "fra_Latn"],
      "Arabic": ["arb_Arab"],
      "Cyrillic": ["rus_Cyrl"]
    },
    "resource_level": {
      "high": ["eng_Latn", "fra_Latn"],
      "medium": ["deu_Latn"],
      "low": ["som_Latn"]
    }
  }
}
```

### Text Measurement Configuration

Control how text "length" is measured for metric normalization using the `--measurement-config` flag:

#### Byte Counting (Default for compression metrics)
```json
{
  "method": "bytes",
  "byte_counting_method": "utf8"
}
```

**Options:**
- `"utf8"`: Standard UTF-8 encoding (default)
- `"hf_tokenizer"`: Uses HuggingFace tokenizer's pre-tokenizer for byte counting

#### Character Counting
```json
{
  "method": "characters"
}
```

#### Line Counting (Default for Gini metrics)
```json
{
  "method": "lines",
  "line_counting_method": "python_split",
  "include_empty_lines": false
}
```

**Options:**
- `"python_split"`: Uses Python's `str.splitlines()` (default)
- `"regex"`: Custom regex-based line splitting (requires `custom_regex`)

#### Word Counting (Default for fertility metrics)  
```json
{
  "method": "words",
  "word_counting_method": "whitespace",
  "include_empty_words": false
}
```

**Options:**
- `"whitespace"`: Simple whitespace splitting
- `"hf_whitespace"`: HuggingFace whitespace pre-tokenizer
- `"regex"`: Custom regex-based word splitting (requires `custom_regex`)

#### Custom Regex Patterns
```json
{
  "method": "words",
  "word_counting_method": "regex",
  "custom_regex": "\\S+",
  "include_empty_words": false
}
```

### MorphScore Configuration

**Note**: MorphScore integration is work-in-progress with current limitations.

#### Default Configuration
If MorphScore data files are placed in `morphscore_data/` directory and you use , no configuration is needed.

#### Custom Data Location Configuration
Specify custom MorphScore data location using the `--morphscore-config` flag; any other configuration parameter accepted by morphscore can also be specified in this config (e.g., 'freq_scale'). By default, MorhpScore parameters are those recommended by the authors:

```json
{
    "data_dir": "/path/to/morphscore/datasets",
    "by_split": false,
    "freq_scale": true,
    "exclude_single_tok": false
}
```
By default, MorphScore analysis is filtered to the languages given in the data configuration file (`--language-config`). This filtering requires languages be specified in particular language codes.
**Supported Language Codes**:
- ISO 639-3 with script codes (e.g., `eng_Latn`, `spa_Latn`, `arb_Arab`)

Note that you can override this behavior by setting 'language_subset' in `--morphscore-config` manually to the list of languages you'd like the MorphScore analysis performed on (needs to be given as a list of the root of the file name for per-language files in `morphscore_data/`). This bypasses the need to use ISO 639-3 codes.

**Data Requirements**:
- Download MorphScore datasets from links in [MorphScore README](morphscore/README.md)
- Ensure language codes in your language config match supported mappings

### Pre-tokenized Data

#### Generating Pre-tokenized Data
```bash
# Save tokenized data for reuse (automatically creates config and vocab files)
python scripts/run_tokenizer_analysis.py --use-sample-data \
    --save-tokenized-data --tokenized-data-output-path my_data.pkl

# Use pre-tokenized data with auto-generated files
python scripts/run_tokenizer_analysis.py \
    --tokenized-data-file my_data.pkl \
    --tokenized-data-config my_data_config.json

# Optional: Add language config, e.g., `--language-config configs/language_config.json` for custom analysis groupings  

```

#### Manual Pre-tokenized Data Setup
If you have pre-tokenized data not generated by this framework, you need to provide:

**Required Files:**
1. **Tokenized data file** (`.pkl` or `.json`): Dictionary mapping tokenizer names to lists of TokenizedData objects:
   ```python
   # Pickle format (Dict[str, List[TokenizedData]])
   {
     "my_tokenizer": [
       {
         "tokenizer_name": "my_tokenizer",
         "language": "eng_Latn", 
         "tokens": [101, 7592, 2003, 1037, 6509, 102],
         "text": "This is a sample sentence.",
         "metadata": {"source": "my_dataset"}
       },
       {
         "tokenizer_name": "my_tokenizer",
         "language": "spa_Latn",
         "tokens": [101, 2023, 2003, 102], 
         "text": "Esta es una oración.",
         "metadata": {"source": "my_dataset"}
       }
     ]
   }
   ```

2. **Tokenized data config file** (`.json`): Configuration for vocabulary information:
   ```json
   {
     "vocabulary_files": {
       "my_tokenizer": "my_tokenizer_vocab.txt"
     }
   }
   ```

3. **Vocabulary files** (`<tokenizer_name>_vocab.txt`): Line-by-line vocabulary files:
   ```text
   <pad>
   <unk>
   <s>
   </s>
   the
   is
   ...
   ```

**Optional Files:**
- **Language config file** (`.json`): Only needed for custom analysis groupings by script family, resource level, etc. The `data_path` field is not used since texts are already included in the tokenized data.

### Output Structure
```
results/
├── fertility.png              # Tokens per word/character
├── compression_rate.png       # Text compression efficiency
├── vocabulary_utilization.png # Vocabulary usage
├── <metric name>.png          # Other supported metrics
├── grouped_plots/                  # Cross-tokenizer comparisons
│   ├── script_family_comparison.png
│   └── resource_level_analysis.png
├── per-language/                   # Language-specific analysis
│   ├── fertility_by_language.png
│   └── compression_by_language.png
├── latex_tables/                   # Academic publication tables
│   ├── basic_metrics.tex
│   └── comprehensive_analysis.tex
├── analysis_results.json   # Key metrics summary
├── analysis_results_full.json     # Detailed results (with --save-full-results)
├── tokenized_data.pkl       # Pre-tokenized data (with --save-tokenized-data)
├── tokenized_data_config.json     # Auto-generated config (with --save-tokenized-data)
├── <tokenizer_name>_vocab.txt     # Auto-generated vocab files (with --save-tokenized-data)
└── ...
```


### Custom Tokenizer Integration

The framework uses a unified `TokenizerWrapper` interface, making it easy to integrate custom tokenizers:

```python
from tokenizer_analysis.core.tokenizer_wrapper import TokenizerWrapper, register_tokenizer_class

class SentencePieceTokenizer(TokenizerWrapper):
    """Example custom tokenizer for SentencePiece."""
    
    def __init__(self, name: str, model_path: str):
        import sentencepiece as spm
        self._name = name
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(model_path)
    
    def get_name(self) -> str:
        return self._name
    
    def get_vocab_size(self) -> int:
        return self._sp.get_piece_size()
    
    def get_vocab(self) -> Dict[str, int]:
        return {self._sp.id_to_piece(i): i for i in range(self.get_vocab_size())}
    
    def can_encode(self) -> bool:
        return True
    
    def encode(self, text: str) -> List[int]:
        return self._sp.encode(text, out_type=int)
    
    def can_pretokenize(self) -> bool:
        return True
    
    def pretokenize(self, text: str) -> List[str]:
        return self._sp.encode(text, out_type=str)
    
    @classmethod  
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'SentencePieceTokenizer':
        return cls(name, config['model_path'])

# Register the custom tokenizer
register_tokenizer_class('sentencepiece', SentencePieceTokenizer)
```

Then use in your tokenizer config (see Configuration Files section for format details):
```json
{
  "my_spm_tokenizer": {
    "class": "sentencepiece",
    "model_path": "/path/to/model.spm"
  }
}
```

## Data Format Requirements

The framework supports various input text formats:

### **Raw Text Files**
```text
# Plain text files (.txt)
This is a sentence in English.
This is another sentence.

# One sentence per line (recommended for parallel corpora)
Sentence one.
Sentence two.
Sentence three.
```

### **JSON Files**
```json
{
  "texts": [
    "First text sample here.",
    "Second text sample here.",
    "Third text sample here."
  ]
}
```

### **Parquet Files**
```python
# Expected column: 'text'
import pandas as pd
df = pd.DataFrame({
    'text': ['Sample text 1', 'Sample text 2', 'Sample text 3']
})
df.to_parquet('data.parquet')
```


## Metrics

### Basic Tokenization Metrics
- **Compression Rate**: Text size (bytes/chars/lines) per token - measures encoding efficiency
- **Fertility**: Tokens per word and per character - measures tokenization granularity  
- **Token Length**: Average token size in bytes/characters - measures vocabulary efficiency
- **Type-Token Ratio**: Unique tokens / total tokens - measures vocabulary usage diversity

### Information-Theoretic Metrics  
- **Rényi Entropy**: Information content at different α values - generalizes Shannon entropy
- **Vocabulary Utilization**: Fraction of vocabulary actually used - measures vocabulary efficiency
- **Entropy Analysis**: Token frequency distributions and information content
- **Average Token Rank**: Typical position of tokens in a tokenized text within the frequency-ordered vocabulary

### Morphological Metrics
- **Boundary Precision/Recall**: How well tokens align with morpheme boundaries
- **Morpheme Preservation**: Whether morphemes remain intact after tokenization
- **MorphScore V2**: Advanced morphological evaluation ([Arnett et. al. 2025](https://arxiv.org/abs/2507.06378))

### Multilingual Fairness
- **Tokenizer Gini Coefficient**: Measures equitable treatment across languages, defined as:  

* $`L = \{1, \dots, n\}`$ be the set of languages, each weighted equally.  
* For every language $`\ell \in L`$, define the **token cost**  
```math
  c_\ell \;=\;
  \frac{\text{number of tokens produced by the tokenizer on language }\ell}
       {\text{number of raw bytes (or lines for parallel ds) in the same text}}
```
  (lower $`c_\ell`$ ⇒ cheaper encoding, higher ⇒ more byte-hungry).

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


## Output Interpretation

### **JSON Output Structure**

The framework generates JSON results files with comprehensive analysis data:

#### **analysis_results.json (Summary)**
Contains key metrics and aggregated results:
```json
{
  "fertility": {
    "per_tokenizer": {
      "tokenizer_name": {
        "global": {
          "mean": 1.25,
          "std": 0.15,
          "min": 1.05,
          "max": 1.45
        },
        "per_language": {
          "eng_Latn": {"mean": 1.20, "std": 0.12},
          "spa_Latn": {"mean": 1.30, "std": 0.18}
        }
      }
    },
    "metadata": {
      "normalization_method": "words",
      "description": "Tokens per word/character"
    }
  },
  "compression_ratio": {...},
  "vocabulary_utilization": {...},
  "gini": {...}
}
```

#### **analysis_results_full.json (Detailed)**
Includes additional detailed breakdowns and raw data:
- Sample-level results for each text
- Token frequency distributions
- Detailed statistical breakdowns
- Raw measurements before aggregation

#### **Pre-tokenized Data Config**
Auto-generated config for cached tokenized data:
```json
{
  "tokenizer_name": {
    "class": "pretokenized",
    "vocab_size": 128000,
    "name": "tokenizer_name"
  }
}
```

#### **Key JSON Fields**
- **`per_tokenizer`**: Results organized by tokenizer name
- **`per_language`**: Language-specific breakdowns
- **`global`**: Aggregated statistics across all languages
- **`metadata`**: Configuration and normalization information
- **`mean/std/min/max`**: Statistical summaries
- **`normalization_method`**: Text measurement method used (words/bytes/lines/chars)

### **Understanding Key Metrics**

#### **Fertility (tokens per word/character)**
- **Lower values**: More efficient tokenization (fewer tokens needed)
- **Typical ranges**: 0.5-2.0 tokens/word for subword tokenizers
- **Language differences**: Morphologically rich languages typically have higher fertility

#### **Compression Rate (text units per token)**
- **Higher values**: Better compression (more text compressed per token)
- **Interpretation**: Bytes/characters/lines encoded per token
- **Comparison**: Higher compression rate = more efficient tokenizer

#### **Vocabulary Utilization (0-1)**
- **Higher values**: Better vocabulary usage (less waste)
- **Typical ranges**: 0.3-0.8 for well-trained tokenizers
- **Low values**: May indicate over-sized vocabulary or limited training data

#### **Tokenizer Fairness Gini (0-1)**
- **0.0**: Perfect fairness (equal treatment across languages)
- **1.0**: Maximum unfairness (some languages heavily penalized)
- **Typical values**: 0.1-0.4 for multilingual tokenizers
- **Target**: Lower values indicate better multilingual performance

### **Interpreting Plots**

#### **Individual Plots**
- **Bar charts**: Compare tokenizers across single metric
- **Error bars**: Show variance across languages/samples
- **Global lines**: Average performance across all languages

#### **Grouped Plots**  
- **Script family comparison**: How tokenizers perform on different scripts
- **Resource level analysis**: Performance on high/medium/low resource languages
- **Fairness analysis**: Cross-language equity measurements

#### **Per-Language Plots**
- **Language-specific**: How each tokenizer performs on individual languages
- **Identifies**: Which languages are well/poorly served by each tokenizer

### **What Makes a "Good" Tokenizer**
1. **Low fertility**: Efficient tokenization with fewer tokens
2. **High compression rate**: Good text compression per token
3. **High vocabulary utilization**: Efficient use of vocabulary space
4. **Low Gini coefficient**: Fair treatment across languages
5. **Consistent performance**: Stable metrics across different languages

## Troubleshooting

### Common Issues and Solutions

#### **Installation Problems**
```bash
# Issue: Submodule not found
Error: No module named 'morphscore'

# Solution: Initialize submodules
git submodule update --init --recursive
cd morphscore && pip install -e . && cd ..
```

#### **Tokenizer Loading Errors**
```bash
# Issue: Unknown tokenizer class
ERROR - Unknown tokenizer class: my_tokenizer

# Solution: Check available classes or register custom class
# Available classes: "huggingface", "pretokenized"
# Or register your custom tokenizer class (see Custom Tokenizer Integration)
```

#### **Path and Data Issues**
```bash
# Issue: File not found
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/data'

# Solution: Check paths in config files
# Ensure paths are absolute or relative to working directory
# For directories, ensure they contain the expected files (e.g., eval.txt)
```

#### **Plot Generation Issues**
```bash
# Issue: Display backend errors on servers
_tkinter.TclError: no display name

# Solution: Use non-interactive backend
export MPLBACKEND=Agg
python scripts/run_tokenizer_analysis.py --use-sample-data
```

## Performance Guidelines


### **Performance Optimization**
```bash
# Use pre-tokenized data caching for repeated analysis
python scripts/run_tokenizer_analysis.py --save-tokenized-data --tokenized-data-output-path cache.pkl

# Reuse cached data (much faster)
python scripts/run_tokenizer_analysis.py --tokenized-data-file cache.pkl --tokenized-data-config cache_config.json

# Limit sample size for testing
python scripts/run_tokenizer_analysis.py --use-sample-data --samples-per-lang 50

# Skip expensive plots for faster analysis
python scripts/run_tokenizer_analysis.py --use-sample-data --no-plots
```

### **Scaling Considerations**
- **Encoding optimization**: Texts are encoded once and reused across metrics
- **Grouped analysis**: Automatically filters pre-encoded data (no re-encoding)
- **Memory-efficient**: Results stored incrementally, not held entirely in memory


## Module Structure

```
tokenizer_analysis/
├── __init__.py                    # Main package exports
├── main.py                        # UnifiedTokenizerAnalyzer orchestration class
├── constants.py                   # Package-level constants
├── config/                        # Configuration modules
│   ├── __init__.py
│   ├── language_metadata.py      # LanguageMetadata for grouping analysis
│   └── text_measurement.py       # Text measurement configuration
├── core/                          # Core data structures and providers
│   ├── input_providers.py        # InputProvider implementations
│   ├── input_types.py            # TokenizedData and core types
│   ├── input_utils.py            # Input loading and validation utilities
│   ├── tokenizer_wrapper.py      # Generic wrapper for tokenizer objects
│   └── validation.py             # Data validation functions
├── metrics/                       # Metrics computation modules
│   ├── __init__.py
│   ├── base.py                   # BaseMetrics with common utilities
│   ├── basic.py                  # Basic tokenization metrics
│   ├── information_theoretic.py  # Information-theoretic metrics
│   ├── morphological.py          # Morphological boundary alignment
│   ├── morphscore.py             # MorphScore neural evaluation
│   └── gini.py                   # Multilingual fairness metrics
├── loaders/                       # Data loading modules
│   ├── __init__.py
│   ├── constants.py              # Language code mappings (ISO639-1 to FLORES)
│   ├── morphological.py          # Morphological dataset loader
│   └── multilingual_data.py      # Multilingual text dataset loader
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── text_utils.py             # Text processing utilities
│   └── tokenizer_utils.py        # Tokenizer loading utilities
└── visualization/                 # Plotting and visualization
    ├── __init__.py
    ├── plotter.py                # TokenizerVisualizer main class
    ├── plots.py                  # Core plotting functions
    ├── data_extraction.py        # Data extraction for plotting
    ├── latex_tables.py           # LaTeX table generation
    └── visualization_config.py   # Visualization configuration
```
## Contributing

We welcome contributions to improve the tokenizer analysis framework!

### **Adding New Tokenizers**
1. Create a new `TokenizerWrapper` subclass
2. Implement required abstract methods
3. Register with `register_tokenizer_class()`
4. Add configuration examples to Configuration Files section

### **Adding New Metrics**
1. Inherit from `BaseMetrics` class
2. Implement `compute()` method
3. Add to metrics registry in `main.py`
4. Include plot generation logic

### **Submitting Changes**
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all existing tests pass
5. Submit a pull request with clear description

## API Reference

The framework can be used programmatically as a Python library:

### **Basic Programmatic Usage**
```python
from tokenizer_analysis.main import UnifiedTokenizerAnalyzer
from tokenizer_analysis.config.language_metadata import LanguageMetadata

# Load language configuration
language_metadata = LanguageMetadata.from_file("language_config.json")

# Create analyzer
analyzer = UnifiedTokenizerAnalyzer(language_metadata)

# Load tokenizers from config
tokenizer_configs = {"my_tokenizer": {"class": "huggingface", "path": "bert-base-uncased"}}
analyzer.load_tokenizers_from_config(tokenizer_configs)

# Load language texts  
language_texts = {"en": ["Sample text 1", "Sample text 2"]}
results, encodings = analyzer.run_full_analysis(language_texts=language_texts)

# Access results
fertility_results = results["fertility"]["per_tokenizer"]["my_tokenizer"]
print(f"Global fertility: {fertility_results['global']['mean']}")
```


### **Advanced Analysis Options**
```python
# Grouped analysis by script families
grouped_results = analyzer.run_grouped_analysis(
    encodings=encodings, 
    group_types=['script_family', 'resource_level']
)

# Generate visualizations
from tokenizer_analysis.visualization.plotter import TokenizerVisualizer
visualizer = TokenizerVisualizer(list(tokenizer_configs.keys()))
visualizer.generate_all_plots(results, "output_directory/")

# Save results
import json
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### **Working with Pre-tokenized Data**
```python
# Save tokenized data for reuse
analyzer.save_tokenized_data(encodings, "cached_tokens.pkl", "cache_config.json")

# Load pre-tokenized data
from tokenizer_analysis.core.input_utils import load_vocabularies_from_config
tokenizer_wrappers = load_vocabularies_from_config("cache_config.json")

# Create analyzer from pre-tokenized data
analyzer_from_cache = UnifiedTokenizerAnalyzer.create_analyzer_from_tokenized_data(
    "cached_tokens.pkl", tokenizer_wrappers, language_metadata
)

# Run analysis (much faster - no re-encoding)
cached_results, cached_encodings = analyzer_from_cache.run_full_analysis(all_encodings=cached_encodings)
```

If you use this tokenizer analysis framework in your research, please cite it as follows:

```bibtex
@software{meister_tokenizer_analysis_2025,
  title = {TokEval: A Tokenizer Analysis Suite},
  author = {Meister, Clara},
  year = {2025},
  url = {https://github.com/cimeister/tokenizer-analysis}
}
```
