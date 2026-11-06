"""
Unified tokenizer analysis script supporting both raw tokenizers and pre-tokenized data.

Raw tokenizer examples:
tokenizer-analysis --use-sample-data
tokenizer-analysis --tokenizer-config configs/sample_tokenizers.json --language-config configs/core_lang_config.json --measurement-config configs/text_measurement_config_bytes.json --samples-per-lang 3000 --output-dir analysis_results --verbose --run-grouped-analysis

Pre-tokenized data examples:
tokenizer-analysis --tokenized-data-file tokenized_data.json --language-config configs/core_lang_config.json
tokenizer-analysis --tokenized-data-file tokenized_data.pkl --tokenized-data-config tokenized_config.json --language-config configs/core_lang_config.json --run-grouped-analysis
"""
import argparse
import logging
import math
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tokenizer_analysis import create_analyzer_from_raw_inputs, create_analyzer_from_tokenized_data
from tokenizer_analysis.utils import setup_environment
from tokenizer_analysis.config.language_metadata import (
    LanguageMetadata,
    PACKAGE_ROOT,
    resolve_corpus_path,
)
from tokenizer_analysis.loaders.multilingual_data import (
    load_multilingual_data,
    ParquetEngineMissing,
)
from tokenizer_analysis.core.input_utils import InputLoader
from tokenizer_analysis.metrics.redundancy import MERGES as _MERGES
from tokenizer_analysis.constants import (
    DEFAULT_CER_TIME_BUDGET_S,
    LARGE_ARRAY_THRESHOLD,
    ARRAY_SAMPLING_POINTS,
    DEFAULT_MAX_SAMPLES,
    MIN_TOKENIZERS_FOR_PLOTS,
)
from tokenizer_analysis.visualization.visualization_config import LaTeXFormatting


# Defined in tokenizer_analysis/exceptions.py since 1.0.3, so that core/ can
# raise it without importing from cli/. Re-exported here because callers and
# tests import it from this module.
from tokenizer_analysis.exceptions import ConfigurationError


class OutputGenerationError(RuntimeError):
    """A requested output artifact could not be written.

    Raised at the end of a run so the results file is still saved and reported.
    main() turns it into exit 1 with the message and no traceback, since the
    cause is already logged and names the flag and the path.
    """


logger = logging.getLogger(__name__)


def _configure_cli_logging(log_file: str = "tokenizer_analysis.log",
                           quiet: bool = False) -> None:
    """Configure root logging for the console script.

    Called from ``main()``, never at import. Doing this at module scope meant
    that merely importing ``tokenizer_analysis.cli.run_analysis`` from another
    program reconfigured that program's root logger and created a log file in
    whatever directory it happened to be running in. A library must leave
    logging configuration to the application.

    *quiet* raises the console handler to WARNING and leaves the file handler at
    INFO, so quietening the terminal never costs the record. The root logger
    stays at INFO for the same reason: raising it would silence the file too.
    """
    setup_environment()
    console = logging.StreamHandler()
    if quiet:
        console.setLevel(logging.WARNING)
    # force=True because setup_environment() above calls basicConfig itself, and
    # basicConfig returns without doing anything when the root already has a
    # handler. The file handler was therefore constructed and dropped on every
    # run: tokenizer_analysis.log was created and left empty, so the log this
    # function exists to write has never been written. Found while adding
    # --quiet, whose contract is that the file keeps what the console drops.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            console,
        ],
        force=True,
    )


def load_config_from_file(config_path: str) -> Dict:
    """Load one JSON config, naming the file when it cannot be read or parsed.

    Every ``--*-config`` flag resolves through here, so a run with several of
    them used to report a bare JSONDecodeError giving a line and column and no
    file. With more than one config flag in play the user had to guess which one
    it meant.
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(
            f"{config_path} is not valid JSON: {e.msg} at line {e.lineno} "
            f"column {e.colno}."
        ) from e
    except OSError as e:
        raise ConfigurationError(
            f"{config_path} could not be read: {e.strerror or e}."
        ) from e


def create_sample_configs() -> Dict[str, Dict]:
    """Create sample tokenizer configurations for testing."""
    return {
        "bpe": {
            "class": "huggingface",
            "path": "tokenizers/bpe.json"
        },
        "unigramlm": {
            "class": "huggingface",
            "path": "tokenizers/unigramlm.json"
        }
    }


# One message, used wherever a shipped config names a corpus file that is not
# on disk. FLORES+ is CC-BY-SA 4.0 and is fetched rather than redistributed, so
# a fresh checkout has the configs but not the corpus, and the error has to say
# which command produces it.
FLORES_MISSING_HINT = (
    "The FLORES+ corpus is not in {directory}. This repository does not "
    "redistribute it: fetch it with `uv run python scripts/fetch_flores.py` "
    "(the Hugging Face dataset is gated, so run `hf auth login` first). Use "
    "`--input` or `--language-config` to analyze your own corpus instead."
)


def _require_flores_corpus(paths) -> None:
    """Abort with the fetch instruction when a named FLORES+ file is absent."""
    missing = [p for p in paths if not Path(p).exists()]
    if not missing:
        return
    directory = Path(missing[0]).parent
    raise FileNotFoundError(
        f"{len(missing)} of {len(paths)} corpus file(s) named by this run do "
        f"not exist, starting with {missing[0]}. "
        + FLORES_MISSING_HINT.format(directory=directory)
    )


def create_sample_language_metadata() -> str:
    """Create sample LanguageMetadata configuration and return path to temp file."""
    import tempfile
    
    sample_metadata = {
        "languages": {
            "eng_Latn": {
                "name": "English",
                "iso_code": "en", 
                "data_path": "parallel/eng_Latn.txt"
            },
            "spa_Latn": {
                "name": "Spanish",
                "iso_code": "es",
                "data_path": "parallel/spa_Latn.txt"
            },
            "deu_Latn": {
                "name": "German", 
                "iso_code": "de",
                "data_path": "parallel/deu_Latn.txt"
            },
            "arb_Arab": {
                "name": "Arabic",
                "iso_code": "ar",
                "data_path": "parallel/arb_Arab.txt"
            },
            "rus_Cyrl": {
                "name": "Russian",
                "iso_code": "ru",
                "data_path": "parallel/rus_Cyrl.txt"
            }
        },
        "analysis_groups": {
            "script_family": {
                "Latin": ["eng_Latn", "spa_Latn", "deu_Latn"],
                "Arabic": ["arb_Arab"],
                "Cyrillic": ["rus_Cyrl"]
            },
            "resource_level": {
                "high": ["eng_Latn", "spa_Latn", "deu_Latn"],
                "medium": ["arb_Arab", "rus_Cyrl"],
                "low": []
            },
            "geographic_region": {
                "Western_Europe": ["eng_Latn", "spa_Latn", "deu_Latn"],
                "Middle_East": ["arb_Arab"],
                "Eastern_Europe": ["rus_Cyrl"]
            },
            "language_family": {
                "Indo_European": ["eng_Latn", "spa_Latn", "deu_Latn", "rus_Cyrl"],
                "Afro_Asiatic": ["arb_Arab"]
            }
        }
    }
    
    # Check before writing the config, so --use-sample-data on a fresh checkout
    # names the fetch command instead of failing several steps later with a
    # per-language load report.
    _require_flores_corpus([
        resolve_corpus_path(entry["data_path"])
        for entry in sample_metadata["languages"].values()
    ])

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_metadata, f, indent=2)
        return f.name


def create_sample_morphscore_config(data_dir: str = "morphscore_data") -> Dict[str, any]:
    """Create sample MorphScore configuration with default settings."""
    return {
        "data_dir": data_dir,
        "language_subset": None,
        "by_split": False,
        "freq_scale": True,
        "exclude_single_tok": False
    }

def _resolve_code_ast_config(args) -> Optional[Dict]:
    """Return the code-AST config dict from CLI args, or None if disabled.

    SILENT-FALLBACK HAZARD (fixed 2026-07-13). Omitting --code-ast-config does NOT disable the
    code metrics: it runs them on SYNTHETIC generated code. That produced a whole intrinsic run
    whose code metrics (AST boundary alignment, identifier fragmentation, indentation
    consistency) looked plausible but were computed on toy samples rather than real StarCoder
    data. AST full-alignment differed by 14% (0.562 synthetic vs 0.493 StarCoder) and silently
    corrupted every downstream correlation that used it. Warn loudly.
    """
    if args.code_ast_config:
        config = load_config_from_file(args.code_ast_config)
        # Check the shape here, where the flag and the file path are both in
        # hand. Left to the loader, a JSON array surfaced two files later as
        # AttributeError: 'list' object has no attribute 'items', swallowed at
        # one call site and uncaught at the other.
        from tokenizer_analysis.loaders.code_data import CodeDataLoader
        try:
            CodeDataLoader._validate_config(config)
        except (TypeError, ValueError) as e:
            raise ConfigurationError(f"--code-ast-config {args.code_ast_config}: {e}")
        return config
    if not args.no_code_ast:
        logger.warning(
            "=" * 78 + "\n"
            "NO --code-ast-config GIVEN: the code metrics (AST alignment, identifier "
            "fragmentation, indentation consistency) will be computed on SYNTHETIC sample code, "
            "NOT on real code. These numbers are NOT comparable to a run that passed "
            "--code-ast-config starcoder_ast_config.json. Pass --no-code-ast to disable them "
            "outright, or --code-ast-config to use real data.\n" + "=" * 78
        )
        return {}  # Empty dict triggers synthetic sample generation
    return None


_DROPPABLE_STAT_KEYS = {'sum', 'std_err', 'min', 'max'}


def _strip_stats(d):
    """Keep only {mean, std, median, count} from a compute_basic_stats() dict."""
    if isinstance(d, dict) and {'mean', 'std', 'count'} <= d.keys():
        return {k: v for k, v in d.items() if k not in _DROPPABLE_STAT_KEYS}
    return d


def _strip_per_language(per_lang):
    """Apply _strip_stats to each entry in a per-language dict."""
    if not isinstance(per_lang, dict):
        return per_lang
    return {lang: _strip_stats(v) if isinstance(v, dict) else v
            for lang, v in per_lang.items()}


def _pool_per_language(per_language: dict, fields) -> dict:
    """Count-weighted pool of per-language means, plus the summed count.

    Used by the metrics whose per-language entries are a mean over items and a
    ``count`` of those items. Pooling ``sum(mean_l * count_l) / sum(count_l)``
    recovers the mean over every item in every language, which is what
    ``mean_of_ratios`` names for these metrics: the per-language entries are
    themselves means of per-item ratios, so the count-weighted pool of them is
    the mean over all items, not a ratio of summed counts. It is derived from the per-language block rather
    than read from the metric's ``summary``. Historically that was because the
    grouped filter copied the whole-corpus ``summary`` into each group; the
    filter now re-aggregates ``summary`` per group by the same weighting
    (``UnifiedTokenizerAnalyzer._regroup_digit_summary``), and this stays
    per-language-derived so the slim file does not depend on which path
    produced the block. Reordering the same additions leaves a relative
    difference around 1e-16 against the value ``summary`` reports.

    Every field is null and ``count`` is 0 when no language contributed an
    item, so an empty measurement is not published as a measured zero.
    """
    totals = {field: 0.0 for field in fields}
    weights = {field: 0 for field in fields}
    total_count = 0

    for entry in (per_language or {}).values():
        if not isinstance(entry, dict):
            continue
        count = entry.get('count')
        if not count:
            continue
        total_count += count
        for field in fields:
            value = entry.get(field)
            if value is None:
                continue
            totals[field] += value * count
            weights[field] += count

    pooled = {
        field: (totals[field] / weights[field] if weights[field] else None)
        for field in fields
    }
    pooled['count'] = total_count
    return pooled


def _sample_array(value):
    """Downsample large arrays for JSON export."""
    if isinstance(value, list) and len(value) > LARGE_ARRAY_THRESHOLD:
        step = len(value) // ARRAY_SAMPLING_POINTS
        return value[::step] if step > 1 else value[:ARRAY_SAMPLING_POINTS]
    return value


def _rename_by_category(obj):
    """Recursively rename by_category to per_category in a results dict."""
    if isinstance(obj, dict):
        return {
            ('per_category' if k == 'by_category' else k): _rename_by_category(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_rename_by_category(item) for item in obj]
    return obj


def _normalize_tokenizer_entry(
    metric_name: str,
    tok_data: dict,
    tok_name: Optional[str] = None,
    metric_data: Optional[dict] = None,
) -> dict:
    """Rewrite one tokenizer's raw entry to its published key names, dropping nothing.

    This is the per-tokenizer half of normalize_results. Every rename or pivot
    that the old single-pass slimming step used to perform while also deleting
    fields happens here without deleting anything: a field with a canonical
    published name (for example overall, which becomes global) is moved to
    that name, and a field with no canonical published name (for example
    tokenizer_fairness_gini's most_efficient_language) is left where it was.
    Nothing computed here is new: every value already existed somewhere in
    tok_data or, for the operator_isolation_rate branch, in a sibling block of
    the same metric.

    tok_name and metric_data are only needed by the operator_isolation_rate
    branch, which copies its pooled per-tokenizer rate out of
    metric_data['summary'] and its per-domain split out of
    metric_data['by_domain']. Both stay in place at their own paths too, so
    the value is reachable from either location in the full file.

    Six branches below name a metric no CLI run reaches: avg_tokens_per_line,
    type_token_ratio, unigram_distribution_metrics, lorenz_curve_data,
    utf8_char_split and digit_split_variability. Each is marked "unreachable
    from a CLI run" at its branch. merge_redundant_metrics runs inside
    UnifiedTokenizerAnalyzer.run_analysis, on the whole-corpus results and
    again on every grouped block, and removes all six from the top level
    before normalize_results is called.

    They are kept rather than removed for two reasons. normalize_results (and
    the slim_results_for_json wrapper built on it) is importable and is called
    directly on hand-built results dicts, including in
    tests/test_cli_and_config.py. And merge_redundant_metrics skips a merge
    when either member of a pair is absent, so a caller assembling results
    from part of the pipeline can still have a top-level secondary metric in
    its dict. Removing a branch here would not raise for such a caller: the
    entry would fall through to the generic branch at the end of this function
    and be published in a different shape, with no error.
    """
    if not isinstance(tok_data, dict):
        return tok_data

    out = dict(tok_data)

    def _move(old_key, new_key):
        if old_key in out:
            out[new_key] = out.pop(old_key)

    # --- Metrics with a standard 'global' stats dict ---
    # avg_tokens_per_line: unreachable from a CLI run (merged into
    # compression_rate). See the docstring.
    if metric_name in ('fertility', 'avg_tokens_per_line'):
        if 'global' not in out and 'stats' in out:
            out['global'] = dict(out.pop('stats'))
        if metric_name == 'avg_tokens_per_line' and 'total_lines' in out:
            out['global'] = out.get('global', {})
            out['global']['total_lines'] = out['total_lines']

    # --- Compression rate: already at its published names ---
    elif metric_name == 'compression_rate':
        pass

    # --- Vocabulary utilization: the flat global_* siblings pivot into a
    # nested global block ---
    elif metric_name == 'vocabulary_utilization':
        out['global'] = {
            'utilization': out.pop('global_utilization', None),
            'used_tokens': out.pop('global_used_tokens', None),
            'vocab_size': out.pop('global_vocab_size', None),
        }

    # --- Type-token ratio: unreachable from a CLI run (merged into
    # vocabulary_utilization). See the docstring. Same flat-to-nested pivot. ---
    elif metric_name == 'type_token_ratio':
        out['global'] = {
            'ttr': out.pop('global_ttr', None),
            'types': out.pop('global_types', None),
            'tokens': out.pop('global_tokens', None),
        }

    # --- Unigram distribution metrics: unreachable from a CLI run (merged
    # into renyi_efficiency). See the docstring. Same flat-to-nested pivot. ---
    elif metric_name == 'unigram_distribution_metrics':
        out['global'] = {
            'unigram_entropy': out.pop('global_unigram_entropy', None),
            'avg_token_rank': out.pop('global_avg_token_rank', None),
        }

    # --- Bigram entropy: same flat-to-nested pivot ---
    elif metric_name == 'bigram_entropy':
        out['global'] = {
            'bigram_entropy': out.pop('global_bigram_entropy', None),
            'total_bigrams': out.pop('global_total_bigrams', None),
            'types_evaluated': out.pop('global_types_evaluated', None),
            'types_excluded': out.pop('global_types_excluded', None),
        }

    # --- Trigram entropy: the same pivot as bigram, one n-gram order up ---
    elif metric_name == 'trigram_entropy':
        out['global'] = {
            'trigram_entropy': out.pop('global_trigram_entropy', None),
            'total_trigrams': out.pop('global_total_trigrams', None),
            'types_evaluated': out.pop('global_types_evaluated', None),
            'types_excluded': out.pop('global_types_excluded', None),
        }

    # --- Rényi efficiency: transpose {renyi_X: {overall, lang...}} into a
    # global block keyed by alpha and a per_language block keyed by language.
    # The source per-alpha dicts and token_counts are left in place: the
    # transpose is many-to-many, not a one-to-one rename, so nothing is
    # removed here. ---
    elif metric_name == 'renyi_efficiency':
        global_vals = {}
        lang_vals = {}  # lang -> {renyi_X: v}
        for key, value in tok_data.items():
            if key.startswith('renyi_') and isinstance(value, dict):
                if 'overall' in value:
                    global_vals[key] = value['overall']
                for lang, score in value.items():
                    if lang != 'overall':
                        lang_vals.setdefault(lang, {})[key] = score
        # The number of tokens each entropy was computed over travels in its
        # own dict on the source block, because the pivot above turns one dict
        # per alpha into one dict per language and there is nowhere else to put
        # a per-language count.
        token_counts = tok_data.get('token_counts') or {}
        out['global'] = global_vals
        if lang_vals:
            for lang, entry in lang_vals.items():
                if lang in token_counts:
                    entry['count'] = token_counts[lang]
            out['per_language'] = lang_vals

    # --- Tokenizer fairness (Gini): nest the global-scope scalars under
    # 'global' and rename language_costs to per_language. The remaining
    # fields (most_efficient_language, least_efficient_language,
    # sorted_language_costs) have no canonical published name, so they stay
    # where they are. lorenz_curve and per_line_normalization are already at
    # their published names; _move below is top-level only, so the
    # language_costs nested inside per_line_normalization is not renamed and
    # keeps that name in both output files. ---
    elif metric_name == 'tokenizer_fairness_gini':
        gini_global_fields = (
            'gini_coefficient', 'mean_cost', 'std_cost', 'min_cost',
            'max_cost', 'cost_ratio', 'num_languages', 'warning',
        )
        out['global'] = {
            field: out.pop(field) for field in gini_global_fields if field in out
        }
        _move('language_costs', 'per_language')

    # --- Lorenz curve data: unreachable from a CLI run (merged into
    # tokenizer_fairness_gini). See the docstring. Already at its own flat
    # names; downsampling large arrays is a selection-time concern, not a
    # rename. ---
    elif metric_name == 'lorenz_curve_data':
        pass

    # --- Token length: already at its published names ---
    elif metric_name == 'token_length':
        pass

    # --- UTF-8 metrics: already at their published names.
    # utf8_char_split: unreachable from a CLI run (merged into
    # utf8_token_integrity). See the docstring. ---
    elif metric_name in ('utf8_token_integrity', 'utf8_char_split'):
        pass

    # --- MorphScore: rename summary to global ---
    elif metric_name == 'morphscore':
        _move('summary', 'global')

    # --- AST boundary alignment: rename overall to global, by_language to
    # per_language. by_category was already renamed to per_category by the
    # recursive pass at the top of normalize_results and is left in place. ---
    elif metric_name == 'ast_boundary_alignment':
        _move('overall', 'global')
        _move('by_language', 'per_language')

    # --- Digit boundary metrics: 'overall' holds per-language values, so it
    # is renamed to per_language rather than global.
    # three_digit_boundary_alignment additionally gets a computed global: a
    # count-weighted pool of the per-language mean_f1/mean_precision/
    # mean_recall. The old slimming step computed this while selecting, which
    # meant the pooled value existed only in analysis_results.json; it is
    # computed here instead so the full file carries it too.
    # digit_split_variability: unreachable from a CLI run (merged into
    # three_digit_boundary_alignment). See the docstring. ---
    elif metric_name in ('three_digit_boundary_alignment', 'digit_split_variability'):
        _move('overall', 'per_language')
        if metric_name == 'three_digit_boundary_alignment':
            out['global'] = _pool_per_language(
                out.get('per_language'),
                ('mean_f1', 'mean_precision', 'mean_recall'),
            )

    # --- Numeric magnitude consistency: same per-language rename, plus a
    # computed global pool of mean_fertility, moved here for the same reason
    # as three_digit_boundary_alignment above. ---
    elif metric_name == 'numeric_magnitude_consistency':
        _move('overall', 'per_language')
        out['global'] = _pool_per_language(out.get('per_language'), ('mean_fertility',))

    # --- Operator isolation rate: rename by_language to per_language, and
    # copy in the pooled rate and the by_domain split. Both live as siblings
    # of per_tokenizer on metric_data rather than inside tok_data, and both
    # are left in place there, so this is a copy, not a move: the same value
    # is reachable at metric_data.summary.<tok> (or
    # metric_data.by_domain.<domain>.summary.<tok>) and at
    # metric_data.per_tokenizer.<tok>.global (or ...by_domain.<domain>). ---
    elif metric_name == 'operator_isolation_rate':
        _move('by_language', 'per_language')

        if metric_data is not None and tok_name is not None:
            global_stats = metric_data.get('summary', {}).get(tok_name)
            if global_stats:
                out['global'] = global_stats

            by_domain_full = metric_data.get('by_domain')
            if by_domain_full:
                by_domain_out = {}
                for domain, dom_data in by_domain_full.items():
                    dom_stats = dom_data.get('summary', {}).get(tok_name)
                    if dom_stats:
                        by_domain_out[domain] = dom_stats
                if by_domain_out:
                    out['by_domain'] = by_domain_out

    # --- Reconstruction fidelity: rename overall to global, by_domain to
    # per_domain, by_language to per_language ---
    elif metric_name == 'reconstruction_fidelity':
        _move('overall', 'global')
        _move('by_domain', 'per_domain')
        _move('by_language', 'per_language')

    # --- Identifier fragmentation: rename overall to global, and by_language
    # to per_language unless per_language is already published under that
    # name (a hand-built results dict could set both) ---
    elif metric_name == 'identifier_fragmentation':
        _move('overall', 'global')
        if 'by_language' in out and 'per_language' not in out:
            _move('by_language', 'per_language')

    # --- Indentation consistency ---
    elif metric_name in ('indentation_preservation', 'indentation_consistency'):
        _move('by_language', 'per_language')
        _move('overall', 'global')

    # --- Fallback: the same generic rename the old code applied to any
    # metric not named above ---
    else:
        _move('overall', 'global')
        _move('by_language', 'per_language')

    return out


def _normalize_metric_block(metric_name: str, metric_data: dict) -> dict:
    """Rewrite one metric's block to its published key names, dropping nothing.

    Copies every key metric_data has, then overwrites per_tokenizer with the
    normalized per-tokenizer entries and renames a top-level by_language to
    per_language when nothing is already published under that name. summary,
    metadata, vocabulary_sizes, by_domain, domain_operator_counts,
    observed_normalization, reference_definition, and any other key, stay
    exactly as they were.
    """
    normalized = dict(metric_data)

    if 'per_tokenizer' in metric_data:
        normalized['per_tokenizer'] = {
            tok: _normalize_tokenizer_entry(metric_name, tok_data, tok, metric_data)
            for tok, tok_data in metric_data['per_tokenizer'].items()
        }

    if 'by_language' in normalized and 'per_language' not in normalized:
        normalized['per_language'] = normalized.pop('by_language')

    return normalized


def normalize_results(results: Dict) -> Dict:
    """Rewrite raw analysis results to their published key names, dropping nothing.

    This is the first of the two passes that build analysis_results_full.json
    and analysis_results.json. It applies every rename and pivot the old
    single-pass slimming step used to perform (overall to global, by_language
    to per_language, by_category to per_category, the flat global_* siblings
    folded into a nested global block, and so on), but it never deletes a
    value: a field with a canonical published name is moved there, and a
    field with no canonical name (summary, by_category's un-nested siblings,
    the redundant stat fields sum/std_err/min/max, and so on) is left where it
    was. select_results then builds analysis_results.json by deleting from
    this function's output, never renaming, so every path in the slim file
    also exists, under the same name and with the same value, in the full
    file.

    run_metadata is provenance, not a metric. It has no per_tokenizer block
    and is carried through unchanged. grouped_analysis is one level deeper
    ({group_type: {group_name: {metric: block}}}) than every other key here,
    so each group's results are normalized recursively with the same rules.
    """
    renamed = _rename_by_category(results)

    normalized = {}
    for metric_name, metric_data in renamed.items():
        if not isinstance(metric_data, dict):
            normalized[metric_name] = metric_data
            continue

        if metric_name == 'run_metadata':
            normalized[metric_name] = metric_data
            continue

        if metric_name == 'grouped_analysis':
            normalized[metric_name] = {
                group_type: {
                    group_name: normalize_results(group_result)
                    for group_name, group_result in groups.items()
                }
                for group_type, groups in metric_data.items()
            }
            continue

        normalized[metric_name] = _normalize_metric_block(metric_name, metric_data)

    return normalized


def _select_tokenizer_entry(
    metric_name: str,
    tok_data: dict,
    tok_name: Optional[str] = None,
    metric_data: Optional[dict] = None,
) -> dict:
    """Delete down to the fields analysis_results.json publishes for one tokenizer.

    tok_data has already been rewritten to its published key names by
    normalize_results (by way of _normalize_tokenizer_entry); this function
    only decides which of those keys survive. No key is renamed here: every
    field this function reads is already at its published name.
    """
    if not isinstance(tok_data, dict):
        return tok_data

    out = {}

    # Fields folded in from a redundant metric (see metrics/redundancy.py)
    # carry through unchanged: without this, a caller assembling results from
    # part of the pipeline would silently lose the merged field rather than
    # see it relocated.
    for _sec, _pri, _field, _ in _MERGES:
        if metric_name == _pri and _field in tok_data:
            out[_field] = tok_data[_field]

    if metric_name in ('fertility', 'avg_tokens_per_line'):
        if 'global' in tok_data:
            out['global'] = _strip_stats(tok_data['global'])
        if 'per_language' in tok_data:
            out['per_language'] = _strip_per_language(tok_data['per_language'])

    elif metric_name == 'compression_rate':
        if 'global' in tok_data:
            out['global'] = tok_data['global']  # {compression_rate, total_units, total_tokens}
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']
        # num_texts_analyzed is dropped, and it is the only field this whole
        # selection drops whose value is not readable somewhere else in
        # analysis_results.json: the global block gives total_units and
        # total_tokens but no count of texts. Not published, by decision
        # (2026-08-09), rather than pending one. Do not add it later on the
        # assumption it was an oversight; it is a deliberate limit of the
        # published schema, and adding it would also leave the committed
        # benchmark artifact a field behind until the next regeneration.
        # test_the_slim_file_drops_only_what_it_declares pins this.

    elif metric_name == 'vocabulary_utilization':
        out['global'] = tok_data.get('global', {})
        # Cross-language dispersion of the utilization ratio
        out['per_language_mean'] = tok_data.get('per_language_mean')
        out['per_language_std'] = tok_data.get('per_language_std')
        out['per_language_cov'] = tok_data.get('per_language_cov')
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']

    elif metric_name == 'type_token_ratio':
        out['global'] = tok_data.get('global', {})

    elif metric_name == 'unigram_distribution_metrics':
        out['global'] = tok_data.get('global', {})
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']

    elif metric_name == 'bigram_entropy':
        out['global'] = tok_data.get('global', {})
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']

    elif metric_name == 'trigram_entropy':
        out['global'] = tok_data.get('global', {})
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']

    elif metric_name == 'renyi_efficiency':
        out['global'] = tok_data.get('global', {})
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']

    elif metric_name == 'tokenizer_fairness_gini':
        global_full = tok_data.get('global', {})
        # Copied only when present. This used to read with .get(), which turns
        # a key the metric did not produce into a null, so the slim file
        # published fields the full file did not have and stopped being a
        # subset of it. A writer whose whole job is to drop fields should not
        # be able to invent one.
        out['global'] = {
            field: global_full[field]
            for field in ('gini_coefficient', 'mean_cost', 'std_cost',
                          'cost_ratio', 'num_languages', 'warning')
            if field in global_full
        }
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']
        # The whole per-line block, including its own language_costs, rather
        # than the coefficient alone: on a parallel corpus this is the
        # coefficient to read, and a reader checking it by hand needs the
        # per-language costs it was computed from. The key is written whether
        # or not the block exists, because the metric writes None when the
        # languages have unequal line counts and that absence is a result.
        if 'per_line_normalization' in tok_data:
            out['per_line_normalization'] = tok_data['per_line_normalization']

    elif metric_name == 'lorenz_curve_data':
        for key, value in tok_data.items():
            out[key] = _sample_array(value)

    # 'global' repeats 'primary_length'. Every metric publishes a global
    # block, with no exceptions, so this one does too even though the same
    # numbers sit beside it under the name that says which of the two lengths
    # it is.
    elif metric_name == 'token_length':
        for key in ('character_length', 'byte_length', 'primary_length', 'global'):
            if key in tok_data:
                out[key] = _strip_stats(tok_data[key])

    elif metric_name in ('utf8_token_integrity', 'utf8_char_split'):
        if 'global' in tok_data:
            out['global'] = tok_data['global']
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']

    elif metric_name == 'morphscore':
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']
        if 'global' in tok_data:
            out['global'] = tok_data['global']
        if 'error' in tok_data:
            out['error'] = tok_data['error']

    elif metric_name == 'ast_boundary_alignment':
        if 'global' in tok_data:
            out['global'] = tok_data['global']
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']
        # per_category (renamed from by_category) dropped: full results only

    elif metric_name in ('three_digit_boundary_alignment', 'digit_split_variability'):
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']
        if 'by_digit_length' in tok_data:
            out['by_digit_length'] = tok_data['by_digit_length']
        if 'by_bucket' in tok_data:
            out['by_bucket'] = tok_data['by_bucket']
        if metric_name == 'three_digit_boundary_alignment':
            out['global'] = tok_data.get('global')

    elif metric_name == 'numeric_magnitude_consistency':
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']
        if 'by_digit_length' in tok_data:
            out['by_digit_length'] = tok_data['by_digit_length']
        if 'scaling' in tok_data:
            out['scaling'] = tok_data['scaling']
        out['global'] = tok_data.get('global')

    elif metric_name == 'operator_isolation_rate':
        if 'per_language' in tok_data:
            per_lang = {}
            for lang, lang_data in tok_data['per_language'].items():
                if isinstance(lang_data, dict):
                    # Drop the nested per_category (renamed from by_category)
                    # from each language entry.
                    per_lang[lang] = {
                        k: v for k, v in lang_data.items() if k != 'per_category'
                    }
                else:
                    per_lang[lang] = lang_data
            out['per_language'] = per_lang
        if 'global' in tok_data:
            out['global'] = tok_data['global']
        if 'by_domain' in tok_data:
            out['by_domain'] = tok_data['by_domain']

    elif metric_name == 'reconstruction_fidelity':
        # cer_skipped travels with the block. mean_cer and whitespace_fidelity
        # are null both when the budget abandoned them and when there was
        # nothing to measure, and this flag is the only thing that tells the
        # two apart. METRICS.md and the benchmark README both name it as the
        # field to read, and it is only present when a tokenizer actually
        # exceeded the budget, so a demo corpus that never skips cannot catch
        # its loss.
        for key in ('global', 'per_domain', 'per_language', 'cer_skipped'):
            if key in tok_data:
                out[key] = tok_data[key]

    elif metric_name == 'identifier_fragmentation':
        if 'global' in tok_data:
            out['global'] = tok_data['global']
        per_lang = tok_data.get('per_language')
        if per_lang:
            out['per_language'] = per_lang

    elif metric_name in ('indentation_preservation', 'indentation_consistency'):
        if 'per_language' in tok_data:
            out['per_language'] = tok_data['per_language']
        if 'global' in tok_data:
            out['global'] = tok_data['global']

    # Example tokens and per-language final-token counts: full results only.
    # The percentages, counts and thresholds in 'global' and the fields that
    # say how the counts were produced are published.
    elif metric_name == 'merge_vocabulary_usage':
        for key, value in tok_data.items():
            if key not in ('examples_scaffold', 'examples_scaffold_byte_fragment',
                           'final_tokens_per_language'):
                out[key] = value

    # --- Fallback: keep every key, stripping stats where the shape matches ---
    else:
        for key, value in tok_data.items():
            if key == 'global' and isinstance(value, dict):
                out['global'] = _strip_stats(value)
            elif key == 'per_language':
                out['per_language'] = _strip_per_language(value)
            elif key == 'per_category':
                pass  # by_category equivalent; full results only
            else:
                out[key] = value

    return out


def _select_metric_block(metric_name: str, metric_data: dict) -> dict:
    """Delete down from a normalized metric block to what analysis_results.json publishes.

    metric_data has already been rewritten to its published key names by
    normalize_results; this function only decides which of those keys
    survive.
    """
    out = {}

    if 'per_tokenizer' in metric_data:
        out['per_tokenizer'] = {
            tok: _select_tokenizer_entry(metric_name, tok_data, tok, metric_data)
            for tok, tok_data in metric_data['per_tokenizer'].items()
        }

    # Cross-tokenizer per-language leaderboard (top-level)
    per_lang = metric_data.get('per_language')
    if per_lang:
        out['per_language'] = per_lang

    # Keep vocabulary sizes if present
    if 'vocabulary_sizes' in metric_data:
        out['vocabulary_sizes'] = metric_data['vocabulary_sizes']

    # Keep metadata
    if 'metadata' in metric_data:
        out['metadata'] = metric_data['metadata']

    # Renyi publishes the pre-1.0 normalization alongside the published one
    # so an older value stays reproducible. Kept: it is not derivable from
    # the primary block, since the divisors differ per tokenizer and per
    # language.
    if 'observed_normalization' in metric_data:
        out['observed_normalization'] = metric_data['observed_normalization']

    # Successor entropy under the reference normalizer and aggregation,
    # published next to this library's own definition for comparison.
    if 'reference_definition' in metric_data:
        out['reference_definition'] = metric_data['reference_definition']

    # Operator isolation: keep the per-domain split (prose / code / math).
    #
    # It is NOT derivable from the selected per_tokenizer block, because that block POOLS the
    # three corpora. The corpus decides the answer (a matched pretokenizer pair differs by 0.60
    # on prose and 0.01 on math), so dropping by_domain here would leave the summary file
    # carrying only the pooled number, and any consumer reading it would silently get a
    # different quantity than the one it asked for. Keep each domain's per-tokenizer summary and
    # the source corpus it was computed on; the bulky per-tokenizer/per-language detail stays in
    # analysis_results_full.json.
    if 'by_domain' in metric_data:
        out['by_domain'] = {
            domain: {
                'summary': dom_data.get('summary', {}),
                'source': dom_data.get('source'),
            }
            for domain, dom_data in metric_data['by_domain'].items()
        }

    # Deliberately dropped: summary, and any other key normalize_results left
    # in place but with no entry above.
    return out


def select_results(normalized: Dict) -> Dict:
    """Delete down from normalize_results' output to analysis_results.json's shape.

    This is the second of the two passes. It only deletes keys: every value it
    publishes already exists, at the same key path and with the same value, in
    ``normalized`` (and therefore in analysis_results_full.json, which is
    written from ``normalized`` directly). No rename and no computation
    happens here.
    """
    selected = {}

    for metric_name, metric_data in normalized.items():
        if not isinstance(metric_data, dict):
            selected[metric_name] = metric_data
            continue

        # Provenance is not a metric; it has no per_tokenizer block and must
        # survive verbatim, so the branches below never see it.
        if metric_name == 'run_metadata':
            selected[metric_name] = metric_data
            continue

        # Grouped results are {group_type: {group_name: {metric: block}}}, one
        # level deeper than everything else here, so a flat pass would find no
        # per_tokenizer key and write an empty dict. Select each group's
        # metric block with the same rules instead.
        if metric_name == 'grouped_analysis':
            selected[metric_name] = {
                group_type: {
                    group_name: select_results(group_result)
                    for group_name, group_result in groups.items()
                }
                for group_type, groups in metric_data.items()
            }
            continue

        selected[metric_name] = _select_metric_block(metric_name, metric_data)

    return selected


def slim_results_for_json(results: Dict) -> Dict:
    """Build analysis_results.json's shape directly from raw results.

    Equivalent to ``select_results(normalize_results(results))``. Kept as one
    entry point because it is imported directly by callers, including tests,
    that only want the slim shape and have no use for the intermediate
    full-shape dict.
    """
    return select_results(normalize_results(results))


# --- Deprecation shims for flags/choices removed in 1.0.0 ---
# The standalone morphological boundary metric was removed; MorphScore replaces it.
# These stubs make old command lines fail with a clear pointer instead of an opaque
# argparse error. See CHANGELOG.md.
_MIGRATION_HINT = "See CHANGELOG.md for the full list of changes."

_REMOVED_FLAG_MESSAGES = {
    "--morphological-config": (
        "--morphological-config was removed in tokenizer-intrinsic-evals 1.0.0. The standalone "
        "morphological boundary metric no longer exists; use MorphScore instead via "
        "--morphscore (defaults) or --morphscore-config <file>. " + _MIGRATION_HINT
    ),
}

_LATEX_TABLE_TYPES = ("basic", "information", "comprehensive", "merge_usage")


class _RemovedFlagAction(argparse.Action):
    """Argparse action for flags removed in 1.0.0: fail loudly with a pointer."""

    def __init__(self, option_strings, dest, **kwargs):
        # Accept an optional trailing value so both "--flag" and "--flag x" reach us
        # with our message rather than an "expected one argument" error.
        kwargs["nargs"] = "?"
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(_REMOVED_FLAG_MESSAGES.get(
            option_string,
            f"{option_string} was removed in tokenizer-intrinsic-evals 1.0.0. " + _MIGRATION_HINT,
        ))


class _LatexTableTypesAction(argparse.Action):
    """Validate --latex-table-types, pointing users off the removed 'morphological' type."""

    def __call__(self, parser, namespace, values, option_string=None):
        for value in values:
            if value == "morphological":
                parser.error(
                    "--latex-table-types no longer supports 'morphological' (removed in "
                    "tokenizer-analysis 1.0.0 with the standalone morphological metric). "
                    "Use MorphScore via --morphscore. Valid types: "
                    + ", ".join(_LATEX_TABLE_TYPES) + ". " + _MIGRATION_HINT
                )
            if value not in _LATEX_TABLE_TYPES:
                parser.error(
                    "argument --latex-table-types: invalid choice: %r (choose from %s)"
                    % (value, ", ".join(repr(v) for v in _LATEX_TABLE_TYPES))
                )
        setattr(namespace, self.dest, values)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for tokenizer analysis."""
    parser = argparse.ArgumentParser(
        description="Enhanced modular tokenizer analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-tokenizer analysis with raw tokenizers (supports any number of tokenizers)
  uv run tokenizer-analysis --use-sample-data
  
  # Load from configuration files (supports 1+ tokenizers)
  uv run tokenizer-analysis --tokenizer-config tokenizers.json --language-config languages.json
  
  # Use pre-tokenized data from file
  uv run tokenizer-analysis --tokenized-data-file tokenized_data.json --language-config languages.json
  
  # Use pre-tokenized data with configuration
  uv run tokenizer-analysis --tokenized-data-file tokenized_data.pkl --tokenized-data-config tokenized_config.json --language-config languages.json
  
  # Filter by script family and run grouped analysis (includes grouped plots)
  uv run tokenizer-analysis --use-sample-data --filter-script-family Latin --run-grouped-analysis
  
  # Filter by resource level  
  uv run tokenizer-analysis --use-sample-data --filter-resource-level high
  
  # Run grouped analysis across all script families and resource levels
  uv run tokenizer-analysis --use-sample-data --run-grouped-analysis
  
  # Pairwise comparison only (restricts to 2 specific tokenizers)
  uv run tokenizer-analysis --pairwise tok1 tok2 --use-sample-data
  
  # Skip plots for faster processing
  uv run tokenizer-analysis --use-sample-data --no-plots
  
  # Enable MorphScore analysis with default settings
  uv run tokenizer-analysis --use-sample-data --morphscore
  
  # Use custom MorphScore configuration
  uv run tokenizer-analysis --tokenizer-config tokenizers.json --language-config languages.json --morphscore-config morphscore.json
  
  # Explicitly disable MorphScore (useful when config file has it enabled)
  uv run tokenizer-analysis --use-sample-data
  
  # Generate LaTeX tables for results
  uv run tokenizer-analysis --use-sample-data --generate-latex-tables
  
  # Generate specific LaTeX table types
  uv run tokenizer-analysis --use-sample-data --generate-latex-tables --latex-table-types basic comprehensive
  
  # Generate per-language plots in addition to standard plots
  uv run tokenizer-analysis --use-sample-data --per-language-plots
  
  # Generate faceted plots (one subplot per tokenizer); the two plot flags are
  # independent, and neither applies to --run-grouped-analysis
  uv run tokenizer-analysis --use-sample-data --faceted-plots
  uv run tokenizer-analysis --use-sample-data --per-language-plots --faceted-plots
  
  # Generate custom LaTeX tables from configuration file
  uv run tokenizer-analysis --use-sample-data --custom-latex-config custom_tables.json
  
  # Save both summary and full detailed results
  uv run tokenizer-analysis --use-sample-data --save-full-results
  
  # Save tokenized data for later reuse
  uv run tokenizer-analysis --use-sample-data --save-tokenized-data
  
  # Save tokenized data to specific path
  uv run tokenizer-analysis --tokenizer-config tokenizers.json --language-config languages.json --save-tokenized-data --tokenized-data-output-path my_tokenized_data.pkl
  
  # Complete workflow: generate tokenized data, then reuse it
  # Step 1: Generate and save tokenized data
  uv run tokenizer-analysis --use-sample-data --save-tokenized-data --tokenized-data-output-path results/tokenized_data.pkl
  # Step 2: Use the saved tokenized data (much faster)
  uv run tokenizer-analysis --tokenized-data-file results/tokenized_data.pkl --tokenized-data-config configs/sample_tokenized_config.json --language-config languages.json
        """
    )
    
    # Configuration options
    parser.add_argument(
        "--tokenizer-config", 
        type=str,
        help="JSON file with tokenizer configurations"
    )
    parser.add_argument(
        "--language-config",
        type=str,
        help="JSON file with LanguageMetadata configuration (languages + analysis groups)"
    )
    parser.add_argument(
        "--input",
        type=str,
        metavar="PATH",
        help="Analyze a single corpus: a file (.txt/.json/.jsonl/.parquet) or a "
             "directory of them. Takes exactly one path; for several corpora use "
             "--language-config. Cannot be combined with --language-config or "
             "--use-sample-data."
    )
    parser.add_argument(
        "--input-label",
        type=str,
        metavar="NAME",
        help="Name for the --input corpus in the results (default: the file or "
             "directory stem). Appears as the single key under per_language."
    )
    parser.add_argument(
        "--morphscore-config",
        type=str,
        help="JSON file with MorphScore configuration (requires raw tokenization)"
    )
    # Removed in 1.0.0: fail with a pointer instead of an opaque argparse error.
    parser.add_argument(
        "--morphological-config",
        action=_RemovedFlagAction,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--morphscore",
        action="store_true",
        help="Enable MorphScore analysis with default settings (requires raw tokenization)"
    )
    parser.add_argument(
        "--morphscore-data-dir",
        type=str,
        default="morphscore_data",
        help="Directory containing data for MorphScore analysis"
    )
    parser.add_argument(
        "--measurement-config",
        type=str,
        help="JSON file with text measurement configuration (method, counting functions, etc.)"
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Run the built-in demo: two bundled tokenizers over five FLORES+ "
             "languages. Supplies its own tokenizers and corpus, so it cannot be "
             "combined with --tokenizer-config, --language-config, --input or "
             "--measurement-config. Requires a source checkout (the demo data "
             "lives in parallel/ and tokenizers/, not in the installed package)."
    )
    
    # NEW: Pre-tokenized data support
    parser.add_argument(
        "--tokenized-data-config",
        type=str,
        help="JSON file with pre-tokenized data configuration including vocabulary file paths"
    )
    parser.add_argument(
        "--tokenized-data-file",
        type=str,
        help="Path to pre-tokenized data file (JSON or pickle)"
    )
    
    # Analysis options
    parser.add_argument(
        "--pairwise",
        nargs=2,
        metavar=("TOK1", "TOK2"),
        help="Perform pairwise comparison between two specific tokenizers"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--no-global-lines",
        action="store_true",
        help="Hide global average reference lines in grouped/per-language plots"
    )
    parser.add_argument(
        "--samples-per-lang",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Number of text samples per language (default: {DEFAULT_MAX_SAMPLES})"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output plots and logs"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Log warnings and errors to the console instead of every progress "
             "line. tokenizer_analysis.log still records everything at INFO, so "
             "this changes what the terminal shows and not what is kept."
    )
    parser.add_argument(
        "--save-full-results",
        action="store_true",
        help="Save full detailed results (large file) in addition to summary"
    )
    
    # NEW: LanguageMetadata filtering options
    parser.add_argument(
        "--filter-script-family",
        type=str,
        help="Filter languages by script family (e.g., 'Latin', 'Arabic', 'CJK')"
    )
    
    parser.add_argument(
        "--filter-resource-level", 
        type=str,
        help="Filter languages by resource level (e.g., 'high', 'medium', 'low')"
    )
    
    parser.add_argument(
        "--run-grouped-analysis",
        action="store_true",
        help="Run analysis grouped by script families and resource levels"
    )
    
    # LaTeX table generation options
    parser.add_argument(
        "--generate-latex-tables",
        action="store_true",
        help="Generate LaTeX tables for analysis results"
    )
    parser.add_argument(
        "--latex-table-types",
        nargs="+",
        default=["basic", "comprehensive", "merge_usage"],
        action=_LatexTableTypesAction,
        metavar="{basic,information,comprehensive,merge_usage}",
        help="Types of (default) LaTeX tables to generate"
    )
    parser.add_argument(
        "--latex-output-dir",
        type=str,
        help="Directory for LaTeX table output. --generate-latex-tables defaults to <output-dir>/latex_tables, --custom-latex-config to <output-dir> itself."
    )
    parser.add_argument(
        "--custom-latex-config",
        type=str,
        help="JSON configuration file for custom LaTeX tables"
    )
    
    # Plot generation options
    parser.add_argument(
        "--per-language-plots",
        action="store_true",
        help="Generate per-language plots in addition to individual plots (does not apply to grouped analysis)"
    )
    parser.add_argument(
        "--faceted-plots",
        action="store_true",
        help="Generate additional faceted plots (one subplot per tokenizer with shared y-axis) "
             "for fertility, compression_rate, vocabulary_utilization and bigram_entropy, in "
             "<output-dir>/faceted_plots. Independent of --per-language-plots, and does not "
             "apply to grouped analysis. Normal plots are still generated."
    )
    
    parser.add_argument(
        "--no-digit-boundary",
        action="store_true",
        help="Skip digit boundary alignment, digit split variability, numeric magnitude consistency, and operator isolation analysis"
    )
    parser.add_argument(
        "--math-data",
        type=str,
        help="Path to math-rich text file (.txt or .json) for digit boundary metrics. "
             "When provided, this data is tokenized per-tokenizer and used instead of "
             "the general dataset for digit boundary, digit split variability, numeric "
             "magnitude consistency, and operator isolation metrics."
    )
    parser.add_argument(
        "--use-builtin-math-data",
        action="store_true",
        help="Use the bundled math sample dataset (285 expressions) for "
             "digit boundary metrics instead of the general input text. "
             "Ignored if --math-data is also provided."
    )
    parser.add_argument(
        "--operator-prose-domain",
        action="store_true",
        help="Score the main corpus as a prose domain of "
             "operator_isolation_rate. Off by default: an operator is a code "
             "construct, and the operator pattern matches a hyphen, a slash and "
             "an exclamation mark, which are ordinary punctuation in prose. On "
             "the nine-tokenizer benchmark prose supplied 568 of 455558 "
             "operator occurrences, 0.12%%, while being the domain that costs "
             "the most to score. With this flag, by_domain gains a 'prose' "
             "entry and the pooled figure includes it."
    )
    parser.add_argument(
        "--code-ast-config",
        type=str,
        help="JSON file mapping programming languages to code file/directory paths for AST boundary analysis"
    )
    parser.add_argument(
        "--no-code-ast",
        action="store_true",
        help="Skip AST boundary alignment analysis"
    )
    from tokenizer_analysis.loaders.code_data import CodeDataLoader
    parser.add_argument(
        "--max-code-files-per-lang",
        type=int,
        default=CodeDataLoader.DEFAULT_MAX_SNIPPETS_PER_LANG,
        metavar="N",
        help="Cap on code files loaded per language from --code-ast-config "
             "paths, feeding the AST boundary alignment, identifier "
             "fragmentation and indentation consistency metrics, and the "
             "code domain of operator isolation. "
             f"0 means no cap: every matching file is loaded (default: "
             f"{CodeDataLoader.DEFAULT_MAX_SNIPPETS_PER_LANG}, no cap)."
    )
    parser.add_argument(
        "--max-code-file-chars",
        type=int,
        default=CodeDataLoader.MAX_SNIPPET_SIZE_CHARS,
        metavar="N",
        help="Truncate each loaded code file to this many characters "
             "before it reaches the code metrics. "
             f"0 means no cap: every file is kept in full (default: "
             f"{CodeDataLoader.MAX_SNIPPET_SIZE_CHARS}, no cap). Loading a "
             "large corpus with no cap is slower: every file is parsed and "
             "encoded in full."
    )
    parser.add_argument(
        "--no-utf8-integrity",
        action="store_true",
        help="Skip UTF-8 character boundary integrity analysis"
    )
    parser.add_argument(
        "--no-reconstruction",
        action="store_true",
        help="Skip reconstruction fidelity analysis (decode round-trip)"
    )
    parser.add_argument(
        "--no-merge-vocabulary-usage",
        action="store_true",
        help="Skip the merge vocabulary usage metric (Active / Rare / Uncommon "
             "/ Unseen merge tokens, and Scaffold)"
    )
    parser.add_argument(
        "--merge-usage-include-code-math",
        action="store_true",
        help="Pool the code and math corpora into the merge vocabulary usage "
             "counts, as well as the main input. Without --code-ast-config or "
             "--math-data these are the bundled samples, and the results "
             "metadata names them."
    )
    parser.add_argument(
        "--cer-time-budget",
        type=float,
        default=DEFAULT_CER_TIME_BUDGET_S,
        metavar="SECONDS",
        help="Max seconds to spend on CER computation per tokenizer. "
             "After a warmup phase the total time is extrapolated; if it "
             "exceeds this budget, CER and whitespace fidelity are skipped "
             "for that tokenizer. 0 disables the budget (default: 10.0)"
    )
    # Tokenized data saving options
    parser.add_argument(
        "--save-tokenized-data",
        action="store_true",
        help="Save tokenized data to file (only when processing raw data)"
    )
    parser.add_argument(
        "--tokenized-data-output-path",
        type=str,
        help="Path to save tokenized data (default: output_dir/tokenized_data.pkl)"
    )
    return parser


def _file_digest(path: str) -> Optional[str]:
    """Short SHA-256 of a file, or None if it cannot be read.

    Truncated to 16 hex characters: enough to tell two artifacts apart in a
    provenance record, short enough to read in a diff.
    """
    import hashlib

    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]
    except OSError:
        return None


def _hub_revision(repo_id: str) -> Optional[str]:
    """Commit sha of the locally cached snapshot of a Hugging Face repo.

    Read from the cache rather than the Hub, so it names the revision this run
    actually loaded and needs no network. Returns None when the repo was not
    loaded from the cache (a local directory, or a class that is not
    huggingface), which is the same case _file_digest already covers.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None

    repo_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    ref = repo_dir / "refs" / "main"
    try:
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip() or None
        snapshots = sorted((repo_dir / "snapshots").iterdir())
        if len(snapshots) == 1:
            return snapshots[0].name
    except OSError:
        return None
    # More than one snapshot and no refs/main: which one was loaded is not
    # decidable here, so record nothing rather than guess.
    return None


def _corpus_digest(language_metadata: Any) -> Dict[str, Any]:
    """Per-language path, byte count and short hash of the corpus that was read.

    The config hashes beside this record which corpus was *requested*. They do
    not record what the file held, so two runs over different snapshots of
    FLORES+ under the same config were indistinguishable after the fact. This
    closes that: a rerun whose numbers moved can be attributed to the corpus or
    ruled out.
    """
    if language_metadata is None:
        return {}
    try:
        languages = list(language_metadata.languages)
    except (AttributeError, TypeError):
        return {}

    files: Dict[str, Any] = {}
    for lang in sorted(languages):
        try:
            path = language_metadata.get_data_path(lang)
        except (AttributeError, KeyError, TypeError):
            continue
        if not path:
            continue
        # Relative to the working directory where it sits under it, so a
        # committed results file records "parallel/eng_Latn.txt" rather than a
        # path naming whoever ran it.
        resolved = Path(path)
        try:
            display = str(resolved.resolve().relative_to(Path.cwd().resolve()))
        except (ValueError, OSError):
            display = str(path)
        entry: Dict[str, Any] = {"path": display}
        try:
            entry["bytes"] = resolved.stat().st_size
        except OSError:
            pass
        digest = _file_digest(str(path))
        if digest:
            entry["sha256_16"] = digest
        files[lang] = entry
    if not files:
        return {}

    out: Dict[str, Any] = {"n_languages": len(files), "files": files}
    # The fetch scripts write a manifest beside the corpus naming the dataset
    # revision they read. Carry it through, so the results file records which
    # snapshot of the corpus produced these numbers rather than only what the
    # bytes hashed to.
    for entry in files.values():
        manifest = Path(entry["path"]).parent / "corpus_manifest.json"
        try:
            out["source"] = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        break
    return out


def _git_state() -> Dict[str, Any]:
    """Commit of the repo this package runs from, and whether it was modified.

    The commit alone does not describe the code that ran: on a modified working
    tree it names the last commit, not what executed. Every development run is
    in that state, so a results file recording only the hash overstates what it
    can reproduce.
    """
    import subprocess

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    state: Dict[str, Any] = {"commit": None, "dirty": None}
    try:
        head = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, timeout=5,
        )
        if head.returncode != 0:
            return state
        state["commit"] = head.stdout.decode().strip()
        status = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, timeout=10,
        )
        if status.returncode == 0:
            state["dirty"] = bool(status.stdout.decode().strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return state


# Arguments that decide where output goes rather than what is measured. Two
# runs differing only in these produce the same numbers, so recording them
# would make identical measurements look different.
_OUTPUT_ONLY_ARGS = frozenset({
    "output_dir", "latex_output_dir", "tokenized_data_output_path",
    "save_full_results", "save_tokenized_data", "no_plots", "per_language_plots",
    "faceted_plots", "no_global_lines", "generate_latex_tables",
    "latex_table_types", "custom_latex_config", "verbose", "quiet",
})


def _non_default_arguments(args: argparse.Namespace) -> Dict[str, Any]:
    """Every argument the caller changed from its default, except output paths.

    Recording a hand-maintained list of interesting flags is what left
    --filter-script-family, --filter-resource-level and --use-builtin-math-data
    out: all three change the numbers and left run_metadata byte-identical.
    Diffing the parsed namespace against the parser's own defaults covers any
    flag added later without anyone remembering to add it here.
    """
    defaults = vars(build_parser().parse_args([]))
    changed = {}
    for name, value in sorted(vars(args).items()):
        if name in _OUTPUT_ONLY_ARGS:
            continue
        if defaults.get(name) != value:
            changed[name] = value
    return changed


def _build_run_metadata(args: argparse.Namespace, tokenizer_configs: Dict,
                        language_metadata: Any = None) -> Dict:
    """Describe the run, so a results file can be traced back to what made it.

    Without this a results file is untraceable: nothing recorded the package
    version, the code revision, which configs were read, or which tokenizer
    files were measured. Two files with different numbers gave no way to tell
    whether the tokenizer changed, the corpus changed, or the library did.

    Hashes cover the inputs that decide the numbers, so a rerun that produces
    different values can be attributed.
    """
    from tokenizer_analysis import __version__

    # On the --tokenized-data-file path no tokenizer file is read, so the only
    # artifacts to hash are the vocabulary dumps the cache names. Those are not
    # the tokenizers that produced the ids, and the class is not known, so the
    # entries say what they are rather than imitating the full form.
    replaying_a_cache = args.tokenized_data_file is not None
    tokenizers = {}
    for name, cfg in (tokenizer_configs or {}).items():
        entry: Dict[str, Any] = {}
        if replaying_a_cache:
            entry["source"] = "vocabulary file from --tokenized-data-config"
        else:
            entry["class"] = cfg.get("class", "huggingface")
        path = cfg.get("path")
        if path:
            entry["path"] = path
            digest = _file_digest(path)
            if digest:
                entry["sha256_16"] = digest
            elif entry.get("class") == "huggingface":
                # Not a readable file, so it is a Hub id. Record the cached
                # snapshot's commit sha: without it a Hub-side retokenization
                # changes every number here and nothing says so.
                revision = _hub_revision(path)
                if revision:
                    entry["hub_revision"] = revision
        tokenizers[name] = entry

    configs = {}
    for flag, value in (
        ("tokenizer_config", args.tokenizer_config),
        ("language_config", args.language_config),
        ("measurement_config", args.measurement_config),
        ("code_ast_config", args.code_ast_config),
        ("morphscore_config", args.morphscore_config),
    ):
        if value:
            configs[flag] = {"path": value, "sha256_16": _file_digest(value)}

    git = _git_state()
    return {
        "package_version": __version__,
        # UTC, ISO 8601. Separates two runs of the same commit over the same
        # configs, which is exactly the case that produced an unexplained
        # difference between two benchmark runs before Gini was made
        # order-independent.
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git["commit"],
        "git_tree_modified": git["dirty"],
        "configs": configs,
        "tokenizers": tokenizers,
        # Everything the caller changed from the defaults, so a flag that
        # narrows or redirects what is measured is on the record even if no
        # named field below happens to cover it.
        "arguments": _non_default_arguments(args),
        "corpus": {
            "input": args.input,
            "input_label": args.input_label,
            "use_sample_data": bool(args.use_sample_data),
            "tokenized_data_file": args.tokenized_data_file,
            "samples_per_lang": args.samples_per_lang,
            "digest": _corpus_digest(language_metadata),
        },
        # Effective caps applied by CodeDataLoader when --code-ast-config
        # names real code files (max_code_files_per_lang feeds the AST
        # metrics and the code domain of operator isolation; both were
        # silent caps of 100 files / 15000 chars before 1.0.0 and now
        # default to 0, no cap). These are the resolved values actually
        # passed to the loader, not just whether the flags were typed, so a
        # reader can tell what was measured without re-deriving argparse
        # defaults.
        "code_corpus_caps": {
            "max_files_per_lang": args.max_code_files_per_lang,
            "max_file_chars": args.max_code_file_chars,
        },
        "metrics_disabled": sorted(
            flag for flag, off in (
                ("code_ast", args.no_code_ast),
                ("digit_boundary", args.no_digit_boundary),
                ("utf8_integrity", args.no_utf8_integrity),
                ("reconstruction", args.no_reconstruction),
                ("merge_vocabulary_usage", args.no_merge_vocabulary_usage),
            ) if off
        ),
    }


def _corpus_source_from_input(input_path: str, label: Optional[str]) -> str:
    """Write a one-language config for --input and return its path.

    Reuses the LanguageMetadata path rather than adding a second corpus-loading
    route, so a single corpus goes through exactly the same loader, measurement
    and metric code as a multilingual run. The only difference is that there is
    one group instead of several.
    """
    import tempfile

    path = Path(input_path)
    if not path.exists():
        raise ConfigurationError(
            f"--input path does not exist: {input_path}. Pass a text file "
            "(.txt/.json/.jsonl/.parquet) or a directory containing them."
        )

    corpus_label = label or path.stem or path.name
    if not corpus_label:
        raise ConfigurationError(
            f"Could not derive a corpus label from --input {input_path!r}; "
            "pass --input-label explicitly."
        )

    # Absolute, because a relative data_path in a language config is anchored
    # at the package root (see config.language_metadata.resolve_corpus_path)
    # while --input is a command-line path and means what it means from the
    # working directory the user typed it in.
    config = {"languages": {corpus_label: {"data_path": str(path.resolve())}}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f, indent=2)
        logger.info(
            "Single-corpus mode: analyzing %s under the label %r.",
            path, corpus_label,
        )
        return f.name


def _validate_corpus_source(args: argparse.Namespace) -> None:
    """Reject ambiguous or absent corpus sources before any work happens.

    Three flags can supply a corpus and they are mutually exclusive.
    --use-sample-data used to win silently: it overwrote --tokenizer-config,
    --language-config and --measurement-config with the demo values, so a run
    that named the user's tokenizers and corpus produced a complete, plausible
    results file describing the bundled demo instead. Nothing in the output said
    so. Refuse the combination rather than pick one.
    """
    if args.tokenized_data_file is not None:
        return  # Pre-tokenized mode supplies its own data; validated separately.

    overridden = [
        flag for flag, value in (
            ("--tokenizer-config", args.tokenizer_config),
            ("--language-config", args.language_config),
            ("--input", args.input),
            ("--measurement-config", args.measurement_config),
        ) if value
    ]
    if args.use_sample_data and overridden:
        raise ConfigurationError(
            "--use-sample-data supplies its own tokenizers, corpus and "
            "measurement settings, so it conflicts with "
            + ", ".join(overridden)
            + ". Earlier versions accepted this and silently ignored those flags, "
            "reporting demo numbers under your configuration. Drop "
            "--use-sample-data to analyze your own data, or drop the other flags "
            "to run the demo."
        )

    if args.input and args.language_config:
        raise ConfigurationError(
            "--input analyzes a single corpus and --language-config analyzes a "
            "set of them; pass one or the other. To analyze several corpora, list "
            "them under 'languages' in the --language-config file."
        )

    if not (args.use_sample_data or args.input or args.language_config):
        raise ConfigurationError(
            "No corpus given. Pass --input PATH for a single corpus, "
            "--language-config FILE for several, or --use-sample-data for the "
            "bundled demo. Earlier versions fell back to the bundled five-language "
            "FLORES+ sample here, which meant results could describe demo data "
            "rather than yours."
        )


def _validate_code_corpus_caps(args: argparse.Namespace) -> None:
    """Reject a negative --max-code-files-per-lang or --max-code-file-chars.

    0 is the documented "no cap" value; a negative number is neither a cap
    nor "no cap" and would otherwise be treated the same as 0 by the loader
    (see CodeDataLoader._load_language's ``cap > 0`` check), silently
    discarding what the flag's value was actually supposed to mean.
    """
    for flag, value in (
        ("--max-code-files-per-lang", args.max_code_files_per_lang),
        ("--max-code-file-chars", args.max_code_file_chars),
    ):
        if value < 0:
            raise ConfigurationError(
                f"{flag} must be 0 (no cap) or a positive integer, got {value}."
            )


def run_from_args(args: argparse.Namespace):
    """Run tokenizer analysis from a parsed CLI namespace."""

    _validate_corpus_source(args)
    _validate_code_corpus_caps(args)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine input mode based on provided arguments
    use_tokenized_data = args.tokenized_data_file is not None

    # Load configurations
    if args.use_sample_data and not use_tokenized_data:
        logger.info("Using sample data for demonstration")
        tokenizer_configs = create_sample_configs()
        language_config_path = create_sample_language_metadata()
        measurement_config = None  # Use default for sample data
        
        # Configure MorphScore for sample data
        morphscore_config = None
        if args.morphscore or args.morphscore_config:
            if args.morphscore_config:
                morphscore_config = load_config_from_file(args.morphscore_config)
            else:
                morphscore_config = create_sample_morphscore_config(args.morphscore_data_dir)

        code_ast_config = _resolve_code_ast_config(args)
    elif use_tokenized_data:
        # Pre-tokenized data mode
        if not args.tokenized_data_file:
            raise ConfigurationError("Must specify --tokenized-data-file for pre-tokenized mode")
        
        # Load tokenized data
        tokenized_data = InputLoader.load_from_file(args.tokenized_data_file)

        # No tokenizer files are read on this path, so run_metadata records the
        # vocabulary files the cache names instead. Leaving the name unbound
        # crashed the whole replay at the last line of the run, after every
        # metric had been computed, with UnboundLocalError: cannot access local
        # variable 'tokenizer_configs'.
        tokenizer_configs: Dict = {}

        # Load vocabulary files if config provided
        vocabularies = {}
        if args.tokenized_data_config:
            config = load_config_from_file(args.tokenized_data_config)
            if 'vocabulary_files' in config:
                vocabularies = InputLoader.load_vocabularies_from_config(config['vocabulary_files'])
                tokenizer_configs = {
                    name: {'path': path}
                    for name, path in config['vocabulary_files'].items()
                }
        
        # If no vocabularies loaded, estimate from data
        if not vocabularies:
            logger.warning("No vocabulary files loaded, will estimate vocabulary sizes from tokenized data")
        
        # Language metadata is still needed for grouping and per-language labels.
        # It is not inferable from a pickle of token ids, so require it rather
        # than substituting the bundled five-language sample, which would label
        # the user's data with FLORES+ language codes it has nothing to do with.
        if not args.language_config:
            raise ConfigurationError(
                "--tokenized-data-file needs --language-config: the pickle holds "
                "token ids, not the language labels or groupings the metrics "
                "report against. Pass the same language config used when the "
                "data was tokenized."
            )
        language_config_path = args.language_config
        
        # Load text measurement configuration
        measurement_config = None
        if args.measurement_config:
            from tokenizer_analysis.config import TextMeasurementConfig
            measurement_config_dict = load_config_from_file(args.measurement_config)
            measurement_config = TextMeasurementConfig.from_dict(measurement_config_dict)

        # MorphScore not supported with pre-tokenized data
        morphscore_config = None
        if args.morphscore or args.morphscore_config:
            logger.warning("MorphScore analysis not supported with pre-tokenized data. Requires raw tokenization.")
            morphscore_config = None

        # AST boundary metrics not supported with pre-tokenized data (needs encode())
        code_ast_config = None
        if args.code_ast_config:
            logger.warning("AST boundary analysis not supported with pre-tokenized data. Requires raw tokenization.")
    else:
        # Raw tokenizer mode
        if not args.tokenizer_config:
            raise ConfigurationError(
                "Must specify --tokenizer-config (a JSON file mapping tokenizer "
                "names to {\"class\": ..., \"path\": ...}), or --use-sample-data "
                "to run the bundled demo."
            )

        tokenizer_configs = load_config_from_file(args.tokenizer_config)

        if args.input:
            language_config_path = _corpus_source_from_input(
                args.input, args.input_label
            )
        else:
            # Load language configuration. This is a JSON file; a directory is an error.
            language_config_path = args.language_config

        # Load text measurement configuration
        measurement_config = None
        if args.measurement_config:
            from tokenizer_analysis.config import TextMeasurementConfig
            measurement_config_dict = load_config_from_file(args.measurement_config)
            measurement_config = TextMeasurementConfig.from_dict(measurement_config_dict)
        
        # Configure MorphScore for raw tokenizer mode
        morphscore_config = None
        if args.morphscore or args.morphscore_config:
            if args.morphscore_config:
                morphscore_config = load_config_from_file(args.morphscore_config)
            else:
                morphscore_config = create_sample_morphscore_config(args.morphscore_data_dir)

        code_ast_config = _resolve_code_ast_config(args)

    # Load language metadata
    logger.info("Loading language metadata...")
    language_metadata = LanguageMetadata(language_config_path)
    
    # Initialize analyzer based on input mode
    logger.info("Initializing unified tokenizer analyzer...")
    
    if use_tokenized_data:
        # Pre-tokenized data mode
        analyzer = create_analyzer_from_tokenized_data(
            tokenized_data=tokenized_data,
            vocabularies=vocabularies,
            measurement_config=measurement_config,
            language_metadata=language_metadata,
            plot_save_dir=args.output_dir,
            morphscore_config=morphscore_config,
            code_ast_config=code_ast_config,
            code_max_snippets_per_lang=args.max_code_files_per_lang,
            code_max_snippet_chars=args.max_code_file_chars,
            show_global_lines=not args.no_global_lines,
            per_language_plots=args.per_language_plots,
            faceted_plots=args.faceted_plots,
            math_data_path=args.math_data,
            use_builtin_math_data=args.use_builtin_math_data,
            include_prose_operators=args.operator_prose_domain
        )
    else:
        # Raw tokenizer mode
        # Validate tokenizer configs
        if not tokenizer_configs or len(tokenizer_configs) < 1:
            raise ConfigurationError("At least one tokenizer must be configured")
        
        if args.pairwise and len(args.pairwise) == 2:
            tok1, tok2 = args.pairwise
            if tok1 not in tokenizer_configs or tok2 not in tokenizer_configs:
                raise ConfigurationError(f"Pairwise tokenizers {tok1}, {tok2} must be in configuration")
            # Filter to only these two tokenizers
            tokenizer_configs = {tok1: tokenizer_configs[tok1], tok2: tokenizer_configs[tok2]}
        
        # Load language texts
        logger.info("Loading language texts...")
        filter_by_group = None
        if args.filter_script_family:
            filter_by_group = ('script_family', args.filter_script_family)
        elif args.filter_resource_level:
            filter_by_group = ('resource_level', args.filter_resource_level)
        
        try:
            language_texts = load_multilingual_data(
                language_metadata=language_metadata,
                max_texts_per_language=args.samples_per_lang,
                filter_by_group=filter_by_group
            )
        except ValueError as e:
            # Every ValueError this loader raises describes the config or the
            # corpus it names: an unknown group, a language with no data path, a
            # path that did not load. The message is the whole of what a caller
            # needs, so print it without a stack.
            message = str(e)
            # A config shipped in configs/ names files under parallel/, which
            # this repository does not redistribute, so on a fresh checkout the
            # per-language report is a wall of missing paths with no hint about
            # what produces them.
            flores_dir = str(PACKAGE_ROOT / "parallel")
            if flores_dir in message and not Path(flores_dir).exists():
                message += " " + FLORES_MISSING_HINT.format(directory=flores_dir)
            raise ConfigurationError(message)
        
        if not language_texts:
            raise ConfigurationError("No valid language texts loaded")
        
        # Initialize unified analyzer using convenience function
        analyzer = create_analyzer_from_raw_inputs(
            tokenizer_configs=tokenizer_configs,
            language_texts=language_texts,
            measurement_config=measurement_config,
            language_metadata=language_metadata,
            plot_save_dir=args.output_dir,
            morphscore_config=morphscore_config,
            code_ast_config=code_ast_config,
            code_max_snippets_per_lang=args.max_code_files_per_lang,
            code_max_snippet_chars=args.max_code_file_chars,
            show_global_lines=not args.no_global_lines,
            per_language_plots=args.per_language_plots,
            faceted_plots=args.faceted_plots,
            math_data_path=args.math_data,
            use_builtin_math_data=args.use_builtin_math_data,
            include_prose_operators=args.operator_prose_domain
        )
    # Run analysis
    logger.info("Starting tokenizer analysis...")
    
    if args.pairwise:
        # Pairwise comparison - for now, just run regular analysis and filter results
        logger.info(f"Running pairwise comparison: {args.pairwise[0]} vs {args.pairwise[1]}")
        results = analyzer.run_analysis(
            save_plots=not args.no_plots,

            include_morphscore=morphscore_config is not None,
            include_digit_boundary=not args.no_digit_boundary,
            include_code_ast=not args.no_code_ast,
            include_utf8_integrity=not args.no_utf8_integrity,
            include_reconstruction=not args.no_reconstruction,
            include_merge_vocabulary_usage=not args.no_merge_vocabulary_usage,
            merge_usage_include_code_math=args.merge_usage_include_code_math,
            verbose=args.verbose,
            save_tokenized_data=args.save_tokenized_data,
            tokenized_data_path=args.tokenized_data_output_path,
            cer_time_budget_s=args.cer_time_budget,
        )
    else:
        # Full multi-tokenizer analysis
        results = analyzer.run_analysis(
            save_plots=not args.no_plots,

            include_morphscore=morphscore_config is not None,
            include_digit_boundary=not args.no_digit_boundary,
            include_code_ast=not args.no_code_ast,
            include_utf8_integrity=not args.no_utf8_integrity,
            include_reconstruction=not args.no_reconstruction,
            include_merge_vocabulary_usage=not args.no_merge_vocabulary_usage,
            merge_usage_include_code_math=args.merge_usage_include_code_math,
            verbose=args.verbose,
            save_tokenized_data=args.save_tokenized_data,
            tokenized_data_path=args.tokenized_data_output_path,
            cer_time_budget_s=args.cer_time_budget,
        )
        
        if args.run_grouped_analysis and analyzer.language_metadata:
            logger.info("Running grouped analysis by script families and resource levels...")
            
            # Use the unified analyzer's built-in grouped analysis
            # Pass base results to avoid recomputing metrics
            grouped_results = analyzer.run_grouped_analysis(
                group_by=analyzer.language_metadata.analysis_groups.keys(),
                save_plots=not args.no_plots,
                base_results=results,
                include_reconstruction=not args.no_reconstruction,
                cer_time_budget_s=args.cer_time_budget,
            )
            
            # Add grouped results to main results
            results['grouped_analysis'] = grouped_results
    
    # Record what produced this file, before writing it.
    results['run_metadata'] = _build_run_metadata(
        args, tokenizer_configs,
        language_metadata=getattr(analyzer, 'language_metadata', None),
    )

    # Both output files are built from the same normalized results: full is
    # normalize_results(results) as-is, and slim is select_results applied on
    # top. That makes slim a strict key-path subset of full by construction,
    # rather than each file separately renaming and selecting from raw results.
    normalized_results = normalize_results(results)

    # Save results to JSON (slimmed version)
    results_file = Path(args.output_dir) / "analysis_results.json"
    logger.info(f"Saving slimmed results to {results_file}")

    # Convert numpy arrays to lists for JSON serialization
    convert_for_json = _convert_for_json_public
    slimmed_results = select_results(normalized_results)
    results_json = convert_for_json(slimmed_results)

    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    # Optionally save full results
    if args.save_full_results:
        full_results_file = Path(args.output_dir) / "analysis_results_full.json"
        logger.info(f"Saving full results to {full_results_file}")
        full_results_json = convert_for_json(normalized_results)
        with open(full_results_file, 'w') as f:
            json.dump(full_results_json, f, indent=2)
    
    # Requested outputs that could not be produced. The analysis itself and its
    # results file are unaffected, so the run finishes and reports where they
    # are, then exits non-zero naming what is missing. Exiting 0 here published
    # a "Results saved to" line for a run that skipped an output the user asked
    # for by name.
    unproduced_outputs: List[str] = []

    # Generate LaTeX tables if requested
    if args.generate_latex_tables:
        logger.info("Generating LaTeX tables...")
        latex_output_dir = args.latex_output_dir or os.path.join(args.output_dir, "latex_tables")
        
        formatting_options = {
            'bold_best': LaTeXFormatting.BOLD_BEST,
            'include_std_err': LaTeXFormatting.INCLUDE_STD_ERR,
            'std_err_size': LaTeXFormatting.STD_ERROR_SIZE
        }
        
        try:
            latex_tables = analyzer.generate_latex_tables(
                results=results,
                output_dir=latex_output_dir,
                table_types=args.latex_table_types,
                **formatting_options
            )
            
            logger.info(f"Generated {len(latex_tables)} LaTeX tables in {latex_output_dir}")
            for table_type, content in latex_tables.items():
                print(f"LaTeX {table_type} table: {latex_output_dir}/{table_type}_metrics_table.tex")
                
        except Exception as e:
            logger.error(
                f"--generate-latex-tables: no table was written to {latex_output_dir}: {e}"
            )
            unproduced_outputs.append(f"--generate-latex-tables ({latex_output_dir})")
    
    # Generate custom LaTeX tables if config provided
    if args.custom_latex_config:
        logger.info(f"Loading custom LaTeX configuration from {args.custom_latex_config}")
        try:
            custom_config = load_config_from_file(args.custom_latex_config)
            latex_output_dir = args.latex_output_dir or args.output_dir
            
            # Prepare formatting options
            formatting_options = {
                'bold_best': LaTeXFormatting.BOLD_BEST,
                'include_std_err': LaTeXFormatting.INCLUDE_STD_ERR,
                'std_err_size': LaTeXFormatting.STD_ERROR_SIZE
            }
            
            # Generate each custom table defined in config
            for table_name, table_config in custom_config.items():
                if not isinstance(table_config, dict):
                    logger.warning(f"Skipping invalid table config: {table_name}")
                    continue
                
                metrics = table_config.get('metrics', [])
                caption = table_config.get('caption', f"Custom Table: {table_name}")
                label = table_config.get('label', f"tab:custom_{table_name}")
                
                if not metrics:
                    logger.warning(f"No metrics specified for table {table_name}")
                    continue
                
                logger.info(f"Generating custom LaTeX table '{table_name}' with metrics: {metrics}")
                
                custom_output_path = f"{latex_output_dir}/custom_{table_name}_table.tex"
                custom_table = analyzer.generate_custom_latex_table(
                    results=results,
                    custom_metrics=metrics,
                    output_path=custom_output_path,
                    caption=caption,
                    label=label,
                    **formatting_options
                )
                
                if custom_table:
                    logger.info(f"Custom LaTeX table '{table_name}' saved to {custom_output_path}")
                    print(f"Custom LaTeX table '{table_name}': {custom_output_path}")
                else:
                    logger.warning(f"Custom LaTeX table '{table_name}' generation failed")
                    
        except Exception as e:
            logger.error(
                f"--custom-latex-config {args.custom_latex_config}: no table was "
                f"written: {e}"
            )
            unproduced_outputs.append(
                f"--custom-latex-config {args.custom_latex_config}"
            )
    
    logger.info("Analysis complete!")
    print(f"\nResults saved to: {args.output_dir}")
    if not args.no_plots:
        print(f"Plots saved to: {args.output_dir}")
    print(f"Summary results: {results_file}")

    if unproduced_outputs:
        raise OutputGenerationError(
            "The analysis finished and its results were written, but these "
            "requested outputs were not produced: "
            + "; ".join(unproduced_outputs)
            + ". See the errors above."
        )
    if args.save_full_results:
        print(f"Full detailed results: {Path(args.output_dir) / 'analysis_results_full.json'}")


def _convert_for_json_public(obj):
    """Make *obj* serializable, turning any non-finite float into None.

    json.dump writes NaN, Infinity and -Infinity as those bare tokens, which no
    strict JSON parser accepts, so one of them anywhere makes the whole results
    file unreadable to a consumer that is not Python. null is also what the rest
    of this release means by a value that could not be computed.

    This is the backstop. Metrics are expected to publish None themselves, and
    the CI job asserts the file parses with parse_constant set. It exists
    because the failure is silent at write time and total at read time:
    numeric_magnitude_consistency shipped a NaN spearman_p on any corpus whose
    numbers span two digit-length buckets, and every CI run on Python 3.10, 3.12
    and 3.13 failed on it.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if hasattr(obj, 'tolist'):
        return _convert_for_json_public(obj.tolist())
    if isinstance(obj, dict):
        return {k: _convert_for_json_public(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_for_json_public(item) for item in obj]
    return obj


def main(argv: Optional[List[str]] = None):
    """Parse CLI arguments and run tokenizer analysis."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_cli_logging(quiet=args.quiet)
    try:
        run_from_args(args)
    except (ParquetEngineMissing, ConfigurationError, OutputGenerationError,
            FileNotFoundError, IsADirectoryError) as e:
        # Each of these names the artifact and what to do about it, so a
        # traceback adds only noise. Every other exception keeps its traceback:
        # it is a defect in this package and the stack is the useful part.
        logger.error(str(e))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
