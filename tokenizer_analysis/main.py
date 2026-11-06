"""
Unified main module supporting both raw and pre-tokenized input modes.
"""

import gc
import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

from .constants import AGGREGATION_MICRO_POOLED, DEFAULT_CER_TIME_BUDGET_S
from .core.input_types import InputSpecification, TokenizedData
from .core.input_providers import InputProvider, create_input_provider
from .core.input_utils import create_simple_specifications, InputValidator
from .core.tokenizer_wrapper import create_tokenizer_wrapper
from .metrics.base import BaseMetrics, format_optional
from .metrics.basic import BasicTokenizationMetrics
from .metrics.information_theoretic import InformationTheoreticMetrics
from .metrics.gini import TokenizerGiniMetrics
from .metrics.morphscore import MorphScoreMetrics
from .metrics.math import (
    DigitBoundaryMetrics, magnitude_metadata, operator_metadata,
)
from .metrics.code_ast import ASTBoundaryMetrics
from .metrics.utf8_integrity import UTF8IntegrityMetrics
from .metrics.merge_usage import MergeVocabularyUsageMetrics
from .metrics.redundancy import merge_redundant_metrics
from .visualization import TokenizerVisualizer
from .visualization.latex_tables import LaTeXTableGenerator
from .config import TextMeasurementConfig, DEFAULT_TEXT_MEASUREMENT_CONFIG
from .config.language_metadata import LanguageMetadata
from .loaders.corpora import resolve_code_corpus, resolve_math_corpus

logger = logging.getLogger(__name__)


class UnifiedTokenizerAnalyzer:
    """
    Unified tokenizer analyzer supporting both raw and pre-tokenized inputs.
    
    This class provides a clean interface for tokenizer analysis using the new
    TokenizedData format without any legacy compatibility.
    """
    
    def __init__(self,
                 input_provider: InputProvider,
                 measurement_config: Optional[TextMeasurementConfig] = None,
                 language_metadata: Optional[LanguageMetadata] = None,
                 plot_save_dir: str = "results",
                 show_global_lines: bool = True,
                 morphscore_config: Optional[Dict[str, Any]] = None,
                 plot_tokenizers: Optional[List[str]] = None,
                 per_language_plots: bool = False,
                 faceted_plots: bool = False,
                 code_ast_config: Optional[Dict[str, str]] = None,
                 code_max_snippets_per_lang: Optional[int] = None,
                 code_max_snippet_chars: Optional[int] = None,
                 math_data_path: Optional[str] = None,
                 use_builtin_math_data: bool = False,
                 include_prose_operators: bool = False):
        """
        Initialize unified analyzer.

        Args:
            input_provider: InputProvider instance with tokenized data
            measurement_config: Configuration for text measurement method
            language_metadata: Optional language metadata for grouping
            plot_save_dir: Directory to save plots
            show_global_lines: Whether to show global average reference lines in plots
            morphscore_config: Optional MorphScore configuration (requires raw tokenization)
            plot_tokenizers: Optional list of tokenizer names to include in plots
            per_language_plots: Whether to generate per-language plots
            faceted_plots: Whether to generate faceted plots (one subplot per tokenizer)
            code_ast_config: Mapping of language name to the file, directory or
                parquet path for that language's code. ``None`` and ``{}`` are
                read differently. With ``None`` the AST boundary metrics are
                not constructed at all; with ``{}`` they are, and they run on
                the bundled samples. Either way the code domain of operator
                isolation runs on the bundled samples and reconstruction
                fidelity gets no code domain, because a corpus marked
                ``synthetic`` does not reach it. Construction aborts here when
                a configured path does not load, rather than continuing with an
                empty code corpus.
            code_max_snippets_per_lang: Cap on code files loaded per language
                for the code corpus (feeds both AST metrics and the code
                domain of operator isolation). ``None`` uses
                ``CodeDataLoader.DEFAULT_MAX_SNIPPETS_PER_LANG`` (0, no cap,
                since 1.0.0).
            code_max_snippet_chars: Cap on characters kept per loaded code
                file; longer files are truncated. ``None`` uses
                ``CodeDataLoader.MAX_SNIPPET_SIZE_CHARS`` (0, no cap, since
                1.0.0).
            include_prose_operators: Whether operator_isolation_rate scores the
                main corpus as a prose domain. Off by default: an operator is a
                code construct, and prose supplied 0.12% of the operator
                occurrences on the nine-tokenizer benchmark.
            math_data_path: Optional path to math-rich text file for digit boundary metrics
            use_builtin_math_data: Whether to use the math samples bundled in
                sample_data/math_samples.json as the math corpus for the digit
                boundary and reconstruction fidelity metrics. Ignored when
                *math_data_path* is set: the caller's path is used and the
                bundled samples are not added to it.
        """
        # Validate input provider
        validation_report = InputValidator.validate_input_provider(input_provider)
        if not validation_report['valid']:
            logger.error("Input provider validation failed:")
            for error in validation_report['errors']:
                logger.error(f"  - {error}")
            raise ValueError("Invalid input provider configuration")
        
        self.input_provider = input_provider
        self.tokenizer_names = input_provider.get_tokenizer_names()
        self.measurement_config = measurement_config or DEFAULT_TEXT_MEASUREMENT_CONFIG
        self.language_metadata = language_metadata
        self.plot_save_dir = plot_save_dir
        
        # Handle plot tokenizer filtering
        if plot_tokenizers:
            # Validate that specified tokenizers exist
            invalid_tokenizers = [name for name in plot_tokenizers if name not in self.tokenizer_names]
            if invalid_tokenizers:
                logger.warning(f"Plot tokenizers not found: {invalid_tokenizers}")
            self.plot_tokenizers = [name for name in plot_tokenizers if name in self.tokenizer_names]
        else:
            self.plot_tokenizers = self.tokenizer_names
        
        # Resolve the code and math corpora once, here, beside the flags that
        # select them, and register them on the provider for every metric that
        # reads them. Two CodeDataLoaders used to be built from the same config
        # and the same caps, one here and one inside ASTBoundaryMetrics, so
        # every configured file was read and truncated twice and nothing
        # checked that the two results agreed.
        #
        # A provider without add_corpus is named here rather than reaching
        # `AttributeError: 'X' object has no attribute 'add_corpus'` from inside
        # this constructor. BaseMetrics._register_corpus gives the same error
        # for a metric that builds its own corpus.
        add_corpus = getattr(input_provider, 'add_corpus', None)
        if not callable(add_corpus):
            raise TypeError(
                f"{type(input_provider).__name__} does not implement "
                "add_corpus, so the code and math corpora this run resolved "
                "cannot be registered for the metrics that read them. Subclass "
                "InputProvider, which implements the corpus registry."
            )
        # Both resolved before either is registered, so a failure in the
        # second leaves the provider untouched. Registering as they resolved
        # meant a math config that loaded 0 texts aborted with the code corpus
        # already on the provider, and the retry then failed with "a corpus
        # named 'code' is already registered", naming neither the original
        # problem nor the fix.
        code_corpus = resolve_code_corpus(
            code_ast_config, code_max_snippets_per_lang, code_max_snippet_chars,
        )
        math_corpus = resolve_math_corpus(math_data_path, use_builtin_math_data)
        # Registered together, so a refusal on the second does not leave the
        # first behind. Resolving both first only covers a failure to load;
        # add_corpus itself refuses a name already registered, and the sequential
        # calls left the provider holding the code corpus when that happened.
        registered = []
        try:
            for corpus in (code_corpus, math_corpus):
                add_corpus(corpus)
                registered.append(corpus.name)
        except Exception:
            registry = getattr(input_provider, '_corpora', None)
            if isinstance(registry, dict):
                for name in registered:
                    registry.pop(name, None)
            raise

        # Initialize metrics classes
        self.basic_metrics = BasicTokenizationMetrics(
            input_provider, measurement_config, language_metadata,
        )

        # Initialize information-theoretic metrics
        self.info_metrics = InformationTheoreticMetrics(
            input_provider, measurement_config=measurement_config, language_metadata=language_metadata
        )
        
        # Initialize Gini metrics
        self.gini_metrics = TokenizerGiniMetrics(
            input_provider, measurement_config=measurement_config, language_metadata=language_metadata
        )
        
        # Initialize MorphScore metrics if config provided
        self.morphscore_metrics = None
        if morphscore_config:
            try:
                self.morphscore_metrics = MorphScoreMetrics(
                    input_provider, 
                    **morphscore_config
                )
            except (ImportError, ValueError) as e:
                logger.warning(f"MorphScore metrics disabled: {e}")

        # Initialize digit boundary metrics (always available: no external data).
        # The registered code and math corpora feed the domains of the
        # operator-isolation split.
        self.digit_boundary_metrics = DigitBoundaryMetrics(
            input_provider,
            include_prose_operators=include_prose_operators,
        )

        # Initialize UTF-8 integrity metrics (always available: no external data)
        self.utf8_integrity_metrics = UTF8IntegrityMetrics(input_provider)

        # Merge vocabulary usage (always available: reads the merge list of
        # each BPE tokenizer and reports the others as not applicable)
        self.merge_usage_metrics = MergeVocabularyUsageMetrics(input_provider)

        # Initialize AST boundary metrics if config provided
        self.ast_boundary_metrics = None
        if code_ast_config is not None:
            try:
                # No code_config: the code corpus was resolved from it above
                # and registered on the provider, so this metric reads it from
                # there. Passing it as well is refused, because the two could
                # name different corpora and the registered one would win
                # without a word. max_snippets_per_lang is still passed, because
                # it bounds the corpus rather than selects it and
                # get_code_snippets applies it to the registered snippets.
                # resolve_code_corpus now bounds the bundled samples itself, so
                # this is a second application rather than the only one; it
                # used to be the only one, which left the operator and digit
                # metrics reading an uncapped corpus.
                # max_snippet_chars is not passed and is refused, because
                # nothing on the registered branch can apply it: the corpus was
                # already truncated when it was resolved.
                self.ast_boundary_metrics = ASTBoundaryMetrics(
                    input_provider,
                    max_snippets_per_lang=code_max_snippets_per_lang,
                )
            except ImportError as e:
                # tree-sitter missing is the one condition that disables these
                # metrics rather than failing the run. A bad code config raises
                # ValueError or TypeError and is no longer caught here: it named
                # data the caller asked to be measured.
                logger.warning(f"AST boundary metrics disabled, tree-sitter unavailable: {e}")

        # Initialize visualizer
        self.visualizer = TokenizerVisualizer(self.plot_tokenizers, plot_save_dir, show_global_lines, per_language_plots, faceted_plots)
        
        logger.info(f"Initialized unified analyzer with {len(self.tokenizer_names)} tokenizers: {self.tokenizer_names}")
        if len(self.plot_tokenizers) < len(self.tokenizer_names):
            logger.info(f"Plot filtering enabled: {len(self.plot_tokenizers)} tokenizers will be plotted: {self.plot_tokenizers}")
        for name in self.tokenizer_names:
            vocab_size = self.input_provider.get_vocab_size(name)
            logger.info(f"  {name}: {vocab_size} tokens")

    def run_analysis(self,
                    save_plots: bool = True,
                    include_morphscore: bool = True,
                    include_digit_boundary: bool = True,
                    include_code_ast: bool = True,
                    include_utf8_integrity: bool = True,
                    include_reconstruction: bool = True,
                    include_merge_vocabulary_usage: bool = True,
                    merge_usage_include_code_math: bool = False,
                    verbose: bool = True,
                    save_tokenized_data: bool = False,
                    tokenized_data_path: str = None,
                    cer_time_budget_s: float = DEFAULT_CER_TIME_BUDGET_S) -> Dict[str, Any]:
        """
        Run the full tokenizer analysis.

        Args:
            save_plots: Whether to generate and save plots
            include_morphscore: Whether to include MorphScore analysis (requires access to tokenizers)
            include_digit_boundary: Whether to include the digit boundary and
                operator isolation metrics
            include_code_ast: Whether to include the code AST boundary metrics.
                Has no effect when the analyzer was constructed without a
                ``code_ast_config``, or when tree-sitter was unavailable, since
                in either case there is no ASTBoundaryMetrics instance to run.
            include_utf8_integrity: Whether to include the UTF-8 token integrity
                and character split metrics
            include_reconstruction: Whether to include reconstruction fidelity analysis
            include_merge_vocabulary_usage: Whether to include the merge
                vocabulary usage metric (Active / Rare / Uncommon / Unseen,
                Scaffold)
            merge_usage_include_code_math: Whether the merge vocabulary usage
                counts also pool the registered code and math corpora
            verbose: Whether to print detailed results
            save_tokenized_data: Whether to save tokenized data to file
            tokenized_data_path: Path to save tokenized data (defaults to plot_save_dir/tokenized_data.pkl)
            cer_time_budget_s: Max seconds for CER per tokenizer (0 disables budget)

        Returns:
            Analysis results dictionary
        """
        logger.info("Starting unified tokenizer analysis...")

        tokenized_data = self.input_provider.get_tokenized_data()
        languages = self.input_provider.get_languages()

        logger.info(f"Analyzing {len(languages)} languages: {languages}")
        logger.info(f"Tokenizers: {self.tokenizer_names}")

        results = {}

        # Collect encoding timing if available
        encode_times = getattr(self.input_provider, 'encode_times', None)
        if encode_times:
            per_tok = {}
            for tok_name, times in encode_times.items():
                if times:
                    arr = np.array(times)
                    timings = {
                        'mean_ms': float(np.mean(arr) * 1000),
                        'total_s': float(np.sum(arr)),
                        'num_samples': len(times),
                    }
                    # 'global' is a deliberate duplicate of the three fields
                    # beside it: the results-file schema gives every metric a
                    # global block with no exceptions, and this metric has only
                    # the one whole-corpus figure. The flat fields stay so a
                    # reader of an older file finds them where they were.
                    per_tok[tok_name] = dict(timings)
                    per_tok[tok_name]['global'] = dict(timings)
            if per_tok:
                results['encoding_speed'] = {
                    'per_tokenizer': per_tok,
                    'metadata': {
                        'description': (
                            'Wall-clock time to encode the corpus: mean '
                            'milliseconds per text, total seconds, and the '
                            'number of texts encoded.'
                        ),
                        'aggregation': AGGREGATION_MICRO_POOLED,
                        'count_unit': 'samples',
                        'per_language': (
                            'Not published. Encoding time is measured once per '
                            'text over the whole corpus and is a property of '
                            'the machine as much as of the tokenizer, so it is '
                            'not broken down by language.'
                        ),
                    },
                }
        
        # Run basic tokenization metrics
        logger.info("Computing basic tokenization metrics...")
        basic_results = self.basic_metrics.compute(
            tokenized_data, include_reconstruction=include_reconstruction,
            cer_time_budget_s=cer_time_budget_s)
        results.update(basic_results)

        if verbose:
            self._print_basic_results(basic_results)

        # Run information-theoretic metrics
        logger.info("Computing information-theoretic metrics...")
        info_results = self.info_metrics.compute(tokenized_data)
        results.update(info_results)
        del info_results

        gc.collect()

        # Run Gini metrics
        logger.info("Computing Gini metrics...")
        gini_results = self.gini_metrics.compute(tokenized_data)
        results.update(gini_results)
        
        # Run MorphScore metrics if available
        if self.morphscore_metrics and include_morphscore:
            logger.info("Computing MorphScore metrics...")
            morphscore_results = self.morphscore_metrics.compute(tokenized_data)
            results.update(morphscore_results)
            
            if verbose:
                self.morphscore_metrics.print_results(morphscore_results)
        
        # Run digit boundary metrics if requested
        if include_digit_boundary:
            logger.info("Computing digit boundary metrics...")
            digit_boundary_results = self.digit_boundary_metrics.compute(tokenized_data)
            results.update(digit_boundary_results)

            if verbose:
                self.digit_boundary_metrics.print_results(digit_boundary_results)

        # Run AST boundary metrics if available
        if self.ast_boundary_metrics and include_code_ast:
            logger.info("Computing AST boundary alignment metrics...")
            ast_results = self.ast_boundary_metrics.compute(tokenized_data)
            results.update(ast_results)

            if verbose:
                self.ast_boundary_metrics.print_results(ast_results)

        # Run UTF-8 integrity metrics if requested
        if include_utf8_integrity:
            logger.info("Computing UTF-8 character boundary integrity metrics...")
            utf8_results = self.utf8_integrity_metrics.compute(tokenized_data)
            results.update(utf8_results)

            if verbose:
                self.utf8_integrity_metrics.print_results(utf8_results)

        if include_merge_vocabulary_usage:
            logger.info("Computing merge vocabulary usage...")
            merge_usage_results = self.merge_usage_metrics.compute(
                tokenized_data, include_code_math=merge_usage_include_code_math)
            results.update(merge_usage_results)

            if verbose:
                self.merge_usage_metrics.print_results(merge_usage_results)

        # Fold metrics that restate one another into the metric that owns the
        # measurement, so the results file does not present one number twice as
        # if it were two pieces of evidence. See metrics/redundancy.py for the
        # correlation and identity behind each merge.
        results = merge_redundant_metrics(results)

        # Save tokenized data if requested
        if save_tokenized_data:
            if not tokenized_data_path:
                tokenized_data_path = f"{self.plot_save_dir}/tokenized_data.pkl"
            self._save_tokenized_data(tokenized_data, tokenized_data_path)
        
        # Generate plots
        if save_plots:
            logger.info("Generating plots...")
            self.visualizer.generate_all_plots(results)
        
        logger.info("Analysis completed successfully!")
        return results
    
    def run_grouped_analysis(self,
                           group_by: Optional[Union[str, List[str]]] = None,
                           save_plots: bool = True,
                           base_results: Optional[Dict[str, Any]] = None,
                           include_reconstruction: bool = True,
                           cer_time_budget_s: float = DEFAULT_CER_TIME_BUDGET_S) -> Dict[str, Dict[str, Any]]:
        """
        Run analysis grouped by language categories.
        
        Args:
            group_by: Group type(s) to analyze by. Defaults to every group type
                present in the language metadata. The previous default was the
                literal list ['script_families', 'resource_levels'], which no
                shipped config uses (they write the singular form), so an API
                caller relying on the default got "group type not found" for
                both and an empty result.
            save_plots: Whether to generate grouped plots
            base_results: Optional pre-computed results to filter instead of recomputing
            include_reconstruction: Whether to include reconstruction fidelity analysis
            cer_time_budget_s: Max seconds for CER computation per tokenizer (0 disables budget)

        Returns:
            Dictionary mapping group types to group analysis results
        """
        if not self.language_metadata:
            raise ValueError("Language metadata required for grouped analysis")

        if group_by is None:
            group_by = list(self.language_metadata.analysis_groups.keys())
        elif isinstance(group_by, str):
            group_by = [group_by]

        if not group_by:
            raise ValueError(
                "Grouped analysis needs at least one group type, but the language "
                f"config {self.language_metadata.config_path!r} defines no "
                "'analysis_groups'. Add a group type (for example 'script_family' "
                "mapping a family name to a list of language codes), or skip "
                "grouped analysis."
            )

        grouped_results = {}

        for group_type in group_by:
            logger.info(f"Running grouped analysis by {group_type}")

            if group_type not in self.language_metadata.analysis_groups:
                available = sorted(self.language_metadata.analysis_groups)
                logger.warning(
                    "Group type %r not found in %s; available group types: %s",
                    group_type, self.language_metadata.config_path,
                    ", ".join(available) if available else "(none)",
                )
                continue
            
            group_results = {}
            
            for group_name, group_languages in self.language_metadata.analysis_groups[group_type].items():
                logger.info(f"Analyzing group: {group_name}")
                
                # Filter tokenized data to this group
                filtered_data = self._filter_data_by_languages(group_languages)
                
                if not filtered_data:
                    logger.warning(f"No data found for group {group_name}")
                    continue
                
                # Run analysis on filtered data (same as main analysis)
                group_result = {}
                
                # Basic metrics. include_code_math=False because a group is a
                # set of prose languages: _filter_data_by_languages selects the
                # prose TokenizedData for the group's languages, and the code
                # and math corpora belong to no language. Reporting the whole of
                # both inside every group made each group's reconstruction
                # `global` a figure measured mostly on the same texts in every
                # group, and put those texts into every group's CER budget.
                basic_results = self.basic_metrics.compute(
                    filtered_data, include_reconstruction=include_reconstruction,
                    cer_time_budget_s=cer_time_budget_s,
                    include_code_math=False)
                group_result.update(basic_results)
                
                # Information-theoretic metrics (includes compression_rate)
                info_results = self.info_metrics.compute(filtered_data)
                group_result.update(info_results)
                
                # Gini metrics
                gini_results = self.gini_metrics.compute(filtered_data)
                group_result.update(gini_results)
                
                # MorphScore metrics - filter from base results if available to avoid recomputation
                if self.morphscore_metrics and base_results and 'morphscore' in base_results:
                    logger.info(f"Filtering MorphScore results for group {group_name} (avoiding recomputation)")
                    morphscore_results = self._filter_morphscore_results(
                        base_results['morphscore'], group_languages
                    )
                    group_result['morphscore'] = morphscore_results
                elif self.morphscore_metrics:
                    logger.info(f"Computing MorphScore results for group {group_name}")
                    morphscore_results = self.morphscore_metrics.compute(filtered_data)
                    group_result.update(morphscore_results)

                # UTF-8 integrity metrics, recomputed on the filtered data
                # because it is cheap. Three cases, the same three the digit
                # metrics below use, and for the same reasons.
                if base_results and 'utf8_token_integrity' in base_results:
                    logger.info(f"Computing UTF-8 integrity results for group {group_name}")
                    utf8_results = self.utf8_integrity_metrics.compute(filtered_data)
                    group_result.update(utf8_results)
                elif base_results:
                    # Truthy base results without the key means the base run
                    # produced no UTF-8 metrics, which is what
                    # --no-utf8-integrity does. The missing key is the only
                    # signal the grouped path gets, so computing here anyway
                    # would switch the metric back on for a caller who asked
                    # for it off.
                    logger.info(
                        "No UTF-8 integrity results in the base run, so group "
                        "%s reports none either.", group_name,
                    )
                else:
                    # Nothing precomputed at all, so compute them. This branch
                    # published no UTF-8 block whatsoever, which made a whole
                    # metric depend on how the caller invoked the function.
                    logger.info(f"Computing UTF-8 integrity results for group {group_name}")
                    group_result.update(
                        self.utf8_integrity_metrics.compute(filtered_data)
                    )

                # Digit boundary metrics - filter from base results if available
                if base_results and 'three_digit_boundary_alignment' in base_results:
                    logger.info(f"Filtering digit boundary results for group {group_name} (avoiding recomputation)")
                    group_result['three_digit_boundary_alignment'] = self._filter_digit_boundary_results(
                        base_results['three_digit_boundary_alignment'], group_languages
                    )
                    # Magnitude consistency uses the same structure as digit
                    # boundary, except its metadata promises the scaling fit
                    # the filter drops, so the grouped variant replaces it.
                    if 'numeric_magnitude_consistency' in base_results:
                        group_result['numeric_magnitude_consistency'] = self._filter_digit_boundary_results(
                            base_results['numeric_magnitude_consistency'], group_languages,
                            grouped_metadata=magnitude_metadata(grouped=True),
                        )
                    # Operator isolation has its own structure
                    if 'operator_isolation_rate' in base_results:
                        group_result['operator_isolation_rate'] = self._filter_operator_results(
                            base_results['operator_isolation_rate'], group_languages
                        )
                elif base_results:
                    # Truthy base_results without the digit keys means the
                    # base run produced no digit metrics, which is what
                    # --no-digit-boundary does, so the groups report none
                    # either rather than computing what the caller turned off.
                    # An empty dict is "nothing precomputed" and falls to the
                    # branch below, as it did before.
                    logger.info(
                        "No digit boundary results in the base run, so group "
                        "%s reports none either.", group_name,
                    )
                else:
                    # No base results to filter, so compute them, with
                    # include_code_math=False for the same reason the basic
                    # metrics use it: a group is a set of prose languages, and
                    # reading the whole code and math corpora here put both
                    # entire corpora into every group's operator isolation.
                    logger.info(f"Computing digit boundary results for group {group_name}")
                    db_results = self.digit_boundary_metrics.compute(
                        filtered_data, include_code_math=False,
                    )
                    group_result.update(db_results)

                # Same merge the top-level results get, so a group block and
                # the whole-corpus block have the same keys. Without it a group
                # still published type_token_ratio and avg_tokens_per_line as
                # top-level metrics after they had been folded into
                # vocabulary_utilization and compression_rate everywhere else.
                group_results[group_name] = merge_redundant_metrics(group_result)

            grouped_results[group_type] = group_results
        
        # Generate grouped plots
        if save_plots and grouped_results:
            logger.info("Generating grouped plots...")
            self.visualizer.plot_grouped_analysis(grouped_results)
        
        return grouped_results
    
    def _filter_data_by_languages(self, target_languages: List[str]) -> Dict[str, List[TokenizedData]]:
        """Filter tokenized data to include only specified languages."""
        all_data = self.input_provider.get_tokenized_data()
        filtered_data = {}
        
        for tok_name, data_list in all_data.items():
            filtered_list = [data for data in data_list if data.language in target_languages]
            if filtered_list:
                filtered_data[tok_name] = filtered_list
        
        return filtered_data
    
    def _filter_digit_boundary_results(
        self, db_results: Dict[str, Any], target_languages: List[str],
        grouped_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Filter digit boundary alignment, entropy, or magnitude results to specified languages.

        Two things are deliberately NOT copied from the base results, because
        both pool every language in the run and a group block carrying them
        published whole-corpus statistics under a group label
        (RELEASE_AUDIT Q35.2 R1): ``summary`` is re-aggregated from the
        group's own per-language blocks, and the magnitude ``scaling`` fit is
        dropped (this filter reads only the per-language ``overall`` blocks,
        which do not determine the fit). The recompute path, reached with no
        base results, computes ``scaling`` from whatever the digit metrics
        measure: the group's own data, unless a dedicated math corpus is
        configured, in which case it is the whole math corpus (RELEASE_AUDIT
        Q35.2 R12). Other unknown keys are passed through, but a key holding
        a corpus-level aggregate must be added to the drop next to
        ``scaling``, not passed through.

        *grouped_metadata* replaces the copied metadata when the group block's
        contract differs from the base run's; the magnitude call site passes
        ``magnitude_metadata(grouped=True)`` because its base description
        promises the scaling fit this filter drops.
        """
        filtered: Dict[str, Any] = {
            "per_tokenizer": {},
            "summary": {},
        }

        _LANG_DICT_KEYS = {"by_digit_length", "by_bucket", "overall"}

        for tok_name, tok_data in db_results.get("per_tokenizer", {}).items():
            ftok: Dict[str, Any] = {}

            # Filter by_digit_length
            if "by_digit_length" in tok_data:
                fbd: Dict[str, Any] = {}
                for dl_str, lang_dict in tok_data["by_digit_length"].items():
                    flang = {l: d for l, d in lang_dict.items() if l in target_languages}
                    if flang:
                        fbd[dl_str] = flang
                if fbd:
                    ftok["by_digit_length"] = fbd

            # Filter by_bucket. Every bucket key the base run had survives,
            # holding {} when the group has no numbers in it: a group with
            # short numbers and none long used to lose the "long" key
            # entirely, so the shape depended on the group's data. by_bucket
            # has a fixed two-key schema, which is why its keys are held;
            # by_digit_length's keys are the digit lengths that occurred, so
            # that block stays data-dependent and can be absent for a group.
            if "by_bucket" in tok_data:
                ftok["by_bucket"] = {
                    bucket: {l: d for l, d in lang_dict.items()
                             if l in target_languages}
                    for bucket, lang_dict in tok_data["by_bucket"].items()
                }

            # Filter overall
            if "overall" in tok_data:
                ftok["overall"] = {
                    l: d for l, d in tok_data["overall"].items() if l in target_languages
                }

            # Pass through unknown keys (e.g. scaling) as-is, except a nested
            # block with the same shape, which is filtered by the same rule.
            # The 1.0 merge moved digit_split_variability under this metric as
            # split_variability, and copying it whole gave a language group a
            # number computed over every language in the run.
            for key, value in tok_data.items():
                if key in _LANG_DICT_KEYS or key in ftok:
                    continue
                if key == "scaling":
                    # The fit pools every language of the run and cannot be
                    # derived from base results for a subset. Copying it gave
                    # every group the whole-corpus rho, cv and linear fit,
                    # kept by the slim writer, so a default grouped run
                    # published them under each group label (Q35.2 R1).
                    continue
                if isinstance(value, dict) and _LANG_DICT_KEYS & set(value):
                    nested = self._filter_digit_boundary_results(
                        {"per_tokenizer": {tok_name: value}}, target_languages
                    )
                    ftok[key] = nested["per_tokenizer"].get(tok_name, {})
                else:
                    ftok[key] = value

            if ftok:
                filtered["per_tokenizer"][tok_name] = ftok

        # Summary: re-aggregated from the group's own per-language blocks.
        # The base run's summary pools every language in the run, so copying
        # it published the whole-corpus figure, byte-identical in every
        # group, next to per-language blocks that can be empty (Q35.2 R1).
        for tok_name, ftok in filtered["per_tokenizer"].items():
            regrouped = self._regroup_digit_summary(ftok.get("overall", {}))
            if regrouped is not None:
                filtered["summary"][tok_name] = regrouped

        # Metadata describes the metric, not the language set, so a group
        # block keeps it, except where the group contract differs from the
        # base run's; the caller then supplies the grouped variant.
        if grouped_metadata is not None:
            filtered["metadata"] = grouped_metadata
        elif "metadata" in db_results:
            filtered["metadata"] = db_results["metadata"]

        return filtered

    # Per-language mean fields and the summary field each re-aggregates to.
    # The per-language blocks hold means over that language's numbers plus
    # the count, so the group's pooled mean is the count-weighted mean of the
    # per-language means, which is the same pooling the base summary applies
    # to the whole run. Summary fields that are not determined by the
    # per-language overall blocks this helper reads (the scaling fit's
    # cv/rho/linear terms, avg_uniform_chunk, single_token_frac) are left
    # out of a group summary rather than copied from the whole corpus; most
    # could be derived from the by_digit_length blocks if a group ever needs
    # them.
    _PER_LANGUAGE_MEAN_TO_SUMMARY = {
        "mean_f1": "avg_f1",
        "mean_precision": "avg_precision",
        "mean_recall": "avg_recall",
        "mean_fertility": "avg_fertility",
    }

    @classmethod
    def _regroup_digit_summary(
        cls, overall: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """A group's digit-metric summary from its own per-language blocks.

        Returns None when the group's languages hold no numbers (for example
        a run whose digit metrics measured a dedicated math corpus, which
        belongs to no language group); the group then publishes no summary
        entry rather than the whole-corpus one.
        """
        entries = [d for d in overall.values()
                   if isinstance(d, dict) and d.get("count", 0)]
        total = sum(d["count"] for d in entries)
        if not total:
            return None
        summary: Dict[str, Any] = {}
        for lang_field, summary_field in cls._PER_LANGUAGE_MEAN_TO_SUMMARY.items():
            pairs = [(d[lang_field], d["count"]) for d in entries
                     if d.get(lang_field) is not None]
            if pairs:
                weight = sum(c for _, c in pairs)
                summary[summary_field] = sum(v * c for v, c in pairs) / weight
        summary["numbers_analyzed"] = total
        summary["languages_analyzed"] = len(entries)
        return summary

    def _filter_operator_results(self, op_results: Dict[str, Any], target_languages: List[str]) -> Dict[str, Any]:
        """Filter operator isolation results to specified languages.

        The top-level ``summary``/``by_category`` pool prose+code+math, so they
        must NOT be copied into a language-group result: a language group is a
        subset of the prose corpus, and inheriting the pooled (code-dominated)
        number would report a figure that has nothing to do with the group. Both
        are therefore recomputed from the group's own languages, using the raw
        counts carried on each ``by_language`` entry.

        Code and math rows are keyed "code_python" and "math" in the pooled
        ``by_language``, matching reconstruction fidelity. Matching against
        FLORES codes therefore selects the prose rows only, and a prose
        language whose name would collide with one of those keys aborts in
        _merge_operator_accs rather than reaching this filter.

        The prose/code/math ``by_domain`` split is a corpus-level view, not a
        per-language-group one; it is reported once in the top-level results.
        """
        filtered: Dict[str, Any] = {
            "per_tokenizer": {},
            "summary": {},
        }

        for tok_name, tok_data in op_results.get("per_tokenizer", {}).items():
            fbl = {
                l: d for l, d in tok_data.get("by_language", {}).items()
                if l in target_languages
            }
            if not fbl:
                continue

            # Re-aggregate this group's categories and totals from its languages.
            cat_totals: Dict[str, Dict[str, int]] = {}
            tot_iso = tot_ops = tot_cok = tot_ctot = 0
            for lang_data in fbl.values():
                tot_iso += lang_data.get("isolated", 0)
                tot_ops += lang_data.get("total", 0)
                tot_cok += lang_data.get("compound_ok", 0)
                tot_ctot += lang_data.get("compound_total", 0)
                for category, cdata in lang_data.get("by_category", {}).items():
                    acc = cat_totals.setdefault(
                        category,
                        {"isolated": 0, "total": 0, "compound_ok": 0, "compound_total": 0},
                    )
                    for key in ("isolated", "total", "compound_ok", "compound_total"):
                        acc[key] += cdata.get(key, 0)

            ftok: Dict[str, Any] = {"by_language": fbl, "by_category": {
                category: {
                    # None, not 0.0, matching _build_operator_results in
                    # metrics/math.py. A category with no operator of that kind
                    # in this language group was not measured, and 0.0 reads as
                    # a tokenizer that isolated none of them. The top level
                    # avoided this zero and the grouped filter reintroduced it.
                    "isolation_rate": (c["isolated"] / c["total"]) if c["total"] else None,
                    "compound_preservation_rate": (
                        (c["compound_ok"] / c["compound_total"])
                        if c["compound_total"] else None
                    ),
                    "total": c["total"],
                    "compound_total": c["compound_total"],
                    "isolated": c["isolated"],
                    "compound_ok": c["compound_ok"],
                }
                for category, c in sorted(cat_totals.items())
            }}
            filtered["per_tokenizer"][tok_name] = ftok

            if tot_ops > 0:
                filtered["summary"][tok_name] = {
                    "overall_isolation_rate": tot_iso / tot_ops,
                    "overall_compound_preservation_rate": (
                        (tot_cok / tot_ctot) if tot_ctot else None
                    ),
                    "total_operators": tot_ops,
                    "total_compound_operators": tot_ctot,
                }

        # Not the base run's metadata: that one was built with
        # include_code_math=True and said "Code and math always run" beside a
        # group block holding neither and no by_domain (RELEASE_AUDIT Q35.2
        # R2). The description is single-sourced in metrics/math.py so this
        # filter and the per-group recompute cannot drift apart.
        filtered["metadata"] = operator_metadata(
            include_code_math=False, filtered=True,
        )

        return filtered

    def _filter_morphscore_results(self, morphscore_results: Dict[str, Any], target_languages: List[str]) -> Dict[str, Any]:
        """Filter MorphScore results to include only specified languages.

        The group block gets the same top-level keys the base block has:
        ``per_tokenizer`` and ``metadata``, with no top-level summary,
        because MorphScoreMetrics.compute publishes none. A tokenizer's own
        ``summary`` is recomputed over the group's languages when it has any
        evaluated language in the group, and is absent otherwise, so a
        per-tokenizer entry can hold fewer keys than the base run's.
        """
        import numpy as np

        filtered_results = {
            'per_tokenizer': {},
        }
        
        # Filter per-tokenizer results
        for tok_name, tok_data in morphscore_results.get('per_tokenizer', {}).items():
            filtered_tok_data = {}
            
            # Filter per-language data
            if 'per_language' in tok_data:
                filtered_per_lang = {}
                for lang, lang_data in tok_data['per_language'].items():
                    if lang in target_languages:
                        filtered_per_lang[lang] = lang_data
                
                if filtered_per_lang:
                    filtered_tok_data['per_language'] = filtered_per_lang
                    
                    # Recompute summary statistics based on filtered languages
                    recall_values = []
                    precision_values = []
                    micro_f1_values = []
                    macro_f1_values = []
                    total_samples = 0
                    
                    for lang_data in filtered_per_lang.values():
                        if 'morphscore_recall' in lang_data:
                            recall_values.append(lang_data['morphscore_recall'])
                            precision_values.append(lang_data['morphscore_precision'])
                            micro_f1_values.append(lang_data['micro_f1'])
                            macro_f1_values.append(lang_data['macro_f1'])
                            total_samples += lang_data.get('num_samples', 0)
                    
                    # Compute summary statistics for filtered languages
                    if recall_values:
                        n_languages = len(recall_values)
                        filtered_tok_data['summary'] = {
                            'languages_evaluated': n_languages,
                            'total_samples': total_samples,
                            'avg_morphscore_recall': np.mean(recall_values),
                            'avg_morphscore_precision': np.mean(precision_values),
                            'avg_micro_f1': np.mean(micro_f1_values),
                            'avg_macro_f1': np.mean(macro_f1_values),
                            'avg_morphscore_recall_std': np.std(recall_values),
                            'avg_morphscore_precision_std': np.std(precision_values),
                            'avg_micro_f1_std': np.std(micro_f1_values),
                            'avg_macro_f1_std': np.std(macro_f1_values),
                            'avg_morphscore_recall_std_err': np.std(recall_values) / np.sqrt(n_languages),
                            'avg_morphscore_precision_std_err': np.std(precision_values) / np.sqrt(n_languages),
                            'avg_micro_f1_std_err': np.std(micro_f1_values) / np.sqrt(n_languages),
                            'avg_macro_f1_std_err': np.std(macro_f1_values) / np.sqrt(n_languages)
                        }
            
            # Copy other non-language-specific data (excluding original summary)
            for key, value in tok_data.items():
                if key not in ['per_language', 'summary']:
                    filtered_tok_data[key] = value
            
            if filtered_tok_data:
                filtered_results['per_tokenizer'][tok_name] = filtered_tok_data
        
        # No top-level summary: MorphScoreMetrics.compute publishes none, so a
        # group block does not invent one. The one built here averaged
        # per-tokenizer means and summed languages_evaluated and total_samples
        # over tokenizers, so a 9-tokenizer group over 5 languages reported
        # total_languages_evaluated 45, a field no ungrouped block carries
        # (RELEASE_AUDIT Q35.2 R1).

        # Add any metadata
        if 'metadata' in morphscore_results:
            filtered_results['metadata'] = morphscore_results['metadata']
        
        return filtered_results
    
    def _print_basic_results(self, results: Dict[str, Any]):
        """Print basic metrics results."""
        print("\n" + "="*60)
        print("BASIC TOKENIZATION METRICS RESULTS")
        print("="*60)
        
        # Print fertility results
        if 'fertility' in results:
            fertility_data = results['fertility']
            metadata = fertility_data.get('metadata', {})
            measurement_method = metadata.get('normalization_method', 'units')
            
            print(f"\nFERTILITY ANALYSIS ({measurement_method})")
            print("-" * 40)
            
            for tok_name in self.tokenizer_names:
                if tok_name in fertility_data['per_tokenizer']:
                    global_stats = fertility_data['per_tokenizer'][tok_name]['global']
                    # empty_stats() stores None under 'mean', so the 0.0
                    # default never fires and a plain :.3f raises TypeError.
                    # format_optional prints 'n/a' for a value that was never
                    # measured.
                    mean_fertility = format_optional(global_stats.get('mean'), '.3f')
                    std_fertility = format_optional(global_stats.get('std'), '.3f')
                    print(f"{tok_name:20}: {mean_fertility} ± {std_fertility} tokens/{measurement_method[:-1]}")
        
        # Print token length results
        if 'token_length' in results:
            print(f"\nTOKEN LENGTH ANALYSIS")
            print("-" * 40)
            
            for tok_name in self.tokenizer_names:
                if tok_name in results['token_length']['per_tokenizer']:
                    char_stats = results['token_length']['per_tokenizer'][tok_name]['character_length']
                    mean_length = format_optional(char_stats.get('mean'), '.2f')
                    std_length = format_optional(char_stats.get('std'), '.2f')
                    print(f"{tok_name:20}: {mean_length} ± {std_length} chars/token")
        
        # Print vocabulary utilization
        if 'vocabulary_utilization' in results:
            print(f"\nVOCABULARY UTILIZATION")
            print("-" * 40)
            
            for tok_name in self.tokenizer_names:
                if tok_name in results['vocabulary_utilization']['per_tokenizer']:
                    util_data = results['vocabulary_utilization']['per_tokenizer'][tok_name]
                    utilization = util_data.get('global_utilization', 0.0)
                    used_tokens = util_data.get('global_used_tokens', 0)
                    vocab_size = util_data.get('global_vocab_size', 0)
                    print(f"{tok_name:20}: {format_optional(utilization, '.1%')} "
                          f"({used_tokens:,}/{vocab_size:,} tokens)")
        
        # Print type-token ratio
        if 'type_token_ratio' in results:
            print(f"\nTYPE-TOKEN RATIO")
            print("-" * 40)
            
            for tok_name in self.tokenizer_names:
                if tok_name in results['type_token_ratio']['per_tokenizer']:
                    ttr_data = results['type_token_ratio']['per_tokenizer'][tok_name]
                    ttr = ttr_data.get('global_ttr', 0.0)
                    types = ttr_data.get('global_types', 0)
                    tokens = ttr_data.get('global_tokens', 0)
                    print(f"{tok_name:20}: {format_optional(ttr, '.4f')} "
                          f"({types:,} types / {tokens:,} tokens)")

        # Print reconstruction fidelity
        if 'reconstruction_fidelity' in results:
            summary = results['reconstruction_fidelity'].get('summary', {})
            if summary:
                print(f"\nRECONSTRUCTION FIDELITY")
                print("-" * 40)

                for tok_name in self.tokenizer_names:
                    if tok_name in summary:
                        s = summary[tok_name]
                        # `or 0.0` printed the stand-in default that the null
                        # contract removed, so a domain where every decode
                        # failed showed EM=0.000 rather than saying nothing was
                        # measured. SKIP is reserved for the CER budget, which
                        # is a different reason for a null and is what
                        # docs/METRICS.md defines it as; n/a is no denominator.
                        skipped = bool(s.get('cer_skipped'))

                        def _rate(value, places):
                            if value is not None:
                                return f"{value:.{places}f}"
                            return "SKIP" if skipped else "n/a"

                        n = s.get('texts_analyzed', 0)
                        failures = s.get('decode_failures', 0)
                        line = (
                            f"{tok_name:20}: EM={_rate(s.get('exact_match_rate'), 3)}"
                            f"  CER={_rate(s.get('mean_cer'), 4)}"
                            f"  UNK={_rate(s.get('unk_token_rate'), 4)}"
                            f"  WS={_rate(s.get('whitespace_fidelity'), 3)}"
                            f"  ({n} texts)"
                        )
                        if failures:
                            # Otherwise every rate above is over survivors with
                            # nothing on the line saying texts were dropped.
                            line += f"  [{failures} decode failure(s)]"
                        print(line)

        print("\n" + "="*60)
    
    def generate_latex_tables(self, 
                             results: Dict[str, Any],
                             output_dir: str = None,
                             table_types: List[str] = None,
                             metrics: Dict[str, List[str]] = None,
                             **formatting_options) -> Dict[str, str]:
        """
        Generate LaTeX tables from analysis results.
        
        Args:
            results: Analysis results dictionary
            output_dir: Output directory for table files. If None, uses plot_save_dir
            table_types: List of table types to generate. Options: 'basic', 'information', 'comprehensive', 'merge_usage'
            metrics: Dict mapping table types to specific metrics to include
            **formatting_options: Additional formatting options for LaTeX tables
            
        Returns:
            Dict mapping table types to LaTeX table strings
        """
        if output_dir is None:
            output_dir = os.path.join(self.plot_save_dir, "latex_tables")
        
        if table_types is None:
            table_types = ['basic', 'comprehensive', 'merge_usage']
        
        if metrics is None:
            metrics = {}
        
        # Initialize LaTeX table generator
        latex_generator = LaTeXTableGenerator(results, self.tokenizer_names)
        
        # Apply formatting options
        if formatting_options:
            latex_generator.set_formatting_options(**formatting_options)
        
        generated_tables = {}
        
        for table_type in table_types:
            logger.info(f"Generating {table_type} LaTeX table...")
            
            try:
                if table_type == 'basic':
                    table_content = latex_generator.generate_basic_metrics_table(
                        metrics.get('basic', None)
                    )
                    caption = "Basic Tokenization Metrics"
                    label = "tab:basic_metrics"
                    
                elif table_type == 'information':
                    table_content = latex_generator.generate_information_theory_table(
                        metrics.get('information', None)
                    )
                    caption = "Information-Theoretic Metrics"
                    label = "tab:information_metrics"
                    
                elif table_type == 'comprehensive':
                    table_content = latex_generator.generate_comprehensive_table(
                        metrics.get('comprehensive', None)
                    )
                    caption = "Comprehensive Tokenizer Analysis"
                    label = "tab:comprehensive_metrics"

                elif table_type == 'merge_usage':
                    table_content = latex_generator.generate_merge_usage_table(
                        metrics.get('merge_usage', None)
                    )
                    caption = "Merge Vocabulary Usage (\\% of merge tokens)"
                    label = "tab:merge_usage"
                    
                else:
                    logger.warning(f"Unknown table type: {table_type}")
                    continue
                
                if table_content:
                    generated_tables[table_type] = table_content
                    
                    # Save to file
                    output_path = f"{output_dir}/{table_type}_metrics_table.tex"
                    latex_generator.save_table(table_content, output_path, caption, label)
                    
                else:
                    logger.warning(f"No content generated for {table_type} table")
                    
            except Exception as e:
                logger.error(f"Error generating {table_type} table: {e}")
                continue
        
        return generated_tables
    
    def generate_custom_latex_table(self,
                                   results: Dict[str, Any],
                                   custom_metrics: List[str],
                                   output_path: str = None,
                                   caption: str = None,
                                   label: str = None,
                                   **formatting_options) -> str:
        """
        Generate a custom LaTeX table with specified metrics across categories.
        
        Args:
            results: Analysis results dictionary
            custom_metrics: List of metrics to include (can be from different categories)
            output_path: Optional output file path
            caption: Optional table caption
            label: Optional table label
            **formatting_options: Additional formatting options
            
        Returns:
            LaTeX table string
        """
        logger.info(f"Generating custom LaTeX table with metrics: {custom_metrics}")
        
        # Initialize LaTeX table generator
        latex_generator = LaTeXTableGenerator(results, self.tokenizer_names)
        
        # Apply formatting options
        if formatting_options:
            latex_generator.set_formatting_options(**formatting_options)
        
        # Generate the custom table using the basic table method with custom metrics
        table_content = latex_generator.generate_basic_metrics_table(custom_metrics)
        
        if not table_content:
            logger.warning("No content generated for custom table")
            return ""
        
        # Save to file if path provided
        if output_path:
            latex_generator.save_table(table_content, output_path, caption, label)
            logger.info(f"Custom LaTeX table saved to {output_path}")
        
        return table_content

    def _save_tokenized_data(self, tokenized_data: Dict[str, List], save_path: str):
        """Save tokenized data in format compatible with InputLoader."""
        import pickle
        import json
        
        logger.info(f"Saving tokenized data to {save_path}")
        
        # Create directory if needed
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save tokenized data in pickle format
        with open(save_path, 'wb') as f:
            pickle.dump(tokenized_data, f)
        
        # Save vocabulary files in line-by-line text format
        save_dir = os.path.dirname(save_path)
        for tok_name in self.tokenizer_names:
            vocab_file_path = os.path.join(save_dir, f"{tok_name}_vocab.txt")
            
            # Export the tokenizer's real vocabulary. Refuse to fabricate one:
            # a placeholder vocabulary file would silently corrupt downstream
            # vocabulary-based metrics (junk tokens, dead vocab, avg langs/token).
            try:
                tokenizer = self.input_provider.get_tokenizer(tok_name)
            except Exception as e:
                raise RuntimeError(
                    f"Cannot export vocabulary for '{tok_name}': failed to load "
                    f"the tokenizer ({e})."
                ) from e

            if hasattr(tokenizer, 'get_vocab'):
                vocab_dict = tokenizer.get_vocab()
            elif hasattr(tokenizer, 'vocab'):
                vocab_dict = tokenizer.vocab
            else:
                raise RuntimeError(
                    f"Cannot export vocabulary for '{tok_name}': the tokenizer "
                    f"exposes neither get_vocab() nor a vocab attribute."
                )
            sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
            tokens = [token for token, _ in sorted_vocab]
            
            # Save vocabulary as line-by-line text file
            with open(vocab_file_path, 'w', encoding='utf-8') as f:
                for token in tokens:
                    f.write(f"{token}\n")
            
            logger.info(f"Vocabulary for {tok_name} saved to {vocab_file_path} ({len(tokens)} tokens)")
        
        # Generate tokenized data config file
        config_data = {
            "vocabulary_files": {
                tok_name: f"{tok_name}_vocab.txt" for tok_name in self.tokenizer_names
            }
        }
        
        config_file_path = save_path.replace('.pkl', '_config.json')
        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Tokenized data saved to {save_path}")
        logger.info(f"Configuration file saved to {config_file_path}")

        # Copy the language metadata next to the cache so replaying it is
        # self-sufficient. The pickle holds token ids and language labels but no
        # groupings, and replaying against a different language config silently
        # relabels the data, so the cache has to carry its own.
        if self.language_metadata is None:
            logger.warning(
                "No language metadata to save alongside %s. Replaying this cache "
                "will require --language-config.", save_path,
            )
            return

        lang_config_path = save_path.replace('.pkl', '_language_config.json')
        with open(lang_config_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "languages": self.language_metadata.languages,
                    "analysis_groups": self.language_metadata.analysis_groups,
                },
                f, indent=2, ensure_ascii=False,
            )
        logger.info(
            "Language config saved to %s; replay with "
            "--tokenized-data-file %s --tokenized-data-config %s --language-config %s",
            lang_config_path, save_path, config_file_path, lang_config_path,
        )


# Convenience functions for creating analyzers from different input types

def create_analyzer_from_raw_inputs(tokenizer_configs: Dict[str, Dict],
                                   language_texts: Dict[str, Union[str, List[str]]],
                                   **kwargs) -> UnifiedTokenizerAnalyzer:
    """
    Create analyzer from raw tokenizer configs and texts.
    
    Args:
        tokenizer_configs: Dict mapping tokenizer names to configs
        language_texts: Dict mapping languages to texts (strings or lists of strings)
        **kwargs: Additional arguments for UnifiedTokenizerAnalyzer
        
    Returns:
        UnifiedTokenizerAnalyzer instance
    """    
    # Extract plot filtering from tokenizer configs
    plot_tokenizers = None
    actual_tokenizer_configs = {}
    
    for key, value in tokenizer_configs.items():
        if key == 'plot_tokenizers':
            plot_tokenizers = value
        else:
            actual_tokenizer_configs[key] = value
    
    # Load tokenizers
    tokenizers = {}
    for name, config in actual_tokenizer_configs.items():
        logger.info(f"Loading tokenizer: {name}")
        tokenizers[name] = create_tokenizer_wrapper(name, config)
    
    # Validate plot_tokenizers if provided
    if plot_tokenizers:
        invalid_tokenizers = [name for name in plot_tokenizers if name not in tokenizers]
        if invalid_tokenizers:
            logger.warning(f"Plot tokenizers not found in config: {invalid_tokenizers}")
            plot_tokenizers = [name for name in plot_tokenizers if name in tokenizers]
    
    # Create specifications
    tokenizer_text_pairs = {}
    for name, tokenizer in tokenizers.items():
        tokenizer_text_pairs[name] = (tokenizer, language_texts)
    
    specifications = create_simple_specifications(tokenizer_text_pairs)
    input_provider = create_input_provider(specifications)
    
    # Pass plot_tokenizers to analyzer
    if plot_tokenizers:
        kwargs['plot_tokenizers'] = plot_tokenizers
    
    return UnifiedTokenizerAnalyzer(input_provider, **kwargs)


def create_analyzer_from_tokenized_data(tokenized_data: Dict[str, List[TokenizedData]],
                                       vocabularies: Dict[str, Union[int, 'TokenizerWrapper']],
                                       **kwargs) -> UnifiedTokenizerAnalyzer:
    """
    Create analyzer from pre-tokenized data.
    
    Args:
        tokenized_data: Dict mapping tokenizer names to TokenizedData lists
        vocabularies: Dict mapping tokenizer names to vocab sizes or TokenizerWrapper objects
        **kwargs: Additional arguments for UnifiedTokenizerAnalyzer
        
    Returns:
        UnifiedTokenizerAnalyzer instance
    """
    from .core.tokenizer_wrapper import PreTokenizedDataTokenizer, TokenizerWrapper
    
    specifications = {}
    for tok_name, data_list in tokenized_data.items():
        # Create tokenizer wrapper
        if tok_name in vocabularies:
            vocab = vocabularies[tok_name]
            if isinstance(vocab, int):
                tokenizer = PreTokenizedDataTokenizer(tok_name, vocab)
            elif isinstance(vocab, TokenizerWrapper):
                tokenizer = vocab
            else:
                raise ValueError(f"Invalid vocabulary for {tok_name}: must be int or TokenizerWrapper")
        else:
            # Estimate vocab size and create tokenizer
            max_token_id = max(max(data.tokens) for data in data_list if data.tokens)
            tokenizer = PreTokenizedDataTokenizer(tok_name, max_token_id + 1)
        
        spec = InputSpecification(
            tokenizer=tokenizer,
            tokenized_data=data_list
        )
        specifications[tok_name] = spec
    
    input_provider = create_input_provider(specifications)
    return UnifiedTokenizerAnalyzer(input_provider, **kwargs)

