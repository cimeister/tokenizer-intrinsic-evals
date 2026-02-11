"""
Unified main module supporting both raw and pre-tokenized input modes.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

from .core.input_types import TokenizedData, InputSpecification
from .core.input_providers import InputProvider, create_input_provider
from .core.input_utils import create_simple_specifications, InputValidator
from .core.tokenizer_wrapper import create_tokenizer_wrapper
from .metrics.base import BaseMetrics
from .metrics.basic import BasicTokenizationMetrics
from .metrics.information_theoretic import InformationTheoreticMetrics
from .metrics.gini import TokenizerGiniMetrics
from .metrics.morphological import MorphologicalMetrics
from .metrics.morphscore import MorphScoreMetrics
from .visualization import TokenizerVisualizer
from .visualization.latex_tables import LaTeXTableGenerator
from .config import TextMeasurementConfig, DEFAULT_TEXT_MEASUREMENT_CONFIG
from .config.language_metadata import LanguageMetadata

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
                 morphological_config: Optional[Dict[str, str]] = None,
                 show_global_lines: bool = True,
                 morphscore_config: Optional[Dict[str, Any]] = None,
                 plot_tokenizers: Optional[List[str]] = None,
                 per_language_plots: bool = False,
                 faceted_plots: bool = False):
        """
        Initialize unified analyzer.
        
        Args:
            input_provider: InputProvider instance with tokenized data
            measurement_config: Configuration for text measurement method
            language_metadata: Optional language metadata for grouping
            plot_save_dir: Directory to save plots
            morphological_config: Optional morphological dataset configuration
            show_global_lines: Whether to show global average reference lines in plots
            morphscore_config: Optional MorphScore configuration (requires raw tokenization)
            plot_tokenizers: Optional list of tokenizer names to include in plots
            per_language_plots: Whether to generate per-language plots
            faceted_plots: Whether to generate faceted plots (one subplot per tokenizer)
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
        
        # Initialize metrics classes
        self.basic_metrics = BasicTokenizationMetrics(
            input_provider, measurement_config, language_metadata
        )
        
        # Initialize information-theoretic metrics
        self.info_metrics = InformationTheoreticMetrics(
            input_provider, measurement_config=measurement_config, language_metadata=language_metadata
        )
        
        # Initialize Gini metrics
        self.gini_metrics = TokenizerGiniMetrics(
            input_provider, measurement_config=measurement_config, language_metadata=language_metadata
        )
        
        # Initialize morphological metrics if config provided
        self.morphological_metrics = None
        if morphological_config:
            self.morphological_metrics = MorphologicalMetrics(
                input_provider, morphological_config=morphological_config
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
                self.morphscore_metrics = None
        
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
                    include_morphological: bool = True,  
                    include_morphscore: bool = True,
                    verbose: bool = True,
                    save_tokenized_data: bool = False,
                    tokenized_data_path: str = None) -> Dict[str, Any]:
        """
        Run comprehensive tokenizer analysis.
        
        Args:
            save_plots: Whether to generate and save plots
            include_morphological: Whether to include morphological analysis (not yet implemented)
            include_morphscore: Whether to include MorphScore analysis (requires access to tokenizers)
            verbose: Whether to print detailed results
            save_tokenized_data: Whether to save tokenized data to file
            tokenized_data_path: Path to save tokenized data (defaults to plot_save_dir/tokenized_data.pkl)
            
        Returns:
            Analysis results dictionary
        """
        logger.info("Starting unified tokenizer analysis...")
        
        tokenized_data = self.input_provider.get_tokenized_data()
        languages = self.input_provider.get_languages()
        
        logger.info(f"Analyzing {len(languages)} languages: {languages}")
        logger.info(f"Tokenizers: {self.tokenizer_names}")
        
        results = {}
        
        # Run basic tokenization metrics
        logger.info("Computing basic tokenization metrics...")
        basic_results = self.basic_metrics.compute(tokenized_data)
        results.update(basic_results)
        
        if verbose:
            self._print_basic_results(basic_results)
        
        # Run information-theoretic metrics
        logger.info("Computing information-theoretic metrics...")
        info_results = self.info_metrics.compute(tokenized_data)
        results.update(info_results)
        
        # Run Gini metrics
        logger.info("Computing Gini metrics...")
        gini_results = self.gini_metrics.compute(tokenized_data)
        results.update(gini_results)
        
        # Run morphological metrics if available
        if self.morphological_metrics and include_morphological:
            logger.info("Computing morphological metrics...")
            morphological_results = self.morphological_metrics.compute(tokenized_data)
            results.update(morphological_results)
            
            if verbose:
                self.morphological_metrics.print_results(morphological_results)
        
        # Run MorphScore metrics if available
        if self.morphscore_metrics and include_morphscore:
            logger.info("Computing MorphScore metrics...")
            morphscore_results = self.morphscore_metrics.compute(tokenized_data)
            results.update(morphscore_results)
            
            if verbose:
                self.morphscore_metrics.print_results(morphscore_results)
        
        # Save tokenized data if requested
        if save_tokenized_data:
            if not tokenized_data_path:
                tokenized_data_path = f"{self.plot_save_dir}/tokenized_data.pkl"
            self._save_tokenized_data(tokenized_data, tokenized_data_path)
        
        # Generate plots
        if save_plots:
            logger.info("Generating plots...")
            self.visualizer.generate_all_plots(results, print_pairwise=False)
        
        logger.info("Analysis completed successfully!")
        return results
    
    def run_grouped_analysis(self,
                           group_by: Union[str, List[str]] = ['script_families', 'resource_levels'],
                           save_plots: bool = True,
                           base_results: Optional[Dict[str, Any]] = None,
                           reference_line_method: str = 'macro') -> Dict[str, Dict[str, Any]]:
        """
        Run analysis grouped by language categories.
        
        Args:
            group_by: Group type(s) to analyze by
            save_plots: Whether to generate grouped plots
            base_results: Optional pre-computed results to filter instead of recomputing
            reference_line_method: Method for reference lines ('macro' for average across groups, 'micro' for overall global)
            
        Returns:
            Dictionary mapping group types to group analysis results
        """
        if not self.language_metadata:
            raise ValueError("Language metadata required for grouped analysis")
        
        if isinstance(group_by, str):
            group_by = [group_by]
        
        grouped_results = {}
        
        for group_type in group_by:
            logger.info(f"Running grouped analysis by {group_type}")
            
            if group_type not in self.language_metadata.analysis_groups:
                logger.warning(f"Group type {group_type} not found in language metadata")
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
                
                # Basic metrics
                basic_results = self.basic_metrics.compute(filtered_data)
                group_result.update(basic_results)
                
                # Information-theoretic metrics (includes compression_ratio)
                info_results = self.info_metrics.compute(filtered_data)
                group_result.update(info_results)
                
                # Gini metrics
                gini_results = self.gini_metrics.compute(filtered_data)
                group_result.update(gini_results)
                
                # Morphological metrics - filter from base results if available to avoid recomputation
                if self.morphological_metrics and base_results and 'morphological_alignment' in base_results:
                    logger.info(f"Filtering morphological results for group {group_name} (avoiding recomputation)")
                    morphological_results = self._filter_morphological_results(
                        base_results['morphological_alignment'], group_languages
                    )
                    group_result['morphological_alignment'] = morphological_results
                elif self.morphological_metrics:
                    logger.info(f"Computing morphological results for group {group_name}")
                    morphological_results = self.morphological_metrics.compute(filtered_data)
                    group_result.update(morphological_results)
                
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
                
                group_results[group_name] = group_result
            
            grouped_results[group_type] = group_results
        
        # Generate grouped plots
        if save_plots and grouped_results:
            logger.info("Generating grouped plots...")
            self.visualizer.plot_grouped_analysis(grouped_results, reference_line_method=reference_line_method)
        
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
    
    def _filter_morphological_results(self, morph_results: Dict[str, Any], target_languages: List[str]) -> Dict[str, Any]:
        """Filter morphological results to include only specified languages."""
        filtered_results = {
            'per_tokenizer': {},
            'summary': {}
        }
        
        # Filter per-tokenizer results
        for tok_name, tok_data in morph_results.get('per_tokenizer', {}).items():
            filtered_tok_data = {}
            
            # Filter each metric type
            for metric_type, metric_data in tok_data.items():
                if isinstance(metric_data, dict):
                    filtered_metric_data = {}
                    for lang, lang_data in metric_data.items():
                        if lang in target_languages:
                            filtered_metric_data[lang] = lang_data
                    if filtered_metric_data:
                        filtered_tok_data[metric_type] = filtered_metric_data
                else:
                    # Non-dict data (e.g., scalars) - keep as is
                    filtered_tok_data[metric_type] = metric_data
            
            if filtered_tok_data:
                filtered_results['per_tokenizer'][tok_name] = filtered_tok_data
        
        # Filter summary if it exists
        if 'summary' in morph_results:
            # Summary typically contains aggregate statistics that should be recomputed
            # For now, copy the original summary (could be improved to recompute)
            filtered_results['summary'] = morph_results['summary']
        
        # Add any metadata
        if 'metadata' in morph_results:
            filtered_results['metadata'] = morph_results['metadata']
        
        return filtered_results
    
    def _filter_morphscore_results(self, morphscore_results: Dict[str, Any], target_languages: List[str]) -> Dict[str, Any]:
        """Filter MorphScore results to include only specified languages and recompute summary statistics."""
        import numpy as np
        
        filtered_results = {
            'per_tokenizer': {},
            'summary': {}
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
        
        # Recompute global summary based on filtered tokenizer summaries
        if filtered_results['per_tokenizer']:
            # Aggregate summary across all tokenizers for the filtered group
            all_recall_values = []
            all_precision_values = []
            all_micro_f1_values = []
            all_macro_f1_values = []
            total_langs_evaluated = 0
            total_samples_all = 0
            
            for tok_data in filtered_results['per_tokenizer'].values():
                if 'summary' in tok_data:
                    summary = tok_data['summary']
                    # Use per-tokenizer averages weighted by number of languages
                    langs_count = summary.get('languages_evaluated', 0)
                    if langs_count > 0:
                        all_recall_values.append(summary['avg_morphscore_recall'])
                        all_precision_values.append(summary['avg_morphscore_precision'])
                        all_micro_f1_values.append(summary['avg_micro_f1'])
                        all_macro_f1_values.append(summary['avg_macro_f1'])
                        total_langs_evaluated += langs_count
                        total_samples_all += summary.get('total_samples', 0)
            
            if all_recall_values:
                filtered_results['summary'] = {
                    'avg_morphscore_recall': np.mean(all_recall_values),
                    'avg_morphscore_precision': np.mean(all_precision_values),
                    'avg_micro_f1': np.mean(all_micro_f1_values),
                    'avg_macro_f1': np.mean(all_macro_f1_values),
                    'total_languages_evaluated': total_langs_evaluated,
                    'total_samples': total_samples_all,
                    'avg_morphscore_recall_std_err': np.std(all_recall_values) / np.sqrt(len(all_recall_values)) if len(all_recall_values) > 1 else 0.0,
                    'avg_morphscore_precision_std_err': np.std(all_precision_values) / np.sqrt(len(all_precision_values)) if len(all_precision_values) > 1 else 0.0
                }
        
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
            
            print(f"\n📊 FERTILITY ANALYSIS ({measurement_method})")
            print("-" * 40)
            
            for tok_name in self.tokenizer_names:
                if tok_name in fertility_data['per_tokenizer']:
                    global_stats = fertility_data['per_tokenizer'][tok_name]['global']
                    mean_fertility = global_stats.get('mean', 0.0)
                    std_fertility = global_stats.get('std', 0.0)
                    print(f"{tok_name:20}: {mean_fertility:.3f} ± {std_fertility:.3f} tokens/{measurement_method[:-1]}")
        
        # Print token length results
        if 'token_length' in results:
            print(f"\n📏 TOKEN LENGTH ANALYSIS")
            print("-" * 40)
            
            for tok_name in self.tokenizer_names:
                if tok_name in results['token_length']['per_tokenizer']:
                    char_stats = results['token_length']['per_tokenizer'][tok_name]['character_length']
                    mean_length = char_stats.get('mean', 0.0)
                    std_length = char_stats.get('std', 0.0)
                    print(f"{tok_name:20}: {mean_length:.2f} ± {std_length:.2f} chars/token")
        
        # Print vocabulary utilization
        if 'vocabulary_utilization' in results:
            print(f"\n📚 VOCABULARY UTILIZATION")
            print("-" * 40)
            
            for tok_name in self.tokenizer_names:
                if tok_name in results['vocabulary_utilization']['per_tokenizer']:
                    util_data = results['vocabulary_utilization']['per_tokenizer'][tok_name]
                    utilization = util_data.get('global_utilization', 0.0)
                    used_tokens = util_data.get('global_used_tokens', 0)
                    vocab_size = util_data.get('global_vocab_size', 0)
                    print(f"{tok_name:20}: {utilization:.1%} ({used_tokens:,}/{vocab_size:,} tokens)")
        
        # Print type-token ratio
        if 'type_token_ratio' in results:
            print(f"\n🔤 TYPE-TOKEN RATIO")
            print("-" * 40)
            
            for tok_name in self.tokenizer_names:
                if tok_name in results['type_token_ratio']['per_tokenizer']:
                    ttr_data = results['type_token_ratio']['per_tokenizer'][tok_name]
                    ttr = ttr_data.get('global_ttr', 0.0)
                    types = ttr_data.get('global_types', 0)
                    tokens = ttr_data.get('global_tokens', 0)
                    print(f"{tok_name:20}: {ttr:.4f} ({types:,} types / {tokens:,} tokens)")
        
        print("\n" + "="*60)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analysis configuration and capabilities."""
        return {
            'tokenizer_names': self.tokenizer_names,
            'num_tokenizers': len(self.tokenizer_names),
            'languages': self.input_provider.get_languages(),
            'num_languages': len(self.input_provider.get_languages()),
            'vocab_sizes': {name: self.input_provider.get_vocab_size(name) for name in self.tokenizer_names},
            'measurement_method': self.measurement_config.method.value,
            'has_language_metadata': self.language_metadata is not None,
            'analysis_groups': list(self.language_metadata.analysis_groups.keys()) if self.language_metadata else [],
            'plot_save_dir': self.plot_save_dir
        }
    
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
            table_types: List of table types to generate. Options: 'basic', 'information', 'morphological', 'comprehensive'
            metrics: Dict mapping table types to specific metrics to include
            **formatting_options: Additional formatting options for LaTeX tables
            
        Returns:
            Dict mapping table types to LaTeX table strings
        """
        if output_dir is None:
            output_dir = os.path.join(self.plot_save_dir, "latex_tables")
        
        if table_types is None:
            table_types = ['basic', 'comprehensive']
        
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
                    
                elif table_type == 'morphological':
                    table_content = latex_generator.generate_morphological_table(
                        metrics.get('morphological', None)
                    )
                    caption = "Morphological Alignment Metrics"
                    label = "tab:morphological_metrics"
                    
                elif table_type == 'comprehensive':
                    table_content = latex_generator.generate_comprehensive_table(
                        metrics.get('comprehensive', None)
                    )
                    caption = "Comprehensive Tokenizer Analysis"
                    label = "tab:comprehensive_metrics"
                    
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
            latex_generator.save_table(table_content, output_path)#, caption, label)
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
            
            # Try to get actual vocabulary tokens if available
            try:
                tokenizer = self.input_provider.get_tokenizer(tok_name)
                if hasattr(tokenizer, 'get_vocab'):
                    # Get vocabulary mapping and sort by token IDs
                    vocab_dict = tokenizer.get_vocab()
                    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
                    tokens = [token for token, _ in sorted_vocab]
                elif hasattr(tokenizer, 'vocab'):
                    # Alternative vocabulary access
                    vocab_dict = tokenizer.vocab
                    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
                    tokens = [token for token, _ in sorted_vocab]
                else:
                    # Fallback: create dummy tokens
                    vocab_size = self.input_provider.get_vocab_size(tok_name)
                    tokens = [f"<token_{i}>" for i in range(vocab_size)]
            except:
                # Fallback: create dummy tokens if tokenizer access fails
                vocab_size = self.input_provider.get_vocab_size(tok_name)
                tokens = [f"<token_{i}>" for i in range(vocab_size)]
            
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


def create_analyzer_from_input_provider(input_provider: InputProvider,
                                       **kwargs) -> UnifiedTokenizerAnalyzer:
    """
    Create analyzer from existing InputProvider.
    
    Args:
        input_provider: InputProvider instance
        **kwargs: Additional arguments for UnifiedTokenizerAnalyzer
        
    Returns:
        UnifiedTokenizerAnalyzer instance
    """
    return UnifiedTokenizerAnalyzer(input_provider, **kwargs)