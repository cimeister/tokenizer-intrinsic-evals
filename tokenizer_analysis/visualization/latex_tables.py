"""
LaTeX table generation for tokenizer analysis results.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class LaTeXTableGenerator:
    """Generate LaTeX tables from tokenizer analysis results."""
    
    def __init__(self, results: Dict[str, Any], tokenizer_names: List[str]):
        """
        Initialize LaTeX table generator.
        
        Args:
            results: Analysis results dictionary
            tokenizer_names: List of tokenizer names to include
        """
        self.results = results
        self.tokenizer_names = tokenizer_names
        
        # Default formatting options
        self.decimal_places = 3
        self.bold_best = True
        self.include_std_err = True
        self.std_err_size = "\\small"
        
        # Metric configurations
        self.metric_configs = {
            'fertility': {
                'title': 'Fertility',
                'key_path': ['fertility', 'per_tokenizer'],
                'value_key': 'global',
                'stat_key': 'mean',
                'err_key': 'std_err',
                'format': '{:.3f}',
                'lower_is_better': True
            },
            'vocabulary_utilization': {
                'title': 'Vocab Utilization',
                'key_path': ['vocabulary_utilization', 'per_tokenizer'],
                'value_key': 'global_utilization',
                'stat_key': None,
                'err_key': None,
                'format': '{:.1f}\\%',
                'lower_is_better': False,
                'multiplier': 100. #Turn into percentage but with escaped % for LaTeX
            },
            'type_token_ratio': {
                'title': 'TTR',
                'key_path': ['type_token_ratio', 'per_tokenizer'],
                'value_key': 'global_ttr',
                'stat_key': None,
                'err_key': None,
                'format': '{:.4f}',
                'lower_is_better': False
            },
            'renyi_1.0': {
                'title': 'Rényi ($\\alpha$=1)',
                'key_path': ['renyi_efficiency', 'per_tokenizer'],
                'value_key': 'renyi_1.0',
                'stat_key': 'overall',
                'err_key': None,
                'format': '{:.2f}',
                'lower_is_better': False
            },
            'renyi_2.5': {
                'title': 'Rényi ($\\alpha$=2.5)',
                'key_path': ['renyi_efficiency', 'per_tokenizer'],
                'value_key':  'renyi_2.5',
                'stat_key': 'overall',
                'err_key': None,
                'format': '{:.2f}',
                'lower_is_better': False
            },
            'tokenizer_fairness_gini': {
                'title': 'Gini',
                'key_path': ['tokenizer_fairness_gini', 'per_tokenizer'],
                'value_key': 'gini_coefficient',
                'stat_key': None,
                'err_key': None,
                'format': '{:.3f}',
                'lower_is_better': True
            },
            'morphological_alignment': {
                'title': 'Morphological F1',
                'key_path': ['morphological_alignment', 'summary'],
                'value_key': 'avg_boundary_f1',
                'stat_key': None,
                'err_key': 'avg_boundary_f1_std_err',
                'format': '{:.3f}',
                'lower_is_better': False
            },
            'morphscore_recall': {
                'title': 'MorphScore Recall',
                'key_path': ['morphscore', 'per_tokenizer'],
                'value_key': 'summary',
                'stat_key': 'avg_morphscore_recall',
                'err_key': 'avg_morphscore_recall_std_err',
                'format': '{:.3f}',
                'lower_is_better': False
            },
            'morphscore_precision': {
                'title': 'MorphScore Precision',
                'key_path': ['morphscore', 'per_tokenizer'],
                'value_key': 'summary',
                'stat_key': 'avg_morphscore_precision',
                'err_key': 'avg_morphscore_precision_std_err',
                'format': '{:.3f}',
                'lower_is_better': False
            },
            'avg_token_rank': {
                'title': 'Avg Token Rank',
                'key_path': ['unigram_distribution_metrics', 'per_tokenizer'],
                'value_key': 'global_avg_token_rank',
                'stat_key': None,
                'err_key': None,
                'format': '{:.1f}',
                'lower_is_better': True
            },
            'token_length': {
                'title': 'Token Length',
                'key_path': ['token_length', 'per_tokenizer'],
                'value_key': 'character_length',
                'stat_key': 'mean',
                'err_key': 'std_err',
                'format': '{:.2f}',
                'lower_is_better': False
            },
            'avg_tokens_per_line': {
                'title': 'Tokens per Line',
                'key_path': ['avg_tokens_per_line', 'per_tokenizer'],
                'value_key': 'global_avg',
                'stat_key': None,
                'err_key': 'global_std_err',
                'format': '{:.1f}',
                'lower_is_better': False
            },
            'compression_rate': {
                'title': 'Comp. Rate',
                'key_path': ['compression_rate', 'per_tokenizer'],
                'value_key': 'global',
                'stat_key': 'compression_rate',
                'err_key': None,
                'format': '{:.4f}',
                'lower_is_better': False
            }
        }
    
    def _extract_metric_value(self, metric_config: Dict[str, Any], tokenizer_name: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract metric value and error for a tokenizer.
        
        Args:
            metric_config: Metric configuration
            tokenizer_name: Name of tokenizer
            
        Returns:
            Tuple of (value, error) or (None, None) if not found
        """
        try:
            # Navigate to the data using key_path
            data = self.results
            for key in metric_config['key_path']:
                if key not in data:
                    return None, None
                data = data[key]
            
            # Get tokenizer-specific data
            if tokenizer_name not in data:
                return None, None
            
            tokenizer_data = data[tokenizer_name]
            
            # Extract value
            if metric_config['stat_key']:
                # Value is nested (e.g., global.mean)
                value_data = tokenizer_data.get(metric_config['value_key'], {})
                if isinstance(value_data, dict):
                    value = value_data.get(metric_config['stat_key'])
                else:
                    value = value_data
            else:
                # Value is direct
                value = tokenizer_data.get(metric_config['value_key'])
            
            # Extract error if available
            error = None
            if metric_config['err_key'] and self.include_std_err:
                if metric_config['stat_key']:
                    error_data = tokenizer_data.get(metric_config['value_key'], {})
                    if isinstance(error_data, dict):
                        error = error_data.get(metric_config['err_key'])
                else:
                    error = tokenizer_data.get(metric_config['err_key'])
            
            return value, error
            
        except Exception as e:
            logger.warning(f"Error extracting metric {metric_config['title']} for {tokenizer_name}: {e}")
            return None, None
    
    def _wrap_column_title(self, title: str, max_length: int = 15) -> str:
        """
        Wrap long column titles into multiple lines for LaTeX.
        
        Args:
            title: Original column title
            max_length: Maximum length per line before wrapping
            
        Returns:
            LaTeX-formatted title with line breaks
        """
        if len(title) <= max_length:
            return title
            
        # Split on spaces and wrap
        words = title.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_length:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Use \makecell for multi-line columns
        if len(lines) > 1:
            line_separator = ' \\\\ '
            return f"\\makecell{{{line_separator.join(lines)}}}"
        else:
            return lines[0] if lines else title
    
    def _format_value(self, value: float, error: Optional[float], format_str: str, is_best: bool = False, multiplier: Optional[float] = None) -> str:
        """
        Format a value with optional error and styling.
        
        Args:
            value: The main value
            error: Optional error value
            format_str: Format string for the value
            is_best: Whether this is the best value (for bolding)
            
        Returns:
            Formatted LaTeX string
        """
        if value is None:
            return "---"
        
        # Format main value
        formatted_value = format_str.format(value*multiplier if multiplier else value)
        
        # Add error if available
        if error is not None and not np.isnan(error):
            error_str = f" {{{self.std_err_size} $\pm$ {format_str.format(error)}}}"
            formatted_value += error_str
        
        # Bold if best
        if is_best and self.bold_best:
            formatted_value = f"\\textbf{{{formatted_value}}}"
        
        return formatted_value
    
    def _find_best_values(self, values: Dict[str, Tuple[Optional[float], Optional[float]]], 
                         lower_is_better: bool) -> str:
        """
        Find the best tokenizer for a metric.
        
        Args:
            values: Dict mapping tokenizer names to (value, error) tuples
            lower_is_better: Whether lower values are better
            
        Returns:
            Name of best tokenizer or empty string if no valid values
        """
        valid_values = {name: val for name, (val, _) in values.items() if val is not None}
        
        if not valid_values:
            return ""
        
        if lower_is_better:
            best_tokenizer = min(valid_values.keys(), key=lambda k: valid_values[k])
        else:
            best_tokenizer = max(valid_values.keys(), key=lambda k: valid_values[k])
        
        return best_tokenizer
    
    def generate_basic_metrics_table(self, metrics: List[str] = None) -> str:
        """
        Generate LaTeX table for basic metrics.
        
        Args:
            metrics: List of metrics to include. If None, uses default set.
            
        Returns:
            LaTeX table string
        """
        if metrics is None:
            metrics = [
                'compression_rate',
                'type_token_ratio',
                'vocabulary_utilization',
                'fertility',
                'renyi_2.5',
                'tokenizer_fairness_gini',
                'morphscore_precision',
                'morphscore_recall',
            ]
        
        # Validate metrics
        valid_metrics = [m for m in metrics if m in self.metric_configs]
        if not valid_metrics:
            logger.warning("No valid metrics found for basic table")
            return ""
        
        # Create table header
        num_cols = len(valid_metrics) + 1  # +1 for tokenizer name column
        col_spec = "l" + "c" * len(valid_metrics)
        
        # Create header with wrapped titles + a direction-of-goodness arrow
        # ($\downarrow$ when lower is better, $\uparrow$ otherwise). We append
        # the arrow after wrapping so it always appears on the last line of a
        # multi-line header.
        header_titles = []
        for m in valid_metrics:
            cfg = self.metric_configs[m]
            arrow = "$\\downarrow$" if cfg.get('lower_is_better') else "$\\uparrow$"
            wrapped_title = self._wrap_column_title(cfg['title'])
            if wrapped_title.startswith("\\makecell{") and wrapped_title.endswith("}"):
                # Append the arrow inside the last makecell line.
                inner = wrapped_title[len("\\makecell{"):-1]
                header_titles.append(f"\\makecell{{{inner} \\\\ ({arrow})}}")
            else:
                header_titles.append(f"\\makecell{{{wrapped_title} \\\\ ({arrow})}}")
        
        table_lines = [
            "% Requires \\usepackage{makecell} in preamble",
            "\\begin{tabular}{" + col_spec + "}",
            "\\toprule",
            "Tokenizer & " + " & ".join(header_titles) + " \\\\",
            "\\midrule"
        ]
        
        # Extract values for all metrics and tokenizers
        all_values = {}
        for metric in valid_metrics:
            all_values[metric] = {}
            for tokenizer in self.tokenizer_names:
                value, error = self._extract_metric_value(self.metric_configs[metric], tokenizer)
                all_values[metric][tokenizer] = (value, error)
        
        # Find best values for each metric
        best_tokenizers = {}
        for metric in valid_metrics:
            best_tokenizers[metric] = self._find_best_values(
                all_values[metric], 
                self.metric_configs[metric]['lower_is_better']
            )
        
        # Generate table rows
        for tokenizer in self.tokenizer_names:
            row_values = []
            for metric in valid_metrics:
                value, error = all_values[metric][tokenizer]
                is_best = best_tokenizers[metric] == tokenizer
                formatted_value = self._format_value(
                    value, error, 
                    self.metric_configs[metric]['format'], 
                    is_best,
                    self.metric_configs[metric].get('multiplier')
                )
                row_values.append(formatted_value)
            
            # Clean tokenizer name for LaTeX
            clean_name = tokenizer.replace("_", "\\_")
            row_line = clean_name + " & " + " & ".join(row_values) + " \\\\"
            table_lines.append(row_line)
        
        # Close table
        table_lines.extend([
            "\\bottomrule",
            "\\end{tabular}"
        ])
        
        return "\n".join(table_lines)
    
    def generate_information_theory_table(self, metrics: List[str] = None) -> str:
        """
        Generate LaTeX table for information theory metrics.
        
        Args:
            metrics: List of metrics to include. If None, uses default set.
            
        Returns:
            LaTeX table string
        """
        if metrics is None:
            metrics = ['renyi_1.0', 'avg_token_rank']
        
        return self.generate_basic_metrics_table(metrics)
    
    def generate_morphological_table(self, metrics: List[str] = None) -> str:
        """
        Generate LaTeX table for morphological metrics.
        
        Args:
            metrics: List of metrics to include. If None, uses default set.
            
        Returns:
            LaTeX table string
        """
        if metrics is None:
            metrics = ['morphological_alignment', 'morphscore_recall', 'morphscore_precision']
        
        return self.generate_basic_metrics_table(metrics)
    
    def generate_comprehensive_table(self, metrics: List[str] = None) -> str:
        """
        Generate comprehensive LaTeX table with all available metrics.
        
        Args:
            metrics: List of metrics to include. If None, uses all available.
            
        Returns:
            LaTeX table string
        """
        if metrics is None:
            # Use all available metrics that have data
            metrics = []
            for metric_key in self.metric_configs:
                # Check if any tokenizer has data for this metric
                has_data = False
                for tokenizer in self.tokenizer_names:
                    value, _ = self._extract_metric_value(self.metric_configs[metric_key], tokenizer)
                    if value is not None:
                        has_data = True
                        break
                
                if has_data:
                    metrics.append(metric_key)
        
        return self.generate_basic_metrics_table(metrics)
    
    def save_table(self, table_content: str, output_path: str, 
                   caption: str = None, label: str = None) -> None:
        """
        Save LaTeX table to file with optional caption and label.
        
        Args:
            table_content: LaTeX table content
            output_path: Output file path
            caption: Optional table caption
            label: Optional table label
        """
        lines = []
        
        if caption or label:
            lines.append("\\begin{table}[htbp]")
            lines.append("\\centering")
        
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        
        if label:
            lines.append(f"\\label{{{label}}}")
        
        lines.append(table_content)
        
        if caption or label:
            lines.append("\\end{table}")
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        logger.info(f"LaTeX table saved to {output_path}")
    
    def set_formatting_options(self, decimal_places: int = None, bold_best: bool = None, 
                              include_std_err: bool = None, std_err_size: str = None) -> None:
        """
        Update formatting options.
        
        Args:
            decimal_places: Number of decimal places for numeric values
            bold_best: Whether to bold the best values
            include_std_err: Whether to include standard errors
            std_err_size: LaTeX size command for standard errors
        """
        if decimal_places is not None:
            self.decimal_places = decimal_places
        if bold_best is not None:
            self.bold_best = bold_best
        if include_std_err is not None:
            self.include_std_err = include_std_err
        if std_err_size is not None:
            self.std_err_size = std_err_size