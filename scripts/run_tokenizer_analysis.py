"""
Unified tokenizer analysis script supporting both raw tokenizers and pre-tokenized data.

Raw tokenizer examples:
python scripts/run_tokenizer_analysis.py --use-sample-data
python scripts/run_tokenizer_analysis.py --tokenizer-config configs/tokenizer_config.json --language-config configs/language_config.json --morphological-config configs/morphological_config.json --measurement-config configs/text_measurement_config_bytes.json --samples-per-lang 3000 --output-dir analysis_results --verbose --run-grouped-analysis

Pre-tokenized data examples:
python scripts/run_tokenizer_analysis.py --tokenized-data-file tokenized_data.json --language-config configs/language_config.json
python scripts/run_tokenizer_analysis.py --tokenized-data-file tokenized_data.pkl --tokenized-data-config tokenized_config.json --language-config configs/language_config.json --run-grouped-analysis
"""
import logging
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from tokenizer_analysis import create_analyzer_from_raw_inputs, create_analyzer_from_tokenized_data
from tokenizer_analysis.utils import setup_environment
from tokenizer_analysis.config.language_metadata import LanguageMetadata
from tokenizer_analysis.loaders.multilingual_data import load_multilingual_data
from tokenizer_analysis.core.input_utils import InputLoader
from tokenizer_analysis.constants import (
    TextProcessing,
    DataProcessing,
    MIN_TOKENIZERS_FOR_PLOTS
)
from tokenizer_analysis.visualization.visualization_config import LaTeXFormatting

# Setup environment
setup_environment()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tokenizer_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config_from_file(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_sample_configs() -> Dict[str, Dict]:
    """Create sample tokenizer configurations for testing."""
    return {
        "bpe": {
            "class": "huggingface",
            "path": "sample_tokenizers/bpe.json"
        },
        "unigramlm": {
            "class": "huggingface",
            "path": "sample_tokenizers/unigramlm.json"
        }
    }


def create_sample_language_metadata() -> str:
    """Create sample LanguageMetadata configuration and return path to temp file."""
    import tempfile
    
    sample_metadata = {
        "languages": {
            "eng_Latn": {
                "name": "English",
                "iso_code": "en", 
                "data_path": "parallel/en/eval.txt"
            },
            "spa_Latn": {
                "name": "Spanish",
                "iso_code": "es",
                "data_path": "parallel/es/eval.txt"
            },
            "deu_Latn": {
                "name": "German", 
                "iso_code": "de",
                "data_path": "parallel/de/eval.txt"
            },
            "arb_Arab": {
                "name": "Arabic",
                "iso_code": "ar",
                "data_path": "parallel/ar/eval.txt"
            },
            "rus_Cyrl": {
                "name": "Russian",
                "iso_code": "ru",
                "data_path": "parallel/ru/eval.txt"
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
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_metadata, f, indent=2)
        return f.name


def create_sample_morphological_config() -> Dict[str, str]:
    """Create sample morphological dataset configuration."""
    return {}


def create_sample_morphscore_config(data_dir: str = "morphscore_data") -> Dict[str, any]:
    """Create sample MorphScore configuration with default settings."""
    return {
        "data_dir": data_dir,
        "language_subset": None,
        "by_split": False,
        "freq_scale": True,
        "exclude_single_tok": False
    }

def slim_results_for_json(results: Dict) -> Dict:
    """Create a slimmed-down version of results for JSON export."""
    slimmed = {}
    
    # Keep only essential summary statistics, not raw data
    for metric_name, metric_data in results.items():
        if isinstance(metric_data, dict):
            slimmed_metric = {}
            
            # For per-tokenizer results, keep only summary stats
            if 'per_tokenizer' in metric_data:
                slimmed_metric['per_tokenizer'] = {}
                for tok_name, tok_data in metric_data['per_tokenizer'].items():
                    if isinstance(tok_data, dict):
                        # Keep essential stats but remove raw arrays
                        tok_summary = {}
                        for key, value in tok_data.items():
                            if key in ['global', 'global_ttr', 'global_utilization', 'global_avg']:
                                if isinstance(value, dict):
                                    # Keep only mean/std, not raw values
                                    filtered_value = {k: v for k, v in value.items() 
                                                    if k in ['mean', 'std', 'median', 'count'] and not k.endswith('_lengths')}
                                    tok_summary[key] = filtered_value
                                else:
                                    tok_summary[key] = value
                            elif key == 'per_language':
                                # Include per-language results for analysis
                                tok_summary[key] = value
                            elif key.startswith('renyi_') and isinstance(value, dict):
                                # Keep overall entropy values but not per-language details
                                tok_summary[key] = {'overall': value.get('overall')}
                            elif key in ['gini_coefficient', 'mean_cost', 'std_cost', 'min_cost', 'max_cost', 
                                        'cost_ratio', 'most_efficient_language', 'least_efficient_language', 
                                        'num_languages', 'language_costs', 'warning']:
                                # Keep all Gini-related metrics
                                tok_summary[key] = value
                            elif key in ['sorted_languages', 'sorted_costs', 'total_cost', 'n_languages', 
                                        'x_values', 'y_values', 'equality_line']:
                                # Keep Lorenz curve data but limit array sizes if needed
                                if isinstance(value, list) and len(value) > TextProcessing.LARGE_ARRAY_THRESHOLD:
                                    # For very large arrays, keep only key points
                                    step = len(value) // TextProcessing.ARRAY_SAMPLING_POINTS  # Keep ~50 points
                                    tok_summary[key] = value[::step] if step > 1 else value[:TextProcessing.ARRAY_SAMPLING_POINTS]
                                else:
                                    tok_summary[key] = value
                        slimmed_metric['per_tokenizer'][tok_name] = tok_summary
            
            # Keep pairwise comparisons (they're already summary data)
            if 'pairwise_comparisons' in metric_data:
                slimmed_metric['pairwise_comparisons'] = metric_data['pairwise_comparisons']
            
            # Keep vocabulary sizes
            if 'vocabulary_sizes' in metric_data:
                slimmed_metric['vocabulary_sizes'] = metric_data['vocabulary_sizes']
            
            # Keep summary stats for morphological analysis
            if metric_name == 'morphological_alignment' and 'summary' in metric_data:
                slimmed_metric['summary'] = metric_data['summary']
            
            # Keep summary stats for MorphScore analysis
            if metric_name == 'morphscore' and 'summary' in metric_data:
                slimmed_metric['summary'] = metric_data['summary']
            
            # Keep metadata for Gini metrics
            if metric_name in ['tokenizer_fairness_gini', 'lorenz_curve_data'] and 'metadata' in metric_data:
                slimmed_metric['metadata'] = metric_data['metadata']
            
            # Keep global results
            if 'global' in metric_data and metric_name not in ['morphological_alignment', 'morphscore']:
                slimmed_metric['global'] = metric_data['global']
            
            # Include per-language results at the top level
            if 'per_language' in metric_data:
                slimmed_metric['per_language'] = metric_data['per_language']
            
            slimmed[metric_name] = slimmed_metric
        else:
            slimmed[metric_name] = metric_data
    
    return slimmed


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Enhanced modular tokenizer analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-tokenizer analysis with raw tokenizers (supports any number of tokenizers)
  python scripts/run_tokenizer_analysis.py --use-sample-data
  
  # Load from configuration files (supports 1+ tokenizers)
  python scripts/run_tokenizer_analysis.py --tokenizer-config tokenizers.json --language-config languages.json
  
  # Use pre-tokenized data from file
  python scripts/run_tokenizer_analysis.py --tokenized-data-file tokenized_data.json --language-config languages.json
  
  # Use pre-tokenized data with configuration
  python scripts/run_tokenizer_analysis.py --tokenized-data-file tokenized_data.pkl --tokenized-data-config tokenized_config.json --language-config languages.json
  
  # Filter by script family and run grouped analysis (includes grouped plots)
  python scripts/run_tokenizer_analysis.py --use-sample-data --filter-script-family Latin --run-grouped-analysis
  
  # Filter by resource level  
  python scripts/run_tokenizer_analysis.py --use-sample-data --filter-resource-level high
  
  # Run grouped analysis across all script families and resource levels
  python scripts/run_tokenizer_analysis.py --use-sample-data --run-grouped-analysis
  
  # Pairwise comparison only (restricts to 2 specific tokenizers)
  python scripts/run_tokenizer_analysis.py --pairwise tok1 tok2 --use-sample-data
  
  # Skip morphological analysis and plots for faster processing
  python scripts/run_tokenizer_analysis.py --use-sample-data --no-plots
  
  # Enable MorphScore analysis with default settings
  python scripts/run_tokenizer_analysis.py --use-sample-data --morphscore
  
  # Use custom MorphScore configuration
  python scripts/run_tokenizer_analysis.py --tokenizer-config tokenizers.json --language-config languages.json --morphscore-config morphscore.json
  
  # Explicitly disable MorphScore (useful when config file has it enabled)
  python scripts/run_tokenizer_analysis.py --use-sample-data
  
  # Generate LaTeX tables for results
  python scripts/run_tokenizer_analysis.py --use-sample-data --generate-latex-tables
  
  # Generate specific LaTeX table types
  python scripts/run_tokenizer_analysis.py --use-sample-data --generate-latex-tables --latex-table-types basic morphological
  
  # Generate per-language plots in addition to standard plots
  python scripts/run_tokenizer_analysis.py --use-sample-data --per-language-plots
  
  # Generate per-language plots with additional faceted plots (subplots per tokenizer)
  python scripts/run_tokenizer_analysis.py --use-sample-data --per-language-plots --faceted-plots
  
  # Generate grouped analysis with additional faceted plots for grouped metrics
  python scripts/run_tokenizer_analysis.py --use-sample-data --run-grouped-analysis --faceted-plots
  
  # Generate custom LaTeX tables from configuration file
  python scripts/run_tokenizer_analysis.py --use-sample-data --custom-latex-config custom_tables.json
  
  # Save both summary and full detailed results
  python scripts/run_tokenizer_analysis.py --use-sample-data --save-full-results
  
  # Save tokenized data for later reuse
  python scripts/run_tokenizer_analysis.py --use-sample-data --save-tokenized-data
  
  # Save tokenized data to specific path
  python scripts/run_tokenizer_analysis.py --tokenizer-config tokenizers.json --language-config languages.json --save-tokenized-data --tokenized-data-output-path my_tokenized_data.pkl
  
  # Complete workflow: generate tokenized data, then reuse it
  # Step 1: Generate and save tokenized data
  python scripts/run_tokenizer_analysis.py --use-sample-data --save-tokenized-data --tokenized-data-output-path results/tokenized_data.pkl
  # Step 2: Use the saved tokenized data (much faster)
  python scripts/run_tokenizer_analysis.py --tokenized-data-file results/tokenized_data.pkl --tokenized-data-config configs/sample_tokenized_config.json --language-config languages.json
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
        "--morphological-config",
        type=str,
        help="JSON file with morphological dataset configurations"
    )
    parser.add_argument(
        "--morphscore-config",
        type=str,
        help="JSON file with MorphScore configuration (requires raw tokenization)"
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
        help="Directory containing morphological data for MorphScore analysis"
    )
    parser.add_argument(
        "--measurement-config",
        type=str,
        help="JSON file with text measurement configuration (method, counting functions, etc.)"
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Use sample/demo data for testing"
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
        default=DataProcessing.DEFAULT_MAX_SAMPLES,
        help=f"Number of text samples per language (default: {DataProcessing.DEFAULT_MAX_SAMPLES})"
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
        "--test",
        action="store_true",
        help="Run test functions"
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
        default=["basic", "comprehensive"],
        choices=["basic", "information", "morphological", "comprehensive"],
        help="Types of (default) LaTeX tables to generate"
    )
    parser.add_argument(
        "--latex-output-dir",
        type=str,
        help="Directory for LaTeX table output (default: same as --output-dir)"
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
        help="Generate additional faceted plots (one subplot per tokenizer with shared y-axis) for grouped analysis (--run-grouped-analysis) and per-language plots (--per-language-plots). Normal plots are still generated."
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
    
    args = parser.parse_args()
    
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
        morphological_config = create_sample_morphological_config()
        measurement_config = None  # Use default for sample data
        
        # Configure MorphScore for sample data
        morphscore_config = None
        if args.morphscore or args.morphscore_config:
            if args.morphscore_config:
                morphscore_config = load_config_from_file(args.morphscore_config)
            else:
                morphscore_config = create_sample_morphscore_config(args.morphscore_data_dir)
    elif use_tokenized_data:
        # Pre-tokenized data mode
        if not args.tokenized_data_file:
            raise ValueError("Must specify --tokenized-data-file for pre-tokenized mode")
        
        # Load tokenized data
        tokenized_data = InputLoader.load_from_file(args.tokenized_data_file)
        
        # Load vocabulary files if config provided
        vocabularies = {}
        if args.tokenized_data_config:
            config = load_config_from_file(args.tokenized_data_config)
            if 'vocabulary_files' in config:
                vocabularies = InputLoader.load_vocabularies_from_config(config['vocabulary_files'])
        
        # If no vocabularies loaded, estimate from data
        if not vocabularies:
            logger.warning("No vocabulary files loaded, will estimate vocabulary sizes from tokenized data")
        
        # Still need language config for metadata
        if args.language_config:
            language_config_path = args.language_config
        else:
            language_config_path = create_sample_language_metadata()
            logger.warning("No language config specified, using sample metadata")
        
        morphological_config = None
        if args.morphological_config:
            morphological_config = load_config_from_file(args.morphological_config)
        
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
    else:
        # Raw tokenizer mode
        if not args.tokenizer_config:
            raise ValueError("Must specify --tokenizer-config or use --use-sample-data")
        
        tokenizer_configs = load_config_from_file(args.tokenizer_config)
        
        if args.language_config:
            # Load language configuration (supports both directory and file paths)
            language_config_path = args.language_config
        else:
            language_config_path = create_sample_language_metadata()
            logger.warning("No language config specified, using sample metadata")
        
        morphological_config = None
        if args.morphological_config:
            morphological_config = load_config_from_file(args.morphological_config)
        
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
            morphological_config=morphological_config,
            morphscore_config=morphscore_config,
            show_global_lines=not args.no_global_lines,
            per_language_plots=args.per_language_plots,
            faceted_plots=args.faceted_plots
        )
    else:
        # Raw tokenizer mode
        # Validate tokenizer configs
        if not tokenizer_configs or len(tokenizer_configs) < 1:
            raise ValueError("At least one tokenizer must be configured")
        
        if args.pairwise and len(args.pairwise) == 2:
            tok1, tok2 = args.pairwise
            if tok1 not in tokenizer_configs or tok2 not in tokenizer_configs:
                raise ValueError(f"Pairwise tokenizers {tok1}, {tok2} must be in configuration")
            # Filter to only these two tokenizers
            tokenizer_configs = {tok1: tokenizer_configs[tok1], tok2: tokenizer_configs[tok2]}
        
        # Load language texts
        logger.info("Loading language texts...")
        filter_by_group = None
        if args.filter_script_family:
            filter_by_group = ('script_family', args.filter_script_family)
        elif args.filter_resource_level:
            filter_by_group = ('resource_level', args.filter_resource_level)
        
        language_texts = load_multilingual_data(
            language_metadata=language_metadata,
            max_texts_per_language=args.samples_per_lang,
            filter_by_group=filter_by_group
        )
        
        if not language_texts:
            raise ValueError("No valid language texts loaded")
        
        # Initialize unified analyzer using convenience function
        analyzer = create_analyzer_from_raw_inputs(
            tokenizer_configs=tokenizer_configs,
            language_texts=language_texts,
            measurement_config=measurement_config,
            language_metadata=language_metadata,
            plot_save_dir=args.output_dir,
            morphological_config=morphological_config,
            morphscore_config=morphscore_config,
            show_global_lines=not args.no_global_lines,
            per_language_plots=args.per_language_plots,
            faceted_plots=args.faceted_plots
        )
    if args.test:
        logger.warning("Test methods not yet updated for unified system")
        exit(0)
    
    # Run analysis
    logger.info("Starting tokenizer analysis...")
    
    if args.pairwise:
        # Pairwise comparison - for now, just run regular analysis and filter results
        logger.info(f"Running pairwise comparison: {args.pairwise[0]} vs {args.pairwise[1]}")
        results = analyzer.run_analysis(
            save_plots=not args.no_plots,
            include_morphological=morphological_config is not None,
            include_morphscore=morphscore_config is not None,
            verbose=args.verbose,
            save_tokenized_data=args.save_tokenized_data,
            tokenized_data_path=args.tokenized_data_output_path
        )
    else:
        # Full multi-tokenizer analysis
        results = analyzer.run_analysis(
            save_plots=not args.no_plots,
            include_morphological=morphological_config is not None,
            include_morphscore=morphscore_config is not None,
            verbose=args.verbose,
            save_tokenized_data=args.save_tokenized_data,
            tokenized_data_path=args.tokenized_data_output_path
        )
        
        if args.run_grouped_analysis and analyzer.language_metadata:
            logger.info("Running grouped analysis by script families and resource levels...")
            
            # Use the unified analyzer's built-in grouped analysis
            # Pass base results to avoid recomputing morphological metrics
            grouped_results = analyzer.run_grouped_analysis(
                group_by=analyzer.language_metadata.analysis_groups.keys(),
                save_plots=not args.no_plots,
                base_results=results
            )
            
            # Add grouped results to main results
            results['grouped_analysis'] = grouped_results
    
    # Save results to JSON (slimmed version)
    results_file = Path(args.output_dir) / "analysis_results.json"
    logger.info(f"Saving slimmed results to {results_file}")
    
    # Create slimmed version and convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    slimmed_results = slim_results_for_json(results)
    results_json = convert_for_json(slimmed_results)
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Optionally save full results
    if args.save_full_results:
        full_results_file = Path(args.output_dir) / "analysis_results_full.json"
        logger.info(f"Saving full results to {full_results_file}")
        full_results_json = convert_for_json(results)
        with open(full_results_file, 'w') as f:
            json.dump(full_results_json, f, indent=2)
    
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
            logger.error(f"Error generating LaTeX tables: {e}")
    
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
            logger.error(f"Error generating custom LaTeX tables: {e}")
    
    logger.info("Analysis complete!")
    print(f"\nResults saved to: {args.output_dir}")
    if not args.no_plots:
        print(f"Plots saved to: {args.output_dir}")
    print(f"Summary results: {results_file}")
    if args.save_full_results:
        print(f"Full detailed results: {Path(args.output_dir) / 'analysis_results_full.json'}")

if __name__ == "__main__":
    main()
