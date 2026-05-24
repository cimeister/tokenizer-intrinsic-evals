"""
MorphScore metrics for tokenizer analysis.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging

from .base import BaseMetrics
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider, RawTokenizationProvider
from ..loaders.constants import ISO639_1_to_FLORES

logger = logging.getLogger(__name__)

try:
    from morphscore import MorphScore
    MORPHSCORE_AVAILABLE = True
except ImportError:
    logger.warning("MorphScore library not available. MorphScore metrics will be disabled.")
    MORPHSCORE_AVAILABLE = False


# FLORES codes used in this library that have corresponding morphscore CSVs but
# are absent from MorphScore.FLORES_TO_MS_FILES. Merged into the upstream map
# at construction time so input-data subsetting works without monkey-patching
# every call site.
EXTRA_FLORES_TO_MS_FILES = {
    'ceb_Latn': 'cebuano_data.csv',
    'hrv_Latn': 'croatian_data.csv',
    'isl_Latn': 'icelandic_data.csv',
    'gle_Latn': 'irish_data.csv',
    'kaz_Cyrl': 'kazakh_data.csv',
    'kir_Cyrl': 'kirghiz_data.csv',
    'lit_Latn': 'lithuanian_data.csv',
    'lav_Latn': 'latvian_data.csv',  # alt to lvs_Latn
    'pus_Arab': 'pashto_data.csv',
    'srp_Cyrl': 'serbian_data.csv',
    'snd_Arab': 'sindhi_data.csv',
    'sin_Sinh': 'sinhala_data.csv',
    'slv_Latn': 'slovenian_data.csv',
    'wol_Latn': 'wolof_data.csv',
    'zho_Hant': 'mandarin_data.csv',  # share Mandarin dataset for Hant code
}


class MorphScoreMetrics(BaseMetrics):
    """MorphScore metrics for tokenizer evaluation."""
    
    def __init__(self, 
                 input_provider: InputProvider,
                 data_dir: str = "morphscore_data",
                 language_subset: Optional[List[str]] = None,
                 by_split: bool = False,
                 freq_scale: bool = True,
                 exclude_single_tok: bool = False
                ):
        """
        Initialize MorphScore metrics.
        
        Args:
            input_provider: InputProvider instance
            data_dir: Path to morphological data directory
            language_subset: Optional list of language codes to evaluate
            by_split: Whether to evaluate train/val/test separately
            freq_scale: Whether to use frequency scaling
            exclude_single_tok: Whether to exclude single token words
        """
        super().__init__(input_provider)

        if not MORPHSCORE_AVAILABLE:
            raise ImportError("MorphScore library is required for MorphScore metrics")

        # Validate that input provider supports tokenizer access
        if not isinstance(input_provider, RawTokenizationProvider):
            raise ValueError("MorphScore metrics require RawTokenizationProvider (tokenizer access)")

        # Merge any additional FLORES → CSV entries into the upstream map so that
        # subsetting and dataset resolution see the same key space.
        for k, v in EXTRA_FLORES_TO_MS_FILES.items():
            MorphScore.FLORES_TO_MS_FILES.setdefault(k, v)

        # Store MorphScore configuration
        self.data_dir = data_dir
        self.language_subset = language_subset
        self.by_split = by_split
        self.freq_scale = freq_scale
        self.exclude_single_tok = exclude_single_tok

        # Get available languages from input provider
        self.available_languages = input_provider.get_languages()

        # Use provided subset or all available languages
        if language_subset is None:
            requested_languages = list(self.available_languages)
        else:
            requested_languages = [lang for lang in language_subset if lang in self.available_languages]

        # Translate any ISO 639-1 codes to FLORES form
        requested_languages = [ISO639_1_to_FLORES.get(lang, lang) for lang in requested_languages]

        # Restrict to languages with a morphscore dataset (either an explicit
        # FLORES mapping or a CSV whose stem matches the lowercased FLORES code)
        ms_keys = set(MorphScore.FLORES_TO_MS_FILES.keys())
        from pathlib import Path
        data_root = Path(self.data_dir)
        self.target_languages = []
        skipped = []
        for lang in requested_languages:
            if lang in ms_keys:
                self.target_languages.append(lang)
            elif (data_root / f"{lang.lower()}_data.csv").exists():
                self.target_languages.append(lang)
            else:
                skipped.append(lang)

        if skipped:
            logger.warning(
                f"MorphScore: skipping {len(skipped)} language(s) with no morphscore "
                f"dataset: {skipped}"
            )
        logger.info(
            f"MorphScore metrics initialized for {len(self.target_languages)} "
            f"language(s): {self.target_languages}"
        )
    
    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        """
        Compute MorphScore metrics.
        
        Args:
            tokenized_data: Optional tokenized data (not used for MorphScore)
            
        Returns:
            Dictionary with MorphScore results
        """
        if not MORPHSCORE_AVAILABLE:
            return {
                'morphscore': {
                    'error': 'MorphScore library not available'
                }
            }
        
        logger.info("Computing MorphScore metrics...")
        
        # Initialize MorphScore evaluator
        morph_score = MorphScore(
            data_dir=self.data_dir,
            language_subset=self.target_languages,
            by_split=self.by_split,
            freq_scale=self.freq_scale,
            exclude_single_tok=self.exclude_single_tok
        )
        
        results = {
            'per_tokenizer': {},
            'metadata': {
                'by_split': self.by_split,
                'freq_scale': self.freq_scale,
                'exclude_single_tok': self.exclude_single_tok,
                'target_languages': self.target_languages
            }
        }
        
        # Evaluate each tokenizer
        for tok_name in self.tokenizer_names:
            logger.info(f"Evaluating MorphScore for tokenizer: {tok_name}")
            try:
                # Get tokenizer wrapper
                tokenizer_wrapper = self.input_provider.get_tokenizer(tok_name)
                
                # Get underlying tokenizer for MorphScore (requires raw tokenizer)
                underlying_tokenizer = tokenizer_wrapper.get_underlying_tokenizer()
                if underlying_tokenizer is None:
                    logger.warning(f"MorphScore not available for {tok_name}: no underlying tokenizer available")
                    tokenizer_results = {
                        'error': 'No underlying tokenizer available for MorphScore evaluation'
                    }
                    results['per_tokenizer'][tok_name] = tokenizer_results
                    continue
                
                # Evaluate with MorphScore using the raw tokenizer
                morph_results = morph_score.eval(underlying_tokenizer)
                
                # Process results
                tokenizer_results = {
                    'per_language': {},
                    'summary': {
                        'languages_evaluated': 0,
                        'total_samples': 0,
                        'avg_morphscore_recall': 0.0,
                        'avg_morphscore_precision': 0.0,
                        'avg_micro_f1': 0.0,
                        'avg_macro_f1': 0.0
                    }
                }
                
                # Collect per-language results
                recall_values = []
                precision_values = []
                micro_f1_values = []
                macro_f1_values = []
                total_samples = 0
                
                for lang_code, lang_results in morph_results.items():
                    if lang_results and lang_code in self.target_languages:
                        if 'error' in lang_results:
                            logger.warning(f"Error evaluating MorphScore for {tok_name} on {lang_code}: {lang_results['error']}")
                            continue
                            
                        tokenizer_results['per_language'][lang_code] = lang_results

                        # Collect values for averaging
                        recall_values.append(lang_results['morphscore_recall'])
                        precision_values.append(lang_results['morphscore_precision'])
                        micro_f1_values.append(lang_results['micro_f1'])
                        macro_f1_values.append(lang_results['macro_f1'])
                        total_samples += lang_results['num_samples']
                
                # Compute summary statistics
                if recall_values:
                    n_languages = len(recall_values)
                    tokenizer_results['summary'] = {
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
                
                results['per_tokenizer'][tok_name] = tokenizer_results
                
                logger.info(f"MorphScore evaluation completed for {tok_name}: "
                           f"{len(recall_values)} languages evaluated, "
                           f"{total_samples} total samples")
                
            except Exception as e:
                logger.error(f"Error evaluating MorphScore for {tok_name}: {e}")
                results['per_tokenizer'][tok_name] = {
                    'error': str(e),
                    'per_language': {},
                    'summary': {
                        'languages_evaluated': 0,
                        'total_samples': 0,
                        'avg_morphscore_recall': 0.0,
                        'avg_morphscore_precision': 0.0,
                        'avg_micro_f1': 0.0,
                        'avg_macro_f1': 0.0
                    }
                }
        
        return {'morphscore': results}
    
    def print_results(self, results: Dict[str, Any], per_lang: bool = False):
        """Print MorphScore metrics results."""
        if 'morphscore' not in results:
            return
        
        morphscore_data = results['morphscore']
        
        # Handle error case
        if 'error' in morphscore_data:
            print(f"\nMORPHSCORE ANALYSIS")
            print("-" * 40)
            print(f"Error: {morphscore_data['error']}")
            return
        
        print("\n" + "="*60)
        print("MORPHSCORE RESULTS")
        print("="*60)
        
        # Print configuration info
        metadata = morphscore_data.get('metadata', {})
        print(f"\nCONFIGURATION")
        print("-" * 40)
        print(f"By split: {metadata.get('by_split', 'N/A')}")
        print(f"Frequency scale: {metadata.get('freq_scale', 'N/A')}")
        print(f"Exclude single token: {metadata.get('exclude_single_tok', 'N/A')}")
        print(f"Target languages: {len(metadata.get('target_languages', []))}")
        
        # Print summary statistics
        print(f"\nSUMMARY STATISTICS")
        print("-" * 40)
        
        for tok_name in self.tokenizer_names:
            if tok_name in morphscore_data['per_tokenizer']:
                tok_data = morphscore_data['per_tokenizer'][tok_name]
                
                if 'error' in tok_data:
                    print(f"{tok_name:20}: Error - {tok_data['error']}")
                    continue
                
                summary = tok_data.get('summary', {})
                recall = summary.get('avg_morphscore_recall', 0.0)
                precision = summary.get('avg_morphscore_precision', 0.0)
                micro_f1 = summary.get('avg_micro_f1', 0.0)
                macro_f1 = summary.get('avg_macro_f1', 0.0)
                langs = summary.get('languages_evaluated', 0)
                samples = summary.get('total_samples', 0)
                
                print(f"{tok_name:20}:")
                print(f"  {'Recall':15}: {recall:.3f}")
                print(f"  {'Precision':15}: {precision:.3f}")
                print(f"  {'Micro F1':15}: {micro_f1:.3f}")
                print(f"  {'Macro F1':15}: {macro_f1:.3f}")
                print(f"  {'Languages':15}: {langs}")
                print(f"  {'Samples':15}: {samples:,}")
        
        # Print detailed per-language results
        if per_lang:
            print(f"\nPER-LANGUAGE RESULTS")
            print("-" * 60)
            
            for tok_name in self.tokenizer_names:
                if tok_name not in morphscore_data['per_tokenizer']:
                    continue
                    
                tok_data = morphscore_data['per_tokenizer'][tok_name]
                
                if 'error' in tok_data or not tok_data.get('per_language'):
                    continue
                
                print(f"\n{tok_name}:")
                print("-" * 30)
                
                for lang_code, lang_results in tok_data['per_language'].items():
                    recall = lang_results.get('morphscore_recall', 0.0)
                    precision = lang_results.get('morphscore_precision', 0.0)
                    micro_f1 = lang_results.get('micro_f1', 0.0)
                    macro_f1 = lang_results.get('macro_f1', 0.0)
                    samples = lang_results.get('num_samples', 0)
                    
                    print(f"  {lang_code}:")
                    print(f"    Recall: {recall:.3f}, Precision: {precision:.3f}")
                    print(f"    Micro F1: {micro_f1:.3f}, Macro F1: {macro_f1:.3f}")
                    print(f"    Samples: {samples:,}")
            
        print("\n" + "="*60)