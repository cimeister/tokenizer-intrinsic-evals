"""
Basic tokenization metrics using unified TokenizedData interface.
"""

from bisect import bisect_left
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging
import time

from .base import BaseMetrics, TokenizedDataProcessor
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider
from ..config import TextMeasurementConfig, TextMeasurer, DEFAULT_TEXT_MEASUREMENT_CONFIG, DEFAULT_WORD_MEASUREMENT_CONFIG
from ..config.language_metadata import LanguageMetadata
from ..utils.text_utils import load_math_data, BUILTIN_MATH_SAMPLES_PATH

logger = logging.getLogger(__name__)

_WS_CHARS = frozenset(' \t\n\r')
_CER_WARMUP = 50


class BasicTokenizationMetrics(BaseMetrics):
    """Basic tokenization metrics: fertility, token_length, type_token_ratio, vocabulary_utilization, avg_tokens_per_line, reconstruction_fidelity."""

    def __init__(self,
                 input_provider: InputProvider,
                 measurement_config: Optional[TextMeasurementConfig] = None,
                 language_metadata: Optional[LanguageMetadata] = None,
                 code_texts: Optional[Dict[str, List[str]]] = None,
                 math_data_path: Optional[str] = None,
                 use_builtin_math_data: bool = False,
                 fertility_use_global_config: bool = False):
        """
        Initialize basic metrics.

        Args:
            input_provider: InputProvider instance
            measurement_config: Configuration for text measurement method
            language_metadata: Optional language metadata for grouping
            code_texts: Optional pre-loaded code texts mapping languages to snippets
            math_data_path: Optional path to math-rich text file
            use_builtin_math_data: Whether to use built-in math samples
            fertility_use_global_config: If True, fertility uses *measurement_config*
                instead of the default words-based normalization.
        """
        super().__init__(input_provider)
        self.measurement_config = measurement_config or DEFAULT_TEXT_MEASUREMENT_CONFIG
        self.language_metadata = language_metadata
        if fertility_use_global_config:
            self.fertility_measurement_config = self.measurement_config
        else:
            self.fertility_measurement_config = DEFAULT_WORD_MEASUREMENT_CONFIG
        self.fertility_text_measurer = TextMeasurer(self.fertility_measurement_config)

        # Code data for reconstruction fidelity (pre-loaded)
        self._code_texts: Dict[str, List[str]] = code_texts or {}

        # Load math data for reconstruction fidelity
        self._math_texts: List[str] = []
        if math_data_path:
            self._math_texts = load_math_data(math_data_path)
        elif use_builtin_math_data:
            self._math_texts = load_math_data(BUILTIN_MATH_SAMPLES_PATH)

    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None,
                include_reconstruction: bool = True,
                cer_time_budget_s: float = 30.0) -> Dict[str, Any]:
        """
        Compute basic tokenization metrics.

        Args:
            tokenized_data: Optional tokenized data dict. If None, uses input_provider data.
            include_reconstruction: Whether to include reconstruction fidelity analysis.
            cer_time_budget_s: Max seconds to spend on CER per tokenizer before
                skipping.  0 disables the budget (always compute).

        Returns:
            Dictionary with basic metrics results
        """
        if tokenized_data is None:
            tokenized_data = self.get_tokenized_data()

        results = {}

        # Compute fertility analysis
        results.update(self.compute_fertility_analysis(tokenized_data))

        # Compute token length analysis
        results.update(self.compute_token_length_analysis(tokenized_data))

        # Compute vocabulary utilization
        results.update(self.compute_vocabulary_utilization_analysis(tokenized_data))

        # Compute type-token ratio
        results.update(self.compute_type_token_ratio_analysis(tokenized_data))

        # Compute average tokens per line
        results.update(self.compute_avg_tokens_per_line_analysis(tokenized_data))

        # Compute reconstruction fidelity
        if include_reconstruction:
            results.update(self.compute_reconstruction_fidelity_analysis(
                tokenized_data, cer_time_budget_s=cer_time_budget_s))

        return results
    
    def compute_fertility_analysis(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute fertility analysis using configured normalization method.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict with fertility results
        """
        normalization_unit = self.fertility_measurement_config.method.value.lower()
        
        results = {
            'fertility': {
                'per_tokenizer': {},
                'per_language': {},
                'pairwise_comparisons': {},
                'metadata': {
                    'normalization_method': normalization_unit,
                    'description': f'Average number of tokens per {normalization_unit[:-1]}',
                    'short_description': f'tokens/{normalization_unit[:-1]}'
                }
            }
        }
        
        global_values = {}
        
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
            
            tok_data = tokenized_data[tok_name]
            
            # Compute global fertility
            global_fertility = self._compute_fertility_stats(tok_data, normalization_unit)
            
            # Compute per-language fertility
            per_lang_fertility = {}
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
            
            for language, lang_data in lang_groups.items():
                lang_fertility = self._compute_fertility_stats(lang_data, normalization_unit)
                if lang_fertility['count'] == 0 and lang_data:
                    logger.warning(
                        "Fertility: language '%s' produced 0 valid samples. "
                        "This typically occurs for languages without whitespace "
                        "word boundaries (e.g., Chinese, Japanese, Thai) when "
                        "using whitespace-based word counting.",
                        language,
                    )
                per_lang_fertility[language] = lang_fertility
            
            results['fertility']['per_tokenizer'][tok_name] = {
                'global': global_fertility,
                'per_language': per_lang_fertility
            }
            
            global_values[tok_name] = global_fertility.get('mean', 0.0)
        
        # Compute pairwise comparisons
        if len(global_values) >= 2:
            results['fertility']['pairwise_comparisons'] = self.compute_pairwise_comparisons(
                global_values, 'fertility'
            )
        
        return results
    
    def _compute_fertility_stats(self, tokenized_data: List[TokenizedData], 
                                normalization_unit: str) -> Dict[str, float]:
        """Compute fertility statistics for a list of TokenizedData."""
        if not tokenized_data:
            return self.empty_stats()
        
        fertilities = []
        
        for data in tokenized_data:
            if not data.text or not data.text.strip():
                continue  # Skip if no text available

            num_tokens = len(data.tokens)
            num_units = self.fertility_text_measurer.get_unit_count(data.text)
            
            if num_units > 0:
                fertility = num_tokens / num_units
                fertilities.append(fertility)
        
        if not fertilities:
            return self.empty_stats()
        
        return self.compute_basic_stats(fertilities)
    
    def compute_token_length_analysis(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute token length analysis.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict with token length results
        """
        results = {
            'token_length': {
                'per_tokenizer': {},
                'metadata': {
                    'units': ['characters', 'bytes'],
                    'description': 'Average character and byte length per token'
                }
            }
        }

        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue

            tok_data = tokenized_data[tok_name]

            char_lengths = []
            byte_lengths = []
            for data in tok_data:
                if data.text and data.text.strip() and data.tokens:
                    n_tokens = len(data.tokens)
                    char_lengths.append(len(data.text) / n_tokens)
                    byte_lengths.append(len(data.text.encode('utf-8')) / n_tokens)

            if char_lengths:
                char_stats = self.compute_basic_stats(char_lengths)
                byte_stats = self.compute_basic_stats(byte_lengths)
                results['token_length']['per_tokenizer'][tok_name] = {
                    'character_length': char_stats,
                    'byte_length': byte_stats,
                    'primary_length': char_stats
                }
            else:
                empty_stats = self.empty_stats()
                results['token_length']['per_tokenizer'][tok_name] = {
                    'character_length': empty_stats,
                    'byte_length': empty_stats,
                    'primary_length': empty_stats
                }

        return results
    
    def compute_vocabulary_utilization_analysis(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute vocabulary utilization analysis.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict with vocabulary utilization results
        """
        results = {
            'vocabulary_utilization': {
                'per_tokenizer': {},
                'metadata': {
                    'description': 'Proportion of vocabulary actually used',
                    'metric_range': '[0.0, 1.0]'
                }
            }
        }
        
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
            
            tok_data = tokenized_data[tok_name]
            vocab_size = self.get_vocab_size(tok_name)
            
            # Compute global utilization
            global_util = self._compute_vocabulary_utilization(tok_data, vocab_size)
            
            # Compute per-language utilization
            per_lang_util = {}
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
            
            for language, lang_data in lang_groups.items():
                lang_util = self._compute_vocabulary_utilization(lang_data, vocab_size)
                per_lang_util[language] = lang_util
            
            results['vocabulary_utilization']['per_tokenizer'][tok_name] = {
                'global_utilization': global_util['utilization'],
                'global_used_tokens': global_util['used_tokens'],
                'global_vocab_size': global_util['vocab_size'],
                'per_language': per_lang_util
            }
        
        return results
    
    def _compute_vocabulary_utilization(self, tokenized_data: List[TokenizedData], vocab_size: int) -> Dict[str, Any]:
        """Compute vocabulary utilization for a list of TokenizedData."""
        unique_tokens = TokenizedDataProcessor.get_unique_tokens(tokenized_data)
        used_tokens = len(unique_tokens)
        
        return {
            'utilization': self.safe_divide(used_tokens, vocab_size, 0.0),
            'used_tokens': used_tokens,
            'vocab_size': vocab_size,
            'unused_tokens': vocab_size - used_tokens
        }
    
    def compute_type_token_ratio_analysis(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute type-token ratio analysis.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict with type-token ratio results
        """
        results = {
            'type_token_ratio': {
                'per_tokenizer': {},
                'per_language': {},
                'metadata': {
                    'description': 'Ratio of unique tokens to total tokens (lexical diversity)',
                    'metric_range': '[0.0, 1.0]'
                }
            }
        }
        
        # Global per-language TTR (aggregated across tokenizers)
        all_languages = set()
        for tok_data in tokenized_data.values():
            for data in tok_data:
                all_languages.add(data.language)
        
        per_language_results = {}
        for language in all_languages:
            per_language_results[language] = {}
            
            for tok_name in self.tokenizer_names:
                if tok_name not in tokenized_data:
                    continue
                
                # Get data for this tokenizer and language
                lang_data = [data for data in tokenized_data[tok_name] 
                           if data.language == language]
                
                if lang_data:
                    ttr_stats = self._compute_type_token_ratio(lang_data)
                    per_language_results[language][tok_name] = ttr_stats['ttr']
        
        results['type_token_ratio']['per_language'] = per_language_results
        
        # Global per-tokenizer TTR
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
            
            tok_data = tokenized_data[tok_name]
            global_ttr = self._compute_type_token_ratio(tok_data)
            
            results['type_token_ratio']['per_tokenizer'][tok_name] = {
                'global_ttr': global_ttr['ttr'],
                'global_types': global_ttr['types'],
                'global_tokens': global_ttr['tokens']
            }
        
        return results
    
    def _compute_type_token_ratio(self, tokenized_data: List[TokenizedData]) -> Dict[str, Any]:
        """Compute type-token ratio for a list of TokenizedData."""
        all_tokens = TokenizedDataProcessor.flatten_all_tokens(tokenized_data)
        unique_tokens = set(all_tokens)
        
        total_tokens = len(all_tokens)
        unique_count = len(unique_tokens)
        
        return {
            'ttr': self.safe_divide(unique_count, total_tokens, 0.0),
            'types': unique_count,
            'tokens': total_tokens
        }
    
    def compute_avg_tokens_per_line_analysis(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute average tokens per line analysis.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict with average tokens per line results
        """
        results = {
            'avg_tokens_per_line': {
                'per_tokenizer': {},
                'metadata': {
                    'description': 'Average number of tokens per line of text',
                    'unit': 'tokens/line'
                }
            }
        }
        
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
            
            tok_data = tokenized_data[tok_name]
            
            # Calculate tokens per line where text is available
            tokens_per_line = []
            total_lines = 0
            
            for data in tok_data:
                if data.text and data.text.strip() and data.tokens:
                    # Exclude blank lines from the count
                    lines = [l for l in data.text.split('\n') if l.strip()]
                    num_lines = len(lines)
                    total_lines += num_lines

                    if num_lines > 0:
                        tpl = len(data.tokens) / num_lines
                        tokens_per_line.append(tpl)
            
            if tokens_per_line:
                tpl_stats = self.compute_basic_stats(tokens_per_line)
                results['avg_tokens_per_line']['per_tokenizer'][tok_name] = {
                    'global_avg': tpl_stats['mean'],
                    'global_std': tpl_stats['std'],
                    'global_std_err': tpl_stats['std_err'],
                    'total_lines': total_lines,
                    'stats': tpl_stats
                }
            else:
                results['avg_tokens_per_line']['per_tokenizer'][tok_name] = {
                    'global_avg': 0.0,
                    'global_std': 0.0,
                    'global_std_err': 0.0,
                    'total_lines': 0,
                    'stats': self.empty_stats()
                }

        return results

    # ------------------------------------------------------------------
    # Reconstruction Fidelity
    # ------------------------------------------------------------------

    def compute_reconstruction_fidelity_analysis(
        self, tokenized_data: Dict[str, List[TokenizedData]],
        cer_time_budget_s: float = 30.0,
    ) -> Dict[str, Any]:
        """Compute encode-decode round-trip fidelity metrics.

        For each tokenizer that supports decoding, collects text from
        language data, code data, and math data, then measures:
        - Exact match rate
        - Character error rate (CER)
        - UNK token rate
        - Whitespace fidelity

        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            cer_time_budget_s: Max seconds to spend on CER per tokenizer.
                After ``_CER_WARMUP`` non-exact texts, the total CER time is
                extrapolated.  If the projection exceeds this budget the CER
                and whitespace-fidelity computations are skipped for the rest
                of the tokenizer and reported as ``None``.
                Set to ``0`` to disable the budget (always compute).
        """
        results: Dict[str, Any] = {
            'reconstruction_fidelity': {
                'per_tokenizer': {},
                'summary': {},
            }
        }

        for tok_name in self.tokenizer_names:
            try:
                tokenizer = self.input_provider.get_tokenizer(tok_name)
            except Exception:
                continue

            if not tokenizer.can_decode():
                logger.info("Reconstruction fidelity: skipping %s (no decode support)", tok_name)
                continue

            unk_id = tokenizer.get_unk_token_id()

            # Per-domain accumulators
            domain_stats: Dict[str, Dict[str, Any]] = defaultdict(
                lambda: {
                    'exact_matches': 0, 'total': 0,
                    'cer_sum': 0.0,
                    'unk_tokens': 0, 'total_tokens': 0,
                    'ws_preserved': 0, 'ws_total': 0,
                }
            )

            has_data = False

            # CER time-budget state
            cer_elapsed = 0.0
            cer_skipped = False
            n_cer_calls = 0
            # Count total texts for extrapolation
            total_lang_texts = 0
            if tok_name in tokenized_data:
                total_lang_texts = sum(
                    1 for td in tokenized_data[tok_name]
                    if td.text and td.text.strip()
                )
            total_code_math_texts = (
                sum(len(snippets) for snippets in self._code_texts.values())
                + len(self._math_texts)
            )
            total_all_texts = total_lang_texts + total_code_math_texts
            texts_processed = 0

            # Language data — reuse tokens/offsets already stored in TokenizedData
            if tok_name in tokenized_data:
                for td in tokenized_data[tok_name]:
                    if not td.text or not td.text.strip():
                        continue
                    has_data = True
                    text = td.text
                    texts_processed += 1

                    token_ids = td.tokens
                    stats = domain_stats[td.language]
                    stats['total_tokens'] += len(token_ids)

                    if unk_id is not None:
                        stats['unk_tokens'] += token_ids.count(unk_id)

                    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                    if decoded is None:
                        continue

                    stats['total'] += 1

                    if decoded == text:
                        stats['exact_matches'] += 1
                        if not cer_skipped:
                            ws_total = sum(1 for c in text if c in _WS_CHARS)
                            stats['ws_preserved'] += ws_total
                            stats['ws_total'] += ws_total
                    else:
                        if not cer_skipped:
                            t0 = time.monotonic()
                            stats['cer_sum'] += self._character_error_rate(text, decoded)
                            cer_elapsed += time.monotonic() - t0
                            n_cer_calls += 1

                            ws_preserved, ws_total = self._whitespace_fidelity(text, decoded)
                            stats['ws_preserved'] += ws_preserved
                            stats['ws_total'] += ws_total

                            # Check budget after warmup
                            if (cer_time_budget_s > 0
                                    and n_cer_calls == _CER_WARMUP):
                                # Estimate remaining non-exact texts
                                exact_so_far = sum(
                                    ds['exact_matches']
                                    for ds in domain_stats.values()
                                )
                                total_so_far = sum(
                                    ds['total']
                                    for ds in domain_stats.values()
                                )
                                exact_rate = (exact_so_far / total_so_far
                                              if total_so_far else 0.0)
                                remaining_texts = total_all_texts - texts_processed
                                est_remaining_nonexact = (
                                    (1 - exact_rate) * remaining_texts
                                )
                                time_per_cer = cer_elapsed / n_cer_calls
                                projected = (
                                    cer_elapsed
                                    + time_per_cer * est_remaining_nonexact
                                )
                                if projected > cer_time_budget_s:
                                    cer_skipped = True
                                    logger.warning(
                                        "CER time budget exceeded for %s: "
                                        "%.1fs projected (budget %.1fs). "
                                        "Skipping CER and whitespace fidelity "
                                        "for remaining texts.",
                                        tok_name, projected,
                                        cer_time_budget_s,
                                    )

            # Code/math data — encode on the fly (not in TokenizedData)
            code_math_pairs: List[Tuple[str, str]] = []
            for lang, snippets in self._code_texts.items():
                domain = f"code_{lang}"
                for snippet in snippets:
                    if snippet and snippet.strip():
                        code_math_pairs.append((snippet, domain))
            for text in self._math_texts:
                if text and text.strip():
                    code_math_pairs.append((text, "math"))

            for text, domain in code_math_pairs:
                has_data = True
                texts_processed += 1
                stats = domain_stats[domain]

                token_ids, _ = tokenizer.encode_with_offsets(text)
                stats['total_tokens'] += len(token_ids)

                if unk_id is not None:
                    stats['unk_tokens'] += token_ids.count(unk_id)

                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                if decoded is None:
                    continue

                stats['total'] += 1

                if decoded == text:
                    stats['exact_matches'] += 1
                    if not cer_skipped:
                        ws_total = sum(1 for c in text if c in _WS_CHARS)
                        stats['ws_preserved'] += ws_total
                        stats['ws_total'] += ws_total
                else:
                    if not cer_skipped:
                        t0 = time.monotonic()
                        stats['cer_sum'] += self._character_error_rate(text, decoded)
                        cer_elapsed += time.monotonic() - t0
                        n_cer_calls += 1

                        ws_preserved, ws_total = self._whitespace_fidelity(text, decoded)
                        stats['ws_preserved'] += ws_preserved
                        stats['ws_total'] += ws_total

                        # Check budget after warmup (if not already checked)
                        if (cer_time_budget_s > 0
                                and not cer_skipped
                                and n_cer_calls == _CER_WARMUP):
                            exact_so_far = sum(
                                ds['exact_matches']
                                for ds in domain_stats.values()
                            )
                            total_so_far = sum(
                                ds['total']
                                for ds in domain_stats.values()
                            )
                            exact_rate = (exact_so_far / total_so_far
                                          if total_so_far else 0.0)
                            remaining_texts = total_all_texts - texts_processed
                            est_remaining_nonexact = (
                                (1 - exact_rate) * remaining_texts
                            )
                            time_per_cer = cer_elapsed / n_cer_calls
                            projected = (
                                cer_elapsed
                                + time_per_cer * est_remaining_nonexact
                            )
                            if projected > cer_time_budget_s:
                                cer_skipped = True
                                logger.warning(
                                    "CER time budget exceeded for %s: "
                                    "%.1fs projected (budget %.1fs). "
                                    "Skipping CER and whitespace fidelity "
                                    "for remaining texts.",
                                    tok_name, projected,
                                    cer_time_budget_s,
                                )

            if not has_data:
                continue

            # Log after main loop using accumulated counts
            n_lang = sum(
                ds['total'] for dom, ds in domain_stats.items()
                if not dom.startswith('code_') and dom != 'math'
            )
            n_code_math = sum(
                ds['total'] for dom, ds in domain_stats.items()
                if dom.startswith('code_') or dom == 'math'
            )
            logger.info(
                "Reconstruction fidelity: decoding %d texts for %s (%d language, %d code/math)",
                n_lang + n_code_math, tok_name, n_lang, n_code_math,
            )

            # Aggregate per-domain and overall
            per_domain: Dict[str, Dict[str, Any]] = {}
            overall = {
                'exact_matches': 0, 'total': 0,
                'cer_sum': 0.0,
                'unk_tokens': 0, 'total_tokens': 0,
                'ws_preserved': 0, 'ws_total': 0,
            }

            for domain, ds in domain_stats.items():
                count = ds['total']
                per_domain[domain] = {
                    'exact_match_rate': self.safe_divide(ds['exact_matches'], count, 0.0),
                    'mean_cer': None if cer_skipped else self.safe_divide(ds['cer_sum'], count, 0.0),
                    'unk_token_rate': self.safe_divide(ds['unk_tokens'], ds['total_tokens'], 0.0),
                    'whitespace_fidelity': None if cer_skipped else self.safe_divide(ds['ws_preserved'], ds['ws_total'], 1.0),
                    'count': count,
                    'total_tokens': ds['total_tokens'],
                }
                for k in overall:
                    overall[k] += ds[k]

            total = overall['total']
            tok_result = {
                'by_domain': per_domain,
                'overall': {
                    'exact_match_rate': self.safe_divide(overall['exact_matches'], total, 0.0),
                    'mean_cer': None if cer_skipped else self.safe_divide(overall['cer_sum'], total, 0.0),
                    'unk_token_rate': self.safe_divide(overall['unk_tokens'], overall['total_tokens'], 0.0),
                    'whitespace_fidelity': None if cer_skipped else self.safe_divide(overall['ws_preserved'], overall['ws_total'], 1.0),
                    'count': total,
                    'total_tokens': overall['total_tokens'],
                },
            }
            if cer_skipped:
                tok_result['cer_skipped'] = True
            results['reconstruction_fidelity']['per_tokenizer'][tok_name] = tok_result
            results['reconstruction_fidelity']['summary'][tok_name] = {
                'exact_match_rate': tok_result['overall']['exact_match_rate'],
                'mean_cer': tok_result['overall']['mean_cer'],
                'unk_token_rate': tok_result['overall']['unk_token_rate'],
                'whitespace_fidelity': tok_result['overall']['whitespace_fidelity'],
                'texts_analyzed': total,
                'total_tokens_analyzed': overall['total_tokens'],
            }
            if cer_skipped:
                results['reconstruction_fidelity']['summary'][tok_name]['cer_skipped'] = True
            cer_msg = "SKIPPED" if cer_skipped else f"{tok_result['overall']['mean_cer']:.4f}"
            logger.info(
                "Reconstruction fidelity: %s done — %d texts decoded, "
                "exact_match=%.3f, mean_cer=%s",
                tok_name, total,
                tok_result['overall']['exact_match_rate'],
                cer_msg,
            )

        return results

    @staticmethod
    def _character_error_rate(reference: str, hypothesis: str) -> float:
        """Levenshtein edit distance between *reference* and *hypothesis*,
        normalized by the length of *reference*.

        CER = levenshtein(reference, hypothesis) / len(reference)

        Returns 0.0 when the strings are identical or the reference is empty.
        CER can exceed 1.0 when the hypothesis is longer than the reference.
        Uses a two-row DP approach: O(n*m) time, O(min(n,m)) space.

        Optimized with common prefix/suffix stripping to reduce the DP
        matrix to the differing region only.
        """
        if reference == hypothesis:
            return 0.0
        ref_len = len(reference)
        if ref_len == 0:
            return 0.0
        if not hypothesis:
            return 1.0

        # Strip common prefix
        prefix = 0
        min_len = min(ref_len, len(hypothesis))
        while prefix < min_len and reference[prefix] == hypothesis[prefix]:
            prefix += 1

        # Strip common suffix (don't overlap with prefix)
        r_end = ref_len
        h_end = len(hypothesis)
        while r_end > prefix and h_end > prefix and reference[r_end - 1] == hypothesis[h_end - 1]:
            r_end -= 1
            h_end -= 1

        # Work on the differing region only
        a = reference[prefix:r_end]
        b = hypothesis[prefix:h_end]
        n = len(a)
        m = len(b)

        # If one side is empty after stripping, distance = length of the other
        if n == 0:
            return m / ref_len
        if m == 0:
            return n / ref_len

        # Swap so the DP row is the shorter dimension (space optimization).
        # Levenshtein distance is symmetric so the result is unchanged.
        rows, cols = n, m
        if cols > rows:
            a, b = b, a
            rows, cols = cols, rows

        prev = list(range(cols + 1))
        curr = [0] * (cols + 1)

        for i in range(1, rows + 1):
            curr[0] = i
            ai = a[i - 1]
            prev_im1 = prev[0]
            for j in range(1, cols + 1):
                pj = prev[j]
                sub = prev_im1 if ai == b[j - 1] else prev_im1 + 1
                ins = curr[j - 1] + 1
                dele = pj + 1
                curr[j] = sub if sub <= ins and sub <= dele else (ins if ins <= dele else dele)
                prev_im1 = pj
            prev, curr = curr, prev

        return prev[cols] / ref_len

    @staticmethod
    def _whitespace_fidelity(
        original: str,
        decoded: str,
    ) -> Tuple[int, int]:
        """Count whitespace chars preserved through encode-decode round-trip.

        Uses a greedy forward scan with indexed lookup to align original
        characters to decoded characters, then checks whether each
        whitespace position in the original is preserved.

        Returns ``(num_preserved, num_total_ws)`` in the original.
        """
        total_ws = sum(1 for c in original if c in _WS_CHARS)
        if total_ws == 0:
            return (0, 0)

        if original == decoded:
            return (total_ws, total_ws)

        # Greedy forward scan with indexed lookup
        char_positions = defaultdict(list)
        for i, c in enumerate(decoded):
            char_positions[c].append(i)

        preserved = 0
        j = 0
        for c in original:
            if c not in _WS_CHARS:
                positions = char_positions.get(c)
                if positions is not None:
                    idx = bisect_left(positions, j)
                    if idx < len(positions):
                        j = positions[idx] + 1
            else:
                if j < len(decoded) and decoded[j] == c:
                    preserved += 1
                    j += 1

        return (preserved, total_ws)