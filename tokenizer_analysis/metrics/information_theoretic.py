"""
Information-theoretic metrics including entropy, compression, and vocabulary utilization.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from collections import Counter, defaultdict
import logging

from .base import BaseMetrics, TokenizedDataProcessor
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider
from ..config import TextMeasurementConfig, TextMeasurer, DEFAULT_LINE_MEASUREMENT_CONFIG, DEFAULT_TEXT_MEASUREMENT_CONFIG
from ..config.language_metadata import LanguageMetadata
from ..constants import DEFAULT_RENYI_ALPHAS, SHANNON_ENTROPY_ALPHA

logger = logging.getLogger(__name__)

MIN_BIGRAM_OCCURRENCES = 3
MIN_TRIGRAM_OCCURRENCES = 3


class InformationTheoreticMetrics(BaseMetrics):
    """Information-theoretic analysis metrics."""
    
    def __init__(self, input_provider: InputProvider,
                 renyi_alphas: Optional[List[float]] = None,
                 measurement_config: Optional[TextMeasurementConfig] = None,
                 language_metadata: Optional[LanguageMetadata] = None,
                 min_bigram_occurrences: int = MIN_BIGRAM_OCCURRENCES,
                 min_trigram_occurrences: int = MIN_TRIGRAM_OCCURRENCES):
        """
        Initialize information-theoretic metrics.

        Args:
            input_provider: InputProvider instance
            renyi_alphas: List of alpha values for Rényi entropy (default: [1.0, 2.0, 3.0])
            measurement_config: Configuration for text measurement method
            language_metadata: Optional language metadata for grouping
            min_bigram_occurrences: Minimum number of bigram occurrences for a
                token type to be included in bigram entropy (default: 3)
            min_trigram_occurrences: Minimum number of trigram occurrences for a
                context pair to be included in trigram entropy (default: 3)
        """
        super().__init__(input_provider)
        self.renyi_alphas = renyi_alphas or DEFAULT_RENYI_ALPHAS
        # Default to lines for information-theoretic analysis (as was hardcoded before)
        self.measurement_config = measurement_config or DEFAULT_TEXT_MEASUREMENT_CONFIG
        self.text_measurer = TextMeasurer(self.measurement_config)
        self.language_metadata = language_metadata
        self.min_bigram_occurrences = min_bigram_occurrences
        self.min_trigram_occurrences = min_trigram_occurrences
    
    def compute_renyi_entropy(self, token_counts: Counter, alpha: float) -> float:
        """
        Compute Rényi entropy of order alpha for token distribution.
        
        Args:
            token_counts: Counter of token frequencies
            alpha: Order of Rényi entropy
            
        Returns:
            Rényi entropy value
        """
        if not token_counts:
            return 0.0
        
        total_count = sum(token_counts.values())
        probabilities = [count / total_count for count in token_counts.values()]
        
        if alpha == SHANNON_ENTROPY_ALPHA:
            # Shannon entropy (limit case)
            return -sum(p * np.log2(p) for p in probabilities if p > 0)
        else:
            # General Rényi entropy
            sum_p_alpha = sum(p ** alpha for p in probabilities if p > 0)
            if sum_p_alpha <= 0:
                return 0.0
            return (1 / (1 - alpha)) * np.log2(sum_p_alpha)
    
    def compute_renyi_efficiency_analysis(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute Rényi efficiency metrics for all tokenizers.
        
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
            
        Returns:
            Dict with Rényi efficiency results
        """
        
        results = {
            'per_tokenizer': {},
            'per_language': {},
            'pairwise_comparisons': {}
        }
        
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
                
            tok_results = {}
            tok_data = tokenized_data[tok_name]
            
            # Collect all tokens for global entropy
            global_token_counts = Counter()
            per_lang_token_counts = {}
            
            # Group data by language
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
            
            for lang, lang_data in lang_groups.items():
                lang_token_counts = Counter()
                
                for data in lang_data:
                    for token in data.tokens:
                        global_token_counts[token] += 1
                        lang_token_counts[token] += 1
                
                per_lang_token_counts[lang] = lang_token_counts
            
            # Compute Rényi entropy for each alpha
            for alpha in self.renyi_alphas:
                alpha_key = f'renyi_{alpha}'
                tok_results[alpha_key] = {}
                
                # Global entropy
                global_entropy = self.compute_renyi_entropy(global_token_counts, alpha)
                global_vocab = len(global_token_counts)
                tok_results[alpha_key]['overall'] = global_entropy / np.log2(global_vocab) if global_vocab > 1 else 0.0
                
                # Per-language entropy
                for lang, lang_counts in per_lang_token_counts.items():
                    lang_entropy = self.compute_renyi_entropy(lang_counts, alpha)
                    lang_vocab = len(lang_counts)
                    tok_results[alpha_key][lang] = lang_entropy / np.log2(lang_vocab) if lang_vocab > 1 else 0.0
            
            results['per_tokenizer'][tok_name] = tok_results
        
        # Aggregate per-language results
        all_languages = set()
        for tok_results in results['per_tokenizer'].values():
            for alpha in self.renyi_alphas:
                alpha_key = f'renyi_{alpha}'
                if alpha_key in tok_results:
                    all_languages.update(k for k in tok_results[alpha_key].keys() if k != 'overall')
        
        for alpha in self.renyi_alphas:
            alpha_key = f'renyi_{alpha}'
            results['per_language'][alpha_key] = {}
            for lang in all_languages:
                results['per_language'][alpha_key][lang] = {}
                for tok_name in self.tokenizer_names:
                    if (alpha_key in results['per_tokenizer'][tok_name] and 
                        lang in results['per_tokenizer'][tok_name][alpha_key]):
                        results['per_language'][alpha_key][lang][tok_name] = results['per_tokenizer'][tok_name][alpha_key][lang]
        
        # Compute pairwise comparisons for Shannon entropy (alpha=1.0)
        if 1.0 in self.renyi_alphas:
            shannon_entropies = {name: results['per_tokenizer'][name]['renyi_1.0']['overall'] 
                               for name in self.tokenizer_names}
            results['pairwise_comparisons']['shannon'] = self.compute_pairwise_comparisons(
                shannon_entropies, 'shannon_entropy'
            )
        
        return results
    
    def compute_compression_rate(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Compute compression rates using ratio-of-means: total_units / total_tokens.

        This produces a single global statistic rather than averaging per-sample
        ratios, which avoids bias from short texts.

        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists

        Returns:
            Dict with compression rate results
        """

        results = {
            'per_tokenizer': {},
            'per_language': {},
            'pairwise_comparisons': {}
        }

        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue

            tok_data = tokenized_data[tok_name]
            per_lang_ratios = {}
            total_units = 0
            total_tokens = 0
            num_texts = 0

            # Group data by language
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)

            for lang, lang_data in lang_groups.items():
                lang_units = 0
                lang_tokens = 0

                for data in lang_data:
                    if data.text and data.text.strip():
                        normalization_count = self.text_measurer.get_unit_count(data.text)
                        if normalization_count > 0 and data.tokens:
                            lang_units += normalization_count
                            lang_tokens += len(data.tokens)
                            num_texts += 1

                if lang_tokens > 0:
                    per_lang_ratios[lang] = lang_units / lang_tokens
                    total_units += lang_units
                    total_tokens += lang_tokens

            # Global compression: ratio of totals
            if total_tokens > 0:
                global_rate = total_units / total_tokens
            else:
                global_rate = 1.0  # Default compression rate

            results['per_tokenizer'][tok_name] = {
                'global': {
                    'compression_rate': global_rate,
                    'total_units': total_units,
                    'total_tokens': total_tokens,
                },
                'per_language': per_lang_ratios,
                'num_texts_analyzed': num_texts
            }

        # Add metadata
        results['metadata'] = {
            'normalization_method': self.measurement_config.method.value
        }

        # Compute pairwise comparisons
        global_ratios = {name: results['per_tokenizer'][name]['global']['compression_rate']
                        for name in self.tokenizer_names if name in results['per_tokenizer']}
        results['pairwise_comparisons'] = self.compute_pairwise_comparisons(
            global_ratios, 'compression_rate'
        )

        return results
        
    

    def compute_unigram_distribution_metrics(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """
        Computes metrics based on the unigram distribution of tokens for each language.
    
        This includes:
        1.  Unigram Distribution Entropy: The Shannon entropy of the token frequency
            distribution for each language.
        2.  Average Token Rank: The average rank of tokens (by frequency) observed
            in the corpus for each language.
    
        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists
    
        Returns:
            A dictionary containing the computed metrics, structured by tokenizer and language,
            including global metrics and pairwise comparisons.
        """
        
        results = {
            'per_tokenizer': {},
            'per_language': {
                'unigram_entropy': {},
                'avg_token_rank': {}
            },
            'pairwise_comparisons': {}
        }
    
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue
                
            tok_data = tokenized_data[tok_name]
            per_lang_metrics = {}
            global_token_counts = Counter()
            all_token_sequences = []
    
            # Group data by language
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)
    
            for lang, lang_data in lang_groups.items():
                # Flatten all tokens for the language
                lang_tokens = TokenizedDataProcessor.flatten_all_tokens(lang_data)
                if not lang_tokens:
                    continue
    
                # 1. Compute per-language unigram distribution and metrics
                lang_token_counts = Counter(lang_tokens)
                unigram_entropy = self.compute_renyi_entropy(lang_token_counts, alpha=1.0)
                
                ranked_tokens = [token for token, count in lang_token_counts.most_common()]
                token_to_rank = {token: rank + 1 for rank, token in enumerate(ranked_tokens)}
                
                lang_ranks = [token_to_rank[token] for token in lang_tokens]
                avg_token_rank = np.mean(lang_ranks) if lang_ranks else 0.0
                
                per_lang_metrics[lang] = {
                    'unigram_entropy': unigram_entropy,
                    'avg_token_rank': avg_token_rank,
                    'total_tokens': len(lang_tokens),
                    'unique_tokens': len(lang_token_counts)
                }
    
                # Aggregate for global metrics
                global_token_counts.update(lang_tokens)
                all_token_sequences.extend([data.tokens for data in lang_data])
    
            # 2. Compute global metrics for the tokenizer
            global_unigram_entropy = self.compute_renyi_entropy(global_token_counts, alpha=1.0)
            
            global_avg_token_rank = 0.0
            if global_token_counts:
                globally_ranked_tokens = [token for token, count in global_token_counts.most_common()]
                global_token_to_rank = {token: rank + 1 for rank, token in enumerate(globally_ranked_tokens)}
                
                all_global_ranks = [global_token_to_rank[token] for seq in all_token_sequences for token in seq]
                global_avg_token_rank = np.mean(all_global_ranks) if all_global_ranks else 0.0
    
            results['per_tokenizer'][tok_name] = {
                'global_unigram_entropy': global_unigram_entropy,
                'global_avg_token_rank': global_avg_token_rank,
                'per_language': per_lang_metrics
            }
    
        # 3. Aggregate per-language results for easier comparison across tokenizers
        all_languages = set()
        for tok_results in results['per_tokenizer'].values():
            all_languages.update(tok_results['per_language'].keys())
    
        for lang in all_languages:
            results['per_language']['unigram_entropy'][lang] = {}
            results['per_language']['avg_token_rank'][lang] = {}
            for tok_name in self.tokenizer_names:
                lang_stats = results['per_tokenizer'][tok_name]['per_language'].get(lang)
                if lang_stats:
                    results['per_language']['unigram_entropy'][lang][tok_name] = lang_stats['unigram_entropy']
                    results['per_language']['avg_token_rank'][lang][tok_name] = lang_stats['avg_token_rank']
    
        # 4. Compute pairwise comparisons for global metrics
        global_entropies = {name: res['global_unigram_entropy'] for name, res in results['per_tokenizer'].items()}
        global_ranks = {name: res['global_avg_token_rank'] for name, res in results['per_tokenizer'].items()}
        
        results['pairwise_comparisons']['global_unigram_entropy'] = self.compute_pairwise_comparisons(
            global_entropies, 'global_unigram_entropy'
        )
        results['pairwise_comparisons']['global_avg_token_rank'] = self.compute_pairwise_comparisons(
            global_ranks, 'global_avg_token_rank'
        )
    
        return results

    @staticmethod
    def _compute_weighted_entropy(right_accessors: Dict, min_occ: int) -> dict:
        """Compute frequency-weighted normalized Shannon entropy (η) over
        right-accessor distributions.

        Args:
            right_accessors: Mapping from context (any hashable key) to a
                Counter of successor token IDs.
            min_occ: Minimum total occurrences for a context to be included.

        Returns:
            Dict with keys 'entropy', 'total_ngrams', 'types_evaluated',
            'types_excluded'.
        """
        weighted_sum = 0.0
        weight_total = 0
        total_ngrams = 0
        types_evaluated = 0
        types_excluded = 0

        for t, successors in right_accessors.items():
            ta = sum(successors.values())
            total_ngrams += ta
            if ta < min_occ:
                types_excluded += 1
                continue
            n_unique = len(successors)
            if n_unique <= 1:
                # Only one successor type → entropy is 0, eta is 0
                types_evaluated += 1
                weight_total += ta
                continue
            # Shannon entropy
            h = -sum((c / ta) * np.log2(c / ta) for c in successors.values())
            # Normalize by max possible entropy
            max_h = np.log2(min(n_unique, ta))
            eta = h / max_h if max_h > 0 else 0.0
            weighted_sum += ta * eta
            weight_total += ta
            types_evaluated += 1

        entropy = weighted_sum / weight_total if weight_total else 0.0
        return {
            'entropy': entropy,
            'total_ngrams': total_ngrams,
            'types_evaluated': types_evaluated,
            'types_excluded': types_excluded,
        }

    def compute_bigram_entropy(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """Compute frequency-weighted normalized Shannon entropy of right-accessor distributions.

        For each token type t that appears as the left element of at least
        MIN_BIGRAM_OCCURRENCES bigrams, we compute the normalized Shannon entropy
        (η) of its right-successor distribution. The final score is the
        frequency-weighted mean of η across all qualifying types.

        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists

        Returns:
            Dict with bigram entropy results per tokenizer and language.
        """
        min_occ = self.min_bigram_occurrences

        results = {
            'per_tokenizer': {},
            'per_language': {},
            'pairwise_comparisons': {},
            'metadata': {
                'description': 'Bigram Entropy — frequency-weighted normalized Shannon entropy of right-accessor distributions',
                'reference': 'Poelman et al. 2025, EMNLP (Shannon efficiency η)',
                'metric_range': '[0.0, 1.0]',
                'interpretation': 'Higher = more uniform successor distributions',
                'min_bigram_occurrences': min_occ,
            }
        }

        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue

            tok_data = tokenized_data[tok_name]
            per_lang_metrics = {}
            global_right_accessors: Dict[int, Counter] = defaultdict(Counter)

            # Group data by language
            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)

            for lang, lang_data in lang_groups.items():
                lang_right_accessors: Dict[int, Counter] = defaultdict(Counter)
                for data in lang_data:
                    tokens = data.tokens
                    for i in range(len(tokens) - 1):
                        lang_right_accessors[tokens[i]][tokens[i + 1]] += 1
                        global_right_accessors[tokens[i]][tokens[i + 1]] += 1

                lang_stats = self._compute_weighted_entropy(lang_right_accessors, min_occ)
                per_lang_metrics[lang] = {
                    'bigram_entropy': lang_stats['entropy'],
                    'total_bigrams': lang_stats['total_ngrams'],
                    'types_evaluated': lang_stats['types_evaluated'],
                    'types_excluded': lang_stats['types_excluded'],
                }

            global_stats = self._compute_weighted_entropy(global_right_accessors, min_occ)

            results['per_tokenizer'][tok_name] = {
                'global_bigram_entropy': global_stats['entropy'],
                'global_total_bigrams': global_stats['total_ngrams'],
                'global_types_evaluated': global_stats['types_evaluated'],
                'global_types_excluded': global_stats['types_excluded'],
                'per_language': per_lang_metrics,
            }

        # Aggregate per-language results for cross-tokenizer comparison
        all_languages = set()
        for tok_results in results['per_tokenizer'].values():
            all_languages.update(tok_results['per_language'].keys())

        for lang in all_languages:
            results['per_language'][lang] = {}
            for tok_name in self.tokenizer_names:
                if tok_name in results['per_tokenizer']:
                    lang_data = results['per_tokenizer'][tok_name]['per_language'].get(lang)
                    if lang_data:
                        results['per_language'][lang][tok_name] = lang_data['bigram_entropy']

        # Pairwise comparisons on global bigram entropy
        global_entropies = {
            name: res['global_bigram_entropy']
            for name, res in results['per_tokenizer'].items()
        }
        results['pairwise_comparisons'] = self.compute_pairwise_comparisons(
            global_entropies, 'bigram_entropy'
        )

        return results

    def compute_trigram_entropy(self, tokenized_data: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        """Compute frequency-weighted normalized Shannon entropy of right-accessor
        distributions conditioned on bigram context.

        For each bigram context (t₁, t₂) that appears as the left context of at
        least min_trigram_occurrences trigrams, we compute the normalized Shannon
        entropy (η) of the successor distribution P(t₃ | t₁, t₂). The final
        score is the frequency-weighted mean of η across all qualifying contexts.

        Args:
            tokenized_data: Dict mapping tokenizer names to TokenizedData lists

        Returns:
            Dict with trigram entropy results per tokenizer and language.
        """
        min_occ = self.min_trigram_occurrences

        results = {
            'per_tokenizer': {},
            'per_language': {},
            'pairwise_comparisons': {},
            'metadata': {
                'description': 'Trigram Entropy — frequency-weighted normalized Shannon entropy of right-accessor distributions conditioned on bigram context',
                'reference': 'Extension of Poelman et al. 2025 (Shannon efficiency η) to trigram contexts',
                'metric_range': '[0.0, 1.0]',
                'interpretation': 'Higher = more uniform successor distributions given bigram context',
                'min_trigram_occurrences': min_occ,
            }
        }

        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue

            tok_data = tokenized_data[tok_name]
            per_lang_metrics = {}
            global_right_accessors: Dict[tuple, Counter] = defaultdict(Counter)

            lang_groups = TokenizedDataProcessor.group_by_language(tok_data)

            for lang, lang_data in lang_groups.items():
                lang_right_accessors: Dict[tuple, Counter] = defaultdict(Counter)
                for data in lang_data:
                    tokens = data.tokens
                    for i in range(len(tokens) - 2):
                        context = (tokens[i], tokens[i + 1])
                        lang_right_accessors[context][tokens[i + 2]] += 1
                        global_right_accessors[context][tokens[i + 2]] += 1

                lang_stats = self._compute_weighted_entropy(lang_right_accessors, min_occ)
                per_lang_metrics[lang] = {
                    'trigram_entropy': lang_stats['entropy'],
                    'total_trigrams': lang_stats['total_ngrams'],
                    'types_evaluated': lang_stats['types_evaluated'],
                    'types_excluded': lang_stats['types_excluded'],
                }

            global_stats = self._compute_weighted_entropy(global_right_accessors, min_occ)

            results['per_tokenizer'][tok_name] = {
                'global_trigram_entropy': global_stats['entropy'],
                'global_total_trigrams': global_stats['total_ngrams'],
                'global_types_evaluated': global_stats['types_evaluated'],
                'global_types_excluded': global_stats['types_excluded'],
                'per_language': per_lang_metrics,
            }

        # Aggregate per-language results for cross-tokenizer comparison
        all_languages = set()
        for tok_results in results['per_tokenizer'].values():
            all_languages.update(tok_results['per_language'].keys())

        for lang in all_languages:
            results['per_language'][lang] = {}
            for tok_name in self.tokenizer_names:
                if tok_name in results['per_tokenizer']:
                    lang_data = results['per_tokenizer'][tok_name]['per_language'].get(lang)
                    if lang_data:
                        results['per_language'][lang][tok_name] = lang_data['trigram_entropy']

        # Pairwise comparisons on global trigram entropy
        global_entropies = {
            name: res['global_trigram_entropy']
            for name, res in results['per_tokenizer'].items()
        }
        results['pairwise_comparisons'] = self.compute_pairwise_comparisons(
            global_entropies, 'trigram_entropy'
        )

        return results

    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        """Compute all information-theoretic metrics."""
        if tokenized_data is None:
            tokenized_data = self.get_tokenized_data()

        results = {}

        results['compression_rate'] = self.compute_compression_rate(tokenized_data)
        results['renyi_efficiency'] = self.compute_renyi_efficiency_analysis(tokenized_data)
        results['unigram_distribution_metrics'] = self.compute_unigram_distribution_metrics(tokenized_data)
        results['bigram_entropy'] = self.compute_bigram_entropy(tokenized_data)
        results['trigram_entropy'] = self.compute_trigram_entropy(tokenized_data)

        return results
