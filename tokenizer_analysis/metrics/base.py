"""
Base metrics class for unified TokenizedData interface - skeleton only.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import re
import numpy as np
import scipy
from collections import defaultdict
import logging

from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider
from ..constants import (
    DEFAULT_SAFE_DIVIDE_VALUE,
    PERCENTAGE_MULTIPLIER,
    MAX_ERROR_DISPLAY_COUNT,
)

logger = logging.getLogger(__name__)


class BaseMetrics(ABC):
    """Base class for tokenizer metrics using TokenizedData interface."""

    # Pre-compiled regex patterns for subword marker handling.
    # Shared by DigitBoundaryMetrics, ASTBoundaryMetrics, and MorphologicalMetrics.
    _SPACE_PREFIX = re.compile(r'^[Ġ▁ ]')
    _CONTINUATION = re.compile(r'^##')
    _END_WORD = re.compile(r'</w>$')
    _CONTINUATION_END = re.compile(r'@@$')
    _SPECIAL_TOKEN = re.compile(r'^(<\||\[).*(\|>|\])$')

    # Known byte-level BPE / SentencePiece character remappings.
    _DEFAULT_CHAR_DECODE: Dict[str, str] = {
        'Ġ': ' ', '▁': ' ', 'Ċ': '\n', 'ĉ': '\t', 'č': '\r',
    }

    def __init__(self, input_provider: InputProvider):
        self.input_provider = input_provider
        self.tokenizer_names = input_provider.get_tokenizer_names()
        self.language_metadata = None  # Can be set by subclasses
        self._tokenizer_vocab_cache: Dict[int, Dict[int, str]] = {}
        self._warned_tokenizers: set = set()
        self._char_decode_table: Optional[Dict[str, str]] = None
    
    def get_tokenized_data(self) -> Dict[str, List[TokenizedData]]:
        """Get tokenized data organized by tokenizer."""
        return self.input_provider.get_tokenized_data()
    
    def get_vocab_size(self, tokenizer_name: str) -> int:
        """Get vocabulary size for a tokenizer."""
        return self.input_provider.get_vocab_size(tokenizer_name)
    
    def get_languages(self, tokenizer_name: Optional[str] = None) -> List[str]:
        """Get available languages."""
        return self.input_provider.get_languages(tokenizer_name)

    # ------------------------------------------------------------------
    # Shared token conversion / cleaning helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_char_decode_table(tokenizer: Any) -> Dict[str, str]:
        """Probe *tokenizer* to discover its character remapping table.

        Encodes known whitespace characters (space, newline, tab, carriage
        return) and inspects the raw token strings for non-matching
        characters.  Returns a mapping from encoded character to the
        original character, e.g. ``{'Ġ': ' ', 'Ċ': '\\n'}``.

        Returns an empty dict when the tokenizer does not remap characters
        (e.g. WordPiece), or when probing fails.
        """
        if not hasattr(tokenizer, 'encode'):
            return {}

        probes = [
            ('a a', 1, ' '),   # space
            ('a\na', 1, '\n'), # newline
            ('a\ta', 1, '\t'), # tab
            ('a\ra', 1, '\r'), # carriage return
        ]
        table: Dict[str, str] = {}
        for text, target_pos, original_char in probes:
            try:
                token_ids = tokenizer.encode(text)
            except Exception:
                continue
            if not token_ids:
                continue
            # Convert IDs → raw token strings
            raw_tokens: Optional[List[str]] = None
            try:
                if hasattr(tokenizer, 'convert_ids_to_tokens'):
                    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
            except Exception:
                pass
            if not raw_tokens:
                continue
            # Look at the token at target_pos (the one after 'a')
            # and also search through all tokens for the remapping.
            for raw_tok in raw_tokens:
                if not isinstance(raw_tok, str) or len(raw_tok) < 1:
                    continue
                for ch in raw_tok:
                    if ch != original_char and ch not in 'aA' and ord(ch) > 127:
                        # This high-unicode char might be a remapping.
                        # Verify: does the token content make sense with this
                        # substitution?
                        if ch not in table:
                            table[ch] = original_char
        return table

    def _convert_ids_to_tokens(self, tokenizer: Any, token_ids: List[int]) -> List[str]:
        """Convert token IDs to strings with multiple fallback strategies.

        Fallback order:
        1. ``tokenizer.convert_ids_to_tokens``
        2. ``tokenizer.get_vocab`` → reverse mapping (cached)
        3. ``tokenizer.model.id_to_token``
        4. Placeholder strings ``<TOKEN_{id}>``
        """
        if not token_ids:
            return []

        tokenizer_id = id(tokenizer)

        # Fast path: use cached vocab reverse-mapping if available
        if tokenizer_id in self._tokenizer_vocab_cache:
            id_to_token = self._tokenizer_vocab_cache[tokenizer_id]
            return [id_to_token.get(tid, f"<UNK_{tid}>") for tid in token_ids]

        try:
            if hasattr(tokenizer, 'convert_ids_to_tokens'):
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                if tokens and all(isinstance(t, str) for t in tokens):
                    return tokens
        except Exception as e:
            logger.debug("convert_ids_to_tokens failed: %s", e)

        try:
            vocab = None
            if hasattr(tokenizer, 'get_vocab'):
                vocab = tokenizer.get_vocab()
            if vocab:
                self._tokenizer_vocab_cache[tokenizer_id] = {
                    v: (k.decode('utf-8') if isinstance(k, bytes) else str(k))
                    for k, v in vocab.items()
                }
                id_to_token = self._tokenizer_vocab_cache[tokenizer_id]
                return [id_to_token.get(tid, f"<UNK_{tid}>") for tid in token_ids]
        except Exception as e:
            logger.debug("Vocabulary lookup fallback failed: %s", e)

        try:
            if hasattr(tokenizer, 'model') and hasattr(tokenizer.model, 'id_to_token'):
                tokens = [tokenizer.model.id_to_token(tid) for tid in token_ids]
                if tokens and all(t is not None for t in tokens):
                    return [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in tokens]
        except Exception as e:
            logger.debug("Model id_to_token fallback failed: %s", e)

        if tokenizer_id not in self._warned_tokenizers:
            self._warned_tokenizers.add(tokenizer_id)
            logger.warning(
                "All token conversion methods failed for %s. Using placeholders.",
                type(tokenizer),
            )
        return [f"<TOKEN_{tid}>" for tid in token_ids]

    def _process_token(self, raw_token: str, preserve_space: bool = False) -> Optional[str]:
        """Shared token processing: strip subword markers, returning ``None`` for special tokens.

        Args:
            raw_token: Raw token string from the tokenizer vocabulary.
            preserve_space: If ``False`` (default), space-prefix markers (Ġ, ▁,
                leading space) are stripped entirely — the ``_clean_token`` path.
                If ``True``, space-prefix markers are replaced with a literal
                space — the ``_decode_raw_token`` path used for
                whitespace-preserving alignment.
        """
        if self._SPECIAL_TOKEN.match(raw_token):
            return None

        # Build effective decode table: always start with defaults, overlay
        # probed table when available.  This is safe because the default
        # entries (Ġ→' ', Ċ→'\n', etc.) are harmless for tokenizers that
        # never emit those characters.
        if self._char_decode_table:
            table = {**self._DEFAULT_CHAR_DECODE, **self._char_decode_table}
        else:
            table = self._DEFAULT_CHAR_DECODE

        # Apply character decode table to ALL characters
        decoded = ''.join(table.get(ch, ch) for ch in raw_token)

        # Check subword markers on the decoded result
        if self._CONTINUATION.match(decoded):
            return decoded[2:]
        if self._END_WORD.search(decoded):
            return decoded[:-4]
        if self._CONTINUATION_END.search(decoded):
            return decoded[:-2]

        # Handle leading space
        if decoded and decoded[0] == ' ':
            if preserve_space:
                return decoded
            return decoded[1:]

        return decoded

    def _clean_token(self, token: str) -> Optional[str]:
        """Strip subword markers from *token*, returning ``None`` for special tokens."""
        return self._process_token(token, preserve_space=False)

    def _build_char_to_token_map(
        self, token_strings: List[str]
    ) -> Tuple[str, List[int]]:
        """Build a mapping from character offset to token index.

        Returns ``(reconstructed_text, char_to_token)`` where
        ``char_to_token[i]`` is the token index that produced character *i*
        in the reconstructed text.
        """
        reconstructed: List[str] = []
        char_to_token: List[int] = []

        for idx, raw_token in enumerate(token_strings):
            cleaned = self._clean_token(raw_token)
            if cleaned is None:
                continue
            for ch in cleaned:
                reconstructed.append(ch)
                char_to_token.append(idx)

        return "".join(reconstructed), char_to_token

    @staticmethod
    def _build_source_to_recon_map(
        source_text: str, recon_text: str
    ) -> List[Optional[int]]:
        """Map each source-text character position to its position in the
        reconstructed (whitespace-stripped) text.

        Uses a greedy forward scan with exact (case-sensitive) matching.
        Characters dropped during reconstruction (e.g. whitespace consumed
        by subword prefixes) get ``None``.

        Returns a list of length ``len(source_text)`` where each entry is
        either a valid index into *recon_text* or ``None``.
        """
        source_to_recon: List[Optional[int]] = [None] * len(source_text)
        recon_idx = 0
        for src_idx in range(len(source_text)):
            if recon_idx >= len(recon_text):
                break
            if source_text[src_idx] == recon_text[recon_idx]:
                source_to_recon[src_idx] = recon_idx
                recon_idx += 1
        return source_to_recon

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_basic_stats(values: List[float]) -> Dict[str, float]:
        """Compute basic statistics for a list of values."""
        if not values:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'std_err': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0,
                'sum': 0
            }
            
        n = len(values)
        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values, ddof=1)) if n > 1 else 0.0,
            'std_err': float(scipy.stats.sem(values)) if n > 1 else 0.0,
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': n,
            'sum': sum(values)
        }
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = DEFAULT_SAFE_DIVIDE_VALUE) -> float:
        """Safely divide two numbers, returning default if denominator is zero."""
        return numerator / denominator if denominator != 0 else default
    
    def compute_pairwise_comparisons(self, values: Dict[str, float], metric_name: str = "metric") -> Dict[str, Dict[str, Any]]:
        """Compute pairwise comparisons between tokenizers."""
        comparisons = {}
        
        tokenizer_list = list(values.keys())
        for i, tok1 in enumerate(tokenizer_list):
            for j, tok2 in enumerate(tokenizer_list[i+1:], i+1):
                val1, val2 = values[tok1], values[tok2]
                
                comparison_key = f"{tok1}_vs_{tok2}"
                comparisons[comparison_key] = {
                    'tokenizer_1': tok1,
                    'tokenizer_2': tok2,
                    'value_1': val1,
                    'value_2': val2,
                    'difference': val1 - val2,
                    'ratio': self.safe_divide(val1, val2, 1.0),
                    'percent_difference': self.safe_divide(abs(val1 - val2), (val1 + val2) / 2, 0.0) * PERCENTAGE_MULTIPLIER
                }
        
        return comparisons
    
    @staticmethod
    def empty_stats() -> Dict[str, float]:
        """Return empty statistics dictionary with zero values."""
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'std_err': 0.0,
            'min': 0.0,
            'max': 0.0,
            'count': 0,
            'sum': 0
        }
    
    @staticmethod
    def validate_non_empty_data(data: Any, name: str) -> None:
        """Raise ValueError if data is empty."""
        if not data:
            raise ValueError(f"{name} cannot be empty")
    
    @staticmethod
    def validate_minimum_count(items: List[Any], min_count: int, name: str) -> None:
        """Raise ValueError if len(items) < min_count."""
        if len(items) < min_count:
            raise ValueError(f"{name} must have at least {min_count} items, got {len(items)}")
    
    @staticmethod
    def validate_positive_number(value: float, name: str) -> None:
        """Raise ValueError if value <= 0."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    
    @staticmethod
    def truncate_for_display(items: List[Any], max_count: int = MAX_ERROR_DISPLAY_COUNT) -> List[Any]:
        if len(items) <= max_count:
            return items
        return items[:max_count]
    
    @staticmethod
    def format_list_for_display(items: List[Any], max_count: int = MAX_ERROR_DISPLAY_COUNT) -> str:
        if len(items) <= max_count:
            return str(items)
        
        truncated = items[:max_count]
        return f"{truncated}... (showing {max_count}/{len(items)})"
    
    @abstractmethod
    def compute(self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None) -> Dict[str, Any]:
        """If tokenized_data is None, uses input_provider data."""
        pass


class TokenizedDataProcessor:
    """Utility class for processing TokenizedData objects."""
    
    @staticmethod
    def group_by_language(tokenized_data: List[TokenizedData]) -> Dict[str, List[TokenizedData]]:
        """Group TokenizedData objects by language."""
        grouped = defaultdict(list)
        for data in tokenized_data:
            grouped[data.language].append(data)
        return dict(grouped)
    
    @staticmethod
    def extract_tokens(tokenized_data: List[TokenizedData]) -> List[List[int]]:
        """Extract token lists from TokenizedData objects."""
        return [data.tokens for data in tokenized_data]
    
    @staticmethod
    def extract_texts(tokenized_data: List[TokenizedData]) -> List[str]:
        """Extract text strings from TokenizedData objects (where available)."""
        return [data.text for data in tokenized_data if data.text is not None]
    
    @staticmethod
    def flatten_all_tokens(tokenized_data: List[TokenizedData]) -> List[int]:
        """Flatten all tokens into a single list."""
        all_tokens = []
        for data in tokenized_data:
            all_tokens.extend(data.tokens)
        return all_tokens
    
    @staticmethod
    def count_total_tokens(tokenized_data: List[TokenizedData]) -> int:
        """Count total number of tokens across all data."""
        return sum(len(data.tokens) for data in tokenized_data)
    
    @staticmethod
    def get_unique_tokens(tokenized_data: List[TokenizedData]) -> set:
        """Get set of all unique token IDs."""
        unique_tokens = set()
        for data in tokenized_data:
            unique_tokens.update(data.tokens)
        return unique_tokens
    
