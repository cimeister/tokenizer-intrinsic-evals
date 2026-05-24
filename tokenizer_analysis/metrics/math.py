"""
Digit boundary alignment, digit split variability, numeric magnitude
consistency, and operator isolation metrics.

Based on Singh & Strouse (2024, arXiv:2402.14903), which showed that
right-to-left tokenization of numbers (enforced by comma-grouped formats)
improved arithmetic accuracy by >22 pp.  The mechanism: when digit
positions within operands and result occupy the same position within their
respective tokens, the model can learn a consistent positional mapping.

Four failure modes are measured here:

1. **Three-Digit Boundary Alignment Score** -- the tokenizer splits at the
   wrong positions inside a number.
2. **Digit Split Variability** -- the tokenizer splits numbers of
   the same digit length at *different* positions depending on the
   specific digit values (value-dependent BPE merges).
3. **Numeric Magnitude Consistency** -- does fertility (tokens per digit)
   scale predictably across digit magnitudes?
4. **Operator Isolation Rate** -- are mathematical operators tokenized as
   isolated units?
"""

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import spearmanr
import logging

from .base import BaseMetrics, TokenizedDataProcessor
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider
from ..utils.text_utils import load_math_data, BUILTIN_MATH_SAMPLES_PATH

logger = logging.getLogger(__name__)


class DigitBoundaryMetrics(BaseMetrics):
    """Digit boundary alignment and digit split variability.

    Worked examples -- Three-Digit Boundary Alignment
    =============================================

    ``"1234567"`` (L=7) -- ideal boundaries: {1, 4}  (i.e. ``"1|234|567"``)

    * Tokenized as ``"1" "234" "567"``   -> actual {1,4} -> P=1.0 R=1.0 F1=1.0
    * Tokenized as ``"1234" "567"``      -> actual {4}   -> P=1.0 R=0.5 F1=0.67
    * Tokenized as ``"12" "345" "67"``   -> actual {2,5} -> P=0.0 R=0.0 F1=0.0
    * Tokenized as ``"1234567"``         -> actual {}    -> P=1.0 R=0.0 F1=0.0

    ``"42"`` (L=2) -- ideal boundaries: {}

    * Tokenized as ``"42"``   -> actual {} -> P=1.0 R=1.0 F1=1.0
    * Tokenized as ``"4" "2"`` -> actual {1} -> P=0.0 R=1.0 F1=0.0

    Worked examples -- Digit Split Variability
    =================================================

    L=4, numbers ``["1234","5678","9012","3456"]`` all tokenized as
    ``["X","XXX"]`` -> patterns: {(1,): 4}.  H = 0.0, normalised = 0.0

    L=4, ``"1234"`` -> ``(2,)``, ``"5678"`` -> ``(1,)``,
    ``"9012"`` -> ``(2,)``, ``"3456"`` -> ``(1,)``.
    Distribution: {(2,): 0.5, (1,): 0.5}.
    H = 1.0 bit, normalised = 1.0
    """

    _DIGIT_SPAN = re.compile(r'\d+')
    # NOTE: The hyphen-minus ``-`` is always treated as an operator, even when
    # it appears as a unary negative sign (e.g. ``-42``).  Disambiguating
    # unary minus from subtraction requires expression parsing, and for
    # tokenizer evaluation the simpler rule is sufficient.
    _OPERATOR_SPAN = re.compile(r'(?:\*\*|//|<<|>>|<=|>=|=>|==|!=|&&|\|\||\?:|[+\-*/=<>!&|^~%])')

    _OPERATOR_CATEGORIES: Dict[str, List[str]] = {
        "arithmetic": ["+", "-", "*", "/", "//", "%", "**"],
        "comparison": ["<", ">", "<=", ">=", "==", "!="],
        "assignment": ["=", "=>"],
        "logical_bitwise": ["&", "|", "^", "~", "&&", "||"],
        "shift": ["<<", ">>"],
        "ternary": ["?:"],
    }
    _OPERATOR_TO_CATEGORY: Dict[str, str] = {}
    for _cat, _ops in _OPERATOR_CATEGORIES.items():
        for _op in _ops:
            _OPERATOR_TO_CATEGORY[_op] = _cat

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        input_provider: InputProvider,
        math_data_path: Optional[str] = None,
        use_builtin_math_data: bool = False,
    ):
        super().__init__(input_provider)
        self._math_data_path = math_data_path
        self._math_texts: List[str] = []
        if math_data_path:
            self._math_texts = load_math_data(math_data_path)
            logger.info(
                "Loaded %d math texts from %s", len(self._math_texts), math_data_path
            )
        elif use_builtin_math_data:
            self._math_texts = load_math_data(BUILTIN_MATH_SAMPLES_PATH)
            logger.info(
                "Loaded %d built-in math samples", len(self._math_texts)
            )

    # ------------------------------------------------------------------
    # Digit-span boundary extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _find_number_spans(text: str) -> List[Tuple[int, int, str]]:
        """Return ``[(start, end, digit_string), ...]`` for every ``\\d+`` in *text*."""
        return [(m.start(), m.end(), m.group()) for m in DigitBoundaryMetrics._DIGIT_SPAN.finditer(text)]

    @staticmethod
    def _get_digit_span_boundaries(
        char_to_token: List[int],
        span_start: int,
        span_end: int,
    ) -> Optional[List[int]]:
        """Return internal boundary positions *within* a digit span.

        *span_start* and *span_end* are character offsets **in the
        reconstructed text** (the same text that *char_to_token* was built
        from).  A boundary at position *k* means there is a token change
        between the (k-1)-th and k-th digit of the span.

        Returns ``None`` when the span exceeds the mapped region.
        """
        num_digits = span_end - span_start

        if span_end > len(char_to_token):
            return None

        boundaries: List[int] = []
        for k in range(1, num_digits):
            if char_to_token[span_start + k] != char_to_token[span_start + k - 1]:
                boundaries.append(k)

        return boundaries

    # ------------------------------------------------------------------
    # Ideal (right-aligned) boundaries
    # ------------------------------------------------------------------

    @staticmethod
    def _ideal_boundaries(num_digits: int) -> Set[int]:
        """Right-aligned grouping boundaries at positions 3, 6, 9, ... from the right.

        >>> DigitBoundaryMetrics._ideal_boundaries(7)
        {1, 4}
        >>> DigitBoundaryMetrics._ideal_boundaries(3)
        set()
        >>> DigitBoundaryMetrics._ideal_boundaries(6)
        {3}
        """
        result: Set[int] = set()
        pos = num_digits - 3
        while pos > 0:
            result.add(pos)
            pos -= 3
        return result

    # ------------------------------------------------------------------
    # Boundary scoring with vacuous-case semantics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_boundaries(
        actual: Set[int], ideal: Set[int]
    ) -> Dict[str, float]:
        """Compute precision / recall / F1 with vacuous-case rules.

        Vacuous-case table
        (P = precision, R = recall):

        ========  =====  ===  ===  ===  =========================================
        actual    ideal   P    R   F1   Rationale
        --------  -----  ---  ---  ---  -----------------------------------------
        empty     empty  1.0  1.0  1.0  Short number, single token -- nothing to
                                         get wrong, nothing to miss.
        non-empty empty  0.0  1.0  0.0  Short number needlessly split -- all
                                         boundaries are spurious (P=0).  R=1 by
                                         vacuous truth (zero ideal boundaries
                                         were all trivially recovered).  F1=0
                                         because precision dominates.
        empty     non-∅  1.0  0.0  0.0  Long number kept as single token -- no
                                         spurious boundaries (P=1) but none of
                                         the ideal ones were produced (R=0).
        non-empty non-∅  TP/(TP+FP)  TP/(TP+FN)  harmonic mean
        ========  =====  ===  ===  ===  =========================================
        """
        if not actual and not ideal:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if actual and not ideal:
            return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
        if not actual and ideal:
            return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

        tp = len(actual & ideal)
        fp = len(actual - ideal)
        fn = len(ideal - actual)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    # ------------------------------------------------------------------
    # Pattern entropy
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pattern_entropy(
        patterns: List[tuple],
    ) -> Dict[str, Any]:
        """Shannon entropy (bits) of the boundary-pattern distribution.

        Returns a dict with *entropy* (in bits), *num_patterns*,
        *dominant_pattern*, *dominant_pattern_freq*, and *count*.
        """
        if not patterns:
            return {
                "entropy": 0.0,
                "num_patterns": 0,
                "dominant_pattern": (),
                "dominant_pattern_freq": 0.0,
                "count": 0,
            }

        counts: Dict[tuple, int] = defaultdict(int)
        for p in patterns:
            counts[p] += 1

        total = len(patterns)
        k = len(counts)

        dominant_pattern = max(counts, key=counts.get)  # type: ignore[arg-type]
        dominant_freq = counts[dominant_pattern] / total

        if k <= 1:
            return {
                "entropy": 0.0,
                "num_patterns": k,
                "dominant_pattern": dominant_pattern,
                "dominant_pattern_freq": dominant_freq,
                "count": total,
            }

        entropy = 0.0
        for cnt in counts.values():
            p = cnt / total
            if p > 0:
                entropy -= p * math.log2(p)

        return {
            "entropy": entropy,
            "num_patterns": k,
            "dominant_pattern": dominant_pattern,
            "dominant_pattern_freq": dominant_freq,
            "count": total,
        }

    # ------------------------------------------------------------------
    # Bucket helpers
    # ------------------------------------------------------------------

    _SHORT_THRESHOLD = 3  # digit lengths <= this are "short"

    @staticmethod
    def _digit_length_bucket(length: int) -> str:
        """Return the per-length key: ``'1'`` .. ``'9'`` or ``'10+'``."""
        if length <= 9:
            return str(length)
        return "10+"

    @staticmethod
    def _is_short_bucket(bucket: str) -> bool:
        """Whether *bucket* (a digit-length key) falls in the short category.

        Short = digit lengths 1..3.  The ``'10+'`` bucket is always long.
        """
        if bucket.endswith("+"):
            return False
        return int(bucket) <= DigitBoundaryMetrics._SHORT_THRESHOLD

    # ------------------------------------------------------------------
    # Main compute
    # ------------------------------------------------------------------

    def compute(
        self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None
    ) -> Dict[str, Any]:
        """Compute digit boundary alignment, digit split variability,
        numeric magnitude consistency, and operator isolation rate.

        When *math_data_path* was provided at construction time, this method
        tokenizes the loaded math texts with each tokenizer and uses that
        data **instead of** the ``tokenized_data`` parameter.

        Returns a dict with four top-level keys:
        ``three_digit_boundary_alignment``, ``digit_split_variability``,
        ``numeric_magnitude_consistency``, and ``operator_isolation_rate``.
        """
        if self._math_texts:
            # Build tokenized data from the dedicated math texts.
            tokenized_data = {}
            for tok_name in self.tokenizer_names:
                tokenizer_obj = self.input_provider.get_tokenizer(tok_name)
                items: List[TokenizedData] = []
                for text in self._math_texts:
                    tokens = tokenizer_obj.encode(text)
                    items.append(
                        TokenizedData(
                            tokenizer_name=tok_name,
                            language="math",
                            tokens=tokens,
                            text=text,
                        )
                    )
                tokenized_data[tok_name] = items
            logger.info(
                "Using %d dedicated math texts for digit boundary metrics",
                len(self._math_texts),
            )
        elif tokenized_data is None:
            tokenized_data = self.input_provider.get_tokenized_data()

        # ---- accumulators ----
        # alignment: tok -> lang -> digit_length_str -> list of per-number dicts
        alignment_acc: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        # entropy: tok -> lang -> digit_length_str -> list of boundary pattern tuples
        entropy_acc: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        # operator: tok -> lang -> category -> {isolated, total, compound_ok, compound_total}
        operator_acc: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(
                lambda: {"isolated": 0, "total": 0, "compound_ok": 0, "compound_total": 0}
            ))
        )

        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue

            tokenizer_obj = self.input_provider.get_tokenizer(tok_name)
            self._char_decode_table = self._build_char_decode_table(tokenizer_obj)
            lang_groups = TokenizedDataProcessor.group_by_language(
                tokenized_data[tok_name]
            )

            for lang, data_list in lang_groups.items():
                for item in data_list:
                    if item.text is None or not item.text.strip():
                        continue

                    # Quick check: any digits or operators at all?
                    has_digits = bool(self._DIGIT_SPAN.search(item.text))
                    has_operators = bool(self._OPERATOR_SPAN.search(item.text))
                    if not has_digits and not has_operators:
                        continue

                    token_strings = self._convert_ids_to_tokens(
                        tokenizer_obj, item.tokens
                    )
                    recon_text, char_to_token = self._build_char_to_token_map(
                        token_strings
                    )

                    # Map original source positions → reconstructed-text positions
                    # so we can find spans on the original (whitespace-preserving)
                    # text and still look up token indices via char_to_token.
                    source_to_recon = self._build_source_to_recon_map(
                        item.text, recon_text
                    )

                    if has_digits:
                        # Find digit spans on the *original* text to avoid
                        # merging adjacent numbers separated by whitespace.
                        spans = self._find_number_spans(item.text)

                        for src_start, src_end, digit_str in spans:
                            num_digits = len(digit_str)
                            if num_digits == 0:
                                continue

                            # Map source span to reconstructed-text positions
                            recon_positions = [
                                source_to_recon[i]
                                for i in range(src_start, src_end)
                                if source_to_recon[i] is not None
                            ]
                            if len(recon_positions) != num_digits:
                                # Not all digits mapped — skip this span
                                continue
                            span_start = recon_positions[0]
                            span_end = recon_positions[-1] + 1

                            bucket = self._digit_length_bucket(num_digits)

                            boundaries = self._get_digit_span_boundaries(
                                char_to_token, span_start, span_end,
                            )
                            if boundaries is None:
                                continue

                            actual = set(boundaries)
                            ideal = self._ideal_boundaries(num_digits)
                            scores = self._score_boundaries(actual, ideal)

                            # Uniform-chunk: all token pieces have same digit count
                            # Reconstruct chunk lengths from boundaries
                            bnd_list = sorted(boundaries)
                            chunk_lengths = []
                            prev = 0
                            for b in bnd_list:
                                chunk_lengths.append(b - prev)
                                prev = b
                            chunk_lengths.append(num_digits - prev)
                            uniform_chunk = 1.0 if len(set(chunk_lengths)) <= 1 else 0.0

                            single_token = 1 if len(bnd_list) == 0 else 0

                            num_tokens = len(bnd_list) + 1
                            fertility_per_digit = num_tokens / num_digits

                            alignment_acc[tok_name][lang][bucket].append({
                                "precision": scores["precision"],
                                "recall": scores["recall"],
                                "f1": scores["f1"],
                                "uniform_chunk": uniform_chunk,
                                "single_token": single_token,
                                "fertility_per_digit": fertility_per_digit,
                            })

                            pattern = tuple(sorted(boundaries))
                            entropy_acc[tok_name][lang][bucket].append(pattern)

                    if has_operators:
                        # Build reverse map: token_index -> set of char positions
                        token_to_chars: Dict[int, Set[int]] = defaultdict(set)
                        for ci, ti in enumerate(char_to_token):
                            token_to_chars[ti].add(ci)

                        for m in self._OPERATOR_SPAN.finditer(recon_text):
                            op_str = m.group()
                            op_start = m.start()
                            op_end = m.end()
                            category = self._OPERATOR_TO_CATEGORY.get(op_str)
                            if category is None:
                                continue

                            # Get token indices covering this operator
                            op_token_indices = set()
                            for i in range(op_start, op_end):
                                if i < len(char_to_token):
                                    op_token_indices.add(char_to_token[i])

                            if not op_token_indices:
                                continue

                            cat_acc = operator_acc[tok_name][lang][category]
                            cat_acc["total"] += 1

                            # Isolated = all chars of those tokens fall within the operator span
                            op_char_set = set(range(op_start, op_end))
                            all_token_chars: Set[int] = set()
                            for ti in op_token_indices:
                                all_token_chars |= token_to_chars[ti]
                            isolated = all_token_chars.issubset(op_char_set)
                            if isolated:
                                cat_acc["isolated"] += 1

                            # Compound preservation: multi-char operator maps to exactly 1 token
                            if len(op_str) > 1:
                                cat_acc["compound_total"] += 1
                                if len(op_token_indices) == 1:
                                    cat_acc["compound_ok"] += 1

        self._char_decode_table = None

        # ---- build result structures ----
        alignment_results = self._build_alignment_results(alignment_acc)
        entropy_results = self._build_entropy_results(entropy_acc)
        magnitude_results = self._build_magnitude_results(alignment_acc)
        operator_results = self._build_operator_results(operator_acc)

        return {
            "three_digit_boundary_alignment": alignment_results,
            "digit_split_variability": entropy_results,
            "numeric_magnitude_consistency": magnitude_results,
            "operator_isolation_rate": operator_results,
        }

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_alignment_results(
        self, acc: Dict[str, Dict[str, Dict[str, list]]]
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {"per_tokenizer": {}, "summary": {}}

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {
                "by_digit_length": {},
                "by_bucket": {"short": {}, "long": {}},
                "overall": {},
            }

            all_f1: List[float] = []
            all_prec: List[float] = []
            all_rec: List[float] = []
            all_uniform: List[float] = []
            all_single: List[int] = []
            total_numbers = 0
            languages_seen: set = set()

            for lang in sorted(acc.get(tok_name, {})):
                lang_all_f1: List[float] = []
                lang_all_prec: List[float] = []
                lang_all_rec: List[float] = []
                lang_short_f1: List[float] = []
                lang_long_f1: List[float] = []

                for dl_str in sorted(acc[tok_name][lang]):
                    items = acc[tok_name][lang][dl_str]
                    if not items:
                        continue

                    is_short = self._is_short_bucket(dl_str)

                    f1s = [it["f1"] for it in items]
                    precs = [it["precision"] for it in items]
                    recs = [it["recall"] for it in items]
                    uniforms = [it["uniform_chunk"] for it in items]
                    singles = [it["single_token"] for it in items]

                    if dl_str not in tok_data["by_digit_length"]:
                        tok_data["by_digit_length"][dl_str] = {}

                    tok_data["by_digit_length"][dl_str][lang] = {
                        "mean_f1": float(np.mean(f1s)),
                        "mean_precision": float(np.mean(precs)),
                        "mean_recall": float(np.mean(recs)),
                        "mean_uniform_chunk": float(np.mean(uniforms)),
                        "single_token_frac": float(np.mean(singles)),
                        "count": len(items),
                    }

                    lang_all_f1.extend(f1s)
                    lang_all_prec.extend(precs)
                    lang_all_rec.extend(recs)

                    if is_short:
                        lang_short_f1.extend(f1s)
                    else:
                        lang_long_f1.extend(f1s)

                # Per-language bucket aggregation
                if lang_short_f1:
                    tok_data["by_bucket"]["short"][lang] = {
                        "mean_f1": float(np.mean(lang_short_f1)),
                        "count": len(lang_short_f1),
                    }
                if lang_long_f1:
                    tok_data["by_bucket"]["long"][lang] = {
                        "mean_f1": float(np.mean(lang_long_f1)),
                        "count": len(lang_long_f1),
                    }

                # Overall per-language
                if lang_all_f1:
                    tok_data["overall"][lang] = {
                        "mean_f1": float(np.mean(lang_all_f1)),
                        "mean_precision": float(np.mean(lang_all_prec)),
                        "mean_recall": float(np.mean(lang_all_rec)),
                        "count": len(lang_all_f1),
                    }
                    all_f1.extend(lang_all_f1)
                    all_prec.extend(lang_all_prec)
                    all_rec.extend(lang_all_rec)
                    all_uniform.extend(
                        it["uniform_chunk"]
                        for dl_items in acc[tok_name][lang].values()
                        for it in dl_items
                    )
                    all_single.extend(
                        it["single_token"]
                        for dl_items in acc[tok_name][lang].values()
                        for it in dl_items
                    )
                    total_numbers += len(lang_all_f1)
                    languages_seen.add(lang)

            results["per_tokenizer"][tok_name] = tok_data

            # Summary
            if all_f1:
                results["summary"][tok_name] = {
                    "avg_f1": float(np.mean(all_f1)),
                    "avg_precision": float(np.mean(all_prec)),
                    "avg_recall": float(np.mean(all_rec)),
                    "avg_uniform_chunk": float(np.mean(all_uniform)),
                    "single_token_frac": float(np.mean(all_single)),
                    "numbers_analyzed": total_numbers,
                    "languages_analyzed": len(languages_seen),
                }

        return results

    def _build_entropy_results(
        self, acc: Dict[str, Dict[str, Dict[str, list]]]
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {"per_tokenizer": {}, "summary": {}}

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {
                "by_digit_length": {},
                "by_bucket": {"short": {}, "long": {}},
                "overall": {},
            }

            # Collect raw patterns across all languages for pooled summary
            pooled_long_patterns: List[tuple] = []
            pooled_short_patterns: List[tuple] = []
            total_numbers = 0
            languages_seen: set = set()

            for lang in sorted(acc.get(tok_name, {})):
                lang_short_patterns: List[tuple] = []
                lang_long_patterns: List[tuple] = []

                for dl_str in sorted(acc[tok_name][lang]):
                    patterns = acc[tok_name][lang][dl_str]
                    if not patterns:
                        continue

                    is_short = self._is_short_bucket(dl_str)

                    stats = self._compute_pattern_entropy(patterns)

                    if dl_str not in tok_data["by_digit_length"]:
                        tok_data["by_digit_length"][dl_str] = {}

                    tok_data["by_digit_length"][dl_str][lang] = stats

                    if is_short:
                        lang_short_patterns.extend(patterns)
                    else:
                        lang_long_patterns.extend(patterns)

                # Per-language bucket aggregation
                if lang_short_patterns:
                    tok_data["by_bucket"]["short"][lang] = self._compute_pattern_entropy(
                        lang_short_patterns
                    )
                if lang_long_patterns:
                    tok_data["by_bucket"]["long"][lang] = self._compute_pattern_entropy(
                        lang_long_patterns
                    )

                # Overall per-language
                all_lang_patterns = lang_short_patterns + lang_long_patterns
                if all_lang_patterns:
                    tok_data["overall"][lang] = self._compute_pattern_entropy(
                        all_lang_patterns
                    )
                    total_numbers += len(all_lang_patterns)
                    languages_seen.add(lang)

                # Accumulate for pooled summary
                pooled_short_patterns.extend(lang_short_patterns)
                pooled_long_patterns.extend(lang_long_patterns)

            results["per_tokenizer"][tok_name] = tok_data

            if languages_seen:
                long_stats = self._compute_pattern_entropy(pooled_long_patterns)
                short_stats = self._compute_pattern_entropy(pooled_short_patterns)
                results["summary"][tok_name] = {
                    "entropy_long": long_stats["entropy"],
                    "entropy_short": short_stats["entropy"],
                    "numbers_analyzed": total_numbers,
                    "languages_analyzed": len(languages_seen),
                }

        return results

    # ------------------------------------------------------------------
    # Fertility scaling (for numeric magnitude consistency)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fertility_scaling(
        bucket_fertilities: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """Compute scaling statistics across digit-length buckets.

        Given ``{bucket_str: [fertility_per_digit_values, ...]}``, returns
        per-bucket stats plus overall scaling indicators (Spearman rho,
        coefficient of variation of mean fertility, and linear fit).
        """
        per_bucket: Dict[str, Dict[str, float]] = {}
        digit_lengths: List[float] = []
        mean_fertilities: List[float] = []

        for bucket_str in sorted(bucket_fertilities, key=lambda x: (len(x), x)):
            values = bucket_fertilities[bucket_str]
            if not values:
                continue
            m = float(np.mean(values))
            s = float(np.std(values))
            per_bucket[bucket_str] = {
                "mean_fertility": m,
                "std_fertility": s,
                "count": len(values),
            }
            # For scaling stats, use numeric digit length
            if bucket_str.endswith("+"):
                dl = 10.0
            else:
                dl = float(bucket_str)
            digit_lengths.append(dl)
            mean_fertilities.append(m)

        result: Dict[str, Any] = {"per_bucket": per_bucket}

        if len(digit_lengths) < 2:
            result["spearman_rho"] = None
            result["spearman_p"] = None
            result["cv_of_mean_fertility"] = 0.0
            result["linear_fit"] = None
            return result

        dl_arr = np.array(digit_lengths)
        mf_arr = np.array(mean_fertilities)

        # Spearman rank correlation (guard against constant input)
        if np.std(mf_arr) == 0.0:
            result["spearman_rho"] = 0.0
            result["spearman_p"] = 1.0
        else:
            rho, p_val = spearmanr(dl_arr, mf_arr)
            result["spearman_rho"] = float(rho)
            result["spearman_p"] = float(p_val)

        # Coefficient of variation of mean fertility across buckets
        overall_mean = float(np.mean(mf_arr))
        overall_std = float(np.std(mf_arr))
        result["cv_of_mean_fertility"] = (
            overall_std / overall_mean if overall_mean > 0 else 0.0
        )

        # Linear fit: mean_tokens = slope * num_digits + intercept
        # We fit mean_tokens (= mean_fertility * digit_length) vs digit_length
        mean_tokens_arr = mf_arr * dl_arr
        coeffs = np.polyfit(dl_arr, mean_tokens_arr, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])

        # R^2
        predicted = slope * dl_arr + intercept
        ss_res = float(np.sum((mean_tokens_arr - predicted) ** 2))
        ss_tot = float(np.sum((mean_tokens_arr - np.mean(mean_tokens_arr)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        result["linear_fit"] = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
        }

        return result

    # ------------------------------------------------------------------
    # Magnitude result builder
    # ------------------------------------------------------------------

    def _build_magnitude_results(
        self, alignment_acc: Dict[str, Dict[str, Dict[str, list]]]
    ) -> Dict[str, Any]:
        """Build the ``numeric_magnitude_consistency`` result dict.

        Derives fertility values from the ``fertility_per_digit`` field
        stored in each *alignment_acc* entry, eliminating the need for a
        separate magnitude accumulator.
        """
        results: Dict[str, Any] = {"per_tokenizer": {}, "summary": {}}

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {
                "by_digit_length": {},
                "overall": {},
                "scaling": {},
            }

            all_fertilities: List[float] = []
            # Collect bucket fertilities across all languages for global scaling
            global_bucket_fertilities: Dict[str, List[float]] = defaultdict(list)
            languages_seen: set = set()

            for lang in sorted(alignment_acc.get(tok_name, {})):
                lang_fertilities: List[float] = []

                for dl_str in sorted(alignment_acc[tok_name][lang]):
                    items = alignment_acc[tok_name][lang][dl_str]
                    if not items:
                        continue

                    values = [it["fertility_per_digit"] for it in items]

                    if dl_str not in tok_data["by_digit_length"]:
                        tok_data["by_digit_length"][dl_str] = {}

                    tok_data["by_digit_length"][dl_str][lang] = {
                        "mean_fertility": float(np.mean(values)),
                        "std_fertility": float(np.std(values)),
                        "count": len(values),
                    }

                    lang_fertilities.extend(values)
                    global_bucket_fertilities[dl_str].extend(values)

                if lang_fertilities:
                    tok_data["overall"][lang] = {
                        "mean_fertility": float(np.mean(lang_fertilities)),
                        "count": len(lang_fertilities),
                    }
                    all_fertilities.extend(lang_fertilities)
                    languages_seen.add(lang)

            # Scaling stats (across all languages pooled)
            tok_data["scaling"] = self._compute_fertility_scaling(
                global_bucket_fertilities
            )

            results["per_tokenizer"][tok_name] = tok_data

            # Summary
            if all_fertilities:
                scaling = tok_data["scaling"]
                summary: Dict[str, Any] = {
                    "avg_fertility": float(np.mean(all_fertilities)),
                    "cv_of_mean_fertility": scaling.get("cv_of_mean_fertility", 0.0),
                    "spearman_rho": scaling.get("spearman_rho"),
                    "numbers_analyzed": len(all_fertilities),
                    "languages_analyzed": len(languages_seen),
                }
                if scaling.get("linear_fit"):
                    summary["linear_r_squared"] = scaling["linear_fit"]["r_squared"]
                    summary["linear_slope"] = scaling["linear_fit"]["slope"]
                results["summary"][tok_name] = summary

        return results

    # ------------------------------------------------------------------
    # Operator result builder
    # ------------------------------------------------------------------

    def _build_operator_results(
        self, acc: Dict[str, Dict[str, Dict[str, Dict[str, int]]]]
    ) -> Dict[str, Any]:
        """Build the ``operator_isolation_rate`` result dict."""
        results: Dict[str, Any] = {"per_tokenizer": {}, "summary": {}}

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {
                "by_category": {},
                "by_language": {},
            }

            # Global totals across all languages and categories
            total_isolated = 0
            total_ops = 0
            total_compound_ok = 0
            total_compound = 0

            # Aggregate by category (across all languages)
            category_totals: Dict[str, Dict[str, int]] = defaultdict(
                lambda: {"isolated": 0, "total": 0, "compound_ok": 0, "compound_total": 0}
            )

            for lang in sorted(acc.get(tok_name, {})):
                lang_data: Dict[str, Any] = {"by_category": {}}
                lang_isolated = 0
                lang_total = 0
                lang_compound_ok = 0
                lang_compound_total = 0

                for category in sorted(acc[tok_name][lang]):
                    cat_data = acc[tok_name][lang][category]
                    t = cat_data["total"]
                    iso = cat_data["isolated"]
                    cok = cat_data["compound_ok"]
                    ctot = cat_data["compound_total"]

                    lang_data["by_category"][category] = {
                        "isolation_rate": iso / t if t > 0 else 0.0,
                        "compound_preservation_rate": cok / ctot if ctot > 0 else 0.0,
                        "total": t,
                        "compound_total": ctot,
                    }

                    lang_isolated += iso
                    lang_total += t
                    lang_compound_ok += cok
                    lang_compound_total += ctot

                    category_totals[category]["isolated"] += iso
                    category_totals[category]["total"] += t
                    category_totals[category]["compound_ok"] += cok
                    category_totals[category]["compound_total"] += ctot

                if lang_total > 0:
                    lang_data["isolation_rate"] = lang_isolated / lang_total
                    lang_data["compound_preservation_rate"] = (
                        lang_compound_ok / lang_compound_total
                        if lang_compound_total > 0 else 0.0
                    )
                    lang_data["total"] = lang_total
                    tok_data["by_language"][lang] = lang_data

                total_isolated += lang_isolated
                total_ops += lang_total
                total_compound_ok += lang_compound_ok
                total_compound += lang_compound_total

            # Per-category aggregation
            for category in sorted(category_totals):
                ct = category_totals[category]
                t = ct["total"]
                tok_data["by_category"][category] = {
                    "isolation_rate": ct["isolated"] / t if t > 0 else 0.0,
                    "compound_preservation_rate": (
                        ct["compound_ok"] / ct["compound_total"]
                        if ct["compound_total"] > 0 else 0.0
                    ),
                    "total": t,
                    "compound_total": ct["compound_total"],
                }

            results["per_tokenizer"][tok_name] = tok_data

            # Summary
            if total_ops > 0:
                results["summary"][tok_name] = {
                    "overall_isolation_rate": total_isolated / total_ops,
                    "overall_compound_preservation_rate": (
                        total_compound_ok / total_compound
                        if total_compound > 0 else 0.0
                    ),
                    "total_operators": total_ops,
                    "total_compound_operators": total_compound,
                }

        return results

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print digit boundary alignment and digit split variability results."""
        # ---- Alignment ----
        align = results.get("three_digit_boundary_alignment")
        if align:
            print("\n" + "=" * 60)
            print("THREE-DIGIT BOUNDARY ALIGNMENT RESULTS")
            print("=" * 60)

            if "summary" in align:
                print("\nSUMMARY STATISTICS")
                print("-" * 40)
                for tok_name in self.tokenizer_names:
                    if tok_name in align["summary"]:
                        s = align["summary"][tok_name]
                        print(f"{tok_name}:")
                        print(f"  {'Avg F1':20}: {s['avg_f1']:.3f}")
                        print(f"  {'Avg Precision':20}: {s['avg_precision']:.3f}")
                        print(f"  {'Avg Recall':20}: {s['avg_recall']:.3f}")
                        print(f"  {'Uniform Chunk':20}: {s['avg_uniform_chunk']:.3f}")
                        print(f"  {'Single-Token Frac':20}: {s['single_token_frac']:.3f}")
                        print(f"  {'Numbers Analyzed':20}: {s['numbers_analyzed']:,}")
                        print(f"  {'Languages':20}: {s['languages_analyzed']}")

            if "per_tokenizer" in align:
                print("\nDETAILED (by digit length)")
                print("-" * 60)
                for tok_name in self.tokenizer_names:
                    tok = align["per_tokenizer"].get(tok_name, {})
                    by_dl = tok.get("by_digit_length", {})
                    if not by_dl:
                        continue
                    print(f"\n{tok_name}:")
                    for dl in sorted(by_dl, key=lambda x: (len(x), x)):
                        lang_data = by_dl[dl]
                        for lang, d in sorted(lang_data.items()):
                            print(
                                f"  L={dl:>3} {lang:12}: "
                                f"F1={d['mean_f1']:.3f}  "
                                f"P={d['mean_precision']:.3f}  "
                                f"R={d['mean_recall']:.3f}  "
                                f"uniform={d['mean_uniform_chunk']:.2f}  "
                                f"n={d['count']}"
                            )

        # ---- Entropy ----
        ent = results.get("digit_split_variability")
        if ent:
            print("\n" + "=" * 60)
            print("DIGIT SPLIT VARIABILITY RESULTS")
            print("=" * 60)

            if "summary" in ent:
                print("\nSUMMARY STATISTICS (pooled across languages)")
                print("-" * 40)
                for tok_name in self.tokenizer_names:
                    if tok_name in ent["summary"]:
                        s = ent["summary"][tok_name]
                        print(f"{tok_name}:")
                        print(f"  {'Entropy (long)':25}: {s['entropy_long']:.3f} bits")
                        print(f"  {'Entropy (short)':25}: {s['entropy_short']:.3f} bits")
                        print(f"  {'Numbers Analyzed':25}: {s['numbers_analyzed']:,}")
                        print(f"  {'Languages':25}: {s['languages_analyzed']}")

            if "per_tokenizer" in ent:
                print("\nDETAILED (by digit length)")
                print("-" * 60)
                for tok_name in self.tokenizer_names:
                    tok = ent["per_tokenizer"].get(tok_name, {})
                    by_dl = tok.get("by_digit_length", {})
                    if not by_dl:
                        continue
                    print(f"\n{tok_name}:")
                    for dl in sorted(by_dl, key=lambda x: (len(x), x)):
                        lang_data = by_dl[dl]
                        for lang, d in sorted(lang_data.items()):
                            dom = d.get("dominant_pattern", ())
                            print(
                                f"  L={dl:>3} {lang:12}: "
                                f"H={d['entropy']:.3f} bits  "
                                f"K={d['num_patterns']}  "
                                f"dom={dom} ({d['dominant_pattern_freq']:.0%})  "
                                f"n={d['count']}"
                            )

        # ---- Magnitude Consistency ----
        mag = results.get("numeric_magnitude_consistency")
        if mag:
            print("\n" + "=" * 60)
            print("NUMERIC MAGNITUDE CONSISTENCY RESULTS")
            print("=" * 60)

            if "summary" in mag:
                print("\nSUMMARY STATISTICS")
                print("-" * 40)
                for tok_name in self.tokenizer_names:
                    if tok_name in mag["summary"]:
                        s = mag["summary"][tok_name]
                        print(f"{tok_name}:")
                        print(f"  {'Avg Fertility':25}: {s['avg_fertility']:.3f}")
                        print(f"  {'CV of Mean Fertility':25}: {s['cv_of_mean_fertility']:.3f}")
                        rho = s.get('spearman_rho')
                        rho_str = f"{rho:.3f}" if rho is not None else "N/A"
                        print(f"  {'Spearman rho':25}: {rho_str}")
                        if 'linear_r_squared' in s:
                            print(f"  {'Linear R^2':25}: {s['linear_r_squared']:.3f}")
                            print(f"  {'Linear Slope':25}: {s['linear_slope']:.3f}")
                        print(f"  {'Numbers Analyzed':25}: {s['numbers_analyzed']:,}")
                        print(f"  {'Languages':25}: {s['languages_analyzed']}")

            if "per_tokenizer" in mag:
                print("\nDETAILED (by digit length)")
                print("-" * 60)
                for tok_name in self.tokenizer_names:
                    tok = mag["per_tokenizer"].get(tok_name, {})
                    by_dl = tok.get("by_digit_length", {})
                    if not by_dl:
                        continue
                    print(f"\n{tok_name}:")
                    for dl in sorted(by_dl, key=lambda x: (len(x), x)):
                        lang_data = by_dl[dl]
                        for lang, d in sorted(lang_data.items()):
                            print(
                                f"  L={dl:>3} {lang:12}: "
                                f"fert={d['mean_fertility']:.3f} "
                                f"(std={d['std_fertility']:.3f})  "
                                f"n={d['count']}"
                            )

        # ---- Operator Isolation ----
        ops = results.get("operator_isolation_rate")
        if ops:
            print("\n" + "=" * 60)
            print("OPERATOR ISOLATION RATE RESULTS")
            print("=" * 60)

            if "summary" in ops:
                print("\nSUMMARY STATISTICS")
                print("-" * 40)
                for tok_name in self.tokenizer_names:
                    if tok_name in ops["summary"]:
                        s = ops["summary"][tok_name]
                        print(f"{tok_name}:")
                        print(f"  {'Isolation Rate':30}: {s['overall_isolation_rate']:.3f}")
                        print(f"  {'Compound Preservation':30}: {s['overall_compound_preservation_rate']:.3f}")
                        print(f"  {'Total Operators':30}: {s['total_operators']:,}")
                        print(f"  {'Total Compound Ops':30}: {s['total_compound_operators']:,}")

            if "per_tokenizer" in ops:
                print("\nBY CATEGORY")
                print("-" * 60)
                for tok_name in self.tokenizer_names:
                    tok = ops["per_tokenizer"].get(tok_name, {})
                    by_cat = tok.get("by_category", {})
                    if not by_cat:
                        continue
                    print(f"\n{tok_name}:")
                    for cat, d in sorted(by_cat.items()):
                        cpr = d.get('compound_preservation_rate', 0.0)
                        ct = d.get('compound_total', 0)
                        print(
                            f"  {cat:20}: "
                            f"isolation={d['isolation_rate']:.3f}  "
                            f"compound={cpr:.3f} ({ct} compound)  "
                            f"n={d['total']}"
                        )

            print("\n" + "=" * 60)

    # ------------------------------------------------------------------
    # Per-text entry point (independent of compute()'s aggregator pipeline)
    # ------------------------------------------------------------------

    def compute_per_text(
        self,
        tokenizer_obj: Any,
        text: str,
        char_decode_table: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Compute digit boundary alignment metrics for ONE text under ONE
        tokenizer, returning a per-document summary.

        Mirrors the per-text body inside ``compute()`` but does not pool across
        a corpus. Used by per-example correlation analysis. The standard
        ``compute()`` workflow is unaffected; this method snapshots and
        restores ``self._char_decode_table`` so calling it does not mutate
        aggregator state.

        Returns a dict with keys: ``n_digit_spans``, ``mean_digit_f1``,
        ``mean_fertility_per_digit``, ``single_token_number_rate``,
        ``uniform_chunk_rate``, ``n_operators``, ``operator_isolation_rate``,
        ``n_compound_operators``, ``compound_operator_preserved_rate``,
        ``n_tokens`` (the token count produced for the whole text — useful
        as a regression covariate).
        NaN is returned for ratios with empty denominators (e.g. no digit
        spans in the text).
        """
        empty = {
            "n_digit_spans": 0,
            "mean_digit_f1": float("nan"),
            "mean_fertility_per_digit": float("nan"),
            "single_token_number_rate": float("nan"),
            "uniform_chunk_rate": float("nan"),
            "n_operators": 0,
            "operator_isolation_rate": float("nan"),
            "n_compound_operators": 0,
            "compound_operator_preserved_rate": float("nan"),
            "n_tokens": 0,
            "parse_status": "empty",
        }

        if not text or not text.strip():
            return empty

        # Snapshot + restore aggregator state so compute() callers are unaffected.
        saved_table = getattr(self, "_char_decode_table", None)
        try:
            self._char_decode_table = (
                char_decode_table
                if char_decode_table is not None
                else self._build_char_decode_table(tokenizer_obj)
            )

            # ---- Encode and build char→token map (mirrors compute() body) ----
            try:
                try:
                    token_ids_raw = tokenizer_obj.encode(text, add_special_tokens=False)
                except TypeError:
                    token_ids_raw = tokenizer_obj.encode(text)
            except Exception as e:
                out = dict(empty)
                out["parse_status"] = f"encode_error:{type(e).__name__}"
                return out
            token_ids = (
                list(token_ids_raw.ids)
                if hasattr(token_ids_raw, "ids")
                else list(token_ids_raw)
            )

            token_strings = self._convert_ids_to_tokens(tokenizer_obj, token_ids)
            recon_text, char_to_token = self._build_char_to_token_map(token_strings)
            source_to_recon = self._build_source_to_recon_map(text, recon_text)

            # Match compute() early-exit semantics: both checks on the original text.
            has_digits = bool(self._DIGIT_SPAN.search(text))
            has_operators = bool(self._OPERATOR_SPAN.search(text))

            if not has_digits and not has_operators:
                out = dict(empty)
                out["n_tokens"] = len(token_ids)
                out["parse_status"] = "no_digits_or_operators"
                return out

            # ---- Per-digit-span scoring ----
            digit_f1s: List[float] = []
            digit_fertility: List[float] = []
            single_token_flags: List[float] = []
            uniform_chunk_flags: List[float] = []

            if has_digits:
                for src_start, src_end, digit_str in self._find_number_spans(text):
                    num_digits = len(digit_str)
                    if num_digits == 0:
                        continue
                    recon_positions = [
                        source_to_recon[i]
                        for i in range(src_start, src_end)
                        if source_to_recon[i] is not None
                    ]
                    if len(recon_positions) != num_digits:
                        continue
                    span_start = recon_positions[0]
                    span_end = recon_positions[-1] + 1

                    boundaries = self._get_digit_span_boundaries(
                        char_to_token, span_start, span_end,
                    )
                    if boundaries is None:
                        continue

                    actual = set(boundaries)
                    ideal = self._ideal_boundaries(num_digits)
                    scores = self._score_boundaries(actual, ideal)

                    bnd_list = sorted(boundaries)
                    chunk_lengths: List[int] = []
                    prev = 0
                    for b in bnd_list:
                        chunk_lengths.append(b - prev)
                        prev = b
                    chunk_lengths.append(num_digits - prev)
                    uniform_chunk = 1.0 if len(set(chunk_lengths)) <= 1 else 0.0
                    single_token = 1.0 if len(bnd_list) == 0 else 0.0
                    num_tokens = len(bnd_list) + 1
                    fertility_per_digit = num_tokens / num_digits

                    digit_f1s.append(scores["f1"])
                    digit_fertility.append(fertility_per_digit)
                    single_token_flags.append(single_token)
                    uniform_chunk_flags.append(uniform_chunk)

            # ---- Per-operator scoring ----
            isolated_flags: List[float] = []
            compound_total = 0
            compound_ok = 0

            if has_operators:
                token_to_chars: Dict[int, Set[int]] = defaultdict(set)
                for ci, ti in enumerate(char_to_token):
                    token_to_chars[ti].add(ci)
                for m in self._OPERATOR_SPAN.finditer(recon_text):
                    op_str = m.group()
                    op_start = m.start()
                    op_end = m.end()
                    category = self._OPERATOR_TO_CATEGORY.get(op_str)
                    if category is None:
                        continue
                    op_token_indices: Set[int] = set()
                    for i in range(op_start, op_end):
                        if i < len(char_to_token):
                            op_token_indices.add(char_to_token[i])
                    if not op_token_indices:
                        continue
                    op_char_set = set(range(op_start, op_end))
                    all_token_chars: Set[int] = set()
                    for ti in op_token_indices:
                        all_token_chars |= token_to_chars[ti]
                    isolated_flags.append(
                        1.0 if all_token_chars.issubset(op_char_set) else 0.0
                    )
                    if len(op_str) > 1:
                        compound_total += 1
                        if len(op_token_indices) == 1:
                            compound_ok += 1

            n_digit_spans = len(digit_f1s)
            n_operators = len(isolated_flags)
            return {
                "n_digit_spans": n_digit_spans,
                "mean_digit_f1": float(np.mean(digit_f1s)) if n_digit_spans else float("nan"),
                "mean_fertility_per_digit": (
                    float(np.mean(digit_fertility)) if n_digit_spans else float("nan")
                ),
                "single_token_number_rate": (
                    float(np.mean(single_token_flags)) if n_digit_spans else float("nan")
                ),
                "uniform_chunk_rate": (
                    float(np.mean(uniform_chunk_flags)) if n_digit_spans else float("nan")
                ),
                "n_operators": n_operators,
                "operator_isolation_rate": (
                    float(np.mean(isolated_flags)) if n_operators else float("nan")
                ),
                "n_compound_operators": compound_total,
                "compound_operator_preserved_rate": (
                    (compound_ok / compound_total) if compound_total else float("nan")
                ),
                "n_tokens": len(token_ids),
                "parse_status": "ok",
            }
        finally:
            self._char_decode_table = saved_table
