"""Per-example tokenization metrics.

The aggregate pipeline (BasicTokenizationMetrics, UTF8IntegrityMetrics) computes
metrics over language groups and corpora. This module exposes the same underlying
computations at single-document granularity — designed for joining with LM
eval `_samples.json` files where each entry has a `doc_id` and a single prompt.

All UTF-8 byte-conversion / validity / boundary-crossing logic is delegated to
the static methods on `UTF8IntegrityMetrics`. We do not duplicate that logic
here; we only iterate over a single document's tokens and aggregate per-doc.
"""

from typing import Any, Dict, List, Optional

from .config import (
    DEFAULT_WORD_MEASUREMENT_CONFIG,
    TextMeasurementConfig,
    TextMeasurer,
)
from .metrics.utf8_integrity import (
    UTF8IntegrityMetrics,
    _GPT2_DETECTION_THRESHOLD,
    _GPT2_MARKER_CHARS,
    _GPT2_UNICODE_TO_BYTE,
)


def _encode_to_ids(tokenizer, text: str) -> List[int]:
    """Encode text and return a plain list[int] of token ids.

    Handles three common tokenizer interfaces:
    - `tokenizers.Tokenizer` (HF rust): `.encode(text)` returns an Encoding with `.ids`
    - `transformers.PreTrainedTokenizer*`: `.encode(text, add_special_tokens=False)` returns list[int]
    - Custom: `.encode(text)` returns list[int]
    """
    try:
        # transformers tokenizers accept add_special_tokens; do NOT add them
        # so the per-example metrics reflect raw content tokenization.
        out = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        out = tokenizer.encode(text)
    if hasattr(out, "ids"):
        return list(out.ids)
    return list(out)


def _ids_to_token_strings(tokenizer, ids: List[int]) -> List[str]:
    """Convert token ids back to their vocab strings."""
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        # transformers
        return tokenizer.convert_ids_to_tokens(ids)
    if hasattr(tokenizer, "id_to_token"):
        # tokenizers.Tokenizer
        return [tokenizer.id_to_token(i) for i in ids]
    raise ValueError(
        f"Tokenizer {type(tokenizer)} has neither convert_ids_to_tokens nor id_to_token"
    )


def maybe_detect_gpt2(tokenizer) -> Optional[Dict[str, int]]:
    """Heuristic detection of GPT-2-style byte encoding. Returns the unicode_to_byte
    table if detected (so per-example callers can pass it through), else None.

    Mirrors the cached detection in UTF8IntegrityMetrics._detect_gpt2_encoding
    but without requiring an instance.
    """
    try:
        vocab = None
        if hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()
        elif hasattr(tokenizer, "vocab"):
            vocab = tokenizer.vocab
        if not vocab:
            return None
        marker_count = 0
        for token_str in vocab:
            if isinstance(token_str, bytes):
                token_str = token_str.decode("utf-8", errors="replace")
            if len(str(token_str)) == 1 and token_str in _GPT2_MARKER_CHARS:
                marker_count += 1
                if marker_count >= _GPT2_DETECTION_THRESHOLD:
                    return _GPT2_UNICODE_TO_BYTE
        return None
    except Exception:
        return None


def per_example_basic(
    tokenizer,
    text: str,
    measurer: Optional[TextMeasurer] = None,
) -> Dict[str, float]:
    """Basic per-example tokenization metrics for one (tokenizer, text) pair.

    Returned keys:
    - n_tokens: tokens emitted (no specials)
    - n_bytes: UTF-8 byte length of `text`
    - n_chars: codepoint count of `text`
    - n_words: word count via the default word measurer (Python str.split)
    - fertility_words: n_tokens / n_words (the value reported in the aggregate pipeline)
    - bytes_per_token: n_bytes / n_tokens
    - chars_per_token: n_chars / n_tokens
    - tokens_per_byte: n_tokens / n_bytes (inverse of bytes_per_token; sometimes preferred)
    """
    if measurer is None:
        measurer = TextMeasurer(DEFAULT_WORD_MEASUREMENT_CONFIG)

    ids = _encode_to_ids(tokenizer, text)
    n_tokens = len(ids)
    n_bytes = len(text.encode("utf-8"))
    n_chars = len(text)
    n_words = measurer.get_unit_count(text)

    return {
        "n_tokens": n_tokens,
        "n_bytes": n_bytes,
        "n_chars": n_chars,
        "n_words": n_words,
        "fertility_words": (n_tokens / n_words) if n_words > 0 else float("nan"),
        "bytes_per_token": (n_bytes / n_tokens) if n_tokens > 0 else float("nan"),
        "chars_per_token": (n_chars / n_tokens) if n_tokens > 0 else float("nan"),
        "tokens_per_byte": (n_tokens / n_bytes) if n_bytes > 0 else float("nan"),
    }


def per_example_utf8_integrity(
    tokenizer,
    text: str,
    gpt2_unicode_to_byte: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """UTF-8 boundary integrity per single example.

    Per-doc version of UTF8IntegrityMetrics.compute(). Uses the same
    `_token_string_to_bytes` / `_is_valid_complete_utf8` /
    `_crosses_character_boundary` helpers as the aggregate pipeline.

    Returns:
    - n_content_tokens: tokens excluding specials / placeholders
    - n_valid_complete: tokens whose bytes form complete valid UTF-8
    - n_malformed: tokens whose bytes are not valid UTF-8
    - n_boundary_crossing: tokens whose bytes span a multi-byte character boundary
        (BPE merges that fused fragments of two adjacent characters)
    - n_byte_fallback: byte-fallback tokens (`<0xAB>`)
    - integrity_rate: n_valid_complete / n_content_tokens
    - boundary_crossing_rate: n_boundary_crossing / n_content_tokens
    - byte_fallback_rate: n_byte_fallback / n_content_tokens
    """
    if gpt2_unicode_to_byte is None:
        gpt2_unicode_to_byte = maybe_detect_gpt2(tokenizer)

    ids = _encode_to_ids(tokenizer, text)
    token_strs = _ids_to_token_strings(tokenizer, ids)

    n_total = 0
    n_byte_fallback = 0
    n_boundary_crossing = 0
    n_malformed = 0
    n_valid_complete = 0

    for token_str in token_strs:
        if token_str is None:
            continue
        # Byte-fallback marker
        if (
            len(token_str) >= 6
            and token_str.startswith("<0x")
            and token_str.endswith(">")
        ):
            n_byte_fallback += 1

        b = UTF8IntegrityMetrics._token_string_to_bytes(token_str, gpt2_unicode_to_byte)
        if b is None:
            continue  # specials / placeholders excluded from the denominator
        n_total += 1

        if UTF8IntegrityMetrics._is_valid_complete_utf8(b):
            n_valid_complete += 1
        else:
            n_malformed += 1

        if UTF8IntegrityMetrics._crosses_character_boundary(b):
            n_boundary_crossing += 1

    safe_n = max(1, n_total)
    return {
        "n_content_tokens": n_total,
        "n_valid_complete": n_valid_complete,
        "n_malformed": n_malformed,
        "n_boundary_crossing": n_boundary_crossing,
        "n_byte_fallback": n_byte_fallback,
        "integrity_rate": n_valid_complete / safe_n,
        "boundary_crossing_rate": n_boundary_crossing / safe_n,
        "byte_fallback_rate": n_byte_fallback / safe_n,
    }


def per_example_all(
    tokenizer,
    text: str,
    measurer: Optional[TextMeasurer] = None,
    gpt2_unicode_to_byte: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: returns the union of `per_example_basic` and
    `per_example_utf8_integrity` for one (tokenizer, text) pair.

    GPT-2 detection is cached in the optional `gpt2_unicode_to_byte` argument —
    callers iterating many texts for the same tokenizer should detect once and
    pass it in to avoid re-scanning the vocab per call.
    """
    out = per_example_basic(tokenizer, text, measurer=measurer)
    out.update(per_example_utf8_integrity(tokenizer, text, gpt2_unicode_to_byte))
    return out


# ----------------------------------------------------------------------
# Domain-specific per-example metric wrappers (math + code)
# ----------------------------------------------------------------------

# A trivial InputProvider stub for instantiating metric classes that
# normally require one. The metric classes we use here (DigitBoundaryMetrics,
# ASTBoundaryMetrics) only consult ``self.input_provider`` inside their
# aggregator ``compute()`` paths — the per-text methods don't touch it.
class _StubInputProvider:
    """Minimal InputProvider stub used solely to satisfy BaseMetrics.__init__.
    Not used by compute_per_text; provided so the metric classes can be
    instantiated outside the standard pipeline.
    """
    def get_tokenizer(self, name):  # pragma: no cover (unused on the per-text path)
        return None

    def get_tokenizer_names(self):  # pragma: no cover
        return []

    def get_vocab_size(self, name):  # pragma: no cover
        return 0

    def get_tokenized_data(self):  # pragma: no cover
        return {}


# Module-level singletons. Instantiation is cheap (no data load required for
# DigitBoundaryMetrics; ASTBoundaryMetrics loads synthetic samples it doesn't
# use on the per-text path, which is also cheap).
#
# NOT THREAD-SAFE: ``compute_per_text`` snapshots and restores
# ``self._char_decode_table`` (a shared mutable attribute) around its body.
# Two threads calling ``per_example_*_alignment`` concurrently on the same
# singleton can race and observe each other's char-decode table during the
# critical section. For multi-threaded use, either (a) instantiate a fresh
# ``DigitBoundaryMetrics`` / ``ASTBoundaryMetrics`` per thread (cheap), or
# (b) guard the call site with a ``threading.Lock``. Single-threaded use
# (the default for our extraction scripts) is unaffected.
_DIGIT_INSTANCE = None
_AST_INSTANCE = None


def _get_digit_metric():
    global _DIGIT_INSTANCE
    if _DIGIT_INSTANCE is None:
        from .metrics.math import DigitBoundaryMetrics
        _DIGIT_INSTANCE = DigitBoundaryMetrics(_StubInputProvider())
    return _DIGIT_INSTANCE


def _get_ast_metric():
    global _AST_INSTANCE
    if _AST_INSTANCE is None:
        from .metrics.code_ast import ASTBoundaryMetrics
        _AST_INSTANCE = ASTBoundaryMetrics(_StubInputProvider())
    return _AST_INSTANCE


def per_example_digit_alignment(
    tokenizer,
    text: str,
    char_decode_table: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Per-doc digit boundary + operator isolation metrics for math prompts.

    Delegates to ``DigitBoundaryMetrics.compute_per_text`` — uses the EXACT
    canonical alignment scoring (``_score_boundaries``,
    ``_get_digit_span_boundaries``, ``_find_number_spans``).
    """
    return _get_digit_metric().compute_per_text(
        tokenizer, text, char_decode_table=char_decode_table,
    )


def per_example_ast_alignment(
    tokenizer,
    source_code: str,
    language: str = "python",
    char_decode_table: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Per-doc AST boundary alignment metrics for code prompts.

    Delegates to ``ASTBoundaryMetrics.compute_per_text`` — uses the EXACT
    canonical alignment scoring (``_check_boundary_alignment_fast``,
    ``_count_identifier_tokens_fast``) plus in-process tree-sitter parsing
    (``parse_snippets``).

    ``language``: pass the source code's language (e.g. ``"python"``,
    ``"javascript"``). MBPP and HumanEval are Python.
    """
    return _get_ast_metric().compute_per_text(
        tokenizer, source_code, language=language, char_decode_table=char_decode_table,
    )
