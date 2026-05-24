"""Tests for tokenizer_analysis.metrics.basic (BasicTokenizationMetrics)."""

import pytest

from tokenizer_analysis.metrics.basic import BasicTokenizationMetrics
from tokenizer_analysis.core.input_types import TokenizedData
from typing import Dict, List, Optional, Tuple

from .conftest import SimpleProvider as _SimpleProvider


def _make_td(tok_name: str, text: str, tokens: List[int], lang: str = "en") -> TokenizedData:
    return TokenizedData(
        tokenizer_name=tok_name,
        language=lang,
        tokens=tokens,
        text=text,
    )


# ======================================================================
# T5: Blank-line exclusion in avg_tokens_per_line
# ======================================================================

class TestBlankLineExclusion:

    def test_blank_lines_not_counted(self):
        """Blank lines should be excluded from line count."""
        tok_name = "test_tok"
        provider = _SimpleProvider(tok_name)
        metrics = BasicTokenizationMetrics(provider)

        # Text with 2 non-blank lines and 2 blank lines
        text = "hello world\n\ngoodbye world\n\n"
        td = {tok_name: [_make_td(tok_name, text, [1, 2, 3, 4])]}

        results = metrics.compute_avg_tokens_per_line_analysis(td)
        tpl_data = results["avg_tokens_per_line"]["per_tokenizer"][tok_name]
        # 4 tokens / 2 non-blank lines = 2.0
        assert tpl_data["global_avg"] == pytest.approx(2.0)

    def test_all_blank_lines(self):
        """Text with only blank lines should produce 0 tokens per line."""
        tok_name = "test_tok"
        provider = _SimpleProvider(tok_name)
        metrics = BasicTokenizationMetrics(provider)

        text = "\n\n\n"
        td = {tok_name: [_make_td(tok_name, text, [1])]}

        results = metrics.compute_avg_tokens_per_line_analysis(td)
        tpl_data = results["avg_tokens_per_line"]["per_tokenizer"][tok_name]
        assert tpl_data["global_avg"] == 0.0


# ======================================================================
# T6: Fertility skip when text is None
# ======================================================================

class TestFertilitySkip:

    def test_no_text_skipped(self):
        """Samples without text should be skipped, not use a fallback."""
        tok_name = "test_tok"
        provider = _SimpleProvider(tok_name)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [
            # Sample WITH text
            _make_td(tok_name, "hello world", [1, 2]),
            # Sample WITHOUT text
            TokenizedData(tokenizer_name=tok_name, language="en", tokens=[3, 4, 5]),
        ]}

        results = metrics.compute(td)
        fertility_data = results["fertility"]["per_tokenizer"][tok_name]["global"]
        # Only the first sample (2 tokens / 2 words = 1.0) should be counted
        assert fertility_data["count"] == 1

    def test_whitespace_only_text_skipped(self):
        """Whitespace-only texts should be skipped."""
        tok_name = "test_tok"
        provider = _SimpleProvider(tok_name)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [
            _make_td(tok_name, "   \n\t  ", [1, 2]),
            _make_td(tok_name, "actual text", [3, 4]),
        ]}

        results = metrics.compute(td)
        fertility_data = results["fertility"]["per_tokenizer"][tok_name]["global"]
        assert fertility_data["count"] == 1


# ======================================================================
# T7: Bytes-per-token metric
# ======================================================================

class TestBytesPerToken:

    def test_ascii_text(self):
        """For ASCII text, bytes_per_token == chars_per_token."""
        tok_name = "test_tok"
        provider = _SimpleProvider(tok_name)
        metrics = BasicTokenizationMetrics(provider)

        text = "hello"  # 5 ASCII chars = 5 bytes
        td = {tok_name: [_make_td(tok_name, text, [1, 2])]}

        results = metrics.compute_token_length_analysis(td)
        tok_data = results["token_length"]["per_tokenizer"][tok_name]
        assert "byte_length" in tok_data
        char_mean = tok_data["character_length"]["mean"]
        byte_mean = tok_data["byte_length"]["mean"]
        assert char_mean == pytest.approx(byte_mean)  # ASCII: same
        assert char_mean == pytest.approx(2.5)  # 5 chars / 2 tokens

    def test_multibyte_text(self):
        """For multi-byte UTF-8, bytes_per_token > chars_per_token."""
        tok_name = "test_tok"
        provider = _SimpleProvider(tok_name)
        metrics = BasicTokenizationMetrics(provider)

        text = "\u00e9\u00e9"  # 2 chars, each 2 bytes in UTF-8 = 4 bytes total
        td = {tok_name: [_make_td(tok_name, text, [1, 2])]}

        results = metrics.compute_token_length_analysis(td)
        tok_data = results["token_length"]["per_tokenizer"][tok_name]
        char_mean = tok_data["character_length"]["mean"]
        byte_mean = tok_data["byte_length"]["mean"]
        assert char_mean == pytest.approx(1.0)   # 2 chars / 2 tokens
        assert byte_mean == pytest.approx(2.0)   # 4 bytes / 2 tokens


# ======================================================================
# Mock decodable tokenizer and provider
# ======================================================================

class _MockDecodableTokenizer:
    """Minimal tokenizer wrapper with configurable encode/decode for tests."""

    def __init__(self, encode_fn=None, decode_fn=None, unk_id=None):
        self._encode_fn = encode_fn or (lambda t: list(range(len(t.split()))))
        self._decode_fn = decode_fn  # None means decode not supported
        self._unk_id = unk_id

    def get_name(self) -> str:
        return "mock_tok"

    def get_vocab_size(self) -> int:
        return 100

    def get_vocab(self) -> Optional[Dict[str, int]]:
        return None

    def can_encode(self) -> bool:
        return True

    def encode(self, text: str) -> List[int]:
        return self._encode_fn(text)

    def can_pretokenize(self) -> bool:
        return False

    def pretokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    def can_decode(self) -> bool:
        return self._decode_fn is not None

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> Optional[str]:
        if self._decode_fn is None:
            return None
        try:
            return self._decode_fn(token_ids)
        except Exception:
            return None

    def encode_with_offsets(self, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        return self.encode(text), None

    def get_unk_token_id(self) -> Optional[int]:
        return self._unk_id

    def has_unk_token(self) -> bool:
        return self._unk_id is not None

    @classmethod
    def from_config(cls, name, config):
        return cls()


class _MockDecodableProvider(_SimpleProvider):
    """Provider that wraps a _MockDecodableTokenizer."""

    def __init__(self, tok_name: str, tokenizer: _MockDecodableTokenizer):
        super().__init__(tok_name)
        self._tokenizer = tokenizer

    def get_tokenizer(self, name: str):
        return self._tokenizer


# ======================================================================
# T8: Reconstruction fidelity
# ======================================================================

class TestReconstructionFidelity:

    def test_perfect_roundtrip(self):
        """Perfect round-trip -> exact_match=1.0, CER=0.0."""
        tok_name = "mock_tok"
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, 2, 3],
            decode_fn=lambda ids: "hello world",
        )
        provider = _MockDecodableProvider(tok_name, tok)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [_make_td(tok_name, "hello world", [1, 2, 3])]}
        results = metrics.compute_reconstruction_fidelity_analysis(td)

        summary = results["reconstruction_fidelity"]["summary"][tok_name]
        assert summary["exact_match_rate"] == pytest.approx(1.0)
        assert summary["mean_cer"] == pytest.approx(0.0)

    def test_lossy_roundtrip(self):
        """Lossy round-trip -> exact_match=0.0, CER>0."""
        tok_name = "mock_tok"
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, 2],
            decode_fn=lambda ids: "helo world",  # missing 'l'
        )
        provider = _MockDecodableProvider(tok_name, tok)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [_make_td(tok_name, "hello world", [1, 2])]}
        results = metrics.compute_reconstruction_fidelity_analysis(td)

        summary = results["reconstruction_fidelity"]["summary"][tok_name]
        assert summary["exact_match_rate"] == pytest.approx(0.0)
        assert summary["mean_cer"] > 0.0

    def test_unk_counting(self):
        """UNK tokens should be counted correctly."""
        tok_name = "mock_tok"
        unk_id = 99
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, unk_id, 2, unk_id],  # 2 UNKs out of 4
            decode_fn=lambda ids: "test text",
            unk_id=unk_id,
        )
        provider = _MockDecodableProvider(tok_name, tok)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [_make_td(tok_name, "test text", [1, unk_id, 2, unk_id])]}
        results = metrics.compute_reconstruction_fidelity_analysis(td)

        summary = results["reconstruction_fidelity"]["summary"][tok_name]
        assert summary["unk_token_rate"] == pytest.approx(0.5)

    def test_no_unk_id_defined(self):
        """When no UNK ID is defined, UNK rate should be 0.0."""
        tok_name = "mock_tok"
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, 2, 3],
            decode_fn=lambda ids: "hello",
            unk_id=None,
        )
        provider = _MockDecodableProvider(tok_name, tok)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [_make_td(tok_name, "hello", [1, 2, 3])]}
        results = metrics.compute_reconstruction_fidelity_analysis(td)

        summary = results["reconstruction_fidelity"]["summary"][tok_name]
        assert summary["unk_token_rate"] == pytest.approx(0.0)

    def test_whitespace_preserved(self):
        """All whitespace preserved -> fidelity=1.0."""
        tok_name = "mock_tok"
        text = "a b\tc"
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, 2, 3],
            decode_fn=lambda ids: text,  # perfect decode
        )
        provider = _MockDecodableProvider(tok_name, tok)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [_make_td(tok_name, text, [1, 2, 3])]}
        results = metrics.compute_reconstruction_fidelity_analysis(td)

        summary = results["reconstruction_fidelity"]["summary"][tok_name]
        assert summary["whitespace_fidelity"] == pytest.approx(1.0)

    def test_non_decodable_tokenizer_skipped(self):
        """Non-decodable tokenizer should be silently skipped."""
        tok_name = "mock_tok"
        tok = _MockDecodableTokenizer(
            encode_fn=lambda t: [1, 2],
            decode_fn=None,  # can't decode
        )
        provider = _MockDecodableProvider(tok_name, tok)
        metrics = BasicTokenizationMetrics(provider)

        td = {tok_name: [_make_td(tok_name, "hello", [1, 2])]}
        results = metrics.compute_reconstruction_fidelity_analysis(td)

        assert tok_name not in results["reconstruction_fidelity"]["summary"]


# ======================================================================
# T9: _character_error_rate edge cases
# ======================================================================

class TestCharacterErrorRate:

    def test_identical_strings(self):
        assert BasicTokenizationMetrics._character_error_rate("abc", "abc") == pytest.approx(0.0)

    def test_single_char_missing(self):
        # "abc" vs "ac": Levenshtein distance 1 (1 deletion), CER = 1/3
        assert BasicTokenizationMetrics._character_error_rate("abc", "ac") == pytest.approx(1.0 / 3.0)

    def test_empty_reference(self):
        # Empty reference -> 0.0 (nothing to measure against)
        assert BasicTokenizationMetrics._character_error_rate("", "abc") == pytest.approx(0.0)

    def test_both_empty(self):
        assert BasicTokenizationMetrics._character_error_rate("", "") == pytest.approx(0.0)

    def test_empty_hypothesis(self):
        # "abc" -> "": 3 deletions / 3 chars -> 1.0
        assert BasicTokenizationMetrics._character_error_rate("abc", "") == pytest.approx(1.0)

    def test_hypothesis_longer_than_reference(self):
        # "ab" vs "aXbY": Levenshtein distance 2 (2 insertions), CER = 2/2 = 1.0
        assert BasicTokenizationMetrics._character_error_rate("ab", "aXbY") == pytest.approx(1.0)

    def test_always_non_negative(self):
        # CER is always >= 0.0 but can exceed 1.0
        assert BasicTokenizationMetrics._character_error_rate("a", "bcdefg") >= 0.0
        assert BasicTokenizationMetrics._character_error_rate("abcdef", "x") >= 0.0

    def test_cer_can_exceed_one(self):
        # "a" vs "abcde": Levenshtein distance 4 (4 insertions), CER = 4/1 = 4.0
        assert BasicTokenizationMetrics._character_error_rate("a", "abcde") == pytest.approx(4.0)

    def test_common_prefix_suffix(self):
        # Shared prefix "hello world, goodby" and suffix "!"; differ by 1 char
        # "hello world, goodbye!" vs "hello world, goodby!" -> distance 1, len 21
        assert BasicTokenizationMetrics._character_error_rate(
            "hello world, goodbye!", "hello world, goodby!"
        ) == pytest.approx(1.0 / 21.0)

    def test_differ_only_in_middle(self):
        # 100 A's + "XYZ" + 100 B's vs same with "X_Z" -> 1 substitution, len 203
        ref = "A" * 100 + "XYZ" + "B" * 100
        hyp = "A" * 100 + "X_Z" + "B" * 100
        assert BasicTokenizationMetrics._character_error_rate(ref, hyp) == pytest.approx(1.0 / 203.0)


# ======================================================================
# T10: Whitespace fidelity
# ======================================================================

class TestWhitespaceFidelity:

    def test_whitespace_stripped(self):
        """All whitespace stripped -> 0 preserved."""
        original = "a b c"
        decoded = "abc"
        preserved, total = BasicTokenizationMetrics._whitespace_fidelity(
            original, decoded
        )
        assert total == 2
        assert preserved == 0

    def test_partial_whitespace_loss(self):
        """One of two spaces lost -> 1/2 preserved."""
        original = "a b c"
        decoded = "ab c"  # first space lost
        preserved, total = BasicTokenizationMetrics._whitespace_fidelity(
            original, decoded
        )
        assert total == 2
        assert preserved == 1

    def test_no_whitespace(self):
        """Text with no whitespace -> (0, 0)."""
        original = "abc"
        decoded = "abc"
        preserved, total = BasicTokenizationMetrics._whitespace_fidelity(
            original, decoded
        )
        assert total == 0
        assert preserved == 0

    def test_unicode_zs_separators_count_as_whitespace(self):
        """whitespace_fidelity counts Unicode Zs separators (NBSP / thin /
        ideographic), not just ASCII.  NBSP->space is a real loss."""
        from tokenizer_analysis.metrics.basic import _is_ws
        for ch in (" ", "\t", "\n", "\r", " ", " ", "　"):
            assert _is_ws(ch), repr(ch)
        # NBSP folded to a regular space = the non-breaking property lost
        assert BasicTokenizationMetrics._whitespace_fidelity(
            "a b", "a b") == (0, 1)
        assert BasicTokenizationMetrics._whitespace_fidelity(
            "a　b", "a　b") == (1, 1)

    def test_zwsp_cf_excluded_from_whitespace(self):
        """ZWSP (U+200B, category Cf) is deliberately NOT whitespace -- its
        loss is captured by exact_match_rate / CER, not whitespace_fidelity."""
        from tokenizer_analysis.metrics.basic import _is_ws
        assert _is_ws("​") is False
        # ZWSP not counted -> total_ws stays 0 here
        assert BasicTokenizationMetrics._whitespace_fidelity(
            "a​b", "ab") == (0, 0)

    def test_reconstruction_metadata_self_describes_definition(self):
        """The widened definition is traceable in result metadata."""
        from tokenizer_analysis.metrics.basic import WHITESPACE_DEFINITION
        assert WHITESPACE_DEFINITION == "ascii(space,tab,nl,cr)+unicode_Zs"


# ======================================================================
# Vocabulary-utilization cross-language dispersion (per_language_std / cov)
# ======================================================================

class TestVocabUtilDispersion:

    def test_dispersion_zero_when_one_language(self):
        """Single language → SD == 0.0, CoV is None."""
        tok = "t"
        provider = _SimpleProvider(tok, vocab_size=100)
        metrics = BasicTokenizationMetrics(provider)
        td = {tok: [_make_td(tok, "x", [1, 2, 3, 4, 5], lang="eng_Latn")]}
        out = metrics.compute_vocabulary_utilization_analysis(td)
        per_tok = out["vocabulary_utilization"]["per_tokenizer"][tok]
        assert per_tok["per_language_std"] == 0.0
        assert per_tok["per_language_cov"] is None
        assert per_tok["per_language_mean"] == pytest.approx(0.05)  # 5/100

    def test_dispersion_known_value(self):
        """Two langs with utilizations [0.2, 0.4] → mean 0.3, sd≈0.1414, cov≈0.4714."""
        tok = "t"
        provider = _SimpleProvider(tok, vocab_size=10)
        metrics = BasicTokenizationMetrics(provider)
        # eng uses 2 unique tokens out of 10 -> 0.2; fra uses 4 unique out of 10 -> 0.4
        td = {tok: [
            _make_td(tok, "x", [1, 2], lang="eng_Latn"),
            _make_td(tok, "y", [3, 4, 5, 6], lang="fra_Latn"),
        ]}
        out = metrics.compute_vocabulary_utilization_analysis(td)
        per_tok = out["vocabulary_utilization"]["per_tokenizer"][tok]
        assert per_tok["per_language_mean"] == pytest.approx(0.3)
        # Sample SD with ddof=1 of [0.2, 0.4]:
        #   variance = ((0.2-0.3)^2 + (0.4-0.3)^2) / (2-1) = 0.02
        #   sd = sqrt(0.02) ≈ 0.14142135
        assert per_tok["per_language_std"] == pytest.approx(0.14142135, abs=1e-7)
        assert per_tok["per_language_cov"] == pytest.approx(0.14142135 / 0.3, abs=1e-7)

    def test_dispersion_uses_ratio_not_absolute_count(self):
        """Two tokenizers with the same per-language ratios but different vocab
        sizes must produce identical dispersion."""
        small_tok, big_tok = "small", "big"

        class TwoTokProvider(_SimpleProvider):
            def get_tokenizer_names(self):
                return [small_tok, big_tok]
            def get_vocab_size(self, name):
                return 10 if name == small_tok else 100
            def get_languages(self, tokenizer_name=None):
                return ["eng_Latn", "fra_Latn"]
        provider = TwoTokProvider("ignored")
        metrics = BasicTokenizationMetrics(provider)

        td = {
            # small_tok: 2/10 and 4/10  → ratios [0.2, 0.4]
            small_tok: [
                _make_td(small_tok, "x", [1, 2], lang="eng_Latn"),
                _make_td(small_tok, "y", [3, 4, 5, 6], lang="fra_Latn"),
            ],
            # big_tok: 20/100 and 40/100 → ratios [0.2, 0.4]
            big_tok: [
                _make_td(big_tok, "x", list(range(20)), lang="eng_Latn"),
                _make_td(big_tok, "y", list(range(20, 60)), lang="fra_Latn"),
            ],
        }
        out = metrics.compute_vocabulary_utilization_analysis(td)
        small = out["vocabulary_utilization"]["per_tokenizer"][small_tok]
        big = out["vocabulary_utilization"]["per_tokenizer"][big_tok]
        assert small["per_language_std"] == pytest.approx(big["per_language_std"], abs=1e-9)
        assert small["per_language_cov"] == pytest.approx(big["per_language_cov"], abs=1e-9)
