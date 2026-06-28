"""Tests for UTF8IntegrityMetrics."""

import pytest

from tokenizer_analysis.metrics.utf8_integrity import (
    UTF8IntegrityMetrics,
    _GPT2_UNICODE_TO_BYTE,
    _GPT2_BYTE_TO_UNICODE,
    _GPT2_MARKER_CHARS,
)
from tokenizer_analysis.core.input_types import TokenizedData
from .conftest import MockTokenizer, MockProvider


# ---------------------------------------------------------------
# Helper: bare instance for static-method tests
# ---------------------------------------------------------------

@pytest.fixture
def inst():
    """Bare UTF8IntegrityMetrics instance (no InputProvider needed for statics)."""
    return object.__new__(UTF8IntegrityMetrics)


# ---------------------------------------------------------------
# GPT-2-style mock tokenizer with get_vocab()
# ---------------------------------------------------------------

class GPT2MockTokenizer:
    """Mock tokenizer that exposes a GPT-2-style vocabulary."""

    def __init__(self, id_to_token: dict):
        self._map = id_to_token
        # Build a vocab dict containing GPT-2 marker chars to trigger detection
        self._vocab = {}
        # Add the id_to_token entries
        for tid, tok_str in id_to_token.items():
            self._vocab[tok_str] = tid
        # Add enough single-char GPT-2 marker tokens to pass the threshold
        base_id = max(id_to_token.keys()) + 1 if id_to_token else 0
        for i, ch in enumerate(sorted(_GPT2_MARKER_CHARS)):
            self._vocab[ch] = base_id + i
            if i >= 60:  # Well above the threshold of 50
                break

    def convert_ids_to_tokens(self, ids):
        return [self._map[i] for i in ids]

    def get_vocab(self):
        return self._vocab


# ===================================================================
# GPT-2 byte tables sanity checks
# ===================================================================

class TestGPT2ByteTables:

    def test_table_covers_all_256_bytes(self):
        assert len(_GPT2_BYTE_TO_UNICODE) == 256
        assert len(_GPT2_UNICODE_TO_BYTE) == 256

    def test_roundtrip(self):
        """Every byte -> unicode -> byte roundtrips."""
        for b in range(256):
            ch = _GPT2_BYTE_TO_UNICODE[b]
            assert _GPT2_UNICODE_TO_BYTE[ch] == b

    def test_space_maps_to_g(self):
        """Byte 0x20 (space) maps to Ġ (U+0120)."""
        assert _GPT2_BYTE_TO_UNICODE[0x20] == '\u0120'
        assert _GPT2_UNICODE_TO_BYTE['\u0120'] == 0x20

    def test_printable_ascii_maps_to_self(self):
        """Printable ASCII chars (0x21-0x7E) map to themselves."""
        for b in range(0x21, 0x7F):
            assert _GPT2_BYTE_TO_UNICODE[b] == chr(b)

    def test_marker_chars_above_0x100(self):
        """Marker chars for non-printable bytes are at U+0100+."""
        assert len(_GPT2_MARKER_CHARS) > 0
        for ch in _GPT2_MARKER_CHARS:
            assert ord(ch) >= 0x100


# ===================================================================
# _token_string_to_bytes — non-GPT-2 path
# ===================================================================

class TestTokenStringToBytes:

    def test_plain_ascii(self, inst):
        assert inst._token_string_to_bytes("hello") == b"hello"

    def test_g_space_prefix(self, inst):
        # Ġ represents a space, NOT its own UTF-8 encoding
        result = inst._token_string_to_bytes("\u0120hello")
        assert result == b" hello"

    def test_sentencepiece_space_prefix(self, inst):
        result = inst._token_string_to_bytes("\u2581hello")
        assert result == b" hello"

    def test_byte_fallback_upper(self, inst):
        assert inst._token_string_to_bytes("<0xC3>") == bytes([0xC3])

    def test_byte_fallback_lower(self, inst):
        assert inst._token_string_to_bytes("<0xc3>") == bytes([0xC3])

    def test_bert_continuation(self, inst):
        assert inst._token_string_to_bytes("##ing") == b"ing"

    def test_special_token(self, inst):
        assert inst._token_string_to_bytes("<|endoftext|>") is None

    def test_special_token_brackets(self, inst):
        assert inst._token_string_to_bytes("[CLS]") is None

    def test_placeholder_unk(self, inst):
        assert inst._token_string_to_bytes("<UNK_42>") is None

    def test_placeholder_token(self, inst):
        assert inst._token_string_to_bytes("<TOKEN_99>") is None

    def test_multibyte_char(self, inst):
        # café contains é (U+00E9) -> C3 A9
        assert inst._token_string_to_bytes("café") == b"caf\xc3\xa9"

    def test_cjk_char(self, inst):
        # 的 (U+7684) -> E7 9A 84
        assert inst._token_string_to_bytes("的") == b"\xe7\x9a\x84"

    def test_end_word_suffix(self, inst):
        assert inst._token_string_to_bytes("word</w>") == b"word"

    def test_bpe_continuation_suffix(self, inst):
        assert inst._token_string_to_bytes("sub@@") == b"sub"

    def test_markers_are_mutually_exclusive(self, inst):
        """Marker stripping should use early returns — only one marker type applied."""
        # A token starting with ▁ should NOT also strip ## or @@
        result = inst._token_string_to_bytes("\u2581##foo")
        # ▁ prefix matched first → " ##foo"
        assert result == b" ##foo"

        result = inst._token_string_to_bytes("##foo@@")
        # ## prefix matched first → "foo@@"
        assert result == b"foo@@"


# ===================================================================
# _token_string_to_bytes — GPT-2 path
# ===================================================================

class TestTokenStringToBytesGPT2:

    def test_gpt2_space_via_g(self, inst):
        """Ġ in GPT-2 mode maps to byte 0x20 (space), not C4 A0."""
        result = inst._token_string_to_bytes("\u0120hello", _GPT2_UNICODE_TO_BYTE)
        assert result == b" hello"

    def test_gpt2_plain_ascii(self, inst):
        """Printable ASCII maps to itself in GPT-2 mode."""
        result = inst._token_string_to_bytes("hello", _GPT2_UNICODE_TO_BYTE)
        assert result == b"hello"

    def test_gpt2_c3_char(self, inst):
        """In GPT-2 mode, the Unicode char representing byte 0xC3 should
        map to the single byte 0xC3, not the 2-byte UTF-8 of Ã."""
        # Ã is U+00C3, which in GPT-2 represents byte 0xC3
        gpt2_c3 = _GPT2_BYTE_TO_UNICODE[0xC3]
        result = inst._token_string_to_bytes(gpt2_c3, _GPT2_UNICODE_TO_BYTE)
        assert result == bytes([0xC3])

    def test_gpt2_split_e_accent(self, inst):
        """GPT-2 token for byte 0xC3 followed by token for byte 0xA9
        should each produce a single byte (together forming é)."""
        ch_c3 = _GPT2_BYTE_TO_UNICODE[0xC3]
        ch_a9 = _GPT2_BYTE_TO_UNICODE[0xA9]
        assert inst._token_string_to_bytes(ch_c3, _GPT2_UNICODE_TO_BYTE) == bytes([0xC3])
        assert inst._token_string_to_bytes(ch_a9, _GPT2_UNICODE_TO_BYTE) == bytes([0xA9])

    def test_gpt2_multibyte_intact(self, inst):
        """A GPT-2 token like 'café' should produce the correct raw bytes."""
        # In GPT-2 encoding, 'c', 'a', 'f' map to themselves (printable ASCII),
        # 'é' (U+00E9) maps to byte 0xE9 — but é is actually 2 UTF-8 bytes.
        # In GPT-2 vocab, 'café' would be represented as 'cafÃ©' where
        # Ã = byte 0xC3, © = byte 0xA9 (since é = C3 A9 in UTF-8).
        ch_c3 = _GPT2_BYTE_TO_UNICODE[0xC3]
        ch_a9 = _GPT2_BYTE_TO_UNICODE[0xA9]
        token_str = f"caf{ch_c3}{ch_a9}"
        result = inst._token_string_to_bytes(token_str, _GPT2_UNICODE_TO_BYTE)
        assert result == b"caf\xc3\xa9"

    def test_gpt2_byte_fallback_still_works(self, inst):
        """Byte-fallback tokens should still work even with GPT-2 table."""
        result = inst._token_string_to_bytes("<0xC3>", _GPT2_UNICODE_TO_BYTE)
        assert result == bytes([0xC3])

    def test_gpt2_special_still_skipped(self, inst):
        """Special tokens return None even in GPT-2 mode."""
        assert inst._token_string_to_bytes("<|endoftext|>", _GPT2_UNICODE_TO_BYTE) is None


# ===================================================================
# _is_valid_complete_utf8
# ===================================================================

class TestIsValidCompleteUTF8:

    def test_valid_ascii(self, inst):
        assert inst._is_valid_complete_utf8(b"hello") is True

    def test_valid_2byte(self, inst):
        # é = C3 A9
        assert inst._is_valid_complete_utf8(b"\xc3\xa9") is True

    def test_valid_3byte(self, inst):
        # 的 = E7 9A 84
        assert inst._is_valid_complete_utf8(b"\xe7\x9a\x84") is True

    def test_valid_4byte(self, inst):
        # 🎉 = F0 9F 8E 89
        assert inst._is_valid_complete_utf8(b"\xf0\x9f\x8e\x89") is True

    def test_orphan_continuation_byte(self, inst):
        # A9 alone is an orphan continuation byte
        assert inst._is_valid_complete_utf8(b"\xa9") is False

    def test_truncated_leading_byte(self, inst):
        # C3 alone expects a continuation byte
        assert inst._is_valid_complete_utf8(b"\xc3") is False

    def test_mixed_valid_plus_truncated(self, inst):
        # 'a' followed by truncated C3
        assert inst._is_valid_complete_utf8(b"a\xc3") is False

    def test_empty_bytes(self, inst):
        assert inst._is_valid_complete_utf8(b"") is True


# ===================================================================
# _classify_malformation
# ===================================================================

class TestClassifyMalformation:

    def test_valid_utf8_returns_none(self, inst):
        assert inst._classify_malformation(b"hello") is None
        assert inst._classify_malformation(b"\xc3\xa9") is None  # é
        assert inst._classify_malformation(b"") is None

    def test_trailing_incomplete_2byte(self, inst):
        # C3 alone: leading byte for 2-byte char, missing continuation
        assert inst._classify_malformation(b"\xc3") == 'trailing_incomplete'

    def test_trailing_incomplete_3byte(self, inst):
        # E7 9A: leading byte for 3-byte char, only 1 continuation
        assert inst._classify_malformation(b"\xe7\x9a") == 'trailing_incomplete'

    def test_trailing_incomplete_4byte(self, inst):
        # F0 9F: leading byte for 4-byte char, only 1 continuation
        assert inst._classify_malformation(b"\xf0\x9f") == 'trailing_incomplete'

    def test_orphan_continuation(self, inst):
        # A9 alone: continuation byte without leading byte
        assert inst._classify_malformation(b"\xa9") == 'orphan_continuation'

    def test_orphan_continuation_after_ascii(self, inst):
        # 'a' then orphan continuation
        assert inst._classify_malformation(b"a\xa9") == 'orphan_continuation'

    def test_trailing_after_valid_char(self, inst):
        # Valid é (C3 A9) followed by truncated C3
        assert inst._classify_malformation(b"\xc3\xa9\xc3") == 'trailing_incomplete'

    def test_leader_followed_by_non_continuation(self, inst):
        # E7 followed by 0x41 ('A') — not a continuation byte.
        # The leading byte has a follower, but it's invalid — this is
        # orphan_continuation (the non-continuation byte is orphaned
        # from any valid sequence), NOT trailing_incomplete.
        assert inst._classify_malformation(b"\xe7\x41") == 'orphan_continuation'

    def test_4byte_leader_followed_by_non_continuation(self, inst):
        # F0 9F followed by 0x41 — 4-byte leader with valid first
        # continuation but invalid second continuation.
        assert inst._classify_malformation(b"\xf0\x9f\x41") == 'orphan_continuation'

    def test_2byte_leader_followed_by_ascii(self, inst):
        # C3 followed by 'a' (0x61) — not a continuation byte.
        assert inst._classify_malformation(b"\xc3\x61") == 'orphan_continuation'


# ===================================================================
# _crosses_character_boundary
# ===================================================================

class TestCrossesCharacterBoundary:

    def test_single_byte_never_crosses(self, inst):
        assert inst._crosses_character_boundary(b"\xc3") is False
        assert inst._crosses_character_boundary(b"\xa9") is False
        assert inst._crosses_character_boundary(b"a") is False

    def test_empty_never_crosses(self, inst):
        assert inst._crosses_character_boundary(b"") is False

    def test_complete_ascii_no_crossing(self, inst):
        assert inst._crosses_character_boundary(b"hello") is False

    def test_complete_multibyte_no_crossing(self, inst):
        # Complete é (C3 A9) — one char, complete
        assert inst._crosses_character_boundary(b"\xc3\xa9") is False

    def test_multiple_complete_chars_no_crossing(self, inst):
        # "aé" — two complete chars
        assert inst._crosses_character_boundary(b"a\xc3\xa9") is False

    def test_continuation_then_leader_crosses(self, inst):
        # A9 (tail of é) + E4 (lead of 你) — two chars, both incomplete
        assert inst._crosses_character_boundary(b"\xa9\xe4") is True

    def test_complete_char_plus_orphan_crosses(self, inst):
        # "a" + A9 (orphan continuation) — two chars, second is incomplete
        assert inst._crosses_character_boundary(b"a\xa9") is True

    def test_leader_plus_non_continuation_crosses(self, inst):
        # C3 (lead of 2-byte) + 41 ('A') — two chars, first is incomplete
        assert inst._crosses_character_boundary(b"\xc3\x41") is True

    def test_full_boundary_crossing_example(self, inst):
        # A9 E4 — tail of é merged with lead of CJK char
        assert inst._crosses_character_boundary(b"\xa9\xe4") is True

    def test_two_orphan_continuations_crosses(self, inst):
        # A9 BD — two orphan continuation bytes, each an incomplete char
        assert inst._crosses_character_boundary(b"\xa9\xbd") is True

    def test_byte_fallback_single_does_not_cross(self, inst):
        # Single-byte tokens from byte fallback never cross
        for b in [0xC3, 0xA9, 0xE4, 0xBD, 0xA0]:
            assert inst._crosses_character_boundary(bytes([b])) is False


# ===================================================================
# _align_byte_sequences
# ===================================================================

class TestAlignByteSequences:

    def test_identical(self, inst):
        data = b"hello"
        mapping, mismatches = inst._align_byte_sequences(data, data)
        assert mapping == [0, 1, 2, 3, 4]
        assert mismatches == 0

    def test_small_insertion_in_reconstructed(self, inst):
        # Reconstructed has an extra byte at position 2
        source = b"abcd"
        recon = b"abXcd"
        mapping, mismatches = inst._align_byte_sequences(source, recon)
        assert mapping[0] == 0  # a
        assert mapping[1] == 1  # b
        # c and d should still align via lookahead
        assert mapping[2] == 3  # c
        assert mapping[3] == 4  # d
        assert mismatches == 0

    def test_extra_source_byte(self, inst):
        """Source has an extra byte (reverse case) — should skip and continue."""
        source = b"abXcd"
        recon = b"abcd"
        mapping, mismatches = inst._align_byte_sequences(source, recon)
        assert mapping[0] == 0  # a
        assert mapping[1] == 1  # b
        assert mapping[2] is None  # X has no match
        assert mapping[3] == 2  # c
        assert mapping[4] == 3  # d
        assert mismatches == 1

    def test_empty_sequences(self, inst):
        mapping, mismatches = inst._align_byte_sequences(b"", b"")
        assert mapping == []
        assert mismatches == 0


# ===================================================================
# _count_split_characters
# ===================================================================

class TestCountSplitCharacters:

    def test_all_ascii(self, inst):
        source = b"abc"
        mapping = [0, 1, 2]
        byte_to_token = [0, 0, 0]
        splits, mb, total, per_width, unaligned = inst._count_split_characters(
            source, mapping, byte_to_token
        )
        assert splits == 0
        assert mb == 0
        assert total == 3
        assert per_width == {2: (0, 0), 3: (0, 0), 4: (0, 0)}

    def test_multibyte_intact(self, inst):
        # é = C3 A9, both bytes in token 0
        source = b"\xc3\xa9"
        mapping = [0, 1]
        byte_to_token = [0, 0]
        splits, mb, total, per_width, unaligned = inst._count_split_characters(
            source, mapping, byte_to_token
        )
        assert splits == 0
        assert mb == 1
        assert total == 1
        assert per_width[2] == (0, 1)  # 1 two-byte char, 0 splits

    def test_multibyte_split_across_tokens(self, inst):
        # é = C3 A9, C3 in token 0, A9 in token 1
        source = b"\xc3\xa9"
        mapping = [0, 1]
        byte_to_token = [0, 1]
        splits, mb, total, per_width, unaligned = inst._count_split_characters(
            source, mapping, byte_to_token
        )
        assert splits == 1
        assert mb == 1
        assert total == 1
        assert per_width[2] == (1, 1)  # 1 two-byte char, 1 split

    def test_3byte_split(self, inst):
        # 你 = E4 BD A0 — split across 3 tokens
        source = b"\xe4\xbd\xa0"
        mapping = [0, 1, 2]
        byte_to_token = [0, 1, 2]
        splits, mb, total, per_width, unaligned = inst._count_split_characters(
            source, mapping, byte_to_token
        )
        assert splits == 1
        assert per_width[3] == (1, 1)

    def test_4byte_intact(self, inst):
        # 🎉 = F0 9F 8E 89, all in token 0
        source = b"\xf0\x9f\x8e\x89"
        mapping = [0, 1, 2, 3]
        byte_to_token = [0, 0, 0, 0]
        splits, mb, total, per_width, unaligned = inst._count_split_characters(
            source, mapping, byte_to_token
        )
        assert splits == 0
        assert per_width[4] == (0, 1)

    def test_partially_unmapped_char_excluded(self, inst):
        """A char with some bytes unaligned cannot be classified split-or-not,
        so it is excluded from both numerator and denominator and counted as
        unaligned. (Previously it was counted as a split, inflating the rate.)"""
        # é = C3 A9: first byte mapped (token 0), second byte unaligned
        source = b"\xc3\xa9"
        mapping = [0, None]
        byte_to_token = [0]
        splits, mb, total, per_width, unaligned = inst._count_split_characters(
            source, mapping, byte_to_token
        )
        assert splits == 0
        assert mb == 0
        assert unaligned == 1
        assert per_width[2] == (0, 0)

    def test_fully_unmapped_char_excluded(self, inst):
        """A char with all bytes unaligned is excluded from both numerator and
        denominator. (Previously it stayed in the denominator, deflating the
        rate.)"""
        source = b"\xc3\xa9"
        mapping = [None, None]
        byte_to_token = []
        splits, mb, total, per_width, unaligned = inst._count_split_characters(
            source, mapping, byte_to_token
        )
        assert splits == 0
        assert mb == 0
        assert unaligned == 1


# ===================================================================
# GPT-2 detection
# ===================================================================

class TestGPT2Detection:

    def test_detects_gpt2_tokenizer(self):
        """Tokenizer with GPT-2 marker chars in vocab is detected."""
        tok = GPT2MockTokenizer({0: "hello"})
        prov = MockProvider("gpt2_tok", tok)
        metrics = UTF8IntegrityMetrics(prov)
        result = metrics._detect_gpt2_encoding(tok)
        assert result is not None
        assert result is _GPT2_UNICODE_TO_BYTE

    def test_non_gpt2_tokenizer(self):
        """Regular tokenizer without GPT-2 markers is not detected."""
        tok = MockTokenizer({0: "hello", 1: "world"})
        prov = MockProvider("reg_tok", tok)
        metrics = UTF8IntegrityMetrics(prov)
        result = metrics._detect_gpt2_encoding(tok)
        assert result is None

    def test_detection_is_cached(self):
        """Repeated calls return cached result."""
        tok = GPT2MockTokenizer({0: "hello"})
        prov = MockProvider("gpt2_tok", tok)
        metrics = UTF8IntegrityMetrics(prov)
        r1 = metrics._detect_gpt2_encoding(tok)
        r2 = metrics._detect_gpt2_encoding(tok)
        assert r1 is r2


# ===================================================================
# End-to-end compute() tests
# ===================================================================

class TestComputeEndToEnd:

    def _make_data(self, tok_name, text, token_ids, lang="en"):
        return {
            tok_name: [
                TokenizedData(
                    tokens=token_ids,
                    text=text,
                    language=lang,
                    tokenizer_name=tok_name,
                )
            ]
        }

    def test_good_tokenizer_intact_chars(self):
        """Tokenizer keeps multi-byte chars intact -> integrity=1, splits=0."""
        # "café" tokenized as ["ca", "fé"]
        tok = MockTokenizer({0: "ca", 1: "fé"})
        prov = MockProvider("good_tok", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = self._make_data("good_tok", "café", [0, 1])
        results = metrics.compute(data)

        integrity = results['utf8_token_integrity']['summary']['good_tok']
        assert integrity['completeness_rate'] == 1.0
        assert integrity['total_incomplete_tokens'] == 0

        char_split = results['utf8_char_split']['summary']['good_tok']
        assert char_split['total_splits'] == 0

    def test_bad_tokenizer_byte_fallback(self):
        """Tokenizer splits é into byte-fallback tokens -> integrity<1, splits>0."""
        # "café" tokenized as ["caf", "<0xC3>", "<0xA9>"]
        tok = MockTokenizer({0: "caf", 1: "<0xC3>", 2: "<0xA9>"})
        prov = MockProvider("bad_tok", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = self._make_data("bad_tok", "café", [0, 1, 2])
        results = metrics.compute(data)

        integrity = results['utf8_token_integrity']['summary']['bad_tok']
        # Token "caf" is valid, "<0xC3>" alone is invalid, "<0xA9>" alone is invalid
        assert integrity['completeness_rate'] < 1.0
        assert integrity['total_incomplete_tokens'] == 2

        char_split = results['utf8_char_split']['summary']['bad_tok']
        assert char_split['total_splits'] > 0

    def test_bad_tokenizer_malformation_subtypes(self):
        """Verify malformation sub-types are reported."""
        tok = MockTokenizer({0: "caf", 1: "<0xC3>", 2: "<0xA9>"})
        prov = MockProvider("bad_tok", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = self._make_data("bad_tok", "café", [0, 1, 2])
        results = metrics.compute(data)

        integrity = results['utf8_token_integrity']['summary']['bad_tok']
        # <0xC3> = byte 0xC3 alone = trailing_incomplete (leading byte missing continuation)
        # <0xA9> = byte 0xA9 alone = orphan_continuation (continuation byte without leader)
        assert integrity['trailing_incomplete'] == 1
        assert integrity['orphan_continuation'] == 1

    def test_byte_fallback_no_boundary_crossings(self):
        """Single-byte fallback tokens are incomplete but do NOT cross boundaries."""
        tok = MockTokenizer({0: "caf", 1: "<0xC3>", 2: "<0xA9>"})
        prov = MockProvider("bf_tok", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = self._make_data("bf_tok", "café", [0, 1, 2])
        results = metrics.compute(data)

        integrity = results['utf8_token_integrity']['summary']['bf_tok']
        assert integrity['total_incomplete_tokens'] == 2
        # Single-byte tokens don't cross boundaries
        assert integrity['boundary_crossings'] == 0
        assert integrity['boundary_crossing_rate'] == 0.0

    def test_boundary_crossing_token(self):
        """A token whose bytes span two incomplete chars crosses a boundary."""
        # Token 0: bytes A9 E4 — tail of é (A9) + lead of 你 (E4)
        # This is a BPE merge across a character boundary.
        # We need a non-GPT-2 tokenizer where _token_string_to_bytes
        # produces these raw bytes. Use byte-fallback-style but as a
        # 2-char token won't work since it would UTF-8-encode.
        # Instead: construct via GPT-2 path where we control exact bytes.
        ch_a9 = _GPT2_BYTE_TO_UNICODE[0xA9]
        ch_e4 = _GPT2_BYTE_TO_UNICODE[0xE4]

        tok = GPT2MockTokenizer({
            0: f"{ch_a9}{ch_e4}",  # bytes: A9 E4 — crosses boundary
        })
        prov = MockProvider("cross_tok", tok)
        metrics = UTF8IntegrityMetrics(prov)

        # Source text doesn't matter much for this test, but provide something
        data = self._make_data("cross_tok", "x", [0])
        results = metrics.compute(data)

        integrity = results['utf8_token_integrity']['summary']['cross_tok']
        assert integrity['boundary_crossings'] == 1
        assert integrity['boundary_crossing_rate'] > 0.0

    def test_ascii_only_text(self):
        """ASCII-only text -> integrity=1.0, splits=0 for any tokenizer."""
        tok = MockTokenizer({0: "hello", 1: "\u0120world"})
        prov = MockProvider("ascii_tok", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = self._make_data("ascii_tok", "hello world", [0, 1])
        results = metrics.compute(data)

        integrity = results['utf8_token_integrity']['summary']['ascii_tok']
        assert integrity['completeness_rate'] == 1.0

        char_split = results['utf8_char_split']['summary']['ascii_tok']
        assert char_split['total_splits'] == 0
        assert char_split['total_multibyte_chars'] == 0

    def test_cjk_intact(self):
        """CJK chars kept intact should have perfect scores."""
        # 你好 = two 3-byte chars
        tok = MockTokenizer({0: "你", 1: "好"})
        prov = MockProvider("cjk_tok", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = self._make_data("cjk_tok", "你好", [0, 1])
        results = metrics.compute(data)

        integrity = results['utf8_token_integrity']['summary']['cjk_tok']
        assert integrity['completeness_rate'] == 1.0

        char_split = results['utf8_char_split']['summary']['cjk_tok']
        assert char_split['total_splits'] == 0

    def test_cjk_split(self):
        """CJK char split across byte-fallback tokens."""
        # 你 = E4 BD A0 split into 3 byte tokens
        tok = MockTokenizer({0: "<0xE4>", 1: "<0xBD>", 2: "<0xA0>"})
        prov = MockProvider("cjk_bad", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = self._make_data("cjk_bad", "你", [0, 1, 2])
        results = metrics.compute(data)

        integrity = results['utf8_token_integrity']['summary']['cjk_bad']
        assert integrity['completeness_rate'] < 1.0

        char_split = results['utf8_char_split']['summary']['cjk_bad']
        assert char_split['total_splits'] == 1

    def test_cjk_split_byte_width_stratification(self):
        """Verify byte-width stratification for CJK split."""
        tok = MockTokenizer({0: "<0xE4>", 1: "<0xBD>", 2: "<0xA0>"})
        prov = MockProvider("cjk_bad", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = self._make_data("cjk_bad", "你", [0, 1, 2])
        results = metrics.compute(data)

        pw = results['utf8_char_split']['summary']['cjk_bad']['per_byte_width']
        assert pw['3_byte']['splits'] == 1
        assert pw['3_byte']['total'] == 1
        assert pw['2_byte']['splits'] == 0
        assert pw['4_byte']['splits'] == 0

    def test_gpt2_tokenizer_detects_invalid(self):
        """GPT-2-style tokenizer: byte-split é is correctly detected as invalid."""
        # In GPT-2 encoding, byte 0xC3 maps to Ã (U+00C3),
        # byte 0xA9 maps to © (U+00A9)
        ch_c3 = _GPT2_BYTE_TO_UNICODE[0xC3]
        ch_a9 = _GPT2_BYTE_TO_UNICODE[0xA9]

        tok = GPT2MockTokenizer({
            0: "caf",
            1: ch_c3,   # represents byte 0xC3
            2: ch_a9,   # represents byte 0xA9
        })
        prov = MockProvider("gpt2_tok", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = self._make_data("gpt2_tok", "café", [0, 1, 2])
        results = metrics.compute(data)

        integrity = results['utf8_token_integrity']['summary']['gpt2_tok']
        # Without GPT-2 detection, "Ã" and "©" would each appear as valid UTF-8.
        # WITH GPT-2 detection, they map to bytes 0xC3 and 0xA9 respectively,
        # which are individually invalid UTF-8.
        assert integrity['completeness_rate'] < 1.0
        assert integrity['total_incomplete_tokens'] == 2

        char_split = results['utf8_char_split']['summary']['gpt2_tok']
        assert char_split['total_splits'] == 1

    def test_gpt2_tokenizer_intact_multibyte(self):
        """GPT-2-style tokenizer with intact multi-byte chars -> valid."""
        ch_c3 = _GPT2_BYTE_TO_UNICODE[0xC3]
        ch_a9 = _GPT2_BYTE_TO_UNICODE[0xA9]

        # Token "café" in GPT-2 encoding = "caf" + Ã + ©
        tok = GPT2MockTokenizer({
            0: f"caf{ch_c3}{ch_a9}",  # entire word in one token
        })
        prov = MockProvider("gpt2_good", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = self._make_data("gpt2_good", "café", [0])
        results = metrics.compute(data)

        integrity = results['utf8_token_integrity']['summary']['gpt2_good']
        assert integrity['completeness_rate'] == 1.0
        assert integrity['total_incomplete_tokens'] == 0

        char_split = results['utf8_char_split']['summary']['gpt2_good']
        assert char_split['total_splits'] == 0

    def test_multiple_languages(self):
        """Verify per-language breakdown in results."""
        tok = MockTokenizer({0: "hello", 1: "café"})
        prov = MockProvider("multi_tok", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = {
            "multi_tok": [
                TokenizedData(
                    tokens=[0], text="hello", language="en",
                    tokenizer_name="multi_tok"
                ),
                TokenizedData(
                    tokens=[1], text="café", language="fr",
                    tokenizer_name="multi_tok"
                ),
            ]
        }
        results = metrics.compute(data)

        per_lang = results['utf8_token_integrity']['per_tokenizer']['multi_tok']['per_language']
        assert 'en' in per_lang
        assert 'fr' in per_lang

        summary = results['utf8_token_integrity']['summary']['multi_tok']
        assert summary['languages_analyzed'] == 2

    def test_no_text_field(self):
        """When text is None, char split metric should be empty but integrity works."""
        tok = MockTokenizer({0: "hello", 1: "<0xC3>"})
        prov = MockProvider("notxt", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = {
            "notxt": [
                TokenizedData(
                    tokens=[0, 1], text=None, language="en",
                    tokenizer_name="notxt"
                ),
            ]
        }
        results = metrics.compute(data)

        # Integrity should still work
        integrity = results['utf8_token_integrity']['summary']['notxt']
        assert integrity['total_content_tokens'] == 2

        # Char split should have no data (text was None)
        char_split = results['utf8_char_split']['summary']['notxt']
        assert char_split['total_splits'] == 0
        assert char_split['total_multibyte_chars'] == 0

    def test_byte_width_stratification_mixed(self):
        """Mixed 2-byte and 3-byte chars, only 3-byte split."""
        # é = C3 A9 (2-byte, intact in token 0)
        # 你 = E4 BD A0 (3-byte, split across tokens 1,2,3)
        tok = MockTokenizer({
            0: "é",         # intact 2-byte
            1: "<0xE4>",    # 3-byte split
            2: "<0xBD>",
            3: "<0xA0>",
        })
        prov = MockProvider("mix_tok", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = self._make_data("mix_tok", "é你", [0, 1, 2, 3])
        results = metrics.compute(data)

        pw = results['utf8_char_split']['summary']['mix_tok']['per_byte_width']
        assert pw['2_byte']['total'] == 1
        assert pw['2_byte']['splits'] == 0
        assert pw['3_byte']['total'] == 1
        assert pw['3_byte']['splits'] == 1

    def test_alignment_failure_not_counted_as_split(self):
        """A multi-byte char the tokenizer fails to reproduce is excluded as
        unaligned, not counted as a split. With no aligned multi-byte char,
        split_rate is None rather than a fabricated value. Regression for the
        alignment-failure-as-split bug."""
        # Source "café"; tokenizer reconstructs only "caf" (é is dropped),
        # so é's two bytes do not align.
        tok = MockTokenizer({0: "caf"})
        prov = MockProvider("drop_tok", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = self._make_data("drop_tok", "café", [0])
        res = metrics.compute(data)['utf8_char_split']

        g = res['per_tokenizer']['drop_tok']['global']
        assert g['total_splits'] == 0
        assert g['total_multibyte_chars'] == 0        # é excluded from denominator
        assert g['unaligned_multibyte_chars'] == 1
        assert g['split_rate'] is None                # no aligned multi-byte char
        assert g['aligned_fraction'] == 0.0
        assert g['alignment_mismatches'] == 2         # é's two unaligned bytes

    def test_split_rate_uses_only_aligned_denominator(self):
        """When one char aligns (and is split) and another fails to align,
        split_rate is computed over the aligned char only, and the unaligned
        char is reported separately."""
        # 你 (E4 BD A0) split across 3 byte-fallback tokens -> aligned, split.
        # é is in the source but not reproduced by the tokens -> unaligned.
        tok = MockTokenizer({0: "<0xE4>", 1: "<0xBD>", 2: "<0xA0>"})
        prov = MockProvider("mix_align", tok)
        metrics = UTF8IntegrityMetrics(prov)

        data = self._make_data("mix_align", "你é", [0, 1, 2])
        g = metrics.compute(data)['utf8_char_split']['per_tokenizer']['mix_align']['global']

        assert g['total_splits'] == 1
        assert g['total_multibyte_chars'] == 1        # only 你 aligned
        assert g['unaligned_multibyte_chars'] == 1    # é
        assert g['split_rate'] == 1.0
        assert g['aligned_fraction'] == 0.5
        assert g['alignment_mismatches'] == 2         # é's two bytes
