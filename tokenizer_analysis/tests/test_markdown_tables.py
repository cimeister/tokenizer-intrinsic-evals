"""Tests for markdown results table generation and parsing."""
import os
import textwrap
from pathlib import Path

import pytest

from tokenizer_analysis.visualization.markdown_tables import (
    MarkdownTableGenerator,
    _COMPOSITE_KEY_RE,
    _DISPLAY_NAME_RE,
)


# ── Regex tests (H6) ──────────────────────────────────────────────────


class TestCompositeKeyRegex:
    """_COMPOSITE_KEY_RE must handle names that contain parentheses."""

    def test_simple_name(self):
        m = _COMPOSITE_KEY_RE.match("Classical (meistecl, flores)")
        assert m is not None
        assert m.group(1).strip() == "Classical"
        assert m.group(2) == "meistecl"
        assert m.group(3) == "flores"

    def test_name_with_parens(self):
        m = _COMPOSITE_KEY_RE.match(
            "Gemma 3 (512 codebook) (saibo, flores)"
        )
        assert m is not None
        assert m.group(1).strip() == "Gemma 3 (512 codebook)"
        assert m.group(2) == "saibo"
        assert m.group(3) == "flores"

    def test_name_with_multiple_parens(self):
        m = _COMPOSITE_KEY_RE.match(
            "Foo (bar) (baz) (alice, dataset1)"
        )
        assert m is not None
        assert m.group(1).strip() == "Foo (bar) (baz)"
        assert m.group(2) == "alice"
        assert m.group(3) == "dataset1"

    def test_plain_name_no_match(self):
        assert _COMPOSITE_KEY_RE.match("plain_name") is None

    def test_brackets_no_match(self):
        assert _COMPOSITE_KEY_RE.match("Tok [128k]") is None


class TestVocabUtilCovColumn:
    """Cross-lingual vocab-utilization CoV is surfaced as a markdown column,
    reading the already-computed `per_language_cov` (None for <2-language
    tokenizers -> the standard '---' placeholder)."""

    @staticmethod
    def _results(cov_a, cov_b):
        return {
            'vocabulary_utilization': {
                'per_tokenizer': {
                    'A': {'global_utilization': 0.5,
                          'per_language_cov': cov_a},
                    'B': {'global_utilization': 0.5,
                          'per_language_cov': cov_b},
                },
                'metadata': {},
            }
        }

    def test_cov_column_formatted_for_multilang(self):
        md = MarkdownTableGenerator(
            self._results(0.12345, 0.4),
            ['A', 'B'],
        ).generate_markdown_table(metrics=['vocab_util_cross_lingual_cov'])
        assert 'Vocab Util. CoV' in md
        assert '0.123' in md          # {:.3f} of 0.12345
        assert '0.400' in md

    def test_cov_placeholder_for_single_language_none(self):
        # B is single-language -> per_language_cov is None -> '---'
        md = MarkdownTableGenerator(
            self._results(0.12345, None),
            ['A', 'B'],
        ).generate_markdown_table(metrics=['vocab_util_cross_lingual_cov'])
        assert '0.123' in md          # A still rendered
        assert '---' in md            # B rendered as the standard placeholder
        # Column not dropped (not all-None): the title is present.
        assert 'Vocab Util. CoV' in md


class TestDisplayNameRegex:
    """_DISPLAY_NAME_RE must match names with [Nk] suffix."""

    def test_standard(self):
        m = _DISPLAY_NAME_RE.match("Classical [128k]")
        assert m is not None
        assert m.group(1).strip() == "Classical"

    def test_name_with_parens(self):
        m = _DISPLAY_NAME_RE.match("Gemma 3 (512 codebook) [128k]")
        assert m is not None
        assert m.group(1).strip() == "Gemma 3 (512 codebook)"

    def test_no_bracket_no_match(self):
        assert _DISPLAY_NAME_RE.match("Classical") is None

    def test_composite_key_no_match(self):
        assert _DISPLAY_NAME_RE.match("Classical (alice, flores)") is None


# ── Parse round-trip tests (H1 / H2) ──────────────────────────────────


_SAMPLE_NEW_FORMAT = textwrap.dedent("""\
    # Tokenizer Evaluation Results

    _Last updated: 2025-06-01 12:00:00_

    | Tokenizer | Fertility ↓ | Dataset | User | Date |
    | --- | --- | --- | --- | --- |
    | Classical [128k] | 1.234 | flores | meistecl | 2025-06-01 |
    | Gemma 3 (512 codebook) [256k] | 0.987 | flores | saibo | 2025-06-01 |
""")


class TestParseRoundTrip:
    """Parsing new-format markdown must reconstruct composite keys and
    preserve the display name in the ``Tokenizer`` column of each row_map.
    """

    def test_composite_key_reconstruction(self, tmp_path):
        md_file = tmp_path / "RESULTS_flores.md"
        md_file.write_text(_SAMPLE_NEW_FORMAT)
        headers, rows = MarkdownTableGenerator.parse_existing_markdown(
            str(md_file)
        )
        assert "Tokenizer" in headers
        # Composite keys should be reconstructed
        assert "Classical (meistecl, flores)" in rows
        assert "Gemma 3 (512 codebook) (saibo, flores)" in rows

    def test_display_name_preserved(self, tmp_path):
        md_file = tmp_path / "RESULTS_flores.md"
        md_file.write_text(_SAMPLE_NEW_FORMAT)
        _, rows = MarkdownTableGenerator.parse_existing_markdown(str(md_file))
        row = rows["Classical (meistecl, flores)"]
        assert row["Tokenizer"] == "Classical [128k]"

    def test_display_name_with_parens_preserved(self, tmp_path):
        md_file = tmp_path / "RESULTS_flores.md"
        md_file.write_text(_SAMPLE_NEW_FORMAT)
        _, rows = MarkdownTableGenerator.parse_existing_markdown(str(md_file))
        row = rows["Gemma 3 (512 codebook) (saibo, flores)"]
        assert row["Tokenizer"] == "Gemma 3 (512 codebook) [256k]"
