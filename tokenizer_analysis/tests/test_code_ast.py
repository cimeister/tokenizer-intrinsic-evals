"""Tests for tokenizer_analysis.metrics.code_ast (ASTBoundaryMetrics)."""

import pytest

from tokenizer_analysis.metrics.code_ast import (
    ASTBoundaryMetrics,
    _IDENTIFIER_TYPES,
    _LITERAL_TYPES,
    _DELIMITER_CHARS,
    _KNOWN_KEYWORDS,
    _NON_OPERATOR_PUNCTUATION,
    _WHITESPACE_SIGNIFICANT_LANGS,
)
# _WHITESPACE_SIGNIFICANT_LANGS is now defined in code_ast.py (not the worker)
from tokenizer_analysis.loaders.code_data import CodeDataLoader
from tokenizer_analysis.core.input_types import TokenizedData


# ======================================================================
# Helpers
# ======================================================================

_EPS = 1e-9


def _make_instance():
    """Return a bare ASTBoundaryMetrics without a live InputProvider.

    Only usable for calling static / class methods and helpers that
    don't touch ``self.input_provider``.
    """
    inst = object.__new__(ASTBoundaryMetrics)
    inst._tokenizer_vocab_cache = {}
    inst._warned_tokenizers = set()
    inst._char_decode_table = None
    return inst


# ======================================================================
# _build_source_to_recon_map
# ======================================================================

class TestSourceToReconMap:
    """Verify source-code to reconstructed-text coordinate mapping."""

    def test_no_whitespace(self):
        source = "abc"
        recon = "abc"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        assert result == [0, 1, 2]

    def test_leading_whitespace_dropped(self):
        source = "  abc"
        recon = "abc"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        # first two chars are spaces -> None; then abc maps to 0,1,2
        assert result == [None, None, 0, 1, 2]

    def test_internal_whitespace_dropped(self):
        source = "a b"
        recon = "ab"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        assert result == [0, None, 1]

    def test_newlines_dropped(self):
        source = "a\nb"
        recon = "ab"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        assert result == [0, None, 1]

    def test_tab_dropped(self):
        source = "a\tb"
        recon = "ab"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        assert result == [0, None, 1]

    def test_case_mismatch_produces_none(self):
        """Case-mismatched characters are not mapped (exact match only)."""
        source = "ABC"
        recon = "abc"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        assert result == [None, None, None]

    def test_identical_source_and_recon(self):
        source = "def fibonacci(n):"
        recon = "deffibonacci(n):"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        # 'd','e','f' match; space->None; then rest matches
        assert result[0] == 0  # 'd'
        assert result[1] == 1  # 'e'
        assert result[2] == 2  # 'f'
        assert result[3] is None  # ' '
        assert result[4] == 3  # 'f' (of fibonacci)

    def test_case_sensitive_matching(self):
        """Exact match pairs characters correctly."""
        source = "aA"
        recon = "aA"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        # 'a' matches recon[0] exactly, 'A' matches recon[1] exactly
        assert result == [0, 1]

    def test_dropped_character(self):
        """A character dropped in recon leaves None; subsequent chars align."""
        source = "a b"
        recon = "ab"
        result = ASTBoundaryMetrics._build_source_to_recon_map(source, recon)
        assert result == [0, None, 1]


# ======================================================================
# _byte_to_char_offsets
# ======================================================================

class TestByteToCharOffsets:

    def test_ascii(self):
        source = "abc".encode("utf-8")
        offsets = ASTBoundaryMetrics._byte_to_char_offsets(source)
        # 3 bytes + sentinel = 4 entries
        assert offsets == [0, 1, 2, 3]

    def test_multibyte_utf8(self):
        source = "aé".encode("utf-8")  # 'a' = 1 byte, 'é' = 2 bytes
        offsets = ASTBoundaryMetrics._byte_to_char_offsets(source)
        # byte 0 -> char 0 ('a')
        # bytes 1,2 -> char 1 ('é')
        # sentinel -> char 2
        assert offsets == [0, 1, 1, 2]

    def test_empty(self):
        offsets = ASTBoundaryMetrics._byte_to_char_offsets(b"")
        assert offsets == [0]  # just sentinel

    def test_cjk_character(self):
        # Chinese character: 3 bytes in UTF-8
        source = "x中".encode("utf-8")
        offsets = ASTBoundaryMetrics._byte_to_char_offsets(source)
        # x: 1 byte -> char 0
        # 中: 3 bytes -> char 1,1,1
        # sentinel -> char 2
        assert offsets == [0, 1, 1, 1, 2]


# ======================================================================
# _check_boundary_alignment
# ======================================================================

class TestBoundaryAlignment:
    """Test boundary alignment check logic."""

    def test_perfect_alignment_at_start(self):
        # Tokens: [0,0,0, 1,1,1] representing "abcdef"
        # Node spans chars 0..3 (abc) -> all token 0
        source_to_recon = [0, 1, 2, 3, 4, 5]
        char_to_token = [0, 0, 0, 1, 1, 1]
        result = ASTBoundaryMetrics._check_boundary_alignment(
            0, 3, source_to_recon, char_to_token
        )
        assert result is not None
        assert result["start_aligned"] is True
        assert result["end_aligned"] is True
        assert result["fully_aligned"] is True
        assert result["cross_boundary"] is False

    def test_perfect_alignment_in_middle(self):
        # Tokens: [0,0, 1,1, 2,2]
        # Node spans chars 2..4 (token 1) -> start changes from 0 to 1, end changes from 1 to 2
        source_to_recon = [0, 1, 2, 3, 4, 5]
        char_to_token = [0, 0, 1, 1, 2, 2]
        result = ASTBoundaryMetrics._check_boundary_alignment(
            2, 4, source_to_recon, char_to_token
        )
        assert result is not None
        assert result["fully_aligned"] is True

    def test_cross_boundary_start(self):
        # Tokens: [0,0,0, 1,1,1]
        # Node spans chars 1..4 -> starts mid-token-0, ends mid-token-1
        source_to_recon = [0, 1, 2, 3, 4, 5]
        char_to_token = [0, 0, 0, 1, 1, 1]
        result = ASTBoundaryMetrics._check_boundary_alignment(
            1, 4, source_to_recon, char_to_token
        )
        assert result is not None
        assert result["start_aligned"] is False
        assert result["fully_aligned"] is False
        assert result["cross_boundary"] is True

    def test_unmappable_span_returns_none(self):
        # All whitespace in source -> all None in source_to_recon
        source_to_recon = [None, None, None]
        char_to_token = [0, 0, 0]
        result = ASTBoundaryMetrics._check_boundary_alignment(
            0, 3, source_to_recon, char_to_token
        )
        assert result is None

    def test_node_at_end_of_text(self):
        # Node spans to end of text -> end_aligned should be True
        source_to_recon = [0, 1, 2]
        char_to_token = [0, 0, 1]
        result = ASTBoundaryMetrics._check_boundary_alignment(
            2, 3, source_to_recon, char_to_token
        )
        assert result is not None
        assert result["end_aligned"] is True

    def test_with_whitespace_in_source(self):
        # Source: "a b" -> recon: "ab"
        # source_to_recon: [0, None, 1]
        # char_to_token: [0, 1]  (tokens for "ab")
        # Node spans source chars 2..3 ('b') -> recon pos 1..2
        source_to_recon = [0, None, 1]
        char_to_token = [0, 1]
        result = ASTBoundaryMetrics._check_boundary_alignment(
            2, 3, source_to_recon, char_to_token
        )
        assert result is not None
        # Token at recon pos 1 differs from token at recon pos 0 -> start aligned
        assert result["start_aligned"] is True
        # recon_end=2 >= len(char_to_token) -> end aligned
        assert result["end_aligned"] is True
        assert result["fully_aligned"] is True


# ======================================================================
# _classify_node (requires tree-sitter)
# ======================================================================

class _MockNode:
    """Minimal tree-sitter node substitute for classification tests."""

    def __init__(self, node_type, text, is_named=True, child_count=0):
        self.type = node_type
        self.text = text.encode("utf-8") if isinstance(text, str) else text
        self.is_named = is_named
        self.child_count = child_count


class TestClassifyNode:
    """Test AST node classification without tree-sitter."""

    def test_delimiter_paren(self):
        node = _MockNode("(", "(", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "delimiter"

    def test_delimiter_brace(self):
        node = _MockNode("{", "{", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "delimiter"

    def test_delimiter_semicolon(self):
        node = _MockNode(";", ";", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "delimiter"

    def test_identifier(self):
        node = _MockNode("identifier", "fibonacci", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "identifier"

    def test_type_identifier(self):
        node = _MockNode("type_identifier", "Int", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "identifier"

    def test_field_identifier(self):
        node = _MockNode("field_identifier", "name", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "identifier"

    def test_string_literal(self):
        node = _MockNode("string_literal", '"hello"', is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "literal"

    def test_integer(self):
        node = _MockNode("integer", "42", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "literal"

    def test_float_literal(self):
        node = _MockNode("float_literal", "3.14", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "literal"

    def test_true_literal(self):
        node = _MockNode("true", "true", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "literal"

    def test_false_literal(self):
        node = _MockNode("false", "false", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "literal"

    def test_null_literal(self):
        node = _MockNode("null", "null", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "literal"

    def test_keyword_anonymous_if(self):
        # Tree-sitter represents keywords as anonymous nodes whose type == text
        node = _MockNode("if", "if", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "keyword"

    def test_keyword_anonymous_return(self):
        node = _MockNode("return", "return", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "keyword"

    def test_keyword_anonymous_class(self):
        node = _MockNode("class", "class", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "keyword"

    def test_keyword_def_in_known_set(self):
        # Named node with text in _KNOWN_KEYWORDS
        node = _MockNode("keyword", "def", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) == "keyword"

    def test_operator_plus(self):
        node = _MockNode("+", "+", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "operator"

    def test_operator_equals_equals(self):
        node = _MockNode("==", "==", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "operator"

    def test_operator_ampersand_ampersand(self):
        node = _MockNode("&&", "&&", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "operator"

    def test_operator_arrow(self):
        node = _MockNode("->", "->", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "operator"

    def test_empty_text_returns_none(self):
        node = _MockNode("", "", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) is None

    def test_non_leaf_generic_node_returns_none(self):
        # A named node with non-matching type
        node = _MockNode("expression_statement", "x + 1", is_named=True)
        assert ASTBoundaryMetrics._classify_node(node) is None

    # -- Punctuation exclusion from operator category --

    def test_colon_not_operator(self):
        node = _MockNode(":", ":", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) is None

    def test_dot_not_operator(self):
        node = _MockNode(".", ".", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) is None

    def test_double_colon_not_operator(self):
        node = _MockNode("::", "::", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) is None

    def test_at_sign_not_operator(self):
        node = _MockNode("@", "@", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) is None

    def test_hash_not_operator(self):
        node = _MockNode("#", "#", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) is None

    def test_spread_not_operator(self):
        node = _MockNode("...", "...", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) is None

    def test_arrow_still_operator(self):
        node = _MockNode("->", "->", is_named=False)
        assert ASTBoundaryMetrics._classify_node(node) == "operator"


# ======================================================================
# CodeDataLoader
# ======================================================================

class TestCodeDataLoader:
    """Tests for the code data loader."""

    def test_synthetic_samples_all_languages(self):
        samples = CodeDataLoader.generate_synthetic_samples()
        expected_langs = set(CodeDataLoader._LANG_TO_TREESITTER.keys())
        assert set(samples.keys()) == expected_langs

    def test_synthetic_samples_nonempty(self):
        samples = CodeDataLoader.generate_synthetic_samples()
        for lang, snippets in samples.items():
            assert len(snippets) > 0, f"No snippets for {lang}"
            for snippet in snippets:
                assert len(snippet.strip()) > 0, f"Empty snippet for {lang}"

    def test_get_languages_after_synthetic(self):
        loader = CodeDataLoader()
        synthetic = CodeDataLoader.generate_synthetic_samples()
        for lang, snippets in synthetic.items():
            loader.code_snippets[lang] = snippets
        assert len(loader.get_languages()) == len(CodeDataLoader._LANG_TO_TREESITTER)

    def test_get_code_snippets_missing_lang(self):
        loader = CodeDataLoader()
        assert loader.get_code_snippets("nonexistent") == []

    def test_cap_limits_loaded_snippets(self, tmp_path):
        """Loading from a directory with many files respects the cap."""
        lang_dir = tmp_path / "python"
        lang_dir.mkdir()
        for i in range(10):
            (lang_dir / f"f{i}.py").write_text(f"x = {i}\n")

        loader = CodeDataLoader(
            {"python": str(lang_dir)}, max_snippets_per_lang=3
        )
        loader.load_all()
        assert len(loader.get_code_snippets("python")) == 3

    def test_cap_limits_parquet_snippets(self, tmp_path):
        """Parquet loading also respects the cap."""
        import pandas as pd
        df = pd.DataFrame({"content": [f"x = {i}" for i in range(20)]})
        path = tmp_path / "code.parquet"
        df.to_parquet(path)

        loader = CodeDataLoader(
            {"python": str(path)}, max_snippets_per_lang=5
        )
        loader.load_all()
        assert len(loader.get_code_snippets("python")) == 5

    def test_lang_to_treesitter_consistency(self):
        """All extension-mapped languages should have a tree-sitter grammar mapping."""
        for lang in CodeDataLoader._LANG_EXTENSIONS:
            assert lang in CodeDataLoader._LANG_TO_TREESITTER, (
                f"Language {lang} has extensions but no tree-sitter mapping"
            )


# ======================================================================
# StarCoder metadata stripping
# ======================================================================

class TestStarCoderMetadataStripping:
    """Verify that StarCoder metadata prefixes are stripped from content."""

    def test_all_three_tags(self):
        raw = "<reponame>user/repo<filename>main.py<gh_stars>1-10\nimport os\n"
        assert CodeDataLoader._strip_starcoder_metadata(raw) == "import os\n"

    def test_filename_only(self):
        raw = "<filename>setup.py\nimport setuptools\n"
        assert CodeDataLoader._strip_starcoder_metadata(raw) == "import setuptools\n"

    def test_gh_stars_only(self):
        raw = "<gh_stars>1-10\nfn main() {}\n"
        assert CodeDataLoader._strip_starcoder_metadata(raw) == "fn main() {}\n"

    def test_no_tags(self):
        raw = "import os\nprint('hello')\n"
        assert CodeDataLoader._strip_starcoder_metadata(raw) == raw

    def test_tags_not_at_start_preserved(self):
        raw = "# comment\n<filename>not_a_prefix.py\n"
        assert CodeDataLoader._strip_starcoder_metadata(raw) == raw


# ======================================================================
# Parquet loading
# ======================================================================

class TestParquetLoading:
    """Verify that CodeDataLoader can read parquet files."""

    @pytest.fixture()
    def parquet_file(self, tmp_path):
        """Create a small parquet file with a content column."""
        import pandas as pd
        df = pd.DataFrame({
            "content": [
                "<reponame>u/r<filename>a.py<gh_stars>0\ndef foo(): pass\n",
                "class Bar:\n    x = 1\n",
                "",       # empty — should be skipped
                "  \t  ", # whitespace-only — should be skipped
            ]
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path)
        return str(path)

    def test_read_parquet_strips_metadata(self, parquet_file):
        snippets = CodeDataLoader._read_parquet(parquet_file)
        assert len(snippets) == 2
        assert snippets[0] == "def foo(): pass"
        assert snippets[1] == "class Bar:\n    x = 1"

    def test_load_language_parquet(self, parquet_file):
        loader = CodeDataLoader({"python": parquet_file})
        loader.load_all()
        assert "python" in loader.get_languages()
        assert len(loader.get_code_snippets("python")) == 2

    def test_read_parquet_missing_column(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({"code": ["print(1)"]})
        path = tmp_path / "no_content.parquet"
        df.to_parquet(path)
        snippets = CodeDataLoader._read_parquet(str(path))
        assert snippets == []


# ======================================================================
# Synthetic samples parse without tree-sitter errors
# ======================================================================

class TestSyntheticSamplesParsing:
    """Verify that all synthetic code snippets parse without tree-sitter errors.

    This test is skipped if tree-sitter-language-pack is not installed.
    """

    @pytest.fixture(scope="class")
    def ts_pack(self):
        try:
            import tree_sitter_language_pack
            return tree_sitter_language_pack
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

    def test_all_snippets_parse(self, ts_pack):
        samples = CodeDataLoader.generate_synthetic_samples()
        for lang, snippets in samples.items():
            ts_name = CodeDataLoader._LANG_TO_TREESITTER.get(lang)
            if ts_name is None:
                continue
            try:
                parser = ts_pack.get_parser(ts_name)
            except Exception:
                pytest.skip(f"No tree-sitter grammar for {ts_name}")

            for i, snippet in enumerate(snippets):
                tree = parser.parse(snippet.encode("utf-8"))
                root = tree.root_node

                # Count ERROR nodes
                errors = []
                def _find_errors(node):
                    if node.type == "ERROR":
                        errors.append(
                            f"ERROR at {node.start_point}-{node.end_point}: "
                            f"'{snippet[node.start_byte:node.end_byte][:50]}'"
                        )
                    for child in node.children:
                        _find_errors(child)

                _find_errors(root)
                assert len(errors) == 0, (
                    f"Parse errors in {lang} snippet #{i}: {errors}"
                )

    def test_extract_leaf_spans_nonempty(self, ts_pack):
        """Verify that extracted spans cover all 5 categories for key languages."""
        inst = _make_instance()
        inst._ts_pack = ts_pack
        inst._treesitter_available = True

        # Test a subset of languages likely to cover all categories
        test_langs = ["python", "javascript", "java", "rust", "go"]
        samples = CodeDataLoader.generate_synthetic_samples()

        for lang in test_langs:
            ts_name = CodeDataLoader._LANG_TO_TREESITTER.get(lang)
            if ts_name is None:
                continue

            parser = ts_pack.get_parser(ts_name)
            snippet = samples[lang][0]
            tree = parser.parse(snippet.encode("utf-8"))
            spans = inst._extract_leaf_spans(tree)

            # Each language snippet should produce at least identifiers,
            # keywords, operators, and delimiters
            for cat in ("identifier", "keyword", "operator", "delimiter"):
                assert len(spans[cat]) > 0, (
                    f"No {cat} spans extracted from {lang} snippet"
                )


from .conftest import MockTokenizer, MockProvider as _MockProvider


# ======================================================================
# Mock infrastructure for end-to-end compute() tests
# ======================================================================

class _MockTokenizer(MockTokenizer):
    """Extends MockTokenizer with encode/can_encode for AST tests."""

    def can_encode(self):
        return True

    def encode(self, text):
        """Character-level encoding: one token per character."""
        return list(range(len(text)))

    def encode_with_offsets(self, text):
        ids = self.encode(text)
        offsets = [(i, i + 1) for i in range(len(text))]
        return ids, offsets


class _CharTokenizer:
    """Simple character-level tokenizer for testing."""

    def __init__(self):
        pass

    def convert_ids_to_tokens(self, ids):
        # The IDs are character ordinals
        return [chr(i) for i in ids]

    def can_encode(self):
        return True

    def encode(self, text):
        return [ord(c) for c in text]

    def encode_with_offsets(self, text):
        ids = self.encode(text)
        offsets = [(i, i + 1) for i in range(len(text))]
        return ids, offsets

    def get_vocab(self):
        return {}


class _PerfectTokenizer:
    """Tokenizer that returns pre-defined tokens for specific snippets."""

    # Character decode table matching BaseMetrics._DEFAULT_CHAR_DECODE
    _DECODE = {'Ġ': ' ', '▁': ' ', 'Ċ': '\n', 'ĉ': '\t', 'č': '\r'}

    def __init__(self, snippet_to_tokens):
        """
        Args:
            snippet_to_tokens: Dict mapping snippet text -> list of token strings
        """
        self._snippet_map = snippet_to_tokens
        self._build_vocab()

    def _build_vocab(self):
        self._token_to_id = {}
        self._id_to_token = {}
        next_id = 0
        for tokens in self._snippet_map.values():
            for t in tokens:
                if t not in self._token_to_id:
                    self._token_to_id[t] = next_id
                    self._id_to_token[next_id] = t
                    next_id += 1

    def _decode_token(self, raw):
        """Decode a raw token string to its source characters."""
        decoded = ''.join(self._DECODE.get(ch, ch) for ch in raw)
        # Strip ## prefix (continuation marker)
        if decoded.startswith('##'):
            return decoded[2:]
        # Strip </w> suffix
        if decoded.endswith('</w>'):
            return decoded[:-4]
        # Strip @@ suffix
        if decoded.endswith('@@'):
            return decoded[:-2]
        return decoded

    def can_encode(self):
        return True

    def encode(self, text):
        tokens = self._snippet_map.get(text, [])
        return [self._token_to_id[t] for t in tokens]

    def encode_with_offsets(self, text):
        """Return (ids, offsets) by greedily matching decoded tokens to source."""
        token_strs = self._snippet_map.get(text, [])
        ids = [self._token_to_id[t] for t in token_strs]
        offsets = []
        src_idx = 0
        for raw in token_strs:
            decoded = self._decode_token(raw)
            start = src_idx
            for ch in decoded:
                if src_idx >= len(text):
                    break
                if text[src_idx] == ch:
                    src_idx += 1
                elif ch in (' ', '\t') and text[src_idx] in (' ', '\t'):
                    src_idx += 1
                # else: skip decoded char
            offsets.append((start, src_idx))
        return ids, offsets

    def convert_ids_to_tokens(self, ids):
        return [self._id_to_token[i] for i in ids]

    def get_vocab(self):
        return dict(self._token_to_id)


# ======================================================================
# Graceful degradation when tree-sitter is unavailable
# ======================================================================

class TestGracefulDegradation:
    """Verify compute() returns error dict when tree-sitter is missing."""

    def test_returns_error_when_unavailable(self):
        """When tree-sitter import fails, compute returns an error dict."""
        inst = _make_instance()
        inst._treesitter_available = False
        inst.input_provider = _MockProvider("test", _CharTokenizer())
        inst.tokenizer_names = ["test"]

        loader = CodeDataLoader()
        synthetic = CodeDataLoader.generate_synthetic_samples()
        for lang, snippets in synthetic.items():
            loader.code_snippets[lang] = snippets
        inst.code_loader = loader
        inst.max_snippets_per_lang = 1

        result = inst.compute()
        assert "ast_boundary_alignment" in result
        assert "error" in result["ast_boundary_alignment"]


# ======================================================================
# End-to-end compute() test with tree-sitter
# ======================================================================

class TestEndToEnd:
    """Full pipeline test with real tree-sitter parsing."""

    @pytest.fixture(scope="class")
    def ts_pack(self):
        try:
            import tree_sitter_language_pack
            return tree_sitter_language_pack
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

    def test_compute_with_char_tokenizer(self, ts_pack):
        """Character-level tokenizer should produce non-zero alignment rates.

        A character-level tokenizer splits every character into its own token,
        so every AST node boundary aligns perfectly with token boundaries.
        """
        char_tok = _CharTokenizer()
        provider = _MockProvider("char_tok", char_tok)

        # Build metrics with a single Python snippet
        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["char_tok"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [
            'def add(a, b):\n'
            '    return a + b\n'
        ]}
        inst.code_loader = loader

        result = inst.compute()

        assert "ast_boundary_alignment" in result
        ast = result["ast_boundary_alignment"]
        assert "error" not in ast
        assert "per_tokenizer" in ast
        assert "char_tok" in ast["per_tokenizer"]

        tok_data = ast["per_tokenizer"]["char_tok"]
        assert "overall" in tok_data
        # Character-level tokenizer: every boundary aligns
        assert tok_data["overall"]["full_alignment_rate"] == pytest.approx(1.0)
        assert tok_data["overall"]["count"] > 0

    def test_compute_summary_structure(self, ts_pack):
        """Verify the summary structure is populated."""
        char_tok = _CharTokenizer()
        provider = _MockProvider("test_tok", char_tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["test_tok"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [
            'x = 1 + 2\n'
        ]}
        inst.code_loader = loader

        result = inst.compute()
        ast = result["ast_boundary_alignment"]

        # Check summary
        assert "summary" in ast
        assert "test_tok" in ast["summary"]
        s = ast["summary"]["test_tok"]
        assert "avg_full_alignment_rate" in s
        assert "total_nodes_analyzed" in s
        assert s["total_nodes_analyzed"] > 0
        assert "languages_analyzed" in s
        assert s["languages_analyzed"] == 1

    def test_compute_by_category(self, ts_pack):
        """Verify by_category results contain known categories."""
        char_tok = _CharTokenizer()
        provider = _MockProvider("test_tok", char_tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["test_tok"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [CodeDataLoader.generate_synthetic_samples()["python"][0]]}
        inst.code_loader = loader

        result = inst.compute()
        ast = result["ast_boundary_alignment"]
        by_cat = ast["per_tokenizer"]["test_tok"]["by_category"]

        # At least identifiers, keywords, operators, delimiters should be present
        for cat in ("identifier", "keyword", "operator", "delimiter"):
            assert cat in by_cat, f"Missing category {cat}"
            assert "python" in by_cat[cat], f"Missing python in {cat}"
            assert by_cat[cat]["python"]["count"] > 0

    def test_compute_by_language(self, ts_pack):
        """Verify by_language results appear for each loaded language."""
        char_tok = _CharTokenizer()
        provider = _MockProvider("test_tok", char_tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["test_tok"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        samples = CodeDataLoader.generate_synthetic_samples()
        # Test with 3 languages
        for lang in ["python", "javascript", "go"]:
            loader.code_snippets[lang] = samples[lang]
        inst.code_loader = loader

        result = inst.compute()
        ast = result["ast_boundary_alignment"]
        by_lang = ast["per_tokenizer"]["test_tok"]["by_language"]

        for lang in ["python", "javascript", "go"]:
            assert lang in by_lang, f"Missing language {lang}"
            assert by_lang[lang]["count"] > 0

    def test_perfect_tokenizer_high_alignment(self, ts_pack):
        """A tokenizer that preserves AST boundaries should score well.

        We use a manually crafted tokenizer that splits a simple Python
        snippet exactly at keyword/identifier/operator/delimiter boundaries.
        """
        snippet = 'x = 1 + 2'
        # Tokens: "x" " " "=" " " "1" " " "+" " " "2"
        # This tokenizer keeps each meaningful AST node as its own token
        tokens = ["x", "Ġ=", "Ġ1", "Ġ+", "Ġ2"]
        tok = _PerfectTokenizer({snippet: tokens})
        provider = _MockProvider("perfect", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["perfect"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        ast = result["ast_boundary_alignment"]
        overall = ast["per_tokenizer"]["perfect"]["overall"]

        # Should have very high alignment since tokens match AST boundaries
        assert overall["full_alignment_rate"] > 0.5
        assert overall["count"] > 0


# ======================================================================
# print_results smoke test
# ======================================================================

class TestPrintResults:

    def test_print_error(self, capsys):
        inst = _make_instance()
        inst.tokenizer_names = ["test"]
        inst.print_results({"ast_boundary_alignment": {"error": "no tree-sitter"}})
        captured = capsys.readouterr()
        assert "no tree-sitter" in captured.out

    def test_print_empty(self, capsys):
        inst = _make_instance()
        inst.tokenizer_names = ["test"]
        inst.print_results({})
        captured = capsys.readouterr()
        # Should produce no output
        assert captured.out == ""

    def test_print_results_with_data(self, capsys):
        inst = _make_instance()
        inst.tokenizer_names = ["test_tok"]
        results = {
            "ast_boundary_alignment": {
                "per_tokenizer": {
                    "test_tok": {
                        "by_category": {
                            "identifier": {
                                "python": {
                                    "full_alignment_rate": 0.9,
                                    "start_alignment_rate": 0.95,
                                    "end_alignment_rate": 0.92,
                                    "cross_boundary_rate": 0.1,
                                    "count": 50,
                                }
                            }
                        },
                        "by_language": {
                            "python": {
                                "overall_full_alignment_rate": 0.85,
                                "overall_start_alignment_rate": 0.90,
                                "overall_end_alignment_rate": 0.88,
                                "overall_cross_boundary_rate": 0.15,
                                "count": 100,
                            }
                        },
                        "overall": {
                            "full_alignment_rate": 0.85,
                            "start_alignment_rate": 0.90,
                            "end_alignment_rate": 0.88,
                            "cross_boundary_rate": 0.15,
                            "count": 100,
                        }
                    }
                },
                "summary": {
                    "test_tok": {
                        "avg_full_alignment_rate": 0.85,
                        "avg_start_alignment_rate": 0.90,
                        "avg_end_alignment_rate": 0.88,
                        "avg_cross_boundary_rate": 0.15,
                        "total_nodes_analyzed": 100,
                        "languages_analyzed": 1,
                    }
                }
            }
        }
        inst.print_results(results)
        captured = capsys.readouterr()
        assert "AST BOUNDARY ALIGNMENT" in captured.out
        assert "SUMMARY STATISTICS" in captured.out
        assert "test_tok" in captured.out
        assert "identifier" in captured.out
        assert "python" in captured.out


# ======================================================================
# _decode_raw_token
# ======================================================================

class TestDecodeRawToken:
    """Verify whitespace-preserving token decoding."""

    def setup_method(self):
        self.inst = _make_instance()

    def test_g_prefix_to_space(self):
        assert self.inst._decode_raw_token("Ġdef") == " def"

    def test_underscore_prefix_to_space(self):
        assert self.inst._decode_raw_token("▁def") == " def"

    def test_space_prefix_to_space(self):
        assert self.inst._decode_raw_token(" def") == " def"

    def test_continuation_stripped(self):
        assert self.inst._decode_raw_token("##ing") == "ing"

    def test_end_word_stripped(self):
        assert self.inst._decode_raw_token("word</w>") == "word"

    def test_continuation_end_stripped(self):
        assert self.inst._decode_raw_token("word@@") == "word"

    def test_special_token_returns_none(self):
        assert self.inst._decode_raw_token("<|endoftext|>") is None

    def test_special_token_bracket(self):
        assert self.inst._decode_raw_token("[CLS]") is None

    def test_plain_unchanged(self):
        assert self.inst._decode_raw_token("abc") == "abc"

    def test_multi_g_four_spaces(self):
        """ĠĠĠĠ → 4 spaces (all chars decoded)."""
        assert self.inst._decode_raw_token("ĠĠĠĠ") == "    "

    def test_newline_char(self):
        assert self.inst._decode_raw_token("Ċ") == "\n"

    def test_colon_newline(self):
        assert self.inst._decode_raw_token(":Ċ") == ":\n"

    def test_tab_char(self):
        assert self.inst._decode_raw_token("ĉ") == "\t"

    def test_g_with_embedded_newline(self):
        """Ġ followed by text then Ċ."""
        assert self.inst._decode_raw_token("Ġif:Ċ") == " if:\n"


# ======================================================================
# _process_token (shared helper in BaseMetrics)
# ======================================================================

class TestProcessToken:
    """Verify the shared _process_token helper produces results consistent
    with both _clean_token (preserve_space=False) and _decode_raw_token
    (preserve_space=True)."""

    def setup_method(self):
        self.inst = _make_instance()

    # -- preserve_space=False (mirrors _clean_token) --

    def test_strip_g_prefix(self):
        assert self.inst._process_token("Ġdef", preserve_space=False) == "def"

    def test_strip_underscore_prefix(self):
        assert self.inst._process_token("▁def", preserve_space=False) == "def"

    def test_strip_continuation(self):
        assert self.inst._process_token("##ing", preserve_space=False) == "ing"

    def test_strip_end_word(self):
        assert self.inst._process_token("word</w>", preserve_space=False) == "word"

    def test_strip_continuation_end(self):
        assert self.inst._process_token("word@@", preserve_space=False) == "word"

    def test_special_returns_none(self):
        assert self.inst._process_token("<|endoftext|>", preserve_space=False) is None

    def test_plain_unchanged(self):
        assert self.inst._process_token("abc", preserve_space=False) == "abc"

    # -- preserve_space=True (mirrors _decode_raw_token) --

    def test_preserve_g_prefix(self):
        assert self.inst._process_token("Ġdef", preserve_space=True) == " def"

    def test_preserve_underscore_prefix(self):
        assert self.inst._process_token("▁def", preserve_space=True) == " def"

    def test_preserve_continuation(self):
        assert self.inst._process_token("##ing", preserve_space=True) == "ing"

    def test_preserve_end_word(self):
        assert self.inst._process_token("word</w>", preserve_space=True) == "word"

    def test_preserve_continuation_end(self):
        assert self.inst._process_token("word@@", preserve_space=True) == "word"

    def test_preserve_special_returns_none(self):
        assert self.inst._process_token("<|endoftext|>", preserve_space=True) is None

    def test_preserve_plain_unchanged(self):
        assert self.inst._process_token("abc", preserve_space=True) == "abc"

    # -- Multi-char decode (byte-level BPE) --

    def test_multi_g_preserve(self):
        """ĠĠĠĠ should decode to 4 spaces with preserve_space=True."""
        assert self.inst._process_token("ĠĠĠĠ", preserve_space=True) == "    "

    def test_multi_g_clean(self):
        """ĠĠĠĠ should decode to 3 spaces with preserve_space=False (leading space stripped)."""
        assert self.inst._process_token("ĠĠĠĠ", preserve_space=False) == "   "

    def test_newline_alone(self):
        assert self.inst._process_token("Ċ", preserve_space=True) == "\n"

    def test_newline_alone_clean(self):
        assert self.inst._process_token("Ċ", preserve_space=False) == "\n"

    def test_colon_newline(self):
        assert self.inst._process_token(":Ċ", preserve_space=True) == ":\n"

    def test_tab_decode(self):
        assert self.inst._process_token("ĉ", preserve_space=True) == "\t"

    def test_cr_decode(self):
        assert self.inst._process_token("č", preserve_space=True) == "\r"

    def test_sentencepiece_multi_preserve(self):
        """▁▁▁▁ should decode to 4 spaces with preserve_space=True."""
        assert self.inst._process_token("▁▁▁▁", preserve_space=True) == "    "

    def test_sentencepiece_multi_clean(self):
        """▁▁▁▁ should decode to 3 spaces with preserve_space=False."""
        assert self.inst._process_token("▁▁▁▁", preserve_space=False) == "   "

    # -- Consistency: _clean_token and _decode_raw_token delegate correctly --

    def test_clean_token_matches_process_token(self):
        tokens = ["Ġdef", "▁x", "##ing", "word</w>", "tok@@", "<|pad|>", "abc"]
        for t in tokens:
            assert self.inst._clean_token(t) == self.inst._process_token(t, preserve_space=False)

    def test_decode_raw_token_matches_process_token(self):
        tokens = ["Ġdef", "▁x", "##ing", "word</w>", "tok@@", "<|pad|>", "abc"]
        for t in tokens:
            assert self.inst._decode_raw_token(t) == self.inst._process_token(t, preserve_space=True)

    # -- SentencePiece byte-fallback decoding (<0xNN> → chr(NN)) --

    def test_sp_byte_fallback_newline(self):
        """<0x0A> should decode to a literal newline (the bug that broke EuroLLM)."""
        assert self.inst._decode_byte_fallback("<0x0A>") == "\n"
        assert self.inst._process_token("<0x0A>", preserve_space=False) == "\n"
        assert self.inst._process_token("<0x0A>", preserve_space=True) == "\n"

    def test_sp_byte_fallback_tab(self):
        assert self.inst._process_token("<0x09>", preserve_space=True) == "\t"

    def test_sp_byte_fallback_lowercase_hex(self):
        """Accept lowercase hex even though SentencePiece normally emits uppercase."""
        assert self.inst._decode_byte_fallback("<0x0a>") == "\n"

    def test_sp_byte_fallback_within_token(self):
        """Byte-fallback substring inside a larger string should be decoded in place."""
        assert self.inst._decode_byte_fallback("Ġ<0x0A>") == "Ġ\n"

    def test_sp_byte_fallback_no_match_unchanged(self):
        """Tokens without the <0xNN> pattern pass through unchanged."""
        assert self.inst._decode_byte_fallback("def") == "def"
        assert self.inst._decode_byte_fallback("<0x>") == "<0x>"      # invalid (no hex)
        assert self.inst._decode_byte_fallback("<0xZZ>") == "<0xZZ>"  # invalid hex

    def test_build_char_to_token_map_with_byte_fallback(self):
        """A <0xNN> token in the middle of a token sequence should produce one
        decoded character and attribute char_to_token to its index."""
        tokens = ["def", "<0x0A>", "x"]
        recon, c2t = self.inst._build_char_to_token_map(tokens)
        assert recon == "def\nx"
        assert c2t == [0, 0, 0, 1, 2]


# ======================================================================
# _build_source_char_to_token_map
# ======================================================================

class TestSourceCharToTokenMap:
    """Verify whitespace-inclusive source → token mapping."""

    def setup_method(self):
        self.inst = _make_instance()

    def test_simple_tokens(self):
        source = "abc"
        tokens = ["a", "b", "c"]
        result = self.inst._build_source_char_to_token_map(source, tokens)
        assert result == [0, 1, 2]

    def test_g_prefix_tokens_with_whitespace(self):
        source = "a b"
        tokens = ["a", "Ġb"]
        result = self.inst._build_source_char_to_token_map(source, tokens)
        assert result == [0, 1, 1]  # space is part of token 1

    def test_special_tokens_skipped(self):
        source = "ab"
        tokens = ["<|bos|>", "a", "b"]
        result = self.inst._build_source_char_to_token_map(source, tokens)
        assert result == [1, 2]

    def test_indentation_mapping(self):
        source = "    x"
        # Four spaces as a single token, then 'x'
        tokens = ["Ġ   ", "x"]  # Ġ decodes to space, so " " + "   " = "    "
        result = self.inst._build_source_char_to_token_map(source, tokens)
        assert result == [0, 0, 0, 0, 1]

    def test_tab_space_mismatch(self):
        """Source has tab but token decodes as space — pointer must not stall.

        Without the whitespace-class fallback the space in the decoded
        token never matches the tab in the source, so src_idx stays at 0
        and every subsequent position is None.
        """
        source = "\tx = 1"
        # Token " x" decodes a leading space where source has a tab
        tokens = [" x", "Ġ=", "Ġ1"]
        result = self.inst._build_source_char_to_token_map(source, tokens)
        # ' ' matches '\t' via whitespace-class fallback, then rest aligns
        assert result == [0, 0, 1, 1, 2, 2]

    def test_longer_than_source(self):
        """Tokens with more characters than source should stop cleanly."""
        source = "ab"
        tokens = ["abc"]
        result = self.inst._build_source_char_to_token_map(source, tokens)
        assert result == [0, 0]

    def test_bpe_multiline_no_none_gaps(self):
        """Byte-level BPE tokens with ĠĠĠĠ and Ċ should map all chars."""
        source = "if True:\n    x = 1"
        # BPE-style encoding: Ċ for newline, ĠĠĠĠ for 4-space indentation
        tokens = ["if", "ĠTrue", "Ġ:", "Ċ", "ĠĠĠĠ", "x", "Ġ=", "Ġ1"]
        result = self.inst._build_source_char_to_token_map(source, tokens)
        assert len(result) == len(source)
        # No None gaps
        assert None not in result

    def test_sentencepiece_multiline_no_none_gaps(self):
        """SentencePiece tokens with ▁▁▁▁ should map all chars."""
        source = "a b\n    c"
        tokens = ["a", "▁b", "\n", "▁▁▁▁", "c"]
        result = self.inst._build_source_char_to_token_map(source, tokens)
        assert len(result) == len(source)
        assert None not in result

    def test_bpe_multi_g_whitespace(self):
        """Multi-Ġ whitespace tokens should correctly map to source spaces."""
        source = "    x"
        tokens = ["ĠĠĠĠ", "x"]
        result = self.inst._build_source_char_to_token_map(source, tokens)
        # All 4 spaces should map to token 0, 'x' to token 1
        assert result == [0, 0, 0, 0, 1]


# ======================================================================
# _extract_line_indentation
# ======================================================================

class TestExtractLineIndentation:
    """Verify line indentation extraction."""

    def test_python_style(self):
        source = "def f():\n    return 1"
        result = ASTBoundaryMetrics._extract_line_indentation(source)
        assert len(result) == 2
        ws0, start0, end0 = result[0]
        assert ws0 == ""  # no indentation on first line
        ws1, start1, end1 = result[1]
        assert ws1 == "    "
        assert source[start1:end1] == "    "

    def test_blank_lines_excluded(self):
        source = "a\n\nb"
        result = ASTBoundaryMetrics._extract_line_indentation(source)
        assert len(result) == 2  # blank line excluded

    def test_tab_indentation(self):
        source = "\tif True:\n\t\tpass"
        result = ASTBoundaryMetrics._extract_line_indentation(source)
        assert result[0][0] == "\t"
        assert result[1][0] == "\t\t"

    def test_no_indentation(self):
        source = "x = 1\ny = 2"
        result = ASTBoundaryMetrics._extract_line_indentation(source)
        assert all(ws == "" for ws, _, _ in result)


# ======================================================================
# _infer_indent_unit
# ======================================================================

class TestInferIndentUnit:
    """Verify indentation unit detection."""

    def test_four_space_indent(self):
        indentation = [("", 0, 0), ("    ", 10, 14), ("        ", 25, 33)]
        assert ASTBoundaryMetrics._infer_indent_unit(indentation) == 4

    def test_two_space_indent(self):
        indentation = [("", 0, 0), ("  ", 5, 7), ("    ", 15, 19), ("      ", 25, 31)]
        assert ASTBoundaryMetrics._infer_indent_unit(indentation) == 2

    def test_tab_indent(self):
        # With expandtabs(), "\t" → 8 chars, "\t\t" → 16 chars, GCD = 8
        indentation = [("", 0, 0), ("\t", 5, 6), ("\t\t", 10, 12)]
        assert ASTBoundaryMetrics._infer_indent_unit(indentation) == 8

    def test_no_indentation(self):
        indentation = [("", 0, 0), ("", 5, 5)]
        assert ASTBoundaryMetrics._infer_indent_unit(indentation) == 1

    def test_empty_list(self):
        assert ASTBoundaryMetrics._infer_indent_unit([]) == 1

    def test_single_level(self):
        indentation = [("", 0, 0), ("    ", 10, 14)]
        assert ASTBoundaryMetrics._infer_indent_unit(indentation) == 4

    def test_mixed_widths_gcd(self):
        # 6 and 3 -> GCD = 3
        indentation = [("   ", 0, 3), ("      ", 10, 16)]
        assert ASTBoundaryMetrics._infer_indent_unit(indentation) == 3

    def test_mixed_tabs_and_spaces(self):
        """Tab (expands to 8) + 4 spaces → GCD should be 4, not 1."""
        indentation = [("\t", 0, 1), ("    ", 10, 14)]
        assert ASTBoundaryMetrics._infer_indent_unit(indentation) == 4

    def test_single_stray_tab(self):
        """One tab line among 4-space lines → GCD should be 4, not 1."""
        indentation = [
            ("    ", 0, 4),
            ("\t", 10, 11),
            ("        ", 20, 28),
        ]
        assert ASTBoundaryMetrics._infer_indent_unit(indentation) == 4


# ======================================================================
# _spearman_correlation
# ======================================================================

class TestSpearmanCorrelation:
    """Verify the Spearman rank correlation helper."""

    def test_perfect_positive(self):
        rho = ASTBoundaryMetrics._spearman_correlation(
            [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]
        )
        assert rho == pytest.approx(1.0)

    def test_perfect_negative(self):
        rho = ASTBoundaryMetrics._spearman_correlation(
            [1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]
        )
        assert rho == pytest.approx(-1.0)

    def test_no_correlation(self):
        # With only 2 points, any two distinct pairs have perfect correlation.
        # Use 4 scrambled points.
        rho = ASTBoundaryMetrics._spearman_correlation(
            [1.0, 2.0, 3.0, 4.0], [3.0, 1.0, 4.0, 2.0]
        )
        assert -1.0 <= rho <= 1.0

    def test_single_element_returns_zero(self):
        rho = ASTBoundaryMetrics._spearman_correlation([1.0], [2.0])
        assert rho == 0.0

    def test_empty_returns_zero(self):
        rho = ASTBoundaryMetrics._spearman_correlation([], [])
        assert rho == 0.0


# ======================================================================
# _count_identifier_tokens
# ======================================================================

class TestCountIdentifierTokens:
    """Verify identifier token counting."""

    def test_single_token_identifier(self):
        # source: "abc" => recon: "abc" => all token 0
        source_to_recon = [0, 1, 2]
        char_to_token = [0, 0, 0]
        result = ASTBoundaryMetrics._count_identifier_tokens(
            0, 3, source_to_recon, char_to_token
        )
        assert result == 1

    def test_multi_token_identifier(self):
        # source: "abc" => recon: "abc" => tokens [0,0,1]
        source_to_recon = [0, 1, 2]
        char_to_token = [0, 0, 1]
        result = ASTBoundaryMetrics._count_identifier_tokens(
            0, 3, source_to_recon, char_to_token
        )
        assert result == 2

    def test_unmappable_returns_none(self):
        source_to_recon = [None, None, None]
        char_to_token = [0, 0, 0]
        result = ASTBoundaryMetrics._count_identifier_tokens(
            0, 3, source_to_recon, char_to_token
        )
        assert result is None


# ======================================================================
# End-to-end: Identifier Fragmentation
# ======================================================================

class TestIdentifierFragmentationE2E:
    """End-to-end tests for identifier fragmentation metric."""

    @pytest.fixture(scope="class")
    def ts_pack(self):
        try:
            import tree_sitter_language_pack
            return tree_sitter_language_pack
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

    def test_char_tokenizer_high_fragmentation(self, ts_pack):
        """A character-level tokenizer should fragment most identifiers."""
        char_tok = _CharTokenizer()
        provider = _MockProvider("char_tok", char_tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["char_tok"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [
            'def fibonacci(n):\n    return n\n'
        ]}
        inst.code_loader = loader

        result = inst.compute()
        ident = result["identifier_fragmentation"]
        assert "per_tokenizer" in ident
        assert "char_tok" in ident["per_tokenizer"]

        overall = ident["per_tokenizer"]["char_tok"]["overall"]
        # "fibonacci" is 9 chars -> 9 tokens -> fragmented
        # "n" is 1 char -> 1 token -> not fragmented
        assert overall["fragmentation_rate"] > 0.0
        assert overall["count"] > 0

    def test_perfect_tokenizer_zero_fragmentation(self, ts_pack):
        """A tokenizer that keeps identifiers whole should have zero fragmentation."""
        snippet = 'x = 1'
        tokens = ["x", "Ġ=", "Ġ1"]
        tok = _PerfectTokenizer({snippet: tokens})
        provider = _MockProvider("perfect", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["perfect"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        ident = result["identifier_fragmentation"]
        overall = ident["per_tokenizer"]["perfect"]["overall"]
        assert overall["fragmentation_rate"] == pytest.approx(0.0)


# ======================================================================
# End-to-end: Indentation Consistency
# ======================================================================

class TestIndentationConsistencyE2E:
    """End-to-end tests for indentation consistency metric."""

    @pytest.fixture(scope="class")
    def ts_pack(self):
        try:
            import tree_sitter_language_pack
            return tree_sitter_language_pack
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

    def test_python_present(self, ts_pack):
        """Python snippets should produce indentation consistency data."""
        char_tok = _CharTokenizer()
        provider = _MockProvider("char_tok", char_tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["char_tok"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [
            'def f():\n    return 1\n    x = 2\n'
        ]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        assert "per_tokenizer" in indent
        tok_data = indent["per_tokenizer"]["char_tok"]
        assert "python" in tok_data["by_language"]
        py = tok_data["by_language"]["python"]
        assert py["total_indented_lines"] > 0
        assert py["num_depth_levels"] > 0

    def test_non_ws_lang_excluded(self, ts_pack):
        """Non-whitespace-significant languages should not appear."""
        char_tok = _CharTokenizer()
        provider = _MockProvider("char_tok", char_tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["char_tok"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"javascript": [
            'function f() {\n    return 1;\n}\n'
        ]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        tok_data = indent["per_tokenizer"]["char_tok"]
        # javascript is not whitespace-significant, so no by_language data
        assert "javascript" not in tok_data.get("by_language", {})

    def test_consistent_indentation_pattern_stability(self, ts_pack):
        """When indentation is uniform, pattern stability should be 1.0."""
        snippet = 'if True:\n    a = 1\n    b = 2\n    c = 3\n'
        # Each "    " (4 spaces) should produce the same token pattern
        tokens = [
            "if", "ĠTrue", "Ġ:", "\n",
            "Ġ   ", "a", "Ġ=", "Ġ1", "\n",
            "Ġ   ", "b", "Ġ=", "Ġ2", "\n",
            "Ġ   ", "c", "Ġ=", "Ġ3", "\n",
        ]
        tok = _PerfectTokenizer({snippet: tokens})
        provider = _MockProvider("perf", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["perf"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        py = indent["per_tokenizer"]["perf"]["by_language"]["python"]
        assert py["pattern_stability_rate"] == pytest.approx(1.0)

    def test_depth_corr_zero_for_constant_ws_tokens(self, ts_pack):
        """When the tokenizer produces exactly 1 ws token at every depth,
        num_ws_tokens is constant → Spearman ρ is 0.0."""
        snippet = (
            'if True:\n'
            '    a = 1\n'
            '    if True:\n'
            '        b = 2\n'
            '        if True:\n'
            '            c = 3\n'
        )
        # Single ws token per indented line regardless of width:
        #   "    " → "Ġ   " (1 token), "        " → "Ġ       " (1 token),
        #   "            " → "Ġ           " (1 token)
        tokens = [
            "if", "ĠTrue", "Ġ:", "\n",
            "Ġ   ", "a", "Ġ=", "Ġ1", "\n",
            "Ġ   ", "if", "ĠTrue", "Ġ:", "\n",
            "Ġ       ", "b", "Ġ=", "Ġ2", "\n",
            "Ġ       ", "if", "ĠTrue", "Ġ:", "\n",
            "Ġ           ", "c", "Ġ=", "Ġ3", "\n",
        ]
        tok = _PerfectTokenizer({snippet: tokens})
        provider = _MockProvider("const_ws", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["const_ws"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        py = indent["per_tokenizer"]["const_ws"]["by_language"]["python"]
        # Constant num_ws_tokens → correlation should be 0.0
        assert py["depth_proportionality_correlation"] == pytest.approx(0.0)

    def test_depth_corr_negative_for_inverse_tokenizer(self, ts_pack):
        """When deeper indentation uses *fewer* tokens, correlation is negative."""
        snippet = (
            'if True:\n'
            '    a = 1\n'
            '    if True:\n'
            '        b = 2\n'
            '        if True:\n'
            '            c = 3\n'
        )
        # Inverse: depth 1 (4 sp) → 3 ws tokens, depth 2 (8 sp) → 2 ws tokens,
        # depth 3 (12 sp) → 1 ws token
        tokens = [
            "if", "ĠTrue", "Ġ:", "\n",
            "Ġ", " ", " ", "a", "Ġ=", "Ġ1", "\n",
            "Ġ", " ", " ", "if", "ĠTrue", "Ġ:", "\n",
            "Ġ   ", "Ġ   ", "b", "Ġ=", "Ġ2", "\n",
            "Ġ   ", "Ġ   ", "if", "ĠTrue", "Ġ:", "\n",
            "Ġ           ", "c", "Ġ=", "Ġ3", "\n",
        ]
        tok = _PerfectTokenizer({snippet: tokens})
        provider = _MockProvider("inv", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["inv"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        py = indent["per_tokenizer"]["inv"]["by_language"]["python"]
        assert py["depth_proportionality_correlation"] is not None
        assert py["depth_proportionality_correlation"] < 0.0

    def test_depth_corr_none_with_fewer_than_3_depths(self, ts_pack):
        """With only 2 distinct depth levels, correlation should be None."""
        snippet = (
            'if True:\n'
            '    a = 1\n'
            '    b = 2\n'
        )
        tokens = [
            "if", "ĠTrue", "Ġ:", "\n",
            "Ġ   ", "a", "Ġ=", "Ġ1", "\n",
            "Ġ   ", "b", "Ġ=", "Ġ2", "\n",
        ]
        tok = _PerfectTokenizer({snippet: tokens})
        provider = _MockProvider("shallow", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["shallow"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        py = indent["per_tokenizer"]["shallow"]["by_language"]["python"]
        # Only 2 depth levels (0 is excluded since depth=0 lines have no ws)
        # → correlation should be None
        assert py["depth_proportionality_correlation"] is None

    def test_depth_proportionality_high_for_proportional_tokenizer(self, ts_pack):
        """A tokenizer that uses proportional tokens for deeper indentation
        should have a high depth-proportionality correlation."""
        # Snippet with 3 depth levels: depth 1 (4 spaces), depth 2 (8 spaces),
        # depth 3 (12 spaces)
        snippet = (
            'if True:\n'
            '    a = 1\n'
            '    if True:\n'
            '        b = 2\n'
            '        if True:\n'
            '            c = 3\n'
        )
        # Tokenizer: depth 1 = 1 ws token, depth 2 = 2 ws tokens, depth 3 = 3 ws tokens
        tokens = [
            "if", "ĠTrue", "Ġ:", "\n",
            "Ġ   ", "a", "Ġ=", "Ġ1", "\n",
            "Ġ   ", "if", "ĠTrue", "Ġ:", "\n",
            "Ġ   ", "Ġ   ", "b", "Ġ=", "Ġ2", "\n",
            "Ġ   ", "Ġ   ", "if", "ĠTrue", "Ġ:", "\n",
            "Ġ   ", "Ġ   ", "Ġ   ", "c", "Ġ=", "Ġ3", "\n",
        ]
        tok = _PerfectTokenizer({snippet: tokens})
        provider = _MockProvider("prop", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["prop"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        py = indent["per_tokenizer"]["prop"]["by_language"]["python"]
        assert py["depth_proportionality_correlation"] is not None
        assert py["depth_proportionality_correlation"] > 0.8


# ======================================================================
# _build_char_decode_table
# ======================================================================

class _MockBPETokenizer:
    """Mock tokenizer that remaps space→Ġ and newline→Ċ."""

    _REMAP = {' ': 'Ġ', '\n': 'Ċ', '\t': 'ĉ', '\r': 'č'}
    _REV = {v: k for k, v in _REMAP.items()}

    def encode(self, text):
        return list(range(len(text)))

    def convert_ids_to_tokens(self, ids):
        # Not a real tokenizer, but mimics BPE raw token output:
        # For probing "a a", return ["a", "Ġa"]
        # We'll just remap each character
        results = []
        for i in ids:
            # We don't have the original text, so return generic tokens
            results.append(f"tok_{i}")
        return results


class _MockBPETokenizerProbe:
    """Mock BPE tokenizer that properly responds to probe strings."""

    _REMAP = {' ': 'Ġ', '\n': 'Ċ', '\t': 'ĉ', '\r': 'č'}

    def encode(self, text):
        # Return one ID per character
        return list(range(len(text)))

    def convert_ids_to_tokens(self, ids):
        # For "a a": ids [0,1,2] → ["a", "Ġ", "a"]
        # For "a\na": ids [0,1,2] → ["a", "Ċ", "a"]
        # We need to know what text was encoded... simulate it
        return [f"tok{i}" for i in ids]


class _BPEStyleTokenizer:
    """BPE tokenizer that remaps whitespace in raw tokens."""

    _CHAR_MAP = {' ': 'Ġ', '\n': 'Ċ', '\t': 'ĉ', '\r': 'č'}

    def encode(self, text):
        return list(range(len(text)))

    def convert_ids_to_tokens(self, ids):
        # Store last encoded text for token conversion
        return [self._CHAR_MAP.get(chr(i), chr(i)) if i < 128 else f"<{i}>" for i in ids]

    def _encode_and_convert(self, text):
        """Helper: encode then convert."""
        ids = self.encode(text)
        mapped = []
        for ch in text:
            mapped.append(self._CHAR_MAP.get(ch, ch))
        return mapped


class _ProbeableBPETokenizer:
    """Tokenizer that responds correctly to the probing mechanism."""

    _CHAR_MAP = {' ': 'Ġ', '\n': 'Ċ', '\t': 'ĉ', '\r': 'č'}

    def __init__(self):
        self._last_text = None

    def encode(self, text):
        self._last_text = text
        return list(range(len(text)))

    def convert_ids_to_tokens(self, ids):
        if self._last_text is None:
            return [f"<{i}>" for i in ids]
        tokens = []
        for i, ch in enumerate(self._last_text):
            tokens.append(self._CHAR_MAP.get(ch, ch))
        return tokens


class _NoRemapTokenizer:
    """Tokenizer with no character remapping."""

    def encode(self, text):
        return list(range(len(text)))

    def convert_ids_to_tokens(self, ids):
        return [chr(i + 97) for i in ids]  # just 'a', 'b', 'c', ...


class TestMapFromOffsets:
    """Unit tests for _map_from_offsets static method."""

    def test_perfect_coverage(self):
        """Every character mapped when offsets tile the source exactly."""
        # 3 tokens: [0,3), [3,5), [5,8) covering 8 chars
        offsets = [(0, 3), (3, 5), (5, 8)]
        result = ASTBoundaryMetrics._map_from_offsets(8, offsets)
        assert result == [0, 0, 0, 1, 1, 2, 2, 2]

    def test_special_tokens_skipped(self):
        """(0,0) offsets (special tokens like <s>) leave chars as None."""
        offsets = [(0, 0), (0, 3), (3, 5), (0, 0)]
        result = ASTBoundaryMetrics._map_from_offsets(5, offsets)
        # token 0 is special, token 1 → [0..3), token 2 → [3..5), token 3 special
        assert result == [1, 1, 1, 2, 2]

    def test_gaps_are_none(self):
        """Characters not covered by any offset remain None."""
        offsets = [(0, 2), (4, 6)]
        result = ASTBoundaryMetrics._map_from_offsets(6, offsets)
        assert result == [0, 0, None, None, 1, 1]

    def test_overlapping_offsets_last_wins(self):
        """Overlapping offsets: later token overwrites earlier."""
        offsets = [(0, 5), (3, 7)]
        result = ASTBoundaryMetrics._map_from_offsets(7, offsets)
        assert result == [0, 0, 0, 1, 1, 1, 1]

    def test_empty_source(self):
        """Zero-length source returns empty list."""
        result = ASTBoundaryMetrics._map_from_offsets(0, [(0, 0)])
        assert result == []

    def test_offset_beyond_source_clamped(self):
        """Offsets extending past source_len are safely clamped."""
        offsets = [(0, 2), (2, 10)]
        result = ASTBoundaryMetrics._map_from_offsets(4, offsets)
        assert result == [0, 0, 1, 1]

    def test_multiline_bpe_offsets(self):
        """Offsets covering multi-line source with BPE-style tokens."""
        source = "if True:\n    x = 1\n"
        # Suppose: "if"→[0,2), " True"→[2,7), ":"→[7,8), "\n"→[8,9),
        #          "    "→[9,13), "x"→[13,14), " ="→[14,16), " 1"→[16,18), "\n"→[18,19)
        offsets = [
            (0, 2), (2, 7), (7, 8), (8, 9),
            (9, 13), (13, 14), (14, 16), (16, 18), (18, 19),
        ]
        result = ASTBoundaryMetrics._map_from_offsets(len(source), offsets)
        assert len(result) == len(source)
        # No None gaps
        assert None not in result
        # Whitespace region [9..13) all maps to token 4
        assert result[9:13] == [4, 4, 4, 4]
        # Newline at position 8 maps to token 3
        assert result[8] == 3


class TestWhitespaceStrippingTokenizer:
    """E2E: tokenizers that strip whitespace from tokens (custom BPE)
    should produce num_ws_tokens=0 and never-negative correlation."""

    @pytest.fixture(scope="class")
    def ts_pack(self):
        try:
            import tree_sitter_language_pack
            return tree_sitter_language_pack
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

    def test_ws_stripping_no_negative_correlation(self, ts_pack):
        """Whitespace-stripping tokenizer: offsets skip whitespace → 0 ws tokens,
        correlation is None (not enough data) rather than negative."""
        snippet = (
            'if True:\n'
            '    a = 1\n'
            '    if True:\n'
            '        b = 2\n'
        )
        # Whitespace-stripping tokenizer: whitespace not in any token's source coverage
        # Offsets skip over whitespace positions entirely
        tokens = ["if", "True", ":", "a", "=", "1", "if", "True", ":", "b", "=", "2"]
        # Build offsets that skip whitespace
        offsets = []
        src = snippet
        pos = 0
        for tok in tokens:
            idx = src.find(tok, pos)
            offsets.append((idx, idx + len(tok)))
            pos = idx + len(tok)

        class _WSStrippingTokenizer:
            def __init__(self, toks, offs):
                self._toks = toks
                self._offs = offs
                self._tok_to_id = {t: i for i, t in enumerate(set(toks))}
                self._id_to_tok = {v: k for k, v in self._tok_to_id.items()}

            def can_encode(self):
                return True

            def encode(self, text):
                return [self._tok_to_id[t] for t in self._toks]

            def encode_with_offsets(self, text):
                return self.encode(text), list(self._offs)

            def convert_ids_to_tokens(self, ids):
                return [self._id_to_tok[i] for i in ids]

            def get_vocab(self):
                return dict(self._tok_to_id)

        tok = _WSStrippingTokenizer(tokens, offsets)
        provider = _MockProvider("ws_strip", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["ws_strip"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        py = indent["per_tokenizer"]["ws_strip"]["by_language"]["python"]
        # All whitespace positions are None in the map → num_ws_tokens=0
        # With constant 0 ws tokens, correlation should be None or 0, never negative
        corr = py["depth_proportionality_correlation"]
        if corr is not None:
            assert corr >= 0.0, (
                f"Whitespace-stripping tokenizer should never have negative "
                f"correlation, got {corr}"
            )


class TestBuildCharDecodeTable:
    """Test _build_char_decode_table probing."""

    def test_detects_bpe_remapping(self):
        """A BPE-style tokenizer with Ġ→space, Ċ→newline should be detected."""
        tok = _ProbeableBPETokenizer()
        table = ASTBoundaryMetrics._build_char_decode_table(tok)
        assert table.get('Ġ') == ' '
        assert table.get('Ċ') == '\n'

    def test_no_remap_returns_empty(self):
        """Tokenizer without character remapping → empty table."""
        tok = _NoRemapTokenizer()
        table = ASTBoundaryMetrics._build_char_decode_table(tok)
        assert table == {}

    def test_no_encode_returns_empty(self):
        """Tokenizer without encode() method → empty table."""
        table = ASTBoundaryMetrics._build_char_decode_table(object())
        assert table == {}


# ======================================================================
# End-to-end: Indentation Consistency with BPE-encoded whitespace
# ======================================================================

class TestIndentationConsistencyBPE:
    """E2E tests for indentation with ĠĠĠĠ/Ċ encoding (byte-level BPE)."""

    @pytest.fixture(scope="class")
    def ts_pack(self):
        try:
            import tree_sitter_language_pack
            return tree_sitter_language_pack
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

    def test_proportional_bpe_high_correlation(self, ts_pack):
        """Proportional tokenizer with ĠĠĠĠ/Ċ encoding → ρ > 0.8."""
        snippet = (
            'if True:\n'
            '    a = 1\n'
            '    if True:\n'
            '        b = 2\n'
            '        if True:\n'
            '            c = 3\n'
        )
        # BPE: Ċ for newlines, ĠĠĠĠ for 4-space blocks
        # depth 1 = 1 ws token, depth 2 = 2 ws tokens, depth 3 = 3 ws tokens
        tokens = [
            "if", "ĠTrue", "Ġ:", "Ċ",
            "ĠĠĠĠ", "a", "Ġ=", "Ġ1", "Ċ",
            "ĠĠĠĠ", "if", "ĠTrue", "Ġ:", "Ċ",
            "ĠĠĠĠ", "ĠĠĠĠ", "b", "Ġ=", "Ġ2", "Ċ",
            "ĠĠĠĠ", "ĠĠĠĠ", "if", "ĠTrue", "Ġ:", "Ċ",
            "ĠĠĠĠ", "ĠĠĠĠ", "ĠĠĠĠ", "c", "Ġ=", "Ġ3", "Ċ",
        ]
        tok = _PerfectTokenizer({snippet: tokens})
        provider = _MockProvider("bpe", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["bpe"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        py = indent["per_tokenizer"]["bpe"]["by_language"]["python"]
        assert py["depth_proportionality_correlation"] is not None
        assert py["depth_proportionality_correlation"] > 0.8

    def test_uniform_bpe_stability(self, ts_pack):
        """Uniform indentation with ĠĠĠĠ/Ċ → stability = 1.0."""
        snippet = 'if True:\n    a = 1\n    b = 2\n    c = 3\n'
        tokens = [
            "if", "ĠTrue", "Ġ:", "Ċ",
            "ĠĠĠĠ", "a", "Ġ=", "Ġ1", "Ċ",
            "ĠĠĠĠ", "b", "Ġ=", "Ġ2", "Ċ",
            "ĠĠĠĠ", "c", "Ġ=", "Ġ3", "Ċ",
        ]
        tok = _PerfectTokenizer({snippet: tokens})
        provider = _MockProvider("bpe_uni", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["bpe_uni"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        py = indent["per_tokenizer"]["bpe_uni"]["by_language"]["python"]
        assert py["pattern_stability_rate"] == pytest.approx(1.0)

    def test_constant_ws_tokens_zero_corr(self, ts_pack):
        """Constant ws tokens across depths with ĠĠĠĠ/Ċ → ρ = 0.0."""
        snippet = (
            'if True:\n'
            '    a = 1\n'
            '    if True:\n'
            '        b = 2\n'
            '        if True:\n'
            '            c = 3\n'
        )
        # 1 ws token at every depth (each indentation is a single big token)
        tokens = [
            "if", "ĠTrue", "Ġ:", "Ċ",
            "ĠĠĠĠ", "a", "Ġ=", "Ġ1", "Ċ",
            "ĠĠĠĠ", "if", "ĠTrue", "Ġ:", "Ċ",
            "ĠĠĠĠĠĠĠĠ", "b", "Ġ=", "Ġ2", "Ċ",
            "ĠĠĠĠĠĠĠĠ", "if", "ĠTrue", "Ġ:", "Ċ",
            "ĠĠĠĠĠĠĠĠĠĠĠĠ", "c", "Ġ=", "Ġ3", "Ċ",
        ]
        tok = _PerfectTokenizer({snippet: tokens})
        provider = _MockProvider("bpe_const", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["bpe_const"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        py = indent["per_tokenizer"]["bpe_const"]["by_language"]["python"]
        assert py["depth_proportionality_correlation"] == pytest.approx(0.0)


# ======================================================================
# End-to-end: Indentation Consistency with SentencePiece encoding
# ======================================================================

class TestIndentationConsistencySP:
    """E2E tests for indentation with ▁▁▁▁ / newline encoding (SentencePiece)."""

    @pytest.fixture(scope="class")
    def ts_pack(self):
        try:
            import tree_sitter_language_pack
            return tree_sitter_language_pack
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

    def test_proportional_sp_high_correlation(self, ts_pack):
        """Proportional tokenizer with ▁▁▁▁/newline encoding → ρ > 0.8."""
        snippet = (
            'if True:\n'
            '    a = 1\n'
            '    if True:\n'
            '        b = 2\n'
            '        if True:\n'
            '            c = 3\n'
        )
        # SentencePiece: ▁ for space, literal \n
        tokens = [
            "if", "▁True", "▁:", "\n",
            "▁▁▁▁", "a", "▁=", "▁1", "\n",
            "▁▁▁▁", "if", "▁True", "▁:", "\n",
            "▁▁▁▁", "▁▁▁▁", "b", "▁=", "▁2", "\n",
            "▁▁▁▁", "▁▁▁▁", "if", "▁True", "▁:", "\n",
            "▁▁▁▁", "▁▁▁▁", "▁▁▁▁", "c", "▁=", "▁3", "\n",
        ]
        tok = _PerfectTokenizer({snippet: tokens})
        provider = _MockProvider("sp", tok)

        inst = object.__new__(ASTBoundaryMetrics)
        inst._tokenizer_vocab_cache = {}
        inst._warned_tokenizers = set()
        inst._char_decode_table = None
        inst._treesitter_available = True
        inst._ts_pack = ts_pack
        inst._parser_cache = {}
        inst.input_provider = provider
        inst.tokenizer_names = ["sp"]
        inst.max_snippets_per_lang = 1

        loader = CodeDataLoader()
        loader.code_snippets = {"python": [snippet]}
        inst.code_loader = loader

        result = inst.compute()
        indent = result["indentation_consistency"]
        py = indent["per_tokenizer"]["sp"]["by_language"]["python"]
        assert py["depth_proportionality_correlation"] is not None
        assert py["depth_proportionality_correlation"] > 0.8


# ======================================================================
# print_results for new metrics
# ======================================================================

class TestPrintNewMetrics:

    def test_print_fragmentation(self, capsys):
        inst = _make_instance()
        inst.tokenizer_names = ["test_tok"]
        results = {
            "ast_boundary_alignment": {
                "per_tokenizer": {
                    "test_tok": {
                        "by_category": {},
                        "by_language": {},
                        "overall": {},
                    }
                },
                "summary": {},
            },
            "identifier_fragmentation": {
                "per_tokenizer": {
                    "test_tok": {
                        "by_language": {
                            "python": {
                                "fragmentation_rate": 0.75,
                                "avg_tokens_per_identifier": 3.2,
                                "count": 100,
                            }
                        },
                        "overall": {
                            "fragmentation_rate": 0.75,
                            "avg_tokens_per_identifier": 3.2,
                            "count": 100,
                        },
                    }
                },
                "summary": {
                    "test_tok": {
                        "fragmentation_rate": 0.75,
                        "avg_tokens_per_identifier": 3.2,
                        "identifiers_analyzed": 100,
                        "languages_analyzed": 1,
                    }
                },
            },
            "indentation_consistency": {"per_tokenizer": {}, "summary": {}},
        }
        inst.print_results(results)
        captured = capsys.readouterr()
        assert "IDENTIFIER FRAGMENTATION" in captured.out
        assert "0.750" in captured.out
        assert "3.20" in captured.out
        assert "python" in captured.out

    def test_print_indentation(self, capsys):
        inst = _make_instance()
        inst.tokenizer_names = ["test_tok"]
        results = {
            "ast_boundary_alignment": {
                "per_tokenizer": {
                    "test_tok": {
                        "by_category": {},
                        "by_language": {},
                        "overall": {},
                    }
                },
                "summary": {},
            },
            "identifier_fragmentation": {"per_tokenizer": {}, "summary": {}},
            "indentation_consistency": {
                "per_tokenizer": {
                    "test_tok": {
                        "by_language": {
                            "python": {
                                "depth_proportionality_correlation": 0.9,
                                "pattern_stability_rate": 0.95,
                                "num_depth_levels": 3,
                                "total_indented_lines": 20,
                            }
                        },
                    }
                },
                "summary": {
                    "test_tok": {
                        "avg_depth_proportionality_correlation": 0.9,
                        "avg_pattern_stability_rate": 0.95,
                        "languages_analyzed": 1,
                    }
                },
            },
        }
        inst.print_results(results)
        captured = capsys.readouterr()
        assert "INDENTATION CONSISTENCY" in captured.out
        assert "0.900" in captured.out
        assert "0.950" in captured.out
        assert "python" in captured.out


# ======================================================================
# Subprocess worker isolation
# ======================================================================

class TestSubprocessWorker:
    """Verify that tree-sitter parsing via the subprocess worker produces
    correct results and completes within a reasonable time.

    These tests exercise the actual subprocess path (``_treesitter_worker.py``
    invoked via ``subprocess.run``) so they catch pickle, import, and
    timeout regressions without running a full tokenizer analysis.
    """

    @pytest.fixture(scope="class")
    def ts_available(self):
        try:
            import tree_sitter_language_pack  # noqa: F401
            return True
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

    @staticmethod
    def _run_worker(code_snippets, lang_to_ts, timeout=30):
        """Helper: run the tree-sitter worker subprocess using temp files."""
        import os, pickle, subprocess, sys, tempfile

        worker_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "metrics", "_treesitter_worker.py",
        )

        tmp_in = tmp_out = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pkl", prefix="ts_test_in_", delete=False
            ) as f_in:
                tmp_in = f_in.name
                pickle.dump((code_snippets, lang_to_ts), f_in)

            with tempfile.NamedTemporaryFile(
                suffix=".pkl", prefix="ts_test_out_", delete=False
            ) as f_out:
                tmp_out = f_out.name

            proc = subprocess.run(
                [sys.executable, worker_path, tmp_in, tmp_out],
                capture_output=True,
                timeout=timeout,
            )
            assert proc.returncode == 0, (
                f"Worker failed: {proc.stderr.decode(errors='replace')}"
            )

            with open(tmp_out, "rb") as f:
                return pickle.load(f)
        finally:
            for p in (tmp_in, tmp_out):
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    def test_subprocess_roundtrip(self, ts_available):
        """Spawn the worker, send a small snippet, verify parsed output.

        The subprocess now returns spans only (no snippet text,
        byte_to_char, or indentation).
        """
        code_snippets = {
            "python": ["def add(a, b):\n    return a + b\n"],
        }
        lang_to_ts = CodeDataLoader._LANG_TO_TREESITTER

        parsed = self._run_worker(code_snippets, lang_to_ts)
        assert "python" in parsed
        assert len(parsed["python"]) == 1

        # Each entry is now just a categorized_spans dict (not a 4-tuple)
        spans = parsed["python"][0]
        assert isinstance(spans, dict)

        # spans should contain at least identifiers and keywords
        assert len(spans["identifier"]) > 0, "Expected identifier spans"
        assert len(spans["keyword"]) > 0, "Expected keyword spans"

        # Each span is a (start_byte, end_byte) tuple
        for cat, cat_spans in spans.items():
            for span in cat_spans:
                assert len(span) == 2
                assert span[0] < span[1]

    def test_subprocess_non_ws_lang(self, ts_available):
        """Non-whitespace-significant languages should still return spans."""
        code_snippets = {
            "javascript": ["function add(a, b) { return a + b; }"],
        }
        lang_to_ts = CodeDataLoader._LANG_TO_TREESITTER

        parsed = self._run_worker(code_snippets, lang_to_ts)
        spans = parsed["javascript"][0]
        assert isinstance(spans, dict)
        # Should have parsed some identifiers and keywords
        assert len(spans["identifier"]) > 0
        assert len(spans["keyword"]) > 0

    def test_subprocess_unknown_language_skipped(self, ts_available):
        """Languages without a tree-sitter mapping should be silently skipped."""
        code_snippets = {
            "brainfuck": ["+++++[>+++<-]>."],
        }
        lang_to_ts = CodeDataLoader._LANG_TO_TREESITTER

        parsed = self._run_worker(code_snippets, lang_to_ts)
        assert "brainfuck" not in parsed

    def test_subprocess_matches_in_process(self, ts_available):
        """Subprocess result should be identical to in-process result."""
        from tokenizer_analysis.metrics._treesitter_worker import parse_snippets

        samples = CodeDataLoader.generate_synthetic_samples()
        # Use one snippet per language to keep it fast
        code_snippets = {lang: snips[:1] for lang, snips in samples.items()}
        lang_to_ts = CodeDataLoader._LANG_TO_TREESITTER

        # In-process
        in_proc = parse_snippets(code_snippets, lang_to_ts)

        # Subprocess
        sub_proc = self._run_worker(code_snippets, lang_to_ts, timeout=60)

        # Same languages parsed
        assert set(in_proc.keys()) == set(sub_proc.keys())

        for lang in in_proc:
            assert len(in_proc[lang]) == len(sub_proc[lang]), f"Mismatch for {lang}"
            for i, (ip_spans, sp_spans) in enumerate(zip(in_proc[lang], sub_proc[lang])):
                assert ip_spans == sp_spans, f"Span mismatch: {lang} snippet {i}"


class TestPerSnippetTimeout:
    """Verify that the thread-based per-snippet timeout in
    ``_parse_one_snippet`` correctly handles hanging parsers and that
    ``parse_snippets`` emits aligned empty spans for timed-out snippets
    while still producing results for healthy ones.
    """

    def test_parse_one_snippet_timeout_returns_none(self):
        """A parser whose .parse() blocks indefinitely should be timed out."""
        import time
        from unittest.mock import MagicMock
        from tokenizer_analysis.metrics._treesitter_worker import _parse_one_snippet

        # Create a mock parser whose parse() sleeps far longer than the
        # timeout.  The daemon thread will be left behind but is harmless.
        mock_parser = MagicMock()
        mock_parser.parse.side_effect = lambda _src: time.sleep(30)

        start = time.monotonic()
        result = _parse_one_snippet(mock_parser, "x = 1", timeout=0.5)
        elapsed = time.monotonic() - start

        assert result is None, "Expected None for a timed-out snippet"
        assert elapsed < 5, f"Should have returned quickly after timeout, took {elapsed:.1f}s"

    def test_parse_one_snippet_success(self):
        """A well-behaved parser should return spans normally."""
        try:
            import tree_sitter_language_pack  # noqa: F401
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

        import tree_sitter_language_pack as ts_pack
        from tokenizer_analysis.metrics._treesitter_worker import (
            _parse_one_snippet,
            CATEGORIES,
        )

        parser = ts_pack.get_parser("python")
        result = _parse_one_snippet(parser, "x = 1\n", timeout=10)

        assert result is not None, "Expected spans for a simple snippet"
        assert isinstance(result, dict)
        for cat in CATEGORIES:
            assert cat in result

    def test_parse_one_snippet_exception_returns_none(self):
        """If parse() raises, _parse_one_snippet returns None (not crash)."""
        from unittest.mock import MagicMock
        from tokenizer_analysis.metrics._treesitter_worker import _parse_one_snippet

        mock_parser = MagicMock()
        mock_parser.parse.side_effect = RuntimeError("boom")

        result = _parse_one_snippet(mock_parser, "x = 1", timeout=5)
        assert result is None

    def test_parse_snippets_mixed_healthy_and_hanging(self):
        """parse_snippets with a hanging snippet should skip it (empty spans)
        while still returning real spans for the healthy snippet.

        Index alignment between input and output lists must be preserved.
        """
        import time
        from unittest.mock import patch, MagicMock
        from tokenizer_analysis.metrics._treesitter_worker import (
            parse_snippets,
            CATEGORIES,
        )

        try:
            import tree_sitter_language_pack  # noqa: F401
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

        good_snippet = "def foo():\n    return 42\n"
        bad_snippet = "x = 1\n"  # content doesn't matter — we'll make it hang

        code_snippets = {"python": [good_snippet, bad_snippet, good_snippet]}
        lang_to_ts = CodeDataLoader._LANG_TO_TREESITTER

        # Patch _parse_one_snippet: let good_snippet through normally,
        # simulate a timeout (return None) for bad_snippet.
        original_parse_one = None
        import tokenizer_analysis.metrics._treesitter_worker as worker_mod
        original_parse_one = worker_mod._parse_one_snippet

        call_count = [0]

        def patched_parse_one(parser, snippet, timeout):
            call_count[0] += 1
            if snippet is bad_snippet:
                return None  # simulate timeout
            return original_parse_one(parser, snippet, timeout)

        with patch.object(worker_mod, "_parse_one_snippet", side_effect=patched_parse_one):
            start = time.monotonic()
            result = parse_snippets(code_snippets, lang_to_ts, per_snippet_timeout=5)
            elapsed = time.monotonic() - start

        assert "python" in result
        spans_list = result["python"]

        # Must have exactly 3 entries — one per input snippet
        assert len(spans_list) == 3, f"Expected 3 entries, got {len(spans_list)}"

        # First and third (good) should have real spans
        for idx in (0, 2):
            assert any(
                len(spans_list[idx].get(cat, [])) > 0 for cat in CATEGORIES
            ), f"Snippet {idx} should have real AST spans"

        # Second (bad) should have all-empty spans
        for cat in CATEGORIES:
            assert spans_list[1].get(cat) == [], (
                f"Timed-out snippet should have empty '{cat}' spans"
            )

        # Should complete quickly (no actual 30s sleep)
        assert elapsed < 10, f"Mixed parse took too long: {elapsed:.1f}s"

    def test_subprocess_per_snippet_timeout(self):
        """End-to-end: the subprocess worker skips a genuinely slow snippet
        without stalling the entire language.

        We send two snippets — one normal and one enormous — and verify
        that the worker completes within its per-snippet timeout window
        and that both entries are present (the slow one with empty spans).
        """
        try:
            import tree_sitter_language_pack  # noqa: F401
        except ImportError:
            pytest.skip("tree-sitter-language-pack not installed")

        import os, pickle, subprocess, sys, tempfile

        worker_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "metrics", "_treesitter_worker.py",
        )

        good_snippet = "def foo():\n    return 42\n"
        # Create a deeply nested snippet that may stress the parser.
        # Even if tree-sitter handles it fast, the test still validates
        # that the output list stays aligned.
        stress_snippet = "(((" * 2000 + ")))" * 2000

        code_snippets = {"python": [good_snippet, stress_snippet]}
        lang_to_ts = CodeDataLoader._LANG_TO_TREESITTER

        tmp_in = tmp_out = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pkl", prefix="ts_test_in_", delete=False
            ) as f_in:
                tmp_in = f_in.name
                pickle.dump((code_snippets, lang_to_ts), f_in)

            with tempfile.NamedTemporaryFile(
                suffix=".pkl", prefix="ts_test_out_", delete=False
            ) as f_out:
                tmp_out = f_out.name

            proc = subprocess.run(
                [sys.executable, worker_path, tmp_in, tmp_out],
                capture_output=True,
                timeout=60,
            )
            assert proc.returncode == 0, (
                f"Worker failed: {proc.stderr.decode(errors='replace')}"
            )

            with open(tmp_out, "rb") as f:
                parsed = pickle.load(f)
        finally:
            for p in (tmp_in, tmp_out):
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

        assert "python" in parsed
        spans_list = parsed["python"]

        # Both entries must be present to preserve index alignment
        assert len(spans_list) == 2, (
            f"Expected 2 entries (good + stress), got {len(spans_list)}"
        )

        # First snippet (good) should have real AST spans
        good_spans = spans_list[0]
        assert isinstance(good_spans, dict)
        assert any(
            len(good_spans.get(cat, [])) > 0
            for cat in ("identifier", "keyword", "literal")
        ), "Good snippet should produce AST spans"

        # Second snippet (stress) — whether it parsed or timed out,
        # it must be a valid spans dict with all categories present
        stress_spans = spans_list[1]
        assert isinstance(stress_spans, dict)
        for cat in ("identifier", "keyword", "operator", "literal", "delimiter"):
            assert cat in stress_spans, f"Missing category '{cat}' in stress result"


class TestFastMethodsParity:
    """Verify that the numpy-accelerated ``_fast`` methods produce identical
    results to the original Python-list-based methods across a range of
    inputs including edge cases.
    """

    @staticmethod
    def _make_arrays(source_to_recon, char_to_token):
        import numpy as np
        s2r_arr = np.array(
            [x if x is not None else -1 for x in source_to_recon],
            dtype=np.int64,
        )
        c2t_arr = np.array(char_to_token, dtype=np.int64)
        return s2r_arr, c2t_arr, len(char_to_token)

    def test_alignment_fast_matches_original(self):
        """_check_boundary_alignment_fast matches _check_boundary_alignment."""
        source_to_recon = [0, None, 1, 2, 3, None, 4, 5]
        char_to_token = [0, 0, 1, 1, 2, 2]
        s2r_arr, c2t_arr, c2t_len = self._make_arrays(
            source_to_recon, char_to_token
        )

        test_spans = [
            (0, 3), (2, 5), (0, 8), (1, 2), (6, 8),
            (0, 1), (7, 8),  # single-char spans
            (5, 6),          # unmapped span (only None)
            (0, 0),          # empty span
            (10, 12),        # out of bounds
        ]
        for c_start, c_end in test_spans:
            original = ASTBoundaryMetrics._check_boundary_alignment(
                c_start, c_end, source_to_recon, char_to_token
            )
            fast = ASTBoundaryMetrics._check_boundary_alignment_fast(
                c_start, c_end, s2r_arr, c2t_arr, c2t_len
            )
            assert original == fast, (
                f"Alignment mismatch for span ({c_start}, {c_end}): "
                f"original={original}, fast={fast}"
            )

    def test_identifier_tokens_fast_matches_original(self):
        """_count_identifier_tokens_fast matches _count_identifier_tokens."""
        source_to_recon = [0, None, 1, 2, 3, None, 4, 5]
        char_to_token = [0, 0, 1, 1, 2, 2]
        s2r_arr, c2t_arr, c2t_len = self._make_arrays(
            source_to_recon, char_to_token
        )

        test_spans = [
            (0, 3), (2, 5), (0, 8), (1, 2), (6, 8),
            (0, 1), (7, 8),
            (5, 6),
            (0, 0),
            (10, 12),
        ]
        for c_start, c_end in test_spans:
            original = ASTBoundaryMetrics._count_identifier_tokens(
                c_start, c_end, source_to_recon, char_to_token
            )
            fast = ASTBoundaryMetrics._count_identifier_tokens_fast(
                c_start, c_end, s2r_arr, c2t_arr, c2t_len
            )
            assert original == fast, (
                f"Token count mismatch for span ({c_start}, {c_end}): "
                f"original={original}, fast={fast}"
            )

    def test_fast_methods_identity_mapping(self):
        """Parity on a simple identity mapping (no Nones, no gaps)."""
        source_to_recon = list(range(10))
        char_to_token = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]
        s2r_arr, c2t_arr, c2t_len = self._make_arrays(
            source_to_recon, char_to_token
        )

        for c_start in range(10):
            for c_end in range(c_start, 11):
                orig_align = ASTBoundaryMetrics._check_boundary_alignment(
                    c_start, c_end, source_to_recon, char_to_token
                )
                fast_align = ASTBoundaryMetrics._check_boundary_alignment_fast(
                    c_start, c_end, s2r_arr, c2t_arr, c2t_len
                )
                assert orig_align == fast_align, (
                    f"Alignment mismatch at ({c_start}, {c_end})"
                )

                orig_count = ASTBoundaryMetrics._count_identifier_tokens(
                    c_start, c_end, source_to_recon, char_to_token
                )
                fast_count = ASTBoundaryMetrics._count_identifier_tokens_fast(
                    c_start, c_end, s2r_arr, c2t_arr, c2t_len
                )
                assert orig_count == fast_count, (
                    f"Token count mismatch at ({c_start}, {c_end})"
                )

    def test_fast_methods_all_none_mapping(self):
        """Both methods return None when source_to_recon is all None."""
        source_to_recon = [None, None, None, None]
        char_to_token = [0, 1, 2]
        s2r_arr, c2t_arr, c2t_len = self._make_arrays(
            source_to_recon, char_to_token
        )

        for c_start, c_end in [(0, 4), (0, 2), (1, 3)]:
            assert ASTBoundaryMetrics._check_boundary_alignment_fast(
                c_start, c_end, s2r_arr, c2t_arr, c2t_len
            ) is None
            assert ASTBoundaryMetrics._count_identifier_tokens_fast(
                c_start, c_end, s2r_arr, c2t_arr, c2t_len
            ) is None
