"""
AST Boundary Alignment metrics for code tokenization evaluation.

Parses source code into an AST using tree-sitter, then measures the fraction
of AST node boundaries (identifiers, keywords, operators, literals,
delimiters) that coincide with token boundaries produced by the tokenizer.

Five AST node categories are tracked independently:

1. **Identifiers** -- variable names, function names, class names, etc.
2. **Keywords** -- language keywords (``if``, ``for``, ``return``, ...).
3. **Operators** -- ``+``, ``==``, ``&&``, etc.
4. **Literals** -- string, numeric, and boolean literals.
5. **Delimiters** -- ``(``, ``)``, ``{``, ``}``, ``[``, ``]``, ``;``, ``,``.
"""

import math
import os
import pickle
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import logging

from .base import BaseMetrics
from ..core.input_providers import InputProvider

# Import constants and helpers from the lightweight worker module.
# The worker is kept free of heavy imports (BaseMetrics, InputProvider, etc.)
# so it can be spawned as a subprocess without pulling in the tokenizer stack.
from ._treesitter_worker import (
    CATEGORIES as _CATEGORIES_TUPLE,
    IDENTIFIER_TYPES as _IDENTIFIER_TYPES,
    LITERAL_TYPES as _LITERAL_TYPES,
    DELIMITER_CHARS as _DELIMITER_CHARS,
    NON_OPERATOR_PUNCTUATION as _NON_OPERATOR_PUNCTUATION,
    KNOWN_KEYWORDS as _KNOWN_KEYWORDS,
    classify_node as _classify_node_fn,
    extract_leaf_spans as _extract_leaf_spans_fn,
    parse_snippets as _parse_snippets,
)

_WHITESPACE_SIGNIFICANT_LANGS: Set[str] = {"python", "haskell"}

logger = logging.getLogger(__name__)


class ASTBoundaryMetrics(BaseMetrics):
    """AST boundary alignment metrics for code tokenization.

    This metric loads its own code data (from config paths or synthetic
    samples) and encodes it with each tokenizer.  It does **not** use the
    ``tokenized_data`` parameter passed to :meth:`compute`.
    """

    _CATEGORIES = _CATEGORIES_TUPLE

    # Timeout (seconds) for each per-language tree-sitter subprocess.
    _PER_LANG_TIMEOUT = 120

    def __init__(
        self,
        input_provider: InputProvider,
        code_config: Optional[Dict[str, str]] = None,
        max_snippets_per_lang: Optional[int] = None,
    ):
        super().__init__(input_provider)

        # Tree-sitter availability (lazy)
        self._treesitter_available: Optional[bool] = None

        # Load code data
        from ..loaders.code_data import CodeDataLoader

        self.code_loader = CodeDataLoader(
            code_config, max_snippets_per_lang=max_snippets_per_lang
        )
        self.max_snippets_per_lang = self.code_loader.max_snippets_per_lang

        if code_config:
            self.code_loader.load_all()

        # If no data was loaded from config, use synthetic samples
        if not self.code_loader.code_snippets:
            synthetic = CodeDataLoader.generate_synthetic_samples()
            for lang, snippets in synthetic.items():
                self.code_loader.code_snippets.setdefault(lang, []).extend(snippets)

    # ------------------------------------------------------------------
    # Tree-sitter helpers
    # ------------------------------------------------------------------

    def _ensure_treesitter(self) -> bool:
        """Lazily import tree-sitter.  Returns ``True`` if available."""
        if self._treesitter_available is not None:
            return self._treesitter_available
        try:
            import tree_sitter_language_pack as _ts_pack  # noqa: F401
            self._treesitter_available = True
        except ImportError:
            logger.warning(
                "tree-sitter-language-pack not installed. "
                "AST boundary metrics disabled. "
                "Install with: pip install tree-sitter-language-pack"
            )
            self._treesitter_available = False
        return self._treesitter_available

    # ------------------------------------------------------------------
    # AST node classification & span extraction
    # Delegated to _treesitter_worker (single source of truth).
    # ------------------------------------------------------------------

    _classify_node = staticmethod(_classify_node_fn)
    _extract_leaf_spans = staticmethod(_extract_leaf_spans_fn)

    @staticmethod
    def _byte_to_char_offsets(source_bytes: bytes) -> List[int]:
        """Map each byte offset to a character offset.

        For pure-ASCII text this is the identity mapping.  Returns a list
        of length ``len(source_bytes) + 1`` so that ``end_byte`` lookups
        (exclusive) work without special-casing.
        """
        result: List[int] = []
        source_str = source_bytes.decode("utf-8")
        char_idx = 0
        for ch in source_str:
            n_bytes = len(ch.encode("utf-8"))
            for _ in range(n_bytes):
                result.append(char_idx)
            char_idx += 1
        # Sentinel for exclusive end positions
        result.append(char_idx)
        return result

    # ------------------------------------------------------------------
    # Source → reconstructed-text coordinate mapping
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Identifier token counting
    # ------------------------------------------------------------------

    @staticmethod
    def _count_identifier_tokens(
        char_start: int,
        char_end: int,
        source_to_recon: List[Optional[int]],
        char_to_token: List[int],
    ) -> Optional[int]:
        """Count the number of distinct token indices spanning a source character range.

        Uses the existing ``source_to_recon`` + ``char_to_token`` coordinate
        chain.  Returns ``None`` if the span cannot be mapped.
        """
        recon_positions = []
        for pos in range(char_start, min(char_end, len(source_to_recon))):
            rp = source_to_recon[pos]
            if rp is not None:
                recon_positions.append(rp)

        if not recon_positions:
            return None

        recon_start = min(recon_positions)
        recon_end = max(recon_positions) + 1  # exclusive

        if recon_end > len(char_to_token):
            return None

        return len(set(char_to_token[recon_start:recon_end]))

    # ------------------------------------------------------------------
    # Whitespace-preserving token decoding
    # ------------------------------------------------------------------

    def _decode_raw_token(self, raw_token: str) -> Optional[str]:
        """Decode a raw token string preserving whitespace.

        Unlike ``_clean_token`` (which strips space prefixes entirely),
        this replaces Ġ / ▁ / leading-space markers with a literal space.
        Returns ``None`` for special tokens.

        Delegates to :meth:`BaseMetrics._process_token` with
        ``preserve_space=True`` so that the branch logic stays in sync
        with ``_clean_token``.
        """
        return self._process_token(raw_token, preserve_space=True)

    # ------------------------------------------------------------------
    # Source char → token map (whitespace-inclusive)
    # ------------------------------------------------------------------

    @staticmethod
    def _map_from_offsets(
        source_len: int,
        offsets: List[Tuple[int, int]],
    ) -> List[Optional[int]]:
        """Build a source-char → token-index map from encoding offsets.

        Each entry in *offsets* is ``(start_char, end_char)`` in the
        original source text for the corresponding token.  Special tokens
        with ``(0, 0)`` are skipped.

        Returns a list of length *source_len* where entry *i* is the
        token index covering that position, or ``None``.
        """
        result: List[Optional[int]] = [None] * source_len
        for tok_idx, (start, end) in enumerate(offsets):
            if start == end:
                # Special token (e.g. <s>, </s>) — no source coverage
                continue
            for pos in range(start, min(end, source_len)):
                result[pos] = tok_idx
        return result

    def _map_from_greedy_decode(
        self, source_code: str, token_strings: List[str],
    ) -> List[Optional[int]]:
        """Fallback: greedy character-by-character alignment.

        Decodes each raw token via :meth:`_decode_raw_token` and greedily
        matches decoded characters against *source_code*.  Allows
        space ↔ tab equivalence but NOT space ↔ newline.
        """
        result: List[Optional[int]] = [None] * len(source_code)
        src_idx = 0

        for tok_idx, raw_token in enumerate(token_strings):
            decoded = self._decode_raw_token(raw_token)
            if decoded is None:
                continue
            for ch in decoded:
                if src_idx >= len(source_code):
                    break
                if source_code[src_idx] == ch:
                    result[src_idx] = tok_idx
                    src_idx += 1
                elif (
                    ch in (' ', '\t') and source_code[src_idx] in (' ', '\t')
                ):
                    result[src_idx] = tok_idx
                    src_idx += 1
                # else: skip character in decoded token (mismatch)

        return result

    def _build_source_char_to_token_map(
        self,
        source_code: str,
        token_strings: List[str],
        offsets: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Optional[int]]:
        """Map each source character (including whitespace) to a token index.

        When *offsets* are provided (from ``encode_with_offsets``), uses
        the direct offset-based mapping which is exact and cannot
        desynchronise.  Otherwise falls back to greedy character decoding.

        Returns a list of length ``len(source_code)`` where entry *i* is
        the token index covering source char *i*, or ``None``.
        """
        if offsets is not None:
            return self._map_from_offsets(len(source_code), offsets)
        return self._map_from_greedy_decode(source_code, token_strings)

    @staticmethod
    def _infer_indent_unit(
        indentation: List[Tuple[str, int, int]],
    ) -> int:
        """Detect the indentation unit size (in characters) for a snippet.

        Collects all non-zero whitespace widths and returns their GCD.
        Falls back to 1 if no indented lines exist or GCD is 0.
        """
        from math import gcd
        widths = [len(ws.expandtabs()) for ws, _, _ in indentation if ws]
        if not widths:
            return 1
        result = widths[0]
        for w in widths[1:]:
            result = gcd(result, w)
        return result if result > 0 else 1

    @staticmethod
    def _extract_line_indentation(
        source_code: str,
    ) -> List[Tuple[str, int, int]]:
        """Extract leading whitespace info for each non-blank line.

        Returns a list of ``(ws_string, line_char_start, ws_char_end)``
        tuples.  Blank / whitespace-only lines are excluded.  Lines with
        no indentation return ``ws_string=""``.
        """
        results: List[Tuple[str, int, int]] = []
        offset = 0
        for line in source_code.split("\n"):
            if line.strip():  # non-blank
                stripped = line.lstrip()
                ws_len = len(line) - len(stripped)
                ws_string = line[:ws_len]
                results.append((ws_string, offset, offset + ws_len))
            offset += len(line) + 1  # +1 for the newline
        return results

    # ------------------------------------------------------------------
    # Boundary alignment check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_boundary_alignment(
        char_start: int,
        char_end: int,
        source_to_recon: List[Optional[int]],
        char_to_token: List[int],
    ) -> Optional[Dict[str, bool]]:
        """Check whether an AST node's boundaries align with token boundaries.

        Parameters use *source-code* character coordinates following the
        half-open ``[start, end)`` convention (mirrors tree-sitter):

        * *char_start* — inclusive start in source-code character coordinates
        * *char_end* — exclusive end in source-code character coordinates

        Positions are translated to the reconstructed-text coordinate space
        via *source_to_recon* before checking against *char_to_token*.

        Returns ``None`` if the span cannot be mapped.
        """
        # Map start position
        recon_start = None
        for pos in range(char_start, min(char_end, len(source_to_recon))):
            if source_to_recon[pos] is not None:
                recon_start = source_to_recon[pos]
                break

        # Map end position (find last mapped char before char_end)
        recon_end = None
        for pos in range(min(char_end - 1, len(source_to_recon) - 1), char_start - 1, -1):
            if pos >= 0 and pos < len(source_to_recon) and source_to_recon[pos] is not None:
                recon_end = source_to_recon[pos] + 1  # exclusive
                break

        if recon_start is None or recon_end is None or recon_start >= recon_end:
            return None
        if recon_end > len(char_to_token):
            return None

        # Start boundary: token changes at recon_start compared to previous
        start_aligned = (
            recon_start == 0
            or char_to_token[recon_start] != char_to_token[recon_start - 1]
        )

        # End boundary: token changes at recon_end compared to previous
        end_aligned = (
            recon_end >= len(char_to_token)
            or char_to_token[recon_end - 1] != char_to_token[recon_end]
        )

        fully_aligned = start_aligned and end_aligned
        cross_boundary = not fully_aligned

        return {
            "start_aligned": start_aligned,
            "end_aligned": end_aligned,
            "fully_aligned": fully_aligned,
            "cross_boundary": cross_boundary,
        }

    # ------------------------------------------------------------------
    # Numpy-accelerated helpers (used by compute() hot loop)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_boundary_alignment_fast(
        char_start: int,
        char_end: int,
        s2r_arr: np.ndarray,
        c2t_arr: np.ndarray,
        c2t_len: int,
    ) -> Optional[Dict[str, bool]]:
        """Fast boundary alignment using pre-built numpy arrays.

        Same semantics as :meth:`_check_boundary_alignment` but operates
        on numpy ``int64`` arrays (``-1`` for unmapped positions) to avoid
        repeated Python list-indexing overhead.
        """
        s2r_len = len(s2r_arr)

        # Forward scan for recon_start
        hi = min(char_end, s2r_len)
        if char_start >= hi:
            return None
        segment = s2r_arr[char_start:hi]
        mapped = segment >= 0
        if not mapped.any():
            return None
        first_offset = int(mapped.argmax())
        recon_start = int(segment[first_offset])

        # Backward scan for recon_end (exclusive)
        last_offset = len(segment) - 1 - int(np.flip(mapped).argmax())
        recon_end = int(segment[last_offset]) + 1

        if recon_start >= recon_end or recon_end > c2t_len:
            return None

        start_aligned = (
            recon_start == 0
            or int(c2t_arr[recon_start]) != int(c2t_arr[recon_start - 1])
        )
        end_aligned = (
            recon_end >= c2t_len
            or int(c2t_arr[recon_end - 1]) != int(c2t_arr[recon_end])
        )
        fully_aligned = start_aligned and end_aligned

        return {
            "start_aligned": start_aligned,
            "end_aligned": end_aligned,
            "fully_aligned": fully_aligned,
            "cross_boundary": not fully_aligned,
        }

    @staticmethod
    def _count_identifier_tokens_fast(
        char_start: int,
        char_end: int,
        s2r_arr: np.ndarray,
        c2t_arr: np.ndarray,
        c2t_len: int,
    ) -> Optional[int]:
        """Fast identifier token counting using pre-built numpy arrays.

        Same semantics as :meth:`_count_identifier_tokens`.
        """
        s2r_len = len(s2r_arr)
        hi = min(char_end, s2r_len)
        if char_start >= hi:
            return None

        segment = s2r_arr[char_start:hi]
        mapped = segment[segment >= 0]

        if len(mapped) == 0:
            return None

        recon_start = int(mapped.min())
        recon_end = int(mapped.max()) + 1  # exclusive

        if recon_end > c2t_len:
            return None

        return int(np.unique(c2t_arr[recon_start:recon_end]).size)

    # ------------------------------------------------------------------
    # Main compute
    # ------------------------------------------------------------------

    def compute(
        self, tokenized_data=None
    ) -> Dict[str, Any]:
        """Compute AST boundary alignment metrics.

        .. note::

           *tokenized_data* is **not used** — the metric loads its own code
           snippets and encodes them with each tokenizer.
        """
        if not self._ensure_treesitter():
            return {
                "ast_boundary_alignment": {
                    "error": "tree-sitter-language-pack not installed",
                },
                "identifier_fragmentation": {},
                "indentation_consistency": {},
            }

        if not self.code_loader.code_snippets:
            return {
                "ast_boundary_alignment": {
                    "error": "No code data loaded",
                },
                "identifier_fragmentation": {},
                "indentation_consistency": {},
            }

        # acc: tok_name -> code_lang -> category -> list of alignment dicts
        acc: Dict[str, Dict[str, Dict[str, List[Dict]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        # ident_acc: tok -> lang -> [{text, num_tokens, fragmented}]
        ident_acc: Dict[str, Dict[str, List[Dict]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # indent_acc: tok -> lang -> [per-line records]
        # Each record: {"depth": int, "num_ws_tokens": int,
        #               "pattern": Tuple[str,...], "ws_width": int}
        indent_acc: Dict[str, Dict[str, List[Dict]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # ----- Phase 1: parse all snippets with tree-sitter -----
        # Tree-sitter uses a C backend.  On some platforms, earlier metrics
        # (DigitBoundaryMetrics, MorphScore, etc.) call tokenizer.encode()
        # through a Rust/C backend that corrupts heap metadata.  By the
        # time AST metrics run, the heap is already corrupted and the first
        # tree-sitter malloc triggers a crash.
        #
        # Fix: run tree-sitter work in *subprocesses* with their own clean
        # heaps.  One subprocess is spawned **per language** so that a
        # pathological snippet in one language cannot stall all others.
        # Each subprocess gets a moderate timeout; languages that exceed
        # it are skipped with a warning.
        #
        # The subprocess returns ONLY categorized byte-offset spans
        # (the one thing that requires tree-sitter's C library).
        # byte_to_char mapping and indentation extraction are pure Python
        # and are computed here in the main process.
        #
        # Structure: parsed_spans[code_lang] = [categorized_spans_dict, ...]

        from ..loaders.code_data import CodeDataLoader

        code_snippets = {
            lang: self.code_loader.get_code_snippets(lang)
            for lang in self.code_loader.get_languages()
        }
        lang_to_treesitter = CodeDataLoader._LANG_TO_TREESITTER

        total_input = sum(len(v) for v in code_snippets.values())
        logger.info(
            "Phase 1: parsing %d snippet(s) across %d language(s) via "
            "per-language subprocesses (timeout=%ds each).",
            total_input, len(code_snippets), self._PER_LANG_TIMEOUT,
        )

        worker_path = os.path.join(os.path.dirname(__file__), "_treesitter_worker.py")

        # Log diagnostic info so subprocess failures can be debugged remotely.
        logger.info(
            "Subprocess config: python=%s, worker=%s, worker_exists=%s",
            sys.executable, worker_path, os.path.isfile(worker_path),
        )

        parsed_spans: Dict[str, list] = {}

        for lang, snippets in code_snippets.items():
            if not snippets:
                continue
            ts_name = lang_to_treesitter.get(lang)
            if ts_name is None:
                logger.debug("No tree-sitter grammar for %s; skipping.", lang)
                continue

            tmp_in = tmp_out = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".pkl", prefix=f"ts_in_{lang}_", delete=False
                ) as f_in:
                    tmp_in = f_in.name
                    pickle.dump(({lang: snippets}, lang_to_treesitter), f_in)

                with tempfile.NamedTemporaryFile(
                    suffix=".pkl", prefix=f"ts_out_{lang}_", delete=False
                ) as f_out:
                    tmp_out = f_out.name

                logger.debug(
                    "  %s: launching subprocess: %s %s %s %s",
                    lang, sys.executable, worker_path, tmp_in, tmp_out,
                )

                proc = subprocess.run(
                    [sys.executable, worker_path, tmp_in, tmp_out],
                    capture_output=True,
                    timeout=self._PER_LANG_TIMEOUT,
                )

                stderr_msg = proc.stderr.decode(errors="replace").strip()

                if proc.returncode != 0:
                    # Log full stderr and the signal number for crash diagnosis.
                    rc = proc.returncode
                    if rc < 0:
                        import signal as _signal
                        try:
                            sig_name = _signal.Signals(-rc).name
                        except (ValueError, AttributeError):
                            sig_name = f"signal {-rc}"
                        logger.error(
                            "  %s: subprocess killed by %s (return code %d). "
                            "This typically indicates a malloc/heap corruption "
                            "crash (SIGSEGV/SIGABRT) in tree-sitter's C backend. "
                            "stderr:\n%s",
                            lang, sig_name, rc, stderr_msg or "(empty)",
                        )
                    else:
                        logger.error(
                            "  %s: subprocess exited with code %d. stderr:\n%s",
                            lang, rc, stderr_msg or "(empty)",
                        )
                    logger.error(
                        "  %s: NOT falling back to in-process parsing to "
                        "avoid heap corruption in the main process.",
                        lang,
                    )
                    continue

                if stderr_msg:
                    logger.info("Tree-sitter worker [%s]: %s", lang, stderr_msg)

                if not os.path.exists(tmp_out) or os.path.getsize(tmp_out) == 0:
                    logger.error(
                        "  %s: subprocess exited successfully but output "
                        "file is missing or empty (%s); skipping language.",
                        lang, tmp_out,
                    )
                    continue

                with open(tmp_out, "rb") as f:
                    lang_result = pickle.load(f)
                parsed_spans.update(lang_result)
                logger.info(
                    "  %s: parsed %d snippet(s) in subprocess.",
                    lang, len(snippets),
                )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "  %s: subprocess timed out after %ds for %d snippet(s); "
                    "skipping language.",
                    lang, self._PER_LANG_TIMEOUT, len(snippets),
                )
            except Exception as e:
                logger.error(
                    "  %s: subprocess failed with %s: %s.  "
                    "NOT falling back to in-process parsing to avoid "
                    "heap corruption.  Skipping language.",
                    lang, type(e).__name__, e,
                )
            finally:
                for tmp_path in (tmp_in, tmp_out):
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

        total_parsed = sum(len(v) for v in parsed_spans.values())
        # Count snippets that produced at least one AST span (non-empty parse).
        non_empty_parsed = sum(
            1
            for spans_list in parsed_spans.values()
            for spans_dict in spans_list
            if any(spans_dict.get(cat) for cat in self._CATEGORIES)
        )
        for lang in sorted(parsed_spans):
            lang_total = len(parsed_spans[lang])
            lang_non_empty = sum(
                1 for sd in parsed_spans[lang]
                if any(sd.get(cat) for cat in self._CATEGORIES)
            )
            logger.info(
                "  %s: %d snippet(s) loaded, %d with AST spans "
                "(%.0f%% non-empty)",
                lang, lang_total, lang_non_empty,
                100 * lang_non_empty / lang_total if lang_total else 0,
            )
        logger.info(
            "Phase 1 complete: parsed %d snippet(s) across %d language(s) "
            "(%d with AST spans). Starting Phase 2 (tokenizer encoding + alignment).",
            total_parsed, len(parsed_spans), non_empty_parsed,
        )

        # ----- Phase 2: encode with tokenizers and measure alignment -----
        # Loop order: (language → snippet → tokenizer) so that
        # tokenizer-independent per-snippet work (byte_to_char, indentation,
        # byte→char span conversion) is computed ONCE and reused across all
        # tokenizers.

        # Pre-filter encodable tokenizers (avoid repeated can_encode() checks).
        active_tokenizers: List[Tuple[str, Any]] = []
        for tok_name in self.tokenizer_names:
            tokenizer = self.input_provider.get_tokenizer(tok_name)
            if tokenizer.can_encode():
                active_tokenizers.append((tok_name, tokenizer))
            else:
                logger.info(
                    "Tokenizer %s cannot encode text; skipping AST metrics",
                    tok_name,
                )

        # Pre-build character decode tables for byte-level BPE / SP tokenizers.
        decode_tables = {n: self._build_char_decode_table(t) for n, t in active_tokenizers}

        for code_lang, spans_list in parsed_spans.items():
            snippets = code_snippets[code_lang]
            is_ws_significant = code_lang in _WHITESPACE_SIGNIFICANT_LANGS

            for si, categorized_spans in enumerate(spans_list):
                snippet = snippets[si]

                # -- Tokenizer-independent pre-computation (ONCE per snippet) --
                source_bytes = snippet.encode("utf-8")
                byte_to_char = self._byte_to_char_offsets(source_bytes)

                indentation = (
                    self._extract_line_indentation(snippet)
                    if is_ws_significant
                    else None
                )
                indent_unit = (
                    self._infer_indent_unit(indentation)
                    if indentation is not None
                    else None
                )

                # Pre-convert ALL byte spans to char spans.
                char_spans_by_category: Dict[str, List[Tuple[int, int]]] = {}
                for category, spans in categorized_spans.items():
                    char_spans: List[Tuple[int, int]] = []
                    for byte_start, byte_end in spans:
                        if byte_start >= len(byte_to_char) or byte_end >= len(byte_to_char):
                            continue
                        char_spans.append(
                            (byte_to_char[byte_start], byte_to_char[byte_end])
                        )
                    char_spans_by_category[category] = char_spans

                # Pre-extract identifier texts.
                ident_char_spans = char_spans_by_category.get("identifier", [])
                ident_texts = [
                    snippet[c_start:c_end]
                    for c_start, c_end in ident_char_spans
                ]

                logger.debug(
                    "Phase 2 pre-computed: lang=%s snippet=%d/%d",
                    code_lang, si + 1, len(spans_list),
                )

                # -- Per-tokenizer work --
                for tok_name, tokenizer in active_tokenizers:
                    self._char_decode_table = decode_tables[tok_name]
                    try:
                        token_ids, enc_offsets = tokenizer.encode_with_offsets(
                            snippet
                        )
                    except Exception as e:
                        logger.debug(
                            "Encoding failed for %s on %s snippet: %s",
                            tok_name, code_lang, e,
                        )
                        continue
                    if not token_ids:
                        continue

                    token_strings = self._convert_ids_to_tokens(
                        tokenizer, token_ids
                    )
                    recon_text, char_to_token = self._build_char_to_token_map(
                        token_strings
                    )
                    if not char_to_token:
                        continue

                    source_to_recon = self._build_source_to_recon_map(
                        snippet, recon_text
                    )

                    # Build numpy arrays ONCE per (snippet, tokenizer).
                    s2r_arr = np.array(
                        [x if x is not None else -1 for x in source_to_recon],
                        dtype=np.int64,
                    )
                    c2t_arr = np.array(char_to_token, dtype=np.int64)
                    c2t_len = len(char_to_token)

                    for category, char_spans in char_spans_by_category.items():
                        for span_idx, (c_start, c_end) in enumerate(char_spans):
                            alignment = self._check_boundary_alignment_fast(
                                c_start, c_end, s2r_arr, c2t_arr, c2t_len
                            )
                            if alignment is None:
                                alignment = {
                                    "start_aligned": False,
                                    "end_aligned": False,
                                    "fully_aligned": False,
                                    "cross_boundary": True,
                                }
                            acc[tok_name][code_lang][category].append(alignment)

                            # Identifier fragmentation tracking
                            if category == "identifier":
                                num_tokens = self._count_identifier_tokens_fast(
                                    c_start, c_end, s2r_arr, c2t_arr, c2t_len
                                )
                                ident_acc[tok_name][code_lang].append({
                                    "text": ident_texts[span_idx],
                                    "num_tokens": num_tokens if num_tokens is not None else -1,
                                    "fragmented": num_tokens is None or num_tokens > 1,
                                })

                    # Indentation consistency (whitespace-significant languages)
                    if indentation is not None:
                        source_char_to_token = (
                            self._build_source_char_to_token_map(
                                snippet, token_strings,
                                offsets=enc_offsets,
                            )
                        )
                        for ws_string, line_start, ws_end in indentation:
                            if not ws_string:
                                continue
                            token_indices: List[int] = []
                            for pos in range(line_start, ws_end):
                                if pos < len(source_char_to_token):
                                    tidx = source_char_to_token[pos]
                                    if tidx is not None and (
                                        not token_indices
                                        or token_indices[-1] != tidx
                                    ):
                                        token_indices.append(tidx)
                            pattern = tuple(
                                token_strings[ti] for ti in token_indices
                            )
                            ws_width = len(ws_string.expandtabs())
                            depth = ws_width // indent_unit if indent_unit else ws_width
                            num_ws_tokens = len(token_indices)
                            indent_acc[tok_name][code_lang].append({
                                "depth": depth,
                                "num_ws_tokens": num_ws_tokens,
                                "pattern": pattern,
                                "ws_width": ws_width,
                            })

        self._char_decode_table = None

        # Log Phase 2 summary: how many AST nodes contributed per tokenizer
        for tok_name in self.tokenizer_names:
            tok_node_count = sum(
                len(items)
                for lang_cats in acc.get(tok_name, {}).values()
                for items in lang_cats.values()
            )
            tok_lang_count = len(acc.get(tok_name, {}))
            logger.info(
                "Phase 2 complete for %s: %d AST nodes aligned across %d language(s).",
                tok_name, tok_node_count, tok_lang_count,
            )

        return {
            "ast_boundary_alignment": self._build_results(acc),
            "identifier_fragmentation": self._build_identifier_fragmentation_results(ident_acc),
            "indentation_consistency": self._build_indentation_consistency_results(indent_acc),
        }

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_results(
        self, acc: Dict[str, Dict[str, Dict[str, List[Dict]]]]
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {"per_tokenizer": {}, "summary": {}}

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {
                "by_category": {},
                "by_language": {},
                "overall": {},
            }

            all_full: List[float] = []
            all_start: List[float] = []
            all_end: List[float] = []
            all_cross: List[float] = []
            total_count = 0
            languages_seen: set = set()

            for code_lang in sorted(acc.get(tok_name, {})):
                lang_full: List[float] = []
                lang_start: List[float] = []
                lang_end: List[float] = []
                lang_cross: List[float] = []

                for category in sorted(acc[tok_name][code_lang]):
                    items = acc[tok_name][code_lang][category]
                    if not items:
                        continue

                    s_rates = [1.0 if it["start_aligned"] else 0.0 for it in items]
                    e_rates = [1.0 if it["end_aligned"] else 0.0 for it in items]
                    f_rates = [1.0 if it["fully_aligned"] else 0.0 for it in items]
                    c_rates = [1.0 if it["cross_boundary"] else 0.0 for it in items]

                    if category not in tok_data["by_category"]:
                        tok_data["by_category"][category] = {}

                    tok_data["by_category"][category][code_lang] = {
                        "start_alignment_rate": float(np.mean(s_rates)),
                        "end_alignment_rate": float(np.mean(e_rates)),
                        "full_alignment_rate": float(np.mean(f_rates)),
                        "cross_boundary_rate": float(np.mean(c_rates)),
                        "count": len(items),
                    }

                    lang_full.extend(f_rates)
                    lang_start.extend(s_rates)
                    lang_end.extend(e_rates)
                    lang_cross.extend(c_rates)

                if lang_full:
                    tok_data["by_language"][code_lang] = {
                        "overall_full_alignment_rate": float(np.mean(lang_full)),
                        "overall_start_alignment_rate": float(np.mean(lang_start)),
                        "overall_end_alignment_rate": float(np.mean(lang_end)),
                        "overall_cross_boundary_rate": float(np.mean(lang_cross)),
                        "count": len(lang_full),
                    }
                    all_full.extend(lang_full)
                    all_start.extend(lang_start)
                    all_end.extend(lang_end)
                    all_cross.extend(lang_cross)
                    total_count += len(lang_full)
                    languages_seen.add(code_lang)

            if all_full:
                tok_data["overall"] = {
                    "full_alignment_rate": float(np.mean(all_full)),
                    "start_alignment_rate": float(np.mean(all_start)),
                    "end_alignment_rate": float(np.mean(all_end)),
                    "cross_boundary_rate": float(np.mean(all_cross)),
                    "count": total_count,
                }

            results["per_tokenizer"][tok_name] = tok_data

            if all_full:
                results["summary"][tok_name] = {
                    "avg_full_alignment_rate": float(np.mean(all_full)),
                    "avg_start_alignment_rate": float(np.mean(all_start)),
                    "avg_end_alignment_rate": float(np.mean(all_end)),
                    "avg_cross_boundary_rate": float(np.mean(all_cross)),
                    "total_nodes_analyzed": total_count,
                    "languages_analyzed": len(languages_seen),
                }

        return results

    def _build_identifier_fragmentation_results(
        self, ident_acc: Dict[str, Dict[str, List[Dict]]]
    ) -> Dict[str, Any]:
        """Build identifier fragmentation results from accumulated data."""
        results: Dict[str, Any] = {"per_tokenizer": {}, "summary": {}}

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {"by_language": {}, "overall": {}}
            all_items: List[Dict] = []
            languages_seen: set = set()

            for code_lang in sorted(ident_acc.get(tok_name, {})):
                items = ident_acc[tok_name][code_lang]
                if not items:
                    continue

                frag_rate = sum(1 for it in items if it["fragmented"]) / len(items)
                avg_tokens = sum(it["num_tokens"] for it in items) / len(items)

                tok_data["by_language"][code_lang] = {
                    "fragmentation_rate": float(frag_rate),
                    "avg_tokens_per_identifier": float(avg_tokens),
                    "count": len(items),
                }
                all_items.extend(items)
                languages_seen.add(code_lang)

            if all_items:
                overall_frag = sum(1 for it in all_items if it["fragmented"]) / len(all_items)
                overall_avg = sum(it["num_tokens"] for it in all_items) / len(all_items)
                tok_data["overall"] = {
                    "fragmentation_rate": float(overall_frag),
                    "avg_tokens_per_identifier": float(overall_avg),
                    "count": len(all_items),
                }

            results["per_tokenizer"][tok_name] = tok_data

            if all_items:
                results["summary"][tok_name] = {
                    "fragmentation_rate": float(overall_frag),
                    "avg_tokens_per_identifier": float(overall_avg),
                    "identifiers_analyzed": len(all_items),
                    "languages_analyzed": len(languages_seen),
                }

        return results

    @staticmethod
    def _spearman_correlation(x: List[float], y: List[float]) -> float:
        """Compute Spearman rank correlation between two lists.

        Uses scipy.stats.spearmanr if available, otherwise falls back to a
        pure-Python implementation.  Returns 0.0 if the correlation is
        undefined (e.g. constant input).
        """
        n = len(x)
        if n < 2:
            return 0.0
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(x, y)
            if math.isnan(rho):
                return 0.0
            return float(rho)
        except ImportError:
            pass

        # Pure-Python rank correlation
        def _rank(vals):
            indexed = sorted(range(n), key=lambda i: vals[i])
            ranks = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    ranks[indexed[k]] = avg_rank
                i = j + 1
            return ranks

        rx = _rank(x)
        ry = _rank(y)
        d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
        denom = n * (n * n - 1)
        if denom == 0:
            return 0.0
        rho = 1.0 - 6.0 * d_sq / denom
        return rho

    def _build_indentation_consistency_results(
        self, indent_acc: Dict[str, Dict[str, List[Dict]]]
    ) -> Dict[str, Any]:
        """Build indentation consistency results from accumulated data.

        Computes two metrics per language per tokenizer:
        - depth_proportionality_correlation: Spearman ρ between logical
          nesting depth and number of whitespace tokens.
        - pattern_stability_rate: weighted fraction of lines at each depth
          that share the dominant tokenization pattern.
        """
        results: Dict[str, Any] = {"per_tokenizer": {}, "summary": {}}

        for tok_name in self.tokenizer_names:
            tok_data: Dict[str, Any] = {"by_language": {}}
            lang_correlations: List[float] = []
            lang_stabilities: List[float] = []
            languages_seen: set = set()

            for code_lang in sorted(indent_acc.get(tok_name, {})):
                records = indent_acc[tok_name][code_lang]
                if not records:
                    continue

                total_indented_lines = len(records)
                depths = [r["depth"] for r in records]
                num_ws_tokens = [r["num_ws_tokens"] for r in records]
                distinct_depths = len(set(depths))

                # Depth-proportionality correlation (Spearman ρ)
                if distinct_depths >= 3:
                    corr = self._spearman_correlation(
                        [float(d) for d in depths],
                        [float(t) for t in num_ws_tokens],
                    )
                else:
                    corr = float("nan")

                # Pattern stability rate
                depth_groups: Dict[int, List[Tuple]] = defaultdict(list)
                for r in records:
                    depth_groups[r["depth"]].append(r["pattern"])

                dominant_total = 0
                for d, patterns in depth_groups.items():
                    counter = Counter(patterns)
                    dominant_total += counter.most_common(1)[0][1]

                stability = dominant_total / total_indented_lines if total_indented_lines else 0.0

                tok_data["by_language"][code_lang] = {
                    "depth_proportionality_correlation": float(corr) if not math.isnan(corr) else None,
                    "pattern_stability_rate": float(stability),
                    "num_depth_levels": distinct_depths,
                    "total_indented_lines": total_indented_lines,
                }

                if not math.isnan(corr):
                    lang_correlations.append(corr)
                lang_stabilities.append(stability)
                languages_seen.add(code_lang)

            results["per_tokenizer"][tok_name] = tok_data

            if languages_seen:
                summary: Dict[str, Any] = {
                    "avg_pattern_stability_rate": float(np.mean(lang_stabilities)),
                    "languages_analyzed": len(languages_seen),
                }
                if lang_correlations:
                    summary["avg_depth_proportionality_correlation"] = float(
                        np.mean(lang_correlations)
                    )
                results["summary"][tok_name] = summary

        return results

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print AST boundary alignment results."""
        ast = results.get("ast_boundary_alignment")
        if not ast:
            return

        if "error" in ast:
            print(f"\nAST BOUNDARY ALIGNMENT: {ast['error']}")
            return

        print("\n" + "=" * 60)
        print("AST BOUNDARY ALIGNMENT RESULTS")
        print("=" * 60)

        # Summary
        if "summary" in ast:
            print("\nSUMMARY STATISTICS")
            print("-" * 40)
            for tok_name in self.tokenizer_names:
                if tok_name in ast["summary"]:
                    s = ast["summary"][tok_name]
                    print(f"{tok_name}:")
                    print(f"  {'Full Alignment':25}: {s['avg_full_alignment_rate']:.3f}")
                    print(f"  {'Start Alignment':25}: {s['avg_start_alignment_rate']:.3f}")
                    print(f"  {'End Alignment':25}: {s['avg_end_alignment_rate']:.3f}")
                    print(f"  {'Cross-Boundary Rate':25}: {s['avg_cross_boundary_rate']:.3f}")
                    print(f"  {'Nodes Analyzed':25}: {s['total_nodes_analyzed']:,}")
                    print(f"  {'Languages':25}: {s['languages_analyzed']}")

        # By category
        if "per_tokenizer" in ast:
            print("\nBY CATEGORY")
            print("-" * 60)
            for tok_name in self.tokenizer_names:
                tok = ast["per_tokenizer"].get(tok_name, {})
                by_cat = tok.get("by_category", {})
                if not by_cat:
                    continue
                print(f"\n{tok_name}:")
                for category in self._CATEGORIES:
                    if category not in by_cat:
                        continue
                    lang_data = by_cat[category]
                    total_items = sum(d["count"] for d in lang_data.values())
                    if total_items == 0:
                        continue
                    weighted_full = sum(
                        d["full_alignment_rate"] * d["count"]
                        for d in lang_data.values()
                    ) / total_items
                    print(
                        f"  {category:15}: "
                        f"full_align={weighted_full:.3f}  "
                        f"n={total_items}"
                    )

            # By language
            print("\nBY LANGUAGE")
            print("-" * 60)
            for tok_name in self.tokenizer_names:
                tok = ast["per_tokenizer"].get(tok_name, {})
                by_lang = tok.get("by_language", {})
                if not by_lang:
                    continue
                print(f"\n{tok_name}:")
                for lang in sorted(by_lang):
                    d = by_lang[lang]
                    print(
                        f"  {lang:15}: "
                        f"full_align={d['overall_full_alignment_rate']:.3f}  "
                        f"start={d['overall_start_alignment_rate']:.3f}  "
                        f"end={d['overall_end_alignment_rate']:.3f}  "
                        f"n={d['count']}"
                    )

        # --- Identifier Fragmentation ---
        ident = results.get("identifier_fragmentation")
        if ident and "summary" in ident and ident["summary"]:
            print("\nIDENTIFIER FRAGMENTATION")
            print("-" * 60)
            for tok_name in self.tokenizer_names:
                if tok_name in ident.get("summary", {}):
                    s = ident["summary"][tok_name]
                    print(f"{tok_name}:")
                    print(f"  {'Fragmentation Rate':25}: {s['fragmentation_rate']:.3f}")
                    print(f"  {'Avg Tokens/Identifier':25}: {s['avg_tokens_per_identifier']:.2f}")
                    print(f"  {'Identifiers Analyzed':25}: {s['identifiers_analyzed']:,}")
                    print(f"  {'Languages':25}: {s['languages_analyzed']}")

                    # By language breakdown
                    tok_detail = ident.get("per_tokenizer", {}).get(tok_name, {})
                    by_lang = tok_detail.get("by_language", {})
                    if by_lang:
                        for lang in sorted(by_lang):
                            d = by_lang[lang]
                            print(
                                f"    {lang:13}: "
                                f"frag_rate={d['fragmentation_rate']:.3f}  "
                                f"avg_tok={d['avg_tokens_per_identifier']:.2f}  "
                                f"n={d['count']}"
                            )

        # --- Indentation Consistency ---
        indent = results.get("indentation_consistency")
        if indent and "summary" in indent and indent["summary"]:
            print("\nINDENTATION CONSISTENCY")
            print("-" * 60)
            for tok_name in self.tokenizer_names:
                if tok_name in indent.get("summary", {}):
                    s = indent["summary"][tok_name]
                    print(f"{tok_name}:")
                    if "avg_depth_proportionality_correlation" in s:
                        print(f"  {'Avg Depth Correlation':25}: {s['avg_depth_proportionality_correlation']:.3f}")
                    print(f"  {'Avg Pattern Stability':25}: {s['avg_pattern_stability_rate']:.3f}")
                    print(f"  {'Languages':25}: {s['languages_analyzed']}")

                    tok_detail = indent.get("per_tokenizer", {}).get(tok_name, {})
                    by_lang = tok_detail.get("by_language", {})
                    if by_lang:
                        for lang in sorted(by_lang):
                            d = by_lang[lang]
                            corr = d.get("depth_proportionality_correlation")
                            corr_str = f"{corr:.3f}" if corr is not None else "N/A"
                            print(
                                f"    {lang:13}: "
                                f"depth_corr={corr_str}  "
                                f"stability={d['pattern_stability_rate']:.3f}  "
                                f"levels={d['num_depth_levels']}  "
                                f"lines={d['total_indented_lines']}"
                            )

        print("\n" + "=" * 60)
