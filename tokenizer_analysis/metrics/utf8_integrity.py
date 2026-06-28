"""
UTF-8 character boundary integrity metrics.

Two complementary metrics detect when byte-level tokenizers split
multi-byte UTF-8 characters across token boundaries:

1. **Token Boundary Integrity Rate** (token-centric) -- fraction of
   content tokens whose bytes form valid, complete UTF-8.
2. **Character Boundary Split Count** (text-centric) -- how many
   multi-byte characters in the source text have their bytes spread
   across multiple tokens.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import logging

from .base import BaseMetrics, TokenizedDataProcessor
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider

logger = logging.getLogger(__name__)

# Byte-fallback token: <0xAB>
_BYTE_FALLBACK_RE = re.compile(r'^<0x([0-9A-Fa-f]{2})>$')

# Placeholder tokens produced by failed conversions
_PLACEHOLDER_RE = re.compile(r'^<(UNK|TOKEN)_\d+>$')


# ------------------------------------------------------------------
# GPT-2 byte-to-Unicode tables
# ------------------------------------------------------------------

def _build_gpt2_byte_tables() -> Tuple[Dict[int, str], Dict[str, int]]:
    """Build the GPT-2 byte <-> Unicode mapping tables.

    GPT-2 maps every byte 0x00-0xFF to a unique Unicode character:
    - Printable ASCII / Latin-1 ranges map to themselves.
    - All other bytes map to U+0100 and above.

    Returns ``(byte_to_unicode, unicode_to_byte)``.
    """
    # Ranges that map to themselves (printable ASCII + Latin-1 supplement)
    byte_values: List[int] = (
        list(range(ord('!'), ord('~') + 1))
        + list(range(ord('\xa1'), ord('\xac') + 1))
        + list(range(ord('\xae'), ord('\xff') + 1))
    )
    char_codes = list(byte_values)

    # Everything else gets mapped starting from U+0100
    n = 0
    for b in range(256):
        if b not in byte_values:
            byte_values.append(b)
            char_codes.append(256 + n)
            n += 1

    byte_to_unicode = {b: chr(c) for b, c in zip(byte_values, char_codes)}
    unicode_to_byte = {chr(c): b for b, c in zip(byte_values, char_codes)}
    return byte_to_unicode, unicode_to_byte


_GPT2_BYTE_TO_UNICODE, _GPT2_UNICODE_TO_BYTE = _build_gpt2_byte_tables()

# Marker characters that only appear in GPT-2-style byte encodings
# (characters at U+0100+ that are used to represent non-printable bytes)
_GPT2_MARKER_CHARS: Set[str] = {
    ch for ch in _GPT2_UNICODE_TO_BYTE if ord(ch) >= 0x100
}

# Threshold: if a tokenizer's vocab contains at least this many single-char
# tokens matching GPT-2 marker characters, we treat it as GPT-2-style.
_GPT2_DETECTION_THRESHOLD = 50


class UTF8IntegrityMetrics(BaseMetrics):
    """Measure UTF-8 character boundary integrity of tokenizers."""

    def __init__(self, input_provider: InputProvider):
        super().__init__(input_provider)
        self._gpt2_detection_cache: Dict[int, Optional[Dict[str, int]]] = {}

    # ------------------------------------------------------------------
    # GPT-2 detection
    # ------------------------------------------------------------------

    def _detect_gpt2_encoding(self, tokenizer: Any) -> Optional[Dict[str, int]]:
        """Detect whether *tokenizer* uses GPT-2-style byte encoding.

        Returns the ``unicode_to_byte`` table if detected, else ``None``.
        Results are cached per tokenizer object identity.
        """
        tok_id = id(tokenizer)
        if tok_id in self._gpt2_detection_cache:
            return self._gpt2_detection_cache[tok_id]

        result = None
        try:
            vocab = None
            if hasattr(tokenizer, 'get_vocab'):
                vocab = tokenizer.get_vocab()
            elif hasattr(tokenizer, 'vocab'):
                vocab = tokenizer.vocab

            if vocab:
                # Count single-char tokens that match GPT-2 marker characters
                marker_count = 0
                for token_str in vocab:
                    if isinstance(token_str, bytes):
                        token_str = token_str.decode('utf-8', errors='replace')
                    token_str = str(token_str)
                    if len(token_str) == 1 and token_str in _GPT2_MARKER_CHARS:
                        marker_count += 1
                        if marker_count >= _GPT2_DETECTION_THRESHOLD:
                            result = _GPT2_UNICODE_TO_BYTE
                            break
        except Exception as e:
            logger.debug("GPT-2 encoding detection failed: %s", e)

        self._gpt2_detection_cache[tok_id] = result
        if result is not None:
            logger.debug("Detected GPT-2-style byte encoding for tokenizer %s", type(tokenizer))
        return result

    # ------------------------------------------------------------------
    # Static / instance helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _token_string_to_bytes(
        token_str: str,
        unicode_to_byte: Optional[Dict[str, int]] = None,
    ) -> Optional[bytes]:
        """Convert a vocabulary token string to the raw bytes it represents.

        When *unicode_to_byte* is provided (GPT-2-style tokenizer), each
        character is mapped through the table to recover the original byte.
        Otherwise, subword markers are stripped and the result is UTF-8 encoded.

        Returns ``None`` for special tokens and placeholder strings.
        """
        # Special tokens — always skip regardless of encoding style
        if BaseMetrics._SPECIAL_TOKEN.match(token_str):
            return None

        # Placeholder tokens from failed conversions
        if _PLACEHOLDER_RE.match(token_str):
            return None

        # Byte-fallback tokens: <0xC3> -> bytes([0xC3])
        m = _BYTE_FALLBACK_RE.match(token_str)
        if m:
            return bytes([int(m.group(1), 16)])

        # --- GPT-2-style byte encoding ---
        if unicode_to_byte is not None:
            raw = bytearray()
            for ch in token_str:
                if ch in unicode_to_byte:
                    raw.append(unicode_to_byte[ch])
                else:
                    # Character not in GPT-2 table — encode as UTF-8
                    # (shouldn't happen for well-formed GPT-2 vocab)
                    raw.extend(ch.encode('utf-8'))
            return bytes(raw)

        # --- Non-GPT-2 tokenizers: strip markers, encode as UTF-8 ---
        # Use early returns for mutually exclusive marker types.

        # SentencePiece ▁ prefix
        if token_str and token_str[0] == '\u2581':
            return (' ' + token_str[1:]).encode('utf-8')

        # GPT-NeoX / non-GPT-2 Ġ prefix (only reached when unicode_to_byte is None)
        if token_str and token_str[0] == '\u0120':
            return (' ' + token_str[1:]).encode('utf-8')

        # BERT continuation prefix
        if token_str.startswith('##'):
            return token_str[2:].encode('utf-8')

        # </w> suffix
        if token_str.endswith('</w>'):
            return token_str[:-4].encode('utf-8')

        # @@ suffix
        if token_str.endswith('@@'):
            return token_str[:-2].encode('utf-8')

        return token_str.encode('utf-8')

    @staticmethod
    def _is_valid_complete_utf8(data: bytes) -> bool:
        """Return True if *data* decodes as complete, valid UTF-8."""
        try:
            data.decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False

    @staticmethod
    def _crosses_character_boundary(data: bytes) -> bool:
        """Return True if *data* contains bytes from more than one UTF-8
        character and at least one of those characters is incomplete.

        A boundary-crossing token is the product of a BPE merge that fused
        bytes across a character boundary.  For example, ``b'\\xa9\\xe4'``
        contains the tail of one character and the lead of another —
        neither is complete.

        Single-byte tokens and tokens that decode cleanly as one or more
        complete characters do NOT cross a boundary.
        """
        if len(data) <= 1:
            return False

        # Walk the bytes and identify character fragments.
        char_count = 0
        has_incomplete = False
        i = 0
        n = len(data)

        while i < n:
            b = data[i]
            if b < 0x80:
                # ASCII — always complete
                char_count += 1
                i += 1
            elif 0x80 <= b <= 0xBF:
                # Orphan continuation byte — incomplete fragment
                has_incomplete = True
                char_count += 1
                i += 1
            elif 0xC0 <= b <= 0xDF:
                expected = 2
                available = min(expected, n - i)
                # Check that all following bytes are continuations
                actual = 1
                for k in range(1, available):
                    if 0x80 <= data[i + k] <= 0xBF:
                        actual += 1
                    else:
                        break
                char_count += 1
                if actual < expected:
                    has_incomplete = True
                i += actual
            elif 0xE0 <= b <= 0xEF:
                expected = 3
                available = min(expected, n - i)
                actual = 1
                for k in range(1, available):
                    if 0x80 <= data[i + k] <= 0xBF:
                        actual += 1
                    else:
                        break
                char_count += 1
                if actual < expected:
                    has_incomplete = True
                i += actual
            elif 0xF0 <= b <= 0xF7:
                expected = 4
                available = min(expected, n - i)
                actual = 1
                for k in range(1, available):
                    if 0x80 <= data[i + k] <= 0xBF:
                        actual += 1
                    else:
                        break
                char_count += 1
                if actual < expected:
                    has_incomplete = True
                i += actual
            else:
                # Invalid byte (0xF8+)
                has_incomplete = True
                char_count += 1
                i += 1

        return char_count > 1 and has_incomplete

    @staticmethod
    def _classify_malformation(data: bytes) -> Optional[str]:
        """Classify the type of UTF-8 malformation in *data*.

        Returns:
            ``'trailing_incomplete'`` — ends with a leading byte lacking
            enough continuation bytes (the data simply runs out).
            ``'orphan_continuation'`` — contains a continuation byte (0x80-0xBF)
            not preceded by a valid leading byte, **or** a leading byte is
            followed by a non-continuation byte (the follower byte is
            effectively orphaned from any valid sequence).
            ``None`` — data is valid UTF-8.
        """
        try:
            data.decode('utf-8')
            return None
        except UnicodeDecodeError:
            pass

        if not data:
            return None

        i = 0
        n = len(data)
        while i < n:
            b = data[i]
            if b < 0x80:
                # ASCII — valid, advance
                i += 1
            elif 0x80 <= b <= 0xBF:
                # Continuation byte without preceding leader
                return 'orphan_continuation'
            elif 0xC0 <= b <= 0xDF:
                if i + 1 >= n:
                    return 'trailing_incomplete'
                if not (0x80 <= data[i + 1] <= 0xBF):
                    return 'orphan_continuation'
                i += 2
            elif 0xE0 <= b <= 0xEF:
                needed = 2
                for k in range(1, needed + 1):
                    if i + k >= n:
                        return 'trailing_incomplete'
                    if not (0x80 <= data[i + k] <= 0xBF):
                        return 'orphan_continuation'
                i += 3
            elif 0xF0 <= b <= 0xF7:
                needed = 3
                for k in range(1, needed + 1):
                    if i + k >= n:
                        return 'trailing_incomplete'
                    if not (0x80 <= data[i + k] <= 0xBF):
                        return 'orphan_continuation'
                i += 4
            else:
                # Invalid leading byte (0xF8+)
                return 'orphan_continuation'

        return None

    # ------------------------------------------------------------------
    # Byte-stream helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_byte_stream(
        token_bytes_list: List[Tuple[int, bytes]],
    ) -> Tuple[bytearray, List[int]]:
        """Concatenate token bytes into a flat stream with origin mapping.

        Returns ``(byte_stream, byte_to_token)`` where
        ``byte_to_token[i]`` is the token index that produced byte *i*.
        """
        byte_stream = bytearray()
        byte_to_token: List[int] = []
        for tok_idx, raw_bytes in token_bytes_list:
            for b in raw_bytes:
                byte_stream.append(b)
                byte_to_token.append(tok_idx)
        return byte_stream, byte_to_token

    @staticmethod
    def _align_byte_sequences(
        source_bytes: bytes, reconstructed_bytes: bytes,
    ) -> Tuple[List[Optional[int]], int]:
        """Bidirectional greedy alignment of source bytes to reconstructed bytes.

        Handles both extra bytes in *reconstructed_bytes* (forward lookahead)
        and extra bytes in *source_bytes* (skip source byte and continue).

        Returns ``(mapping, mismatch_count)`` where ``mapping[i]`` is the
        index into *reconstructed_bytes* for source byte *i*, or ``None``
        if unaligned.
        """
        n_src = len(source_bytes)
        n_rec = len(reconstructed_bytes)

        if source_bytes == reconstructed_bytes:
            return list(range(n_src)), 0

        mapping: List[Optional[int]] = [None] * n_src
        mismatches = 0
        j = 0  # pointer into reconstructed

        for i in range(n_src):
            if j >= n_rec:
                mismatches += 1
                continue
            if source_bytes[i] == reconstructed_bytes[j]:
                mapping[i] = j
                j += 1
            else:
                # Forward lookahead: reconstructed has extra bytes
                found = False
                for look in range(1, 5):
                    if j + look < n_rec and source_bytes[i] == reconstructed_bytes[j + look]:
                        mapping[i] = j + look
                        j = j + look + 1
                        found = True
                        break
                if not found:
                    # Reverse case: source has extra bytes — skip this
                    # source byte and try matching the next source byte
                    # against the same reconstructed position.
                    mismatches += 1
                    # Do NOT advance j — the current reconstructed byte
                    # may match a later source byte.

        return mapping, mismatches

    @staticmethod
    def _count_split_characters(
        source_bytes: bytes,
        mapping: List[Optional[int]],
        byte_to_token: List[int],
    ) -> Tuple[int, int, int, Dict[int, Tuple[int, int]], int]:
        """Count multi-byte UTF-8 characters whose bytes span multiple tokens.

        A multi-byte character is classified as split only when all of its
        bytes aligned to the reconstructed token stream and those bytes come
        from more than one token. A character with any unaligned byte (a
        ``None`` entry in *mapping*, produced when ``_align_byte_sequences``
        could not match it) cannot be classified as split-or-not, so it is
        excluded from both the split count and the multi-byte total and is
        reported separately as *unaligned*. This keeps alignment failures out
        of the split rate; see _align_byte_sequences for when bytes go
        unaligned.

        Returns ``(split_count, aligned_multibyte_chars, total_chars,
        per_width, unaligned_multibyte_chars)`` where *aligned_multibyte_chars*
        is the split-rate denominator (fully-aligned multi-byte chars only) and
        *per_width* maps byte width (2, 3, 4) to ``(splits_at_this_width,
        aligned_chars_at_this_width)``.
        """
        splits = 0
        aligned_multibyte = 0
        unaligned_multibyte = 0
        total_chars = 0
        # per_width: width -> [splits, aligned_total]
        per_width: Dict[int, List[int]] = {2: [0, 0], 3: [0, 0], 4: [0, 0]}
        i = 0
        n = len(source_bytes)

        while i < n:
            b = source_bytes[i]
            # Determine UTF-8 character length from leading byte
            if b < 0x80:
                char_len = 1
            elif b < 0xC0:
                # Orphan continuation byte — skip
                i += 1
                continue
            elif b < 0xE0:
                char_len = 2
            elif b < 0xF0:
                char_len = 3
            else:
                char_len = 4

            # Bounds check
            if i + char_len > n:
                break

            total_chars += 1

            if char_len > 1:
                # Collect token indices for each byte of this character.
                token_indices = set()
                has_unmapped = False
                for k in range(i, i + char_len):
                    m = mapping[k] if k < len(mapping) else None
                    if m is not None and m < len(byte_to_token):
                        token_indices.add(byte_to_token[m])
                    else:
                        has_unmapped = True

                if has_unmapped:
                    # An unaligned byte means alignment failed for this
                    # character; we cannot tell whether it is split, so it
                    # is excluded from the split-rate numerator and
                    # denominator and counted separately.
                    unaligned_multibyte += 1
                else:
                    aligned_multibyte += 1
                    if char_len in per_width:
                        per_width[char_len][1] += 1
                    if len(token_indices) > 1:
                        splits += 1
                        if char_len in per_width:
                            per_width[char_len][0] += 1

            i += char_len

        per_width_tuples = {w: (v[0], v[1]) for w, v in per_width.items()}
        return splits, aligned_multibyte, total_chars, per_width_tuples, unaligned_multibyte

    # ------------------------------------------------------------------
    # Main compute
    # ------------------------------------------------------------------

    def compute(
        self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None,
    ) -> Dict[str, Any]:
        """Compute UTF-8 token integrity and character split metrics."""
        if tokenized_data is None:
            tokenized_data = self.input_provider.get_tokenized_data()

        # Accumulators
        # integrity: tok -> lang -> {valid, total, trailing_incomplete,
        #                            orphan_continuation, boundary_crossings}
        integrity_acc: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(
                lambda: {'valid': 0, 'total': 0,
                         'trailing_incomplete': 0, 'orphan_continuation': 0,
                         'boundary_crossings': 0}
            )
        )
        # splits: tok -> lang -> {splits, multibyte, chars, tokens, mismatches,
        #                          w2_splits, w2_total, w3_splits, w3_total, w4_splits, w4_total}
        split_acc: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(
                lambda: {
                    'splits': 0, 'multibyte': 0, 'unaligned': 0, 'chars': 0,
                    'tokens': 0, 'mismatches': 0,
                    'w2_splits': 0, 'w2_total': 0,
                    'w3_splits': 0, 'w3_total': 0,
                    'w4_splits': 0, 'w4_total': 0,
                }
            )
        )

        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue

            tokenizer_obj = self.input_provider.get_tokenizer(tok_name)
            unicode_to_byte = self._detect_gpt2_encoding(tokenizer_obj)

            lang_groups = TokenizedDataProcessor.group_by_language(
                tokenized_data[tok_name]
            )

            for lang, data_list in lang_groups.items():
                for item in data_list:
                    token_strings = self._convert_ids_to_tokens(
                        tokenizer_obj, item.tokens
                    )

                    # Build per-token byte list (content tokens only)
                    token_bytes_list: List[Tuple[int, bytes]] = []
                    for tok_idx, tok_str in enumerate(token_strings):
                        raw_bytes = self._token_string_to_bytes(
                            tok_str, unicode_to_byte
                        )
                        if raw_bytes is None:
                            continue
                        token_bytes_list.append((tok_idx, raw_bytes))

                    total = len(token_bytes_list)
                    if total == 0:
                        continue

                    # --- Metric 1: Token Completeness & Boundary Crossings ---
                    iacc = integrity_acc[tok_name][lang]
                    valid = 0
                    for _, b in token_bytes_list:
                        if self._is_valid_complete_utf8(b):
                            valid += 1
                        else:
                            mtype = self._classify_malformation(b)
                            if mtype == 'trailing_incomplete':
                                iacc['trailing_incomplete'] += 1
                            elif mtype == 'orphan_continuation':
                                iacc['orphan_continuation'] += 1
                            if self._crosses_character_boundary(b):
                                iacc['boundary_crossings'] += 1
                    iacc['valid'] += valid
                    iacc['total'] += total

                    # --- Metric 2: Character Split Count ---
                    if item.text is not None:
                        byte_stream, byte_to_token = self._build_byte_stream(
                            token_bytes_list
                        )
                        source_bytes = item.text.encode('utf-8')
                        mapping, mismatches = self._align_byte_sequences(
                            source_bytes, bytes(byte_stream)
                        )
                        splits, aligned_mb, total_chars, per_width, unaligned_mb = (
                            self._count_split_characters(
                                source_bytes, mapping, byte_to_token
                            )
                        )

                        sacc = split_acc[tok_name][lang]
                        sacc['splits'] += splits
                        sacc['multibyte'] += aligned_mb
                        sacc['unaligned'] += unaligned_mb
                        sacc['chars'] += total_chars
                        sacc['tokens'] += total
                        sacc['mismatches'] += mismatches
                        for w in (2, 3, 4):
                            ws, wt = per_width.get(w, (0, 0))
                            sacc[f'w{w}_splits'] += ws
                            sacc[f'w{w}_total'] += wt

        return {
            'utf8_token_integrity': self._build_integrity_results(integrity_acc),
            'utf8_char_split': self._build_split_results(split_acc),
        }

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_integrity_results(
        self, acc: Dict[str, Dict[str, Dict[str, int]]],
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            'per_tokenizer': {},
            'summary': {},
            'metadata': {
                'description': (
                    'Fraction of content tokens whose bytes form complete '
                    'UTF-8 characters, plus boundary-crossing token counts.'
                ),
            },
        }

        for tok_name in self.tokenizer_names:
            per_lang: Dict[str, Any] = {}
            global_valid = 0
            global_total = 0
            global_trailing = 0
            global_orphan = 0
            global_crossings = 0

            for lang in sorted(acc.get(tok_name, {})):
                d = acc[tok_name][lang]
                v, t = d['valid'], d['total']
                ti, oc = d['trailing_incomplete'], d['orphan_continuation']
                bc = d['boundary_crossings']
                per_lang[lang] = {
                    'completeness_rate': v / t if t > 0 else 1.0,
                    'total_incomplete_tokens': t - v,
                    'total_content_tokens': t,
                    'trailing_incomplete': ti,
                    'orphan_continuation': oc,
                    'boundary_crossings': bc,
                    'boundary_crossing_rate': bc / t if t > 0 else 0.0,
                }
                global_valid += v
                global_total += t
                global_trailing += ti
                global_orphan += oc
                global_crossings += bc

            results['per_tokenizer'][tok_name] = {
                'global': {
                    'completeness_rate': global_valid / global_total if global_total > 0 else 1.0,
                    'total_incomplete_tokens': global_total - global_valid,
                    'total_content_tokens': global_total,
                    'trailing_incomplete': global_trailing,
                    'orphan_continuation': global_orphan,
                    'boundary_crossings': global_crossings,
                    'boundary_crossing_rate': global_crossings / global_total if global_total > 0 else 0.0,
                },
                'per_language': per_lang,
            }

            results['summary'][tok_name] = {
                'completeness_rate': global_valid / global_total if global_total > 0 else 1.0,
                'total_incomplete_tokens': global_total - global_valid,
                'total_content_tokens': global_total,
                'trailing_incomplete': global_trailing,
                'orphan_continuation': global_orphan,
                'boundary_crossings': global_crossings,
                'boundary_crossing_rate': global_crossings / global_total if global_total > 0 else 0.0,
                'languages_analyzed': len(per_lang),
            }

        return results

    def _build_split_results(
        self, acc: Dict[str, Dict[str, Dict[str, int]]],
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            'per_tokenizer': {},
            'summary': {},
            'metadata': {
                'description': (
                    'How many multi-byte UTF-8 characters have their '
                    'bytes spread across multiple tokens.'
                ),
            },
        }

        # split_rate and the splits-per-1k rates are None (JSON null) when
        # their denominator is 0, rather than 0.0, so "no data" is not read as
        # "no splits". aligned_fraction is the share of multi-byte chars that
        # aligned (the rest could not be classified); a low value flags an
        # untrustworthy split_rate.
        def _ratio(num: int, den: int, scale: float = 1.0) -> Optional[float]:
            return (num / den * scale) if den > 0 else None

        for tok_name in self.tokenizer_names:
            per_lang: Dict[str, Any] = {}
            g_splits = 0
            g_mb = 0
            g_unaligned = 0
            g_tokens = 0
            g_mismatches = 0
            g_width: Dict[int, List[int]] = {2: [0, 0], 3: [0, 0], 4: [0, 0]}

            for lang in sorted(acc.get(tok_name, {})):
                d = acc[tok_name][lang]
                sp = d['splits']
                mb = d['multibyte']
                un = d['unaligned']
                tk = d['tokens']
                mm = d['mismatches']

                lang_width: Dict[str, Any] = {}
                for w in (2, 3, 4):
                    ws, wt = d[f'w{w}_splits'], d[f'w{w}_total']
                    lang_width[f'{w}_byte'] = {
                        'splits': ws,
                        'total': wt,
                        'split_rate': _ratio(ws, wt),
                    }
                    g_width[w][0] += ws
                    g_width[w][1] += wt

                per_lang[lang] = {
                    'split_rate': _ratio(sp, mb),
                    'splits_per_1k_tokens': _ratio(sp, tk, 1000),
                    'splits_per_1k_multibyte': _ratio(sp, mb, 1000),
                    'total_splits': sp,
                    'total_multibyte_chars': mb,
                    'unaligned_multibyte_chars': un,
                    'aligned_fraction': _ratio(mb, mb + un),
                    'total_content_tokens': tk,
                    'alignment_mismatches': mm,
                    'per_byte_width': lang_width,
                }

                g_splits += sp
                g_mb += mb
                g_unaligned += un
                g_tokens += tk
                g_mismatches += mm

            global_width: Dict[str, Any] = {}
            for w in (2, 3, 4):
                ws, wt = g_width[w]
                global_width[f'{w}_byte'] = {
                    'splits': ws,
                    'total': wt,
                    'split_rate': _ratio(ws, wt),
                }

            results['per_tokenizer'][tok_name] = {
                'global': {
                    'split_rate': _ratio(g_splits, g_mb),
                    'splits_per_1k_tokens': _ratio(g_splits, g_tokens, 1000),
                    'splits_per_1k_multibyte': _ratio(g_splits, g_mb, 1000),
                    'total_splits': g_splits,
                    'total_multibyte_chars': g_mb,
                    'unaligned_multibyte_chars': g_unaligned,
                    'aligned_fraction': _ratio(g_mb, g_mb + g_unaligned),
                    'total_content_tokens': g_tokens,
                    'alignment_mismatches': g_mismatches,
                    'per_byte_width': global_width,
                },
                'per_language': per_lang,
            }

            results['summary'][tok_name] = {
                'split_rate': _ratio(g_splits, g_mb),
                'splits_per_1k_tokens': _ratio(g_splits, g_tokens, 1000),
                'total_splits': g_splits,
                'total_multibyte_chars': g_mb,
                'unaligned_multibyte_chars': g_unaligned,
                'aligned_fraction': _ratio(g_mb, g_mb + g_unaligned),
                'languages_analyzed': len(per_lang),
                'per_byte_width': global_width,
            }

        return results

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print UTF-8 completeness and boundary crossing results."""
        integrity = results.get('utf8_token_integrity')
        if integrity and 'summary' in integrity:
            print("\n" + "=" * 60)
            print("UTF-8 TOKEN COMPLETENESS & BOUNDARY CROSSING RESULTS")
            print("=" * 60)
            print("\nSUMMARY STATISTICS")
            print("-" * 40)
            for tok_name in self.tokenizer_names:
                if tok_name in integrity['summary']:
                    s = integrity['summary'][tok_name]
                    print(f"{tok_name}:")
                    print(f"  {'Completeness Rate':25}: {s['completeness_rate']:.4f}")
                    print(f"  {'Incomplete Tokens':25}: {s['total_incomplete_tokens']:,}")
                    print(f"  {'  Trailing Incomplete':25}: {s.get('trailing_incomplete', 0):,}")
                    print(f"  {'  Orphan Continuation':25}: {s.get('orphan_continuation', 0):,}")
                    print(f"  {'Boundary Crossings':25}: {s.get('boundary_crossings', 0):,}")
                    print(f"  {'Boundary Crossing Rate':25}: {s.get('boundary_crossing_rate', 0.0):.4f}")
                    print(f"  {'Content Tokens':25}: {s['total_content_tokens']:,}")
                    print(f"  {'Languages':25}: {s['languages_analyzed']}")

        char_split = results.get('utf8_char_split')
        if char_split and 'summary' in char_split:
            print("\n" + "=" * 60)
            print("UTF-8 CHARACTER BOUNDARY SPLIT RESULTS")
            print("=" * 60)
            print("\nSUMMARY STATISTICS")
            print("-" * 40)
            for tok_name in self.tokenizer_names:
                if tok_name in char_split['summary']:
                    s = char_split['summary'][tok_name]
                    def _fmt(v, spec):
                        return format(v, spec) if v is not None else 'n/a'
                    print(f"{tok_name}:")
                    print(f"  {'Split Rate':25}: {_fmt(s['split_rate'], '.4f')}")
                    print(f"  {'Splits/1k Tokens':25}: {_fmt(s['splits_per_1k_tokens'], '.2f')}")
                    print(f"  {'Total Splits':25}: {s['total_splits']:,}")
                    print(f"  {'Multi-byte Chars':25}: {s['total_multibyte_chars']:,}")
                    print(f"  {'Unaligned Mb Chars':25}: {s.get('unaligned_multibyte_chars', 0):,}")
                    print(f"  {'Aligned Fraction':25}: {_fmt(s.get('aligned_fraction'), '.4f')}")
                    pw = s.get('per_byte_width', {})
                    for w in (2, 3, 4):
                        wd = pw.get(f'{w}_byte', {})
                        ws = wd.get('splits', 0)
                        wt = wd.get('total', 0)
                        wr = wd.get('split_rate')
                        print(f"  {f'  {w}-byte splits':25}: {ws:,}/{wt:,} ({_fmt(wr, '.4f')})")
                    print(f"  {'Languages':25}: {s['languages_analyzed']}")

            print("\n" + "=" * 60)
