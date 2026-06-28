"""Single-tokenizer sanity-check diagnostic.

Emits a pass/warn/fail health report for one tokenizer, run through its own
faithful pipeline (the tokenizer's configured normalizer + pretokenizer are
never bypassed).  No silent fallbacks: anything that cannot be verified is
surfaced as ``unverifiable`` (which forces overall >= warn), never hidden.

Design principle (see plan): a multibyte char — or base char + combining
mark — split across tokens is fine when the byte stream stays valid UTF-8
and the text roundtrips losslessly.  The only defect is an
incomplete/orphaned multibyte grouping (the Character Boundary Crossing Rate
semantics).  A vocab token non-self-reproducing in isolation is normal for
BPE; only a token the tokenizer's own normalizer can never emit is dead.
"""

from __future__ import annotations

import logging
import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..constants import (
    MAX_EXAMPLE_DISPLAY_COUNT,
    SANITY_BYTE_COVERAGE_REQUIRED,
    SANITY_CROSS_BOUNDARY_PROBE,
    SANITY_DIGIT_CONSISTENCY_PASS,
    SANITY_DIGIT_ENTROPY_NORM,
    SANITY_MARK_LEADING_TOKEN_FAIL_FRAC,
    SANITY_MARK_LEADING_TOKEN_WARN_FRAC,
    SANITY_MAX_REASONABLE_TOKEN_CHARS,
    SANITY_MAX_UNREPRESENTABLE_BYTES_WARN,
    SANITY_PRETOK_CONSERVATION_FAIL_FRAC,
    SANITY_ROUNDTRIP_BUG_FAIL_FRAC,
    SANITY_ROUNDTRIP_CLEAN_PASS_FRAC,
    SANITY_UNK_SCRIPT_WARN_RATE,
    SANITY_VOCAB_UNREACHABLE_WARN_COUNT,
    SANITY_VOCAB_NORMALIZATION_DEAD_FAIL_COUNT,
    SANITY_STRICT_BYTE_ALPHABET_WARN_COUNT,
    SANITY_WHITESPACE_FIDELITY_PASS_FRAC,
)
from ..core.tokenizer_wrapper import (
    CustomBPETokenizer,
    HuggingFaceTokenizer,
    SentencePieceTokenizer,
    TokenizerWrapper,
    UniMixLMTokenizer,
)
from ..metrics.base import BaseMetrics
from ..metrics.basic import BasicTokenizationMetrics
from ..metrics.math import DigitBoundaryMetrics
from ..metrics.utf8_integrity import (
    _BYTE_FALLBACK_RE,
    _GPT2_DETECTION_THRESHOLD,
    _GPT2_MARKER_CHARS,
    _GPT2_UNICODE_TO_BYTE,
    UTF8IntegrityMetrics,
)

logger = logging.getLogger(__name__)

# Reused static helpers (audit-confirmed bindings).
_token_string_to_bytes = UTF8IntegrityMetrics._token_string_to_bytes
_is_valid_complete_utf8 = UTF8IntegrityMetrics._is_valid_complete_utf8
_crosses_character_boundary = UTF8IntegrityMetrics._crosses_character_boundary
_classify_malformation = UTF8IntegrityMetrics._classify_malformation
_cer = BasicTokenizationMetrics._character_error_rate
_ws_fidelity = BasicTokenizationMetrics._whitespace_fidelity


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

class Severity:
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    UNVERIFIABLE = "unverifiable"

    # Numeric contribution to the overall verdict.  not_applicable is neutral;
    # unverifiable forces overall >= warn (no-silent-fallback).
    _RANK = {PASS: 0, NOT_APPLICABLE: 0, WARN: 1, UNVERIFIABLE: 1, FAIL: 2}

    @classmethod
    def rank(cls, sev: str) -> int:
        return cls._RANK[sev]

    @classmethod
    def overall(cls, severities: List[str]) -> str:
        worst = max((cls._RANK[s] for s in severities), default=0)
        return {0: cls.PASS, 1: cls.WARN, 2: cls.FAIL}[worst]


def severity_to_exit_code(overall: str) -> int:
    """0 all-pass, 1 >=1 warn, 2 >=1 fail.  3 (execution error) is the CLI's."""
    return {Severity.PASS: 0, Severity.WARN: 1, Severity.FAIL: 2}[overall]


# ---------------------------------------------------------------------------
# Local token cleaning (audit B2: _process_token is an instance method —
# reimplemented here per decision 3, no metrics/ edits).
# ---------------------------------------------------------------------------

def is_special_token(raw: str) -> bool:
    return bool(BaseMetrics._SPECIAL_TOKEN.match(raw))


def clean_token(raw: str, char_decode_table: Dict[str, str]) -> Optional[str]:
    """Strip subword markers; return ``None`` for special tokens.

    Mirrors ``BaseMetrics._process_token(preserve_space=False)``.
    """
    if is_special_token(raw):
        return None
    table = {**BaseMetrics._DEFAULT_CHAR_DECODE, **(char_decode_table or {})}
    decoded = "".join(table.get(ch, ch) for ch in raw)
    if decoded.startswith("##"):
        return decoded[2:]
    if decoded.endswith("</w>"):
        return decoded[:-4]
    if decoded.endswith("@@"):
        return decoded[:-2]
    if decoded and decoded[0] == " ":
        return decoded[1:]
    return decoded


def _is_mark(ch: str) -> bool:
    """True for Unicode combining marks (categories Mn/Mc/Me)."""
    return unicodedata.category(ch) in ("Mn", "Mc", "Me")


# ---------------------------------------------------------------------------
# Byte-encoding detection (decision 3: self-contained scan over the imported
# GPT-2 table; no metrics/ edits).
# ---------------------------------------------------------------------------

def detect_byte_encoding(vocab: Dict[str, int],
                         backend: Any = None) -> Dict[str, Any]:
    """Return ``{'style', 'unicode_to_byte'}``.

    style is one of ``'gpt2'`` / ``'byte_fallback'`` / ``None``.

    The authoritative signal for GPT-2-style byte mapping is a ``ByteLevel``
    pre-tokenizer or decoder (HF ByteLevel uses exactly the GPT-2 byte<->
    unicode table).  The vocab marker-count heuristic is only a fallback —
    it under-counts on small / tiktoken-converted vocabs (observed: a
    ByteLevel gpt4 tokenizer with 37 single-char markers < the threshold).
    """
    if backend is not None:
        comp = " ".join(
            type(getattr(backend, a, None)).__name__
            for a in ("pre_tokenizer", "decoder")
        ) + " " + " ".join(
            str(getattr(backend, a, "")) for a in ("pre_tokenizer", "decoder")
        )
        if "ByteLevel" in comp:
            return {"style": "gpt2", "unicode_to_byte": _GPT2_UNICODE_TO_BYTE}
        # Authoritative non-byte-level signals: a Metaspace / WordPiece /
        # SentencePiece decoder or pre-tokenizer means this is NOT byte-level,
        # regardless of how many vocab chars coincidentally collide with the
        # GPT-2 marker range (observed: xlm-roberta-base, 62 collisions in a
        # 250k SentencePiece vocab -> spurious byte-level misdetection).
        if any(s in comp for s in ("Metaspace", "WordPiece", "BPEDecoder")):
            return {"style": None, "unicode_to_byte": None}
    if not vocab:
        return {"style": None, "unicode_to_byte": None}
    marker_count = 0
    fallback_count = 0
    for tok in vocab:
        if isinstance(tok, bytes):
            tok = tok.decode("utf-8", errors="replace")
        tok = str(tok)
        if len(tok) == 1 and tok in _GPT2_MARKER_CHARS:
            marker_count += 1
        if _BYTE_FALLBACK_RE.match(tok):
            fallback_count += 1
    if marker_count >= _GPT2_DETECTION_THRESHOLD:
        return {"style": "gpt2", "unicode_to_byte": _GPT2_UNICODE_TO_BYTE}
    # A full byte-fallback inventory is 256 <0xNN> tokens; accept a generous
    # majority to allow for tokenizers that also have merged byte tokens.
    if fallback_count >= 200:
        return {"style": "byte_fallback", "unicode_to_byte": None}
    return {"style": None, "unicode_to_byte": None}


# ---------------------------------------------------------------------------
# Faithful-pipeline / normalizer view
# ---------------------------------------------------------------------------

@dataclass
class NormalizerView:
    normalize_fn: Optional[Callable[[str], str]]
    introspectable: bool
    reason: str
    normalizer_repr: str
    pretokenizer_repr: str
    decoder_repr: str


def get_normalizer_view(wrapper: TokenizerWrapper) -> NormalizerView:
    underlying = wrapper.get_underlying_tokenizer()

    if isinstance(wrapper, UniMixLMTokenizer):
        return NormalizerView(
            None, False,
            "langspec per-language normalizer is not exposed by the wrapper",
            "langspec(per-language)", "langspec", "langspec",
        )
    if isinstance(wrapper, SentencePieceTokenizer):
        return NormalizerView(
            None, False,
            "SentencePiece normalization is internal and not introspectable",
            "SentencePiece(internal)", "SentencePiece(internal)",
            "SentencePiece(internal)",
        )
    if isinstance(wrapper, (HuggingFaceTokenizer, CustomBPETokenizer)):
        if underlying is None:
            return NormalizerView(None, False,
                                  "underlying tokenizer not available",
                                  "unknown", "unknown", "unknown")
        backend = getattr(underlying, "backend_tokenizer", None) or underlying
        has_norm_attr = hasattr(backend, "normalizer")
        norm = getattr(backend, "normalizer", None)
        pretok = getattr(backend, "pre_tokenizer", None)
        dec = getattr(backend, "decoder", None)
        pre_repr = type(pretok).__name__ if pretok is not None else "None"
        dec_repr = type(dec).__name__ if dec is not None else "None"
        if norm is not None and hasattr(norm, "normalize_str"):
            return NormalizerView(norm.normalize_str, True, "",
                                  type(norm).__name__, pre_repr, dec_repr)
        if has_norm_attr and norm is None:
            # Verified "no normalization configured" — identity is the truth,
            # not a fallback.
            return NormalizerView(lambda s: s, True, "",
                                  "Identity(no normalizer configured)",
                                  pre_repr, dec_repr)
        return NormalizerView(None, False,
                              "tokenizer exposes no introspectable normalizer",
                              "unknown", pre_repr, dec_repr)
    return NormalizerView(None, False,
                          f"unknown wrapper type {type(wrapper).__name__}",
                          "unknown", "unknown", "unknown")


# ---------------------------------------------------------------------------
# Check result container
# ---------------------------------------------------------------------------

def _mk(name: str, category: str, severity: str, observed: Any,
        threshold: Any, detail: str, rationale: str,
        examples: Optional[List[Any]] = None) -> Dict[str, Any]:
    return {
        "name": name,
        "category": category,
        "severity": severity,
        "observed": observed,
        "threshold": threshold,
        "detail": detail,
        "rationale": rationale,
        "examples": (examples or [])[:MAX_EXAMPLE_DISPLAY_COUNT],
    }


# Roundtrip root-cause buckets.
_RED_FLAG_BUGS = {"unk_loss", "casing_loss_bug", "byte_bug", "merge_or_decode_bug"}


@dataclass
class TokenizerSanityChecker:
    wrapper: TokenizerWrapper
    probes: List[Any]  # probe_corpus.Probe
    name: str = ""

    vocab: Dict[str, int] = field(default_factory=dict)
    _id_to_tok: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.wrapper.get_name()
        self.vocab = self.wrapper.get_vocab() or {}
        self._id_to_tok = {v: k for k, v in self.vocab.items()}
        self.vocab_size = self.wrapper.get_vocab_size()
        underlying = self.wrapper.get_underlying_tokenizer()
        try:
            self.char_decode = (
                BaseMetrics._build_char_decode_table(underlying)
                if underlying is not None else {}
            )
        except Exception:
            self.char_decode = {}
        _backend = (getattr(underlying, "backend_tokenizer", None)
                    or underlying)
        self.byte_enc = detect_byte_encoding(self.vocab, _backend)
        self.normview = get_normalizer_view(self.wrapper)
        self.lowercasing_normalizer = self._detect_lowercasing()

    # -- small encode/decode helpers (always faithful) -------------------

    def _encode(self, text: str) -> List[int]:
        return list(self.wrapper.encode(text))

    def _decode(self, ids: List[int]) -> Optional[str]:
        if not self.wrapper.can_decode():
            return None
        return self.wrapper.decode(ids, skip_special_tokens=True)

    def _token_bytes(self, raw: str) -> Optional[bytes]:
        return _token_string_to_bytes(raw, self.byte_enc["unicode_to_byte"])

    # -- C16 helper ------------------------------------------------------

    def _is_cross_boundary(self) -> bool:
        """True if the tokenizer merges across pretokenizer boundaries (e.g. SuperBPE
        superwords). Detected behaviorally: encode a fixed probe and check whether any
        emitted token's surface contains internal whitespace -- impossible for a normal
        within-pretoken BPE/Unigram, routine for a superword tokenizer. Cached.

        This must be behavioral, not based on pretokenize(): a SuperBPE pretokenizer
        leaves an internal-space superword like ' over the' as ONE piece but splits a
        trailing-space token like ' before ' into two, so pretokenize() is not a reliable
        cross-boundary signal."""
        if getattr(self, "_cross_boundary_cache", None) is not None:
            return self._cross_boundary_cache
        result = False
        if self.wrapper.can_decode():
            try:
                ids = self._encode(SANITY_CROSS_BOUNDARY_PROBE)
                for tid in ids:
                    surf = self.wrapper.decode([tid], skip_special_tokens=False)
                    if surf and any(c.isspace() for c in surf.strip()):
                        result = True
                        break
            except Exception:
                result = False
        self._cross_boundary_cache = result
        return result

    # -- C9 helper -------------------------------------------------------

    def _detect_lowercasing(self) -> bool:
        nv = self.normview
        if nv.introspectable and nv.normalize_fn is not None:
            try:
                return nv.normalize_fn("ABCÉ") == "abcé"
            except Exception:
                return False
        try:
            return (self._encode("ABCDEF") == self._encode("abcdef")
                    and bool(self._encode("ABCDEF")))
        except Exception:
            return False

    # ===================================================================
    # C1 — byte-level 256 coverage
    # ===================================================================

    def check_byte_coverage(self) -> Dict[str, Any]:
        name = "C1 byte-level 256-coverage"
        style = self.byte_enc["style"]
        if style is None:
            return _mk(name, "static", Severity.NOT_APPLICABLE, "non-byte-level",
                       SANITY_BYTE_COVERAGE_REQUIRED,
                       "tokenizer is not byte-level; coverage check skipped "
                       "explicitly (not a silent pass)",
                       "only byte-level tokenizers can losslessly encode "
                       "arbitrary bytes")
        # Representable in vocab?
        representable = set()
        if style == "gpt2":
            present = set(self.vocab.keys())
            for ch, b in _GPT2_UNICODE_TO_BYTE.items():
                if ch in present:
                    representable.add(b)
        else:  # byte_fallback
            for tok in self.vocab:
                m = _BYTE_FALLBACK_RE.match(str(tok))
                if m:
                    representable.add(int(m.group(1), 16))
        missing = sorted(set(range(256)) - representable)
        # Behavioral roundtrip is authoritative: a byte-level tokenizer may
        # not store every byte as an isolated single-char vocab key yet still
        # encode/decode every byte losslessly (observed: gpt-neox-20b — 13
        # vocab-"missing" bytes, 0 behavioral failures).  Only a real
        # roundtrip failure is a defect.
        if self.wrapper.can_decode():
            rt_fail = []
            for b in range(256):
                s = bytes([b]).decode("latin-1")
                try:
                    if self._decode(self._encode(s)) != s:
                        rt_fail.append(b)
                except Exception:
                    rt_fail.append(b)
            if rt_fail:
                return _mk(name, "behavioral", Severity.FAIL, 256 - len(rt_fail),
                           SANITY_BYTE_COVERAGE_REQUIRED,
                           f"{len(rt_fail)} byte value(s) fail encode/decode "
                           f"roundtrip ({style}); "
                           f"{len(missing)} also absent as standalone vocab key",
                           "a byte-level tokenizer that cannot roundtrip a "
                           "byte cannot losslessly encode arbitrary input",
                           [hex(b) for b in rt_fail])
            note = ("" if not missing else
                    f" ({len(missing)} byte(s) absent as standalone vocab "
                    f"keys but still roundtrip)")
            return _mk(name, "behavioral", Severity.PASS, 256,
                       SANITY_BYTE_COVERAGE_REQUIRED,
                       f"all 256 bytes roundtrip ({style}){note}",
                       "byte-level tokenizer covers the full byte range")
        # No decoder: fall back to the (weaker) static vocab check.
        if missing:
            return _mk(name, "static", Severity.FAIL, len(representable),
                       SANITY_BYTE_COVERAGE_REQUIRED,
                       f"{len(missing)} byte value(s) unrepresentable in vocab "
                       "(static check only; tokenizer cannot decode)",
                       "a byte-level tokenizer missing byte values cannot "
                       "losslessly encode arbitrary input",
                       [hex(b) for b in missing])
        return _mk(name, "static", Severity.PASS, 256,
                   SANITY_BYTE_COVERAGE_REQUIRED,
                   f"all 256 bytes representable in vocab ({style}; "
                   "static check only)",
                   "byte-level tokenizer covers the full byte range")

    # ===================================================================
    # C17 — strict byte-alphabet vocab presence
    # ===================================================================
    # C1 is round-trip-based: it returns PASS when every byte round-trips
    # via fallback, even if some byte-alphabet tokens are absent. C17 is the
    # strict variant: every byte must appear as its own single-token vocab
    # key. Missing valid UTF-8 lead bytes (0xC2-0xF4) affect text in
    # Supplementary Unicode planes (rare CJK, ancient scripts), fragment
    # tokenization for those characters, and leave the LM with effectively
    # no learned embedding for them.

    def check_byte_alphabet_strict(self) -> Dict[str, Any]:
        name = "C17 strict byte-alphabet vocab presence"
        style = self.byte_enc["style"]
        if style is None:
            return _mk(name, "static", Severity.NOT_APPLICABLE, 0,
                       SANITY_STRICT_BYTE_ALPHABET_WARN_COUNT,
                       "tokenizer is not byte-level; strict byte-alphabet check skipped",
                       "only byte-level tokenizers have a 256-byte alphabet to enumerate")
        representable = set()
        if style == "gpt2":
            present = set(self.vocab.keys())
            for ch, b in _GPT2_UNICODE_TO_BYTE.items():
                if ch in present:
                    representable.add(b)
        else:  # byte_fallback
            for tok in self.vocab:
                m = _BYTE_FALLBACK_RE.match(str(tok))
                if m:
                    representable.add(int(m.group(1), 16))
        missing = sorted(set(range(256)) - representable)
        if len(missing) > SANITY_STRICT_BYTE_ALPHABET_WARN_COUNT:
            # Valid UTF-8 lead bytes are 0xC2-0xF4 (2/3/4-byte sequence starts).
            # Anything else in `missing` is either a continuation byte (0x80-0xBF)
            # or an invalid lead (0xC0-0xC1, 0xF5-0xFF) that can never appear in
            # valid UTF-8 text.
            valid_leads = [b for b in missing if 0xC2 <= b <= 0xF4]
            other = [b for b in missing if b not in valid_leads]
            note = (
                f"{len(missing)} byte value(s) absent as standalone vocab keys "
                f"({len(valid_leads)} valid UTF-8 lead byte(s), "
                f"{len(other)} non-UTF-8-lead byte(s)). Round-trip still succeeds "
                f"via multi-token fallback (see C1); the valid-lead misses affect "
                f"characters in Supplementary Unicode planes."
            )
            return _mk(name, "static", Severity.WARN, len(missing),
                       SANITY_STRICT_BYTE_ALPHABET_WARN_COUNT,
                       note,
                       "a complete 256-byte vocab alphabet gives deterministic "
                       "single-token encoding for every byte and a real embedding "
                       "slot per byte in the downstream LM",
                       [hex(b) for b in missing])
        return _mk(name, "static", Severity.PASS, 0,
                   SANITY_STRICT_BYTE_ALPHABET_WARN_COUNT,
                   "all 256 byte-alphabet tokens present as standalone vocab keys",
                   "byte-level tokenizer has a complete 256-byte alphabet")

    # ===================================================================
    # C2 — combining-mark mishandling (static is the real signal)
    # ===================================================================

    def check_combining_marks(self) -> Dict[str, Any]:
        name = "C2 combining-mark mishandling"
        bare = []
        leading = 0
        considered = 0
        for raw in self.vocab:
            cleaned = clean_token(str(raw), self.char_decode)
            if not cleaned:
                continue
            considered += 1
            if all(_is_mark(c) for c in cleaned):
                bare.append(cleaned)
            elif _is_mark(cleaned[0]):
                leading += 1
        lead_frac = (leading / considered) if considered else 0.0
        if bare:
            return _mk(name, "static", Severity.FAIL, len(bare), 0,
                       f"{len(bare)} vocab token(s) are bare combining marks",
                       "a token that is only combining marks is a training "
                       "artifact (base+mark split during training)",
                       bare)
        if lead_frac >= SANITY_MARK_LEADING_TOKEN_FAIL_FRAC:
            sev = Severity.FAIL
        elif lead_frac >= SANITY_MARK_LEADING_TOKEN_WARN_FRAC:
            sev = Severity.WARN
        else:
            sev = Severity.PASS
        return _mk(name, "static", sev, round(lead_frac, 6),
                   SANITY_MARK_LEADING_TOKEN_WARN_FRAC,
                   f"{leading}/{considered} tokens begin with a combining mark; "
                   "behavioral mark/byte defects are judged by C3 byte_bug "
                   "(Character Boundary Crossing semantics) — legitimate "
                   "split-but-roundtrips is never penalized",
                   "systematic base+mark splitting corrupts diacritic-heavy "
                   "scripts")

    # ===================================================================
    # C3 — lossy-text root-cause (core) + feeds C5/C12 breakdown
    # ===================================================================

    def _classify_roundtrip(self, text: str) -> str:
        try:
            ids = self._encode(text)
        except Exception:
            return "merge_or_decode_bug"
        d = self._decode(ids)
        if d is None:
            return "lossy_unverifiable" if not self.wrapper.can_decode() \
                else "merge_or_decode_bug"
        if d == text:
            return "clean"
        nv = self.normview
        n = None
        if nv.introspectable and nv.normalize_fn is not None:
            try:
                n = nv.normalize_fn(text)
            except Exception:
                n = None
        if n is not None and (d == n or d == nv.normalize_fn(d)):
            return "lossy_expected"
        # UNK loss
        unk_id = self.wrapper.get_unk_token_id()
        if unk_id is not None and unk_id in ids:
            return "unk_loss"
        # Byte mishandling: concatenated token bytes fail to form valid UTF-8.
        try:
            toks = self.wrapper.convert_ids_to_tokens(ids)
            blob = bytearray()
            ok = True
            for t in toks:
                tb = self._token_bytes(str(t))
                if tb is None:
                    ok = False
                    break
                blob.extend(tb)
            if ok and _classify_malformation(bytes(blob)) is not None:
                return "byte_bug"
        except Exception:
            pass
        # Casing
        if d.casefold() == text.casefold():
            return ("casing_loss_expected" if self.lowercasing_normalizer
                    else "casing_loss_bug")
        # Normalization-form equivalence
        for form in ("NFC", "NFKC"):
            if unicodedata.normalize(form, d) == unicodedata.normalize(form, text):
                if n is not None:
                    return "normalization_loss"
                return "lossy_unverifiable"
        if n is None:
            return "lossy_unverifiable"
        return "merge_or_decode_bug"

    def _roundtrip_breakdown(self) -> Dict[str, Any]:
        buckets: Dict[str, int] = {}
        ascii_bug = []
        examples: Dict[str, List[str]] = {}
        for p in self.probes:
            cat = self._classify_roundtrip(p.text)
            buckets[cat] = buckets.get(cat, 0) + 1
            if cat in _RED_FLAG_BUGS:
                examples.setdefault(cat, []).append(p.text[:80])
                if getattr(p, "category", "") == "ascii_basic":
                    ascii_bug.append(p.text[:80])
        total = sum(buckets.values()) or 1
        clean_like = buckets.get("clean", 0) + buckets.get("lossy_expected", 0) \
            + buckets.get("normalization_loss", 0) \
            + buckets.get("casing_loss_expected", 0)
        bug = sum(buckets.get(k, 0) for k in _RED_FLAG_BUGS)
        unver = buckets.get("lossy_unverifiable", 0)
        return {
            "buckets": buckets, "total": total,
            "clean_frac": clean_like / total,
            "bug_frac": bug / total,
            "unverifiable": unver,
            "ascii_bug": ascii_bug,
            "examples": examples,
        }

    def check_roundtrip(self, bd: Dict[str, Any]) -> Dict[str, Any]:
        name = "C3 lossy-text root-cause"
        ex = [f"{k}: {v}" for k, vs in bd["examples"].items() for v in vs]
        if bd["ascii_bug"]:
            return _mk(name, "behavioral", Severity.FAIL, bd["buckets"],
                       "ascii must be clean",
                       f"ASCII control probes show bugs: {bd['ascii_bug']}",
                       "ASCII text must always roundtrip", ex)
        if bd["bug_frac"] >= SANITY_ROUNDTRIP_BUG_FAIL_FRAC:
            sev = Severity.FAIL
        elif bd["bug_frac"] > 0 or bd["clean_frac"] < SANITY_ROUNDTRIP_CLEAN_PASS_FRAC:
            sev = Severity.WARN
        elif bd["unverifiable"] > 0:
            sev = Severity.UNVERIFIABLE
        else:
            sev = Severity.PASS
        return _mk(name, "behavioral", sev,
                   {"clean_frac": round(bd["clean_frac"], 4),
                    "bug_frac": round(bd["bug_frac"], 4),
                    "unverifiable": bd["unverifiable"]},
                   SANITY_ROUNDTRIP_CLEAN_PASS_FRAC,
                   f"roundtrip buckets: {bd['buckets']}",
                   "separates normalizer-inherent loss (expected) from "
                   "tokenization/decode bugs (red flag)", ex)

    # ===================================================================
    # C4 — faithful-pipeline conformance / transparency
    # ===================================================================

    def check_faithful_pipeline(self) -> Dict[str, Any]:
        name = "C4 faithful-pipeline conformance"
        nv = self.normview
        if not nv.introspectable:
            return _mk(name, "static", Severity.UNVERIFIABLE,
                       nv.normalizer_repr, "introspectable",
                       f"normalizer not introspectable ({nv.reason}); "
                       "lossy/expected discrimination degraded",
                       "transparency + no-silent-fallback ethos")
        # Consistency probe: NFC/NFD must encode-equal iff the normalizer
        # maps them together.
        from .probe_corpus import NFC_NFD_PAIRS
        mismatch = []
        for nfc, nfd in NFC_NFD_PAIRS:
            try:
                same_enc = self._encode(nfc) == self._encode(nfd)
                same_norm = nv.normalize_fn(nfc) == nv.normalize_fn(nfd)
            except Exception:
                continue
            if same_norm and not same_enc:
                mismatch.append(nfc)
        if mismatch:
            return _mk(name, "behavioral", Severity.FAIL, len(mismatch), 0,
                       "encode bypasses the declared normalizer for "
                       f"{len(mismatch)} NFC/NFD pair(s)",
                       "a declared normalizer that is not actually applied is "
                       "a silent-bypass defect", mismatch)
        return _mk(name, "behavioral", Severity.PASS, nv.normalizer_repr,
                   "introspectable",
                   f"normalizer={nv.normalizer_repr}, "
                   f"pretok={nv.pretokenizer_repr}, decoder={nv.decoder_repr}",
                   "the configured pipeline is applied and introspectable")

    # ===================================================================
    # C5 — whitespace handling + vocab share
    # ===================================================================

    def check_whitespace(self) -> Dict[str, Any]:
        name = "C5 whitespace handling"
        ws_only = 0
        ws_any = 0
        considered = 0
        for raw in self.vocab:
            cleaned = clean_token(str(raw), self.char_decode)
            if cleaned is None:
                continue
            considered += 1
            if cleaned and all(c.isspace() for c in cleaned):
                ws_only += 1
            if any(c.isspace() for c in cleaned):
                ws_any += 1
        ws_frac = (ws_only / considered) if considered else 0.0
        # Behavioral fidelity on whitespace probes.
        preserved = 0
        total_ws = 0
        applicable = 0
        for p in self.probes:
            if getattr(p, "category", "") != "whitespace":
                continue
            try:
                d = self._decode(self._encode(p.text))
            except Exception:
                d = None
            if d is None:
                continue
            pr, tot = _ws_fidelity(p.text, d)
            if tot == 0:
                continue  # not-applicable, do NOT treat as 0%
            applicable += 1
            preserved += pr
            total_ws += tot
        if applicable == 0:
            fidelity = None
            sev = Severity.PASS
            detail = "no whitespace-bearing probes applicable"
        else:
            fidelity = preserved / total_ws if total_ws else 1.0
            # WARN-only by design: many mainstream tokenizers (WordPiece,
            # SentencePiece/Metaspace) are intentionally whitespace-lossy,
            # so this is surfaced, never a hard failure.
            sev = (Severity.PASS if fidelity >= SANITY_WHITESPACE_FIDELITY_PASS_FRAC
                   else Severity.WARN)
            detail = (f"whitespace fidelity={fidelity:.4f}; "
                      f"whitespace-only vocab share={ws_frac:.4f} "
                      f"({ws_only}/{considered})")
        return _mk(name, "behavioral", sev, fidelity,
                   SANITY_WHITESPACE_FIDELITY_PASS_FRAC, detail,
                   "whitespace loss corrupts code/indentation (warn-only: "
                   "WordPiece/SentencePiece are intentionally lossy here)",
                   None)

    # ===================================================================
    # C6 — digit handling + vocab share
    # ===================================================================

    def check_digits(self) -> Dict[str, Any]:
        name = "C6 digit handling"
        pure_digit = 0
        max_run = 0
        considered = 0
        for raw in self.vocab:
            cleaned = clean_token(str(raw), self.char_decode)
            if cleaned is None or cleaned == "":
                continue
            considered += 1
            if cleaned.isdigit():
                pure_digit += 1
                max_run = max(max_run, len(cleaned))
        digit_frac = (pure_digit / considered) if considered else 0.0

        patterns: List[tuple] = []
        directions: Dict[str, int] = {}
        for p in self.probes:
            if getattr(p, "category", "") not in ("digits", "math"):
                continue
            try:
                ids, offsets = self.wrapper.encode_with_offsets(p.text)
            except Exception:
                continue
            if not offsets:
                continue
            recon, c2t = self._char_to_token_from_offsets(p.text, ids, offsets)
            for s, e, digs in DigitBoundaryMetrics._find_number_spans(recon):
                b = DigitBoundaryMetrics._get_digit_span_boundaries(c2t, s, e)
                if b is None:
                    continue
                patterns.append(tuple(b))
                nd = e - s
                if nd < 2:
                    continue
                right = DigitBoundaryMetrics._ideal_boundaries(nd)
                left = {nd - x for x in right}
                bs = set(b)
                if not bs:
                    directions["none"] = directions.get("none", 0) + 1
                elif bs == right:
                    directions["right"] = directions.get("right", 0) + 1
                elif bs == left:
                    directions["left"] = directions.get("left", 0) + 1
                else:
                    directions["other"] = directions.get("other", 0) + 1
        ent = DigitBoundaryMetrics._compute_pattern_entropy(patterns)
        npat = ent["num_patterns"]
        if npat > 1:
            norm_ent = ent["entropy"] / math.log2(npat)
        else:
            norm_ent = 0.0
        consistency = 1.0 - norm_ent
        if directions:
            direction = max(directions, key=directions.get)
            if list(directions.values()).count(max(directions.values())) > 1:
                direction = "mixed"
        else:
            direction = "n/a"
        sev = (Severity.PASS if consistency >= SANITY_DIGIT_CONSISTENCY_PASS
               else Severity.WARN)
        return _mk(name, "behavioral", sev, round(consistency, 4),
                   SANITY_DIGIT_CONSISTENCY_PASS,
                   f"chunking_direction={direction}, "
                   f"consistency={consistency:.4f} "
                   f"(norm basis {SANITY_DIGIT_ENTROPY_NORM}); "
                   f"pure-digit vocab share={digit_frac:.4f}, "
                   f"max digit-run len={max_run}",
                   "inconsistent digit chunking degrades arithmetic")

    @staticmethod
    def _char_to_token_from_offsets(text, ids, offsets) -> Tuple[str, List[int]]:
        """Char->token map over the *source* text using encoder offsets."""
        c2t = [-1] * len(text)
        for tidx, off in enumerate(offsets):
            if not off:
                continue
            s, e = off
            for i in range(s, min(e, len(text))):
                c2t[i] = tidx
        return text, c2t

    # ===================================================================
    # C7 — special-token sanity
    # ===================================================================

    def check_special_tokens(self) -> Dict[str, Any]:
        name = "C7 special-token sanity"
        underlying = self.wrapper.get_underlying_tokenizer()
        specials: Dict[str, int] = {}
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            v = getattr(underlying, attr, None)
            if isinstance(v, int):
                specials[attr] = v
        unk = self.wrapper.get_unk_token_id()
        if isinstance(unk, int):
            specials["unk_token_id"] = unk
        # Robust UNK presence: the wrapper's get_unk_token_id() can return
        # None even when an UNK token exists (observed: bert-base-uncased,
        # [UNK]=100).  Fall back to scanning the vocab for known UNK strings.
        from ..constants import UNK_CANDIDATES
        has_unk = unk is not None or any(c in self.vocab
                                         for c in UNK_CANDIDATES)
        issues = []
        sev = Severity.PASS
        ids = list(specials.values())
        if len(ids) != len(set(ids)) and ids:
            # Intentional aliasing is common and valid (GPT-NeoX
            # bos=eos=unk; Qwen eos=pad) -> surface as WARN, not FAIL.
            issues.append(f"aliased special ids (often intentional): {specials}")
            sev = Severity.WARN
        for label, sid in specials.items():
            if not (0 <= sid < self.vocab_size):
                issues.append(f"{label}={sid} out of [0,{self.vocab_size})")
                sev = Severity.FAIL
            surface = self._id_to_tok.get(sid)
            if surface is not None:
                try:
                    if self._encode(surface) != [sid]:
                        issues.append(f"{label} surface {surface!r} not atomic")
                        sev = Severity.FAIL
                except Exception:
                    pass
        if not has_unk and self.byte_enc["style"] is None:
            issues.append("no UNK token for a non-byte-level tokenizer")
            sev = Severity.WARN if sev == Severity.PASS else sev
        return _mk(name, "static", sev, specials, "unique/in-range/atomic",
                   "; ".join(issues) or "special tokens consistent",
                   "missing/duplicate/non-atomic special tokens are classic "
                   "silent config bugs")

    # ===================================================================
    # C8 — determinism / idempotency
    # ===================================================================

    def check_determinism(self) -> Dict[str, Any]:
        name = "C8 determinism/idempotency"
        sample = [p.text for p in self.probes[:50]]
        for t in sample:
            if self._encode(t) != self._encode(t):
                return _mk(name, "behavioral", Severity.FAIL, "non-deterministic",
                           "deterministic", f"encode({t!r}) not stable",
                           "non-determinism silently corrupts all downstream "
                           "metrics")
        # batch vs loop
        try:
            batch = self.wrapper.encode_batch_with_offsets(sample)
            loop = [self.wrapper.encode_with_offsets(t) for t in sample]
            if [b[0] for b in batch] != [l[0] for l in loop]:
                return _mk(name, "behavioral", Severity.WARN, "batch != loop",
                           "batch == loop",
                           "batched encoding differs from per-text "
                           "(known for langspec); reported, not fatal",
                           "batch/loop divergence is acceptable only when "
                           "documented")
        except Exception as e:
            logger.debug("batch/loop check skipped: %s", e)
        return _mk(name, "behavioral", Severity.PASS, "deterministic",
                   "deterministic", "encode stable; batch == loop",
                   "deterministic tokenization is required for reproducibility")

    # ===================================================================
    # C10 — pretokenizer char conservation
    # ===================================================================

    def check_pretok_conservation(self) -> Dict[str, Any]:
        name = "C10 pretokenizer char conservation"
        if not self.wrapper.can_pretokenize():
            return _mk(name, "behavioral", Severity.NOT_APPLICABLE, "no pretok",
                       SANITY_PRETOK_CONSERVATION_FAIL_FRAC,
                       "tokenizer has no pretokenizer", "n/a")
        nv = self.normview
        kept = 0
        total = 0
        worst = []
        for p in self.probes:
            if getattr(p, "category", "") in ("control_chars",):
                continue  # control chars are legitimately normalizer-dropped
            base = p.text
            if nv.introspectable and nv.normalize_fn is not None:
                try:
                    base = nv.normalize_fn(base)
                except Exception:
                    base = p.text
            try:
                pieces = self.wrapper.pretokenize(p.text)
            except Exception:
                continue
            cleaned = "".join(
                clean_token(str(x), self.char_decode) or "" for x in pieces
            )
            in_chars = [c for c in base if not c.isspace()]
            out_chars = [c for c in cleaned if not c.isspace()]
            total += len(in_chars)
            kept += min(len(in_chars), len(out_chars))
            if len(out_chars) < len(in_chars):
                worst.append(p.text[:60])
        frac = (kept / total) if total else 1.0
        sev = (Severity.FAIL if frac < SANITY_PRETOK_CONSERVATION_FAIL_FRAC
               else Severity.PASS)
        return _mk(name, "behavioral", sev, round(frac, 6),
                   SANITY_PRETOK_CONSERVATION_FAIL_FRAC,
                   f"pretokenizer conserved {frac:.6f} of non-space chars",
                   "a silently char-dropping pretokenizer is invisible data "
                   "loss", worst)

    # ===================================================================
    # C11 — NFC/NFD roundtrip
    # ===================================================================

    def check_nfc_nfd(self) -> Dict[str, Any]:
        name = "C11 NFC/NFD roundtrip"
        from .probe_corpus import NFC_NFD_PAIRS
        identical = 0
        lossy = []
        for nfc, nfd in NFC_NFD_PAIRS:
            try:
                same = self._encode(nfc) == self._encode(nfd)
            except Exception:
                continue
            identical += int(same)
            for form in (nfc, nfd):
                if self._classify_roundtrip(form) in _RED_FLAG_BUGS:
                    lossy.append(form)
        if lossy:
            return _mk(name, "behavioral", Severity.WARN, len(lossy), 0,
                       f"{len(lossy)} NFC/NFD form(s) lossy with a red-flag bug",
                       "NFC/NFD is the cleanest normalization-vs-bug "
                       "discriminator", lossy)
        return _mk(name, "behavioral", Severity.PASS,
                   f"{identical}/{len(NFC_NFD_PAIRS)} encode-identical", None,
                   "NFC/NFD pairs roundtrip without red-flag bugs",
                   "canonical-equivalence robustness")

    # ===================================================================
    # C12 — emoji / ZWJ / control
    # ===================================================================

    def check_emoji_control(self) -> Dict[str, Any]:
        name = "C12 emoji/ZWJ/control"
        lossy: Dict[str, List[str]] = {}
        for p in self.probes:
            if getattr(p, "category", "") not in ("emoji_zwj", "control_chars"):
                continue
            cat = self._classify_roundtrip(p.text)
            if cat in _RED_FLAG_BUGS:
                lossy.setdefault(cat, []).append(p.text[:40])
        if lossy:
            return _mk(name, "behavioral", Severity.WARN, lossy, 0,
                       f"emoji/control red-flag buckets: "
                       f"{ {k: len(v) for k, v in lossy.items()} }",
                       "control/ZWJ sequences are common silent-loss culprits",
                       [v for vs in lossy.values() for v in vs])
        return _mk(name, "behavioral", Severity.PASS, "clean", None,
                   "emoji/ZWJ/control probes roundtrip cleanly",
                   "robust handling of astral/ZWJ/control input")

    # ===================================================================
    # C13 — UNK incidence per script
    # ===================================================================

    def check_unk_per_script(self) -> Dict[str, Any]:
        name = "C13 UNK-per-script"
        unk = self.wrapper.get_unk_token_id()
        if unk is None:
            return _mk(name, "behavioral", Severity.NOT_APPLICABLE, "no UNK",
                       SANITY_UNK_SCRIPT_WARN_RATE,
                       "tokenizer has no UNK token", "n/a")
        per_script: Dict[str, List[int]] = {}
        for p in self.probes:
            if getattr(p, "category", "") not in ("multiscript", "flores"):
                continue
            try:
                ids = self._encode(p.text)
            except Exception:
                continue
            if not ids:
                continue
            s = per_script.setdefault(p.note or "?", [0, 0])
            s[0] += sum(1 for i in ids if i == unk)
            s[1] += len(ids)
        flagged = {k: round(v[0] / v[1], 4) for k, v in per_script.items()
                   if v[1] and v[0] / v[1] > SANITY_UNK_SCRIPT_WARN_RATE}
        sev = Severity.WARN if flagged else Severity.PASS
        return _mk(name, "behavioral", sev, flagged or "all below threshold",
                   SANITY_UNK_SCRIPT_WARN_RATE,
                   f"scripts over UNK threshold: {flagged}" if flagged
                   else "no script exceeds the UNK rate threshold",
                   "high per-script UNK rate indicates an undertrained script")

    # ===================================================================
    # C14 — vocab integrity
    # ===================================================================

    def check_vocab_integrity(self) -> Dict[str, Any]:
        name = "C14 vocab integrity"
        issues = []
        sev = Severity.PASS
        if len(self.vocab) != self.vocab_size:
            issues.append(
                f"len(get_vocab())={len(self.vocab)} != "
                f"get_vocab_size()={self.vocab_size}")
            sev = Severity.FAIL
        ids = list(self.vocab.values())
        if len(ids) != len(set(ids)):
            issues.append("duplicate ids in vocab (surface->id collisions)")
            sev = Severity.FAIL
        if ids:
            lo, hi = min(ids), max(ids)
            if lo != 0 or hi != len(set(ids)) - 1:
                issues.append(f"non-contiguous id range [{lo},{hi}] "
                              f"for {len(set(ids))} ids")
                if sev == Severity.PASS:
                    sev = Severity.WARN
        return _mk(name, "static", sev, len(self.vocab),
                   "contiguous & sized",
                   "; ".join(issues) or "vocab structurally consistent",
                   "corrupt vocab files surface as id gaps / size mismatch")

    # ===================================================================
    # C15 — absurd token-length outliers
    # ===================================================================

    def check_token_outliers(self) -> Dict[str, Any]:
        name = "C15 token-length outliers"
        can_decode = self.wrapper.can_decode()
        outliers = []
        for raw in self.vocab:
            cleaned = clean_token(str(raw), self.char_decode)
            if cleaned and len(cleaned) > SANITY_MAX_REASONABLE_TOKEN_CHARS:
                # Store the human-readable decoded form (the byte-level surface
                # is unreadable mojibake for non-ASCII tokens); fall back to the
                # cleaned surface if the tokenizer can't decode.
                example = cleaned
                if can_decode:
                    try:
                        dec = self.wrapper.decode([self.vocab[raw]])
                        if dec:
                            example = dec
                    except Exception:
                        pass
                outliers.append(example[:80])
        sev = Severity.WARN if outliers else Severity.PASS
        return _mk(name, "static", sev, len(outliers),
                   SANITY_MAX_REASONABLE_TOKEN_CHARS,
                   f"{len(outliers)} token(s) exceed "
                   f"{SANITY_MAX_REASONABLE_TOKEN_CHARS} chars (warn-only; "
                   "long code tokens can be legitimate)",
                   "absurdly long tokens signal training-data contamination",
                   outliers)

    # ===================================================================
    # C16 — vocab reachability under the faithful pipeline
    # ===================================================================

    # Representative "breaker" characters, one per major regex branch class
    # (letter, digit, ASCII punctuation, space, newline). A token whose
    # standalone surface the pretokenizer splits may still be emitted when it
    # sits next to a character of a different class -- the neighbour lets a
    # different arm capture the surface as one pre-token. This is generic to any
    # split regex; it is NOT specific to repeat-run caps.
    _REACH_BREAKERS = ("a", "0", ".", " ", "\n")

    def _embedded_reachable(self, surface: str, tid: int) -> bool:
        """Probe whether a token is emitted by the faithful pipeline in *some*
        embedded context, even though its standalone surface pre-tokenizes into
        >=2 pieces.

        Robust to arbitrary pretokenization regexes, including negative-lookahead
        repeat caps (e.g. ``(?!(?<p>.)\\k<p>{8})``): such regexes condition the
        split on the *surrounding* characters, so a token can be reachable only
        when its neighbours differ from its run character. We therefore probe
        breakers on the prefix, the suffix, and both sides (lookaheads look
        forward, so the suffix probe matters), plus a doubled run for
        single-character tokens (a longer identical run can emit the token as a
        within-run chunk).

        Reachability is decided by the real ``_encode`` pipeline, never by
        reasoning about the regex, so a hit is always a genuine witness input.
        A miss leaves the conservative "unreachable" verdict untouched; the
        probe can only remove false positives, never introduce a false "dead".
        """
        if not surface:
            return False
        probes = []
        for b in self._REACH_BREAKERS:
            probes.append(b + surface)
            probes.append(surface + b)
            probes.append(b + surface + b)
        if len(set(surface)) == 1:
            probes.append(surface + surface)
        for p in probes:
            try:
                if tid in self._encode(p):
                    return True
            except Exception:
                continue
        return False

    def check_vocab_reachability(self) -> Dict[str, Any]:
        name = "C16 vocab reachability"
        buckets = {"self_reproducing": 0, "context_only": 0,
                   "non_self_reproducing": 0, "normalization_unreachable": 0,
                   "pretokenizer_unreachable": 0, "unverifiable": 0}
        dead_examples = []
        nv = self.normview
        can_decode = self.wrapper.can_decode()
        # Pretokenizer-unreachable: a token whose standalone surface the pretokenizer splits
        # into >=2 pre-tokens is a *candidate* for dead vocab, but a standalone split does not
        # prove it: the token may still be emitted when its surface sits inside a larger
        # pre-token (a divider run after a different punct char, a run-capped sub-token inside
        # a longer identical run). _embedded_reachable probes such contexts via the real
        # pipeline before flagging, so reachable-only-in-context tokens are counted as
        # context_only. SuperBPE-style tokenizers deliberately merge across *whitespace*
        # boundaries (superwords), so for them we exempt only tokens whose surface contains
        # internal whitespace; a non-whitespace dead token (e.g. a >3-digit piece under a
        # digit-capping pretokenizer) is still a candidate.
        can_pretok = self.wrapper.can_pretokenize()
        cross_boundary = can_pretok and self._is_cross_boundary()
        special_ids = self.wrapper.get_special_token_ids()
        for tok_str, tid in self.vocab.items():
            tok_str = str(tok_str)
            if tid in special_ids or is_special_token(tok_str):
                continue
            tb = self._token_bytes(tok_str)
            if tb is None or not _is_valid_complete_utf8(tb):
                buckets["context_only"] += 1
                continue
            if not can_decode:
                buckets["unverifiable"] += 1
                continue
            try:
                surface = self.wrapper.decode([tid], skip_special_tokens=False)
            except Exception:
                surface = None
            if not surface:
                buckets["context_only"] += 1
                continue
            try:
                if tid in self._encode(surface):
                    buckets["self_reproducing"] += 1
                    continue
            except Exception:
                buckets["context_only"] += 1
                continue
            # pretokenizer-unreachable: the pretokenizer splits the token's own surface
            # (after normalization, matching the real pipeline) into >=2 pre-tokens.
            # A standalone split is necessary but NOT sufficient for dead vocab: a token
            # can still be emitted when its surface appears inside a larger pre-token
            # (e.g. a divider run '===============' preceded by a different punct char, or
            # a run-capped sub-token emitted inside a longer identical run). Before
            # flagging, _embedded_reachable probes such contexts via the real pipeline; a
            # token reachable in any context is counted as context_only, not dead. For
            # cross-boundary (SuperBPE) tokenizers only the internal-whitespace superwords
            # are exempt from the standalone split test.
            if can_pretok:
                eff = surface
                if nv.introspectable and nv.normalize_fn is not None:
                    try:
                        eff = nv.normalize_fn(surface)
                    except Exception:
                        eff = surface
                try:
                    if cross_boundary:
                        # SuperBPE-style: stage-2 merges cross *whitespace* only, so a token is
                        # unreachable only if some whitespace-delimited chunk of its surface is
                        # itself split by the pretokenizer -- a non-whitespace boundary (e.g. a
                        # digit cap or punctuation split) it cannot bridge. Whitespace-spanning
                        # superwords (e.g. ' over the', 'Aug ') are reachable and not flagged.
                        chunks = [c for c in re.split(r"\s+", eff) if c]
                        dead_pt = any(len(self.wrapper.pretokenize(c)) >= 2 for c in chunks)
                    else:
                        # within-pretoken tokenizer: a standalone pretok split is a candidate
                        dead_pt = len(self.wrapper.pretokenize(eff)) >= 2
                except Exception:
                    dead_pt = False
                if dead_pt:
                    # A standalone split alone is not proof of dead vocab; the token may be
                    # emitted inside a larger pre-token. Probe embedded contexts first.
                    if self._embedded_reachable(surface, tid):
                        buckets["context_only"] += 1
                    else:
                        buckets["pretokenizer_unreachable"] += 1
                        if len(dead_examples) < MAX_EXAMPLE_DISPLAY_COUNT:
                            dead_examples.append(surface[:40])
                    continue
            if nv.introspectable and nv.normalize_fn is not None:
                try:
                    snorm = nv.normalize_fn(surface)
                    if snorm != surface and tid not in self._encode(snorm):
                        buckets["normalization_unreachable"] += 1
                        if len(dead_examples) < MAX_EXAMPLE_DISPLAY_COUNT:
                            dead_examples.append(surface[:40])
                        continue
                except Exception:
                    pass
                buckets["non_self_reproducing"] += 1
            else:
                buckets["unverifiable"] += 1
        # Normalization-dead vocab FAILs: the introspectable normalizer folds the surface,
        # so NO input can ever produce the token -- it signals a vocab built without applying
        # the normalizer. Pretokenizer-dead vocab is only a WARN: the slot is wasted but, like
        # the normalizer case, never corrupts text or emits UNK -- it is a capacity issue, not
        # a construction defect.
        if buckets["normalization_unreachable"] > SANITY_VOCAB_NORMALIZATION_DEAD_FAIL_COUNT:
            sev = Severity.FAIL
        elif buckets["pretokenizer_unreachable"] > SANITY_VOCAB_UNREACHABLE_WARN_COUNT:
            sev = Severity.WARN
        elif buckets["unverifiable"] > 0:
            sev = Severity.UNVERIFIABLE
        else:
            sev = Severity.PASS
        return _mk(name, "behavioral", sev, buckets,
                   SANITY_VOCAB_UNREACHABLE_WARN_COUNT,
                   f"reachability buckets: {buckets}",
                   "normalization-dead vocab FAILs (the normalizer guarantees the token is "
                   "unreachable -> vocab built without the normalizer); pretokenizer-dead vocab "
                   "WARNs (wasted slot, but no input produces it so it cannot corrupt text or "
                   "emit UNK); context-dependent non-self-reproduction and byte fragments are "
                   "legitimate",
                   dead_examples), buckets

    # ===================================================================
    # Orchestration
    # ===================================================================

    def run(self) -> Dict[str, Any]:
        bd = self._roundtrip_breakdown()
        c16, reach_buckets = self.check_vocab_reachability()
        checks = [
            self.check_byte_coverage(),
            self.check_byte_alphabet_strict(),
            self.check_combining_marks(),
            self.check_roundtrip(bd),
            self.check_faithful_pipeline(),
            self.check_whitespace(),
            self.check_digits(),
            self.check_special_tokens(),
            self.check_determinism(),
            self.check_pretok_conservation(),
            self.check_nfc_nfd(),
            self.check_emoji_control(),
            self.check_unk_per_script(),
            self.check_vocab_integrity(),
            self.check_token_outliers(),
            c16,
        ]
        overall = Severity.overall([c["severity"] for c in checks])
        return {
            "overall_severity": overall,
            "checks": {c["name"]: c for c in checks},
            "lossy_breakdown": bd["buckets"],
            "vocab_reachability": reach_buckets,
            "vocab_composition": {
                "vocab_size": self.vocab_size,
                "byte_style": self.byte_enc["style"],
                "n_special_tokens": len(self.wrapper.get_special_token_ids()),
            },
            "components": {
                "normalizer": self.normview.normalizer_repr,
                "pretokenizer": self.normview.pretokenizer_repr,
                "decoder": self.normview.decoder_repr,
                "normalizer_introspectable": self.normview.introspectable,
                "normalizer_reason": self.normview.reason,
            },
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _thresholds_metadata() -> Dict[str, Any]:
    from .. import constants as C
    return {k: getattr(C, k) for k in dir(C) if k.startswith("SANITY_")}


def run_sanity_check(wrappers: Dict[str, TokenizerWrapper],
                     probes: List[Any]) -> Dict[str, Any]:
    """Run the diagnostic over one or more named tokenizer wrappers."""
    per_tok: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    for name, w in wrappers.items():
        rep = TokenizerSanityChecker(w, probes, name=name).run()
        per_tok[name] = rep
        n_fail = sum(1 for c in rep["checks"].values()
                     if c["severity"] == Severity.FAIL)
        n_warn = sum(1 for c in rep["checks"].values()
                     if c["severity"] in (Severity.WARN, Severity.UNVERIFIABLE))
        summary[name] = {"overall_severity": rep["overall_severity"],
                         "n_fail": n_fail, "n_warn": n_warn}
    return {
        "tokenizer_sanity_check": {
            "per_tokenizer": per_tok,
            "summary": summary,
            "metadata": {
                "description": "Single-tokenizer health diagnostic "
                               "(faithful pipeline; no silent fallbacks).",
                "thresholds": _thresholds_metadata(),
                "components": {n: r["components"] for n, r in per_tok.items()},
            },
        }
    }


def render_text(report: Dict[str, Any], quiet: bool = False,
                use_color: Optional[bool] = None) -> str:
    if use_color is None:
        use_color = sys.stdout.isatty()

    def col(sev: str, s: str) -> str:
        if not use_color:
            return s
        code = {Severity.PASS: 32, Severity.WARN: 33, Severity.FAIL: 31,
                Severity.UNVERIFIABLE: 33, Severity.NOT_APPLICABLE: 90}.get(sev, 0)
        return f"\x1b[{code}m{s}\x1b[0m"

    lines: List[str] = []
    root = report["tokenizer_sanity_check"]
    for name, rep in root["per_tokenizer"].items():
        lines.append("=" * 70)
        ov = rep["overall_severity"]
        lines.append(f"{name}: {col(ov, ov.upper())}")
        lines.append("=" * 70)
        for cname, c in rep["checks"].items():
            if quiet and c["severity"] == Severity.PASS:
                continue
            tag = col(c["severity"], f"[{c['severity'].upper()}]")
            lines.append(f"{tag} {cname}: observed={c['observed']} "
                         f"threshold={c['threshold']}")
            lines.append(f"    {c['detail']}")
        lines.append(f"  lossy_breakdown: {rep['lossy_breakdown']}")
        lines.append(f"  vocab_reachability: {rep['vocab_reachability']}")
        lines.append(f"  components: {rep['components']}")
    return "\n".join(lines)
