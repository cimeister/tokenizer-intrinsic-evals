"""Deterministic, offline probe corpus for the tokenizer sanity check.

The built-in probes are the contract: hand-curated, reviewable, no network.
Each :class:`Probe` carries a ``category`` (used by the per-category checks)
and an optional ``note`` describing why it is interesting.  FLORES / math
breadth is opt-in and layered on top by the CLI.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Probe:
    text: str
    category: str
    note: str = ""


# Probe categories (stable identifiers; referenced by checks C2/C3/C5/C6/C12).
CAT_ASCII = "ascii_basic"          # control: must roundtrip exactly
CAT_WHITESPACE = "whitespace"
CAT_DIGITS = "digits"
CAT_MARKS = "combining_marks"
CAT_NFC_NFD = "nfc_nfd_pairs"
CAT_CASING = "casing"
CAT_CONTROL = "control_chars"
CAT_EMOJI = "emoji_zwj"
CAT_MULTISCRIPT = "multiscript"
CAT_FLORES = "flores"
CAT_MATH = "math"

# (NFC form, NFD form) pairs — same grapheme, two normal forms.  Used by C11
# and as the cleanest normalization-vs-bug discriminator in C3.
NFC_NFD_PAIRS: List[Tuple[str, str]] = [
    (unicodedata.normalize("NFC", s), unicodedata.normalize("NFD", s))
    for s in ("café", "résumé", "naïve", "Å", "ñoño", "한국어", "Ðróttning")
]


_ASCII = [
    "The quick brown fox jumps over the lazy dog.",
    "def f(x): return x + 1  # comment",
    "a,b;c:d.e!f?g",
    "1234567890",
]

_WHITESPACE = [
    "a  b",                 # double space
    "a\tb",                 # tab
    "a\nb",                 # newline
    "a\r\nb",               # CRLF
    "  leading",            # leading spaces
    "trailing  ",           # trailing spaces
    "line1\n\n\nline2",     # blank lines
    "indent:\n    four spaces\n\ttab",
    "nbsp gap",        # NBSP
    "thin space",      # thin space
    "ideo　graphic",    # ideographic space
    "zero​width",      # zero-width space
]

_DIGITS = [
    "0", "7", "42", "007", "1000000", "1234567890",
    "3.14159", "1,000,000", "0xDEADBEEF",
    "year 2026 month 05 day 17",
    "²³",                       # superscripts
    "٠١٢٣٤٥٦٧٨٩",               # Arabic-Indic digits
    "०१२३४५६७८९",               # Devanagari digits
    "abc123def456",
]

_CASING = [
    "HELLO", "Hello", "hello",
    "Straße",                   # German eszett
    "İstanbul",                 # Turkish dotted capital I
    "ﬁle",                      # ligature fi (NFKC-sensitive)
]

_MULTISCRIPT = [
    ("Latin sample text", "Latin"),
    ("Образец текста", "Cyrillic"),
    ("نموذج النص", "Arabic"),
    ("文本示例", "Han"),
    ("पाठ नमूना", "Devanagari"),
    ("텍스트 샘플", "Hangul"),
    ("טקסט לדוגמה", "Hebrew"),
    ("δείγμα κειμένου", "Greek"),
    ("ตัวอย่างข้อความ", "Thai"),
]


def _combining_mark_probes() -> List[Probe]:
    probes = [
        Probe("é", CAT_MARKS, "base e + combining acute (decomposed é)"),
        Probe("à́", CAT_MARKS, "base + two stacked combining marks"),
        Probe("́", CAT_MARKS, "bare combining acute accent (no base)"),
        Probe("ö", CAT_MARKS, "base o + combining diaeresis"),
        Probe("ก้", CAT_MARKS, "Thai consonant + tone mark"),
        Probe("क़", CAT_MARKS, "Devanagari ka + nukta"),
        Probe("אָ", CAT_MARKS, "Hebrew alef + qamats (niqqud)"),
        Probe("اً", CAT_MARKS, "Arabic alef + fathatan (harakat)"),
    ]
    return probes


def _control_char_probes() -> List[Probe]:
    return [
        Probe("a\x00b", CAT_CONTROL, "embedded NUL"),
        Probe("a\x07b", CAT_CONTROL, "BEL"),
        Probe("a\x1bb", CAT_CONTROL, "ESC"),
        Probe("a\x7fb", CAT_CONTROL, "DEL"),
        Probe("a\x85b", CAT_CONTROL, "NEL"),
        Probe("﻿bom", CAT_CONTROL, "leading BOM / ZWNBSP"),
    ]


def _emoji_probes() -> List[Probe]:
    return [
        Probe("😀", CAT_EMOJI, "single emoji (astral)"),
        Probe("👍\U0001f3fd", CAT_EMOJI, "emoji + skin-tone modifier"),
        Probe("👨‍👩‍👧", CAT_EMOJI, "ZWJ family sequence"),
        Probe("🇨🇭", CAT_EMOJI, "regional-indicator flag"),
        Probe("1️⃣", CAT_EMOJI, "keycap sequence"),
    ]


def builtin_probes() -> List[Probe]:
    """Return the full deterministic built-in probe set (offline, stable)."""
    probes: List[Probe] = []
    probes += [Probe(t, CAT_ASCII) for t in _ASCII]
    probes += [Probe(t, CAT_WHITESPACE) for t in _WHITESPACE]
    probes += [Probe(t, CAT_DIGITS) for t in _DIGITS]
    probes += _combining_mark_probes()
    for nfc, nfd in NFC_NFD_PAIRS:
        probes.append(Probe(nfc, CAT_NFC_NFD, "NFC form"))
        probes.append(Probe(nfd, CAT_NFC_NFD, "NFD form"))
    probes += [Probe(t, CAT_CASING) for t in _CASING]
    probes += _control_char_probes()
    probes += _emoji_probes()
    probes += [Probe(t, CAT_MULTISCRIPT, script) for t, script in _MULTISCRIPT]
    return probes


def all_byte_strings() -> List[bytes]:
    """The 256 single-byte values, for C1's behavioral byte-coverage probe."""
    return [bytes([b]) for b in range(256)]


def load_flores_probes(language_config_path: str,
                       samples_per_lang: int) -> List[Probe]:
    """Opt-in FLORES breadth via the existing multilingual loader.

    Fails loudly (propagates) if the config or data is missing — no silent
    fallback to an empty set.
    """
    from ..config.language_metadata import LanguageMetadata
    from ..loaders.multilingual_data import load_multilingual_data

    lm = LanguageMetadata(language_config_path)
    lang_texts = load_multilingual_data(lm, max_texts_per_language=samples_per_lang)
    probes: List[Probe] = []
    for lang, texts in lang_texts.items():
        for t in texts:
            probes.append(Probe(t, CAT_FLORES, lang))
    if not probes:
        raise ValueError(
            f"FLORES probe load from {language_config_path} produced 0 texts; "
            f"refusing to continue silently."
        )
    return probes


def load_math_probes(path: Optional[str] = None) -> List[Probe]:
    """Opt-in math breadth via the existing builtin math loader."""
    from ..utils.text_utils import load_math_data, BUILTIN_MATH_SAMPLES_PATH

    src = path or BUILTIN_MATH_SAMPLES_PATH
    texts = load_math_data(src)
    if not texts:
        raise ValueError(f"Math probe load from {src} produced 0 texts.")
    return [Probe(t, CAT_MATH) for t in texts]


def probes_by_category(probes: List[Probe]) -> Dict[str, List[Probe]]:
    out: Dict[str, List[Probe]] = {}
    for p in probes:
        out.setdefault(p.category, []).append(p)
    return out
