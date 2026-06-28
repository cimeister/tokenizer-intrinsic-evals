"""
Constants for tokenizer analysis framework.

This module defines all magic numbers and configuration constants used throughout
the tokenizer analysis codebase to improve maintainability and reduce errors.
"""

from typing import List


# --- Text Processing ---

MIN_PARAGRAPH_LENGTH = 5
MIN_LINE_LENGTH = 5
MIN_SENTENCE_LENGTH = 5
MIN_CONTENT_LENGTH = 5

DEFAULT_CHUNK_SIZE = 500
TRUNCATION_DISPLAY_LENGTH = 100
MAX_TEXTS_FALLBACK = 10

LARGE_ARRAY_THRESHOLD = 100
ARRAY_SAMPLING_POINTS = 50


# --- Statistics ---

DEFAULT_RENYI_ALPHAS: List[float] = [1.0, 2.0, 2.5, 3.0]
SHANNON_ENTROPY_ALPHA = 1.0

DEFAULT_SAFE_DIVIDE_VALUE = 0.0

PERCENTAGE_MULTIPLIER = 100
DEFAULT_PRECISION = 4


# --- Validation ---

MIN_WORD_LENGTH = 2
MIN_LANGUAGES_FOR_GINI = 2
MIN_LANGUAGES_FOR_COMPARISON = 1
MIN_TOKENIZERS_FOR_PLOTS = 1

MAX_ERROR_DISPLAY_COUNT = 5
MAX_TOKEN_DISPLAY_COUNT = 20
MAX_EXAMPLE_DISPLAY_COUNT = 20

MAX_ARRAY_DISPLAY_LENGTH = 5


# --- Data Processing ---

DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_TEXTS_PER_LANGUAGE = 1000
DEFAULT_MAX_SAMPLES = 2000

STEP_SIZE_FOR_LARGE_ARRAYS = 50
OVERLAP_THRESHOLD = 0

DEFAULT_RANK_VALUE = 0.0
DEFAULT_COST_VALUE = 0.0
DEFAULT_UTILIZATION_VALUE = 0.0


# --- File Formats ---

JSON_EXTENSIONS = ['.json']
TEXT_EXTENSIONS = ['.txt', '.text']
PARQUET_EXTENSIONS = ['.parquet']

TEXT_COLUMN_NAMES = ['text', 'content', 'sentence', 'document', 'passage']

DEFAULT_ENCODING = 'utf-8'
ERROR_HANDLING = 'replace'


# --- Morphology ---

BYTE_PREFIXES = ['▁', 'Ġ']
CONTINUATION_PREFIXES = ['##']
SUFFIX_PATTERNS = ['</w>', '@@']

MIN_MORPHEME_LENGTH = 1
MAX_MORPHEME_OVERLAP = 1.0

PUNCTUATION = '.,!?;:"()[]{}'

# Real UNK tokens are always delimited. Bare 'unk'/'UNK' were removed because they
# match ordinary subwords (e.g. SuperBPE has a plain 'unk' subword from words like
# "junk"/"sunk"), which made get_unk_token_id() misidentify it as the UNK token and
# inflated unk_token_rate. '<|endoftext|>' is GPT-style EOS, not UNK, so also dropped.
UNK_CANDIDATES = ['<unk>', '[UNK]', '<UNK>', '<|unk|>', '\u2047']


# --- Tokenizer Sanity Check ---
# Pass/warn/fail thresholds for the single-tokenizer sanity-check diagnostic
# (tokenizer_analysis/diagnostics/sanity_check.py). Every value here is echoed
# verbatim into the report's metadata.thresholds so results stay traceable.

# C1: a byte-level tokenizer must represent all 256 byte values.
SANITY_BYTE_COVERAGE_REQUIRED = 256
# C1: >0 bytes that are in vocab but fail behavioral roundtrip -> warn.
SANITY_MAX_UNREPRESENTABLE_BYTES_WARN = 0
# C17: strict byte-alphabet vocab presence. Above this count of missing single-byte
# tokens, the check is a WARNING. Round-trip can still succeed via multi-token
# fallback (that is what C1 tests), but a strict alphabet is needed for deterministic
# single-token encoding of every byte and to give the LM a real embedding slot for
# each byte. Missing valid UTF-8 lead bytes (0xC2-0xF4) affect text in Supplementary
# Unicode planes (rare CJK extensions, Linear B, Cuneiform, Egyptian hieroglyphs, ...).
SANITY_STRICT_BYTE_ALPHABET_WARN_COUNT = 0
# C2: fraction of vocab tokens that begin with a combining mark.
SANITY_MARK_LEADING_TOKEN_WARN_FRAC = 0.005
SANITY_MARK_LEADING_TOKEN_FAIL_FRAC = 0.02
# C16: count of pretokenizer-unreachable vocab tokens (the pretokenizer splits the
# surface and no embedded context emits it). Above this count the check is a WARNING:
# the slot is wasted capacity but no input produces the token, so it cannot corrupt
# text or emit UNK.
SANITY_VOCAB_UNREACHABLE_WARN_COUNT = 0
# C16: count of normalization-unreachable vocab tokens (the introspectable normalizer
# folds the surface to something else, so NO input can ever produce the token). Above
# this count the check FAILs: a vocab token the normalizer guarantees is unreachable
# signals a vocab built without applying the normalizer, a construction defect distinct
# from a merely wasted pretokenizer slot.
SANITY_VOCAB_NORMALIZATION_DEAD_FAIL_COUNT = 0
# C16: fixed multi-domain text used to detect whether a tokenizer merges across
# pretokenizer boundaries (e.g. SuperBPE superwords). If encoding it emits any token
# whose surface contains internal whitespace, the tokenizer is cross-boundary and the
# pretokenizer-unreachable check is skipped (such tokens are reachable by design).
SANITY_CROSS_BOUNDARY_PROBE = (
    "The quick brown fox jumps over the lazy dog. This is a test of the system, and we "
    "want to know whether superword tokens are emitted in practice. In the beginning "
    "there was nothing and then there was something more than before. "
    "def foo(x): return x + 1\n"
    "Les choses que nous faisons ici. El mundo es grande. Wir gehen nach Hause."
)
# C3: on the curated probe set every probe must be clean or lossy_expected.
SANITY_ROUNDTRIP_CLEAN_PASS_FRAC = 1.0
# C3: any red-flag bug -> at least warn; >= fail frac -> fail.
SANITY_ROUNDTRIP_BUG_WARN_FRAC = 0.0
SANITY_ROUNDTRIP_BUG_FAIL_FRAC = 0.01
# C5: whitespace fidelity below this -> WARN (C5 is warn-only by design;
# WordPiece/SentencePiece are intentionally whitespace-lossy).
SANITY_WHITESPACE_FIDELITY_PASS_FRAC = 1.0
# C6: digit chunking consistency = 1 - normalized boundary-pattern entropy.
SANITY_DIGIT_CONSISTENCY_PASS = 0.99
# C6: documents the entropy normalization basis (string, not a numeric magic value).
SANITY_DIGIT_ENTROPY_NORM = "log2(distinct_patterns)"
# C13: per-script UNK rate above which a script is flagged undertrained.
SANITY_UNK_SCRIPT_WARN_RATE = 0.01
# C10: pretokenizer must conserve at least this fraction of input characters.
SANITY_PRETOK_CONSERVATION_FAIL_FRAC = 0.999
# C15: cleaned single-token length above which a token is flagged as an outlier.
SANITY_MAX_REASONABLE_TOKEN_CHARS = 64
# Default cap on FLORES texts per language when --use-sample-data is passed.
SANITY_PROBE_SAMPLES_PER_LANG = 50
