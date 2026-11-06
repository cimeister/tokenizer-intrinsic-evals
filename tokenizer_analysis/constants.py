"""
Constants for tokenizer analysis framework.

This module defines all magic numbers and configuration constants used throughout
the tokenizer analysis codebase to improve maintainability and reduce errors.
"""

from typing import List

from ._migration import make_module_getattr

# Constants used to live on namespace classes (TextProcessing, DataProcessing,
# Statistics, Validation). They are module-level names now, so an old
# `from ... import DataProcessing` would otherwise fail with a bare ImportError.
__getattr__ = make_module_getattr(__name__, ())


# --- Text Processing ---

MIN_PARAGRAPH_LENGTH = 5
MIN_LINE_LENGTH = 5
MIN_SENTENCE_LENGTH = 5
MIN_CONTENT_LENGTH = 5

DEFAULT_CHUNK_SIZE = 500

LARGE_ARRAY_THRESHOLD = 100
ARRAY_SAMPLING_POINTS = 50


# --- Statistics ---

DEFAULT_RENYI_ALPHAS: List[float] = [1.0, 2.0, 2.5, 3.0]
SHANNON_ENTROPY_ALPHA = 1.0

# Value published when a rate has no denominator, i.e. nothing was measured.
# None, not 0.0: a zero here is indistinguishable from a measured zero, and the
# JSON is the artifact people read. Changed in 1.0; see CHANGELOG.
DEFAULT_SAFE_DIVIDE_VALUE = None

# What the console printers show in place of a number that was not computed,
# i.e. wherever the results file carries the null above. Printing 0.000 there
# puts the same misleading zero the results file avoids into the other output a
# reader sees.
MISSING_VALUE_DISPLAY = "n/a"

PERCENTAGE_MULTIPLIER = 100


# --- Results-file schema ---
# Every metric's ``metadata.aggregation`` in analysis_results.json takes one of
# these five values. The label says which average the metric's ``global`` block
# reports. It is published rather than inferred because on a parallel corpus,
# where every language holds the same number of lines, the micro and macro
# averages agree, so the difference only shows up on an unequal corpus and a
# reader cannot tell the two apart from the numbers.

# One ratio computed from summed counts over every item in every language. A
# language with more items counts for more.
AGGREGATION_MICRO_POOLED = "micro_pooled"
# The unweighted mean of the per-language values. Every language counts the
# same regardless of size.
AGGREGATION_MACRO_LANGUAGES = "macro_languages"
# Total units divided by total tokens. Distinct from micro_pooled in that the
# numerator and denominator are different units, not a count of the same thing.
AGGREGATION_RATIO_OF_SUMS = "ratio_of_sums"
# Not an average. The value is a property of the union of the per-language sets.
AGGREGATION_SET_UNION = "set_union"
# The unweighted mean of a per-item ratio, where the item is a document or a
# number rather than a language. Distinct from micro_pooled, which divides
# summed counts: a long document counts the same as a short one here, and on
# the committed benchmark the two differ by up to 18.7% of the value.
AGGREGATION_MEAN_OF_RATIOS = "mean_of_ratios"

AGGREGATION_LABELS = frozenset({
    AGGREGATION_MICRO_POOLED,
    AGGREGATION_MACRO_LANGUAGES,
    AGGREGATION_RATIO_OF_SUMS,
    AGGREGATION_SET_UNION,
    AGGREGATION_MEAN_OF_RATIOS,
})


# --- Validation ---

MIN_LANGUAGES_FOR_GINI = 2
MIN_TOKENIZERS_FOR_PLOTS = 1

MAX_ERROR_DISPLAY_COUNT = 5
MAX_EXAMPLE_DISPLAY_COUNT = 20


# --- Data Processing ---

# Seconds of CER computation allowed per tokenizer before the measurement is
# dropped and reported as null. CER is an O(n*m) edit distance in Python, so on a
# slow tokenizer it can take longer than every other metric combined. One shared
# value, because the CLI default and the library default were 10.0 and 30.0, and
# the same run gave a different answer depending on which entry point started it.
DEFAULT_CER_TIME_BUDGET_S = 10.0

DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_TEXTS_PER_LANGUAGE = 1000
DEFAULT_MAX_SAMPLES = 2000


# --- Merge vocabulary usage (metrics/merge_usage.py) ---

# Thresholds on a merge token's standalone rate (final emissions per million
# final tokens) and survival (final emissions / times formed). Values are those
# of the research script the metric was ported from, fixed 2026-05-28.
MERGE_USAGE_UNCOMMON_RATE_PER_M = 1.0
MERGE_USAGE_RARE_RATE_PER_M = 5.0
MERGE_USAGE_SCAFFOLD_SURVIVAL_MAX = 0.1
# Below this many final tokens the rate thresholds fall under one occurrence
# (1 per million of fewer than a million tokens), so Uncommon means "never
# emitted on its own". The metric still runs, and warns.
MERGE_USAGE_MIN_FINAL_TOKENS_WARN = 1_000_000
# Example tokens written to the full results file.
MERGE_USAGE_EXAMPLE_COUNT = 10
MERGE_USAGE_BYTE_FRAGMENT_EXAMPLE_COUNT = 5


# --- File Formats ---

JSON_EXTENSIONS = ['.json']
TEXT_EXTENSIONS = ['.txt', '.text']
PARQUET_EXTENSIONS = ['.parquet']

TEXT_COLUMN_NAMES = ['text', 'content', 'sentence', 'document', 'passage']

DEFAULT_ENCODING = 'utf-8'
ERROR_HANDLING = 'replace'


# Real UNK tokens are always delimited. Bare 'unk'/'UNK' were removed because they
# match ordinary subwords (e.g. SuperBPE has a plain 'unk' subword from words like
# "junk"/"sunk"), which made get_unk_token_id() misidentify it as the UNK token and
# inflated unk_token_rate. '<|endoftext|>' is GPT-style EOS, not UNK, so also dropped.
UNK_CANDIDATES = ['<unk>', '[UNK]', '<UNK>', '<|unk|>', '\u2047']


# --- Special tokens ---

# Last-resort special-token set, used only when a tokenizer cannot report its own
# declared special tokens (TokenizerWrapper.get_special_token_strings() returns
# None); the caller warns, naming the tokenizer, before using it.
#
# A token treated as special is deleted from reconstructed text and excluded from
# the UTF-8 content-token denominator, so this decision has to come from the
# tokenizer's metadata. It used to come from the surface pattern
# ^(<\||\[).*(\|>|\])$, which was wrong in both directions. It matched ordinary
# content tokens: 2 vocabulary entries of tokenizers/bpe.json (one of them
# '[...]'), 2 of apertus ('[]' and '[][]') and 5 of llama3. It matched neither
# '<s>' nor '</s>', so the Mistral/Llama BOS and EOS were treated as content
# although tokenizers/bpe.json and apertus both declare them special.
#
# The entries are the spellings used by the tokenizer families this package is run
# against: SentencePiece / Llama / Mistral, GPT-2, and BERT / WordPiece. It is a
# guess, not a declaration: it cannot contain tokenizer-specific forms such as
# llama3's '<|reserved_special_token_0|>'.
GENERIC_SPECIAL_TOKENS = frozenset({
    '<s>', '</s>', '<unk>', '<pad>', '<mask>', '<bos>', '<eos>',
    '<|endoftext|>', '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]',
})


# --- Tokenizer Sanity Check ---
# Pass/warn/fail thresholds for the single-tokenizer sanity-check diagnostic
# (tokenizer_analysis/diagnostics/sanity_check.py). Every value here is echoed
# verbatim into the report's metadata.thresholds so results stay traceable.

# C1: a byte-level tokenizer must represent all 256 byte values.
SANITY_BYTE_COVERAGE_REQUIRED = 256
# There was a SANITY_MAX_UNREPRESENTABLE_BYTES_WARN = 0 here, described as
# warning when a byte that is in the vocabulary still fails the behavioral
# round trip. No check ever read it: C1 has four exits and none of them is a
# warn. Because _thresholds_metadata() reflects every SANITY_* name in this
# module, it was still published in sanity_results.json under
# metadata.thresholds, where it read as a threshold governing the result.
# Removed rather than implemented: adding a warn would change the verdict of
# already-published runs.
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
