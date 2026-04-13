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

UNK_CANDIDATES = ['<unk>', '[UNK]', '<UNK>', 'unk', 'UNK', '\u2047', '<|endoftext|>']
