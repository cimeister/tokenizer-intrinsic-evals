#!/usr/bin/env python3
"""
Text measurement configuration for tokenizer metrics.

This module provides flexible text measurement options for tokenizer analysis,
allowing metrics to be calculated per line, byte, character, or word using
different counting functions for each measurement method.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, fields
import re
from tokenizers.pre_tokenizers import Whitespace, ByteLevel, Sequence

class NormalizationMethod(Enum):
    """Enumeration of available text measurement methods."""
    LINES = "lines"
    BYTES = "bytes" 
    CHARACTERS = "characters"
    WORDS = "words"


class ByteCountingMethod(Enum):
    """Methods for counting bytes."""
    UTF8 = "utf8"  # Standard UTF-8 encoding
    HUGGINGFACE_BYTELEVEL = "hf_bytelevel"  # HuggingFace ByteLevel pretokenizer


class WordCountingMethod(Enum):
    """Methods for counting words."""
    PYTHON_SPLIT = "python_split"  # Default Python .split()
    HUGGINGFACE_WHITESPACE = "hf_whitespace"  # HuggingFace whitespace pretokenizer
    REGEX_WHITESPACE = "regex_whitespace"  # Custom regex for whitespace
    CUSTOM_REGEX = "custom_regex"  # User-defined regex pattern


class LineCountingMethod(Enum):
    """Methods for counting lines."""
    SINGLE = "single"  # Each text = 1 line
    NEWLINE_SPLIT = "newline_split"  # Count actual \n characters
    CUSTOM_REGEX = "custom_regex"  # User-defined regex pattern


@dataclass
class TextMeasurementConfig:
    """Configuration for text measurement methods."""
    
    # Primary measurement method
    method: NormalizationMethod = NormalizationMethod.BYTES
    
    # Method-specific counting configurations
    byte_counting: ByteCountingMethod = ByteCountingMethod.UTF8
    word_counting: WordCountingMethod = WordCountingMethod.PYTHON_SPLIT
    line_counting: LineCountingMethod = LineCountingMethod.SINGLE
    
    # Custom regex overrides the respective counting method
    custom_regex: Optional[str] = None
    
    # Whether to include empty splits (affects word/line counting)
    include_empty_splits: bool = False
    
    def __post_init__(self):
        """Apply custom_regex overrides and validate configuration."""
        if self.custom_regex:
            if self.method == NormalizationMethod.WORDS:
                self.word_counting = WordCountingMethod.CUSTOM_REGEX
            elif self.method == NormalizationMethod.LINES:
                self.line_counting = LineCountingMethod.CUSTOM_REGEX
        
        # Validate that custom_regex is provided when needed
        if ((self.method == NormalizationMethod.WORDS and self.word_counting == WordCountingMethod.CUSTOM_REGEX) or
            (self.method == NormalizationMethod.LINES and self.line_counting == LineCountingMethod.CUSTOM_REGEX)):
            if not self.custom_regex:
                raise ValueError("custom_regex must be provided when using CUSTOM_REGEX counting method")
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TextMeasurementConfig':
        """Create TextMeasurementConfig from dictionary."""
        # Reject unknown keys loudly rather than passing them to the constructor,
        # where they would raise an opaque TypeError. This catches stale key names
        # (e.g. the old "line_counting_method") with a clear, actionable message.
        valid_keys = {f.name for f in fields(cls)}
        unknown = [k for k in config_dict if k not in valid_keys]
        if unknown:
            raise ValueError(
                "Unknown text-measurement config key(s): "
                + ", ".join(sorted(repr(k) for k in unknown))
                + ". Valid keys: " + ", ".join(sorted(valid_keys)) + "."
            )
        # Work on a copy so we never mutate the caller's dict.
        config_dict = dict(config_dict)
        # Convert string values to enums
        if 'method' in config_dict:
            config_dict['method'] = NormalizationMethod(config_dict['method'])
        
        if 'byte_counting' in config_dict:
            config_dict['byte_counting'] = ByteCountingMethod(config_dict['byte_counting'])
            
        if 'word_counting' in config_dict:
            config_dict['word_counting'] = WordCountingMethod(config_dict['word_counting'])
            
        if 'line_counting' in config_dict:
            config_dict['line_counting'] = LineCountingMethod(config_dict['line_counting'])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TextMeasurementConfig to dictionary."""
        return {
            'method': self.method.value,
            'byte_counting': self.byte_counting.value,
            'word_counting': self.word_counting.value,
            'line_counting': self.line_counting.value,
            'custom_regex': self.custom_regex,
            'include_empty_splits': self.include_empty_splits
        }


class TextMeasurer:
    """Measures text properties for normalization purposes."""
    
    def __init__(self, config: TextMeasurementConfig):
        self.config = config
        self._hf_byte_pretokenizer = None
        self._hf_whitespace_pretokenizer = None
        self._compiled_regex = None
        self._initialize_counting_methods()
    
    def _initialize_counting_methods(self):
        """Initialize the specific counting methods based on configuration."""
        if self.config.method == NormalizationMethod.BYTES and self.config.byte_counting == ByteCountingMethod.HUGGINGFACE_BYTELEVEL:
            self._hf_byte_pretokenizer = Sequence([Whitespace(), ByteLevel(use_regex=False)])
        
        if self.config.method == NormalizationMethod.WORDS:
            if self.config.word_counting == WordCountingMethod.HUGGINGFACE_WHITESPACE:
                self._hf_whitespace_pretokenizer = Whitespace()
            elif self.config.word_counting == WordCountingMethod.CUSTOM_REGEX:
                self._compiled_regex = re.compile(self.config.custom_regex)
            elif self.config.word_counting == WordCountingMethod.REGEX_WHITESPACE:
                self._compiled_regex = re.compile(r'\s+')
        
        if self.config.method == NormalizationMethod.LINES:
            if self.config.line_counting == LineCountingMethod.CUSTOM_REGEX:
                self._compiled_regex = re.compile(self.config.custom_regex)
    
    def get_unit_count(self, text: str) -> int:
        """Get the count of units in text according to the configuration."""
        if not text:
            return 0
        
        method_map = {
            NormalizationMethod.LINES: self._count_lines,
            NormalizationMethod.BYTES: self._count_bytes,
            NormalizationMethod.CHARACTERS: self._count_characters,
            NormalizationMethod.WORDS: self._count_words,
        }
        
        return method_map[self.config.method](text)
    
    def _count_lines(self, text: str) -> int:
        """Count lines using the configured method."""
        if self.config.line_counting == LineCountingMethod.SINGLE:
            return 1
        elif self.config.line_counting == LineCountingMethod.NEWLINE_SPLIT:
            return len(text.split('\n'))
        elif self.config.line_counting == LineCountingMethod.CUSTOM_REGEX:
            parts = self._compiled_regex.split(text)
            if not self.config.include_empty_splits:
                parts = [p for p in parts if p.strip()]
            return len(parts)
        else:
            raise ValueError(f"Unsupported line counting method: {self.config.line_counting}")

    def _count_bytes(self, text: str) -> int:
        """Count bytes using the configured method."""
        if self.config.byte_counting == ByteCountingMethod.UTF8:
            return len(text.encode('utf-8'))
        elif self.config.byte_counting == ByteCountingMethod.HUGGINGFACE_BYTELEVEL:
            return self._count_hf_bytes(text)
        else:
            raise ValueError(f"Unsupported byte counting method: {self.config.byte_counting}")
    
    def _count_characters(self, text: str) -> int:
        """Count characters."""
        return len(text)
    
    def _count_words(self, text: str) -> int:
        """Count words using the configured method."""
        method_map = {
            WordCountingMethod.PYTHON_SPLIT: self._count_words_python_split,
            WordCountingMethod.HUGGINGFACE_WHITESPACE: self._count_words_hf_whitespace,
            WordCountingMethod.REGEX_WHITESPACE: self._count_words_regex,
            WordCountingMethod.CUSTOM_REGEX: self._count_words_regex,
        }
        return method_map[self.config.word_counting](text)
    
    def _count_hf_bytes(self, text: str) -> int:
        """Count bytes using HuggingFace ByteLevel pretokenizer."""
        if not text.strip():
            return 0
        pretokenized = self._hf_byte_pretokenizer.pre_tokenize_str(text)
        return sum(len(token_str) for token_str, _ in pretokenized)
    
    def _count_words_python_split(self, text: str) -> int:
        """Count words using Python's split method."""
        words = text.split()
        return len(words) if not self.config.include_empty_splits else len(text.split(' '))
    
    def _count_words_hf_whitespace(self, text: str) -> int:
        """Count words using HuggingFace Whitespace pretokenizer."""
        pretokenized = self._hf_whitespace_pretokenizer.pre_tokenize_str(text)
        return len(pretokenized)
    
    def _count_words_regex(self, text: str) -> int:
        """Count words using regex splitting."""
        parts = self._compiled_regex.split(text)
        if not self.config.include_empty_splits:
            parts = [p for p in parts if p.strip()]
        return len(parts)
    
    def get_unit_label(self) -> str:
        """Get short description for plot labels."""
        return {
            NormalizationMethod.LINES: "line",
            NormalizationMethod.BYTES: "byte",
            NormalizationMethod.CHARACTERS: "char",
            NormalizationMethod.WORDS: "word"
        }[self.config.method]


def create_default_configs() -> Dict[str, TextMeasurementConfig]:
    """Create a set of common text measurement configurations."""
    return {
        'bytes': TextMeasurementConfig(
            method=NormalizationMethod.BYTES,
            byte_counting=ByteCountingMethod.UTF8
        ),
        'bytes_hf': TextMeasurementConfig(
            method=NormalizationMethod.BYTES,
            byte_counting=ByteCountingMethod.HUGGINGFACE_BYTELEVEL
        ),
        'characters': TextMeasurementConfig(
            method=NormalizationMethod.CHARACTERS
        ),
        'lines': TextMeasurementConfig(
            method=NormalizationMethod.LINES,
            line_counting=LineCountingMethod.SINGLE
        ),
        'lines_newline': TextMeasurementConfig(
            method=NormalizationMethod.LINES,
            line_counting=LineCountingMethod.NEWLINE_SPLIT
        ),
        'words_python': TextMeasurementConfig(
            method=NormalizationMethod.WORDS,
            word_counting=WordCountingMethod.PYTHON_SPLIT
        ),
        'words_hf': TextMeasurementConfig(
            method=NormalizationMethod.WORDS,
            word_counting=WordCountingMethod.HUGGINGFACE_WHITESPACE
        ),
        'words_regex': TextMeasurementConfig(
            method=NormalizationMethod.WORDS,
            word_counting=WordCountingMethod.REGEX_WHITESPACE
        )
    }


# Default configuration
DEFAULT_TEXT_MEASUREMENT_CONFIG = create_default_configs()['bytes'] 
DEFAULT_LINE_MEASUREMENT_CONFIG = create_default_configs()['lines'] 
DEFAULT_WORD_MEASUREMENT_CONFIG = create_default_configs()['words_python'] 