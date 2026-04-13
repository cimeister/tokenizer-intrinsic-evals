"""
Comprehensive validation and error handling for the unified tokenizer analysis system.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass

from .input_types import TokenizedData, InputSpecification, VocabularyProvider
from .input_providers import InputProvider

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    
    valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]
    metadata: Optional[Dict[str, Any]] = None
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.valid = False
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_info(self, message: str):
        """Add an info message."""
        self.info.append(message)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        if not other.valid:
            self.valid = False
        
        if other.metadata:
            if self.metadata is None:
                self.metadata = {}
            self.metadata.update(other.metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'valid': self.valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'metadata': self.metadata or {}
        }


class TokenizedDataValidator:
    """Validator for TokenizedData objects."""
    
    @staticmethod
    def validate_single(data: TokenizedData, 
                       vocab_size: Optional[int] = None,
                       expected_tokenizer: Optional[str] = None,
                       expected_language: Optional[str] = None) -> ValidationResult:
        """
        Validate a single TokenizedData object.
        
        Args:
            data: TokenizedData object to validate
            vocab_size: Optional vocabulary size to check against
            expected_tokenizer: Expected tokenizer name
            expected_language: Expected language
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True, errors=[], warnings=[], info=[])
        
        # Tokenizer name validation
        if expected_tokenizer and data.tokenizer_name != expected_tokenizer:
            result.add_error(f"Expected tokenizer '{expected_tokenizer}', got '{data.tokenizer_name}'")
        
        # Language validation
        if expected_language and data.language != expected_language:
            result.add_error(f"Expected language '{expected_language}', got '{data.language}'")
        
        # Token validation
        if not data.tokens:
            result.add_warning("Empty token list")
        else:
            # Check for negative token IDs
            negative_tokens = [t for t in data.tokens if t < 0]
            if negative_tokens:
                result.add_error(f"Found negative token IDs: {negative_tokens[:5]}{'...' if len(negative_tokens) > 5 else ''}")
            
            # Check vocabulary bounds
            if vocab_size is not None:
                out_of_bounds = [t for t in data.tokens if t >= vocab_size]
                if out_of_bounds:
                    result.add_error(f"Token IDs exceed vocabulary size {vocab_size}: {out_of_bounds[:5]}{'...' if len(out_of_bounds) > 5 else ''}")
        
        # Text-token consistency validation
        if data.text and data.tokens:
            text_len = len(data.text)
            token_count = len(data.tokens)
            
            # Basic sanity checks
            if text_len == 0:
                result.add_warning("Empty text with non-empty tokens")
            elif token_count > text_len:
                result.add_warning(f"More tokens ({token_count}) than characters ({text_len}) - unusual but possible")
            elif token_count == 0:
                result.add_warning("Empty tokens with non-empty text")
        
        # Metadata validation
        if data.metadata:
            if not isinstance(data.metadata, dict):
                result.add_error("Metadata must be a dictionary")
        
        return result
    
    @staticmethod
    def validate_batch(data_list: List[TokenizedData],
                      vocab_size: Optional[int] = None,
                      expected_tokenizer: Optional[str] = None,
                      expected_languages: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate a batch of TokenizedData objects.
        
        Args:
            data_list: List of TokenizedData objects
            vocab_size: Optional vocabulary size to check against
            expected_tokenizer: Expected tokenizer name
            expected_languages: Expected languages
            
        Returns:
            ValidationResult with aggregated results
        """
        result = ValidationResult(valid=True, errors=[], warnings=[], info=[])
        
        if not data_list:
            result.add_warning("Empty data list")
            return result
        
        # Validate each item
        for i, data in enumerate(data_list):
            item_result = TokenizedDataValidator.validate_single(
                data, vocab_size, expected_tokenizer
            )
            
            # Prefix errors/warnings with item index
            for error in item_result.errors:
                result.add_error(f"Item {i}: {error}")
            for warning in item_result.warnings:
                result.add_warning(f"Item {i}: {warning}")
        
        # Cross-item validation
        tokenizer_names = set(data.tokenizer_name for data in data_list)
        if len(tokenizer_names) > 1:
            result.add_warning(f"Multiple tokenizer names in batch: {tokenizer_names}")
        
        languages = set(data.language for data in data_list)
        if expected_languages:
            unexpected_languages = languages - set(expected_languages)
            if unexpected_languages:
                result.add_error(f"Unexpected languages: {unexpected_languages}")
        
        # Statistics
        total_tokens = sum(len(data.tokens) for data in data_list)
        total_texts = sum(1 for data in data_list if data.text)
        
        result.metadata = {
            'total_items': len(data_list),
            'total_tokens': total_tokens,
            'items_with_text': total_texts,
            'unique_tokenizers': len(tokenizer_names),
            'unique_languages': len(languages),
            'languages': sorted(list(languages))
        }
        
        return result


class InputProviderValidator:
    """Validator for InputProvider objects."""
    
    @staticmethod
    def validate_provider(provider: InputProvider) -> ValidationResult:
        """
        Validate an InputProvider.
        
        Args:
            provider: InputProvider to validate
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True, errors=[], warnings=[], info=[])
        
        try:
            # Basic interface compliance
            tokenizer_names = provider.get_tokenizer_names()
            if not tokenizer_names:
                result.add_error("No tokenizer names provided")
                return result
            
            # Get tokenized data
            tokenized_data = provider.get_tokenized_data()
            if not tokenized_data:
                result.add_error("No tokenized data provided")
                return result
            
            # Check consistency between interface methods
            data_tokenizers = set(tokenized_data.keys())
            name_tokenizers = set(tokenizer_names)
            
            missing_in_data = name_tokenizers - data_tokenizers
            extra_in_data = data_tokenizers - name_tokenizers
            
            if missing_in_data:
                result.add_error(f"Tokenizers missing from data: {missing_in_data}")
            if extra_in_data:
                result.add_warning(f"Extra tokenizers in data: {extra_in_data}")
            
            # Validate each tokenizer's data
            tokenizer_stats = {}
            for tok_name in tokenizer_names:
                if tok_name not in tokenized_data:
                    continue
                
                try:
                    vocab_size = provider.get_vocab_size(tok_name)
                    languages = provider.get_languages(tok_name)
                    
                    # Validate tokenized data for this tokenizer
                    tok_result = TokenizedDataValidator.validate_batch(
                        tokenized_data[tok_name],
                        vocab_size=vocab_size,
                        expected_tokenizer=tok_name,
                        expected_languages=languages
                    )
                    
                    # Merge results
                    for error in tok_result.errors:
                        result.add_error(f"Tokenizer {tok_name}: {error}")
                    for warning in tok_result.warnings:
                        result.add_warning(f"Tokenizer {tok_name}: {warning}")
                    
                    tokenizer_stats[tok_name] = tok_result.metadata
                    
                except Exception as e:
                    result.add_error(f"Error validating tokenizer {tok_name}: {e}")
            
            # Overall statistics
            all_languages = set()
            total_data_points = 0
            for tok_data in tokenized_data.values():
                for data in tok_data:
                    all_languages.add(data.language)
                total_data_points += len(tok_data)
            
            result.metadata = {
                'num_tokenizers': len(tokenizer_names),
                'num_languages': len(all_languages),
                'total_data_points': total_data_points,
                'languages': sorted(list(all_languages)),
                'tokenizer_stats': tokenizer_stats
            }
            
        except Exception as e:
            result.add_error(f"Provider validation failed: {e}")
        
        return result


class InputSpecificationValidator:
    """Validator for InputSpecification objects."""
    
    @staticmethod
    def validate_specification(spec: InputSpecification) -> ValidationResult:
        """
        Validate an InputSpecification.
        
        Args:
            spec: InputSpecification to validate
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True, errors=[], warnings=[], info=[])
        
        if spec.is_raw_mode:
            result.merge(InputSpecificationValidator._validate_raw_mode(spec))
        elif spec.is_pretokenized_mode:
            result.merge(InputSpecificationValidator._validate_pretokenized_mode(spec))
        else:
            result.add_error("Specification is neither raw nor pre-tokenized mode")
        
        return result
    
    @staticmethod
    def _validate_raw_mode(spec: InputSpecification) -> ValidationResult:
        """Validate raw mode specification."""
        result = ValidationResult(valid=True, errors=[], warnings=[], info=[])
        
        # Check tokenizer
        if not hasattr(spec.tokenizer, 'encode'):
            result.add_error("Tokenizer missing 'encode' method")
        if not hasattr(spec.tokenizer, 'vocab_size'):
            result.add_warning("Tokenizer missing 'vocab_size' property")
        
        # Check texts
        if not spec.texts:
            result.add_error("No texts provided for raw mode")
        else:
            empty_texts = [lang for lang, text in spec.texts.items() if not text or not text.strip()]
            if empty_texts:
                result.add_warning(f"Empty texts for languages: {empty_texts}")
        
        return result
    
    @staticmethod
    def _validate_pretokenized_mode(spec: InputSpecification) -> ValidationResult:
        """Validate pre-tokenized mode specification."""
        result = ValidationResult(valid=True, errors=[], warnings=[], info=[])
        
        # Check vocabulary
        if not hasattr(spec.vocabulary, 'vocab_size'):
            result.add_error("Vocabulary missing 'vocab_size' property")
        else:
            vocab_size = spec.vocabulary.vocab_size
            if vocab_size <= 0:
                result.add_error(f"Invalid vocabulary size: {vocab_size}")
        
        # Check tokenized data
        if not spec.tokenized_data:
            result.add_error("No tokenized data provided")
        else:
            data_result = TokenizedDataValidator.validate_batch(
                spec.tokenized_data,
                vocab_size=getattr(spec.vocabulary, 'vocab_size', None),
                expected_tokenizer=spec.tokenizer_name
            )
            result.merge(data_result)
        
        return result


class AnalysisValidator:
    """High-level validator for complete analysis setup."""
    
    @staticmethod
    def validate_analysis_setup(input_provider: InputProvider,
                               normalization_config: Optional[Any] = None,
                               language_metadata: Optional[Any] = None) -> ValidationResult:
        """
        Validate complete analysis setup.
        
        Args:
            input_provider: InputProvider to validate
            normalization_config: Optional normalization configuration
            language_metadata: Optional language metadata
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True, errors=[], warnings=[], info=[])
        
        # Validate input provider
        provider_result = InputProviderValidator.validate_provider(input_provider)
        result.merge(provider_result)
        
        if not result.valid:
            return result
        
        # Check minimum requirements for analysis
        tokenizer_names = input_provider.get_tokenizer_names()
        languages = input_provider.get_languages()
        
        if len(tokenizer_names) < 1:
            result.add_error("At least one tokenizer required for analysis")
        elif len(tokenizer_names) == 1:
            result.add_info("Single tokenizer analysis - pairwise comparisons will be skipped")
        
        if len(languages) < 1:
            result.add_error("At least one language required for analysis")
        elif len(languages) == 1:
            result.add_info("Single language analysis - cross-language comparisons will be limited")
        
        # Validate data coverage
        tokenized_data = input_provider.get_tokenized_data()
        coverage_stats = {}
        
        for tok_name in tokenizer_names:
            if tok_name in tokenized_data:
                tok_languages = set(data.language for data in tokenized_data[tok_name])
                coverage = len(tok_languages) / len(languages) if languages else 0
                coverage_stats[tok_name] = {
                    'covered_languages': len(tok_languages),
                    'total_languages': len(languages),
                    'coverage_ratio': coverage
                }
                
                if coverage < 0.5:
                    result.add_warning(f"Tokenizer {tok_name} has low language coverage: {coverage:.1%}")
        
        # Validate normalization config
        if normalization_config:
            if not hasattr(normalization_config, 'method'):
                result.add_warning("Normalization config missing 'method' attribute")
        
        # Validate language metadata
        if language_metadata:
            if hasattr(language_metadata, 'analysis_groups'):
                groups = language_metadata.analysis_groups
                result.add_info(f"Language metadata contains {len(groups)} analysis group types: {list(groups.keys())}")
                
                # Check if analysis languages are covered by metadata
                metadata_languages = set()
                for group_dict in groups.values():
                    for lang_list in group_dict.values():
                        metadata_languages.update(lang_list)
                
                missing_in_metadata = set(languages) - metadata_languages
                if missing_in_metadata:
                    result.add_warning(f"Languages not in metadata groups: {missing_in_metadata}")
        
        result.metadata = {
            'tokenizer_count': len(tokenizer_names),
            'language_count': len(languages),
            'coverage_stats': coverage_stats,
            'has_normalization_config': normalization_config is not None,
            'has_language_metadata': language_metadata is not None
        }
        
        return result


def validate_and_report(validation_result: ValidationResult, 
                       logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    Log validation results and return success status.
    
    Args:
        validation_result: ValidationResult to report
        logger_instance: Optional logger to use (defaults to module logger)
        
    Returns:
        True if validation passed, False otherwise
    """
    log = logger_instance or logger
    
    # Log errors
    for error in validation_result.errors:
        log.error(error)

    for warning in validation_result.warnings:
        log.warning(warning)

    for info in validation_result.info:
        log.info(info)

    if validation_result.valid:
        log.info("Validation passed")
    else:
        log.error(f"Validation failed with {len(validation_result.errors)} errors")
    
    return validation_result.valid