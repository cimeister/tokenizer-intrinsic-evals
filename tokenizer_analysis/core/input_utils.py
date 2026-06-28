"""
Utilities for input data validation, conversion, and helper functions.
"""

from typing import Dict, List, Any, Union, Optional, Tuple
import json
import pickle
from pathlib import Path
import logging

from .input_types import (
    TokenizedData, InputSpecification, VocabularyProvider
)
from .input_providers import create_input_provider, InputProvider

logger = logging.getLogger(__name__)


class SimpleVocabulary(VocabularyProvider):
    """Simple vocabulary provider that wraps vocab size and optional vocab dict."""
    
    def __init__(self, vocab_size: int, vocab_dict: Optional[Dict[str, int]] = None):
        """
        Initialize simple vocabulary.
        
        Args:
            vocab_size: Size of vocabulary
            vocab_dict: Optional vocabulary mapping
        """
        self._vocab_size = vocab_size
        self._vocab_dict = vocab_dict or {}
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        return self._vocab_dict


class InputLoader:
    """Utilities for loading input data from various sources."""
    
    @staticmethod
    def load_tokenized_data_from_json(file_path: Union[str, Path]) -> List[TokenizedData]:
        """
        Load tokenized data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of TokenizedData objects
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        
        tokenized_data = []
        for item in data_list:
            data = TokenizedData.from_dict(item)
            tokenized_data.append(data)
        
        return tokenized_data
    
    @staticmethod
    def save_tokenized_data_to_json(tokenized_data: List[TokenizedData], 
                                   file_path: Union[str, Path]) -> None:
        """
        Save tokenized data to JSON file.
        
        Args:
            tokenized_data: List of TokenizedData objects
            file_path: Path to save JSON file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data_list = [data.to_dict() for data in tokenized_data]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(tokenized_data)} tokenized data entries to {file_path}")
    
    @staticmethod
    def load_tokenized_data_from_pickle(file_path: Union[str, Path]) -> List[TokenizedData]:
        """
        Load tokenized data from pickle file.
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            List of TokenizedData objects
        """
        file_path = Path(file_path)
        
        with open(file_path, 'rb') as f:
            data_list = pickle.load(f)
        
        # Validate that all items are TokenizedData
        if not all(isinstance(item, TokenizedData) for item in data_list):
            raise ValueError("Pickle file does not contain valid TokenizedData objects")
        
        return data_list
    
    @staticmethod
    def save_tokenized_data_to_pickle(tokenized_data: List[TokenizedData], 
                                     file_path: Union[str, Path]) -> None:
        """
        Save tokenized data to pickle file.
        
        Args:
            tokenized_data: List of TokenizedData objects
            file_path: Path to save pickle file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(tokenized_data, f)
        
        logger.info(f"Saved {len(tokenized_data)} tokenized data entries to {file_path}")
    
    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> Dict[str, List[TokenizedData]]:
        """
        Load tokenized data from file (auto-detects format).
        
        Args:
            file_path: Path to tokenized data file (.json or .pkl)
            
        Returns:
            Dictionary mapping tokenizer names to tokenized data lists
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.json':
            # JSON format - expect dict with tokenizer names as keys
            with open(file_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            
            result = {}
            for tok_name, data_list in data_dict.items():
                tokenized_data = []
                for item in data_list:
                    data = TokenizedData.from_dict(item)
                    tokenized_data.append(data)
                result[tok_name] = tokenized_data
            
            return result
            
        elif file_path.suffix == '.pkl':
            # Pickle format - expect dict mapping tokenizer names to data lists
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            # Validate structure
            if not isinstance(data_dict, dict):
                raise ValueError("Pickle file should contain a dictionary mapping tokenizer names to data lists")
            
            for tok_name, data_list in data_dict.items():
                if not isinstance(data_list, list):
                    raise ValueError(f"Data for tokenizer '{tok_name}' should be a list")
                if not all(isinstance(item, TokenizedData) for item in data_list):
                    raise ValueError(f"All items for tokenizer '{tok_name}' should be TokenizedData objects")
            
            return data_dict
            
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .json or .pkl")
    
    @staticmethod
    def load_vocabularies_from_config(vocab_config: Dict[str, str]):
        """
        Load vocabularies from text files specified in configuration.
        
        Args:
            vocab_config: Dictionary mapping tokenizer names to vocabulary file paths
            
        Returns:
            Dictionary mapping tokenizer names to TokenizerWrapper objects
        """
        from .tokenizer_wrapper import PreTokenizedDataTokenizer
        
        vocabularies = {}
        for tok_name, vocab_file_path in vocab_config.items():
            vocab_path = Path(vocab_file_path)
            
            if vocab_path.exists():
                try:
                    with open(vocab_path, 'r', encoding='utf-8') as f:
                        vocab_tokens = [line.strip() for line in f if line.strip()]
                    
                    # Create vocab dict mapping tokens to indices
                    vocab_dict = {token: idx for idx, token in enumerate(vocab_tokens)}
                    vocabularies[tok_name] = PreTokenizedDataTokenizer(tok_name, len(vocab_tokens), vocab_dict)
                    logger.info(f"Loaded vocabulary for {tok_name} from {vocab_path} ({len(vocab_tokens)} tokens)")
                except Exception as e:
                    logger.warning(f"Failed to load vocabulary for {tok_name} from {vocab_path}: {e}")
            else:
                logger.warning(f"Vocabulary file not found for {tok_name}: {vocab_path}")
        
        return vocabularies


class InputValidator:
    """Validation for input data."""
    
    @staticmethod
    def validate_tokenized_data(tokenized_data: List[TokenizedData], 
                              expected_tokenizer_name: Optional[str] = None,
                              expected_languages: Optional[List[str]] = None,
                              max_token_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate list of tokenized data.
        
        Args:
            tokenized_data: List of TokenizedData to validate
            expected_tokenizer_name: Expected tokenizer name (optional)
            expected_languages: Expected languages (optional)
            max_token_id: Maximum valid token ID (optional)
            
        Returns:
            Validation report dictionary
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {
                'total_entries': len(tokenized_data),
                'tokenizer_names': set(),
                'languages': set(),
                'total_tokens': 0,
                'unique_tokens': set(),
                'token_range': None
            }
        }
        
        if not tokenized_data:
            report['errors'].append("No tokenized data provided")
            report['valid'] = False
            return report
        
        for i, data in enumerate(tokenized_data):
            try:
                # Basic validation (handled by TokenizedData.__post_init__)
                pass
            except Exception as e:
                report['errors'].append(f"Entry {i}: {e}")
                report['valid'] = False
                continue
            
            # Collect stats
            report['stats']['tokenizer_names'].add(data.tokenizer_name)
            report['stats']['languages'].add(data.language)
            report['stats']['total_tokens'] += len(data.tokens)
            report['stats']['unique_tokens'].update(data.tokens)
            
            # Validate tokenizer name
            if expected_tokenizer_name and data.tokenizer_name != expected_tokenizer_name:
                report['errors'].append(
                    f"Entry {i}: Expected tokenizer '{expected_tokenizer_name}', "
                    f"got '{data.tokenizer_name}'"
                )
                report['valid'] = False
            
            # Validate language
            if expected_languages and data.language not in expected_languages:
                report['errors'].append(
                    f"Entry {i}: Unexpected language '{data.language}'. "
                    f"Expected one of: {expected_languages}"
                )
                report['valid'] = False
            
            # Validate token IDs
            if data.tokens:
                min_token = min(data.tokens)
                max_token = max(data.tokens)
                
                if min_token < 0:
                    report['errors'].append(
                        f"Entry {i}: Negative token ID found: {min_token}"
                    )
                    report['valid'] = False
                
                if max_token_id is not None and max_token >= max_token_id:
                    report['errors'].append(
                        f"Entry {i}: Token ID {max_token} exceeds vocabulary size {max_token_id}"
                    )
                    report['valid'] = False
        
        # Final stats
        if report['stats']['unique_tokens']:
            min_token = min(report['stats']['unique_tokens'])
            max_token = max(report['stats']['unique_tokens'])
            report['stats']['token_range'] = (min_token, max_token)
        
        # Convert sets to lists for JSON serialization
        report['stats']['tokenizer_names'] = list(report['stats']['tokenizer_names'])
        report['stats']['languages'] = list(report['stats']['languages'])
        report['stats']['unique_tokens'] = len(report['stats']['unique_tokens'])
        
        return report
    
    @staticmethod
    def validate_input_provider(input_provider: InputProvider) -> Dict[str, Any]:
        """
        Validate InputProvider instance.
        
        Args:
            input_provider: InputProvider to validate
            
        Returns:
            Validation report dictionary
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'tokenizer_reports': {}
        }
        
        try:
            # Basic validation
            if not input_provider.validate_data():
                report['errors'].append("InputProvider failed internal validation")
                report['valid'] = False
            
            tokenizer_names = input_provider.get_tokenizer_names()
            tokenized_data = input_provider.get_tokenized_data()
            
            # Validate each tokenizer
            for tok_name in tokenizer_names:
                if tok_name not in tokenized_data:
                    report['errors'].append(f"No data found for tokenizer {tok_name}")
                    report['valid'] = False
                    continue
                
                try:
                    vocab_size = input_provider.get_vocab_size(tok_name)
                    languages = input_provider.get_languages(tok_name)
                    
                    # Validate tokenized data for this tokenizer
                    tok_report = InputValidator.validate_tokenized_data(
                        tokenized_data[tok_name],
                        expected_tokenizer_name=tok_name,
                        expected_languages=languages,
                        max_token_id=vocab_size
                    )
                    
                    report['tokenizer_reports'][tok_name] = tok_report
                    
                    if not tok_report['valid']:
                        report['valid'] = False
                        report['errors'].extend([
                            f"Tokenizer {tok_name}: {error}" 
                            for error in tok_report['errors']
                        ])
                
                except Exception as e:
                    report['errors'].append(f"Error validating tokenizer {tok_name}: {e}")
                    report['valid'] = False
        
        except Exception as e:
            report['errors'].append(f"Error during validation: {e}")
            report['valid'] = False
        
        return report


def create_simple_specifications(tokenizer_text_pairs: Dict[str, Tuple['TokenizerWrapper', Dict[str, Union[str, List[str]]]]]) -> Dict[str, InputSpecification]:
    """
    Helper function to create InputSpecifications from simple tokenizer+text pairs.
    
    Args:
        tokenizer_text_pairs: Dict mapping tokenizer_name -> (tokenizer, texts)
                             where texts is Dict[str, Union[str, List[str]]]
        
    Returns:
        Dict of InputSpecification objects in raw mode
    """
    specifications = {}
    
    for name, (tokenizer, texts) in tokenizer_text_pairs.items():
        spec = InputSpecification(
            tokenizer=tokenizer,
            texts=texts,
            metadata={'source': 'simple_creation'}
        )
        specifications[name] = spec
    
    return specifications