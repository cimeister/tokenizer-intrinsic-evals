"""
Utility functions for tokenizer analysis.
Attempts to import from parent codebase first, falls back to standalone versions.
"""

import math
import json
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace, ByteLevel, Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import AutoTokenizer


def load_tokenizer_from_config(config, name: str = "tokenizer"):
    """
    Load tokenizer wrapper from configuration (here for backwards compatibility)
    
    Args:
        config: Configuration dictionary
        name: Tokenizer name (for the wrapper)
        
    Returns:
        TokenizerWrapper instance
    """
    from ..core.tokenizer_wrapper import create_tokenizer_wrapper
    logger.warning("Deprecated function; Use `create_tokenizer_wrapper` in core.tokenizer_wrapper instead")
    return create_tokenizer_wrapper(name, config)


def _load_huggingface_tokenizer(config):
    """
    Internal function to load raw HuggingFace tokenizer from configuration.
    
    This function is used by the HuggingFaceTokenizer wrapper.
    Tries to use original implementation first, falls back to simplified version.
    """
 
    path = config['path']
        
    # Strategy 1: If path points to a JSON file, use Tokenizer.from_file
    if path.endswith('.json') or os.path.isfile(path):
        try:
            logger.info(f"Loading tokenizer from file: {path}")
            tokenizer = Tokenizer.from_file(path)
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from file {path}: {e}")
    
    # Strategy 2: Try loading as HuggingFace tokenizer (directory or model name).
    # Prefer PreTrainedTokenizerFast which loads from tokenizer.json directly.
    # Some repos (e.g. DeepSeek) have a tokenizer.json with full vocabulary
    # but AutoTokenizer resolves to a LlamaTokenizer whose SentencePiece
    # model is incomplete (e.g. missing CJK characters).
    try:
        logger.info(f"Loading tokenizer from HuggingFace: {path}")
        from transformers import PreTrainedTokenizerFast
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
            logger.info(f"Loaded {path} via PreTrainedTokenizerFast")
            return tokenizer
        except Exception:
            pass
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        logger.warning(f"Failed to load HuggingFace tokenizer from {path}: {e}")

    # Strategy 3: If path is a directory, look for tokenizer files
    if os.path.isdir(path):
        # Look for common tokenizer file names
        for filename in ['tokenizer.json', 'vocab.json', 'merges.txt']:
            file_path = os.path.join(path, filename)
            if os.path.exists(file_path):
                try:
                    if filename == 'tokenizer.json':
                        logger.info(f"Loading tokenizer from {file_path}")
                        return Tokenizer.from_file(file_path)
                    elif filename == 'vocab.json' and os.path.exists(os.path.join(path, 'merges.txt')):
                        # Load as BPE tokenizer
                        logger.info(f"Loading BPE tokenizer from directory: {path}")
                        return _load_bpe_from_directory(path)
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer from {file_path}: {e}")
                    continue
    
    raise ValueError(f"Could not load tokenizer from {path}.")


def _load_custom_bpe_from_directory(config):
    """Helper function to load BPE tokenizer from directory with vocab.json and merges.txt"""
    directory_path = config['path']
    vocab_file = os.path.join(directory_path, "vocab.json")
    merges_file = os.path.join(directory_path, "merges.txt")
    
    # Load vocab and merges from files
    with open(vocab_file, "r", encoding="utf-8") as vf:
        vocab = json.load(vf)
    with open(merges_file, "r", encoding="utf-8") as mf:
        merges = [tuple(line.strip().split()) for line in mf if not line.startswith("#")]

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges))

    # Set pre-tokenizer and matching decoder
    tokenizer.pre_tokenizer = Sequence([Whitespace(), ByteLevel(use_regex=False)])
    tokenizer.decoder = ByteLevelDecoder()

    # Set special tokens
    tokenizer.add_special_tokens(["<s>", "</s>", "<unk>", "<pad>"])
    tokenizer.model.unk_token = "<unk>"

    return tokenizer


def setup_environment():
    """Setup environment for tokenizer analysis."""
    # Basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
