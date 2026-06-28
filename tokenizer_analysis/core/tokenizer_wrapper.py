"""
Tokenizer wrapper classes for unified tokenizer interface.

This module provides a common interface for different tokenizer types,
making it easy for users to integrate custom tokenizers into the framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import logging
import os
import glob

from ..constants import UNK_CANDIDATES

logger = logging.getLogger(__name__)


def _setup_fast_decode(tok):
    """Detect and return a fast decode callable for skip_special_tokens=True, or None."""
    # Path A: transformers Fast tokenizer with Rust backend
    backend = getattr(tok, 'backend_tokenizer', None)
    if backend is not None and hasattr(backend, 'decode'):
        return lambda ids: backend.decode(ids, skip_special_tokens=True)

    # Path B: tiktoken-backed tokenizer
    try:
        import tiktoken
    except ImportError:
        return None

    enc = None
    for attr_name in ('tokenizer', 'model', 'encoder'):
        obj = getattr(tok, attr_name, None)
        if isinstance(obj, tiktoken.Encoding):
            enc = obj
            break
    if enc is None:
        for obj in vars(tok).values():
            if isinstance(obj, tiktoken.Encoding):
                enc = obj
                break
    if enc is None:
        return None

    special_ids = frozenset(tok.all_special_ids) if hasattr(tok, 'all_special_ids') else frozenset()
    if special_ids:
        return lambda ids: enc.decode([i for i in ids if i not in special_ids])
    return enc.decode


class TokenizerWrapper(ABC):
    """Abstract interface for tokenizer wrappers."""
    
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        pass
    
    @abstractmethod
    def get_vocab(self) -> Optional[Dict[str, int]]:
        """Get vocabulary mapping. Returns None if not available."""
        pass
    
    @abstractmethod
    def can_encode(self) -> bool:
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Raises NotImplementedError if can_encode() returns False."""
        pass
    
    @abstractmethod
    def can_pretokenize(self) -> bool:
        pass
        
    @abstractmethod
    def pretokenize(self, text: str) -> List[str]:
        """Raises NotImplementedError if can_pretokenize() returns False."""
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'TokenizerWrapper':
        """Create tokenizer wrapper from config."""
        pass
    
    def can_decode(self) -> bool:
        return False

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> Optional[str]:
        """Decode token IDs to text. Returns None if not supported."""
        return None

    def encode_with_offsets(self, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        """Encode text and return ``(token_ids, offsets)``.

        ``offsets[i]`` is ``(start_char, end_char)`` in the original *text*
        for token *i*.  Tokens that do not map to source characters (e.g.
        ``<s>``, ``</s>``) should have offset ``(0, 0)``.

        The default implementation returns ``(self.encode(text), None)``
        (offsets not supported).  Subclasses should override when the
        underlying tokenizer library provides character-offset information.
        """
        return self.encode(text), None

    def encode_batch_with_offsets(
        self, texts: List[str]
    ) -> List[Tuple[List[int], Optional[List[Tuple[int, int]]]]]:
        """Encode a batch of texts, returning ``(token_ids, offsets)`` per text.

        The default implementation loops over :meth:`encode_with_offsets`.
        Subclasses should override when the underlying library provides a
        native batch API for better throughput.
        """
        return [self.encode_with_offsets(text) for text in texts]

    def get_underlying_tokenizer(self):
        """
        Get the underlying raw tokenizer object if available.

        Returns:
            The raw tokenizer object or None if not available.

        Note:
            This method is primarily for specialized use cases like MorphScore
            that require direct access to tokenizer internals.
        """
        return None

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs to token strings.

        Default implementation reverses :meth:`get_vocab`.  Subclasses should
        override with a more efficient method when available.
        """
        vocab = self.get_vocab()
        if vocab:
            id_to_token = {v: k for k, v in vocab.items()}
            return [id_to_token.get(tid, f"<UNK_{tid}>") for tid in token_ids]
        return [f"<UNK_{tid}>" for tid in token_ids]

    def get_unk_token_id(self) -> Optional[int]:
        """
        Get the UNK token ID if available.

        Returns:
            The UNK token ID or None if not available.
        """
        return None

    def has_unk_token(self) -> bool:
        return self.get_unk_token_id() is not None

    def get_special_token_ids(self) -> set:
        """IDs of tokens declared special in the tokenizer's metadata. Default: none."""
        return set()

    def get_metadata(self) -> Dict[str, Any]:
        """Get additional tokenizer metadata."""
        return {
            "name": self.get_name(), 
            "vocab_size": self.get_vocab_size(),
            "can_encode": self.can_encode(),
            "can_pretokenize": self.can_pretokenize()
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.get_name()}', vocab_size={self.get_vocab_size()})"


class HuggingFaceTokenizer(TokenizerWrapper):
    """Wrapper for HuggingFace tokenizers."""
    
    def __init__(self, name: str, tokenizer, config: Dict[str, Any]):
        """
        Initialize HuggingFace tokenizer wrapper.
        
        Args:
            name: Tokenizer name
            tokenizer: HuggingFace tokenizer instance
            config: Original configuration dict
        """
        self._name = name
        self._tokenizer = tokenizer
        self._config = config
        self._fast_decode = _setup_fast_decode(tokenizer)
        logger.info("Creating HF tokenizer")
    
    def get_name(self) -> str:
        return self._name
    
    def get_vocab_size(self) -> int:
        return len(self._tokenizer.get_vocab())
    
    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab()
    
    def can_encode(self) -> bool:
        return True
    
    def encode(self, text: str) -> List[int]:
        result = self._tokenizer.encode(text, add_special_tokens=False)
        # Handle different return types
        if hasattr(result, 'ids'):
            return result.ids
        elif isinstance(result, list):
            return result
        elif isinstance(result, dict) and 'input_ids' in result:
            return result['input_ids']
        else:
            raise ValueError(f"Unexpected encoding result type: {type(result)}")

    def encode_with_offsets(self, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        """Encode text using HuggingFace tokenizer and return offsets."""
        result = self._tokenizer.encode(text, add_special_tokens=False)
        if hasattr(result, 'ids') and hasattr(result, 'offsets'):
            return result.ids, result.offsets
        # transformers.PreTrainedTokenizerFast — use __call__ with offset mapping
        if callable(getattr(self._tokenizer, '__call__', None)):
            try:
                enc = self._tokenizer(text, return_offsets_mapping=True,
                                      add_special_tokens=False)
                if 'offset_mapping' in enc:
                    ids = enc['input_ids']
                    offsets = [tuple(pair) for pair in enc['offset_mapping']]
                    return ids, offsets
            except Exception:
                pass
        # Reuse result — don't call encode() again
        if hasattr(result, 'ids'):
            return result.ids, None
        elif isinstance(result, list):
            return result, None
        elif isinstance(result, dict) and 'input_ids' in result:
            return result['input_ids'], None
        else:
            raise ValueError(f"Unexpected encoding result type: {type(result)}")

    def encode_batch_with_offsets(
        self, texts: List[str]
    ) -> List[Tuple[List[int], Optional[List[Tuple[int, int]]]]]:
        # Path A: tokenizers.Tokenizer with native encode_batch
        if hasattr(self._tokenizer, 'encode_batch'):
            encodings = self._tokenizer.encode_batch(texts, add_special_tokens=False)
            results = []
            for enc in encodings:
                if hasattr(enc, 'ids') and hasattr(enc, 'offsets'):
                    results.append((enc.ids, enc.offsets))
                else:
                    results.append((enc.ids, None))
            return results

        # Path B: transformers.PreTrainedTokenizerFast — batched __call__
        if callable(getattr(self._tokenizer, '__call__', None)):
            try:
                batch_enc = self._tokenizer(
                    texts,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )
                if 'offset_mapping' in batch_enc:
                    return [
                        (ids, [tuple(p) for p in offsets])
                        for ids, offsets in zip(
                            batch_enc['input_ids'],
                            batch_enc['offset_mapping'],
                        )
                    ]
            except Exception:
                pass

        # Fallback: loop
        return [self.encode_with_offsets(text) for text in texts]

    def can_decode(self) -> bool:
        return True

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> Optional[str]:
        try:
            if skip_special_tokens and self._fast_decode is not None:
                return self._fast_decode(token_ids)
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        except Exception as e:
            logger.warning(f"HuggingFace decode failed for {self._name}: {e}")
            return None

    def can_pretokenize(self) -> bool:
        return hasattr(self._tokenizer, 'pre_tokenizer') and self._tokenizer.pre_tokenizer is not None

    def pretokenize(self, text: str) -> List[str]:
        if not self.can_pretokenize():
            raise NotImplementedError(f"Tokenizer {self._name} does not support pretokenization")
        return [token for token, _ in self._tokenizer.pre_tokenizer.pre_tokenize_str(text)]

    def get_underlying_tokenizer(self):
        """Return the underlying HuggingFace tokenizer object."""
        return self._tokenizer

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs using the underlying HuggingFace tokenizer."""
        # tokenizers.Tokenizer uses id_to_token(); transformers.AutoTokenizer
        # uses convert_ids_to_tokens().  Try both.
        if hasattr(self._tokenizer, 'id_to_token'):
            return [self._tokenizer.id_to_token(tid) or f"<UNK_{tid}>" for tid in token_ids]
        if hasattr(self._tokenizer, 'convert_ids_to_tokens'):
            tokens = self._tokenizer.convert_ids_to_tokens(token_ids)
            result = []
            for t, tid in zip(tokens, token_ids):
                if isinstance(t, str):
                    result.append(t)
                elif isinstance(t, bytes):
                    try:
                        result.append(t.decode('utf-8'))
                    except UnicodeDecodeError:
                        result.append(t.decode('utf-8', errors='replace'))
                else:
                    result.append(f"<UNK_{tid}>")
            return result
        return super().convert_ids_to_tokens(token_ids)

    def _added_tokens_decoder(self):
        """Return {id: AddedToken-like} for declared added/special tokens, for either a
        transformers tokenizer or a raw tokenizers.Tokenizer; {} if unavailable."""
        tok = self._tokenizer
        dec = getattr(tok, 'added_tokens_decoder', None)
        if dec:
            return dec
        if hasattr(tok, 'get_added_tokens_decoder'):
            try:
                return tok.get_added_tokens_decoder()
            except Exception:
                return {}
        return {}

    def get_special_token_ids(self) -> set:
        """IDs of all tokens ADDED outside the learned vocabulary -- declared special tokens
        (<bos>, <pad>, <unk>, ...) and reserved/control tokens (<unused123>, [multimodal], ...).
        Taken from the tokenizer's own metadata (added_tokens / all_special_ids), never guessed
        from surface form. These are intentional additions, not learned merges, so callers
        exclude them from junk / unused-vocab statistics."""
        tok = self._tokenizer
        ids = set()
        try:
            ids.update(i for i in getattr(tok, 'all_special_ids', []) or [] if i is not None)
        except Exception:
            pass
        for i in self._added_tokens_decoder().keys():
            ids.add(int(i))
        return ids

    def get_unk_token_id(self) -> Optional[int]:
        """UNK token ID from the tokenizer's declared metadata only. A real UNK token is
        always a declared special token (e.g. transformers' unk_token_id, or an added_tokens
        entry with special=True whose content is a recognized UNK form). A bare 'unk' subword
        is never treated as UNK -- that earlier heuristic mislabeled ordinary subwords."""
        tok = self._tokenizer
        uid = getattr(tok, 'unk_token_id', None)   # transformers: authoritative metadata
        if uid is not None:
            return uid
        for i, a in self._added_tokens_decoder().items():
            content = getattr(a, 'content', None)
            if content is None:
                content = str(a)
            if getattr(a, 'special', False) and content in UNK_CANDIDATES:
                return int(i)
        return None

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'HuggingFaceTokenizer':
        """Create HuggingFace tokenizer wrapper from config."""
        # Import here to avoid circular import
        from ..utils.tokenizer_utils import _load_huggingface_tokenizer
        tokenizer = _load_huggingface_tokenizer(config)
        return cls(name, tokenizer, config)

class UniMixLMTokenizer(HuggingFaceTokenizer):
    """Wrapper for UniMixLM tokenizers.

    Extends :class:`HuggingFaceTokenizer` with language-specific (``langspec``)
    encoding that picks the best per-language tokenizer by log-probability, and
    pre-tokenization via ``base_tokenizer``.
    """

    def __init__(self, name: str, tokenizer, config: Dict[str, Any]):
        super().__init__(name, tokenizer, config)
        self.tokenizer_class = self._config.get('unimixlm_class')
        if self.tokenizer_class == 'langspec':
            from tokenizers import Tokenizer
            self.per_lang_tok = {}
            vocab_tuples = self._get_hf_unigram_tokenizer_vocab(self._tokenizer)
            base_vocab_order = [i for i, j in vocab_tuples]
            language_paths = config.get('language_paths')
            for lang_code, lang_tok_path in language_paths.items():
                tok = Tokenizer.from_file(lang_tok_path)
                vocab_order_per_lang = self._get_hf_unigram_tokenizer_vocab(tok)
                assert [i for i, j in vocab_order_per_lang] == base_vocab_order
                self.per_lang_tok[lang_code] = {
                    "tokenizer": tok,
                    "scores": self._extract_log_scores(tok),
                    "path": lang_tok_path,
                }
        logger.info("Creating UnimixLM tokenizer")

    # ── langspec helpers ─────────────────────────────────────────────

    @staticmethod
    def _get_hf_unigram_tokenizer_vocab(tokenizer):
        state = tokenizer.model.__getstate__()
        attributes = json.loads(state.decode("utf-8"))
        vocab = attributes["vocab"]
        if isinstance(vocab, dict):
            # This is the format the BPE vocab comes in
            tuples = sorted(vocab.items(), key=lambda x: x[1])
            return [(tk, -1) for tk, idx in tuples]
        return vocab

    @staticmethod
    def _extract_log_scores(tok) -> Dict[str, float]:
        """Return dict {token -> log_score} from a HF Unigram tokenizer."""
        vocab_tuples = UniMixLMTokenizer._get_hf_unigram_tokenizer_vocab(tok)
        return {tok_id: tok_tuple[1] for tok_id, tok_tuple in enumerate(vocab_tuples)}

    # ── overrides for langspec encoding ──────────────────────────────

    def _langspec_best_encoding(self, text: str):
        """Find the best langspec encoding by log-probability.

        Returns the raw ``tokenizers.Encoding`` object from the winning
        language tokenizer, or ``None`` if no langspec tokenizers exist.
        """
        best_enc, best_logp = None, float("-inf")
        for info in self.per_lang_tok.values():
            enc = info["tokenizer"].encode(text)
            logp = sum(info["scores"][t] for t in enc.ids)
            if logp > best_logp:
                best_enc, best_logp = enc, logp
        return best_enc

    def encode(self, text: str) -> List[int]:
        if self.tokenizer_class == 'langspec':
            best_enc = self._langspec_best_encoding(text)
            return best_enc.ids if best_enc is not None else []
        return super().encode(text)

    def encode_with_offsets(self, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        if self.tokenizer_class == 'langspec':
            best_enc = self._langspec_best_encoding(text)
            if best_enc is not None and hasattr(best_enc, 'offsets'):
                return best_enc.ids, best_enc.offsets
            return (best_enc.ids if best_enc is not None else []), None
        return super().encode_with_offsets(text)

    def encode_batch_with_offsets(
        self, texts: List[str]
    ) -> List[Tuple[List[int], Optional[List[Tuple[int, int]]]]]:
        # Langspec encoding evaluates each text against all per-language
        # tokenizers; cannot use HuggingFaceTokenizer's native batch path.
        return [self.encode_with_offsets(text) for text in texts]

    def get_underlying_tokenizer(self):
        """Return the base tokenizer.

        For langspec UniMixLM tokenizers, this returns the base tokenizer,
        not the per-language tokenizer selected during encoding.  Use
        ``encode()`` on the wrapper for langspec-aware encoding.
        """
        if self.tokenizer_class == 'langspec':
            logger.warning(
                "%s: get_underlying_tokenizer() returns the base tokenizer. "
                "For langspec-aware encoding, call encode() on the wrapper.",
                self._name,
            )
        return self._tokenizer

    # ── overrides for base_tokenizer pre-tokenization ────────────────

    def can_pretokenize(self) -> bool:
        return (
            hasattr(self._tokenizer, 'base_tokenizer')
            and hasattr(self._tokenizer.base_tokenizer, 'pre_tokenizer')
            and self._tokenizer.base_tokenizer.pre_tokenizer is not None
        )

    def pretokenize(self, text: str) -> List[str]:
        if not self.can_pretokenize():
            raise NotImplementedError(f"Tokenizer {self._name} does not support pretokenization")
        return [token for token, _ in self._tokenizer.base_tokenizer.pre_tokenizer.pre_tokenize_str(text)]

    # ── from_config ──────────────────────────────────────────────────

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'UniMixLMTokenizer':
        """Create tokenizer wrapper from config."""
        from tokenizers import Tokenizer
        tokenizer_class = config.get('unimixlm_class')
        if tokenizer_class is not None:
            tokenizer = Tokenizer.from_file(config['path'])
        else:
            from ..utils.tokenizer_utils import _load_huggingface_tokenizer
            tokenizer = _load_huggingface_tokenizer(config)

        return cls(name, tokenizer, config)


class SentencePieceTokenizer(TokenizerWrapper):
    """Wrapper for SentencePiece tokenizers."""

    def __init__(self, name: str, sp_processor: "spm.SentencePieceProcessor", config: Dict[str, Any]):
        """
        Initialize SentencePiece tokenizer wrapper.

        Args:
            name: Tokenizer name
            sp_processor: sentencepiece.SentencePieceProcessor instance
            config: Original configuration dict
        """
        self._name = name
        self._sp = sp_processor
        self._config = config or {}
        # Optional flags (default False)
        self._add_bos = bool(self._config.get("add_bos", False))
        self._add_eos = bool(self._config.get("add_eos", False))

        logger.info("Creating SentencePiece tokenizer")

    def get_name(self) -> str:
        return self._name

    def get_vocab_size(self) -> int:
        return int(self._sp.get_piece_size())

    def get_vocab(self) -> Dict[str, int]:
        size = self._sp.get_piece_size()
        return {self._sp.id_to_piece(i): i for i in range(size)}

    def can_encode(self) -> bool:
        return True

    def encode(self, text: str) -> List[int]:
        # Return list of ids; optionally prepend/append BOS/EOS if configured and defined
        ids = self._sp.encode(text, out_type=int)

        if self._add_bos:
            bos = self._sp.bos_id()
            if bos is not None and bos >= 0:
                ids = [bos] + ids

        if self._add_eos:
            eos = self._sp.eos_id()
            if eos is not None and eos >= 0:
                ids = ids + [eos]

        return ids

    def encode_with_offsets(self, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        """Encode text using SentencePiece and return character offsets.

        Uses ``encode_as_immutable_proto`` which provides ``begin`` / ``end``
        byte offsets on each piece.  Converts byte offsets to character
        offsets so they index into the original Python string.
        """
        try:
            proto = self._sp.encode_as_immutable_proto(text)
        except Exception:
            return self.encode(text), None

        ids: List[int] = []
        offsets: List[Tuple[int, int]] = []

        # Build byte→char offset map for the UTF-8 encoding of *text*.
        text_bytes = text.encode("utf-8")
        byte_to_char: List[int] = []
        for char_idx, ch in enumerate(text):
            byte_to_char.extend([char_idx] * len(ch.encode("utf-8")))
        # Sentinel for end-of-string
        byte_to_char.append(len(text))

        for piece in proto.pieces:
            ids.append(piece.id)
            b_start = piece.begin
            b_end = piece.end
            if b_start == b_end:
                # Special token or unknown — no source coverage
                offsets.append((0, 0))
            else:
                c_start = byte_to_char[b_start] if b_start < len(byte_to_char) else len(text)
                c_end = byte_to_char[b_end] if b_end < len(byte_to_char) else len(text)
                offsets.append((c_start, c_end))

        # Prepend/append BOS/EOS with (0,0) offsets if configured
        if self._add_bos:
            bos = self._sp.bos_id()
            if bos is not None and bos >= 0:
                ids = [bos] + ids
                offsets = [(0, 0)] + offsets
        if self._add_eos:
            eos = self._sp.eos_id()
            if eos is not None and eos >= 0:
                ids = ids + [eos]
                offsets = offsets + [(0, 0)]

        return ids, offsets

    def can_decode(self) -> bool:
        return True

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> Optional[str]:
        try:
            ids_to_decode = list(token_ids)
            if skip_special_tokens:
                bos = self._sp.bos_id()
                eos = self._sp.eos_id()
                special_ids = set()
                if bos is not None and bos >= 0:
                    special_ids.add(bos)
                if eos is not None and eos >= 0:
                    special_ids.add(eos)
                if special_ids:
                    ids_to_decode = [
                        tid for tid in ids_to_decode
                        if tid not in special_ids
                    ]
            return self._sp.decode(ids_to_decode)
        except Exception as e:
            logger.warning(f"SentencePiece decode failed for {self._name}: {e}")
            return None

    def can_pretokenize(self) -> bool:
        return True

    def pretokenize(self, text: str) -> List[str]:
        # Pieces correspond to subword tokens (e.g., "▁The", "re")
        pieces = self._sp.encode(text, out_type=str)
        pretokens: List[str] = []
        current = ""
    
        for p in pieces:
            if p.startswith("▁"):
                # flush previous
                if current:
                    pretokens.append(current)
                # start new (strip the boundary marker)
                current = p[1:]
            else:
                # continuation of the current pretoken
                current += p
    
        if current:
            pretokens.append(current)
    
        # NOTE: these are "normalized words" per SP's normalization rules,
        return pretokens

    def get_underlying_tokenizer(self):
        """Return the underlying SentencePieceProcessor object."""
        return self._sp

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs using SentencePiece id_to_piece."""
        vocab_size = self._sp.get_piece_size()
        result = []
        for tid in token_ids:
            if 0 <= tid < vocab_size:
                result.append(self._sp.id_to_piece(tid))
            else:
                result.append(f"<UNK_{tid}>")
        return result

    def get_unk_token_id(self) -> Optional[int]:
        """Get the UNK token ID from SentencePiece tokenizer."""
        # SentencePiece exposes unk_id(); returns -1 if undefined
        try:
            unk_id = self._sp.unk_id()
            if unk_id is not None and unk_id >= 0:
                return int(unk_id)
        except Exception:
            pass

        # Fallbacks: check common UNK pieces in the vocab
        vocab = self.get_vocab()
        for candidate in UNK_CANDIDATES:
            if candidate in vocab:
                return vocab[candidate]

        # Last-ditch: ask processor to map a likely token; if unknown, it should map to unk
        try:
            return int(self._sp.piece_to_id("<unk>"))
        except Exception:
            return None

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> "SentencePieceTokenizer":
        """
        Internal function to load a SentencePiece tokenizer from configuration.
    
        Expected config keys:
          - path: path to a .model file OR a directory containing a .model
          - (optional) model_filename: explicit filename to prefer inside a directory
        """
        try:
            import sentencepiece as spm  # lazy import here
        except ImportError as e:
            raise RuntimeError(
                "sentencepiece is required to build SentencePieceTokenizer "
                "from model files. Install with `pip install sentencepiece`."
            ) from e
        sp = None
        if "path" not in config:
            raise ValueError("config must include 'path' to the SentencePiece model (.model or directory)")
    
        path = config["path"]
        prefer_filename = config.get("model_filename")  # optional: e.g., "sp.model"
    
        # Helper: create the processor
        def _init_from_model_file(model_file: str) -> spm.SentencePieceProcessor:
            logger.info(f"Loading SentencePiece model from: {model_file}")
            sp = spm.SentencePieceProcessor()
            # Newer sentencepiece supports load() and constructor arg model_file=
            # Using load() keeps compatibility.
            if not os.path.isfile(model_file):
                raise FileNotFoundError(f"SentencePiece model file not found: {model_file}")
            loaded = sp.load(model_file)
            if not loaded:
                # Some versions return False on failure
                raise RuntimeError(f"SentencePieceProcessor.load failed for: {model_file}")
            return sp
    
        # Strategy 1: Direct path to a model file
        if os.path.isfile(path) and path.endswith(".model"):
            try:
                sp = _init_from_model_file(path)
            except Exception as e:
                logger.warning(f"Failed to load SentencePiece model from file {path}: {e}")
    
        # Strategy 2: Directory containing a model
        if os.path.isdir(path):
            candidates: List[str] = []
    
            # If user provided a preferred model filename, try that first
            if prefer_filename:
                preferred = os.path.join(path, prefer_filename)
                if os.path.isfile(preferred):
                    candidates.append(preferred)
    
            # Common names often used
            common_names = ["sp.model", "sentencepiece.model", "tokenizer.model", "model.model"]
            for name in common_names:
                p = os.path.join(path, name)
                if os.path.isfile(p):
                    candidates.append(p)
    
            # Any *.model in the directory as fallback (sorted for determinism)
            globbed = sorted(glob.glob(os.path.join(path, "*.model")))
            for p in globbed:
                if p not in candidates:
                    candidates.append(p)
    
            # Try candidates in order
            for candidate in candidates:
                try:
                    sp = _init_from_model_file(candidate)
                    break
                except Exception as e:
                    logger.warning(f"Failed to load SentencePiece model from {candidate}: {e}")
    
        # Strategy 3: If the user passed something else (e.g., a bad extension), try appending .model
        if not path.endswith(".model") and os.path.isfile(path + ".model"):
            try:
                sp = _init_from_model_file(path + ".model")
            except Exception as e:
                logger.warning(f"Failed to load SentencePiece model from {path+'.model'}: {e}")
        if sp is not None:
            return cls(name, sp, config)
        # Give up
        raise ValueError(f"Could not load SentencePiece tokenizer from {path}.")


class CustomBPETokenizer(TokenizerWrapper):
    """Wrapper for custom BPE tokenizers loaded from vocab.json and merges.txt."""

    def __init__(self, name: str, tokenizer, config: Dict[str, Any]):
        """
        Args:
            name: Tokenizer name
            tokenizer: HuggingFace tokenizer instance
            config: Original configuration dict
        """
        self._name = name
        self._tokenizer = tokenizer
        self._config = config
    
    def get_name(self) -> str:
        return self._name
    
    def get_vocab_size(self) -> int:
        return len(self._tokenizer.get_vocab())
    
    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab()
    
    def can_encode(self) -> bool:
        return True
    
    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text).ids

    def encode_with_offsets(self, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        """Encode text using custom BPE tokenizer and return offsets."""
        encoding = self._tokenizer.encode(text)
        return encoding.ids, encoding.offsets

    def encode_batch_with_offsets(
        self, texts: List[str]
    ) -> List[Tuple[List[int], Optional[List[Tuple[int, int]]]]]:
        encodings = self._tokenizer.encode_batch(texts)
        return [(enc.ids, enc.offsets) for enc in encodings]

    def can_decode(self) -> bool:
        return True

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> Optional[str]:
        try:
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        except Exception as e:
            logger.warning(f"CustomBPE decode failed for {self._name}: {e}")
            return None

    def can_pretokenize(self) -> bool:
        return hasattr(self._tokenizer, 'pre_tokenizer') and self._tokenizer.pre_tokenizer is not None

    def pretokenize(self, text: str) -> List[str]:
        if not self.can_pretokenize():
            raise NotImplementedError(f"Tokenizer {self._name} does not support pretokenization")
        return [token for token, _ in self._tokenizer.pre_tokenizer.pre_tokenize_str(text)]

    def get_unk_token_id(self) -> Optional[int]:
        """Get the UNK token ID from custom BPE tokenizer."""
        try:
            if hasattr(self._tokenizer, 'unk_token_id'):
                return self._tokenizer.unk_token_id
            if hasattr(self._tokenizer, 'token_to_id'):
                return self._tokenizer.token_to_id('<unk>')
        except Exception:
            pass
        return None

    def get_underlying_tokenizer(self):
        """Return the underlying HuggingFace tokenizer object."""
        return self._tokenizer

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs using the underlying HuggingFace tokenizer."""
        return [self._tokenizer.id_to_token(tid) or f"<UNK_{tid}>" for tid in token_ids]

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'CustomBPETokenizer':
        """Create HuggingFace tokenizer wrapper from config."""
        # Import here to avoid circular import
        from ..utils.tokenizer_utils import _load_custom_bpe_from_directory
        tokenizer = _load_custom_bpe_from_directory(config)
        return cls(name, tokenizer, config)

class PreTokenizedDataTokenizer(TokenizerWrapper):
    """Tokenizer wrapper for pre-tokenized data scenarios."""
    
    def __init__(self, name: str, vocab_size: int, vocab_dict: Optional[Dict[str, int]] = None):
        """
        Initialize pre-tokenized data tokenizer wrapper.
        
        Args:
            name: Tokenizer name
            vocab_size: Size of vocabulary
            vocab_dict: Optional vocabulary mapping
        """
        self._name = name
        self._vocab_size = vocab_size
        self._vocab_dict = vocab_dict or {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_vocab_size(self) -> int:
        return self._vocab_size
    
    def get_vocab(self) -> Optional[Dict[str, int]]:
        return self._vocab_dict if self._vocab_dict else None
    
    def can_encode(self) -> bool:
        return False
    
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError("PreTokenizedDataTokenizer cannot encode raw text")
    
    def can_pretokenize(self) -> bool:
        return False
    
    def pretokenize(self, text: str) -> List[str]:
        raise NotImplementedError("PreTokenizedDataTokenizer cannot pretokenize text")
    
    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'PreTokenizedDataTokenizer':
        """Create pre-tokenized data tokenizer wrapper from config."""
        vocab_size = config.get('vocab_size')
        vocab_dict = config.get('vocab_dict')
        if vocab_size is None:
            raise ValueError("PreTokenizedDataTokenizer requires vocab_size in config")
        return cls(name, vocab_size, vocab_dict)


# Registry for custom tokenizer classes
_TOKENIZER_REGISTRY: Dict[str, type] = {
    'huggingface': HuggingFaceTokenizer,
    'hf': HuggingFaceTokenizer,
    'transformers': HuggingFaceTokenizer,
    'standard': HuggingFaceTokenizer,  # Legacy alias
    'pretokenized': PreTokenizedDataTokenizer,
    'unimixlm': UniMixLMTokenizer,
    'custom_bpe': CustomBPETokenizer,
    'sentencepiece': SentencePieceTokenizer
}


def register_tokenizer_class(class_name: str, tokenizer_class: type) -> None:
    """
    Register a custom tokenizer class.
    
    Args:
        class_name: Name to use in configs
        tokenizer_class: TokenizerWrapper subclass
    """
    if not issubclass(tokenizer_class, TokenizerWrapper):
        raise ValueError("tokenizer_class must be a subclass of TokenizerWrapper")
    _TOKENIZER_REGISTRY[class_name] = tokenizer_class
    logger.info(f"Registered tokenizer class: {class_name} -> {tokenizer_class.__name__}")


def create_tokenizer_wrapper(name: str, config: Dict[str, Any]) -> TokenizerWrapper:
    """
    Factory function to create appropriate tokenizer wrapper from config.
    
    Args:
        name: Tokenizer name
        config: Configuration dictionary
        
    Returns:
        TokenizerWrapper instance
        
    Raises:
        ValueError: If tokenizer class is not recognized
    """
    tokenizer_class_name = config.get('class', 'huggingface')  # Default to HF
    if tokenizer_class_name == 'standard':
        logger.warning("The 'standard' tokenizers class is deprecated. Tokenizers labelled as such class "
                       "are assumed to be 'huggingface' tokenizers.")
    
    if tokenizer_class_name not in _TOKENIZER_REGISTRY:
        available_classes = list(_TOKENIZER_REGISTRY.keys())
        raise ValueError(f"Unknown tokenizer class: {tokenizer_class_name}. "
                        f"Available classes: {available_classes}")
    
    tokenizer_class = _TOKENIZER_REGISTRY[tokenizer_class_name]
    return tokenizer_class.from_config(name, config)