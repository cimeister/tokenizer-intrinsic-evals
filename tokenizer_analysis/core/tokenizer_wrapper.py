"""
Tokenizer wrapper classes for unified tokenizer interface.

This module provides a common interface for different tokenizer types,
making it easy for users to integrate custom tokenizers into the framework.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Union, Any, Tuple
import json
import logging
import os
import glob
import unicodedata
import warnings

from ..config.language_metadata import PACKAGE_ROOT
from ..constants import GENERIC_SPECIAL_TOKENS, UNK_CANDIDATES
from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Ids of tokenizer objects already warned about in resolve_special_token_strings.
# This suppresses duplicate warnings only; no result is cached under an id, so a
# recycled id can at worst drop a warning, never return another tokenizer's set.
_WARNED_MISSING_SPECIALS: set = set()


def _hf_added_tokens_decoder(tok) -> Optional[Dict[int, Any]]:
    """Return ``{id: AddedToken-like}`` for the tokens added to *tok* outside its
    learned vocabulary, or ``None`` when *tok* exposes no such metadata.

    transformers tokenizers carry the ``added_tokens_decoder`` attribute; a raw
    ``tokenizers.Tokenizer`` exposes the ``get_added_tokens_decoder()`` getter.
    ``None`` rather than ``{}`` keeps "this tokenizer added nothing" distinct from
    "this object cannot be asked", which is the difference between reporting an
    empty special-token set and falling back to GENERIC_SPECIAL_TOKENS.
    """
    dec = getattr(tok, 'added_tokens_decoder', None)
    if dec:
        return dec
    getter = getattr(tok, 'get_added_tokens_decoder', None)
    if callable(getter):
        try:
            return getter()
        except Exception as e:
            logger.debug("get_added_tokens_decoder failed: %s", e)
            return None
    return {} if dec is not None else None


def hf_special_token_strings(tok) -> Optional[Set[str]]:
    """Surface strings of every token *tok* declares special or added.

    The string counterpart of :meth:`HuggingFaceTokenizer.get_special_token_ids`:
    same two metadata channels (``all_special_tokens`` on transformers objects,
    the added-tokens decoder on both), same reasoning that an added token is an
    intentional addition rather than a learned merge. Returns ``None`` when
    neither channel can be read, which tells the caller this object cannot report
    its special tokens at all.

    Takes a raw HuggingFace tokenizer, not a wrapper, so the per-example entry
    points in ``tokenizer_analysis.per_example`` (which are handed the raw object)
    can use the same metadata as the pipeline.
    """
    strings: Set[str] = set()
    answered = False

    specials = getattr(tok, 'all_special_tokens', None)
    if specials is not None:
        answered = True
        strings.update(str(s) for s in specials if s is not None)

    dec = _hf_added_tokens_decoder(tok)
    if dec is not None:
        answered = True
        for added in dec.values():
            content = getattr(added, 'content', None)
            strings.add(str(added) if content is None else str(content))

    # Third channel: the model's own unknown token. A SentencePiece-converted
    # vocabulary can carry <unk>, <s>, </s> and <pad> as ordinary vocabulary
    # entries with an empty added_tokens list, and model.unk_token is then the
    # only metadata naming any of them.
    for owner in (getattr(tok, 'model', None), tok):
        unk = getattr(owner, 'unk_token', None)
        if isinstance(unk, str) and unk:
            answered = True
            strings.add(unk)
            break

    if not strings:
        # A readable but empty channel is not a declaration that this tokenizer
        # has no special tokens. The bundled tokenizers/unigramlm.json is the
        # case: added_tokens is empty, yet <unk>, <s>, </s> and <pad> sit at ids
        # 0 to 3. Answering "none" bypassed the generic fallback and let those
        # four be treated as ordinary content, which is the same class of error
        # the surface regex made. Measured before this: 20479 of 37892
        # _process_token calls in one run ran with an empty set.
        #
        # Returning None instead means the caller warns and uses
        # GENERIC_SPECIAL_TOKENS. For a tokenizer that genuinely has no special
        # tokens that is 13 well-known strings unlikely to be content, which is
        # the cheaper direction to be wrong in.
        return None

    return strings if answered else None


def resolve_special_token_strings(tokenizer: Any) -> Set[str]:
    """The special-token strings *tokenizer* declares, or GENERIC_SPECIAL_TOKENS.

    *tokenizer* may be a :class:`TokenizerWrapper` or a raw HuggingFace tokenizer.
    When neither can report its declared special tokens this warns once per
    tokenizer object and returns the generic set, because that set is a guess: it
    holds the common spellings but no tokenizer-specific form such as llama3's
    ``<|reserved_special_token_0|>``.
    """
    declared: Optional[Set[str]] = None
    getter = getattr(tokenizer, 'get_special_token_strings', None)
    if callable(getter):
        try:
            declared = getter()
        except Exception as e:
            logger.debug("get_special_token_strings failed: %s", e)
    else:
        declared = hf_special_token_strings(tokenizer)

    if declared is not None:
        return set(declared)

    key = id(tokenizer)
    if key not in _WARNED_MISSING_SPECIALS:
        _WARNED_MISSING_SPECIALS.add(key)
        name_getter = getattr(tokenizer, 'get_name', None)
        try:
            name = name_getter() if callable(name_getter) else type(tokenizer).__name__
        except Exception:
            name = type(tokenizer).__name__
        logger.warning(
            "Tokenizer %s cannot report its declared special tokens; using the "
            "%d generic ones (GENERIC_SPECIAL_TOKENS). Any special token it "
            "spells differently will be counted as content text.",
            name, len(GENERIC_SPECIAL_TOKENS),
        )
    return set(GENERIC_SPECIAL_TOKENS)


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
    
    @abstractmethod
    def get_special_token_strings(self) -> Optional[Set[str]]:
        """Surface strings of the tokens this tokenizer declares special.

        Read them from the tokenizer's own metadata; never infer them from the
        surface form. Return an empty set when the tokenizer genuinely declares
        none, and ``None`` only when it cannot be asked. ``None`` makes callers
        warn, naming the tokenizer, and fall back to
        :data:`~tokenizer_analysis.constants.GENERIC_SPECIAL_TOKENS`.

        Abstract rather than defaulting to the empty set because the answer
        decides which vocabulary entries are deleted from reconstructed text and
        dropped from the UTF-8 content-token denominator. A wrapper that silently
        inherited "declares none" would publish content-token counts that include
        its BOS and EOS.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'TokenizerWrapper':
        """Create tokenizer wrapper from config."""
        pass

    def pretokenize_with_spans(
        self, text: str
    ) -> Optional[List[Tuple[str, Tuple[int, int]]]]:
        """Pretokenize and return ``[(surface, (start, end)), ...]`` over *text*.

        Returns ``None`` when the backend cannot report spans. Spans are the
        only way to ask what fraction of the source the pretokenizer kept:
        comparing surface string lengths instead counts a byte-level surface,
        where one CJK character becomes three characters, so the inflation hides
        real character loss.
        """
        return None

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

    def get_bpe_merge_model(self):
        """The BPE merge list this wrapper encodes with, or why there is none.

        Returns a ``BPEMergeModel`` or a ``NotApplicable`` from
        ``core/bpe_merges.py``. Only wrappers whose ``encode`` is a single call
        into a ``tokenizers`` BPE model return a merge list; every other class
        returns ``NotApplicable`` naming itself.
        """
        from .bpe_merges import NotApplicable
        return NotApplicable(f"{type(self).__name__} exposes no BPE merge list")

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
        # transformers.PreTrainedTokenizerFast: use __call__ with offset mapping
        if callable(getattr(self._tokenizer, '__call__', None)):
            try:
                enc = self._tokenizer(text, return_offsets_mapping=True,
                                      add_special_tokens=False)
                if 'offset_mapping' in enc:
                    ids = enc['input_ids']
                    offsets = [tuple(pair) for pair in enc['offset_mapping']]
                    return ids, offsets
            except Exception as exc:
                logger.debug("offset-aware encode failed for %s (%s: %s); "
                             "returning ids with no offsets",
                             self.get_name(), type(exc).__name__, exc)
        # Reuse result, do not call encode() again
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

        # Path B: transformers.PreTrainedTokenizerFast, batched __call__
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
            except Exception as exc:
                logger.debug("batched offset encode failed for %s (%s: %s); "
                             "falling back to one text at a time",
                             self.get_name(), type(exc).__name__, exc)

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

    def _pre_tokenizer(self):
        """The Rust pre-tokenizer, or None when this tokenizer has none.

        ``self._tokenizer`` is normally a transformers fast tokenizer, which
        keeps the Rust object one level down at ``backend_tokenizer``; reading
        ``pre_tokenizer`` off the transformers object finds nothing. This
        resolves the same way ``diagnostics/sanity_check.py`` does when it
        reports the pipeline components, so the two agree about whether a
        pre-tokenizer exists.
        """
        backend = getattr(self._tokenizer, 'backend_tokenizer', None) or self._tokenizer
        return getattr(backend, 'pre_tokenizer', None)

    def can_pretokenize(self) -> bool:
        return self._pre_tokenizer() is not None

    def pretokenize(self, text: str) -> List[str]:
        pretok = self._pre_tokenizer()
        if pretok is None:
            raise NotImplementedError(f"Tokenizer {self._name} does not support pretokenization")
        return [token for token, _ in pretok.pre_tokenize_str(text)]

    def pretokenize_with_spans(self, text: str):
        """Surfaces with their character spans, straight from the backend.

        ``pre_tokenize_str`` already returns the spans; the plain
        ``pretokenize`` above throws them away.
        """
        pretok = self._pre_tokenizer()
        if pretok is None:
            return None
        try:
            return list(pretok.pre_tokenize_str(text))
        except Exception as e:
            logger.debug("pre_tokenize_str failed for %s: %s", self._name, e)
            return None

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

    def get_bpe_merge_model(self):
        """The merge list of the tokenizers backend, for this exact class only.

        Subclasses (UniMixLMTokenizer, registered classes) can encode by other
        routes, such as per-language tokenizers, so they do not inherit this.
        ``encode`` passes ``add_special_tokens=False``.
        """
        from .bpe_merges import NotApplicable, merge_model_from_backend
        if type(self) is not HuggingFaceTokenizer:
            return NotApplicable(
                f"{type(self).__name__} may encode by a route other than its "
                "tokenizers backend"
            )
        from tokenizers import Tokenizer
        backend = (self._tokenizer if isinstance(self._tokenizer, Tokenizer)
                   else getattr(self._tokenizer, 'backend_tokenizer', None))
        if backend is None:
            return NotApplicable(
                f"{type(self._tokenizer).__name__} has no tokenizers backend"
            )
        return merge_model_from_backend(backend, self._name, encode_add_special_tokens=False)

    def _added_tokens_decoder(self):
        """Return {id: AddedToken-like} for declared added/special tokens, for either a
        transformers tokenizer or a raw tokenizers.Tokenizer; {} if unavailable."""
        return _hf_added_tokens_decoder(self._tokenizer) or {}

    def get_special_token_strings(self) -> Optional[Set[str]]:
        """Surface strings for the same tokens get_special_token_ids reports: the
        declared special tokens plus everything added outside the learned vocabulary.

        Both come from the tokenizer's own metadata. Measured on the bundled
        tokenizers/bpe.json this returns {'<s>', '</s>', '<pad>', '<unk>'}, on
        apertus 1000 strings and on llama3 256; the surface pattern this replaced
        matched none of bpe.json's four.
        """
        return hf_special_token_strings(self._tokenizer)

    def get_special_token_ids(self) -> set:
        """IDs of all tokens ADDED outside the learned vocabulary: declared special tokens
        (<bos>, <pad>, <unk>, ...) and reserved/control tokens (<unused123>, [multimodal], ...).
        Taken from the tokenizer's own metadata (added_tokens / all_special_ids), never guessed
        from surface form. These are intentional additions, not learned merges, so callers
        exclude them from junk / unused-vocab statistics."""
        tok = self._tokenizer
        ids = set()
        try:
            ids.update(i for i in getattr(tok, 'all_special_ids', []) or [] if i is not None)
        except Exception as exc:
            logger.debug("all_special_ids raised for %s (%s: %s); the special "
                         "token set may be incomplete, so a special token can "
                         "be counted as ordinary content",
                         self.get_name(), type(exc).__name__, exc)
        for i in self._added_tokens_decoder().keys():
            ids.add(int(i))
        return ids

    def get_unk_token_id(self) -> Optional[int]:
        """UNK token ID from the tokenizer's declared metadata only. A real UNK token is
        always a declared special token (e.g. transformers' unk_token_id, or an added_tokens
        entry with special=True whose content is a recognized UNK form). A bare 'unk' subword
        is never treated as UNK, because that earlier heuristic mislabeled ordinary subwords."""
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

    ``get_special_token_strings`` is inherited unchanged: ``__init__`` asserts
    that every langspec per-language tokenizer has the same vocabulary order as
    the base tokenizer, so the base tokenizer's declaration describes all of them.
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

    def _pre_tokenizer(self):
        """The base tokenizer's Rust pre-tokenizer, or None when there is none.

        Overriding the resolution rather than ``can_pretokenize`` and
        ``pretokenize`` separately, because :class:`HuggingFaceTokenizer` builds
        all three of those on this one method. Overriding only the first two
        left ``pretokenize_with_spans`` resolving through the UniMixLM object,
        which has no ``pre_tokenizer`` of its own, so it returned None while
        ``can_pretokenize()`` answered True. Sanity check C10 reads those spans,
        so it reported ``unverifiable`` for a tokenizer whose spans were one
        attribute away.
        """
        base = getattr(self._tokenizer, 'base_tokenizer', None)
        return getattr(base, 'pre_tokenizer', None)

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
                # Special token or unknown: no source coverage
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

    def get_special_token_strings(self) -> Optional[Set[str]]:
        """The configured bos/eos/unk/pad pieces, plus every piece SentencePiece
        itself types as control or unknown.

        The piece type is the model's own metadata, so user-defined control pieces
        (``<|im_start|>`` and the like, added at training time with
        ``--control_symbols``) are covered without matching on surface form.

        Byte-fallback pieces (``<0xNN>``) are deliberately left out although they
        look like markup: each stands for one content byte, and the UTF-8 metrics
        decode them to that byte. SentencePiece types them BYTE, not CONTROL, so
        the scan below already excludes them; likewise UNUSED pieces, which are
        pruned merges rather than special tokens.

        Same ``answered`` structure as :func:`hf_special_token_strings`: a probe
        that raises or is unavailable does not count as an answer, and an empty
        result is reported as ``None`` (cannot determine) rather than ``set()``
        (declares none). A processor whose bos/eos/unk/pad probes all raise and
        whose build exposes no IsControl/IsUnknown predicate has not reported
        that it has no special tokens, only that it could not be asked.
        """
        ids, answered = self._declared_special_ids()
        if not ids:
            return None
        specials = {self._sp.id_to_piece(piece_id) for piece_id in ids}
        return specials if answered else None

    def get_special_token_ids(self) -> set:
        """The ids of the pieces get_special_token_strings reports.

        Both read the same scan, so the two representations cannot disagree.
        The base TokenizerWrapper returns a bare ``set()``, so without this
        override every SentencePiece model reported zero special ids and the
        callers that exclude them (the unused-vocabulary statistic in
        metrics/basic.py, the visualizer, the sanity checker) counted <unk>,
        <s> and </s> as ordinary vocabulary entries.
        """
        return self._declared_special_ids()[0]

    def _declared_special_ids(self) -> Tuple[Set[int], bool]:
        """Ids SentencePiece declares special, and whether any probe answered.

        Two channels. The four role ids (bos, eos, unk, pad), skipping any the
        model reports unset, which sentencepiece signals with a negative id.
        Then every piece the model itself types as control or unknown, which
        covers user-defined control symbols such as <|im_start|> added at
        training time with --control_symbols, without matching on surface form.

        Byte-fallback pieces (<0xNN>) are deliberately left out although they
        look like markup: each stands for one content byte, and the UTF-8
        metrics decode them to that byte. SentencePiece types them BYTE, not
        CONTROL, so the scan already excludes them, as it does UNUSED pieces,
        which are pruned merges rather than special tokens.
        """
        ids: Set[int] = set()
        answered = False

        for id_getter in ('bos_id', 'eos_id', 'unk_id', 'pad_id'):
            try:
                piece_id = getattr(self._sp, id_getter)()
            except Exception as e:
                logger.debug("SentencePiece %s() failed for %s: %s",
                             id_getter, self._name, e)
                continue
            answered = True
            if piece_id is not None and piece_id >= 0:
                ids.add(int(piece_id))

        # Older sentencepiece bindings name these IsControl/IsUnknown, newer ones
        # also expose the snake_case spelling.
        is_control = (getattr(self._sp, 'IsControl', None)
                      or getattr(self._sp, 'is_control', None))
        is_unknown = (getattr(self._sp, 'IsUnknown', None)
                      or getattr(self._sp, 'is_unknown', None))
        if callable(is_control) and callable(is_unknown):
            try:
                for piece_id in range(self._sp.get_piece_size()):
                    if is_control(piece_id) or is_unknown(piece_id):
                        ids.add(int(piece_id))
                answered = True
            except Exception as e:
                logger.debug("SentencePiece piece-type scan failed for %s: %s",
                             self._name, e)
        else:
            logger.debug(
                "SentencePiece build for %s exposes no piece-type predicate; "
                "special tokens are the bos/eos/unk/pad pieces only.", self._name,
            )
        return ids, answered

    def get_unk_token_id(self) -> Optional[int]:
        """Get the UNK token ID from SentencePiece tokenizer."""
        # SentencePiece exposes unk_id(); returns -1 if undefined
        try:
            unk_id = self._sp.unk_id()
            if unk_id is not None and unk_id >= 0:
                return int(unk_id)
        except Exception as exc:
            logger.debug("sentencepiece unk_id() raised for %s (%s: %s); "
                         "trying the vocab candidates",
                         self.get_name(), type(exc).__name__, exc)

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
                "from model files. Install the optional extra with "
                "`uv sync --extra sentencepiece` (or `pip install sentencepiece`)."
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

    def _rust(self):
        """The ``tokenizers.Tokenizer`` this wrapper's methods are written for.

        ``from_config`` builds the wrapper through
        ``_load_custom_bpe_from_directory``, which reads only a directory holding
        vocab.json plus merges.txt and returns a raw ``tokenizers.Tokenizer``.
        A caller constructing the class directly can instead pass a transformers
        fast tokenizer, which keeps the Rust object one level down at
        ``backend_tokenizer``. Every method here reads the Rust API, so on that
        shape ``encode``, ``encode_with_offsets``, ``encode_batch_with_offsets``
        and ``convert_ids_to_tokens`` all raised AttributeError, and
        ``can_pretokenize`` quietly answered False for a tokenizer that has a
        pre-tokenizer. Resolving the same way :class:`HuggingFaceTokenizer` does
        makes both shapes work and leaves the raw-Tokenizer path untouched.
        """
        return getattr(self._tokenizer, 'backend_tokenizer', None) or self._tokenizer

    def get_name(self) -> str:
        return self._name

    def get_vocab_size(self) -> int:
        return len(self._tokenizer.get_vocab())

    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab()

    def can_encode(self) -> bool:
        return True

    def encode(self, text: str) -> List[int]:
        return self._rust().encode(text).ids

    def get_bpe_merge_model(self):
        """The merge list of the tokenizers backend, for this exact class only.

        ``encode`` calls the backend with its default ``add_special_tokens=True``.
        """
        from .bpe_merges import NotApplicable, merge_model_from_backend
        if type(self) is not CustomBPETokenizer:
            return NotApplicable(
                f"{type(self).__name__} may encode by a route other than its "
                "tokenizers backend"
            )
        return merge_model_from_backend(self._rust(), self._name, encode_add_special_tokens=True)

    def encode_with_offsets(self, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        """Encode text using custom BPE tokenizer and return offsets."""
        encoding = self._rust().encode(text)
        return encoding.ids, encoding.offsets

    def encode_batch_with_offsets(
        self, texts: List[str]
    ) -> List[Tuple[List[int], Optional[List[Tuple[int, int]]]]]:
        encodings = self._rust().encode_batch(texts)
        return [(enc.ids, enc.offsets) for enc in encodings]

    def can_decode(self) -> bool:
        return True

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> Optional[str]:
        try:
            return self._rust().decode(token_ids, skip_special_tokens=skip_special_tokens)
        except Exception as e:
            logger.warning(f"CustomBPE decode failed for {self._name}: {e}")
            return None

    def _pre_tokenizer(self):
        """The Rust pre-tokenizer, or None when this tokenizer has none."""
        return getattr(self._rust(), 'pre_tokenizer', None)

    def can_pretokenize(self) -> bool:
        return self._pre_tokenizer() is not None

    def pretokenize(self, text: str) -> List[str]:
        pretok = self._pre_tokenizer()
        if pretok is None:
            raise NotImplementedError(f"Tokenizer {self._name} does not support pretokenization")
        return [token for token, _ in pretok.pre_tokenize_str(text)]

    def pretokenize_with_spans(self, text: str):
        """Surfaces with their character spans, straight from the backend.

        ``pre_tokenize_str`` already returns the spans and ``pretokenize`` above
        throws them away, so this wrapper reported None and sanity check C10
        came out ``unverifiable``. Measured on ``tokenizers/bpe.json``: C10 is
        ``unverifiable`` under the ``custom_bpe`` class and ``pass`` at 1.0 under
        ``huggingface`` for the same file, so the wrapper class rather than the
        tokenizer decided whether that check could run.
        """
        pretok = self._pre_tokenizer()
        if pretok is None:
            return None
        try:
            return list(pretok.pre_tokenize_str(text))
        except Exception as e:
            logger.debug("pre_tokenize_str failed for %s: %s", self._name, e)
            return None

    def get_special_token_strings(self) -> Optional[Set[str]]:
        """Added-token surfaces from the underlying ``tokenizers.Tokenizer``.

        A vocab.json + merges.txt BPE has no special tokens of its own; any it
        reports were attached to the loaded tokenizer object, and that object's
        added-tokens metadata is the only place they are declared.
        """
        return hf_special_token_strings(self._tokenizer)

    def get_unk_token_id(self) -> Optional[int]:
        """Get the UNK token ID from custom BPE tokenizer."""
        try:
            if hasattr(self._tokenizer, 'unk_token_id'):
                return self._tokenizer.unk_token_id
            if hasattr(self._tokenizer, 'token_to_id'):
                return self._tokenizer.token_to_id('<unk>')
        except Exception as exc:
            logger.debug("UNK lookup raised for %s (%s: %s); reporting no UNK "
                         "token, which is not the same as having none",
                         self.get_name(), type(exc).__name__, exc)
        return None

    def get_underlying_tokenizer(self):
        """Return the underlying HuggingFace tokenizer object."""
        return self._tokenizer

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs using the underlying HuggingFace tokenizer."""
        rust = self._rust()
        return [rust.id_to_token(tid) or f"<UNK_{tid}>" for tid in token_ids]

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

    def get_special_token_strings(self) -> Optional[Set[str]]:
        """``None``: this wrapper holds a vocab size and an optional vocab dict
        taken from the config, and the config declares no special tokens.

        An empty set would assert that the pre-tokenized corpus contains none,
        which nothing here establishes. ``None`` makes the caller warn, naming
        this tokenizer, and use GENERIC_SPECIAL_TOKENS instead.
        """
        return None

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'PreTokenizedDataTokenizer':
        """Create pre-tokenized data tokenizer wrapper from config."""
        vocab_size = config.get('vocab_size')
        vocab_dict = config.get('vocab_dict')
        if vocab_size is None:
            raise ValueError("PreTokenizedDataTokenizer requires vocab_size in config")
        return cls(name, vocab_size, vocab_dict)


class _PretokChunkSpans(NamedTuple):
    """One ``script_bpe`` pretokenizer chunk, with source spans.

    ``atom_ids`` is the chunk as ``Pretokenizer.pretokenize`` emits it, and
    ``atom_spans[i]`` is the range of the original text that produced
    ``atom_ids[i]``. ``span`` is the chunk's own range, held separately because
    a chunk can be empty (the digit split produces empty parts) and an empty
    chunk still has a position.
    """

    atom_ids: List[int]
    atom_spans: List[Tuple[int, int]]
    span: Tuple[int, int]


class _ScriptTokTokenizer(TokenizerWrapper):
    """Shared wrapper for ``script_bpe`` tokenizers (SCRIPT BPE and MinGram).

    Both back ends (``script_bpe.BPETokenizer`` and ``script_bpe.MinGramModel``)
    share one contract: a ``.tokens`` dict mapping ``int`` id -> token object, a
    bundled ``.pretokenizer`` that mediates encode/decode, and no bos/eos/unk/pad
    special tokens. The concrete subclasses differ only in how ``from_config``
    loads the back end, so all shared behaviour lives here.

    The back end's native ids are not a dense range. Both tokenizers reserve id 0
    (pad); SCRIPT BPE is then dense from 1, but MinGram's EM pruning leaves gaps,
    so ``max(id)`` can far exceed the token count. Reporting native ids would
    inflate ``vocab_size`` (and the vocabulary-utilization denominator) by the gap
    count and make the size tokenizer-dependent. This wrapper therefore remaps
    native ids to a gap-free, origin-preserving range (see :meth:`__init__`):
    ``encode`` emits compacted ids, ``decode`` maps them back, and ``vocab_size``
    is the compacted bound. The remap is order-preserving but is not the id
    assignment of any downstream model.

    Token strings come from ``pretokenizer.tokens_repr`` rather than per-id
    ``decode``: a lone SCRIPT block/index token is not independently decodable,
    so ``decode([id])`` collapses most ids to the same empty/fragment string,
    whereas ``tokens_repr`` renders each token distinctly. Under the SCRIPT
    pretokenizer those strings carry ``<|BLOCK_...|>`` / ``<|SCRIPT_INDEX_...|>``
    markers for pieces that are not whole characters, so a token string is the
    token's rendered form, not always literal source text (this also reflects
    NFC normalization and digit regrouping applied by the pretokenizer). Metrics
    that read token strings (byte coverage, junk/dead-vocab, whitespace) describe
    this rendered form. Full-sequence :meth:`decode` uses the back end's own
    decoder and round-trips at the character level, except that the pretokenizer
    normalizes (NFC), regroups digits, and drops code points it cannot encode
    (unassigned or noncharacter), so inputs with those do not round-trip exactly.
    """

    def __init__(self, name: str, backend, config: Dict[str, Any]):
        """
        Args:
            name: Tokenizer name.
            backend: A loaded ``script_bpe`` tokenizer (BPETokenizer/MinGramModel).
            config: Original configuration dict.
        """
        self._name = name
        self._backend = backend
        self._config = config or {}
        self._pretokenizer = backend.pretokenizer
        # Remap the back end's (possibly sparse) native ids to a gap-free range,
        # preserving the origin: the smallest native id maps to itself and the
        # rest follow contiguously. A 1-based (id-0-reserved) tokenizer stays
        # 1-based, a 0-based (reindexed) one stays 0-based, and MinGram's pruning
        # gaps are removed. This keeps vocab_size at the token count (not the
        # sparse max-id) so the vocabulary-utilization denominator is not
        # inflated. The map is order-preserving but is not any downstream model's
        # id assignment; it only gives metrics a dense id space.
        native_ids = sorted(backend.tokens.keys())
        base = native_ids[0] if native_ids else 0
        self._native_to_dense: Dict[int, int] = {
            nid: base + rank for rank, nid in enumerate(native_ids)
        }
        self._dense_to_native: Dict[int, int] = {
            dense: nid for nid, dense in self._native_to_dense.items()
        }
        # Compacted id-space bound (== max dense id + 1 == base + token count).
        self._vocab_size: int = base + len(native_ids)
        self._id_to_str_cache: Optional[Dict[int, str]] = None
        self._vocab_cache: Optional[Dict[str, int]] = None
        # Reasons character offsets were already reported unavailable for, so a
        # corpus-wide run logs each cause once instead of once per text.
        self._offset_failures: Set[str] = set()

    def get_name(self) -> str:
        return self._name

    def get_vocab_size(self) -> int:
        # The compacted (gap-free) id-space bound: the token count, plus one for
        # a reserved id 0. The input validator and id-indexed metrics treat
        # vocab_size as a bound on emitted ids (id < vocab_size); encode emits
        # compacted ids, so this holds without the gap inflation that would bias
        # vocabulary utilization and make the size depend on the tokenizer.
        return self._vocab_size

    def _ensure_str_cache(self) -> Dict[int, str]:
        """Build once the compacted-id -> token-string map, and return it."""
        if self._id_to_str_cache is None:
            cache: Dict[int, str] = {}
            for nid, tok in self._backend.tokens.items():
                dense = self._native_to_dense[nid]
                try:
                    cache[dense] = self._pretokenizer.tokens_repr(tok.atomic_tokens)
                except Exception:
                    cache[dense] = f"<TOK_{dense}>"
            self._id_to_str_cache = cache
        return self._id_to_str_cache

    def get_vocab(self) -> Dict[str, int]:
        if self._vocab_cache is None:
            # tokens_repr is distinct per token for these tokenizers, so this is
            # a faithful full-vocabulary map (one entry per compacted id). Were
            # two ids to ever render alike, the later id would win here.
            self._vocab_cache = {s: d for d, s in self._ensure_str_cache().items()}
        # Return a copy so a caller mutating the result cannot corrupt the cache
        # (the HF/SP/custom_bpe wrappers likewise hand back a fresh dict).
        return dict(self._vocab_cache)

    def can_encode(self) -> bool:
        return True

    def encode(self, text: str) -> List[int]:
        # BPETokenizer.encode returns array.array; MinGramModel.encode returns
        # list[int]. Remap each native id to its compacted id (plain Python int).
        to_dense = self._native_to_dense
        return [to_dense[i] for i in self._backend.encode(text)]

    def can_decode(self) -> bool:
        return True

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> Optional[str]:
        # Inputs are compacted ids (what encode emits); map them back to native
        # ids for the back end. skip_special_tokens is a no-op (no special
        # tokens). BPETokenizer.decode defaults errors="replace"; MinGramModel
        # .decode takes no errors kwarg, so passing only ids works for both.
        try:
            native = [self._dense_to_native[i] for i in token_ids]
            return self._backend.decode(native)
        except Exception as e:
            logger.warning(f"script_bpe decode failed for {self._name}: {e}")
            return None

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        cache = self._ensure_str_cache()
        return [cache.get(i, f"<UNK_{i}>") for i in token_ids]

    def can_pretokenize(self) -> bool:
        return True

    def pretokenize(self, text: str) -> List[str]:
        # pretokenizer.pretokenize returns a list of atomic-token chunks
        # (array.array); decode each chunk back to its rendered substring.
        return [self._pretokenizer.decode(chunk)
                for chunk in self._pretokenizer.pretokenize(text)]

    # ---- character offsets ------------------------------------------------
    #
    # The bundled pretokenizer rewrites the text before any token exists: it
    # normalizes it (NFC by default) and drops code points its script config
    # does not cover. Positions in that rewritten text do not index the caller's
    # string, so the methods below replay the pretokenizer's pipeline one step
    # at a time, carrying a source span with every character, and check each
    # step against the pretokenizer's own output. A step that cannot be
    # established exactly gives up on the whole text rather than approximating:
    # an offset that is merely close still reads as a real measurement, which is
    # why metrics/code_ast.py removed its string-matching fallback.

    def _no_offsets(self, reason: str) -> None:
        """Warn that offsets are unavailable, once per distinct *reason*."""
        if reason in self._offset_failures:
            return
        self._offset_failures.add(reason)
        logger.warning(
            "script_bpe tokenizer %s cannot report character offsets: %s. "
            "Offsets are None for inputs of this kind; this reason is logged "
            "once per tokenizer.", self._name, reason)

    @staticmethod
    def _refine_segment(
        text: str, form: str, start: int, end: int
    ) -> List[Tuple[int, int]]:
        """Cut ``text[start:end]`` as finely as normalizing it allows.

        A cut is taken only where normalizing the two sides separately
        reproduces the normalization of the whole range, so however the range
        is cut, the pieces still rebuild it. Cutting as early as possible keeps
        the reported spans as narrow as the normalization permits: characters
        that happen to sit next to a composition get their own span rather than
        inheriting the composition's.
        """
        pieces: List[Tuple[int, int]] = []
        while start < end:
            whole = unicodedata.normalize(form, text[start:end])
            cut = end
            for position in range(start + 1, end):
                if (unicodedata.normalize(form, text[start:position])
                        + unicodedata.normalize(form, text[position:end])) == whole:
                    cut = position
                    break
            pieces.append((start, cut))
            start = cut
        return pieces

    @classmethod
    def _normalization_segments(cls, text: str, form: str) -> List[Tuple[int, int]]:
        """Cut *text* where normalization cannot reach across the cut.

        Two passes. The first cuts before every combining-class-0 character and
        merges a pair back together when normalizing the two sides separately
        does not give what normalizing them together gives, which is what NFC
        composing Hangul jamo (all combining class 0) needs. The second pass
        cuts each of those ranges as finely as its own normalization allows.

        The first pass alone is too coarse: a mark that composes with nothing
        would still share a span with its neighbour. The second pass alone is
        not safe: normalization reorders adjacent marks by combining class, so
        a cut that looks harmless against the next character alone can change
        the result once the character after it is included. The caller also
        checks the joined result against the whole-string normalization, so a
        segmentation these passes get wrong stops the offsets rather than
        shifting them.
        """
        n = len(text)
        if n == 0:
            return []
        starts = [0]
        starts.extend(i for i in range(1, n) if unicodedata.combining(text[i]) == 0)
        starts.append(n)
        segments: List[Tuple[int, int]] = []
        i = 0
        while i < len(starts) - 1:
            start = starts[i]
            j = i + 1
            while j < len(starts) - 1:
                left = unicodedata.normalize(form, text[start:starts[j]])
                right = unicodedata.normalize(form, text[starts[j]:starts[j + 1]])
                if left + right == unicodedata.normalize(form, text[start:starts[j + 1]]):
                    break
                j += 1
            segments.extend(cls._refine_segment(text, form, start, starts[j]))
            i = j
        return segments

    def _normalized_chars_with_spans(
        self, text: str
    ) -> Optional[Tuple[List[str], List[Tuple[int, int]]]]:
        """``(chars, spans)`` for ``pretokenizer.normalize(text)``, or ``None``.

        ``chars`` is that normalized text one character at a time, and
        ``spans[i]`` is the range of the original *text* that produced
        ``chars[i]``. Composition is many-to-one, so a span can be wider than
        one character, and a decomposing form (NFD, NFKD) makes consecutive
        characters share one span.
        """
        pretok = self._pretokenizer
        cfg = pretok.config
        form = cfg.normalization
        normalized = text if form is None else unicodedata.normalize(form, text)
        if normalized == text:
            # Nothing composed, decomposed or reordered, so each character of
            # the input stands for itself.
            chars = list(text)
            spans = [(i, i + 1) for i in range(len(text))]
        else:
            chars = []
            spans = []
            for start, end in self._normalization_segments(text, form):
                for ch in unicodedata.normalize(form, text[start:end]):
                    chars.append(ch)
                    spans.append((start, end))
            if "".join(chars) != normalized:
                self._no_offsets(
                    "normalizing the aligned segments one at a time did not "
                    f"reproduce {form} of the whole text")
                return None
        if cfg.remove_unassigned:
            kept = [i for i, ch in enumerate(chars) if ch in pretok.valid_chars]
            chars = [chars[i] for i in kept]
            spans = [spans[i] for i in kept]
        if "".join(chars) != pretok.normalize(text):
            self._no_offsets(
                "the reconstructed normalized text differs from what "
                "pretokenizer.normalize() returns")
            return None
        return chars, spans

    def _encode_text_with_spans(
        self,
        chars: List[str],
        spans: List[Tuple[int, int]],
        lo: int,
        hi: int,
    ) -> Optional[Tuple[List[Any], List[Tuple[int, int]]]]:
        """``encode_text`` over ``chars[lo:hi]``, with a source span per CharEnc.

        Encoding one character at a time is what pairs a CharEnc with a
        character, since one character can produce several (the UTF-8
        pretokenizer emits one per byte). The resulting atomic-token stream is
        compared against encoding the whole piece at once, so a pretokenizer
        whose encoding depends on neighbouring characters gives up instead of
        pairing them wrongly.
        """
        pretok = self._pretokenizer
        encs: List[Any] = []
        enc_spans: List[Tuple[int, int]] = []
        for i in range(lo, hi):
            for enc in pretok.encode_text(chars[i]):
                encs.append(enc)
                enc_spans.append(spans[i])
        whole = pretok.encode_text("".join(chars[lo:hi]))
        if ([tid for enc in encs for tid in enc.atomic_token_ids]
                != [tid for enc in whole for tid in enc.atomic_token_ids]):
            self._no_offsets(
                "encoding one character at a time gives a different "
                "atomic-token stream than encoding the whole piece")
            return None
        return encs, enc_spans

    def _regex_split_spans(
        self, norm: str, lo: int, hi: int
    ) -> Optional[List[Tuple[int, int]]]:
        """Index ranges of ``regex_split(norm[lo:hi])`` within *norm*, or ``None``.

        ``regex_split`` runs ``findall``, which reports the matched strings
        without saying where they are and silently drops whatever the pattern
        does not match. ``finditer`` over the same pattern gives the positions;
        its matches are compared against ``regex_split``'s own output, so a
        pattern whose capture groups make ``findall`` return something other
        than whole matches gives up instead of pairing the two wrongly.
        """
        pretok = self._pretokenizer
        if pretok.config.regex_pattern is None:
            return [(lo, hi)]
        # The module's own regex binding, so the pattern is matched by the
        # engine split_unencoded_and_encode uses (script_bpe matches with the
        # third-party `regex` module, whose syntax the configured patterns use).
        from script_bpe.pretokenize import pretokenizer as script_bpe_pretokenize
        part = norm[lo:hi]
        matches = list(script_bpe_pretokenize.re.finditer(
            pretok.config.regex_pattern, part))
        if [m.group(0) for m in matches] != list(pretok.regex_split(part)):
            self._no_offsets(
                "the configured regex_pattern does not report whole matches "
                "through findall, so match positions cannot be established")
            return None
        return [(lo + m.start(), lo + m.end()) for m in matches]

    def _append_split_chunks(
        self,
        chunks: List[_PretokChunkSpans],
        encs: List[Any],
        enc_spans: List[Tuple[int, int]],
        piece_span: Tuple[int, int],
    ) -> bool:
        """Run ``split_encoded`` over *encs* and append its groups with spans.

        ``split_encoded`` regroups the CharEncs it is handed (script grouping,
        space merging) and returns the same objects in the same order, which is
        what lets each span follow its character. That is checked here by
        identity, position by position, so an implementation that copied or
        reordered them would stop the offsets rather than shift them.
        """
        pretok = self._pretokenizer
        consumed = 0
        for group in pretok.split_encoded(encs):
            atom_ids: List[int] = []
            atom_spans: List[Tuple[int, int]] = []
            first = consumed
            for enc in group:
                if consumed >= len(encs) or enc is not encs[consumed]:
                    self._no_offsets(
                        "split_encoded did not return the encoded characters it "
                        "was given, in order")
                    return False
                for tid in enc.atomic_token_ids:
                    atom_ids.append(tid)
                    atom_spans.append(enc_spans[consumed])
                consumed += 1
            if atom_spans:
                span = (atom_spans[0][0], atom_spans[-1][1])
            elif first < len(enc_spans):
                span = (enc_spans[first][0], enc_spans[first][0])
            else:
                span = (piece_span[1], piece_span[1])
            chunks.append(_PretokChunkSpans(atom_ids, atom_spans, span))
        if consumed != len(encs):
            self._no_offsets(
                "split_encoded dropped some of the encoded characters")
            return False
        return True

    def _chunks_with_spans(self, text: str) -> Optional[List[_PretokChunkSpans]]:
        """Replay ``pretokenize`` keeping a source span per atomic token.

        Mirrors ``Pretokenizer.pretokenize``: normalize, split digit runs off,
        regex-split the rest, encode, regroup. Returns ``None`` as soon as one
        of those steps cannot be tied back to positions in *text*.
        """
        # The pretokenizer module itself, so the digit split and the digit
        # grouping below are the ones split_unencoded_and_encode runs, down to
        # script_bpe splitting with the third-party `regex` module.
        from script_bpe.pretokenize import pretokenizer as script_bpe_pretokenize

        pretok = self._pretokenizer
        cfg = pretok.config
        aligned = self._normalized_chars_with_spans(text)
        if aligned is None:
            return None
        chars, spans = aligned
        norm = "".join(chars)

        def source_span(lo: int, hi: int) -> Tuple[int, int]:
            if lo >= hi:
                # A zero-width piece is placed where the next kept character
                # starts, or at the end of the last one for a trailing piece.
                pos = spans[lo][0] if lo < len(spans) else (spans[-1][1] if spans else 0)
                return (pos, pos)
            return (spans[lo][0], spans[hi - 1][1])

        # Digit runs are split out first and regrouped, so they carry their own
        # spans; everything else goes through the regex split. Index ranges into
        # `norm` stand in for the substrings the pretokenizer passes around.
        parts: List[Tuple[int, int, bool]] = []
        if cfg.digit_handling is None:
            parts.append((0, len(norm), False))
        else:
            raw = script_bpe_pretokenize.re.split("([0-9]+)", norm)
            if "".join(raw) != norm:
                self._no_offsets(
                    "the digit split did not partition the normalized text")
                return None
            at = 0
            for i, part in enumerate(raw):
                # re.split with one capture group alternates non-digits, digits.
                parts.append((at, at + len(part), i % 2 == 1))
                at += len(part)

        chunks: List[_PretokChunkSpans] = []
        for lo, hi, is_digits in parts:
            if is_digits:
                groups = script_bpe_pretokenize.group_digits(
                    norm[lo:hi], cfg.digit_handling)
                if "".join(groups) != norm[lo:hi]:
                    self._no_offsets(
                        "digit grouping did not partition the digit run")
                    return None
                at = lo
                for group in groups:
                    encs = list(pretok.encode_digits([group]))
                    group_span = source_span(at, at + len(group))
                    if not self._append_split_chunks(
                            chunks, encs, [group_span] * len(encs), group_span):
                        return None
                    at += len(group)
            else:
                pieces = self._regex_split_spans(norm, lo, hi)
                if pieces is None:
                    return None
                for start, end in pieces:
                    encoded = self._encode_text_with_spans(chars, spans, start, end)
                    if encoded is None:
                        return None
                    encs, enc_spans = encoded
                    if not self._append_split_chunks(
                            chunks, encs, enc_spans, source_span(start, end)):
                        return None
        return chunks

    def encode_with_offsets(self, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        """Encode, reporting each token's character range in the original *text*.

        The ids are the ones :meth:`encode` returns. Offsets index the caller's
        text rather than the normalized text the pretokenizer works on, so a
        token holding a composed character reports the range of every original
        character that composed into it. Offsets can overlap: unless the
        training config enforced character boundaries, a token can hold the tail
        of one character and the head of the next, and is reported as covering
        both.

        Characters the pretokenizer drops (outside its script config, or
        unmatched by its regex) are usually covered by no token, and sometimes
        are. A token whose first and last atoms sit on either side of a dropped
        character reports one range across it, dropped character included.
        Measured on the SCRIPT BPE fixture: ``'a￰b'`` covers index 1, which
        was dropped, while ``'ab￰cd'`` leaves index 2 uncovered. An
        uncovered character is therefore evidence of loss, and a covered one is
        not evidence against it.

        Offsets are ``None``, with the ids unaffected, whenever the alignment
        cannot be established; :meth:`_no_offsets` logs why.
        """
        # The native ids are kept alongside the compacted ones because the
        # atomic tokens are looked up by native id below. This repeats what
        # encode() does rather than calling it, which would encode twice.
        native = list(self._backend.encode(text))
        ids = [self._native_to_dense[i] for i in native]
        chunks = self._chunks_with_spans(text)
        if chunks is None:
            return ids, None
        atom_ids = [tid for chunk in chunks for tid in chunk.atom_ids]
        atom_spans = [span for chunk in chunks for span in chunk.atom_spans]
        offsets: List[Tuple[int, int]] = []
        pos = 0
        for nid in native:
            atoms = self._backend.tokens[nid].atomic_tokens
            end = pos + len(atoms)
            if (not len(atoms) or end > len(atom_ids)
                    or list(atoms) != atom_ids[pos:end]):
                self._no_offsets(
                    "the encoded tokens do not spell out the atomic-token "
                    "stream the pretokenizer produced")
                return ids, None
            offsets.append((atom_spans[pos][0], atom_spans[end - 1][1]))
            pos = end
        if pos != len(atom_ids):
            self._no_offsets(
                "the encoded tokens cover only part of the pretokenizer's "
                "atomic-token stream")
            return ids, None
        return ids, offsets

    def pretokenize_with_spans(
        self, text: str
    ) -> Optional[List[Tuple[str, Tuple[int, int]]]]:
        """The pretokenizer's chunks with their character range in *text*.

        The surfaces are :meth:`pretokenize`'s, and the chunks this replays are
        checked against the ones ``pretokenize`` produced before any span is
        reported. A chunk that dropped every character it was given (the digit
        split can produce empty pieces) keeps a zero-width span at the position
        it sits in.
        """
        chunks = self._chunks_with_spans(text)
        if chunks is None:
            return None
        produced = self._pretokenizer.pretokenize(text)
        if len(produced) != len(chunks) or any(
                list(chunk) != replay.atom_ids
                for chunk, replay in zip(produced, chunks)):
            self._no_offsets(
                "the replayed pretokenizer chunks differ from the ones "
                "pretokenize() produced")
            return None
        return [(self._pretokenizer.decode(chunk), replay.span)
                for chunk, replay in zip(produced, chunks)]

    def get_special_token_strings(self) -> Optional[Set[str]]:
        """Empty: the ``script_bpe`` back ends declare no special tokens.

        Both reserve id 0 for padding, but the reserved id is not a key of
        ``backend.tokens``, so it has no token string and never appears in the
        vocabulary this wrapper exposes. Every string :meth:`get_vocab` returns is
        a rendered token (see the class docstring), which is content.
        """
        return set()

    def get_underlying_tokenizer(self):
        # Deliberately None: these are not HuggingFace tokenizers, so MorphScore
        # and the HF-internal sanity checks skip cleanly instead of misreading a
        # foreign object.
        return None


class ScriptBPETokenizer(_ScriptTokTokenizer):
    """Wrapper for SCRIPT BPE tokenizers (``script_bpe.BPETokenizer``, ``sebpe-v1``)."""

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'ScriptBPETokenizer':
        """
        Expected config keys:
          - path: path to a saved script_bpe BPE tokenizer (.json or .json.gz).
        """
        if "path" not in config:
            raise ValueError(
                "ScriptBPETokenizer requires 'path' to a script_bpe tokenizer "
                "file (.json or .json.gz)")
        try:
            from script_bpe import BPETokenizer
        except ImportError as e:
            raise RuntimeError(
                "The 'script_bpe' package is required for the 'script_bpe' "
                "tokenizer class. Install it with `pip install -e "
                "/path/to/script_tok` or add its repo to PYTHONPATH."
            ) from e
        path = config["path"]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"script_bpe tokenizer file not found: {path}")
        backend = BPETokenizer.load(path)
        return cls(name, backend, config)


class MinGramTokenizer(_ScriptTokTokenizer):
    """Wrapper for MinGram tokenizers (``script_bpe.MinGramModel``, ``semingram-v1``)."""

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'MinGramTokenizer':
        """
        Expected config keys:
          - path: path to a saved script_bpe MinGram tokenizer (.json or .json.gz).
          - (optional) reindex: if True, densify token ids to 0..n-1 on load
            (default False; passed to MinGramModel.load).
        """
        if "path" not in config:
            raise ValueError(
                "MinGramTokenizer requires 'path' to a script_bpe tokenizer "
                "file (.json or .json.gz)")
        try:
            from script_bpe import MinGramModel
        except ImportError as e:
            raise RuntimeError(
                "The 'script_bpe' package is required for the 'mingram' "
                "tokenizer class. Install it with `pip install -e "
                "/path/to/script_tok` or add its repo to PYTHONPATH."
            ) from e
        path = config["path"]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"script_bpe tokenizer file not found: {path}")
        reindex = bool(config.get("reindex", False))
        backend = MinGramModel.load(path, reindex=reindex)
        return cls(name, backend, config)


# Registry for custom tokenizer classes
_TOKENIZER_REGISTRY: Dict[str, type] = {
    'huggingface': HuggingFaceTokenizer,
    'hf': HuggingFaceTokenizer,
    'transformers': HuggingFaceTokenizer,
    'standard': HuggingFaceTokenizer,  # Legacy alias
    'pretokenized': PreTokenizedDataTokenizer,
    'unimixlm': UniMixLMTokenizer,
    'custom_bpe': CustomBPETokenizer,
    'sentencepiece': SentencePieceTokenizer,
    'script_bpe': ScriptBPETokenizer,
    'mingram': MinGramTokenizer
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
        warnings.warn(
            "The 'standard' tokenizer class is deprecated; use 'huggingface' "
            f"instead. Tokenizer {name!r} will be loaded as 'huggingface'.",
            DeprecationWarning,
            stacklevel=2,
        )
    
    if tokenizer_class_name not in _TOKENIZER_REGISTRY:
        available_classes = list(_TOKENIZER_REGISTRY.keys())
        # ConfigurationError, not ValueError: a class name that is not in the
        # registry is the user's typo in a config file, and the console script
        # prints this without a traceback. It subclasses ValueError, so a caller
        # catching that still catches this.
        raise ConfigurationError(
            f"Unknown tokenizer class: {tokenizer_class_name}. "
            f"Available classes: {available_classes}")
    
    tokenizer_class = _TOKENIZER_REGISTRY[tokenizer_class_name]
    return tokenizer_class.from_config(name, _resolve_config_path(name, config))


def _resolve_config_path(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Anchor a relative tokenizer ``path`` at the package root when it exists there.

    A relative path in a tokenizer config used to be resolved against the
    process working directory, so `configs/sample_tokenizers.json` naming
    `tokenizers/bpe.json` loaded from a source checkout and failed from
    anywhere else. Language-config corpus paths are anchored unconditionally
    (see config.language_metadata.resolve_corpus_path); a tokenizer path cannot
    be, because a Hub model id such as `meta-llama/Meta-Llama-3-8B` is also a
    relative-looking string that must reach the loader unchanged.

    So the order is: absolute paths and paths that exist relative to the
    working directory are left alone, which means a local file is never
    overridden by a package-root file of the same name; a path that exists only
    under the package root is rewritten, and the log line says so; anything else
    is passed through for the loader to resolve or to report.
    """
    raw = config.get('path')
    if not isinstance(raw, str) or not raw:
        return config
    path = Path(raw)
    if path.is_absolute() or path.exists():
        return config
    candidate = PACKAGE_ROOT / path
    if not candidate.exists():
        return config
    logger.info(
        "Tokenizer %s: %r does not exist in the working directory; using the "
        "package-root copy at %s.", name, raw, candidate,
    )
    resolved = dict(config)
    resolved['path'] = str(candidate)
    return resolved