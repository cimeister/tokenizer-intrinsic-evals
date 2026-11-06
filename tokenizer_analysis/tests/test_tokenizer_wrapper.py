"""Tests for tokenizer wrapper consistency across wrapper types.

Verifies that HuggingFaceTokenizer, SentencePieceTokenizer,
CustomBPETokenizer, and the script_bpe wrappers (ScriptBPETokenizer,
MinGramTokenizer) behave consistently for the operations that intrinsic
metrics depend on: encode, encode_with_offsets, decode,
convert_ids_to_tokens, get_vocab, and get_vocab_size.
"""

import os
import pytest
import tempfile

from tokenizer_analysis.core.tokenizer_wrapper import (
    HuggingFaceTokenizer,
    SentencePieceTokenizer,
    CustomBPETokenizer,
    ScriptBPETokenizer,
    MinGramTokenizer,
    _ScriptTokTokenizer,
    create_tokenizer_wrapper,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# Shared corpus for training tiny tokenizers
_CORPUS = [
    "The theory of relativity revolutionized our understanding of space.",
    "Die Relativitätstheorie revolutionierte unser Verständnis von Raum.",
    "La théorie de la relativité a révolutionné notre compréhension.",
    "Hello world, this is a simple test sentence.",
    "import os; print(os.getcwd())",
    "def foo(x): return x + 1",
    "The quick brown fox jumps over the lazy dog.",
    "Les mathématiques sont belles et utiles.",
    "Spaces   and\ttabs\nand newlines are whitespace.",
    "1234567890 numbers and symbols: @#$%^&*()",
]

_TEST_TEXT = "Hello world, this is a test."
_TEST_TEXT_MULTI = "Die Relativitätstheorie ist wichtig."

# Code- and math-shaped texts: the two corpora the batch-encoding refactor
# shares. Leading indentation and a long digit run are exactly the shapes
# a pretokenizer's whitespace- and digit-splitting rules are most likely to
# treat differently from the plain-prose texts above.
_CODE_TEXT = "    if x:\n        return foo(x, y)\n"
_MATH_TEXT = "The sum is 314159265358979323846264338327950288 exactly."

# Offset inputs, written with escapes so the composition is unambiguous in the
# source rather than depending on how this file was saved.
_NFD_CAFE = "cafe\u0301"           # 5 characters; NFC composes them into 4
_NFC_CAFE = "caf\u00e9"             # the 4-character form of the same word
_HANGUL_JAMO = "\u1100\u1161\u11a8"  # L, V, T; NFC composes all three into one
_HANGUL_SYLLABLE = "\uac01"        # that syllable, as one character
_DROPPED_CHAR = "\ufff0"           # unassigned, so the script config drops it

# Inputs whose every character survives the pretokenizer's normalization, so
# each one must end up inside some token's offsets.
_OFFSET_BATTERY = [
    _TEST_TEXT,
    _TEST_TEXT_MULTI,
    _NFD_CAFE,
    _NFC_CAFE,
    _HANGUL_JAMO,
    "世界 こんにちは 漢字",
    "1234567890 and 42",
    "def f(x):\n    return x + 1\n",
    "   ",
    "  \t \n  ",
    "",
]


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_hf_tokenizer():
    """Train a minimal BPE tokenizer using the tokenizers library."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    tok = Tokenizer(BPE(unk_token="<unk>"))
    # add_prefix_space=False so ByteLevel does not prepend a space to the
    # first token. This ensures decode(encode(text)) == text exactly,
    # without needing whitespace stripping in roundtrip assertions.
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=300,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
    )
    tok.train_from_iterator(_CORPUS, trainer=trainer)
    return tok


@pytest.fixture(scope="module")
def hf_wrapper(trained_hf_tokenizer):
    return HuggingFaceTokenizer("test-hf", trained_hf_tokenizer, {})


@pytest.fixture(scope="module")
def custom_bpe_wrapper(trained_hf_tokenizer):
    return CustomBPETokenizer("test-cbpe", trained_hf_tokenizer, {})


@pytest.fixture(scope="module")
def sp_model_path():
    """Train a tiny SentencePiece model and return the model file path."""
    spm = pytest.importorskip("sentencepiece")
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for line in _CORPUS:
                f.write(line + "\n")

        prefix = os.path.join(tmpdir, "test_sp")
        # Disable SentencePiece's default NFKC normalization, dummy prefix,
        # and extra-whitespace removal so that decode(encode(text)) == text
        # exactly, without needing whitespace stripping in roundtrip assertions.
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=prefix,
            vocab_size=200,
            model_type="bpe",
            character_coverage=1.0,
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
        )
        model_path = prefix + ".model"
        # Read model bytes so the fixture outlives the tmpdir
        with open(model_path, "rb") as f:
            model_bytes = f.read()

        # Write to a persistent temp file (cleaned up at process exit)
        persistent = tempfile.NamedTemporaryFile(suffix=".model", delete=False)
        persistent.write(model_bytes)
        persistent.close()
        yield persistent.name
        os.unlink(persistent.name)


@pytest.fixture(scope="module")
def sp_processor(sp_model_path):
    spm = pytest.importorskip("sentencepiece")
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    return sp


@pytest.fixture(scope="module")
def sp_wrapper(sp_processor):
    return SentencePieceTokenizer("test-sp", sp_processor, {})


@pytest.fixture(scope="module")
def sp_wrapper_with_bos_eos(sp_processor):
    return SentencePieceTokenizer(
        "test-sp-bos-eos", sp_processor,
        {"add_bos": True, "add_eos": True},
    )


@pytest.fixture(scope="module")
def script_bpe_wrapper():
    """Load the committed tiny SCRIPT BPE fixture through the registry.

    Skips when ``script_bpe`` is not installed (it is an optional, non-PyPI
    dependency; the wrapper imports it lazily in from_config).
    """
    pytest.importorskip("script_bpe")
    path = os.path.join(FIXTURE_DIR, "scriptbpe_tiny.json.gz")
    return create_tokenizer_wrapper(
        "test-scriptbpe", {"class": "script_bpe", "path": path}
    )


@pytest.fixture(scope="module")
def mingram_wrapper():
    """Load the committed tiny MinGram fixture through the registry."""
    pytest.importorskip("script_bpe")
    path = os.path.join(FIXTURE_DIR, "mingram_tiny.json.gz")
    return create_tokenizer_wrapper(
        "test-mingram", {"class": "mingram", "path": path}
    )


ALL_WRAPPER_FIXTURES = [
    "hf_wrapper", "sp_wrapper", "custom_bpe_wrapper",
    "script_bpe_wrapper", "mingram_wrapper",
]


# ── Tests ─────────────────────────────────────────────────────────────


class TestEncodeConsistency:
    """encode() and encode_with_offsets() must return the same token IDs."""

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_ids_match(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        for text in [_TEST_TEXT, _TEST_TEXT_MULTI]:
            ids_plain = wrapper.encode(text)
            ids_offsets, _ = wrapper.encode_with_offsets(text)
            assert ids_plain == ids_offsets, (
                f"{wrapper.get_name()}: encode() and encode_with_offsets() "
                f"returned different IDs for {text!r}"
            )

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_batch_ids_match_single_encode(self, wrapper_name, request):
        """encode_batch_with_offsets() must agree with encode() and with
        encode_with_offsets(), text by text, in one batch call.

        A metrics refactor is about to make reconstruction-fidelity and
        fertility read their ids from encode_batch_with_offsets() instead of
        encode(). If a wrapper's batch path (a native tokenizers.Tokenizer
        .encode_batch call, a transformers fast-tokenizer batched __call__,
        or the base class's per-text loop) ever returned ids out of order,
        for the wrong text, or offsets misaligned with the per-text offsets
        encode_with_offsets() reports, published token counts and
        reconstructed text would silently start describing a different text
        than the one that produced them.
        """
        wrapper = request.getfixturevalue(wrapper_name)
        texts = [_TEST_TEXT, _TEST_TEXT_MULTI, _CODE_TEXT, _MATH_TEXT]

        batch_results = wrapper.encode_batch_with_offsets(texts)
        assert len(batch_results) == len(texts)

        for i, text in enumerate(texts):
            batch_ids, batch_offsets = batch_results[i]
            assert batch_ids == wrapper.encode(text), (
                f"{wrapper.get_name()}: encode_batch_with_offsets()[{i}][0] "
                f"!= encode(texts[{i}]) for {text!r}"
            )
            # Some wrappers (e.g. when alignment cannot be established) return
            # None offsets for a given text; that must match single-text
            # encode_with_offsets() exactly, None included, without weakening
            # the id check above.
            _single_ids, single_offsets = wrapper.encode_with_offsets(text)
            assert batch_offsets == single_offsets, (
                f"{wrapper.get_name()}: batch offsets for texts[{i}] "
                f"({text!r}) != encode_with_offsets(texts[{i}])[1]"
            )


class TestNoSpecialTokensInEncode:
    """encode() must not include BOS/EOS unless explicitly configured."""

    def test_hf_no_special_tokens(self, hf_wrapper, trained_hf_tokenizer):
        """HF wrapper with add_special_tokens=False should not add BOS/EOS."""
        text = _TEST_TEXT
        ids = hf_wrapper.encode(text)
        # Encode with special tokens enabled for comparison
        ids_with_special = trained_hf_tokenizer.encode(
            text, add_special_tokens=True
        ).ids
        ids_without_special = trained_hf_tokenizer.encode(
            text, add_special_tokens=False
        ).ids
        assert ids == ids_without_special
        # If the tokenizer has a post-processor that adds specials,
        # the wrapper's output should be shorter
        if len(ids_with_special) > len(ids_without_special):
            assert len(ids) == len(ids_without_special)

    def test_sp_default_no_bos_eos(self, sp_wrapper):
        """SP wrapper without add_bos/add_eos config should not add them."""
        text = _TEST_TEXT
        ids = sp_wrapper.encode(text)
        sp = sp_wrapper._sp
        bos_id = sp.bos_id()
        eos_id = sp.eos_id()
        if bos_id >= 0 and len(ids) > 0:
            assert ids[0] != bos_id, "BOS found but add_bos not configured"
        if eos_id >= 0 and len(ids) > 0:
            assert ids[-1] != eos_id, "EOS found but add_eos not configured"

    def test_sp_with_bos_eos_adds_them(self, sp_wrapper_with_bos_eos):
        """SP wrapper with add_bos=True, add_eos=True should add them."""
        text = _TEST_TEXT
        ids = sp_wrapper_with_bos_eos.encode(text)
        sp = sp_wrapper_with_bos_eos._sp
        bos_id = sp.bos_id()
        eos_id = sp.eos_id()
        if bos_id >= 0:
            assert ids[0] == bos_id, "BOS not found despite add_bos=True"
        if eos_id >= 0:
            assert ids[-1] == eos_id, "EOS not found despite add_eos=True"

    def test_sp_bos_eos_increases_length(self, sp_wrapper, sp_wrapper_with_bos_eos):
        """Adding BOS+EOS should produce exactly 2 more tokens."""
        text = _TEST_TEXT
        ids_plain = sp_wrapper.encode(text)
        ids_bos_eos = sp_wrapper_with_bos_eos.encode(text)
        sp = sp_wrapper._sp
        expected_extra = 0
        if sp.bos_id() >= 0:
            expected_extra += 1
        if sp.eos_id() >= 0:
            expected_extra += 1
        assert len(ids_bos_eos) == len(ids_plain) + expected_extra


class TestSentencePieceSpecialTokenReporting:
    """get_special_token_strings/get_special_token_ids must distinguish "cannot
    report" from "declares none", and must not silently inherit the base
    class's empty-set default for ids.
    """

    def test_special_token_strings_none_when_every_probe_fails(self):
        """A processor whose bos/eos/unk/pad probes all raise, and that exposes
        no IsControl/IsUnknown, cannot be asked for its special tokens at all.
        get_special_token_strings() must return None (the "cannot report"
        signal that makes resolve_special_token_strings warn and fall back to
        GENERIC_SPECIAL_TOKENS), not an empty set, which would assert this
        tokenizer genuinely declares none.
        """
        class UnreadableSPProcessor:
            def bos_id(self):
                raise RuntimeError("simulated failure")

            def eos_id(self):
                raise RuntimeError("simulated failure")

            def unk_id(self):
                raise RuntimeError("simulated failure")

            def pad_id(self):
                raise RuntimeError("simulated failure")

            def get_piece_size(self):
                return 0

        wrapper = SentencePieceTokenizer("unreadable", UnreadableSPProcessor(), {})
        assert wrapper.get_special_token_strings() is None

    def test_special_token_ids_read_declared_roles(self, sp_wrapper, sp_processor):
        """get_special_token_ids() must read the model's own bos/eos/unk/pad ids
        instead of inheriting TokenizerWrapper's default empty set, and must
        skip roles the model reports unset (sentencepiece signals unset with a
        negative id).
        """
        expected = {
            i for i in (sp_processor.bos_id(), sp_processor.eos_id(),
                        sp_processor.unk_id(), sp_processor.pad_id())
            if i >= 0
        }
        assert expected, "fixture model should declare at least one of bos/eos/unk/pad"
        assert sp_wrapper.get_special_token_ids() == expected


class TestConvertIdsRoundtrip:
    """convert_ids_to_tokens(encode(text)) should produce valid token strings."""

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_all_tokens_valid(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        ids = wrapper.encode(_TEST_TEXT)
        tokens = wrapper.convert_ids_to_tokens(ids)
        assert len(tokens) == len(ids)
        for i, tok_str in enumerate(tokens):
            assert isinstance(tok_str, str), (
                f"Token {i} is {type(tok_str)}, expected str"
            )
            assert not tok_str.startswith("<UNK_"), (
                f"Token ID {ids[i]} from encode() mapped to {tok_str!r}, "
                f"all IDs from encode() should be in the vocabulary"
            )


class TestDecodeRoundtrip:
    """decode(encode(text)) must exactly recover the original text.

    The test fixtures are deliberately configured to avoid any
    preprocessing that would alter the text before encoding:
    - HF ByteLevel: add_prefix_space=False (no leading space injection)
    - SentencePiece: normalization_rule_name="identity", add_dummy_prefix=False,
      remove_extra_whitespaces=False (no NFKC normalization, no dummy prefix)

    This ensures the roundtrip test is strict: any mismatch is a real bug,
    not an artifact of pretokenization normalization.
    """

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_roundtrip(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        if not wrapper.can_decode():
            pytest.skip(f"{wrapper.get_name()} does not support decode")
        text = _TEST_TEXT
        ids = wrapper.encode(text)
        decoded = wrapper.decode(ids)
        assert decoded is not None, (
            f"{wrapper.get_name()}: decode returned None"
        )
        assert decoded == text, (
            f"{wrapper.get_name()}: roundtrip mismatch: "
            f"{decoded!r} != {text!r}"
        )


class TestVocabConsistency:
    """get_vocab_size() bounds the number of distinct tokens.

    For gapless id spaces (HF/SP/custom_bpe) it equals len(get_vocab()).
    The script_bpe wrappers reserve id 0 (SCRIPT BPE) or leave gaps from
    MinGram's pruning, so vocab_size (the id-space bound) exceeds the token
    count; get_vocab() still holds one entry per real token.
    """

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_size_matches_dict(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        vocab = wrapper.get_vocab()
        if vocab is not None:
            assert wrapper.get_vocab_size() >= len(vocab), (
                f"{wrapper.get_name()}: get_vocab_size()={wrapper.get_vocab_size()} "
                f"< len(get_vocab())={len(vocab)}"
            )
            if not isinstance(wrapper, _ScriptTokTokenizer):
                assert wrapper.get_vocab_size() == len(vocab), (
                    f"{wrapper.get_name()}: get_vocab_size()={wrapper.get_vocab_size()} "
                    f"!= len(get_vocab())={len(vocab)}"
                )


class TestOffsetsCoverText:
    """Offsets from encode_with_offsets should span the input text."""

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_offsets_span_text(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        text = _TEST_TEXT
        ids, offsets = wrapper.encode_with_offsets(text)
        if offsets is None:
            pytest.skip(f"{wrapper.get_name()} does not provide offsets")
        assert len(offsets) == len(ids), (
            f"Offset count {len(offsets)} != token count {len(ids)}"
        )
        for i, (s, e) in enumerate(offsets):
            assert 0 <= s <= len(text), (
                f"Token {i}: start offset {s} out of range [0, {len(text)}]"
            )
            assert 0 <= e <= len(text), (
                f"Token {i}: end offset {e} out of range [0, {len(text)}]"
            )
            assert s <= e, (
                f"Token {i}: start {s} > end {e}"
            )

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_offsets_cover_all_characters(self, wrapper_name, request):
        """Every character in the text should be owned by at least one token."""
        wrapper = request.getfixturevalue(wrapper_name)
        text = _TEST_TEXT
        ids, offsets = wrapper.encode_with_offsets(text)
        if offsets is None:
            pytest.skip(f"{wrapper.get_name()} does not provide offsets")
        covered = set()
        for s, e in offsets:
            covered.update(range(s, e))
        for i in range(len(text)):
            assert i in covered, (
                f"Character {i} ({text[i]!r}) not covered by any token offset"
            )


class TestBatchEncoding:
    """encode_batch_with_offsets must match per-sample encode_with_offsets."""

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_batch_matches_single(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        texts = [_TEST_TEXT, _TEST_TEXT_MULTI]
        batch_results = wrapper.encode_batch_with_offsets(texts)
        assert len(batch_results) == len(texts)
        for text, (batch_ids, batch_offsets) in zip(texts, batch_results):
            single_ids, single_offsets = wrapper.encode_with_offsets(text)
            assert batch_ids == single_ids, (
                f"{wrapper.get_name()}: batch IDs differ from single for {text!r}"
            )
            assert batch_offsets == single_offsets, (
                f"{wrapper.get_name()}: batch offsets differ from single for {text!r}"
            )

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_empty_batch(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        assert wrapper.encode_batch_with_offsets([]) == []


# ── script_bpe wrappers (SCRIPT BPE + MinGram) ────────────────────────

_SCRIPT_TOK_FIXTURES = ["script_bpe_wrapper", "mingram_wrapper"]


class TestScriptTokWrappers:
    """Behaviour specific to the script_bpe wrappers.

    These tokenizers have no special tokens, expose no HuggingFace-style
    underlying object, and (under the SCRIPT pretokenizer) normalize text, so
    their contract differs from the HF/SP wrappers in ways the shared
    parametrized tests do not cover.
    """

    def test_registry_dispatches_to_correct_class(
        self, script_bpe_wrapper, mingram_wrapper
    ):
        assert isinstance(script_bpe_wrapper, ScriptBPETokenizer)
        assert isinstance(mingram_wrapper, MinGramTokenizer)

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_no_special_or_unk_tokens(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        assert wrapper.get_special_token_ids() == set()
        assert wrapper.get_unk_token_id() is None
        assert wrapper.has_unk_token() is False

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_underlying_tokenizer_is_none(self, wrapper_name, request):
        # None so MorphScore and the HF-internal sanity checks skip cleanly.
        wrapper = request.getfixturevalue(wrapper_name)
        assert wrapper.get_underlying_tokenizer() is None

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_encode_returns_plain_int_list(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        ids = wrapper.encode(_TEST_TEXT)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_offsets_are_reported_and_index_the_source(self, wrapper_name, request):
        # This used to assert offsets were None, on the grounds that the
        # bundled pretokenizer normalizes before it encodes. The wrapper now
        # aligns the normalized text back to the caller's text, so offsets are
        # reported and they index the caller's text: the decomposed spelling
        # below is one character longer than its normalized form, and its last
        # offset ends one character further right than the composed spelling's.
        wrapper = request.getfixturevalue(wrapper_name)
        ids, offsets = wrapper.encode_with_offsets(_TEST_TEXT)
        assert ids == wrapper.encode(_TEST_TEXT)
        assert offsets is not None
        assert len(offsets) == len(ids)

        nfd_ids, nfd_offsets = wrapper.encode_with_offsets(_NFD_CAFE)
        nfc_ids, nfc_offsets = wrapper.encode_with_offsets(_NFC_CAFE)
        assert nfd_ids == nfc_ids, "the two spellings normalize to the same text"
        assert nfd_offsets[-1][1] == len(_NFD_CAFE) == 5
        assert nfc_offsets[-1][1] == len(_NFC_CAFE) == 4

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_pretokenize_returns_strings(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        pretokens = wrapper.pretokenize("Hello 12 世界")
        assert isinstance(pretokens, list)
        assert pretokens
        assert all(isinstance(p, str) for p in pretokens)

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_get_vocab_is_faithful(self, wrapper_name, request):
        # tokens_repr is distinct per token for these fixtures, so get_vocab
        # holds exactly one entry per real token (no collisions collapse it).
        wrapper = request.getfixturevalue(wrapper_name)
        vocab = wrapper.get_vocab()
        assert len(vocab) == len(wrapper._backend.tokens)

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_id_space_is_compacted(self, wrapper_name, request):
        # The wrapper remaps the back end's (possibly sparse) ids to a gap-free
        # range, so vocab_size is the token count (plus one for a reserved id 0),
        # never the sparse max-native-id+1. This is what keeps the
        # vocabulary-utilization denominator from being inflated by MinGram's
        # pruning gaps. (On the tiny MinGram fixture the sparse bound is 2031 for
        # 1993 tokens; compaction must bring it to 1994.)
        wrapper = request.getfixturevalue(wrapper_name)
        n_tokens = len(wrapper._backend.tokens)
        assert n_tokens <= wrapper.get_vocab_size() <= n_tokens + 1
        # The compacted id space is contiguous (no holes) and vocab_size bounds it.
        dense_ids = sorted(wrapper._dense_to_native)
        assert dense_ids == list(range(dense_ids[0], dense_ids[0] + len(dense_ids)))
        assert wrapper.get_vocab_size() == dense_ids[-1] + 1
        # Every id encode can emit stays below the reported vocab size.
        ids = wrapper.encode("The quick brown fox 42 café 世界 x >= 10")
        assert all(0 <= i < wrapper.get_vocab_size() for i in ids)

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_decode_reencode_is_stable(self, wrapper_name, request):
        # decode(encode(text)) yields the normalized form, which re-encodes to
        # the same ids (a normalization-robust round-trip invariant).
        wrapper = request.getfixturevalue(wrapper_name)
        for text in [_TEST_TEXT, "世界 123 café", "def f(): return 1 + 2"]:
            ids = wrapper.encode(text)
            decoded = wrapper.decode(ids)
            assert isinstance(decoded, str)
            assert wrapper.encode(decoded) == ids

    def test_missing_path_raises_value_error(self):
        pytest.importorskip("script_bpe")
        with pytest.raises(ValueError):
            create_tokenizer_wrapper("bad", {"class": "script_bpe"})

    def test_missing_file_raises_file_not_found(self):
        pytest.importorskip("script_bpe")
        with pytest.raises(FileNotFoundError):
            create_tokenizer_wrapper(
                "bad", {"class": "mingram", "path": "/no/such/file.json.gz"}
            )

    def test_unknown_class_lists_new_keys(self):
        # The factory's error enumerates available classes, now including ours.
        with pytest.raises(ValueError) as exc:
            create_tokenizer_wrapper("x", {"class": "does_not_exist", "path": "p"})
        msg = str(exc.value)
        assert "script_bpe" in msg and "mingram" in msg


class TestScriptTokOffsets:
    """Character offsets and pretokenizer spans over the source text.

    The pretokenizer normalizes (NFC) and drops code points its script config
    does not cover before it encodes anything, so positions in the text it
    works on are not positions in the caller's text. These tests hold the
    wrapper to the caller's text: every offset indexes the string that was
    passed in, and the composing, dropping and regrouping steps in between are
    accounted for rather than approximated.

    Offsets may overlap. Unless a tokenizer's training config enforced
    character boundaries, a merged token can hold the tail of one character and
    the head of the next, and both fixtures have such tokens, so no test here
    asserts that offsets are disjoint.
    """

    @staticmethod
    def _covered(offsets):
        covered = set()
        for start, end in offsets:
            covered.update(range(start, end))
        return covered

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_ids_and_offsets_agree_over_a_battery(self, wrapper_name, request):
        """The ids are encode()'s, and every offset is a range within the text.

        Offsets also advance with the tokens: a start or an end that went
        backwards would mean the alignment lost its place in the text.
        """
        wrapper = request.getfixturevalue(wrapper_name)
        for text in _OFFSET_BATTERY:
            ids, offsets = wrapper.encode_with_offsets(text)
            assert ids == wrapper.encode(text), f"ids differ for {text!r}"
            assert offsets is not None, f"no offsets for {text!r}"
            assert len(offsets) == len(ids), f"offset count differs for {text!r}"
            previous = (0, 0)
            for start, end in offsets:
                assert 0 <= start <= end <= len(text), (
                    f"{text!r}: offset ({start}, {end}) is not a range in the text"
                )
                assert previous[0] <= start and previous[1] <= end, (
                    f"{text!r}: offsets went backwards at ({start}, {end})"
                )
                previous = (start, end)

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_every_character_is_covered_when_none_is_dropped(
        self, wrapper_name, request
    ):
        """No character of these inputs is left out of every token's offsets.

        Every character in the battery survives normalization, so each one is
        part of some token. A character the script config drops is a different
        case, covered by test_a_dropped_character_shifts_the_offsets.
        """
        wrapper = request.getfixturevalue(wrapper_name)
        for text in _OFFSET_BATTERY:
            _ids, offsets = wrapper.encode_with_offsets(text)
            covered = self._covered(offsets)
            missing = [i for i in range(len(text)) if i not in covered]
            assert not missing, (
                f"{text!r}: characters {missing} are in no token's offsets"
            )

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_a_composed_character_reaches_both_source_characters(
        self, wrapper_name, request
    ):
        """A letter plus a combining acute is one character after normalizing.

        Whichever token holds the composed character has to report both source
        positions, since neither one alone produced it. Reporting positions in
        the normalized text would stop at index 4 of a 5-character string.
        """
        wrapper = request.getfixturevalue(wrapper_name)
        _ids, offsets = wrapper.encode_with_offsets(_NFD_CAFE)
        assert self._covered(offsets) == set(range(len(_NFD_CAFE)))
        holders = [(s, e) for s, e in offsets if s <= 3 < e]
        assert holders, "no token covers the base letter of the composition"
        assert all(e >= 5 for _s, e in holders), (
            f"a token covering source character 3 stops before the combining "
            f"mark at 4: {holders}"
        )

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_hangul_jamo_compose_across_a_starter_boundary(
        self, wrapper_name, request
    ):
        """Three combining-class-0 characters that NFC composes into one.

        An alignment that cut the text wherever the combining class is 0 would
        cut between the jamo, so its reconstruction would be the three jamo
        rather than the syllable, and the wrapper would report no offsets at
        all. Naming this input separately is the point: it is the case where
        the cheap rule silently produces a different text.
        """
        wrapper = request.getfixturevalue(wrapper_name)
        ids, offsets = wrapper.encode_with_offsets(_HANGUL_JAMO)
        assert ids == wrapper.encode(_HANGUL_SYLLABLE), (
            "the three jamo encode as the syllable they compose into"
        )
        assert offsets is not None, (
            "the jamo alignment could not be established, so no offsets were "
            "reported"
        )
        assert self._covered(offsets) == set(range(len(_HANGUL_JAMO)))
        assert all((s, e) == (0, 3) for s, e in offsets), (
            f"every token here holds the one composed syllable: {offsets}"
        )

    def test_normalization_segments_are_as_fine_as_composition_allows(self):
        """The alignment cuts the text only where normalization lets it.

        Three cases in one call, because they pull in opposite directions:
        "e" and its combining acute compose, so they stay together; the three
        jamo compose across two combining-class-0 characters, so cutting where
        the combining class is 0 would separate them; and the combining
        overline composes with nothing, so it keeps its own position instead of
        inheriting its neighbour's span.
        """
        text = _NFD_CAFE + " " + _HANGUL_JAMO + " a\u0305"
        assert _ScriptTokTokenizer._normalization_segments(text, "NFC") == [
            (0, 1), (1, 2), (2, 3), (3, 5),   # c, a, f, e+acute
            (5, 6),                            # space
            (6, 9),                            # the three jamo
            (9, 10),                           # space
            (10, 11), (11, 12),                # a, and the overline on its own
        ]

    def test_a_range_is_cut_at_the_earliest_valid_position(self):
        """Where several cuts are valid, the first one has to be taken.

        _refine_segment stops at the first position where normalizing the two
        sides separately rebuilds the whole range. Taking a later valid cut
        instead is not caught by any invariant, because a coarser partition
        still reproduces the normalization; it just reports wider spans. The
        case above cannot see the difference, since none of its ranges has a
        second valid cut. This one does: 'b' takes no precomposed form with a
        following acute, so both 4 and 5 are valid cuts of the range 3 to 6,
        and dropping the early exit merges 'b' with its first mark.
        """
        text = "a\u0301\u0301b\u0301\u0301"
        assert _ScriptTokTokenizer._normalization_segments(text, "NFC") == [
            (0, 2),            # a and its acute, which compose
            (2, 3),            # the second acute, which composes with nothing
            (3, 4), (4, 5), (5, 6),   # b and its two marks, each on its own
        ]

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_a_dropped_character_shifts_the_offsets(self, wrapper_name, request):
        """A dropped code point changes the offsets but not the ids.

        U+FFF0 is unassigned, so the pretokenizer removes it and both spellings
        encode identically. The offsets must still index the text as passed in,
        which is what makes them differ between the two. Whether the dropped
        position is covered depends on whether a merged token reaches across
        it, so this asserts only that the surviving characters are covered.
        """
        wrapper = request.getfixturevalue(wrapper_name)
        with_dropped = "a" + _DROPPED_CHAR + "b"
        without = "ab"
        ids_with, offsets_with = wrapper.encode_with_offsets(with_dropped)
        ids_without, offsets_without = wrapper.encode_with_offsets(without)
        assert ids_with == ids_without
        assert offsets_with != offsets_without, (
            "offsets index the source text, so the dropped character must push "
            "everything after it one position right"
        )
        covered = self._covered(offsets_with)
        assert 0 in covered and 2 in covered, (
            f"the surviving characters are not both covered: {offsets_with}"
        )
        assert max(end for _s, end in offsets_with) == len(with_dropped)

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_empty_and_whitespace_only_input(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        assert wrapper.encode_with_offsets("") == ([], [])
        assert wrapper.pretokenize_with_spans("") == []
        whitespace = "  \t \n  "
        _ids, offsets = wrapper.encode_with_offsets(whitespace)
        assert self._covered(offsets) == set(range(len(whitespace)))

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_pretokenize_with_spans_agrees_with_pretokenize(
        self, wrapper_name, request
    ):
        """Same surfaces as pretokenize(), and each span produces its surface.

        Slicing the source text at a span and normalizing it must give that
        span's surface back. That is what lets C10 pretokenizer char
        conservation measure coverage against source positions rather than
        against surface lengths, which a byte-level surface inflates.

        It does not follow that character loss always shows up as a gap. A
        chunk whose characters sit on either side of one the pretokenizer
        dropped reports a single range across it: ``'a￰b'`` on the SCRIPT
        BPE fixture gives one chunk spanning (0, 3), so the dropped character
        at index 1 reads as covered and C10 reports conservation 1.0 on text
        that lost a character. See the C10 entry in docs/SANITY_CHECKS.md.

        One input class is excluded, and none of the battery is in it: when
        normalization reorders a run of combining marks and the reordering
        moves one of them past a chunk boundary, no mark in the run can be
        given a span of its own, so every chunk holding part of the run reports
        the run's whole span and slicing any of them returns the whole run.
        """
        wrapper = request.getfixturevalue(wrapper_name)
        normalize = wrapper._pretokenizer.normalize
        for text in _OFFSET_BATTERY:
            spans = wrapper.pretokenize_with_spans(text)
            assert spans is not None, f"no spans for {text!r}"
            assert [surface for surface, _span in spans] == wrapper.pretokenize(text)
            previous = (0, 0)
            for surface, (start, end) in spans:
                assert 0 <= start <= end <= len(text), f"{text!r}: bad span"
                assert previous[0] <= start and previous[1] <= end, (
                    f"{text!r}: spans went backwards at ({start}, {end})"
                )
                previous = (start, end)
                assert normalize(text[start:end]) == surface, (
                    f"{text!r}: span ({start}, {end}) holds "
                    f"{text[start:end]!r}, not {surface!r}"
                )

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_spans_are_character_positions_not_byte_positions(
        self, wrapper_name, request
    ):
        """A CJK chunk spans its characters, not the bytes they encode to.

        Three bytes per character is what made the C10 conservation check read
        1.000 while half the text was being dropped, so the widths are asserted
        against the character count.
        """
        wrapper = request.getfixturevalue(wrapper_name)
        text = "世界 こんにちは"
        spans = wrapper.pretokenize_with_spans(text)
        assert spans is not None
        assert spans[0] == ("世界", (0, 2))
        assert sum(end - start for _surface, (start, end) in spans) == len(text)

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_digit_run_spans(self, wrapper_name, request):
        """Digits are their own chunk, spanning exactly the source digits.

        These fixtures leave digit_handling unset, so a run stays whole; the
        regrouped case is covered in TestScriptTokOffsetsOtherPretokenizers.
        """
        wrapper = request.getfixturevalue(wrapper_name)
        text = "1234567890 and 42"
        spans = wrapper.pretokenize_with_spans(text)
        assert spans is not None
        assert ("1234567890", (0, 10)) in spans
        assert ("42", (15, 17)) in spans

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_batch_offsets_match_single(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        texts = [t for t in _OFFSET_BATTERY if t]
        assert wrapper.encode_batch_with_offsets(texts) == [
            wrapper.encode_with_offsets(text) for text in texts
        ]

    # ── the inputs that get no offsets ────────────────────────────────

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_no_offsets_when_match_positions_cannot_be_established(
        self, wrapper_name, request, monkeypatch
    ):
        """A capture group in regex_pattern makes findall report groups.

        ``regex_split`` returns what ``findall`` returns, which for a pattern
        with a capture group is the group and not the whole match, so the
        matches cannot be paired with positions. The wrapper reports no offsets
        rather than pairing them anyway; the ids are unaffected.

        The split itself is asserted on directly as well. Every later check
        would also refuse this text, because a replay that split it differently
        from the pretokenizer no longer matches the tokens it produced, so the
        black-box result alone does not say that the positions were the thing
        found wanting.
        """
        wrapper = request.getfixturevalue(wrapper_name)
        monkeypatch.setattr(
            wrapper._pretokenizer.config, "regex_pattern", r"(\w)\w*"
        )
        assert wrapper._regex_split_spans(_TEST_TEXT, 0, len(_TEST_TEXT)) is None
        ids, offsets = wrapper.encode_with_offsets(_TEST_TEXT)
        assert ids == wrapper.encode(_TEST_TEXT)
        assert offsets is None
        assert wrapper.pretokenize_with_spans(_TEST_TEXT) is None

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_no_offsets_when_split_encoded_returns_other_objects(
        self, wrapper_name, request, monkeypatch
    ):
        """Spans follow the encoded characters by identity through split_encoded.

        A ``split_encoded`` that returned copies would still produce the same
        atomic tokens, so the tokenization would look right while the wrapper
        no longer knew which character each group came from. It reports no
        offsets instead of assuming the order was kept.
        """
        import copy

        wrapper = request.getfixturevalue(wrapper_name)
        original = wrapper._pretokenizer.split_encoded
        monkeypatch.setattr(
            wrapper._pretokenizer,
            "split_encoded",
            lambda encs: [[copy.copy(enc) for enc in group]
                          for group in original(encs)],
        )
        ids, offsets = wrapper.encode_with_offsets(_TEST_TEXT)
        assert ids == wrapper.encode(_TEST_TEXT)
        assert offsets is None
        assert wrapper.pretokenize_with_spans(_TEST_TEXT) is None


class TestScriptTokEdgeCases:
    """Edge cases that distinguish these wrappers from the dense HF/SP ones.

    The interesting properties: a non-contiguous id space (1-based, with
    reserved/pruned gaps), a bundled pretokenizer whose normalization makes
    the decode round-trip non-identity, token strings that come from
    ``tokens_repr`` (not per-id decode), the ``reindex`` load knob, and
    loading that must fail loudly when the class does not match the file.
    """

    # ── id space ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_encoded_ids_are_within_vocab_size(self, wrapper_name, request):
        # The input validator and id-indexed metrics require id < vocab_size.
        # These tokenizers have a sparse/1-based id space, so vocab_size must be
        # the id-space bound (max id + 1), not the token count; this guards that.
        wrapper = request.getfixturevalue(wrapper_name)
        vocab_size = wrapper.get_vocab_size()
        text = ("The quick brown fox 42 café Zürich Πολύγλωσσο 世界 "
                "x >= 10 and 3 + 4 * 5 = 23; def f(): return 1 + 2")
        ids = wrapper.encode(text)
        assert ids, "expected a non-empty encoding for a rich sample"
        assert all(0 <= i < vocab_size for i in ids), (
            f"{wrapper.get_name()}: encoded ids outside [0, {vocab_size}): "
            f"{[i for i in ids if not (0 <= i < vocab_size)][:5]}"
        )

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_reserved_and_out_of_range_ids_map_to_unk(self, wrapper_name, request):
        # Id 0 is reserved (absent from the token table) and vocab_size is one
        # past the end; convert must fall back to a sentinel rather than raise
        # or return a real token string.
        wrapper = request.getfixturevalue(wrapper_name)
        vocab_size = wrapper.get_vocab_size()
        assert wrapper.convert_ids_to_tokens([0]) == ["<UNK_0>"]
        assert wrapper.convert_ids_to_tokens([vocab_size]) == [f"<UNK_{vocab_size}>"]

    # ── normalization / round-trip ────────────────────────────────────

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_normalization_is_not_identity(self, wrapper_name, request):
        # The bundled pretokenizer applies NFC, so a decomposed (NFD) input
        # comes back composed (NFC). This is why reconstruction exact-match is
        # below 1.0 for normalizing inputs; it is expected, not a decode bug.
        import unicodedata
        wrapper = request.getfixturevalue(wrapper_name)
        nfd = "café"                      # 'e' + combining acute
        nfc = unicodedata.normalize("NFC", nfd)  # 'é'
        assert nfd != nfc
        decoded = wrapper.decode(wrapper.encode(nfd))
        assert decoded == nfc
        assert decoded != nfd

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_empty_and_whitespace_inputs(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        assert wrapper.encode("") == []
        assert wrapper.decode([]) == ""
        # Whitespace is not normalized away, so it round-trips exactly.
        ws = "  \t \n  "
        assert wrapper.decode(wrapper.encode(ws)) == ws

    # ── vocab views agree ─────────────────────────────────────────────

    @pytest.mark.parametrize("wrapper_name", _SCRIPT_TOK_FIXTURES)
    def test_vocab_and_convert_match_the_backend(self, wrapper_name, request):
        # Independent check against the back end: for real tokens, the compacted
        # id must render (via convert_ids_to_tokens and via get_vocab) to the back
        # end's own tokens_repr string, and the native->compacted remap must land
        # on that id. This catches a wrong remap or wrong rendering, which a plain
        # get_vocab-vs-convert comparison cannot (both share one code path).
        wrapper = request.getfixturevalue(wrapper_name)
        vocab = wrapper.get_vocab()
        backend = wrapper._backend
        native_ids = sorted(backend.tokens)
        mid = len(native_ids) // 2
        # A spread of base atomic tokens (low ids) and merged tokens (high ids).
        sample = native_ids[:3] + native_ids[mid:mid + 3] + native_ids[-3:]
        for nid in sample:
            expected = backend.pretokenizer.tokens_repr(backend.tokens[nid].atomic_tokens)
            dense = wrapper._native_to_dense[nid]
            assert wrapper.convert_ids_to_tokens([dense]) == [expected]
            assert vocab[expected] == dense

    # ── load knobs and failure modes ──────────────────────────────────

    def test_mingram_reindex_changes_origin_not_density(self):
        # The wrapper compacts gaps either way, so both loads give a dense id
        # space; reindex only changes the origin. Reindex packs ids 0-based, so
        # vocab_size == token count; the default keeps the reserved id 0, so
        # vocab_size == token count + 1. Neither is inflated by MinGram's gaps.
        pytest.importorskip("script_bpe")
        path = os.path.join(FIXTURE_DIR, "mingram_tiny.json.gz")
        default = create_tokenizer_wrapper("mg", {"class": "mingram", "path": path})
        reindexed = create_tokenizer_wrapper(
            "mg-ri", {"class": "mingram", "path": path, "reindex": True}
        )
        n = len(default._backend.tokens)
        assert len(reindexed._backend.tokens) == n
        assert reindexed.get_vocab_size() == n          # 0-based dense
        assert default.get_vocab_size() == n + 1        # 1-based dense (reserved id 0)
        # Renumbering ids does not change what either tokenizer represents.
        assert reindexed.decode(reindexed.encode("Hello world 42")) == "Hello world 42"
        assert default.decode(default.encode("Hello world 42")) == "Hello world 42"

    def test_wrong_class_for_file_fails_loudly(self):
        # Pointing the wrong class at a file must raise, not silently build a
        # broken tokenizer (the on-disk formats are class-specific).
        pytest.importorskip("script_bpe")
        bpe_file = os.path.join(FIXTURE_DIR, "scriptbpe_tiny.json.gz")
        mingram_file = os.path.join(FIXTURE_DIR, "mingram_tiny.json.gz")
        with pytest.raises(Exception):
            create_tokenizer_wrapper("x", {"class": "mingram", "path": bpe_file})
        with pytest.raises(Exception):
            create_tokenizer_wrapper("x", {"class": "script_bpe", "path": mingram_file})


class TestScriptTokOffsetsOtherPretokenizers:
    """Offsets under pretokenizer settings no committed fixture uses.

    Both fixtures were trained with the SCRIPT pretokenizer, no regex pattern
    and no digit handling, which leaves three branches of the offset code with
    no coverage: one CharEnc per UTF-8 byte instead of one per character, a
    regex split whose match positions have to be recovered, and digit runs that
    are regrouped before encoding. Training a tokenizer for each setting would
    be a lot of fixture for the question, so these use a back end whose
    vocabulary is exactly the pretokenizer's atomic tokens. Only the offsets are
    of interest here; the merged-token walk is what the two fixtures exercise.
    """

    @staticmethod
    def _wrap(pretokenizer, name):
        from script_bpe.tokenizers.base import BaseToken
        from script_bpe.utils import token_array

        class _AtomicBackend:
            """One token per atomic token, so encode is the pretokenizer's output."""

            def __init__(self, pretok):
                self.pretokenizer = pretok
                self.tokens = {tid: BaseToken(tid, token_array([tid]))
                               for tid in pretok.atomic_tokens}

            def encode(self, text):
                return [tid for chunk in self.pretokenizer.pretokenize(text)
                        for tid in chunk]

            def decode(self, ids, errors="replace"):
                return self.pretokenizer.decode(ids, errors=errors)

        return ScriptBPETokenizer(name, _AtomicBackend(pretokenizer), {})

    @pytest.fixture(scope="module")
    def utf8_regex_digits_wrapper(self):
        """UTF-8 pretokenizer, GPT-4 regex split, digits regrouped in threes."""
        pytest.importorskip("script_bpe")
        from script_bpe.pretokenize import GPT4_REGEX, UTF8PretokenizerConfig
        from script_bpe.pretokenize.pretokenizer import UTF8Pretokenizer

        return self._wrap(
            UTF8Pretokenizer(UTF8PretokenizerConfig(
                regex_pattern=GPT4_REGEX,
                digit_handling="RTL3",
                enforce_char_boundaries=False,
            )),
            "utf8-gpt4-rtl3",
        )

    @pytest.fixture(scope="module")
    def utf8_digits_wrapper(self):
        """UTF-8 pretokenizer with digits regrouped and no regex split.

        Without a pattern, the digit split's empty pieces reach the encoder and
        become empty chunks, which is the case that has no atomic token to take
        a position from.
        """
        pytest.importorskip("script_bpe")
        from script_bpe.pretokenize import UTF8PretokenizerConfig
        from script_bpe.pretokenize.pretokenizer import UTF8Pretokenizer

        return self._wrap(
            UTF8Pretokenizer(UTF8PretokenizerConfig(
                digit_handling="RTL3", enforce_char_boundaries=False)),
            "utf8-rtl3",
        )

    def test_the_bytes_of_one_character_share_that_characters_span(
        self, utf8_regex_digits_wrapper
    ):
        """Each of a CJK character's three tokens reports the character, not a byte.

        This pretokenizer emits one token per UTF-8 byte, so a wrapper that
        counted CharEncs as characters would walk three positions right per
        character and report offsets far past the end of the text.
        """
        wrapper = utf8_regex_digits_wrapper
        text = "a 世界"
        ids, offsets = wrapper.encode_with_offsets(text)
        assert ids == wrapper.encode(text)
        assert offsets is not None
        assert len(offsets) == len("a 世界".encode("utf-8"))
        assert offsets[-6:] == [(2, 3)] * 3 + [(3, 4)] * 3
        assert max(end for _s, end in offsets) == len(text)

    def test_regrouped_digit_runs_span_their_source_digits(
        self, utf8_regex_digits_wrapper
    ):
        """RTL3 cuts 1234567 into 1, 234, 567; each group spans its own digits."""
        wrapper = utf8_regex_digits_wrapper
        text = "abc 1234567 def"
        spans = wrapper.pretokenize_with_spans(text)
        assert spans is not None
        assert [surface for surface, _span in spans] == wrapper.pretokenize(text)
        assert ("1", (4, 5)) in spans
        assert ("234", (5, 8)) in spans
        assert ("567", (8, 11)) in spans

    def test_empty_chunks_keep_a_position(self, utf8_digits_wrapper):
        """The digit split's empty pieces still get a zero-width span.

        pretokenize() reports one surface per chunk including the empty ones,
        so pretokenize_with_spans has to report one span per chunk too, or the
        two lists stop lining up.
        """
        wrapper = utf8_digits_wrapper
        text = "12ab"
        surfaces = wrapper.pretokenize(text)
        spans = wrapper.pretokenize_with_spans(text)
        assert spans is not None
        assert "" in surfaces, "expected the digit split to produce an empty piece"
        assert [surface for surface, _span in spans] == surfaces
        for surface, (start, end) in spans:
            if surface == "":
                assert start == end, f"an empty chunk got a non-empty span at {start}"
        assert ("12", (0, 2)) in spans
        assert ("ab", (2, 4)) in spans

    def test_a_trailing_empty_chunk_sits_at_the_end_of_the_text(
            self, utf8_digits_wrapper):
        """Zero width is not enough; the position has to be right.

        The test above only reaches a leading empty chunk, whose span is (0, 0)
        under any rule, and it asserts start == end, which every wrong position
        also satisfies. Putting the digits last moves the empty chunk to the
        end, where reporting (0, 0) instead of the end of the text would place
        a chunk before the text it follows and still pass that assertion. Full
        spans are asserted here rather than a property, so the position is
        pinned rather than described.
        """
        wrapper = utf8_digits_wrapper
        text = "ab12"
        spans = wrapper.pretokenize_with_spans(text)
        assert spans is not None
        assert [surface for surface, _span in spans] == wrapper.pretokenize(text)
        assert spans == [("ab", (0, 2)), ("12", (2, 4)), ("", (4, 4))]

        # A text that is only digits has an empty chunk at each end, so the two
        # cannot both be reported at the same place.
        assert wrapper.pretokenize_with_spans("12") == [
            ("", (0, 0)), ("12", (0, 2)), ("", (2, 2))]


def test_a_directory_of_vocab_and_merges_loads(tmp_path):
    """The vocab.json plus merges.txt route could never load a tokenizer.

    Strategy 3 of `_load_huggingface_tokenizer` called
    `_load_bpe_from_directory`, a name the module never defined. The resulting
    NameError was caught by the `except Exception` around it, logged as a
    failure to read that one file, and the load fell through to the final
    ValueError naming the path, so the failure read as a bad directory rather
    than as a defect.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    from tokenizer_analysis.utils.tokenizer_utils import _load_huggingface_tokenizer

    trained = Tokenizer(models.BPE(unk_token="<unk>"))
    trained.pre_tokenizer = pre_tokenizers.Whitespace()
    trained.train_from_iterator(
        ["hello world " * 50, "the quick brown fox " * 50],
        trainers.BpeTrainer(vocab_size=100, special_tokens=["<unk>"]),
    )
    trained.model.save(str(tmp_path))
    assert (tmp_path / "vocab.json").exists() and (tmp_path / "merges.txt").exists()

    loaded = _load_huggingface_tokenizer({"path": str(tmp_path)})
    assert loaded.get_vocab_size() > 0
    assert "hello" in loaded.encode("hello world").tokens


class _TransformersLike:
    """The attribute layout of a transformers fast tokenizer.

    The Rust ``tokenizers.Tokenizer`` hangs off ``backend_tokenizer``; the
    outer object carries no ``pre_tokenizer`` of its own.
    """

    def __init__(self, backend):
        self.backend_tokenizer = backend

    def get_vocab(self):
        return self.backend_tokenizer.get_vocab()


class TestPretokenizerResolvesThroughTheTransformersBackend:
    """``can_pretokenize`` must look where a real tokenizer keeps it.

    Every other fixture in this file hands ``HuggingFaceTokenizer`` a raw
    ``tokenizers.Tokenizer``, which carries ``pre_tokenizer`` directly, so the
    suite only ever exercised the shape that already worked. Anything loaded
    through ``AutoTokenizer`` has the other shape. Reading ``pre_tokenizer``
    off the outer object returned ``False`` for all nine tokenizers of
    ``benchmarks/open_source``, which made C10 pretokenizer char conservation
    ``not_applicable`` for every one of them and made C16 report 0
    pretokenizer-unreachable vocabulary tokens for bert-base-uncased where the
    measured count is 6823.
    """

    def test_can_pretokenize_when_only_the_backend_carries_it(self, trained_hf_tokenizer):
        outer = _TransformersLike(trained_hf_tokenizer)
        assert not hasattr(outer, "pre_tokenizer"), "fixture must not expose it directly"
        wrapper = HuggingFaceTokenizer("backend-only", outer, {})
        assert wrapper.can_pretokenize() is True

    def test_spans_cover_the_non_whitespace_characters(self, trained_hf_tokenizer):
        wrapper = HuggingFaceTokenizer(
            "backend-only", _TransformersLike(trained_hf_tokenizer), {}
        )
        text = "hello world"
        spans = wrapper.pretokenize_with_spans(text)
        assert spans, "the backend pre-tokenizer returned no spans"
        covered = set()
        for _surface, (start, end) in spans:
            covered.update(range(start, end))
        uncovered = [i for i, ch in enumerate(text) if not ch.isspace() and i not in covered]
        assert not uncovered, f"characters {uncovered} of {text!r} are in no span"

    def test_pretokenize_returns_surfaces(self, trained_hf_tokenizer):
        wrapper = HuggingFaceTokenizer(
            "backend-only", _TransformersLike(trained_hf_tokenizer), {}
        )
        assert len(wrapper.pretokenize("hello world")) >= 2

    def test_an_absent_pretokenizer_is_still_absent(self):
        """Resolving through the backend must not invent a pre-tokenizer."""
        from tokenizers import Tokenizer
        from tokenizers.models import BPE

        bare = Tokenizer(BPE(unk_token="<unk>"))
        assert bare.pre_tokenizer is None
        wrapper = HuggingFaceTokenizer("no-pretok", _TransformersLike(bare), {})
        assert wrapper.can_pretokenize() is False
        assert wrapper.pretokenize_with_spans("hello") is None
        with pytest.raises(NotImplementedError):
            wrapper.pretokenize("hello")


class TestScriptTokOffsetGuardsRefuseRatherThanGuess:
    """Each runtime invariant must stop the offsets, not shift them.

    The whole design rests on one claim: where the replay of script_bpe's
    pretokenization cannot be shown to match the real pipeline, the wrapper
    reports no offsets rather than approximate ones. Eleven checks enforce that,
    and an adversarial review found three of them that could be deleted with the
    suite still green, because no input reaches them and nothing faked one.
    A guard nobody tests reads as dead code to the next person.

    Each test below breaks one assumption at run time and asserts two things:
    offsets are None, and the ids are still the ids `encode` returns, since a
    failed alignment must not disturb the tokenization.
    """

    @pytest.fixture
    def wrapper(self, script_bpe_wrapper):
        """The shared fixture with its warn-once memory cleared.

        `_no_offsets` records each reason for the lifetime of the wrapper, and
        the fixture is module-scoped, so a reason another test already triggered
        would be silently skipped here.
        """
        script_bpe_wrapper._offset_failures.clear()
        return script_bpe_wrapper

    def test_encoding_that_depends_on_neighbouring_characters(
            self, wrapper, monkeypatch, caplog):
        """Assumption: `encode_text` is character-local.

        The replay pairs one CharEnc with one character by encoding characters
        one at a time. A pretokenizer whose output depended on context would
        make that pairing wrong, so the per-character stream is compared against
        encoding the whole piece.
        """
        pretok = wrapper._pretokenizer
        real = pretok.encode_text

        def context_dependent(text):
            # Longer input takes a different path, so the whole-piece encoding
            # stops matching the concatenated per-character one.
            return real(text if len(text) < 2 else text[:-1])

        monkeypatch.setattr(pretok, "encode_text", context_dependent)
        with caplog.at_level("WARNING"):
            ids, offsets = wrapper.encode_with_offsets("hello world")
        assert offsets is None
        assert ids == wrapper.encode("hello world")
        assert any("one character at a time" in r.message for r in caplog.records), \
            [r.message for r in caplog.records]

    def test_tokens_that_cover_only_part_of_the_atomic_stream(
            self, wrapper, monkeypatch, caplog):
        """Assumption: the tokens spell out the whole pretokenized stream.

        Reporting spans for the tokens that did match would silently describe a
        prefix of the text as though it were all of it.
        """
        backend = wrapper._backend
        real = backend.encode
        monkeypatch.setattr(backend, "encode", lambda text: list(real(text))[:-1])
        with caplog.at_level("WARNING"):
            ids, offsets = wrapper.encode_with_offsets("hello world")
        assert offsets is None
        assert ids == wrapper.encode("hello world")
        assert any("cover only part" in r.message for r in caplog.records), \
            [r.message for r in caplog.records]

    def test_a_segmentation_that_does_not_rebuild_the_normalized_text(
            self, wrapper, monkeypatch, caplog):
        """Assumption: the NFC segmentation reproduces the whole normalization.

        This is the check that makes a wrong segmentation refuse instead of
        reporting shifted spans. A segmentation can only be checked against the
        text it claims to cover, so breaking the segmenter is the only way in.

        The message is asserted, not just the None. The check below this one
        compares the same text after the unassigned-character filter has run, so
        it catches most of the same failures and would leave this test passing
        with this guard deleted. The two are not interchangeable: this one names
        the segmentation, the other names the filtered result, and only this one
        fires when the filter happens to remove exactly the characters the
        segmentation got wrong.
        """
        from tokenizer_analysis.core.tokenizer_wrapper import _ScriptTokTokenizer

        monkeypatch.setattr(
            _ScriptTokTokenizer, "_normalization_segments",
            staticmethod(lambda text, form: [(0, max(1, len(text) - 1))]),
        )
        # Text the pretokenizer's own normalizer changes, so the segmenting
        # branch runs at all.
        text = _NFD_CAFE
        assert wrapper._pretokenizer.normalize(text) != text
        with caplog.at_level("WARNING"):
            ids, offsets = wrapper.encode_with_offsets(text)
        assert offsets is None
        assert ids == wrapper.encode(text)
        assert any("normalizing the aligned segments" in r.message
                   for r in caplog.records), [r.message for r in caplog.records]


class TestCustomBPEHandlesBothShapesItsLoaderReturns:
    """`custom_bpe` is written for the Rust API and can be handed the other one.

    `from_config` loads through `_load_custom_bpe_from_directory`, which reads
    only a directory of vocab.json plus merges.txt and returns a raw
    `tokenizers.Tokenizer`. A caller constructing the class directly can pass a
    transformers fast tokenizer instead (the class name predates this
    correction: the loader itself returns only the raw shape). Every method of
    `CustomBPETokenizer` read the Rust API directly, so on that shape four of
    them raised AttributeError and `can_pretokenize()` answered False for a
    tokenizer that has a pre-tokenizer, which is the one silent failure of the
    five. The fixture below is the shape that already worked; this class covers
    the other one.
    """

    @pytest.fixture(scope="class")
    def transformers_shaped(self, trained_hf_tokenizer):
        return CustomBPETokenizer(
            "cbpe-transformers", _TransformersLike(trained_hf_tokenizer), {})

    def test_every_method_agrees_with_the_raw_shape(
            self, transformers_shaped, custom_bpe_wrapper):
        """Same tokenizer object underneath, so every answer has to match."""
        text = _TEST_TEXT_MULTI
        assert transformers_shaped.encode(text) == custom_bpe_wrapper.encode(text)
        assert (transformers_shaped.encode_with_offsets(text)
                == custom_bpe_wrapper.encode_with_offsets(text))
        assert (transformers_shaped.encode_batch_with_offsets([text, _TEST_TEXT])
                == custom_bpe_wrapper.encode_batch_with_offsets([text, _TEST_TEXT]))
        ids = transformers_shaped.encode(text)
        assert (transformers_shaped.convert_ids_to_tokens(ids)
                == custom_bpe_wrapper.convert_ids_to_tokens(ids))
        assert (transformers_shaped.decode(ids)
                == custom_bpe_wrapper.decode(ids))

    def test_the_pretokenizer_is_found_through_the_backend(
            self, transformers_shaped, custom_bpe_wrapper):
        """The silent one: False here meant no pre-tokenizer, which is wrong."""
        assert transformers_shaped.can_pretokenize() is True
        text = "héllo wörld"
        assert (transformers_shaped.pretokenize(text)
                == custom_bpe_wrapper.pretokenize(text))

    def test_spans_are_reported_rather_than_discarded(self, custom_bpe_wrapper):
        """`pre_tokenize_str` returns spans and `pretokenize` drops them.

        Byte-level surfaces are why the spans matter: 'héllo' renders as
        'hÃ©llo', so a consumer measuring coverage by surface length overcounts
        exactly where a character is multi-byte.
        """
        text = "héllo wörld"
        spans = custom_bpe_wrapper.pretokenize_with_spans(text)
        assert spans is not None
        assert [surface for surface, _span in spans] == custom_bpe_wrapper.pretokenize(text)
        for _surface, (start, end) in spans:
            assert 0 <= start <= end <= len(text)
        assert "".join(text[s:e] for _surface, (s, e) in spans) == text

    def test_c10_no_longer_depends_on_which_wrapper_class_loaded_the_file(
            self, trained_hf_tokenizer):
        """The same tokenizer must get the same health verdict either way.

        Measured before this fix on tokenizers/bpe.json: C10 was `unverifiable`
        under `custom_bpe` and `pass` at 1.0 under `huggingface`, so the wrapper
        class rather than the tokenizer decided whether the check could run, and
        `unverifiable` forces the overall verdict to at least `warn`.
        """
        from tokenizer_analysis.diagnostics.probe_corpus import builtin_probes
        from tokenizer_analysis.diagnostics.sanity_check import (
            Severity, TokenizerSanityChecker,
        )

        probes = builtin_probes()
        results = {}
        for label, wrapper in (
            ("custom_bpe", CustomBPETokenizer("c", trained_hf_tokenizer, {})),
            ("huggingface", HuggingFaceTokenizer("h", trained_hf_tokenizer, {})),
        ):
            check = TokenizerSanityChecker(
                wrapper=wrapper, probes=probes).check_pretok_conservation()
            results[label] = (check["severity"], check["observed"])
        assert results["custom_bpe"] == results["huggingface"], results
        assert results["custom_bpe"][0] != Severity.UNVERIFIABLE, results


class TestUniMixLMResolvesThroughItsBaseTokenizer:
    """UniMixLM overrode two of the three methods built on one resolver.

    `HuggingFaceTokenizer` derives `can_pretokenize`, `pretokenize` and
    `pretokenize_with_spans` from `_pre_tokenizer()`. UniMixLM overrode the
    first two to read `base_tokenizer`, and inherited the third, which resolved
    through the UniMixLM object instead and found nothing. The result was a
    wrapper that said it could pretokenize, did pretokenize, and reported no
    spans, so sanity check C10 came out `unverifiable`.
    """

    class _UniMixLike:
        """A UniMixLM model's layout: the Rust tokenizer under base_tokenizer."""

        def __init__(self, backend):
            self.base_tokenizer = backend

        def get_vocab(self):
            return self.base_tokenizer.get_vocab()

    def _wrapper(self, backend):
        from tokenizer_analysis.core.tokenizer_wrapper import UniMixLMTokenizer
        return UniMixLMTokenizer("unimix", self._UniMixLike(backend), {})

    def test_all_three_pretokenizer_methods_agree(self, trained_hf_tokenizer):
        wrapper = self._wrapper(trained_hf_tokenizer)
        text = "héllo wörld"
        assert wrapper.can_pretokenize() is True
        surfaces = wrapper.pretokenize(text)
        spans = wrapper.pretokenize_with_spans(text)
        assert spans is not None, (
            "can_pretokenize() said yes, so the spans must be reachable too")
        assert [surface for surface, _span in spans] == surfaces

    def test_a_base_tokenizer_without_one_is_still_without_one(self):
        """Resolving through base_tokenizer must not invent a pre-tokenizer."""
        from tokenizers import Tokenizer
        from tokenizers.models import BPE

        bare = Tokenizer(BPE(unk_token="<unk>"))
        assert bare.pre_tokenizer is None
        wrapper = self._wrapper(bare)
        assert wrapper.can_pretokenize() is False
        assert wrapper.pretokenize_with_spans("hello") is None
        with pytest.raises(NotImplementedError):
            wrapper.pretokenize("hello")


class TestUniMixLMBatchEncodingIsPairedWithTheRightText:
    """UniMixLMTokenizer.encode_batch_with_offsets() must return result i for
    text i, same as every other wrapper's batch path.

    UniMixLM is not one of ALL_WRAPPER_FIXTURES, so
    TestEncodeConsistency.test_batch_ids_match_single_encode -- the test that
    checks exactly this pairing for every other wrapper -- never runs against
    it. Its override always falls back to a per-text loop rather than a
    tokenizers.Tokenizer.encode_batch call (langspec needs the per-text
    scoring; non-langspec does not, but the override does not distinguish
    the two), so this pins that the loop's output stays in the input order.
    If it were ever reversed or otherwise reordered, the ids returned for
    texts[i] would silently belong to a different text -- code and math
    reconstruction fidelity, which read exactly these ids, would then score
    every code and math text against another text's tokens.
    """

    def test_batch_results_are_in_the_same_order_as_the_input_texts(
        self, trained_hf_tokenizer,
    ):
        from tokenizer_analysis.core.tokenizer_wrapper import UniMixLMTokenizer

        wrapper = UniMixLMTokenizer("test-unimix", trained_hf_tokenizer, {})
        texts = [_TEST_TEXT, _TEST_TEXT_MULTI, _CODE_TEXT, _MATH_TEXT]

        batch_results = wrapper.encode_batch_with_offsets(texts)
        assert len(batch_results) == len(texts)

        for i, text in enumerate(texts):
            batch_ids, _batch_offsets = batch_results[i]
            assert batch_ids == wrapper.encode(text), (
                f"encode_batch_with_offsets()[{i}][0] != encode(texts[{i}]) "
                f"for {text!r}; the batch result is paired with the wrong text"
            )
