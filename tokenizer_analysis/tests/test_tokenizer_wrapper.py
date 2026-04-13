"""Tests for tokenizer wrapper consistency across wrapper types.

Verifies that HuggingFaceTokenizer, SentencePieceTokenizer, and
CustomBPETokenizer behave consistently for the operations that
intrinsic metrics depend on: encode, encode_with_offsets, decode,
convert_ids_to_tokens, get_vocab, and get_vocab_size.
"""

import os
import pytest
import tempfile

from tokenizer_analysis.core.tokenizer_wrapper import (
    HuggingFaceTokenizer,
    SentencePieceTokenizer,
    CustomBPETokenizer,
)

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
    # first token — this ensures decode(encode(text)) == text exactly,
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


ALL_WRAPPER_FIXTURES = ["hf_wrapper", "sp_wrapper", "custom_bpe_wrapper"]


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
                f"Token ID {ids[i]} from encode() mapped to {tok_str!r} — "
                f"all IDs from encode() should be in the vocabulary"
            )


class TestDecodeRoundtrip:
    """decode(encode(text)) must exactly recover the original text.

    The test fixtures are deliberately configured to avoid any
    preprocessing that would alter the text before encoding:
    - HF ByteLevel: add_prefix_space=False (no leading space injection)
    - SentencePiece: normalization_rule_name="identity", add_dummy_prefix=False,
      remove_extra_whitespaces=False (no NFKC normalization, no dummy prefix)

    This ensures the roundtrip test is strict — any mismatch is a real bug,
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
    """get_vocab_size() should equal len(get_vocab())."""

    @pytest.mark.parametrize("wrapper_name", ALL_WRAPPER_FIXTURES)
    def test_size_matches_dict(self, wrapper_name, request):
        wrapper = request.getfixturevalue(wrapper_name)
        vocab = wrapper.get_vocab()
        if vocab is not None:
            assert wrapper.get_vocab_size() == len(vocab), (
                f"{wrapper.get_name()}: get_vocab_size()={wrapper.get_vocab_size()} "
                f"but len(get_vocab())={len(vocab)}"
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
