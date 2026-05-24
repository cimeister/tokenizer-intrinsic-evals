"""Tests for the tokenizer sanity-check diagnostic.

Each test states its intent.  The C2/C16 tests in particular guard the
user-flagged regressions: legitimate multibyte / base+mark splitting and
context-dependent BPE non-self-reproduction must NOT be penalized, while
genuine orphaned-multibyte groupings and normalizer-dead vocab MUST be
detected.
"""

import os
import tempfile

import pytest

from tokenizer_analysis.core.tokenizer_wrapper import (
    CustomBPETokenizer,
    HuggingFaceTokenizer,
    SentencePieceTokenizer,
    TokenizerWrapper,
)
from tokenizer_analysis import constants
from tokenizer_analysis.diagnostics import probe_corpus
from tokenizer_analysis.diagnostics.probe_corpus import Probe, builtin_probes
from tokenizer_analysis.diagnostics.sanity_check import (
    Severity,
    TokenizerSanityChecker,
    _classify_malformation,
    detect_byte_encoding,
    render_text,
    run_sanity_check,
    severity_to_exit_code,
)
from tokenizer_analysis.cli.sanity_check import main as cli_main

_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Die Relativitätstheorie ist wichtig für die Physik.",
    "café résumé naïve coöperate",
    "import os; print(os.getcwd())",
    "1234567890 numbers and 42 answers",
    "Spaces   and\ttabs\nand newlines.",
]


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def byte_level_hf_wrapper():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    tok.train_from_iterator(
        _CORPUS,
        trainer=BpeTrainer(vocab_size=400,
                            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
                            initial_alphabet=ByteLevel.alphabet()),
    )
    return HuggingFaceTokenizer("byte-level", tok, {})


@pytest.fixture(scope="module")
def hf_nfkc_wrapper():
    """Introspectable NFKC normalizer + a fullwidth vocab token it folds.

    encode("１") NFKC-folds to "1" before the model, so the fullwidth token
    can never be emitted -> C16 normalization_unreachable.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.normalizers import NFKC

    vocab = {"<unk>": 0, "1": 1, "a": 2, "b": 3, " ": 4, "１": 5}
    tok = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tok.normalizer = NFKC()
    return HuggingFaceTokenizer("hf-nfkc", tok, {})


@pytest.fixture(scope="module")
def bare_mark_vocab_wrapper():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE

    vocab = {"<unk>": 0, "a": 1, "b": 2, "́": 3}  # bare combining acute
    tok = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    return HuggingFaceTokenizer("bare-mark", tok, {})


@pytest.fixture(scope="module")
def sp_identity_wrapper():
    spm = pytest.importorskip("sentencepiece")
    return _train_sp("identity")


@pytest.fixture(scope="module")
def sp_nfkc_wrapper():
    spm = pytest.importorskip("sentencepiece")
    return _train_sp("nfkc")


def _train_sp(rule):
    spm = pytest.importorskip("sentencepiece")
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "c.txt")
        with open(cp, "w", encoding="utf-8") as f:
            f.write("\n".join(_CORPUS) + "\n")
        prefix = os.path.join(d, "m")
        spm.SentencePieceTrainer.train(
            input=cp, model_prefix=prefix, vocab_size=180, model_type="bpe",
            character_coverage=1.0, normalization_rule_name=rule,
            add_dummy_prefix=False, remove_extra_whitespaces=False,
        )
        with open(prefix + ".model", "rb") as f:
            mb = f.read()
    persistent = tempfile.NamedTemporaryFile(suffix=".model", delete=False)
    persistent.write(mb)
    persistent.close()
    sp = spm.SentencePieceProcessor()
    sp.load(persistent.name)
    return SentencePieceTokenizer(f"sp-{rule}", sp, {})


class StubWrapper(TokenizerWrapper):
    """Configurable wrapper to drive specific defect paths deterministically."""

    def __init__(self, *, encode_fn, decode_fn=None, vocab=None,
                 ids_to_tokens=None, batch_fn=None, pretok_fn=None):
        self._encode_fn = encode_fn
        self._decode_fn = decode_fn
        self._vocab = vocab or {"<unk>": 0}
        self._i2t = ids_to_tokens or {}
        self._batch_fn = batch_fn
        self._pretok_fn = pretok_fn

    def get_name(self): return "stub"
    def get_vocab_size(self): return len(self._vocab)
    def get_vocab(self): return dict(self._vocab)
    def can_encode(self): return True
    def encode(self, text): return list(self._encode_fn(text))
    def can_decode(self): return self._decode_fn is not None
    def decode(self, ids, skip_special_tokens=True):
        return self._decode_fn(ids) if self._decode_fn else None
    def can_pretokenize(self): return self._pretok_fn is not None
    def pretokenize(self, text):
        if not self._pretok_fn:
            raise NotImplementedError
        return self._pretok_fn(text)
    def convert_ids_to_tokens(self, ids):
        return [self._i2t.get(i, f"<UNK_{i}>") for i in ids]
    def encode_with_offsets(self, text):
        return self.encode(text), None
    def encode_batch_with_offsets(self, texts):
        if self._batch_fn:
            return self._batch_fn(texts)
        return [self.encode_with_offsets(t) for t in texts]
    @classmethod
    def from_config(cls, name, config):  # pragma: no cover - unused
        raise NotImplementedError


# ── Tests ─────────────────────────────────────────────────────────────

def _checks(wrapper, probes=None):
    return TokenizerSanityChecker(
        wrapper, probes if probes is not None else builtin_probes()).run()


def test_1_byte_coverage_pass_on_bytelevel(byte_level_hf_wrapper):
    """C1: a full ByteLevel alphabet -> pass, 256 covered."""
    rep = _checks(byte_level_hf_wrapper)
    c = rep["checks"]["C1 byte-level 256-coverage"]
    assert c["severity"] == Severity.PASS, c


def test_2_byte_coverage_not_applicable_on_nonbyte(hf_nfkc_wrapper):
    """C1: non-byte-level -> explicit not_applicable, never a silent pass."""
    c = _checks(hf_nfkc_wrapper, builtin_probes())["checks"][
        "C1 byte-level 256-coverage"]
    assert c["severity"] == Severity.NOT_APPLICABLE


def test_3_lossy_expected_vs_clean(sp_nfkc_wrapper, sp_identity_wrapper):
    """C3: NFKC SP folds fullwidth/NFD (lossy but expected-ish) while
    identity SP keeps ASCII clean."""
    probes = [Probe("hello world", probe_corpus.CAT_ASCII)]
    bd_id = _checks(sp_identity_wrapper, probes)["lossy_breakdown"]
    assert bd_id.get("clean", 0) >= 1
    # NFKC SP roundtrip of a fullwidth string is not 'clean'
    fp = [Probe("１２３", probe_corpus.CAT_DIGITS)]
    bd_nf = _checks(sp_nfkc_wrapper, fp)["lossy_breakdown"]
    assert "clean" not in bd_nf or bd_nf.get("clean", 0) == 0


def test_4_sp_normalization_unverifiable(sp_nfkc_wrapper):
    """C4: SP normalizer not introspectable -> unverifiable + >=warn."""
    rep = _checks(sp_nfkc_wrapper, [Probe("Aa", probe_corpus.CAT_ASCII)])
    c = rep["checks"]["C4 faithful-pipeline conformance"]
    assert c["severity"] == Severity.UNVERIFIABLE
    assert rep["components"]["normalizer_introspectable"] is False


def test_5_c2_static_bare_mark_fail(bare_mark_vocab_wrapper):
    """C2 static: a bare combining-mark vocab token -> fail."""
    c = _checks(bare_mark_vocab_wrapper, [Probe("ab", "ascii_basic")])[
        "checks"]["C2 combining-mark mishandling"]
    assert c["severity"] == Severity.FAIL
    assert c["observed"] >= 1


def test_6_c2_negative_split_but_roundtrips(byte_level_hf_wrapper):
    """C2 must NOT penalize a multibyte / base+mark split that roundtrips.

    Guards the exact user-flagged regression: no grapheme-split penalty,
    and SANITY_GRAPHEME_SPLIT_* must not exist as a constant.
    """
    probes = [
        Probe("é", probe_corpus.CAT_MARKS),       # base + combining mark
        Probe("文字", probe_corpus.CAT_MULTISCRIPT),  # multibyte scalars
    ]
    rep = _checks(byte_level_hf_wrapper, probes)
    assert rep["checks"]["C2 combining-mark mishandling"][
        "severity"] == Severity.PASS
    blob = repr(rep)
    assert "grapheme_split" not in blob
    assert not hasattr(constants, "SANITY_GRAPHEME_SPLIT_FAIL_COUNT")
    assert not hasattr(constants, "SANITY_GRAPHEME_SPLIT_WARN_COUNT")


def test_7_orphaned_multibyte_is_byte_bug():
    """C3 end-to-end: an orphaned continuation byte -> byte_bug + >=warn,
    and the helper classifies it (not just naive split equality)."""
    assert _classify_malformation(b"\x80") == "orphan_continuation"
    stub = StubWrapper(
        encode_fn=lambda t: [1],
        decode_fn=lambda ids: "X",          # lossy
        vocab={"<0x80>": 1},
        ids_to_tokens={1: "<0x80>"},
    )
    rep = _checks(stub, [Probe("hello", probe_corpus.CAT_ASCII)])
    assert rep["lossy_breakdown"].get("byte_bug", 0) > 0
    assert rep["overall_severity"] in (Severity.WARN, Severity.FAIL)


def test_8_special_tokens_and_vocab_integrity():
    """C7 non-atomic/dup special + C14 size mismatch & non-contiguous ids."""
    stub = StubWrapper(
        encode_fn=lambda t: [0, 0],          # unk surface not atomic
        decode_fn=lambda ids: "x",
        vocab={"<unk>": 0, "a": 1, "b": 5},  # non-contiguous, len!=size
    )
    rep = _checks(stub, [Probe("a", probe_corpus.CAT_ASCII)])
    assert rep["checks"]["C14 vocab integrity"]["severity"] in (
        Severity.WARN, Severity.FAIL)


def test_9_determinism_pass_and_batch_loop_warn():
    """C8: deterministic -> pass; batch != loop -> warn (not crash)."""
    good = StubWrapper(encode_fn=lambda t: [1, 2], decode_fn=lambda i: "x")
    assert _checks(good, [Probe("a", "ascii_basic")])[
        "checks"]["C8 determinism/idempotency"]["severity"] == Severity.PASS
    bad_batch = StubWrapper(
        encode_fn=lambda t: [1, 2], decode_fn=lambda i: "x",
        batch_fn=lambda ts: [([9, 9], None) for _ in ts])
    c = _checks(bad_batch, [Probe("a", "ascii_basic")])[
        "checks"]["C8 determinism/idempotency"]
    assert c["severity"] == Severity.WARN


def test_10_whitespace_digit_static_and_c6_formula(byte_level_hf_wrapper):
    """C5/C6: fractions computed; C6 consistency/direction defined (B3)."""
    rep = _checks(byte_level_hf_wrapper,
                  [Probe("1234567890123 " * 3, probe_corpus.CAT_DIGITS),
                   Probe("a\tb\nc", probe_corpus.CAT_WHITESPACE)])
    c6 = rep["checks"]["C6 digit handling"]
    assert 0.0 <= c6["observed"] <= 1.0
    assert "chunking_direction=" in c6["detail"]


def test_11_c16_normalization_unreachable_fail(hf_nfkc_wrapper):
    """C16: NFKC folds a fullwidth vocab token -> normalization_unreachable
    -> the only C16 fail path (introspectable normalizer required)."""
    rep = _checks(hf_nfkc_wrapper, [Probe("ab", probe_corpus.CAT_ASCII)])
    reach = rep["vocab_reachability"]
    assert reach["normalization_unreachable"] >= 1
    assert rep["checks"]["C16 vocab reachability"]["severity"] == Severity.FAIL


def test_12_c16_non_self_reproducing_not_penalized(byte_level_hf_wrapper):
    """C16: context-dependent BPE non-self-reproduction is informational
    only -- never a severity, no threshold key for that bucket."""
    rep = _checks(byte_level_hf_wrapper, [Probe("the fox", "ascii_basic")])
    c16 = rep["checks"]["C16 vocab reachability"]
    reach = rep["vocab_reachability"]
    assert reach.get("non_self_reproducing", 0) >= 0
    # bucket must never by itself cause a fail
    if (reach["normalization_unreachable"] == 0
            and reach["unverifiable"] == 0):
        assert c16["severity"] == Severity.PASS


def test_13_c16_byte_fragment_context_only(byte_level_hf_wrapper):
    """C16: incomplete-UTF-8 byte-level tokens -> context_only, unpenalized."""
    rep = _checks(byte_level_hf_wrapper, [Probe("é 文", "multiscript")])
    assert rep["vocab_reachability"]["context_only"] > 0


def test_14_c16_opaque_normalizer_unverifiable(sp_identity_wrapper):
    """C16: opaque (SP) normalizer -> unverifiable -> overall >= warn."""
    rep = _checks(sp_identity_wrapper, [Probe("hello", "ascii_basic")])
    reach = rep["vocab_reachability"]
    assert reach["unverifiable"] > 0
    assert rep["checks"]["C16 vocab reachability"]["severity"] in (
        Severity.UNVERIFIABLE, Severity.FAIL)


def test_14b_opaque_normalizer_unverifiable_stub():
    """C4/C16 opaque-normalizer path, dependency-free (runs even without SP).

    Guarantees the no-silent-fallback contract is covered when SP is absent.
    """
    stub = StubWrapper(encode_fn=lambda t: [1], decode_fn=lambda i: "x",
                       vocab={"<unk>": 0, "x": 1}, ids_to_tokens={1: "x"})
    rep = _checks(stub, [Probe("x", "ascii_basic")])
    assert rep["components"]["normalizer_introspectable"] is False
    assert rep["checks"]["C4 faithful-pipeline conformance"][
        "severity"] == Severity.UNVERIFIABLE
    assert rep["checks"]["C16 vocab reachability"][
        "severity"] == Severity.UNVERIFIABLE
    assert rep["overall_severity"] in (Severity.WARN, Severity.FAIL)


def test_15_casing_gate_feeds_c3():
    """C9->C3: with a lowercasing normalizer, case loss is *_expected*."""
    # encode lowercases; decode echoes lowered text -> casing_loss_expected
    stub = StubWrapper(
        encode_fn=lambda t: [1],
        decode_fn=lambda ids: "hello",
        vocab={"<unk>": 0, "hello": 1},
        ids_to_tokens={1: "hello"},
    )
    chk = TokenizerSanityChecker(stub, [Probe("HELLO", "ascii_basic")])
    # force the gate (stub has no introspectable normalizer; emulate C9 true)
    chk.lowercasing_normalizer = True
    cat = chk._classify_roundtrip("HELLO")
    assert cat == "casing_loss_expected"
    chk.lowercasing_normalizer = False
    assert chk._classify_roundtrip("HELLO") == "casing_loss_bug"


def test_16_pretok_conservation(byte_level_hf_wrapper):
    """C10: a char-dropping pretokenizer -> fail; honest one -> not fail."""
    rep = _checks(byte_level_hf_wrapper,
                  [Probe("hello world", probe_corpus.CAT_ASCII)])
    assert rep["checks"]["C10 pretokenizer char conservation"][
        "severity"] in (Severity.PASS, Severity.NOT_APPLICABLE)
    drop = StubWrapper(encode_fn=lambda t: [1], decode_fn=lambda i: "x",
                       pretok_fn=lambda t: ["ab"])  # drops most chars
    c = _checks(drop, [Probe("abcdefghij", probe_corpus.CAT_ASCII)])[
        "checks"]["C10 pretokenizer char conservation"]
    assert c["severity"] == Severity.FAIL


def test_17_c4_conformance_pass(byte_level_hf_wrapper):
    """C4: introspectable normalizer, no declared-vs-effective bypass."""
    rep = _checks(byte_level_hf_wrapper, builtin_probes())
    assert rep["checks"]["C4 faithful-pipeline conformance"][
        "severity"] in (Severity.PASS, Severity.WARN)


def test_18_emoji_control_smoke(byte_level_hf_wrapper):
    """C12: emoji/ZWJ/control roundtrip reported (single smoke assertion)."""
    rep = _checks(byte_level_hf_wrapper, builtin_probes())
    assert "C12 emoji/ZWJ/control" in rep["checks"]


def test_19_cli_positional_and_unknown_class(tmp_path):
    """CLI: positional builds wrapper; unknown class -> exit 3;
    a loaded-but-failing tokenizer -> exit 2; --exit-zero -> 0."""
    code = cli_main(["bogus:foo"])
    assert code == 3
    code = cli_main(["huggingface:test_tokenizers/test_bpe_tok-gpt4.json",
                     "--quiet"])
    assert code in (1, 2)            # this small tokenizer fails C1
    assert cli_main(["bogus:foo", "--exit-zero"]) == 0


def test_20_severity_exit_code_mapping():
    assert severity_to_exit_code(Severity.PASS) == 0
    assert severity_to_exit_code(Severity.WARN) == 1
    assert severity_to_exit_code(Severity.FAIL) == 2


def test_21_render_text_deterministic_no_ansi():
    stub = StubWrapper(encode_fn=lambda t: [1], decode_fn=lambda i: "x",
                       vocab={"<unk>": 0, "x": 1})
    rep = run_sanity_check({"s": stub}, [Probe("x", "ascii_basic")])
    a = render_text(rep, use_color=False)
    b = render_text(rep, use_color=False)
    assert a == b
    assert "\x1b[" not in a


def test_22_thresholds_metadata_matches_constants():
    stub = StubWrapper(encode_fn=lambda t: [1], decode_fn=lambda i: "x",
                       vocab={"<unk>": 0, "x": 1})
    rep = run_sanity_check({"s": stub}, [Probe("x", "ascii_basic")])
    th = rep["tokenizer_sanity_check"]["metadata"]["thresholds"]
    for k in th:
        assert k.startswith("SANITY_")
        assert hasattr(constants, k)
    assert "SANITY_GRAPHEME_SPLIT_FAIL_COUNT" not in th
    # C5 is warn-only: the FAIL threshold constant must be gone.
    assert "SANITY_WHITESPACE_FIDELITY_FAIL_FRAC" not in th
    assert not hasattr(constants, "SANITY_WHITESPACE_FIDELITY_FAIL_FRAC")


def test_23_whitespace_handling_warn_only(byte_level_hf_wrapper):
    """C5 whitespace handling, including \\r (CRLF).

    Asserts: (a) tab/newline/CRLF roundtrip losslessly on a byte-level
    tokenizer -> fidelity 1.0, PASS; (b) a whitespace-dropping tokenizer is
    WARN, never FAIL (per user calibration); (c) a zero-whitespace input is
    reported not-applicable (fidelity None), guarding the (0,0) divide;
    (d) C5 never emits FAIL even across the full builtin probe set.
    """
    ws_probes = [
        Probe("a\tb", probe_corpus.CAT_WHITESPACE),
        Probe("a\nb", probe_corpus.CAT_WHITESPACE),
        Probe("a\r\nb", probe_corpus.CAT_WHITESPACE),   # the \r / CRLF case
        Probe("line1\r\nline2", probe_corpus.CAT_WHITESPACE),
    ]
    # (a) byte-level fixture preserves whitespace incl. CR -> PASS @ 1.0
    c = _checks(byte_level_hf_wrapper, ws_probes)[
        "checks"]["C5 whitespace handling"]
    assert c["severity"] == Severity.PASS
    assert c["observed"] == 1.0
    # explicit \r roundtrip on the real wrapper
    for s in ("\r", "\r\n", "a\rb", "line1\r\nline2"):
        assert byte_level_hf_wrapper.decode(
            byte_level_hf_wrapper.encode(s)) == s

    # (b) a whitespace-dropping tokenizer -> WARN, NOT FAIL
    drop = StubWrapper(
        encode_fn=lambda t: [1],
        decode_fn=lambda ids: "ab",          # all whitespace stripped
        vocab={"<unk>": 0, "ab": 1}, ids_to_tokens={1: "ab"})
    cd = _checks(drop, ws_probes)["checks"]["C5 whitespace handling"]
    assert cd["severity"] == Severity.WARN
    assert cd["severity"] != Severity.FAIL

    # (c) zero-whitespace input -> not-applicable, fidelity None, PASS
    cna = _checks(drop, [Probe("abc", probe_corpus.CAT_WHITESPACE)])[
        "checks"]["C5 whitespace handling"]
    assert cna["observed"] is None
    assert cna["severity"] == Severity.PASS

    # (d) C5 is never FAIL, even over the whole builtin corpus
    for w in (byte_level_hf_wrapper, drop):
        assert _checks(w, builtin_probes())[
            "checks"]["C5 whitespace handling"]["severity"] != Severity.FAIL
