"""Tests for merge_vocabulary_usage (metrics/merge_usage.py, core/bpe_merges.py).

The fixture tokenizer is built so every case the metric distinguishes occurs
once, with counts written out by hand below:

- ``abc`` has two producing merges, ``ab + c`` (rank 2) and ``a + bc`` (rank 3).
  BPE forms ``ab`` first, so ``ab`` is the intermediate. The July script kept
  the last-listed merge and would have counted ``bc`` instead.
- ``âĤ`` is the first two bytes of the euro sign: a byte fragment.
- ``uvw`` is in the vocabulary through ``uv + w``, but ``v + w`` outranks
  ``u + v``, so with ignore_merges off BPE builds ``u`` + ``vw`` and never
  ``uvw``. With ignore_merges on the word is emitted as ``uvw`` directly.
- ``qq`` is never formed; ``mn`` is also an added token, so it is not a merge
  token at all. ``xyq`` exists so that ``xy`` can be formed inside two
  different emitted tokens.
"""

import copy
import json
import random

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing

from tokenizer_analysis.constants import MERGE_USAGE_MIN_FINAL_TOKENS_WARN
from tokenizer_analysis.core import bpe_merges
from tokenizer_analysis.core.bpe_merges import (
    BPEMergeModel, NotApplicable, _merge_all, merge_model_from_spec,
)
from tokenizer_analysis.core.input_types import CODE_CORPUS, MATH_CORPUS, TokenizedData
from tokenizer_analysis.core.tokenizer_wrapper import HuggingFaceTokenizer
from tokenizer_analysis.metrics.merge_usage import METRIC_KEY, MergeVocabularyUsageMetrics
from tokenizer_analysis.metrics.utf8_integrity import _GPT2_BYTE_TO_UNICODE
from tokenizer_analysis.tests.conftest import MockProvider

EURO = "€"  # bytes E2 82 AC
E1, E2, E3 = (_GPT2_BYTE_TO_UNICODE[b] for b in EURO.encode("utf-8"))
FILLER = "z" * 250_000  # base tokens only: no merge involves 'z' twice

MERGES = [
    ["a", "b"], ["b", "c"], ["ab", "c"], ["a", "bc"],
    ["x", "y"], ["xy", "z"],
    ["q", "q"],
    [E1, E2], [E1 + E2, E3],
    ["v", "w"], ["u", "v"], ["uv", "w"],
    ["m", "n"],
    ["xy", "q"],
]


def _spec(ignore_merges, **model_overrides):
    vocab = {_GPT2_BYTE_TO_UNICODE[b]: b for b in range(256)}
    for left, right in MERGES:
        vocab.setdefault(left + right, len(vocab))
    model = {
        "type": "BPE", "dropout": None, "unk_token": None,
        "continuing_subword_prefix": None, "end_of_word_suffix": None,
        "fuse_unk": False, "byte_fallback": False, "ignore_merges": ignore_merges,
        "vocab": vocab, "merges": [list(m) for m in MERGES],
    }
    model.update(model_overrides)
    byte_level = {"type": "ByteLevel", "add_prefix_space": False,
                  "trim_offsets": False, "use_regex": False}
    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": vocab["mn"], "content": "mn", "single_word": False,
                          "lstrip": False, "rstrip": False, "normalized": False,
                          "special": True}],
        "normalizer": None, "pre_tokenizer": byte_level, "post_processor": None,
        "decoder": byte_level, "model": model,
    }


def _wrapper(ignore_merges, name="fixture", **overrides):
    tok = Tokenizer.from_str(json.dumps(_spec(ignore_merges, **overrides)))
    # A post-processor that prepends "mn". The wrapper encodes with
    # add_special_tokens=False, so the ids every metric reads carry no "mn";
    # an encoding route that passed True would differ, and must be caught.
    tok.post_processor = TemplateProcessing(
        single="mn $A", special_tokens=[("mn", tok.token_to_id("mn"))])
    return HuggingFaceTokenizer(name, tok, {})


# "mn" is the added special token: emitted, but neither a merge token nor part of N.
TEXTS = [FILLER, "abc", "abc", "abc", "xyz", "bc", EURO, "uvw", "mn"]


def _records(wrapper, texts=TEXTS, name="fixture"):
    return [TokenizedData(tokenizer_name=name, language="xx", tokens=wrapper.encode(t), text=t)
            for t in texts]


def _run(wrapper, texts=TEXTS, name="fixture", **kwargs):
    provider = MockProvider(name, wrapper)
    metric = MergeVocabularyUsageMetrics(provider)
    out = metric.compute({name: _records(wrapper, texts, name)}, **kwargs)
    return out[METRIC_KEY]["per_tokenizer"][name]


# Hand-computed, flag off. Final tokens: z x250000, abc x3, xyz, bc, euro, u, vw,
# and the special token mn, which N excludes. N = 250008. Active: final >= 5N/1e6 = 1.25, so 2 or more. Rare: final >= N/1e6,
# so exactly 1. Uncommon: formed but never final.
EXPECTED_COUNTS = {
    "active": 1,      # abc
    "rare": 4,        # xyz, bc, euro, vw
    "uncommon": 3,    # ab (formed 3 times inside abc), xy, the euro fragment
    "unseen": 4,      # qq, uv, uvw, xyq
    "scaffold": 3,    # the three Uncommon tokens, all with survival 0
    "scaffold_byte_fragment": 1,
}


@pytest.mark.parametrize("shipped_flag", [False, True])
def test_hand_computed_counts(shipped_flag):
    entry = _run(_wrapper(shipped_flag))
    g = entry["global"]
    assert g["counts"] == EXPECTED_COUNTS
    assert g["n_merge_tokens"] == 12
    assert g["total_final_tokens"] == 250_008
    assert g["n_unreachable_without_ignore_merges"] == 1
    assert entry["n_tokens_with_multiple_merges"] == 1
    assert entry["n_excluded_special_tokens"] == 1
    assert entry["final_tokens_per_language"] == {"prose": {"xx": 250_008}}
    assert entry["final_tokens_per_corpus"] == {"prose": 250_008}
    assert entry["texts_per_corpus"] == {"prose": len(TEXTS)}
    if shipped_flag:
        # Only "uvw" encodes differently: [uvw] shipped, [u, vw] with the flag off.
        assert entry["encoding"] == "reencoded_ignore_merges_false"
        assert entry["texts_differing_after_reencode"] == 1
    else:
        assert entry["encoding"] == "shared"
        assert entry["texts_differing_after_reencode"] is None


def test_intermediate_follows_bpe_order_not_the_last_listed_merge():
    entry = _run(_wrapper(False))
    examples = entry["examples_scaffold"]
    # Ordered by times formed: ab inside each of the three abc, xy inside xyz.
    # bc is emitted once on its own and never formed inside abc, so it is Rare
    # with survival 1 and not an example.
    assert [e["token"] for e in examples] == ["ab", "xy"]
    assert examples[0]["formed"] == 3 and examples[0]["final"] == 0
    assert [e["formed_inside_most"] for e in examples] == ["abc", "xyz"]


# Exactly one million final tokens, so the thresholds are whole numbers of
# emissions: Active from 5, Rare from 1. Each threshold is hit exactly once.
#   abc x9, ab alone x1: ab final 1 (Rare, on the boundary), formed 10,
#       survival exactly 0.1, so not Scaffold. abc final 9: Active.
#   xyz x10, xyq x2, xy alone x1: xy final 1 (Rare), formed 13, survival 1/13:
#       the one Rare Scaffold token, formed inside xyz more often than xyq.
#       xyz final 10: Active. xyq final 2: Rare.
#   bc x5: final 5, exactly the Active threshold.
BOUNDARY_TEXTS = (["z" * (1_000_000 - 28)] + ["abc"] * 9 + ["ab"] + ["xyz"] * 10
                  + ["xyq"] * 2 + ["xy"] + ["bc"] * 5)


def test_every_threshold_is_inclusive_where_defined():
    entry = _run(_wrapper(False), texts=BOUNDARY_TEXTS)
    g = entry["global"]
    assert g["total_final_tokens"] == 1_000_000
    assert g["counts"] == {
        "active": 3,      # abc, xyz, bc (bc exactly 5 per million)
        "rare": 3,        # ab, xy (each exactly 1 per million), xyq
        "uncommon": 0,
        "unseen": 6,      # qq, the euro fragment, euro, vw, uv, uvw
        "scaffold": 1,    # xy; ab sits exactly on survival 0.1 and is excluded
        "scaffold_byte_fragment": 0,
    }
    # Rates of the six formed tokens, per million: 1, 1, 2, 5, 9, 10. The upper
    # of the two middle values, as the research script took it.
    assert g["median_standalone_per_million"] == 5.0
    [xy] = entry["examples_scaffold"]
    assert xy["token"] == "xy" and xy["formed"] == 13
    assert xy["formed_inside_most"] == "xyz"


def test_byte_fragment_is_found_from_bytes():
    entry = _run(_wrapper(False))
    fragments = entry["examples_scaffold_byte_fragment"]
    assert [e["token"] for e in fragments] == [E1 + E2]


def test_small_corpus_warns_and_threshold_does_not():
    wrapper = _wrapper(False)
    below = _run(wrapper, texts=["z" * (MERGE_USAGE_MIN_FINAL_TOKENS_WARN - 1)])
    at = _run(wrapper, texts=["z" * MERGE_USAGE_MIN_FINAL_TOKENS_WARN])
    assert len(below["warnings"]) == 1
    assert at["warnings"] == []


# ----------------------------------------------------------------------
# Replay against the Rust BPE
# ----------------------------------------------------------------------

def _random_bpe(rng):
    """A random merge list over 'abcd' with tokens built several ways and
    ranks shuffled, so ranks are often inconsistent with build order."""
    vocab = {c: i for i, c in enumerate("abcd")}
    merges = []
    pairs = set()
    tokens = list("abcd")
    while len(merges) < 40:
        left, right = rng.choice(tokens), rng.choice(tokens)
        if len(left + right) > 5 or (left, right) in pairs:
            continue
        pairs.add((left, right))
        merges.append((left, right))
        if left + right not in vocab:
            vocab[left + right] = len(vocab)
            tokens.append(left + right)
    rng.shuffle(merges)
    return vocab, merges


def test_replay_matches_rust_and_per_token_replay_matches_whole_word():
    rng = random.Random(0)
    n_multi = 0
    for _ in range(30):
        vocab, merges = _random_bpe(rng)
        rust = BPE(vocab=vocab, merges=merges)
        ranks = {(vocab[a], vocab[b]): (r, vocab[a + b]) for r, (a, b) in enumerate(merges)}
        producers = {}
        for a, b in merges:
            producers[a + b] = producers.get(a + b, 0) + 1
        n_multi += sum(1 for v in producers.values() if v > 1)
        model = BPEMergeModel(vocab=vocab, ranks=ranks, merge_token_ids=frozenset(),
                              added_token_ids=frozenset(), n_tokens_with_multiple_merges=0,
                              shipped_ignore_merges=False, byte_level=False, backend=None,
                              encode_add_special_tokens=False)
        id_to_tok = {i: t for t, i in vocab.items()}
        for _ in range(200):
            word = "".join(rng.choice("abcd") for _ in range(rng.randint(1, 12)))
            whole = _merge_all([vocab[c] for c in word], ranks)
            assert [id_to_tok[i] for i in whole.symbols] == [t.value for t in rust.tokenize(word)]
            per_token = []
            for tid in whole.symbols:
                alone = model.replay(id_to_tok[tid])
                assert alone.symbols == (tid,)
                per_token.extend(alone.created)
            assert sorted(per_token) == sorted(whole.created)
    # The generator has to produce the case the replay exists for.
    assert n_multi > 0


def test_heap_order_differs_from_merging_every_occurrence():
    # (ab, a) at rank 0 and (a, b) at rank 2: Rust gives ['aba', 'b'].
    vocab = {"a": 0, "b": 1, "ab": 2, "aba": 3}
    merges = [("ab", "a"), ("a", "b")]
    ranks = {(2, 0): (0, 3), (0, 1): (1, 2)}
    result = _merge_all([0, 1, 0, 1], ranks)
    rust = [t.value for t in BPE(vocab=vocab, merges=merges).tokenize("abab")]
    assert rust == ["aba", "b"]
    assert [ {0: "a", 1: "b", 2: "ab", 3: "aba"}[i] for i in result.symbols] == rust


# ----------------------------------------------------------------------
# Loud failures
# ----------------------------------------------------------------------

def test_duplicate_pair_raises():
    spec = _spec(False)
    spec["model"]["merges"].append(["a", "b"])
    with pytest.raises(ValueError, match="listed twice"):
        merge_model_from_spec(spec, None, "fixture", False)


def test_merge_part_missing_raises():
    spec = _spec(False)
    spec["model"]["merges"].append(["a", "nonexistent"])
    with pytest.raises(ValueError, match="does not contain"):
        merge_model_from_spec(spec, None, "fixture", False)


def test_missing_added_tokens_list_raises():
    spec = _spec(False)
    del spec["added_tokens"]
    with pytest.raises(ValueError, match="added_tokens"):
        merge_model_from_spec(spec, None, "fixture", False)


def test_ids_the_backend_does_not_reproduce_raise():
    wrapper = _wrapper(False)
    records = _records(wrapper)
    records[1] = TokenizedData(tokenizer_name="fixture", language="xx",
                               tokens=list(reversed(records[1].tokens)) + [0], text="abc")
    metric = MergeVocabularyUsageMetrics(MockProvider("fixture", wrapper))
    with pytest.raises(ValueError, match="does not reproduce"):
        metric.compute({"fixture": records})


@pytest.mark.parametrize("texts", [[], ["mn"]], ids=["no texts", "special tokens only"])
def test_corpus_with_no_final_tokens_raises(texts):
    with pytest.raises(ValueError, match="no final tokens"):
        _run(_wrapper(False), texts=texts)


def test_record_without_text_raises():
    wrapper = _wrapper(True)
    records = _records(wrapper)
    records[2] = TokenizedData(tokenizer_name="fixture", language="xx",
                               tokens=records[2].tokens, text=None)
    metric = MergeVocabularyUsageMetrics(MockProvider("fixture", wrapper))
    with pytest.raises(ValueError, match="carries no text"):
        metric.compute({"fixture": records})


def test_replay_that_does_not_build_an_emitted_token_raises(monkeypatch):
    original = BPEMergeModel.replay

    def broken(self, token_string):
        result = original(self, token_string)
        if token_string == "abc":
            return bpe_merges.ReplayResult(result.symbols[:0] + (self.vocab["a"],), result.created)
        return result

    monkeypatch.setattr(BPEMergeModel, "replay", broken)
    with pytest.raises(ValueError, match="does not build it"):
        _run(_wrapper(False))


def _no_byte_level_component(spec):
    spec["pre_tokenizer"] = {"type": "WhitespaceSplit"}
    spec["decoder"] = None
    return HuggingFaceTokenizer("fixture", Tokenizer.from_str(json.dumps(spec)), {})


def test_byte_vocabulary_without_a_byte_level_component_raises():
    # Every single-character entry is a GPT-2 byte character: byte-level in all
    # but declaration.
    with pytest.raises(ValueError, match="GPT-2 byte character"):
        _run(_no_byte_level_component(_spec(False)), texts=["abc"])


def test_character_vocabulary_with_marker_letters_is_character_level():
    # gemma-2's case: Latin Extended-A letters such as \u0105 are GPT-2 marker
    # characters, but a character vocabulary also holds characters outside the
    # byte table.
    spec = _spec(False)
    spec["model"]["vocab"]["\u4e2d"] = len(spec["model"]["vocab"])
    entry = _run(_no_byte_level_component(spec), texts=["abc"])
    assert entry["applicable"] is True
    assert entry["byte_fragment_detection"].startswith("character vocabulary")


# ----------------------------------------------------------------------
# Not applicable
# ----------------------------------------------------------------------

def test_unigram_is_not_applicable():
    from tokenizers.models import Unigram
    wrapper = HuggingFaceTokenizer("uni", Tokenizer(Unigram([("a", -1.0), ("b", -1.0)], 0)), {})
    entry = _run(wrapper, texts=["ab"], name="uni")
    assert entry == {"applicable": False, "reason": "model type is Unigram, not BPE", "global": None}


def test_slow_tokenizer_is_not_applicable():
    class Slow:
        def encode(self, text, add_special_tokens=False):
            return [0]

        def get_vocab(self):
            return {"a": 0}

    wrapper = HuggingFaceTokenizer("slow", Slow(), {})
    entry = _run(wrapper, texts=["a"], name="slow")
    assert entry["applicable"] is False and "no tokenizers backend" in entry["reason"]


def test_non_empty_affix_is_not_applicable_and_empty_affix_is_applicable():
    # From the spec: the Rust loader cannot load this fixture's unprefixed
    # merges under a '##' prefix, and the decision is made before any loading.
    with_prefix = merge_model_from_spec(
        _spec(False, continuing_subword_prefix="##"), None, "fixture", False)
    assert isinstance(with_prefix, NotApplicable)
    assert "continuing_subword_prefix" in with_prefix.reason
    # o200k stores both affixes as "".
    empty = _run(_wrapper(False, continuing_subword_prefix="", end_of_word_suffix=""))
    assert empty["applicable"] is True


def test_subclass_does_not_inherit_the_merge_list():
    class Sub(HuggingFaceTokenizer):
        pass

    wrapper = Sub("sub", Tokenizer.from_str(json.dumps(_spec(False))), {})
    assert isinstance(wrapper.get_bpe_merge_model(), NotApplicable)


def test_pretokenized_provider_is_not_applicable():
    from tokenizer_analysis.core.input_providers import PreTokenizedProvider

    provider = PreTokenizedProvider.__new__(PreTokenizedProvider)
    provider.specifications = {}
    metric = MergeVocabularyUsageMetrics.__new__(MergeVocabularyUsageMetrics)
    metric.input_provider = provider
    entry = metric._compute_one("fixture", {"prose": []})
    assert entry["applicable"] is False and "pre-tokenized" in entry["reason"]


# ----------------------------------------------------------------------
# Code and math corpora
# ----------------------------------------------------------------------

def test_include_code_math_counts_the_bundled_samples_when_nothing_is_registered():
    wrapper = _wrapper(False)
    provider = MockProvider("fixture", wrapper)
    metric = MergeVocabularyUsageMetrics(provider)
    without = metric.compute({"fixture": _records(wrapper)})[METRIC_KEY]
    with_cm = metric.compute({"fixture": _records(wrapper)}, include_code_math=True)[METRIC_KEY]
    entry = with_cm["per_tokenizer"]["fixture"]
    assert set(entry["final_tokens_per_language"]) == {"prose", CODE_CORPUS, MATH_CORPUS}
    assert (entry["global"]["total_final_tokens"]
            > without["per_tokenizer"]["fixture"]["global"]["total_final_tokens"])
    sources = with_cm["metadata"]["code_math_sources"]
    assert sources[CODE_CORPUS]["synthetic"] is True
    assert sources[MATH_CORPUS]["synthetic"] is True
    assert "code_math_sources" not in without["metadata"]
