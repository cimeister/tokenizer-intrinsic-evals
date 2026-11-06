"""BPE merge lists read from a ``tokenizers`` backend, and merge replay.

Used by ``metrics/merge_usage.py``. A tokenizer either yields a
:class:`BPEMergeModel` or a :class:`NotApplicable` that says why it cannot.

The replay reproduces ``Word::merge_all`` in the Rust ``tokenizers`` BPE
(version 0.22, no dropout): a heap ordered by (rank, position), so the
lowest-ranked pair is merged first and, among equal ranks, the leftmost. A
queued pair whose symbols changed before it was popped is skipped. This is not
the same as "merge every occurrence of the best pair at once": with
inconsistent ranks, ``(ab, a)`` at rank 0 and ``(a, b)`` at rank 2, the word
``abab`` becomes ``['aba', 'b']`` under the heap and ``['ab', 'ab']`` under the
other rule.

Measured on Apertus, llama3 and o200k (about 1.43M pre-tokens, 2026-11-06):
this replay reproduces Rust ``model.tokenize`` with ``ignore_merges=False`` on
every pre-token, and the intermediate symbols formed while merging a whole
pre-token equal those formed by replaying each emitted token's string alone.
"""

import heapq
import json
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

# Names of the tokenizers components that apply the GPT-2 byte-to-character
# remapping. A vocabulary built under any of them spells bytes, not characters.
_BYTE_LEVEL_TYPE = "ByteLevel"


@dataclass(frozen=True)
class NotApplicable:
    """Why a tokenizer has no merge list this package can replay."""

    reason: str


@dataclass(frozen=True)
class ReplayResult:
    """The outcome of running BPE on one string.

    ``symbols`` are the ids left at the end; a string that is a reachable merge
    token ends as exactly one symbol, its own id. ``created`` lists every id a
    merge produced, in order and with repeats, including the last one.
    """

    symbols: Tuple[int, ...]
    created: Tuple[int, ...]


@dataclass
class BPEMergeModel:
    """The merge list of one BPE tokenizer, as ids.

    ``ranks`` maps a (left id, right id) pair to (rank, merged id).
    ``merge_token_ids`` are the ids produced by at least one merge, minus the
    declared added and special tokens in ``added_token_ids``.
    ``shipped_ignore_merges`` is the flag as the tokenizer file sets it.
    ``backend`` is the ``tokenizers.Tokenizer`` the wrapper encodes with, and
    ``encode_add_special_tokens`` the ``add_special_tokens`` value the wrapper's
    ``encode`` passes to it, so an encoding through the backend can reproduce
    the wrapper's ids exactly.
    """

    vocab: Dict[str, int]
    ranks: Dict[Tuple[int, int], Tuple[int, int]]
    merge_token_ids: FrozenSet[int]
    added_token_ids: FrozenSet[int]
    n_tokens_with_multiple_merges: int
    shipped_ignore_merges: bool
    byte_level: bool
    backend: Any
    encode_add_special_tokens: bool
    id_to_token: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id_to_token:
            self.id_to_token = {i: t for t, i in self.vocab.items()}

    def backend_copy(self, ignore_merges: Optional[bool] = None):
        """An independent copy of the backend, optionally with the flag set.

        A copy rather than the backend itself, so the tokenizer every other
        metric reads is never modified.
        """
        from tokenizers import Tokenizer
        copy = Tokenizer.from_str(self.backend.to_str())
        if ignore_merges is not None:
            copy.model.ignore_merges = ignore_merges
        return copy

    def replay(self, token_string: str) -> Optional[ReplayResult]:
        """Run BPE, with ``ignore_merges`` off, on *token_string* alone.

        Each character is one starting symbol, which is how the Rust BPE splits
        a word when ``continuing_subword_prefix`` and ``end_of_word_suffix``
        are empty (the only case :func:`merge_model_from_backend` accepts).
        Returns None when a character of the string is not in the vocabulary,
        so the string cannot be built by merges at all.
        """
        try:
            symbols = [self.vocab[ch] for ch in token_string]
        except KeyError:
            return None
        return _merge_all(symbols, self.ranks)


def _merge_all(symbols: List[int],
               ranks: Dict[Tuple[int, int], Tuple[int, int]]) -> ReplayResult:
    """``Word::merge_all`` over a doubly linked list of symbols."""
    n = len(symbols)
    if n < 2:
        return ReplayResult(tuple(symbols), ())
    sym = list(symbols)
    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))
    nxt[-1] = -1
    alive = [True] * n
    heap: List[Tuple[int, int, int]] = []
    for i in range(n - 1):
        hit = ranks.get((sym[i], sym[i + 1]))
        if hit is not None:
            heap.append((hit[0], i, hit[1]))
    heapq.heapify(heap)
    created: List[int] = []
    while heap:
        _, pos, new_id = heapq.heappop(heap)
        if not alive[pos]:
            continue
        right = nxt[pos]
        if right == -1:
            continue
        # The pair at pos may have changed since this entry was queued: either
        # side merged with something else. The entry is stale unless the pair
        # still produces the id it was queued for.
        current = ranks.get((sym[pos], sym[right]))
        if current is None or current[1] != new_id:
            continue
        sym[pos] = new_id
        created.append(new_id)
        alive[right] = False
        nxt[pos] = nxt[right]
        if nxt[right] != -1:
            prev[nxt[right]] = pos
        if prev[pos] >= 0:
            hit = ranks.get((sym[prev[pos]], sym[pos]))
            if hit is not None:
                heapq.heappush(heap, (hit[0], prev[pos], hit[1]))
        if nxt[pos] != -1:
            hit = ranks.get((sym[pos], sym[nxt[pos]]))
            if hit is not None:
                heapq.heappush(heap, (hit[0], pos, hit[1]))
    final = tuple(sym[i] for i in range(n) if alive[i])
    return ReplayResult(final, tuple(created))


def _mentions_byte_level(component: Any) -> bool:
    """True if a serialized tokenizers component is, or contains, ByteLevel."""
    if isinstance(component, dict):
        if component.get("type") == _BYTE_LEVEL_TYPE:
            return True
        return any(_mentions_byte_level(v) for v in component.values())
    if isinstance(component, list):
        return any(_mentions_byte_level(v) for v in component)
    return False


def _split_merge(entry: Any) -> Tuple[str, str]:
    """A merge as stored in tokenizer.json: ``"a b"`` or ``["a", "b"]``."""
    if isinstance(entry, list) and len(entry) == 2:
        return entry[0], entry[1]
    if isinstance(entry, str):
        parts = entry.split(" ")
        if len(parts) == 2:
            return parts[0], parts[1]
    raise ValueError(f"Unreadable merge entry {entry!r} in the tokenizer's merge list.")


def merge_model_from_backend(backend: Any, tokenizer_name: str,
                             encode_add_special_tokens: bool):
    """A :class:`BPEMergeModel` for a ``tokenizers.Tokenizer``, or why not.

    Returns :class:`NotApplicable` for tokenizers whose encoding this package
    cannot replay: a model other than BPE, dropout, or a non-empty subword
    prefix or end-of-word suffix (the merged string is then not ``a + b``).
    o200k stores both affixes as empty strings, which is the same as unset.

    Raises for a merge list that cannot describe the encoder: a pair listed
    twice, a merge whose parts or result are missing from the vocabulary, or a
    serialized tokenizer with no ``added_tokens`` list. A tokenizer loaded by
    the Rust library never reaches these: its loader drops a repeated pair,
    keeping the later rank, refuses a missing part, and always serializes
    ``added_tokens``. They guard a spec passed to :func:`merge_model_from_spec`
    directly.
    """
    from tokenizers import Tokenizer
    if not isinstance(backend, Tokenizer):
        return NotApplicable(
            f"no tokenizers backend ({type(backend).__name__}); the merge list "
            "cannot be read"
        )
    return merge_model_from_spec(json.loads(backend.to_str()), backend,
                                 tokenizer_name, encode_add_special_tokens)


def merge_model_from_spec(spec: Dict[str, Any], backend: Any, tokenizer_name: str,
                          encode_add_special_tokens: bool):
    """:func:`merge_model_from_backend` on an already serialized tokenizer.

    Separate so a merge list the Rust loader would refuse, such as one naming a
    token missing from the vocabulary, can still be checked.
    """
    model = spec.get("model") or {}
    model_type = model.get("type")
    if model_type != "BPE":
        return NotApplicable(f"model type is {model_type}, not BPE")
    if model.get("dropout"):
        return NotApplicable(f"BPE dropout is {model['dropout']}, so encoding is not deterministic")
    for affix in ("continuing_subword_prefix", "end_of_word_suffix"):
        if model.get(affix):
            return NotApplicable(
                f"{affix} is {model[affix]!r}; merges over affixed symbols are "
                "not supported"
            )

    vocab: Dict[str, int] = model.get("vocab") or {}
    if not vocab:
        raise ValueError(f"{tokenizer_name}: the BPE model has an empty vocabulary.")
    merges = model.get("merges")
    if not merges:
        return NotApplicable("the BPE model has no merges")

    if "added_tokens" not in spec:
        raise ValueError(
            f"{tokenizer_name}: the serialized tokenizer has no 'added_tokens' "
            "list, so special and added tokens cannot be excluded from the "
            "merge vocabulary."
        )
    added_ids = frozenset(int(t["id"]) for t in spec["added_tokens"])

    ranks: Dict[Tuple[int, int], Tuple[int, int]] = {}
    producers: Dict[int, int] = {}
    for rank, entry in enumerate(merges):
        left, right = _split_merge(entry)
        merged = left + right
        missing = [s for s in (left, right, merged) if s not in vocab]
        if missing:
            raise ValueError(
                f"{tokenizer_name}: merge #{rank} ({left!r}, {right!r}) refers "
                f"to {missing} which the vocabulary does not contain."
            )
        key = (vocab[left], vocab[right])
        if key in ranks:
            raise ValueError(
                f"{tokenizer_name}: the pair ({left!r}, {right!r}) is listed "
                f"twice in the merge list (ranks {ranks[key][0]} and {rank}). "
                "The Rust loader keeps the later rank; refusing to pick one."
            )
        ranks[key] = (rank, vocab[merged])
        producers[vocab[merged]] = producers.get(vocab[merged], 0) + 1

    merge_token_ids = frozenset(producers) - added_ids
    if not merge_token_ids:
        return NotApplicable("every merge produces a declared added or special token")

    byte_level = any(
        _mentions_byte_level(spec.get(part))
        for part in ("normalizer", "pre_tokenizer", "decoder")
    )

    return BPEMergeModel(
        vocab=vocab,
        ranks=ranks,
        merge_token_ids=merge_token_ids,
        added_token_ids=added_ids,
        n_tokens_with_multiple_merges=sum(1 for c in producers.values() if c > 1),
        shipped_ignore_merges=bool(model.get("ignore_merges", False)),
        byte_level=byte_level,
        backend=backend,
        encode_add_special_tokens=encode_add_special_tokens,
    )
