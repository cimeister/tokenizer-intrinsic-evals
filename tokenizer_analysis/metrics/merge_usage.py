"""Merge vocabulary usage: how often each BPE merge token is used, and how.

For every vocabulary entry produced by a merge (a merge token), the metric
counts on the corpus how often the token is emitted as a final token and how
often it is formed as an intermediate symbol and then merged further. From those
two counts it partitions the merge tokens into Active, Rare, Uncommon and
Unseen, and marks Scaffold tokens: Rare or Uncommon tokens that are mostly
formed as an intermediate step rather than emitted.

Counts are those of encoding with ``ignore_merges=False``. A tokenizer that
ships with the flag on is re-encoded with it off, because with it on a
pre-token that is already a vocabulary entry is emitted without the merges
that build it, and the intermediate counts would then describe merges the
encoder never ran.

Intermediate symbols are found by replaying BPE on each distinct emitted
token's string (``core/bpe_merges.py``). Every merge applied inside a word joins
two symbols that end up in the same emitted token, so the replay of the emitted
token alone forms the same intermediates as the encoder did inside the word.

Ported from ``scripts/nonemitting_tokens.py`` (2026-07). That script took each
token's merge tree to be fixed, keeping the last merge listed for a token that
several merges produce. Apertus and llama3 have about 90,000 such tokens, so the
script's intermediate counts for them followed an arbitrary choice.
"""

import logging
import math
from collections import Counter
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from ..constants import (
    AGGREGATION_MICRO_POOLED,
    DEFAULT_CHUNK_SIZE,
    MERGE_USAGE_BYTE_FRAGMENT_EXAMPLE_COUNT,
    MERGE_USAGE_EXAMPLE_COUNT,
    MERGE_USAGE_MIN_FINAL_TOKENS_WARN,
    MERGE_USAGE_RARE_RATE_PER_M,
    MERGE_USAGE_SCAFFOLD_SURVIVAL_MAX,
    MERGE_USAGE_UNCOMMON_RATE_PER_M,
)
from ..core.bpe_merges import BPEMergeModel, NotApplicable
from ..core.input_providers import PreTokenizedProvider
from ..core.input_types import CODE_CORPUS, MATH_CORPUS, PROSE_CORPUS, TokenizedData
from .base import BaseMetrics
from .utf8_integrity import (
    _GPT2_DETECTION_THRESHOLD,
    _GPT2_MARKER_CHARS,
    _GPT2_UNICODE_TO_BYTE,
)

logger = logging.getLogger(__name__)

METRIC_KEY = 'merge_vocabulary_usage'

_PER_MILLION = 1_000_000


def _exact(value: float) -> Fraction:
    """A threshold as the decimal it is written as, so comparisons are exact."""
    return Fraction(str(value))


class MergeVocabularyUsageMetrics(BaseMetrics):
    """Active / Rare / Uncommon / Unseen partition of the merge vocabulary, plus Scaffold."""

    def compute(
        self,
        tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None,
        include_code_math: bool = False,
    ) -> Dict[str, Any]:
        """Compute the metric for every tokenizer.

        *include_code_math* adds the provider's registered code and math
        corpora to the pooled counts. With nothing registered, the bundled
        samples are registered and counted, and the metadata says so.
        """
        if tokenized_data is None:
            tokenized_data = self.input_provider.get_tokenized_data()

        extra: Dict[str, Dict[str, List[TokenizedData]]] = {}
        sources: Dict[str, Dict[str, Any]] = {}
        if include_code_math:
            for name in (CODE_CORPUS, MATH_CORPUS):
                corpus = self._code_or_math_corpus(name)
                extra[name] = self.input_provider.get_corpus_data(name)
                sources[name] = {
                    'source': corpus.source,
                    'synthetic': corpus.synthetic,
                    'n_texts': sum(len(texts) for texts in corpus.texts.values()),
                }
                if corpus.synthetic:
                    logger.info(
                        "%s: counting the bundled %s samples from %s (%d texts)",
                        METRIC_KEY, name, corpus.source, sources[name]['n_texts'],
                    )

        per_tokenizer: Dict[str, Any] = {}
        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                raise ValueError(
                    f"{METRIC_KEY}: no tokenized data for {tok_name!r}, which the "
                    "input provider lists. The metric pools the whole corpus, so "
                    "a missing tokenizer cannot be skipped."
                )
            pools = {PROSE_CORPUS: tokenized_data[tok_name]}
            for name, data in extra.items():
                if tok_name not in data:
                    raise ValueError(
                        f"{METRIC_KEY}: the {name!r} corpus has no encoding for "
                        f"{tok_name!r}."
                    )
                pools[name] = data[tok_name]
            per_tokenizer[tok_name] = self._compute_one(tok_name, pools)

        return {METRIC_KEY: {
            'per_tokenizer': per_tokenizer,
            'metadata': self._metadata(include_code_math, sources),
        }}

    # ------------------------------------------------------------------
    # Corpora
    # ------------------------------------------------------------------

    def _code_or_math_corpus(self, name: str):
        """The registered code or math corpus, registering the bundled one if absent."""
        corpus = self._registered_corpus(name)
        if corpus is not None:
            return corpus
        from ..loaders.corpora import resolve_code_corpus, resolve_math_corpus
        built = resolve_code_corpus(None) if name == CODE_CORPUS else resolve_math_corpus()
        logger.info(
            "%s: no %s corpus registered; counting the bundled samples from %s",
            METRIC_KEY, name, built.source,
        )
        return self._register_corpus(built)

    # ------------------------------------------------------------------
    # One tokenizer
    # ------------------------------------------------------------------

    def _not_applicable(self, tok_name: str, reason: str) -> Dict[str, Any]:
        logger.warning("%s: not applicable to %s: %s", METRIC_KEY, tok_name, reason)
        return {'applicable': False, 'reason': reason, 'global': None}

    def _compute_one(self, tok_name: str,
                     pools: Dict[str, List[TokenizedData]]) -> Dict[str, Any]:
        if isinstance(self.input_provider, PreTokenizedProvider):
            return self._not_applicable(
                tok_name,
                "pre-tokenized input: the token ids were read from a file, not "
                "produced by this tokenizer's encoder",
            )
        wrapper = self.input_provider.get_tokenizer(tok_name)
        model = wrapper.get_bpe_merge_model()
        if isinstance(model, NotApplicable):
            return self._not_applicable(tok_name, model.reason)
        self._check_byte_level(tok_name, model)

        for corpus, records in pools.items():
            for record in records:
                if record.text is None:
                    raise ValueError(
                        f"{METRIC_KEY}: a {corpus!r} record for {tok_name!r} "
                        f"(language {record.language!r}) carries no text, so it "
                        "cannot be re-encoded to check or recount its ids."
                    )

        # The merge model describes the encoder only if encoding through the
        # backend reproduces the ids every other metric read.
        self._check_backend_reproduces_ids(tok_name, model, pools)

        texts_differing: Optional[int] = None
        if model.shipped_ignore_merges:
            flag_off = model.backend_copy(ignore_merges=False)
            encodings: Dict[str, List[List[int]]] = {}
            texts_differing = 0
            for corpus, records in pools.items():
                ids = self._encode(flag_off, records, model.encode_add_special_tokens)
                texts_differing += sum(
                    1 for record, new in zip(records, ids) if record.tokens != new
                )
                encodings[corpus] = ids
            encoding = 'reencoded_ignore_merges_false'
        else:
            encodings = {corpus: [r.tokens for r in records]
                         for corpus, records in pools.items()}
            encoding = 'shared'

        entry = self._classify(tok_name, model, pools, encodings)
        entry.update({
            'applicable': True,
            'n_excluded_special_tokens': len(model.added_token_ids),
            'shipped_ignore_merges': model.shipped_ignore_merges,
            'encoding': encoding,
            'texts_differing_after_reencode': texts_differing,
            'n_tokens_with_multiple_merges': model.n_tokens_with_multiple_merges,
            'byte_fragment_detection': (
                'byte-level vocabulary' if model.byte_level
                else 'character vocabulary: no merge token can be a byte fragment'
            ),
        })
        return entry

    @staticmethod
    def _encode(backend, records: List[TokenizedData],
                add_special_tokens: bool) -> List[List[int]]:
        ids: List[List[int]] = []
        for start in range(0, len(records), DEFAULT_CHUNK_SIZE):
            chunk = [r.text for r in records[start:start + DEFAULT_CHUNK_SIZE]]
            ids.extend(
                e.ids for e in backend.encode_batch(chunk, add_special_tokens=add_special_tokens)
            )
        return ids

    def _check_backend_reproduces_ids(self, tok_name: str, model: BPEMergeModel,
                                      pools: Dict[str, List[TokenizedData]]) -> None:
        backend = model.backend_copy()
        for corpus, records in pools.items():
            ids = self._encode(backend, records, model.encode_add_special_tokens)
            for record, own in zip(records, ids):
                if list(record.tokens) != own:
                    raise ValueError(
                        f"{METRIC_KEY}: encoding through the tokenizers backend of "
                        f"{tok_name!r} does not reproduce the ids the run holds for "
                        f"a {corpus!r} text (language {record.language!r}, first 80 "
                        f"characters {record.text[:80]!r}): {len(own)} ids against "
                        f"{len(record.tokens)}. The wrapper encodes by some other "
                        "route, so the merge list does not describe its encoder."
                    )

    @staticmethod
    def _check_byte_level(tok_name: str, model: BPEMergeModel) -> None:
        """Refuse a vocabulary that looks byte-level but declares no ByteLevel component.

        A byte-level vocabulary spells every base symbol as one of the 256
        GPT-2 byte characters, so it has no single-character entry outside that
        table. A character vocabulary has many: gemma-2 has 20,465 and the
        Llama 2 tokenizer 1,899, against 0 for llama3, o200k and Apertus
        (measured 2026-11-06). Counting only the marker characters, as the
        UTF-8 metric's fallback heuristic does, misread gemma-2, whose 68 Latin
        Extended-A letters (U+0100 to U+0143) are also GPT-2 markers.
        """
        if model.byte_level:
            return
        singles = [t for i, t in model.id_to_token.items()
                   if len(t) == 1 and i not in model.added_token_ids]
        outside = sum(1 for t in singles if t not in _GPT2_UNICODE_TO_BYTE)
        markers = sum(1 for t in singles if t in _GPT2_MARKER_CHARS)
        if outside == 0 and markers >= _GPT2_DETECTION_THRESHOLD:
            raise ValueError(
                f"{METRIC_KEY}: {tok_name!r} declares no ByteLevel normalizer, "
                "pre-tokenizer or decoder, but every single-character entry of "
                f"its vocabulary is a GPT-2 byte character ({markers} of them "
                "markers). Whether a merge token is a byte fragment cannot be "
                "decided."
            )

    # ------------------------------------------------------------------
    # Counting and classification
    # ------------------------------------------------------------------

    def _classify(self, tok_name: str, model: BPEMergeModel,
                  pools: Dict[str, List[TokenizedData]],
                  encodings: Dict[str, List[List[int]]]) -> Dict[str, Any]:
        added = model.added_token_ids
        merge_ids = model.merge_token_ids

        final: Counter = Counter()
        per_language: Dict[str, Counter] = {}
        for corpus, records in pools.items():
            lang_counts: Counter = Counter()
            for record, ids in zip(records, encodings[corpus]):
                final.update(ids)
                lang_counts[record.language] += sum(1 for i in ids if i not in added)
            per_language[corpus] = lang_counts
        n_final = sum(c for i, c in final.items() if i not in added)
        if n_final == 0:
            raise ValueError(
                f"{METRIC_KEY}: the corpus encodes to no final tokens for "
                f"{tok_name!r} (special and added tokens excluded), so no rate "
                "is defined and every merge token would read as Unseen."
            )

        # Replay every merge token once: emitted ones give the intermediate
        # counts, and all of them give the count that cannot be formed at all
        # with ignore_merges off.
        created_by: Dict[int, Tuple[int, ...]] = {}
        unreachable = 0
        for t in merge_ids:
            result = model.replay(model.id_to_token[t])
            if result is None or result.symbols != (t,):
                unreachable += 1
                if final.get(t, 0):
                    raise ValueError(
                        f"{METRIC_KEY}: {tok_name!r} emitted "
                        f"{model.id_to_token[t]!r} {final[t]} times with "
                        "ignore_merges off, but replaying its merges does not "
                        "build it. The replay does not match this encoder."
                    )
                continue
            created_by[t] = result.created[:-1]

        stepping: Counter = Counter()
        # For each intermediate, how often it is formed inside each emitted
        # token; the examples name the emitted token it is formed inside most.
        inside: Dict[int, Counter] = {}
        for t, count in final.items():
            if t in created_by:
                for intermediate in created_by[t]:
                    stepping[intermediate] += count
                    inside.setdefault(intermediate, Counter())[t] += count

        rare_rate = _exact(MERGE_USAGE_RARE_RATE_PER_M)
        uncommon_rate = _exact(MERGE_USAGE_UNCOMMON_RATE_PER_M)
        survival_max = _exact(MERGE_USAGE_SCAFFOLD_SURVIVAL_MAX)
        rare_bound = rare_rate * n_final          # compared against final * 1e6
        uncommon_bound = uncommon_rate * n_final

        buckets = {'active': [], 'rare': [], 'uncommon': [], 'unseen': []}
        scaffold: List[int] = []
        built_rates: List[Fraction] = []
        for t in merge_ids:
            fin = final.get(t, 0)
            formed = fin + stepping.get(t, 0)
            if formed == 0:
                buckets['unseen'].append(t)
                continue
            scaled = fin * _PER_MILLION
            if scaled >= rare_bound:
                buckets['active'].append(t)
            elif scaled >= uncommon_bound:
                buckets['rare'].append(t)
            else:
                buckets['uncommon'].append(t)
            if scaled < rare_bound and fin < survival_max * formed:
                scaffold.append(t)
            if n_final:
                built_rates.append(Fraction(fin, n_final))

        fragments = [t for t in scaffold if self._is_byte_fragment(tok_name, model, t)]

        n_merge = len(merge_ids)

        def pct(count: int) -> float:
            return round(100 * count / n_merge, 2)

        built_rates.sort()
        median = (round(float(built_rates[len(built_rates) // 2] * _PER_MILLION), 3)
                  if built_rates else None)

        warnings: List[str] = []
        if n_final < MERGE_USAGE_MIN_FINAL_TOKENS_WARN:
            # Emissions are whole numbers, so a threshold of r emissions is
            # met by ceil(r) of them.
            active_from = math.ceil(rare_bound / _PER_MILLION)
            rare_from = math.ceil(uncommon_bound / _PER_MILLION)
            message = (
                f"{n_final} final tokens, under {MERGE_USAGE_MIN_FINAL_TOKENS_WARN}. "
                f"Active means emitted at least {active_from} times, Rare at "
                f"least {rare_from} times, and Uncommon never on its own, so the "
                "categories are counts of a few emissions rather than rates."
            )
            if active_from <= rare_from:
                message += (" No token can be Rare: one emission already reaches "
                            "the Active threshold.")
            warnings.append(message)
            logger.warning("%s: %s: %s", METRIC_KEY, tok_name, message)

        examples = self._examples(model, final, stepping, inside, scaffold, fragments)
        per_corpus = {corpus: sum(counts.values()) for corpus, counts in per_language.items()}
        texts_per_corpus = {corpus: len(records) for corpus, records in pools.items()}

        return {
            'global': {
                'pct_active': pct(len(buckets['active'])),
                'pct_rare': pct(len(buckets['rare'])),
                'pct_uncommon': pct(len(buckets['uncommon'])),
                'pct_unseen': pct(len(buckets['unseen'])),
                'pct_scaffold': pct(len(scaffold)),
                'pct_scaffold_byte_fragment': pct(len(fragments)),
                'counts': {
                    'active': len(buckets['active']),
                    'rare': len(buckets['rare']),
                    'uncommon': len(buckets['uncommon']),
                    'unseen': len(buckets['unseen']),
                    'scaffold': len(scaffold),
                    'scaffold_byte_fragment': len(fragments),
                },
                'n_merge_tokens': n_merge,
                'total_final_tokens': n_final,
                'threshold_counts': {
                    'uncommon_below': float(uncommon_rate * n_final / _PER_MILLION),
                    'rare_below': float(rare_rate * n_final / _PER_MILLION),
                },
                'n_unreachable_without_ignore_merges': unreachable,
                'median_standalone_per_million': median,
            },
            'warnings': warnings,
            'final_tokens_per_corpus': per_corpus,
            'texts_per_corpus': texts_per_corpus,
            'final_tokens_per_language': {
                corpus: dict(sorted(counts.items())) for corpus, counts in per_language.items()
            },
            'examples_scaffold': examples[0],
            'examples_scaffold_byte_fragment': examples[1],
        }

    @staticmethod
    def _token_bytes(tok_name: str, model: BPEMergeModel, t: int) -> bytes:
        token = model.id_to_token[t]
        try:
            return bytes(_GPT2_UNICODE_TO_BYTE[ch] for ch in token)
        except KeyError as e:
            raise ValueError(
                f"{METRIC_KEY}: {tok_name!r} is byte-level but merge token "
                f"{token!r} contains {e.args[0]!r}, which is not a GPT-2 byte "
                "character."
            ) from None

    def _is_byte_fragment(self, tok_name: str, model: BPEMergeModel, t: int) -> bool:
        if not model.byte_level:
            return False
        try:
            self._token_bytes(tok_name, model, t).decode('utf-8')
        except UnicodeDecodeError:
            return True
        return False

    def _examples(self, model: BPEMergeModel, final: Counter, stepping: Counter,
                  inside: Dict[int, Counter], scaffold: List[int], fragments: List[int]):
        def formed(t: int) -> int:
            return final.get(t, 0) + stepping.get(t, 0)

        def text(t: int) -> str:
            token = model.id_to_token[t]
            if not model.byte_level:
                return token
            return bytes(_GPT2_UNICODE_TO_BYTE[ch] for ch in token).decode(
                'utf-8', errors='backslashreplace')

        def describe(t: int) -> Dict[str, Any]:
            # The emitted token t is formed inside most often, ties to the
            # lower id; None for a token that is only ever emitted.
            hosts = inside.get(t)
            parent = (min(hosts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
                      if hosts else None)
            built = formed(t)
            return {
                'token': model.id_to_token[t],
                'text': text(t),
                'final': final.get(t, 0),
                'formed': built,
                'survival': round(final.get(t, 0) / built, 4) if built else None,
                'formed_inside_most': model.id_to_token[parent] if parent is not None else None,
            }

        fragment_set = set(fragments)
        # Ties broken by id so the examples do not depend on set order.
        order = lambda t: (-formed(t), t)
        plain = sorted((t for t in scaffold if t not in fragment_set), key=order)
        return ([describe(t) for t in plain[:MERGE_USAGE_EXAMPLE_COUNT]],
                [describe(t) for t in sorted(fragments, key=order)[:MERGE_USAGE_BYTE_FRAGMENT_EXAMPLE_COUNT]])

    # ------------------------------------------------------------------
    # Metadata and printing
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata(include_code_math: bool, sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        meta = {
            'description': (
                'Use of the merge vocabulary on the corpus, encoded with '
                'ignore_merges=False. A merge token is a vocabulary entry produced '
                'by at least one merge, excluding declared special and added '
                'tokens. final = emissions as a final token; formed = final + '
                'times formed as an intermediate symbol and merged further; '
                'N = final tokens excluding special and added tokens; '
                'standalone rate = final / N; survival = final / formed. '
                'Active: formed > 0 and rate >= %g per million. Rare: formed > 0 '
                'and %g <= rate per million < %g. Uncommon: formed > 0 and rate '
                '< %g per million. Unseen: formed = 0. The four sum to 100%% of '
                'merge tokens. Scaffold, counted separately: Rare or Uncommon, '
                'and survival < %g.'
                % (MERGE_USAGE_RARE_RATE_PER_M, MERGE_USAGE_UNCOMMON_RATE_PER_M,
                   MERGE_USAGE_RARE_RATE_PER_M, MERGE_USAGE_UNCOMMON_RATE_PER_M,
                   MERGE_USAGE_SCAFFOLD_SURVIVAL_MAX)
            ),
            'constants': {
                'uncommon_rate_per_million': MERGE_USAGE_UNCOMMON_RATE_PER_M,
                'rare_rate_per_million': MERGE_USAGE_RARE_RATE_PER_M,
                'scaffold_survival_max': MERGE_USAGE_SCAFFOLD_SURVIVAL_MAX,
                'min_final_tokens_warn': MERGE_USAGE_MIN_FINAL_TOKENS_WARN,
            },
            'aggregation': AGGREGATION_MICRO_POOLED,
            'count_unit': 'merge tokens',
            'corpus': ('main input plus the registered code and math corpora'
                       if include_code_math else 'main input'),
            'per_language': (
                'Not published. The counts are pooled over every language, so '
                'the language mix of the input affects the percentages; the '
                'final-token count of each language is in the full results file.'
            ),
        }
        if include_code_math:
            meta['code_math_sources'] = sources
        return meta

    def print_results(self, results: Dict[str, Any]) -> None:
        block = results.get(METRIC_KEY, {}).get('per_tokenizer', {})
        print("\nMERGE VOCABULARY USAGE (% of merge tokens)")
        for tok_name, entry in block.items():
            if not entry.get('applicable'):
                print(f"  {tok_name:32s} not applicable: {entry.get('reason')}")
                continue
            g = entry['global']
            print(f"  {tok_name:32s} active={g['pct_active']:.2f} rare={g['pct_rare']:.2f} "
                  f"uncommon={g['pct_uncommon']:.2f} unseen={g['pct_unseen']:.2f} "
                  f"scaffold={g['pct_scaffold']:.2f}")
