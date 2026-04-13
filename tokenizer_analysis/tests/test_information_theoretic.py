"""Tests for tokenizer_analysis.metrics.information_theoretic (compression rate)."""

import pytest

from tokenizer_analysis.metrics.information_theoretic import InformationTheoreticMetrics
from tokenizer_analysis.core.input_types import TokenizedData
from tokenizer_analysis.config import TextMeasurementConfig, NormalizationMethod
from typing import List

from .conftest import SimpleProvider as _SimpleProvider


def _make_td(tok_name: str, text: str, n_tokens: int, lang: str = "en") -> TokenizedData:
    """Create a TokenizedData with *n_tokens* dummy token IDs."""
    return TokenizedData(
        tokenizer_name=tok_name,
        language=lang,
        tokens=list(range(n_tokens)),
        text=text,
    )


# ======================================================================
# T3: Compression rate uses ratio-of-means
# ======================================================================

class TestCompressionRateRatioOfMeans:

    def _make_metrics(self, tok_name: str) -> InformationTheoreticMetrics:
        provider = _SimpleProvider(tok_name)
        # Use bytes normalization for predictable unit counts
        config = TextMeasurementConfig(method=NormalizationMethod.BYTES)
        return InformationTheoreticMetrics(provider, measurement_config=config)

    def test_single_sample(self):
        """Single sample: ratio-of-means == per-sample ratio."""
        tok = "tok"
        m = self._make_metrics(tok)
        text = "hello"  # 5 bytes
        td = {tok: [_make_td(tok, text, 2)]}
        results = m.compute_compression_rate(td)
        rate = results["per_tokenizer"][tok]["global"]["compression_rate"]
        assert rate == pytest.approx(5.0 / 2.0)

    def test_ratio_of_means_not_mean_of_ratios(self):
        """Two samples with different sizes: ratio-of-means != mean-of-ratios.

        Sample 1: 10 bytes, 5 tokens  -> per-sample ratio = 2.0
        Sample 2: 2 bytes,  1 token   -> per-sample ratio = 2.0
        Mean-of-ratios = 2.0
        Ratio-of-means = 12 / 6 = 2.0  (same in this case)

        Now skew it:
        Sample 1: 10 bytes, 2 tokens  -> per-sample ratio = 5.0
        Sample 2: 2 bytes,  4 tokens  -> per-sample ratio = 0.5
        Mean-of-ratios = 2.75
        Ratio-of-means = 12 / 6 = 2.0
        """
        tok = "tok"
        m = self._make_metrics(tok)
        # "helloworld" = 10 bytes, "hi" = 2 bytes
        td = {tok: [
            _make_td(tok, "helloworld", 2),  # 10 bytes / 2 tokens = 5.0
            _make_td(tok, "hi", 4),           # 2 bytes / 4 tokens = 0.5
        ]}
        results = m.compute_compression_rate(td)
        rate = results["per_tokenizer"][tok]["global"]["compression_rate"]
        # Ratio-of-means: (10 + 2) / (2 + 4) = 12 / 6 = 2.0
        assert rate == pytest.approx(2.0)
        # Mean-of-ratios would give (5.0 + 0.5) / 2 = 2.75 — verify it's NOT that
        assert rate != pytest.approx(2.75)

    def test_totals_reported(self):
        """Global dict should include total_units and total_tokens."""
        tok = "tok"
        m = self._make_metrics(tok)
        td = {tok: [_make_td(tok, "abc", 3)]}
        results = m.compute_compression_rate(td)
        g = results["per_tokenizer"][tok]["global"]
        assert g["total_units"] == 3   # 3 ASCII bytes
        assert g["total_tokens"] == 3

    def test_per_language(self):
        """Per-language rates should also be ratio-of-means."""
        tok = "tok"
        m = self._make_metrics(tok)
        td = {tok: [
            _make_td(tok, "hello", 2, lang="en"),       # 5 bytes / 2 tokens
            _make_td(tok, "world!", 3, lang="en"),       # 6 bytes / 3 tokens
        ]}
        results = m.compute_compression_rate(td)
        en_rate = results["per_tokenizer"][tok]["per_language"]["en"]
        # (5 + 6) / (2 + 3) = 11 / 5 = 2.2
        assert en_rate == pytest.approx(11.0 / 5.0)


# ======================================================================
# TestBigramEntropy
# ======================================================================

def _make_td_tokens(tok_name: str, tokens: list, lang: str = "en") -> TokenizedData:
    """Create a TokenizedData with explicit token IDs (no text needed)."""
    return TokenizedData(
        tokenizer_name=tok_name,
        language=lang,
        tokens=tokens,
        text="dummy",
    )


class TestBigramEntropy:

    def _make_metrics(self, tok_name: str) -> InformationTheoreticMetrics:
        provider = _SimpleProvider(tok_name)
        return InformationTheoreticMetrics(provider)

    def test_uniform_successors(self):
        """Token 1 followed equally by 2,3,4,5,6 (5 times each) → η ≈ 1.0.

        Use separate 2-token documents so successor tokens never appear as
        left elements of bigrams (tests document boundary handling too).
        """
        tok = "tok"
        m = self._make_metrics(tok)
        docs = []
        for _ in range(5):
            for succ in [2, 3, 4, 5, 6]:
                docs.append(_make_td_tokens(tok, [1, succ]))
        td = {tok: docs}
        results = m.compute_bigram_entropy(td)
        eta = results['per_tokenizer'][tok]['global_bigram_entropy']
        assert eta == pytest.approx(1.0, abs=0.01)

    def test_dominated_successor_exact(self):
        """Token 1 followed by 2 (20x) and 3 (5x) → exact η value.

        Use separate 2-token documents to isolate token 1 as the only
        left-element type, making the exact value computable.

        H = -(20/25)*log2(20/25) - (5/25)*log2(5/25)
        H_max = log2(2)
        η = H / H_max
        """
        import math
        tok = "tok"
        m = self._make_metrics(tok)
        docs = []
        for _ in range(20):
            docs.append(_make_td_tokens(tok, [1, 2]))
        for _ in range(5):
            docs.append(_make_td_tokens(tok, [1, 3]))
        td = {tok: docs}
        results = m.compute_bigram_entropy(td)
        eta = results['per_tokenizer'][tok]['global_bigram_entropy']

        p1 = 20 / 25
        p2 = 5 / 25
        h = -(p1 * math.log2(p1) + p2 * math.log2(p2))
        expected_eta = h / math.log2(2)
        assert eta == pytest.approx(expected_eta)

    def test_single_successor(self):
        """[1,2,1,2,...] (>=5 bigrams) → only one successor for type 1, η = 0."""
        tok = "tok"
        m = self._make_metrics(tok)
        # 10 repetitions of [1,2] → token 1 always followed by 2
        seq = [1, 2] * 10
        td = {tok: [_make_td_tokens(tok, seq)]}
        results = m.compute_bigram_entropy(td)
        eta = results['per_tokenizer'][tok]['global_bigram_entropy']
        assert eta == pytest.approx(0.0)

    def test_below_threshold(self):
        """[1,2,3] has 2 bigrams, both types have <3 occurrences → all filtered."""
        tok = "tok"
        m = self._make_metrics(tok)
        td = {tok: [_make_td_tokens(tok, [1, 2, 3])]}
        results = m.compute_bigram_entropy(td)
        r = results['per_tokenizer'][tok]
        assert r['global_bigram_entropy'] == pytest.approx(0.0)
        assert r['global_types_evaluated'] == 0

    def test_per_language_separation(self):
        """Uniform lang should have higher η than skewed lang."""
        tok = "tok"
        m = self._make_metrics(tok)
        # Uniform language: token 1 → {2,3,4,5,6} each 5 times
        uniform_seq = []
        for _ in range(5):
            for succ in [2, 3, 4, 5, 6]:
                uniform_seq.extend([1, succ])
        # Skewed language: token 1 → 2 (20x), 3 (5x)
        skewed_seq = []
        for _ in range(20):
            skewed_seq.extend([1, 2])
        for _ in range(5):
            skewed_seq.extend([1, 3])

        td = {tok: [
            _make_td_tokens(tok, uniform_seq, lang="uniform"),
            _make_td_tokens(tok, skewed_seq, lang="skewed"),
        ]}
        results = m.compute_bigram_entropy(td)
        uniform_eta = results['per_tokenizer'][tok]['per_language']['uniform']['bigram_entropy']
        skewed_eta = results['per_tokenizer'][tok]['per_language']['skewed']['bigram_entropy']
        assert uniform_eta > skewed_eta

    def test_schema_keys_present(self):
        """All expected keys should exist in the result."""
        tok = "tok"
        m = self._make_metrics(tok)
        seq = []
        for _ in range(5):
            for succ in [2, 3, 4, 5, 6]:
                seq.extend([1, succ])
        td = {tok: [_make_td_tokens(tok, seq)]}
        results = m.compute_bigram_entropy(td)

        assert 'per_tokenizer' in results
        assert 'per_language' in results
        assert 'pairwise_comparisons' in results
        assert 'metadata' in results

        tok_r = results['per_tokenizer'][tok]
        assert 'global_bigram_entropy' in tok_r
        assert 'global_total_bigrams' in tok_r
        assert 'global_types_evaluated' in tok_r
        assert 'global_types_excluded' in tok_r
        assert 'per_language' in tok_r

    def test_no_bigrams_single_token_docs(self):
        """Single-token documents produce no bigrams → η = 0.0."""
        tok = "tok"
        m = self._make_metrics(tok)
        td = {tok: [
            _make_td_tokens(tok, [1]),
            _make_td_tokens(tok, [2]),
        ]}
        results = m.compute_bigram_entropy(td)
        r = results['per_tokenizer'][tok]
        assert r['global_bigram_entropy'] == pytest.approx(0.0)
        assert r['global_total_bigrams'] == 0

    def test_bigram_entropy_in_compute(self):
        """compute() should include bigram_entropy in its results."""
        tok = "tok"
        provider = _SimpleProvider(tok)
        m = InformationTheoreticMetrics(provider)
        docs = []
        for _ in range(5):
            for succ in [2, 3, 4, 5, 6]:
                docs.append(_make_td_tokens(tok, [1, succ]))
        td = {tok: docs}
        results = m.compute(td)
        assert 'bigram_entropy' in results
        assert 'per_tokenizer' in results['bigram_entropy']
        assert tok in results['bigram_entropy']['per_tokenizer']


# ======================================================================
# TestTrigramEntropy
# ======================================================================


class TestTrigramEntropy:

    def _make_metrics(self, tok_name: str, min_trigram_occurrences: int = 3) -> InformationTheoreticMetrics:
        provider = _SimpleProvider(tok_name)
        return InformationTheoreticMetrics(
            provider, min_trigram_occurrences=min_trigram_occurrences,
        )

    def test_uniform_successors(self):
        """Context (1,2) followed equally by 3,4,5,6,7 (5 times each) → η ≈ 1.0.

        Use separate 3-token documents so context (1,2) is the only trigram
        context and successor tokens never form new contexts.
        """
        tok = "tok"
        m = self._make_metrics(tok)
        docs = []
        for _ in range(5):
            for succ in [3, 4, 5, 6, 7]:
                docs.append(_make_td_tokens(tok, [1, 2, succ]))
        td = {tok: docs}
        results = m.compute_trigram_entropy(td)
        eta = results['per_tokenizer'][tok]['global_trigram_entropy']
        assert eta == pytest.approx(1.0, abs=0.01)

    def test_dominated_successor_exact(self):
        """Context (1,2) followed by 3 (20x) and 4 (5x) → exact η value.

        H = -(20/25)*log2(20/25) - (5/25)*log2(5/25)
        H_max = log2(2)
        η = H / H_max
        """
        import math
        tok = "tok"
        m = self._make_metrics(tok)
        docs = []
        for _ in range(20):
            docs.append(_make_td_tokens(tok, [1, 2, 3]))
        for _ in range(5):
            docs.append(_make_td_tokens(tok, [1, 2, 4]))
        td = {tok: docs}
        results = m.compute_trigram_entropy(td)
        eta = results['per_tokenizer'][tok]['global_trigram_entropy']

        p1 = 20 / 25
        p2 = 5 / 25
        h = -(p1 * math.log2(p1) + p2 * math.log2(p2))
        expected_eta = h / math.log2(2)
        assert eta == pytest.approx(expected_eta)

    def test_single_successor(self):
        """Context (1,2) always followed by 3 → η = 0."""
        tok = "tok"
        m = self._make_metrics(tok)
        docs = []
        for _ in range(10):
            docs.append(_make_td_tokens(tok, [1, 2, 3]))
        td = {tok: docs}
        results = m.compute_trigram_entropy(td)
        eta = results['per_tokenizer'][tok]['global_trigram_entropy']
        assert eta == pytest.approx(0.0)

    def test_below_threshold(self):
        """[1,2,3,4] has 2 trigrams, both contexts have <3 occurrences → all filtered."""
        tok = "tok"
        m = self._make_metrics(tok)
        td = {tok: [_make_td_tokens(tok, [1, 2, 3, 4])]}
        results = m.compute_trigram_entropy(td)
        r = results['per_tokenizer'][tok]
        assert r['global_trigram_entropy'] == pytest.approx(0.0)
        assert r['global_types_evaluated'] == 0

    def test_per_language_separation(self):
        """Uniform lang should have higher η than skewed lang."""
        tok = "tok"
        m = self._make_metrics(tok)
        # Uniform: context (1,2) → {3,4,5,6,7} each 5 times
        uniform_docs = []
        for _ in range(5):
            for succ in [3, 4, 5, 6, 7]:
                uniform_docs.append(_make_td_tokens(tok, [1, 2, succ], lang="uniform"))
        # Skewed: context (1,2) → 3 (20x), 4 (5x)
        skewed_docs = []
        for _ in range(20):
            skewed_docs.append(_make_td_tokens(tok, [1, 2, 3], lang="skewed"))
        for _ in range(5):
            skewed_docs.append(_make_td_tokens(tok, [1, 2, 4], lang="skewed"))

        td = {tok: uniform_docs + skewed_docs}
        results = m.compute_trigram_entropy(td)
        uniform_eta = results['per_tokenizer'][tok]['per_language']['uniform']['trigram_entropy']
        skewed_eta = results['per_tokenizer'][tok]['per_language']['skewed']['trigram_entropy']
        assert uniform_eta > skewed_eta

    def test_no_trigrams_short_docs(self):
        """Documents with ≤2 tokens produce no trigrams → η = 0.0."""
        tok = "tok"
        m = self._make_metrics(tok)
        td = {tok: [
            _make_td_tokens(tok, [1, 2]),
            _make_td_tokens(tok, [3]),
        ]}
        results = m.compute_trigram_entropy(td)
        r = results['per_tokenizer'][tok]
        assert r['global_trigram_entropy'] == pytest.approx(0.0)
        assert r['global_total_trigrams'] == 0

    def test_schema_keys_present(self):
        """All expected keys should exist in the result."""
        tok = "tok"
        m = self._make_metrics(tok)
        docs = []
        for _ in range(5):
            for succ in [3, 4, 5, 6, 7]:
                docs.append(_make_td_tokens(tok, [1, 2, succ]))
        td = {tok: docs}
        results = m.compute_trigram_entropy(td)

        assert 'per_tokenizer' in results
        assert 'per_language' in results
        assert 'pairwise_comparisons' in results
        assert 'metadata' in results

        tok_r = results['per_tokenizer'][tok]
        assert 'global_trigram_entropy' in tok_r
        assert 'global_total_trigrams' in tok_r
        assert 'global_types_evaluated' in tok_r
        assert 'global_types_excluded' in tok_r
        assert 'per_language' in tok_r

    def test_trigram_entropy_in_compute(self):
        """compute() should include trigram_entropy in its results."""
        tok = "tok"
        provider = _SimpleProvider(tok)
        m = InformationTheoreticMetrics(provider)
        docs = []
        for _ in range(5):
            for succ in [3, 4, 5, 6, 7]:
                docs.append(_make_td_tokens(tok, [1, 2, succ]))
        td = {tok: docs}
        results = m.compute(td)
        assert 'trigram_entropy' in results
        assert 'per_tokenizer' in results['trigram_entropy']
        assert tok in results['trigram_entropy']['per_tokenizer']

    def test_separate_threshold(self):
        """Trigram threshold should be independent of bigram threshold."""
        tok = "tok"
        # Set bigram threshold high, trigram threshold low
        provider = _SimpleProvider(tok)
        m = InformationTheoreticMetrics(
            provider, min_bigram_occurrences=100, min_trigram_occurrences=2,
        )
        # 3 occurrences of context (1,2) → should pass trigram threshold (2)
        docs = []
        for succ in [3, 4, 3]:
            docs.append(_make_td_tokens(tok, [1, 2, succ]))
        td = {tok: docs}

        tri_results = m.compute_trigram_entropy(td)
        tri_r = tri_results['per_tokenizer'][tok]
        assert tri_r['global_types_evaluated'] > 0

        bi_results = m.compute_bigram_entropy(td)
        bi_r = bi_results['per_tokenizer'][tok]
        # Bigram threshold is 100, so all types should be excluded
        assert bi_r['global_types_evaluated'] == 0
