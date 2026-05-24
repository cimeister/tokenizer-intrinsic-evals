"""Plot smoke tests.

Kept intentionally minimal — these guard wiring/None-handling, not visual
output.
"""
import matplotlib
matplotlib.use("Agg")  # headless; must precede pyplot import in plots.py

from tokenizer_analysis.visualization.plots import (
    plot_vocab_util_cross_lingual_cov,
)


def test_plot_vocab_util_cov_smoke_and_none_skip(tmp_path):
    """The CoV plot renders for normal tokenizers and silently skips the
    bar for a None-CoV (single-language) tokenizer instead of crashing
    (exercises the plot_metric_bar_chart None-skip guard)."""
    results = {
        'vocabulary_utilization': {
            'per_tokenizer': {
                'A': {'global_utilization': 0.5, 'per_language_cov': 0.20},
                'B': {'global_utilization': 0.5, 'per_language_cov': 0.35},
                'C': {'global_utilization': 0.5, 'per_language_cov': None},
            },
            'metadata': {},
        }
    }
    out = tmp_path / "vocab_util_cov.svg"
    plot_vocab_util_cross_lingual_cov(results, str(out), ['A', 'B', 'C'])
    assert out.exists() and out.stat().st_size > 0
