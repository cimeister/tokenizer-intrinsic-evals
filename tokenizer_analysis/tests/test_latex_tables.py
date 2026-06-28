"""Tests for LaTeX table generation, focused on the metric registry that reads
the per-tokenizer ``summary`` aggregates (math, code-AST, UTF-8, reconstruction,
entropy). These metrics share the canonical key_paths used by the markdown
leaderboard, so the two tables report the same numbers."""
from tokenizer_analysis.visualization.latex_tables import LaTeXTableGenerator


# Minimal results dict in the canonical schema: each metric block has a
# ``summary`` (or ``per_tokenizer``) dict keyed by tokenizer name. A is better
# than B on every metric below.
_RESULTS = {
    'three_digit_boundary_alignment': {
        'summary': {'A': {'avg_f1': 0.80}, 'B': {'avg_f1': 0.60}},
    },
    'utf8_char_split': {
        'summary': {'A': {'split_rate': 0.0100}, 'B': {'split_rate': 0.0500}},
    },
    'reconstruction_fidelity': {
        'summary': {
            'A': {'mean_cer': 0.0020, 'exact_match_rate': 0.990},
            'B': {'mean_cer': 0.0100, 'exact_match_rate': 0.950},
        },
    },
    'ast_boundary_alignment': {
        'summary': {'A': {'avg_full_alignment_rate': 0.70},
                    'B': {'avg_full_alignment_rate': 0.65}},
    },
    'bigram_entropy': {
        'per_tokenizer': {'A': {'global_bigram_entropy': 9.5},
                          'B': {'global_bigram_entropy': 9.0}},
    },
}


class TestSummaryMetricExtraction:
    def test_summary_metrics_render_real_values(self):
        gen = LaTeXTableGenerator(_RESULTS, ['A', 'B'])
        table = gen.generate_basic_metrics_table(
            ['three_digit_boundary_f1', 'utf8_char_split', 'mean_cer',
             'ast_full_alignment', 'exact_match_rate', 'bigram_entropy']
        )
        # Values are pulled from summary[tok][value_key], not rendered as '---'.
        assert '0.800' in table          # avg_f1 for A
        assert '0.0100' in table         # split_rate for A ({:.4f})
        assert '0.0020' in table         # mean_cer for A
        assert '0.700' in table          # ast full alignment for A
        assert '9.500' in table          # bigram entropy for A
        assert '0.990' in table          # exact match for A

    def test_best_is_bolded_per_direction(self):
        gen = LaTeXTableGenerator(_RESULTS, ['A', 'B'])
        # higher-is-better metric: A (0.80) is best and bolded
        t_f1 = gen.generate_basic_metrics_table(['three_digit_boundary_f1'])
        assert '\\textbf{0.800}' in t_f1
        # lower-is-better metric: A (0.0020) is best and bolded
        t_cer = gen.generate_basic_metrics_table(['mean_cer'])
        assert '\\textbf{0.0020}' in t_cer

    def test_direction_arrows(self):
        gen = LaTeXTableGenerator(_RESULTS, ['A', 'B'])
        # 3-digit F1 is higher-is-better -> up arrow
        assert '$\\uparrow$' in gen.generate_basic_metrics_table(['three_digit_boundary_f1'])
        # char split is lower-is-better -> down arrow
        assert '$\\downarrow$' in gen.generate_basic_metrics_table(['utf8_char_split'])

    def test_missing_metric_renders_placeholder(self):
        # No 'operator_isolation_rate' block in _RESULTS -> '---' for both rows.
        gen = LaTeXTableGenerator(_RESULTS, ['A', 'B'])
        table = gen.generate_basic_metrics_table(['operator_isolation'])
        assert table.count('---') == 2

    def test_comprehensive_includes_available_summary_metrics(self):
        # generate_comprehensive_table picks up any registered metric with data.
        # Assert on rendered values and short (un-wrapped) titles; long titles
        # like "3-Digit Align. F1" are split across \makecell lines.
        gen = LaTeXTableGenerator(_RESULTS, ['A', 'B'])
        table = gen.generate_comprehensive_table()
        assert 'AST Align.' in table
        assert '0.800' in table   # 3-digit boundary F1 value (metric included)
        assert '0.0020' in table  # mean CER value
        assert '9.500' in table   # bigram entropy value
