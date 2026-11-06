"""Tests for LaTeX table generation, focused on the metric registry that reads
the per-tokenizer ``summary`` aggregates (math, code-AST, UTF-8, reconstruction,
entropy). The key_paths here are the canonical ones, so a drift between the
registry and a metric's actual output shape shows up in the generated tables.

Since 1.0 that drift surfaces in one of two ways, and both are covered below. A
key_path rooted at one of the six metrics ``merge_redundant_metrics`` deletes
can never resolve on any results file, so ``LaTeXTableGenerator.__init__``
raises; ``test_registry_rejects_a_path_into_a_merged_away_metric`` asserts that.
Any other unresolvable key_path still renders '---' in every row and logs a
warning, because a metric that published nulls for every tokenizer is a
legitimate run outcome and must not abort the run;
``test_every_registered_key_path_resolves_on_merged_results`` is what catches
that case, by running the real merge over each metric's own output shape and
requiring every registry entry to find a value."""
import pytest

from tokenizer_analysis.metrics.redundancy import MERGES, merge_redundant_metrics
from tokenizer_analysis.visualization.latex_tables import LaTeXTableGenerator


# Minimal results dict in the canonical schema: each metric block has a
# ``summary`` (or ``per_tokenizer``) dict keyed by tokenizer name. A is better
# than B on every metric below.
#
# The tables are generated from post-merge results (main.py calls
# merge_redundant_metrics before generate_latex_tables), so the merged metrics
# below sit where merge_redundant_metrics puts them, not at the top level.
_RESULTS = {
    'three_digit_boundary_alignment': {
        'summary': {'A': {'avg_f1': 0.80}, 'B': {'avg_f1': 0.60}},
    },
    'utf8_token_integrity': {
        'summary': {'A': {'boundary_crossing_rate': 0.0010},
                    'B': {'boundary_crossing_rate': 0.0060}},
        'per_tokenizer': {
            'A': {'char_split': {'global': {'split_rate': 0.0100}}},
            'B': {'char_split': {'global': {'split_rate': 0.0500}}},
        },
    },
    'vocabulary_utilization': {
        'per_tokenizer': {
            'A': {'global_utilization': 0.60,
                  'type_token_ratio': {'global_ttr': 0.70}},
            'B': {'global_utilization': 0.40,
                  'type_token_ratio': {'global_ttr': 0.55}},
        },
    },
    'compression_rate': {
        'per_tokenizer': {
            'A': {'global': {'compression_rate': 5.60},
                  'tokens_per_line': {'global_avg': 40.7, 'global_std_err': 3.97}},
            'B': {'global': {'compression_rate': 4.10},
                  'tokens_per_line': {'global_avg': 56.2, 'global_std_err': 5.93}},
        },
    },
    'renyi_efficiency': {
        'per_tokenizer': {
            'A': {'renyi_2.5': {'overall': 0.43},
                  'unigram_distribution': {'global_avg_token_rank': 263.5}},
            'B': {'renyi_2.5': {'overall': 0.33},
                  'unigram_distribution': {'global_avg_token_rank': 252.9}},
        },
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


class TestMergedMetricsStillRender:
    """The four metrics the redundancy merge relocates must still render.

    Their key_paths pointed at top-level metrics that merge_redundant_metrics
    deletes, so every table shipped a column of '---'. type_token_ratio is in
    the default metric list, so that column was in the default table.
    """

    def test_merged_metrics_render_values_not_placeholders(self):
        gen = LaTeXTableGenerator(_RESULTS, ['A', 'B'])
        table = gen.generate_basic_metrics_table(
            ['type_token_ratio', 'avg_token_rank', 'avg_tokens_per_line',
             'utf8_char_split']
        )
        assert '---' not in table
        assert '0.7000' in table   # type_token_ratio for A, {:.4f}
        assert '263.5' in table    # avg_token_rank for A, {:.1f}
        assert '40.7' in table     # avg_tokens_per_line for A, {:.1f}
        assert '0.0100' in table   # utf8_char_split for A, {:.4f}

    def test_tokens_per_line_carries_its_standard_error(self):
        # err_key reads from the relocated block, not from the tokenizer entry.
        # It renders in the column's own format, {:.1f}, so 3.97 prints as 4.0.
        gen = LaTeXTableGenerator(_RESULTS, ['A', 'B'])
        table = gen.generate_basic_metrics_table(['avg_tokens_per_line'])
        assert '40.7 {\\small $\\pm$ 4.0}' in table

    def test_default_basic_table_has_no_empty_column(self):
        gen = LaTeXTableGenerator(_RESULTS, ['A', 'B'])
        table = gen.generate_basic_metrics_table()
        # morphscore and gini are absent from _RESULTS, so they are legitimately
        # '---'. TTR is present via vocabulary_utilization and must not be.
        row_a = next(line for line in table.split('\n') if line.startswith('A &'))
        assert '0.7000' in row_a

    def test_registry_rejects_a_path_into_a_merged_away_metric(self):
        gen = LaTeXTableGenerator(_RESULTS, ['A'])
        gen.metric_configs['stale'] = {
            'title': 'Stale',
            'key_path': ['type_token_ratio', 'per_tokenizer'],
            'value_key': 'global_ttr',
            'stat_key': None,
            'err_key': None,
            'format': '{:.4f}',
            'lower_is_better': False,
        }
        with pytest.raises(ValueError, match='type_token_ratio'):
            gen._validate_registry()

    def test_every_registered_key_path_resolves_on_merged_results(self):
        """Each registry entry finds its value in a post-merge results dict.

        Built by running the real merge over blocks in each metric's own output
        shape, so a merge that moves a metric somewhere else, or a metric that
        renames a field, fails here rather than in a shipped table.
        """
        results = {
            'type_token_ratio': {
                'per_tokenizer': {'A': {'global_ttr': 0.70}},
            },
            'vocabulary_utilization': {
                'per_tokenizer': {'A': {'global_utilization': 0.60}},
            },
            'avg_tokens_per_line': {
                'per_tokenizer': {'A': {'global_avg': 40.7, 'global_std_err': 3.97}},
            },
            'compression_rate': {
                'per_tokenizer': {'A': {'global': {'compression_rate': 5.60}}},
            },
            'unigram_distribution_metrics': {
                'per_tokenizer': {'A': {'global_avg_token_rank': 263.5}},
            },
            'renyi_efficiency': {
                'per_tokenizer': {'A': {'renyi_1.0': {'overall': 0.53},
                                        'renyi_2.5': {'overall': 0.43}}},
            },
            'utf8_char_split': {
                'per_tokenizer': {'A': {'global': {'split_rate': 0.0100}}},
                'summary': {'A': {'split_rate': 0.0100}},
            },
            'utf8_token_integrity': {
                'per_tokenizer': {'A': {'global': {'completeness_rate': 0.99}}},
                'summary': {'A': {'boundary_crossing_rate': 0.0010}},
            },
            'fertility': {
                'per_tokenizer': {'A': {'global': {'mean': 1.64, 'std_err': 0.07}}},
            },
            'token_length': {
                'per_tokenizer': {'A': {'character_length': {'mean': 4.22,
                                                             'std_err': 0.13}}},
            },
            'tokenizer_fairness_gini': {
                'per_tokenizer': {'A': {'gini_coefficient': 0.121}},
            },
            'lorenz_curve_data': {'per_tokenizer': {'A': {'x': [0.0, 1.0]}}},
            'bigram_entropy': {
                'per_tokenizer': {'A': {'global_bigram_entropy': 9.5}},
            },
            'morphscore': {
                'per_tokenizer': {'A': {'summary': {
                    'avg_morphscore_recall': 0.51,
                    'avg_morphscore_recall_std_err': 0.02,
                    'avg_morphscore_precision': 0.49,
                    'avg_morphscore_precision_std_err': 0.03,
                }}},
            },
            'three_digit_boundary_alignment': {
                'per_tokenizer': {'A': {'overall': {}}},
                'summary': {'A': {'avg_f1': 0.80}},
            },
            'digit_split_variability': {
                'per_tokenizer': {'A': {'by_digit_length': {}, 'by_bucket': {}}},
            },
            'operator_isolation_rate': {
                'summary': {'A': {'overall_isolation_rate': 0.73}},
            },
            'ast_boundary_alignment': {
                'summary': {'A': {'avg_full_alignment_rate': 0.70}},
            },
            'identifier_fragmentation': {
                'summary': {'A': {'fragmentation_rate': 0.42}},
            },
            'indentation_consistency': {
                'summary': {'A': {'avg_depth_proportionality_correlation': 0.88}},
            },
            'reconstruction_fidelity': {
                'summary': {'A': {'exact_match_rate': 0.99, 'mean_cer': 0.002,
                                  'unk_token_rate': 0.0,
                                  'whitespace_fidelity': 0.86,
                                  'indentation_fidelity': 0.91,
                                  'newline_fidelity': 0.99,
                                  'tab_fidelity': 1.0}},
            },
            'merge_vocabulary_usage': {
                'per_tokenizer': {'A': {'global': {
                    'pct_active': 15.1, 'pct_rare': 27.5, 'pct_uncommon': 18.6,
                    'pct_unseen': 38.8, 'pct_scaffold': 4.6,
                }}},
            },
        }
        # Every metric the merge relocates is in the fixture, so the merge runs
        # in full rather than skipping a pair for a missing block.
        for secondary, primary, _, _ in MERGES:
            assert secondary in results and primary in results, (
                f"fixture is missing {secondary} or {primary}"
            )
        merged = merge_redundant_metrics(results)

        gen = LaTeXTableGenerator(merged, ['A'])
        unresolved = [
            key for key, config in gen.metric_configs.items()
            if gen._extract_metric_value(config, 'A')[0] is None
        ]
        assert not unresolved, (
            "registry key_paths that find no value in a post-merge results "
            "dict: " + ", ".join(sorted(unresolved))
        )
