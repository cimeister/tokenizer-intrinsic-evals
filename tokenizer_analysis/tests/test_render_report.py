"""The benchmark report renderer had no test at all.

`benchmarks/open_source/render_report.py` writes REPORT.md, which is the most
widely read artifact in the repository, and nothing covered it. The parts that
matter are the ones that can go wrong silently: a column reading the wrong key
renders `n/a` rather than raising, and a legend that has fallen behind the
severities the checker emits renders an unknown state as a bare string in a
cell.

The renderer is a script outside the package, so it is loaded by path.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "benchmarks" / "open_source" / "render_report.py"


@pytest.fixture(scope="module")
def render():
    spec = importlib.util.spec_from_file_location("render_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _results():
    """Two tokenizers, one with a per-line Gini and one without."""
    def block(vocab, comp, tpl, fert, gini, per_line):
        return {
            "vocabulary_utilization": {"global": {"vocab_size": vocab}},
            "compression_rate": {
                "global": {"compression_rate": comp},
                "tokens_per_line": {"global_avg": tpl},
                "per_language": {"eng_Latn": {"compression_rate": comp}},
            },
            "fertility": {"global": {"mean": fert}},
            "tokenizer_fairness_gini": {
                "global": {"gini_coefficient": gini},
                "per_line_normalization": per_line,
            },
        }

    a = block(50257, 3.21, 83.5, 1.42, 0.2000,
              {"gini_coefficient": 0.2933, "num_languages": 13})
    b = block(250002, 4.05, 45.9, 1.10, 0.0976, None)
    out = {}
    for metric in a:
        out[metric] = {"per_tokenizer": {"gpt2": a[metric], "xlm-roberta": b[metric]}}
    out["run_metadata"] = {"package_version": "test"}
    return out


def _sanity(severities):
    """One tokenizer entry per key of *severities*, keyed by full check name."""
    names = [
        "C1 byte-level 256-coverage",
        "C17 strict byte-alphabet vocab presence",
        "C6 digit handling",
        "C16 vocab reachability",
    ]
    per_tok = {}
    for tok, sevs in severities.items():
        per_tok[tok] = {
            "overall_severity": "fail",
            "checks": {n: {"severity": s} for n, s in zip(names, sevs)},
        }
    return {"tokenizer_sanity_check": {"per_tokenizer": per_tok}}


def _section(report, heading):
    """The text of one `## ` section, up to the next one.

    Three tables in the report have a row per tokenizer, so a search over the
    whole file finds the wrong one.
    """
    start = report.index(heading)
    rest = report.index("\n## ", start + len(heading))
    return report[start:rest]


def _headline_row(report, name):
    rows = [l for l in _section(report, "## Headline").splitlines()
            if l.startswith(f"| {name} |")]
    assert len(rows) == 1, f"expected one headline row for {name}, got {rows}"
    return rows[0]


def _render(render, tmp_path, sanity=None):
    results_path = tmp_path / "analysis_results.json"
    results_path.write_text(json.dumps(_results()), encoding="utf-8")
    argv = ["--results", str(results_path), "--output", str(tmp_path / "REPORT.md")]
    if sanity is not None:
        sanity_path = tmp_path / "sanity_results.json"
        sanity_path.write_text(json.dumps(sanity), encoding="utf-8")
        argv += ["--sanity", str(sanity_path)]
    assert render.main(argv) == 0
    return (tmp_path / "REPORT.md").read_text(encoding="utf-8")


class TestHeadlineColumns:
    def test_both_gini_columns_are_present_and_differ(self, render, tmp_path):
        report = _render(render, tmp_path)
        assert "Gini per byte (down)" in report
        assert "Gini per line (down)" in report
        # Distinct values, so the test cannot pass by rendering one twice.
        assert "0.2000" in report and "0.2933" in report

    def test_an_absent_per_line_block_renders_as_not_available(self, render, tmp_path):
        """A null block is a corpus that is not line-parallel, not a zero."""
        report = _render(render, tmp_path)
        row = _headline_row(report, "XLM-RoBERTa base")
        assert "0.0976" in row, row
        assert "| n/a |" in row, row
        assert "0.000" not in row, row

    def test_tokens_per_line_comes_from_the_results_file(self, render, tmp_path):
        report = _render(render, tmp_path)
        assert "Tokens/line (down)" in report
        assert "83.5" in report and "45.9" in report

    def test_no_number_in_the_headline_is_typed_by_hand(self, render, tmp_path):
        """Every headline cell must trace to a key of the fixture.

        A hardcoded figure would survive every other assertion here.
        """
        report = _render(render, tmp_path)
        cells = [c.strip() for c in _headline_row(report, "GPT-2").strip("|").split("|")]
        assert cells == ["GPT-2", "50,257", "3.210", "83.5", "1.420",
                         "0.2000", "0.2933", "n/a", "n/a", "n/a"]


class TestHealthMatrix:
    def test_one_column_per_check_in_the_sanity_file(self, render, tmp_path):
        sanity = _sanity({"gpt2": ["pass", "warn", "warn", "fail"],
                          "xlm-roberta": ["pass", "pass", "pass", "pass"]})
        report = _render(render, tmp_path, sanity)
        header = next(l for l in report.splitlines() if l.startswith("| Tokenizer | [C1]"))
        for cid in ("C1", "C17", "C6", "C16"):
            assert f"[{cid}](" in header, header
        # Four checks plus the name column.
        assert header.count("|") == 6, header

    def test_the_legend_covers_every_severity_emitted(self, render, tmp_path):
        """The assertion that catches a sixth severity being added later.

        A severity with no glyph renders as its own bare string in a cell and
        as nothing in the legend, which reads as a state the reader is meant
        to recognise.
        """
        emitted = ["pass", "warn", "fail", "not_applicable"]
        sanity = _sanity({"gpt2": emitted,
                          "xlm-roberta": ["unverifiable"] * 4})
        report = _render(render, tmp_path, sanity)
        legend = report[report.index("Legend, in worsening order:"):]
        for sev in set(emitted) | {"unverifiable"}:
            assert f"| `{sev}` |" in legend, f"{sev} is in the matrix and not the legend"

    def test_a_severity_absent_from_the_run_is_absent_from_the_legend(self, render, tmp_path):
        sanity = _sanity({"gpt2": ["pass"] * 4, "xlm-roberta": ["pass"] * 4})
        report = _render(render, tmp_path, sanity)
        legend = report[report.index("Legend, in worsening order:"):]
        assert "| `pass` |" in legend
        assert "| `fail` |" not in legend

    def test_the_column_links_resolve_in_the_docs_page(self, render, tmp_path):
        """A header link into SANITY_CHECKS.md must reach a real heading."""
        import re

        sanity = _sanity({"gpt2": ["pass"] * 4, "xlm-roberta": ["pass"] * 4})
        report = _render(render, tmp_path, sanity)
        page = (REPO_ROOT / "docs" / "SANITY_CHECKS.md").read_text(encoding="utf-8")
        headings = set()
        for line in page.splitlines():
            if line.startswith("#"):
                text = line.lstrip("#").strip().lower()
                headings.add(re.sub(r"[^\w\s-]", "", text).replace(" ", "-"))
        anchors = re.findall(r"\]\(\.\./\.\./docs/SANITY_CHECKS\.md#([\w-]+)\)", report)
        assert anchors, "the health matrix emitted no column links"
        missing = [a for a in anchors if a not in headings]
        assert not missing, f"docs/SANITY_CHECKS.md has no heading for {missing}"

    def test_no_sanity_file_omits_the_section_rather_than_half_rendering_it(
            self, render, tmp_path):
        report = _render(render, tmp_path)
        assert "## Tokenizer health" not in report
        # The rest of the report is still written.
        assert "## Headline" in report

    def test_a_sanity_file_with_no_tokenizers_emits_no_section(self, render, tmp_path):
        """Present but empty is the case the missing-file check does not cover.

        A header over an empty table reads as nine tokenizers with nothing
        wrong, which is the opposite of what an empty file means.
        """
        report = _render(render, tmp_path, {"tokenizer_sanity_check": {"per_tokenizer": {}}})
        assert "## Tokenizer health" not in report
        assert "## Headline" in report

    def test_a_named_sanity_file_that_is_absent_is_an_error(self, render, tmp_path):
        """Silently skipping a file the caller named would hide a typo."""
        results_path = tmp_path / "analysis_results.json"
        results_path.write_text(json.dumps(_results()), encoding="utf-8")
        with pytest.raises(SystemExit, match="does not exist"):
            render.main(["--results", str(results_path),
                         "--output", str(tmp_path / "REPORT.md"),
                         "--sanity", str(tmp_path / "nope.json")])


def _render_custom(render, tmp_path, results, name="custom"):
    """Render an arbitrary results dict, for prose that reads more than one row."""
    results_path = tmp_path / f"{name}_results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    out = tmp_path / f"{name}_REPORT.md"
    assert render.main(["--results", str(results_path), "--output", str(out)]) == 0
    return out.read_text(encoding="utf-8")


class TestGiniProseIsComputed:
    """The Gini comparison was typed into the renderer until 1.0.3.

    Spearman 0.650 and four coefficients sat in a string literal, so a
    regeneration on different data would have left the sentence contradicting
    the table directly above it, and the module docstring already promised that
    nothing in the report is transcribed by hand.
    """

    def _two_tokenizers(self, gini_a, line_a, gini_b, line_b):
        """gpt2 and xlm-roberta with the two coefficients set independently."""
        results = _results()
        block = results["tokenizer_fairness_gini"]["per_tokenizer"]
        block["gpt2"]["global"]["gini_coefficient"] = gini_a
        block["gpt2"]["per_line_normalization"] = {"gini_coefficient": line_a}
        block["xlm-roberta"]["global"]["gini_coefficient"] = gini_b
        block["xlm-roberta"]["per_line_normalization"] = {"gini_coefficient": line_b}
        return results

    def test_the_two_units_disagreeing_names_both_tokenizers_and_both_numbers(
            self, render, tmp_path):
        # gpt2 is lower per byte, xlm-roberta is lower per line.
        report = _render_custom(
            render, tmp_path,
            self._two_tokenizers(0.1000, 0.4000, 0.3000, 0.2000), "disagree")
        assert "They disagree on which tokenizer is the most equitable" in report
        assert "GPT-2 is lowest per byte at 0.1000" in report
        assert "XLM-RoBERTa base is lowest per line at 0.2000" in report
        assert "against 0.4000 per line" in report
        assert "against 0.3000 per byte" in report

    def test_the_two_units_agreeing_says_so_instead(self, render, tmp_path):
        """The disagreement claim must not be printed when there is none."""
        report = _render_custom(
            render, tmp_path,
            self._two_tokenizers(0.1000, 0.2000, 0.3000, 0.4000), "agree")
        assert "They disagree" not in report
        assert "GPT-2 has the lowest coefficient under both units" in report

    def test_the_correlation_is_recomputed_rather_than_quoted(
            self, render, tmp_path):
        """Two orderings, opposite correlations, from the same code path."""
        same = _render_custom(
            render, tmp_path,
            self._two_tokenizers(0.1000, 0.2000, 0.3000, 0.4000), "rho_pos")
        flipped = _render_custom(
            render, tmp_path,
            self._two_tokenizers(0.1000, 0.4000, 0.3000, 0.2000), "rho_neg")
        assert "Spearman 1.000" in same, same
        assert "Spearman -1.000" in flipped, flipped
        # The figure the renderer used to hardcode must not appear for data
        # that does not produce it.
        assert "0.650" not in same and "0.650" not in flipped

    def test_fewer_than_two_pairs_makes_no_claim(self, render, tmp_path):
        """The shipped fixture has one per-line value, which is this case."""
        report = _render(render, tmp_path)
        assert "Spearman" not in report
        assert "1 of the 2 tokenizers here" in report


class TestRoundTripProseIsComputed:
    def _with_skipped_cer(self, rate_a, rate_b):
        results = _results()
        results["reconstruction_fidelity"] = {
            "per_tokenizer": {
                "gpt2": {"global": {"exact_match_rate": rate_a, "mean_cer": None},
                         "cer_skipped": True},
                "xlm-roberta": {"global": {"exact_match_rate": rate_b,
                                           "mean_cer": None},
                                "cer_skipped": True},
            }
        }
        return results

    def test_the_worst_skipped_tokenizer_is_named_with_its_rate(
            self, render, tmp_path):
        """The rate was the literal 0.031, tying the sentence to one corpus."""
        report = _render_custom(render, tmp_path,
                                self._with_skipped_cer(0.5000, 0.0125), "cer")
        assert "XLM-RoBERTa base reconstructs 0.013 of its texts exactly" in report
        assert "0.031" not in report

    def test_the_named_tokenizer_follows_the_data(self, render, tmp_path):
        """Swap which one is worse; the sentence has to follow."""
        report = _render_custom(render, tmp_path,
                                self._with_skipped_cer(0.0125, 0.5000), "cer2")
        assert "GPT-2 reconstructs 0.013 of its texts exactly" in report


class TestMergeUsageSection:
    """Every number and every conditional sentence comes from the results file."""

    @staticmethod
    def _results(total_final_tokens=271_337):
        results = _results()
        results["merge_vocabulary_usage"] = {
            "metadata": {"corpus": "main input",
                         "constants": {"min_final_tokens_warn": 1_000_000}},
            "per_tokenizer": {
                "gpt2": {
                    "applicable": True,
                    "encoding": "reencoded_ignore_merges_false",
                    "texts_differing_after_reencode": 7,
                    "texts_per_corpus": {"prose": 3250},
                    "global": {
                        "n_merge_tokens": 50_000, "pct_active": 12.34,
                        "pct_rare": 5.67, "pct_uncommon": 8.9, "pct_unseen": 73.09,
                        "pct_scaffold": 1.23,
                        "total_final_tokens": total_final_tokens,
                        "threshold_counts": {"rare_below": 5 * total_final_tokens / 1e6,
                                             "uncommon_below": total_final_tokens / 1e6},
                        "n_unreachable_without_ignore_merges": 3,
                    },
                },
                "xlm-roberta": {"applicable": False, "global": None,
                                "reason": "model type is Unigram, not BPE"},
            },
        }
        return results

    def test_rows_read_the_results_file(self, render, tmp_path):
        section = _section(_render_custom(render, tmp_path, self._results()),
                           "## Merge vocabulary usage")
        assert "| GPT-2 | 50,000 | 12.34 | 5.67 | 8.90 | 73.09 | 1.23 |" in section
        assert "| XLM-RoBERTa base | n/a | n/a | n/a | n/a | n/a | n/a |" in section

    def test_sentences_follow_the_data(self, render, tmp_path):
        section = _section(_render_custom(render, tmp_path, self._results()),
                           "## Merge vocabulary usage")
        # 5 x 271,337 / 1e6 = 1.36, so Active starts at 2 emissions.
        assert "between 271,337 and 271,337 final tokens" in section
        assert "Active means emitted at least 2 times" in section
        assert ("GPT-2 ships with `ignore_merges` on and was encoded a second time "
                "with it off; this changed the token ids of 7 of 3,250 texts for "
                "GPT-2.") in section
        assert "GPT-2 has 3 merge tokens that cannot be built" in section
        assert ("The metric does not apply to XLM-RoBERTa base (model type is "
                "Unigram, not BPE).") in section

    def test_no_corpus_size_sentence_at_the_limit(self, render, tmp_path):
        section = _section(
            _render_custom(render, tmp_path, self._results(total_final_tokens=1_000_000)),
            "## Merge vocabulary usage")
        assert "final tokens per tokenizer" not in section

    def test_a_results_file_without_the_block_omits_the_section(self, render, tmp_path):
        assert "## Merge vocabulary usage" not in _render(render, tmp_path)
