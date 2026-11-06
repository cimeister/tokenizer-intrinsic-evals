"""Tests for the contract `analysis_results.json` makes to its readers.

The README documents this file's shape and nothing enforced it. These tests
cover the properties a downstream consumer relies on and that the 1.0 audit
found broken: the metric key set, that absent values are null rather than a
plausible zero, that the file is strict JSON, and that it says what produced it.

They run the CLI in-process against the bundled demo, so they exercise the real
assembly path rather than a mock of it.
"""
import json
import subprocess
import sys

import pytest

from .conftest import requires_flores

from tokenizer_analysis.metrics.base import BaseMetrics
from tokenizer_analysis.metrics.redundancy import MERGES, merge_redundant_metrics


REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def _reject_non_standard(constant: str):
    raise ValueError(f"non-standard JSON token: {constant}")


@pytest.fixture(scope="module")
def demo_results(tmp_path_factory):
    """Run the bundled demo once and return the parsed results file.

    Kept module-scoped because encoding dominates the runtime; the tests below
    only read it.
    """
    out = tmp_path_factory.mktemp("demo")
    proc = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--use-sample-data", "--samples-per-lang", "10",
         "--no-plots", "--no-code-ast", "--output-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            "demo run failed with exit "
            f"{proc.returncode}:\n{proc.stderr.decode(errors='replace')[-3000:]}"
        )
    return json.loads((out / "analysis_results.json").read_text())


@requires_flores
def test_results_file_is_strict_json(demo_results, tmp_path_factory):
    """No NaN or Infinity tokens, which Python writes but strict parsers reject.

    `gini.cost_ratio` used to be float('inf') when the minimum cost was zero,
    which json.dump serializes as the bare token Infinity. JavaScript's
    JSON.parse and most non-Python parsers refuse to read that file at all.
    """
    text = json.dumps(demo_results)
    json.loads(text, parse_constant=_reject_non_standard)


@requires_flores
def test_run_metadata_identifies_what_produced_the_file(demo_results):
    """A results file must be traceable to the code and inputs behind it."""
    meta = demo_results.get("run_metadata")
    assert meta is not None, "results file carries no provenance block"
    assert meta["package_version"]
    assert "tokenizers" in meta and meta["tokenizers"], "no tokenizers recorded"
    for name, entry in meta["tokenizers"].items():
        assert "class" in entry, f"{name} has no class recorded"
    assert meta["corpus"]["samples_per_lang"] == 10


@requires_flores
def test_merged_metrics_are_not_top_level_and_are_reachable(demo_results):
    """Each merged metric moved under its primary, and the move is recorded.

    A merge must relocate the data, not delete it. The slim writer rebuilds each
    per-tokenizer entry from a per-metric whitelist, so a folded field is only
    one missing whitelist entry away from vanishing silently.
    """
    for secondary, primary, field, _ in MERGES:
        if primary not in demo_results:
            continue
        assert secondary not in demo_results, (
            f"{secondary} should be reported under {primary}, not at top level"
        )
        per_tok = demo_results[primary]["per_tokenizer"]
        for tok, entry in per_tok.items():
            assert field in entry, (
                f"{primary}.{tok} is missing the merged field {field!r} "
                f"(from {secondary})"
            )
        merged = demo_results[primary].get("metadata", {}).get("merged_metrics", {})
        assert secondary in merged, (
            f"{primary} does not record that it absorbed {secondary}"
        )


@requires_flores
def test_every_metric_has_a_per_tokenizer_block(demo_results):
    """The documented shape is {metric: {per_tokenizer: {tok: ...}}}."""
    for name, block in demo_results.items():
        if name == "run_metadata":
            continue
        assert isinstance(block, dict), f"{name} is not a dict"
        assert "per_tokenizer" in block, f"{name} has no per_tokenizer block"


@pytest.fixture(scope="module")
def full_and_slim_results(tmp_path_factory):
    """Run the bundled demo with --save-full-results and return both files, parsed.

    A separate run from demo_results above: that fixture never passes
    --save-full-results, so it has no full file to compare against.

    --run-grouped-analysis is here because without it this fixture could not
    see the defect it exists to catch. The slim file invented two fields per
    single-language group, twenty in a real run, and no group of one language
    exists unless the run is grouped. The check was live and passing the whole
    time.
    """
    out = tmp_path_factory.mktemp("full_and_slim")
    proc = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--use-sample-data", "--samples-per-lang", "10",
         "--no-plots", "--no-code-ast", "--save-full-results",
         "--run-grouped-analysis",
         "--output-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            "demo run failed with exit "
            f"{proc.returncode}:\n{proc.stderr.decode(errors='replace')[-3000:]}"
        )
    slim = json.loads((out / "analysis_results.json").read_text())
    full = json.loads((out / "analysis_results_full.json").read_text())
    return slim, full


def _leaf_paths(obj, prefix=()):
    """Yield (key_path, value) for every leaf under obj, including empty dicts.

    A key path is the tuple of keys walked to reach a value that is not
    itself a dict, or is an empty dict (which has no further keys to walk
    into but is still a value a consumer can read).
    """
    if isinstance(obj, dict):
        if not obj:
            yield prefix, {}
        for key, value in obj.items():
            yield from _leaf_paths(value, prefix + (key,))
    elif isinstance(obj, list):
        yield prefix, obj
    else:
        yield prefix, obj


@requires_flores
def test_slim_file_is_a_strict_projection_of_the_full_file(full_and_slim_results):
    """Every value analysis_results.json publishes must exist, unchanged, at the
    same key path in analysis_results_full.json.

    This is the property normalize_results and select_results exist to
    guarantee: normalize_results renames and pivots raw results to their
    published key names without deleting anything, analysis_results_full.json
    is written from that output directly, and select_results (used by
    slim_results_for_json) only deletes keys from it to build
    analysis_results.json. Because the second pass never renames, every path
    the slim file publishes must already be present, under the same name and
    with the same value, in the output of the first pass.

    Before this split, a single function renamed keys while selecting them
    (overall to global, by_language to per_language, and so on), so a value
    read out of the slim file could not be looked up at the same path in the
    full file: the full file still used the raw, pre-rename names. A demo run
    measured 1022 slim paths with no counterpart in the full file. This test
    asserts that count is zero, so a future change to either pass cannot
    reintroduce a rename that only one of the two files sees.

    run_metadata is provenance, added to the slim file only, after both files
    are written from the same normalized results, and is excluded from the
    comparison for that reason.
    """
    slim, full = full_and_slim_results

    full_paths = dict(_leaf_paths(full))
    slim_paths = {
        path: value for path, value in _leaf_paths(slim) if path[0] != "run_metadata"
    }

    missing = [path for path in slim_paths if path not in full_paths]
    differing = [
        path for path in slim_paths
        if path in full_paths and full_paths[path] != slim_paths[path]
    ]

    assert not missing, (
        f"{len(missing)} slim path(s) have no counterpart in the full file, "
        f"starting with {missing[:5]}"
    )
    assert not differing, (
        f"{len(differing)} slim path(s) differ in value from the full file, "
        f"starting with {differing[:5]}"
    )


# Every key analysis_results_full.json holds that analysis_results.json is
# meant not to publish, per metric. 'per_tokenizer' lists names inside each
# tokenizer's entry, 'global' lists names inside that entry's global block, and
# 'metric' lists names beside per_tokenizer.
#
# Each entry records where the same information is still readable, because a
# drop is only acceptable when nothing is lost:
#   compression_rate.num_texts_analyzed  is the one exception. The published
#       global block gives total_units and total_tokens but no count of texts,
#       and no other published field carries it.
#   operator_isolation_rate.summary      equals per_tokenizer.<tok>.global,
#       which normalize_results copies from it.
#   the other four .summary blocks       hold the same numbers as the published
#       global block, sometimes under another name (reconstruction_fidelity's
#       texts_analyzed is global.count, its total_tokens_analyzed is
#       global.total_tokens), plus languages_analyzed, which is len(per_language).
#   domain_operator_counts               equals by_domain.<domain>.total_operators.
#   the renyi_<alpha> dicts and token_counts  are the untransposed source of
#       global.renyi_<alpha> and per_language.<lang>.{renyi_<alpha>,count}.
#   the gini extras                      are all derivable from per_language:
#       the two named languages are its argmin and argmax, and
#       sorted_language_costs is it sorted.
#   per_category                         is the only genuine detail loss, and
#       both branches that drop it say so.
#   sum, std_err, min and max            are stripped from every stats block by
#       _strip_stats and are all recoverable from the published mean, std and
#       count. OUTPUT.md names them.
#   the gini global block's min_cost and max_cost  are the smallest and largest
#       value of the published per_language costs.
#   the merge_vocabulary_usage examples and final_tokens_per_language  are
#       detail: ten and five example tokens, and each language's share of the
#       published total_final_tokens.
_DELIBERATELY_UNPUBLISHED = {
    "compression_rate": {"per_tokenizer": {"num_texts_analyzed"}},
    "fertility": {"global": {"sum", "std_err", "min", "max"}},
    "token_length": {"global": {"sum", "std_err", "min", "max"}},
    "numeric_magnitude_consistency": {"metric": {"summary"}},
    "operator_isolation_rate": {
        "per_tokenizer": {"per_category"},
        "metric": {"domain_operator_counts", "summary"},
    },
    "reconstruction_fidelity": {"metric": {"summary"}},
    "renyi_efficiency": {"per_tokenizer": {"token_counts"}},
    "three_digit_boundary_alignment": {"metric": {"summary"}},
    "tokenizer_fairness_gini": {
        "per_tokenizer": {
            "least_efficient_language",
            "most_efficient_language",
            "sorted_language_costs",
        },
        "global": {"min_cost", "max_cost"},
    },
    "utf8_token_integrity": {"metric": {"summary"}},
    "ast_boundary_alignment": {"per_tokenizer": {"per_category"}},
    "merge_vocabulary_usage": {
        "per_tokenizer": {
            "examples_scaffold",
            "examples_scaffold_byte_fragment",
            "final_tokens_per_language",
        },
    },
}


def _is_allowed_drop(metric, scope, key):
    """Whether *key* is a declared drop for *metric* at *scope*.

    renyi_efficiency publishes one dict per configured alpha, named
    renyi_<alpha>, so the alpha values come from the run's configuration rather
    than from a fixed list. Matching them by prefix keeps a configuration change
    from failing this test for a reason that has nothing to do with slimming.
    """
    if metric == "renyi_efficiency" and scope == "per_tokenizer":
        if key.startswith("renyi_"):
            return True
    return key in _DELIBERATELY_UNPUBLISHED.get(metric, {}).get(scope, set())


@requires_flores
def test_the_slim_file_drops_only_what_it_declares(full_and_slim_results):
    """Nothing may disappear from the published file without being declared.

    test_slim_file_is_a_strict_projection_of_the_full_file checks the other
    direction, and deleting a key is legal under that property, so it cannot see
    a field that stopped being published. That is not hypothetical: the
    tokenizer_fairness_gini branch of _select_tokenizer_entry enumerates the
    fields it copies, so per_line_normalization reached
    analysis_results_full.json and never analysis_results.json until the branch
    was changed to carry it.

    Most branches of _select_tokenizer_entry enumerate fields the same way, so
    the same thing happens to any field added to a metric that has a named
    branch. This test fails when that happens, and the fix is either to publish
    the field or to add it to _DELIBERATELY_UNPUBLISHED with a note saying where
    the same information is still readable.

    A key whose value in the full file is an empty container is not reported: it
    carries nothing to lose, and whether the selection keeps it depends on the
    corpus rather than on the code.

    This covers the metrics the bundled demo run produces. The code metrics are
    off in that run (--no-code-ast), so ast_boundary_alignment's entry below is
    declared from its branch rather than observed here.
    """
    slim, full = full_and_slim_results
    undeclared = []

    for metric, full_block in full.items():
        if metric == "run_metadata" or not isinstance(full_block, dict):
            continue
        slim_block = slim.get(metric, {})

        for key, value in full_block.items():
            if key == "per_tokenizer" or key in slim_block:
                continue
            if value in ({}, [], None):
                continue
            if not _is_allowed_drop(metric, "metric", key):
                undeclared.append(f"{metric}.{key}")

        full_toks = full_block.get("per_tokenizer") or {}
        slim_toks = slim_block.get("per_tokenizer") or {}
        for tok, full_entry in full_toks.items():
            if not isinstance(full_entry, dict):
                continue
            slim_entry = slim_toks.get(tok) or {}
            for key, value in full_entry.items():
                if key in slim_entry or value in ({}, [], None):
                    continue
                if not _is_allowed_drop(metric, "per_tokenizer", key):
                    undeclared.append(f"{metric}.per_tokenizer.{tok}.{key}")

            # The global block is checked one level in because OUTPUT.md calls
            # it the headline block, and because the branch for
            # tokenizer_fairness_gini rebuilds it field by field, which is the
            # same shape of hazard one level down.
            full_global = full_entry.get("global")
            slim_global = slim_entry.get("global")
            if isinstance(full_global, dict) and isinstance(slim_global, dict):
                for key, value in full_global.items():
                    if key in slim_global or value in ({}, [], None):
                        continue
                    if not _is_allowed_drop(metric, "global", key):
                        undeclared.append(
                            f"{metric}.per_tokenizer.{tok}.global.{key}"
                        )

    assert not undeclared, (
        f"{len(undeclared)} field(s) are in analysis_results_full.json, absent "
        "from analysis_results.json, and not declared in "
        "_DELIBERATELY_UNPUBLISHED: "
        f"{sorted(set(undeclared))}"
    )


def test_merge_is_a_no_op_when_the_secondary_is_absent():
    """A run that disabled a metric family must not trip the merge step."""
    results = {"compression_rate": {"per_tokenizer": {"t": {"global": {}}}}}
    merged = merge_redundant_metrics(results)
    assert "compression_rate" in merged
    assert merged["compression_rate"]["per_tokenizer"]["t"] == {"global": {}}


class TestNoDataIsNull:
    """Absent values must be null, never a number that reads as a measurement.

    0.0 is a legal value for nearly every rate here, so publishing it for the
    empty case made 'no UNK token exists' and 'no UNK was emitted' identical in
    the output, among others.
    """

    def test_safe_divide_returns_none_on_zero_denominator(self):
        assert BaseMetrics.safe_divide(1.0, 0.0) is None

    def test_safe_divide_still_divides(self):
        assert BaseMetrics.safe_divide(1.0, 4.0) == 0.25

    def test_safe_divide_honours_an_explicit_default(self):
        """A caller wanting a number for the empty case must say so."""
        assert BaseMetrics.safe_divide(1.0, 0.0, default=1.0) == 1.0

    def test_empty_stats_reports_none_not_zero(self):
        stats = BaseMetrics.empty_stats()
        for key in ("mean", "median", "std", "std_err", "min", "max"):
            assert stats[key] is None, f"{key} should be None for an empty sample"
        # Sample size is genuinely zero, so it stays numeric.
        assert stats["count"] == 0
        assert stats["sum"] == 0


@requires_flores
def test_single_corpus_does_not_report_perfect_fairness(tmp_path):
    """Gini is undefined for one corpus and must say so, not report 0.0.

    0.0 is the value for perfect equality, so the degenerate case was
    indistinguishable from the best possible result. It was also contradicted by
    the language_costs entry in the same object.
    """
    out = tmp_path / "mono"
    proc = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--tokenizer-config", "configs/sample_tokenizers.json",
         "--input", "parallel/eng_Latn.txt",
         "--samples-per-lang", "10", "--no-plots", "--no-code-ast",
         "--output-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-2000:]

    results = json.loads((out / "analysis_results.json").read_text())
    gini = results["tokenizer_fairness_gini"]["per_tokenizer"]["bpe"]["global"]
    assert gini["gini_coefficient"] is None
    assert gini["num_languages"] == 1
    assert "warning" in gini
    # mean_cost is a real measurement even with one corpus, so it stays a number.
    assert isinstance(gini["mean_cost"], float) and gini["mean_cost"] > 0


@requires_flores
def test_grouped_analysis_runs_after_the_metric_merge(tmp_path):
    """--run-grouped-analysis must read the merged results, not the old keys.

    run_analysis folds six metrics into the metric that owns the measurement
    before returning, and run_grouped_analysis then filters those base results
    by language group. It went on reading the pre-merge top-level key
    digit_split_variability, so the whole flag exited 1 with
    KeyError: 'digit_split_variability'. Nothing covered the flag, so the merge
    shipped with it broken.
    """
    out = tmp_path / "grouped"
    proc = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--use-sample-data", "--samples-per-lang", "5",
         "--run-grouped-analysis", "--no-plots", "--no-code-ast",
         "--output-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-3000:]


def test_a_nested_merged_block_is_filtered_to_the_group_languages():
    """The block the merge nested must be language-filtered like its parent.

    _filter_digit_boundary_results passes unrecognized keys through untouched.
    After the merge, split_variability is one of those keys and holds
    per-language numbers, so a language group inherited values computed over
    every language in the run.
    """
    from tokenizer_analysis.main import UnifiedTokenizerAnalyzer

    merged_shape = {'per_tokenizer': {'bpe': {
        'by_digit_length': {'2': {'eng_Latn': 0.5, 'arb_Arab': 0.6}},
        'overall': {'eng_Latn': 0.7, 'arb_Arab': 0.8},
        'split_variability': {
            'by_digit_length': {'2': {'eng_Latn': 0.1, 'arb_Arab': 0.3}},
            'by_bucket': {'short': {'eng_Latn': 1.0, 'arb_Arab': 2.0}, 'long': {}},
        },
    }}}
    filtered = UnifiedTokenizerAnalyzer._filter_digit_boundary_results(
        object.__new__(UnifiedTokenizerAnalyzer), merged_shape, ['eng_Latn']
    )
    nested = filtered['per_tokenizer']['bpe']['split_variability']
    assert list(nested['by_digit_length']['2']) == ['eng_Latn']
    assert list(nested['by_bucket']['short']) == ['eng_Latn']


@requires_flores
def test_tokenized_data_cache_can_be_replayed(tmp_path):
    """--save-tokenized-data then --tokenized-data-file must complete.

    The replay path never bound `tokenizer_configs`, and the last line of
    run_from_args reads it to build run_metadata, so every replay crashed with
    UnboundLocalError after computing every metric and wrote nothing. The
    provenance block broke the cache workflow, and neither had a test.
    """
    saved = tmp_path / "saved"
    save = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--use-sample-data", "--samples-per-lang", "5", "--no-plots",
         "--no-code-ast", "--save-tokenized-data", "--output-dir", str(saved)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    assert save.returncode == 0, save.stderr.decode(errors="replace")[-2000:]

    replayed = tmp_path / "replayed"
    replay = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--tokenized-data-file", str(saved / "tokenized_data.pkl"),
         "--tokenized-data-config", str(saved / "tokenized_data_config.json"),
         "--language-config", str(saved / "tokenized_data_language_config.json"),
         "--no-plots", "--no-code-ast", "--output-dir", str(replayed)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    assert replay.returncode == 0, replay.stderr.decode(errors="replace")[-2000:]

    first = json.loads((saved / "analysis_results.json").read_text())
    again = json.loads((replayed / "analysis_results.json").read_text())
    assert (first["fertility"]["per_tokenizer"]["bpe"]["global"]["mean"]
            == again["fertility"]["per_tokenizer"]["bpe"]["global"]["mean"])
    assert "run_metadata" in again


def test_run_metadata_records_the_flags_that_change_the_numbers():
    """A flag that narrows what is measured must be on the record.

    --filter-script-family, --filter-resource-level and --use-builtin-math-data
    all change the published values and all left run_metadata byte-identical,
    because the block named a hand-maintained list of interesting flags and
    those three were not on it. The block now diffs the parsed namespace against
    the parser's own defaults, so a flag added later is covered without anyone
    remembering to add it.
    """
    from tokenizer_analysis.cli.run_analysis import build_parser, _non_default_arguments

    base = build_parser().parse_args(["--use-sample-data"])
    assert _non_default_arguments(base) == {"use_sample_data": True}

    for flag, value in (
        ("--filter-script-family", "Latin"),
        ("--filter-resource-level", "high"),
    ):
        args = build_parser().parse_args(["--use-sample-data", flag, value])
        recorded = _non_default_arguments(args)
        assert value in recorded.values(), f"{flag} is absent from {recorded}"

    args = build_parser().parse_args(["--use-sample-data", "--use-builtin-math-data"])
    assert _non_default_arguments(args).get("use_builtin_math_data") is True

    # Where the output goes is not what was measured, so two runs differing only
    # in that must not look like different measurements.
    args = build_parser().parse_args(["--use-sample-data", "--output-dir", "/tmp/x",
                                      "--no-plots", "--save-full-results"])
    assert _non_default_arguments(args) == {"use_sample_data": True}


# ----------------------------------------------------------------------
# The schema contract, asserted over every metric rather than a sample.
# ----------------------------------------------------------------------

# Imported rather than restated, so the test cannot drift from the constant the
# metrics assign from.
from tokenizer_analysis.constants import AGGREGATION_LABELS as _ALLOWED_AGGREGATIONS


def _metrics(results):
    """Every metric block in a results file, by name."""
    return {
        name: block for name, block in results.items()
        if name != "run_metadata" and isinstance(block, dict)
    }


@requires_flores
def test_every_metric_publishes_a_global(demo_results):
    """No exemptions. A consumer reads one key for the headline value.

    Five metrics had none before 1.0.0, and one of them, trigram_entropy,
    published the same four numbers as flat global_* siblings instead, so a
    parser written against bigram_entropy silently found nothing. token_length
    and encoding_speed carry a global that duplicates an existing block, which
    is deliberate: an exception in the schema costs a reader more than a
    duplicated number does.
    """
    missing = []
    for name, block in _metrics(demo_results).items():
        for tok, entry in (block.get("per_tokenizer") or {}).items():
            if isinstance(entry, dict) and "global" not in entry:
                missing.append(f"{name}.per_tokenizer.{tok}")
    assert not missing, "no global block on: " + ", ".join(sorted(missing))


@requires_flores
def test_every_metric_declares_its_aggregation(demo_results):
    """Which average `global` reports, from a fixed set.

    global meant a ratio of sums in one metric, a mean of per-document ratios
    in another, an unweighted mean across languages in a third and a set union
    in a fourth, with nothing in the output saying which. On the bundled
    parallel corpus every language holds the same number of lines, so micro and
    macro agree and the difference is invisible until someone runs an unequal
    corpus.
    """
    bad = {}
    for name, block in _metrics(demo_results).items():
        label = (block.get("metadata") or {}).get("aggregation")
        if label not in _ALLOWED_AGGREGATIONS:
            bad[name] = label
    assert not bad, (
        "aggregation missing or not one of "
        f"{sorted(_ALLOWED_AGGREGATIONS)}: {bad}"
    )


@requires_flores
def test_every_per_language_entry_carries_a_count_or_says_why_not(demo_results):
    """So a consumer can re-derive the other weighting.

    A metric whose per-language entry is one language, so that the count would
    be 1 for every entry, says so in metadata.per_language_count instead. That
    is the only accepted reason to omit it: an entry that is simply silent is
    the failure this asserts against.
    """
    offenders = []
    for name, block in _metrics(demo_results).items():
        metadata = block.get("metadata") or {}
        if metadata.get("per_language_count"):
            continue
        for tok, entry in (block.get("per_tokenizer") or {}).items():
            per_lang = (entry or {}).get("per_language")
            if not isinstance(per_lang, dict):
                continue
            for lang, value in per_lang.items():
                if isinstance(value, dict) and "count" not in value:
                    offenders.append(f"{name}.{tok}.{lang}")
                    break
    assert not offenders, (
        "per_language entries with no count and no stated reason: "
        + ", ".join(sorted(offenders))
    )


@requires_flores
def test_every_metric_names_the_unit_its_count_is_in(demo_results):
    """A count is not interpretable without its unit.

    Documents, tokens, digit spans and AST nodes are not interchangeable, and
    the same field name carries all four across the file.
    """
    missing = [
        name for name, block in _metrics(demo_results).items()
        if not (block.get("metadata") or {}).get("count_unit")
    ]
    assert not missing, "no count_unit in metadata: " + ", ".join(sorted(missing))


def test_a_two_bucket_corpus_still_writes_strict_json(tmp_path):
    """The corpus shape that broke CI: `NaN` is not valid JSON.

    `scipy.stats.spearmanr` over two points returns a defined rho and an
    undefined p, because a rank correlation over two points has no significance
    level. `numeric_magnitude_consistency` wrote that NaN straight out, and
    `json.dump` renders it as the bare token `NaN`, which no strict parser
    accepts. The demo corpus happens to produce three or more digit-length
    buckets, so the existing strict-JSON test never saw it; a corpus whose
    numbers span two buckets does.

    Two layers are asserted here: the metric publishes null, and the serializer
    would have converted any remaining non-finite float anyway.
    """
    corpus = tmp_path / "eng_Latn.txt"
    # Numbers 0 to 199, so digit lengths 1, 2 and 3, which collapse to two
    # populated buckets after the short/long split.
    corpus.write_text(
        "\n".join(f"Item number {i} costs {i * 7} in total." for i in range(200)) + "\n"
    )
    languages = tmp_path / "languages.json"
    languages.write_text(json.dumps({
        "languages": {"eng_Latn": {"name": "English", "data_path": str(corpus)}},
    }))

    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--tokenizer-config", "configs/sample_tokenizers.json",
         "--language-config", str(languages), "--samples-per-lang", "50",
         "--no-plots", "--no-code-ast", "--output-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-2000:]

    # parse_constant fires on NaN, Infinity and -Infinity, the three tokens
    # json.dump will happily write and no other parser will read.
    json.loads((out / "analysis_results.json").read_text(),
               parse_constant=_reject_non_standard)


def test_the_serializer_converts_any_non_finite_float_to_null():
    """The backstop, independent of which metric produced the value."""
    from tokenizer_analysis.cli.run_analysis import _convert_for_json_public as convert

    converted = convert({
        "a": float("nan"), "b": float("inf"), "c": float("-inf"),
        "d": [1.0, float("nan")], "e": {"f": float("inf")}, "g": 0.5,
    })
    assert converted == {
        "a": None, "b": None, "c": None,
        "d": [1.0, None], "e": {"f": None}, "g": 0.5,
    }
    json.dumps(converted, allow_nan=False)


def test_cer_skipped_survives_slimming():
    """A null mean_cer is ambiguous without the flag that explains it.

    mean_cer and whitespace_fidelity are null both when the character error
    rate exceeded --cer-time-budget and when there was nothing to measure.
    cer_skipped is the only field that separates the two, and docs/METRICS.md
    the benchmark README both tell readers to consult it.

    It is written only when a tokenizer actually exceeds the budget, so a demo
    corpus small enough to finish never produces it. That is why this is a
    unit test over a hand-built results dict rather than an assertion on a demo
    run: a comparison against demo output cannot see this field at all, which
    is how the 1.0.2 schema refactor dropped it without any gate noticing.
    """
    from tokenizer_analysis.cli.run_analysis import slim_results_for_json

    results = {
        "reconstruction_fidelity": {
            "per_tokenizer": {
                "tok": {
                    "overall": {
                        "exact_match_rate": 0.03,
                        "mean_cer": None,
                        "whitespace_fidelity": None,
                        "count": 10,
                        "total_tokens": 100,
                    },
                    "by_domain": {},
                    "cer_skipped": True,
                }
            },
            "metadata": {"aggregation": "micro_pooled"},
        }
    }

    slimmed = slim_results_for_json(results)
    entry = slimmed["reconstruction_fidelity"]["per_tokenizer"]["tok"]
    assert entry.get("cer_skipped") is True, (
        "cer_skipped was dropped, so a reader cannot tell a skipped "
        "mean_cer from one that had nothing to measure"
    )
    assert entry["global"]["mean_cer"] is None


class TestProseBatchPairing:
    """A short batch must not silently shrink the prose corpus.

    RawTokenizationProvider pairs a batch with its texts through ``zip``, which
    stops at the shorter side. A backend returning fewer encodings than it was
    given therefore drops the trailing texts, and every metric downstream is
    computed over a corpus smaller than the one the run declared, with nothing
    in the output saying so. Nothing else in the repository covers this
    provider.
    """

    class _ShortBatch:
        """Encodes correctly, but returns one fewer result than it was given."""

        def __init__(self, drop=1):
            self._drop = drop

        def can_encode(self):
            return True

        def encode(self, text):
            return list(range(len(text.split())))

        def encode_with_offsets(self, text):
            return self.encode(text), None

        def encode_batch_with_offsets(self, texts):
            kept = list(texts)[: len(texts) - self._drop]
            return [(self.encode(t), None) for t in kept]

        def get_vocab_size(self):
            return 100

    def _provider(self, tokenizer):
        from tokenizer_analysis.core.input_providers import RawTokenizationProvider
        from tokenizer_analysis.core.input_types import InputSpecification

        return RawTokenizationProvider({
            "tok": InputSpecification(
                tokenizer=tokenizer,
                texts={"eng_Latn": ["one two", "three four", "five six"]},
            )
        })

    def test_a_short_batch_raises_naming_both_counts(self):
        with pytest.raises(ValueError) as excinfo:
            self._provider(self._ShortBatch()).get_tokenized_data()
        message = str(excinfo.value)
        assert "2 encodings" in message and "3 'eng_Latn' texts" in message
        assert "prose corpus" in message

    def test_a_complete_batch_still_pairs_every_text(self):
        """The guard must not reject a correct batch."""
        data = self._provider(self._ShortBatch(drop=0)).get_tokenized_data()
        assert [d.text for d in data["tok"]] == ["one two", "three four", "five six"]


def _gini_entry(per_line):
    """One tokenizer's raw tokenizer_fairness_gini entry, before normalizing."""
    return {
        "tokenizer_fairness_gini": {
            "per_tokenizer": {
                "tok": {
                    "gini_coefficient": 0.0976,
                    "mean_cost": 0.31,
                    "std_cost": 0.04,
                    "min_cost": 0.26,
                    "max_cost": 0.38,
                    "cost_ratio": 1.46,
                    "num_languages": 2,
                    "language_costs": {"eng_Latn": 0.26, "zho_Hans": 0.38},
                    "most_efficient_language": ["eng_Latn", 0.26],
                    "least_efficient_language": ["zho_Hans", 0.38],
                    "sorted_language_costs": [["eng_Latn", 0.26], ["zho_Hans", 0.38]],
                    "per_line_normalization": per_line,
                }
            },
            "metadata": {"aggregation": "macro_languages"},
        }
    }


class TestPerLineGiniSurvivesSlimming:
    """The per-line coefficient has to reach the file everything reads.

    render_report.py opens analysis_results.json, and
    test_docs_match_results.py resolves documented key paths against it, so a
    field that only reaches analysis_results_full.json is unreadable by both.
    The selection branch for this metric enumerates its fields rather than
    passing the entry through, so a new field is dropped unless it is added
    there. test_slim_file_is_a_strict_projection_of_the_full_file cannot catch
    that: deleting a key is exactly what slimming is allowed to do.
    """

    def test_the_block_reaches_the_slim_file_intact(self):
        from tokenizer_analysis.cli.run_analysis import slim_results_for_json

        # The real shape: lines_per_language is the single shared count, since
        # the block is published only when every language has the same one.
        per_line = {
            "gini_coefficient": 0.0494,
            "lines_per_language": 250,
            "mean_cost": 47.2,
            "cost_ratio": 1.21,
            "language_costs": {"eng_Latn": 42.8, "zho_Hans": 51.6},
            "num_languages": 2,
        }
        slimmed = slim_results_for_json(_gini_entry(per_line))
        entry = slimmed["tokenizer_fairness_gini"]["per_tokenizer"]["tok"]

        assert entry.get("per_line_normalization") == per_line
        # The two coefficients differ, so a test cannot pass by finding the
        # per-byte number under the per-line key.
        assert entry["global"]["gini_coefficient"] == 0.0976
        assert entry["per_line_normalization"]["gini_coefficient"] == 0.0494

    def test_an_absent_block_is_published_as_null_not_omitted(self):
        """Unequal line counts are a result, not a missing measurement.

        The metric writes None rather than skipping the key, so a consumer can
        tell "this corpus is not line-parallel" from "this file predates the
        field".
        """
        from tokenizer_analysis.cli.run_analysis import slim_results_for_json

        slimmed = slim_results_for_json(_gini_entry(None))
        entry = slimmed["tokenizer_fairness_gini"]["per_tokenizer"]["tok"]

        assert "per_line_normalization" in entry
        assert entry["per_line_normalization"] is None


@pytest.fixture(scope="module")
def degenerate_run(tmp_path_factory):
    """A corpus too small for most metrics to have anything to measure.

    One short document and two tokenizers.  Bigram and trigram contexts do not
    clear their occurrence thresholds, there is one language so cross-language
    dispersion is undefined, and several counts are zero.  This is the shape
    that produced the 1.0.2 defects: a value that could not be computed was
    published as 0.0 or 1.0, and one path raised TypeError instead.
    """
    work = tmp_path_factory.mktemp("degenerate")
    corpus = work / "tiny.txt"
    corpus.write_text("The cat sat.\n", encoding="utf-8")

    toks = work / "toks.json"
    toks.write_text(json.dumps({
        "bpe": {"class": "huggingface", "path": "tokenizers/bpe.json"},
        "unigramlm": {"class": "huggingface", "path": "tokenizers/unigramlm.json"},
    }), encoding="utf-8")

    out = work / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--tokenizer-config", str(toks), "--input", str(corpus),
         "--no-plots", "--no-code-ast", "--output-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            "a corpus with nothing to measure must still complete; exit "
            f"{proc.returncode}:\n{proc.stderr.decode(errors='replace')[-3000:]}"
        )
    return json.loads((out / "analysis_results.json").read_text())


def test_a_corpus_with_nothing_to_measure_publishes_null_not_a_number(degenerate_run):
    """A count of zero may not sit beside a number that reads as a measurement.

    This is the invariant behind every 1.0.2 metric fix.  Where a global block
    reports it measured nothing, every rate and mean in that block has to be
    null: a bigram entropy of 0.0 means every context has exactly one
    successor, a compression rate of 1.0 means one unit per token, and a
    completeness rate of 1.0 means every token was well formed.  None of those
    may stand in for "there was nothing to measure".

    Fields that count things (count, total_tokens, used_tokens and the like)
    stay numeric: zero of something is a true statement about the sample.
    """
    # A field whose zero means "there was nothing to measure". Not every
    # counter qualifies: trailing_incomplete of 0 means no token was truncated,
    # a real finding, and gating on it would have flagged a completeness rate
    # of 1.0 measured over four sound tokens.
    denominators = (
        "count", "total_tokens", "total_units", "total_content_tokens",
        "total_bigrams", "total_trigrams", "total_operators",
        "total_indented_lines", "num_samples", "num_languages", "vocab_size",
        # Zero qualifying context types is the entropy metrics' way of saying
        # nothing was measured; their total_bigrams stays non-zero.
        "types_evaluated",
    )
    offenders = []
    inspected = []

    for name, block in _metrics(degenerate_run).items():
        for tok, entry in (block.get("per_tokenizer") or {}).items():
            glob = (entry or {}).get("global")
            if not isinstance(glob, dict):
                continue
            # Gating on `count == 0` alone skipped seven metric families that
            # put no count in their global at all, including every one this
            # docstring names.
            if not any(glob.get(k) == 0 for k in denominators if k in glob):
                continue
            inspected.append(f"{name}.{tok}")
            for field, value in glob.items():
                # Counts are integers and a zero count is a true statement
                # about the sample. Rates, means and correlations are floats,
                # and those are what must not be invented.
                if isinstance(value, float):
                    offenders.append(
                        f"{name}.{tok}.global.{field} = {value}, "
                        f"beside a zero denominator"
                    )

    # Without this the test can silently inspect nothing, which is how its
    # first version passed while skipping every metric it named.
    assert inspected, (
        "this test inspected no block with a zero denominator, so it asserted "
        "nothing; make the corpus smaller or widen `denominators`"
    )
    assert not offenders, (
        "a measured-looking number published beside a zero denominator:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_run_metadata_records_the_inputs_that_decide_the_numbers(degenerate_run):
    """Provenance the 1.0.1 file did not carry.

    A results file has to say which corpus and which tokenizers produced it,
    not only which config named them.  Two runs over different snapshots of a
    corpus under one config were previously indistinguishable, and a Hub-side
    retokenization would move every number with nothing recording it.
    """
    meta = degenerate_run["run_metadata"]

    assert meta.get("timestamp_utc"), "no timestamp, so two runs of one commit are indistinguishable"
    assert meta["timestamp_utc"].endswith("+00:00"), "timestamp is not UTC"

    for name, entry in meta["tokenizers"].items():
        assert entry.get("sha256_16") or entry.get("hub_revision"), (
            f"tokenizer {name} is recorded by path alone, so a change to it "
            "would move every number with nothing saying so"
        )

    digest = meta["corpus"]["digest"]
    assert digest.get("n_languages"), "no corpus digest"
    for lang, entry in digest["files"].items():
        assert entry.get("sha256_16"), f"corpus {lang} has no hash"
        assert entry.get("bytes"), f"corpus {lang} has no byte count"


@requires_flores
def test_corpus_paths_under_the_working_directory_are_recorded_relative(demo_results):
    """So a committed results file does not name whoever produced it.

    The digest records a path per language. A corpus inside the working
    directory is recorded relative to it; one outside keeps its absolute path,
    because there is nothing to make it relative to. The benchmark reads
    parallel/ from the repository root, so its committed results file must
    carry short paths rather than a home directory.
    """
    files = demo_results["run_metadata"]["corpus"]["digest"]["files"]
    absolute = {lang: e["path"] for lang, e in files.items() if e["path"].startswith("/")}
    assert not absolute, (
        "corpus paths under the working directory recorded as absolute: "
        + ", ".join(f"{k} -> {v}" for k, v in sorted(absolute.items()))
    )


@pytest.fixture(scope="module")
def default_code_config_results(tmp_path_factory):
    """A run with no --code-ast-config, which is the documented default.

    `_resolve_code_ast_config` returns `{}` for this, meaning "use synthetic
    samples", where `None` would mean "disabled". The two are read differently
    on purpose, and the mechanism changed when the corpora moved onto the input
    provider. It is now: `resolve_code_corpus` tests truthiness, so `{}` gives
    the registered corpus the bundled samples with `synthetic=True`;
    `BasicTokenizationMetrics.__init__` reads that flag and takes no code
    domain from a synthetic corpus; and `UnifiedTokenizerAnalyzer` tests
    `is not None` before building the AST metrics, so they are built and score
    the bundled samples.

    The asymmetry is the same one; only where it is decided has moved.
    """
    out = tmp_path_factory.mktemp("default_code_config")
    proc = subprocess.run(
        [sys.executable, "-m", "tokenizer_analysis.cli.run_analysis",
         "--use-sample-data", "--samples-per-lang", "5",
         "--use-builtin-math-data", "--cer-time-budget", "0",
         "--no-plots", "--output-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            "default-config run failed with exit "
            f"{proc.returncode}:\n{proc.stderr.decode(errors='replace')[-3000:]}"
        )
    return json.loads((out / "analysis_results.json").read_text())


@requires_flores
class TestTheDefaultCodeConfigurationIsAsymmetric:
    """Omitting --code-ast-config gives the code metrics synthetic data and
    reconstruction fidelity no code domain at all.

    That asymmetry is deliberate and was confirmed as intended on 2026-08-19.
    It is pinned here because it emerges from two different truth tests on the
    same value rather than from anything stated in one place, so a refactor that
    resolves "which corpus" once, in one rule, would quietly give reconstruction
    fidelity a code domain it has never had. No test covered this before.
    """

    def _one_tokenizer(self, block):
        per_tok = block["per_tokenizer"]
        return per_tok[sorted(per_tok)[0]]

    def test_reconstruction_fidelity_has_no_code_domain(
            self, default_code_config_results):
        entry = self._one_tokenizer(
            default_code_config_results["reconstruction_fidelity"])
        domains = sorted(entry.get("per_domain") or {})
        assert domains, "expected some domains, so the assertion below means something"
        assert not [d for d in domains if d.startswith("code")], domains
        # math is present, so this is specific to code rather than a run in
        # which no derived corpus reached the metric at all.
        assert "math" in domains, domains

    def test_the_code_metrics_do_get_synthetic_code(
            self, default_code_config_results):
        entry = self._one_tokenizer(
            default_code_config_results["ast_boundary_alignment"])
        languages = sorted(entry.get("per_language") or {})
        assert "python" in languages, languages
        assert len(languages) > 3, languages


class TestBaseResultsSemanticsInGroupedAnalysis:
    """base_results={} means "nothing precomputed"; truthy-without-the-key
    means "the base run disabled digit metrics".

    The distinction regressed once (the sixth review's base_results fix) and
    shipped without a test: mutating ``elif base_results`` to
    ``elif base_results is not None`` in run_grouped_analysis passed the
    whole suite (RELEASE_AUDIT Q35.4). These two tests are the guard.
    """

    @pytest.fixture(scope="class")
    def analyzer(self, tmp_path_factory):
        from tokenizer_analysis.main import create_analyzer_from_raw_inputs
        from tokenizer_analysis.config.language_metadata import LanguageMetadata

        cfg = tmp_path_factory.mktemp("grouped") / "langs.json"
        cfg.write_text(json.dumps({
            "languages": {
                "eng_Latn": {"name": "English", "script_family": "Latin",
                             "resource_level": "high"},
                "rus_Cyrl": {"name": "Russian", "script_family": "Cyrillic",
                             "resource_level": "medium"},
            },
            "analysis_groups": {"script_family": {"Latin": ["eng_Latn"],
                                                  "Cyrillic": ["rus_Cyrl"]}},
        }))
        return create_analyzer_from_raw_inputs(
            tokenizer_configs={"bpe": {"class": "huggingface",
                                       "path": "tokenizers/bpe.json"}},
            language_texts={
                "eng_Latn": ["The total was 12 and 345.", "Sum 6789 next."],
                "rus_Cyrl": ["Line 12 and 345 here.", "Total 6789."],
            },
            language_metadata=LanguageMetadata(str(cfg)),
            plot_save_dir=str(tmp_path_factory.mktemp("plots")),
            code_ast_config=None,
        )

    def test_an_empty_dict_means_nothing_precomputed(self, analyzer):
        groups = analyzer.run_grouped_analysis(
            group_by=["script_family"], save_plots=False, base_results={},
            include_reconstruction=False,
        )
        latin = groups["script_family"]["Latin"]
        assert "three_digit_boundary_alignment" in latin
        assert "numeric_magnitude_consistency" in latin
        assert "operator_isolation_rate" in latin

    def test_a_base_run_without_digit_metrics_reports_none(self, analyzer):
        groups = analyzer.run_grouped_analysis(
            group_by=["script_family"], save_plots=False,
            base_results={"fertility": {"per_tokenizer": {}}},
            include_reconstruction=False,
        )
        latin = groups["script_family"]["Latin"]
        assert "three_digit_boundary_alignment" not in latin
        assert "numeric_magnitude_consistency" not in latin
        assert "operator_isolation_rate" not in latin


    def test_utf8_is_computed_when_nothing_was_precomputed(self, analyzer):
        """The gap: with no base results there was no UTF-8 block at all."""
        groups = analyzer.run_grouped_analysis(
            group_by=["script_family"], save_plots=False, base_results={},
            include_reconstruction=False,
        )
        assert "utf8_token_integrity" in groups["script_family"]["Latin"]

    def test_utf8_is_not_resurrected_when_the_caller_turned_it_off(self, analyzer):
        """The half a bare "compute it either way" would break.

        A base run that passed --no-utf8-integrity produces base results with
        no UTF-8 key. That missing key is the only signal the grouped path
        gets, so computing unconditionally would switch the metric back on for
        someone who asked for it off. The digit metrics next door already
        handle this with the same three cases.
        """
        groups = analyzer.run_grouped_analysis(
            group_by=["script_family"], save_plots=False,
            base_results={"fertility": {"per_tokenizer": {}}},
            include_reconstruction=False,
        )
        assert "utf8_token_integrity" not in groups["script_family"]["Latin"]

    def test_utf8_is_filtered_when_the_base_run_had_it(self, analyzer):
        groups = analyzer.run_grouped_analysis(
            group_by=["script_family"], save_plots=False,
            base_results={"utf8_token_integrity": {"per_tokenizer": {}}},
            include_reconstruction=False,
        )
        assert "utf8_token_integrity" in groups["script_family"]["Latin"]


class TestAGroupBlockCarriesNoWholeCorpusStatistics:
    """A group block must hold group numbers or nothing, never corpus ones.

    _filter_digit_boundary_results used to copy the base run's summary into
    every group (byte-identical across groups, pooling the whole corpus) and
    pass the magnitude scaling fit through unfiltered into the slim results
    file; _filter_operator_results copied metadata that said "Code and math
    always run" beside empty blocks; _filter_morphscore_results invented a
    top-level summary whose counters summed over tokenizers. All four were
    published by a default --run-grouped-analysis run (RELEASE_AUDIT Q35.2
    R1/R2).
    """

    @staticmethod
    def _analyzer():
        from tokenizer_analysis.main import UnifiedTokenizerAnalyzer
        return object.__new__(UnifiedTokenizerAnalyzer)

    def test_the_group_summary_is_recomputed_from_the_group_languages(self):
        base = {
            "per_tokenizer": {"bpe": {
                "overall": {
                    "eng_Latn": {"mean_f1": 0.9, "mean_precision": 0.8,
                                 "mean_recall": 0.7, "count": 4},
                    "arb_Arab": {"mean_f1": 0.1, "mean_precision": 0.2,
                                 "mean_recall": 0.3, "count": 6},
                },
            }},
            "summary": {"bpe": {"avg_f1": 0.42, "numbers_analyzed": 10,
                                "languages_analyzed": 2}},
        }
        filtered = self._analyzer()._filter_digit_boundary_results(
            base, ["eng_Latn"]
        )
        summary = filtered["summary"]["bpe"]
        assert summary["avg_f1"] == pytest.approx(0.9)
        assert summary["numbers_analyzed"] == 4
        assert summary["languages_analyzed"] == 1
        assert summary is not base["summary"]["bpe"]

        # Both languages: the re-aggregation must weight each language's mean
        # by its count, reproducing the base run's pooled figure. An
        # unweighted mean of the two per-language means (0.5) passed every
        # other test in the suite, so this assertion is the one guard on the
        # weighting.
        both = self._analyzer()._filter_digit_boundary_results(
            base, ["eng_Latn", "arb_Arab"]
        )["summary"]["bpe"]
        assert both["avg_f1"] == pytest.approx(0.42)
        assert both["avg_precision"] == pytest.approx(0.44)
        assert both["avg_recall"] == pytest.approx(0.46)
        assert both["numbers_analyzed"] == 10
        assert both["languages_analyzed"] == 2

    def test_a_group_with_no_numbers_publishes_no_summary(self):
        """The math-corpus case: digit metrics measured a corpus that belongs
        to no language group, so the group reports nothing rather than the
        whole corpus."""
        base = {
            "per_tokenizer": {"bpe": {
                "overall": {"math": {"mean_f1": 0.6, "count": 627}},
            }},
            "summary": {"bpe": {"avg_f1": 0.6, "numbers_analyzed": 627,
                                "languages_analyzed": 1}},
        }
        filtered = self._analyzer()._filter_digit_boundary_results(
            base, ["eng_Latn"]
        )
        assert filtered["summary"] == {}

    def test_the_corpus_scaling_fit_is_not_copied_into_a_group(self):
        from tokenizer_analysis.metrics.math import magnitude_metadata
        base = {
            "per_tokenizer": {"bpe": {
                "overall": {"eng_Latn": {"mean_fertility": 0.5, "count": 3}},
                "scaling": {"spearman_rho": -0.59, "per_bucket": {"1": {}}},
            }},
            "metadata": magnitude_metadata(),
        }
        filtered = self._analyzer()._filter_digit_boundary_results(
            base, ["eng_Latn"], grouped_metadata=magnitude_metadata(grouped=True)
        )
        assert "scaling" not in filtered["per_tokenizer"]["bpe"]
        assert filtered["metadata"] == magnitude_metadata(grouped=True)
        assert "does not carry one" in filtered["metadata"]["description"]

    def test_every_bucket_key_survives_filtering(self):
        base = {"per_tokenizer": {"bpe": {
            "by_bucket": {"short": {"eng_Latn": {"mean_f1": 1.0, "count": 2}},
                          "long": {"arb_Arab": {"mean_f1": 0.5, "count": 1}}},
        }}}
        filtered = self._analyzer()._filter_digit_boundary_results(
            base, ["eng_Latn"]
        )
        buckets = filtered["per_tokenizer"]["bpe"]["by_bucket"]
        assert set(buckets) == {"short", "long"}
        assert buckets["long"] == {}

    def test_the_grouped_operator_metadata_describes_the_filtered_block(self):
        from tokenizer_analysis.metrics.math import operator_metadata
        base = {
            "per_tokenizer": {"bpe": {"by_language": {
                "eng_Latn": {"isolated": 1, "total": 2,
                             "compound_ok": 0, "compound_total": 0},
            }}},
            "metadata": operator_metadata(include_code_math=True),
        }
        filtered = self._analyzer()._filter_operator_results(
            base, ["eng_Latn"]
        )
        description = filtered["metadata"]["description"]
        assert "always run" not in description
        assert filtered["metadata"] == operator_metadata(
            include_code_math=False, filtered=True
        )

    def test_a_group_gets_no_invented_morphscore_summary(self):
        row = {"morphscore_recall": 0.5, "morphscore_precision": 0.6,
               "micro_f1": 0.55, "macro_f1": 0.54, "num_samples": 5}
        base = {
            "per_tokenizer": {"bpe": {
                "per_language": {"eng_Latn": dict(row), "arb_Arab": dict(row)},
                "summary": {"avg_morphscore_recall": 0.5,
                            "languages_evaluated": 2, "total_samples": 10},
            }},
            "metadata": {"description": "x"},
        }
        filtered = self._analyzer()._filter_morphscore_results(
            base, ["eng_Latn"]
        )
        assert "summary" not in filtered
        per_tok = filtered["per_tokenizer"]["bpe"]["summary"]
        assert per_tok["languages_evaluated"] == 1


def _FERTILITY_METADATA():
    """fertility's published metadata, without needing a corpus."""
    from tokenizer_analysis.metrics.basic import BasicTokenizationMetrics
    from tokenizer_analysis.core.input_types import InputSpecification
    from tokenizer_analysis.core.input_providers import RawTokenizationProvider

    class _T:
        def can_encode(self): return True
        def encode(self, t): return [1]
        def encode_batch_with_offsets(self, ts): return [([1], [(0, 1)]) for _ in ts]
        def encode_with_offsets(self, t): return [1], [(0, 1)]
        def get_vocab_size(self): return 10

    p = RawTokenizationProvider({"t": InputSpecification(tokenizer=_T(), texts={"eng_Latn": ["a"]})})
    m = BasicTokenizationMetrics(p)
    return m.compute_fertility_analysis(p.get_tokenized_data())["fertility"]["metadata"]


def _TOKEN_LENGTH_METADATA():
    from tokenizer_analysis.metrics.basic import BasicTokenizationMetrics
    from tokenizer_analysis.core.input_types import InputSpecification
    from tokenizer_analysis.core.input_providers import RawTokenizationProvider

    class _T:
        def can_encode(self): return True
        def encode(self, t): return [1]
        def encode_batch_with_offsets(self, ts): return [([1], [(0, 1)]) for _ in ts]
        def encode_with_offsets(self, t): return [1], [(0, 1)]
        def get_vocab_size(self): return 10

    p = RawTokenizationProvider({"t": InputSpecification(tokenizer=_T(), texts={"eng_Latn": ["a"]})})
    m = BasicTokenizationMetrics(p)
    return m.compute_token_length_analysis(p.get_tokenized_data())["token_length"]["metadata"]


def _DIGIT_METRIC_METADATA():
    """The two digit blocks' metadata, straight from the builders.

    Read from the built results rather than from the metadata helpers, so a
    builder that stops calling its helper is caught too.
    """
    from tokenizer_analysis.metrics.math import DigitBoundaryMetrics
    # object.__new__: both builders read only their argument, and a real
    # construction would need a provider and a corpus for a metadata read.
    inst = object.__new__(DigitBoundaryMetrics)
    inst.tokenizer_names = []
    align = DigitBoundaryMetrics._build_alignment_results(inst, {})
    mag = DigitBoundaryMetrics._build_magnitude_results(inst, {})
    return {
        "three_digit_boundary_alignment": align["metadata"],
        "numeric_magnitude_consistency": mag["metadata"],
    }


class TestTheAggregationLabelNamesWhatTheGlobalBlockComputes:
    """A label saying micro_pooled over a mean of per-item ratios is a wrong
    number in the metadata: docs/OUTPUT.md invites a consumer to re-derive the
    other weighting from the per-language counts, and the two differ by up to
    18.7% on the committed benchmark.

    test_every_metric_declares_its_aggregation cannot catch this. It imports
    AGGREGATION_LABELS, so adding a fifth member makes both the right and the
    wrong label pass for every metric.
    """

    def test_every_mean_of_ratios_metric_declares_it(self):
        """The plan's named mutation for this item is "leave one of the four
        labelled micro_pooled", and until this test the whole suite stayed
        green with three_digit_boundary_alignment mislabelled: nothing
        asserted its aggregation at all.

        A label assertion is the floor, not the ceiling. It catches the
        relabel-three-miss-one error, which is the one the plan predicted,
        because the two digit metrics come from one helper and one
        accumulator. The value tests in test_basic.py carry the heavier
        claim that the label names what is computed.
        """
        from tokenizer_analysis.constants import (
            AGGREGATION_MEAN_OF_RATIOS, AGGREGATION_MICRO_POOLED,
        )
        from tokenizer_analysis.metrics.basic import BasicTokenizationMetrics
        from tokenizer_analysis.metrics.math import (
            DigitBoundaryMetrics, magnitude_metadata, operator_metadata,
        )

        declared = {}
        for name, block in _DIGIT_METRIC_METADATA().items():
            declared[name] = block["aggregation"]
        declared["fertility"] = _FERTILITY_METADATA()["aggregation"]
        declared["token_length"] = _TOKEN_LENGTH_METADATA()["aggregation"]

        assert declared == {
            "three_digit_boundary_alignment": AGGREGATION_MEAN_OF_RATIOS,
            "numeric_magnitude_consistency": AGGREGATION_MEAN_OF_RATIOS,
            "fertility": AGGREGATION_MEAN_OF_RATIOS,
            "token_length": AGGREGATION_MEAN_OF_RATIOS,
        }, declared
        # The operator rate really is pooled: isolated over total.
        assert operator_metadata()["aggregation"] == AGGREGATION_MICRO_POOLED

    def test_the_two_digit_metrics_built_by_one_helper_agree(self):
        """Relabelling one and missing the other is the plausible error:
        three_digit_boundary_alignment and numeric_magnitude_consistency come
        from the same accumulator, and both average per-number ratios.
        """
        from tokenizer_analysis.constants import AGGREGATION_MEAN_OF_RATIOS
        from tokenizer_analysis.metrics.math import (
            magnitude_metadata, operator_metadata,
        )

        assert magnitude_metadata()["aggregation"] == AGGREGATION_MEAN_OF_RATIOS
        # The operator rate really is pooled: isolated over total.
        assert operator_metadata()["aggregation"] != AGGREGATION_MEAN_OF_RATIOS


class TestOneDefinitionOfAPublishedDomainKey:
    """Two files published the same domain keys, one through a shared function
    and one by writing the strings out by hand in five places. They agreed, and
    nothing kept them agreeing.

    The assertions below are the literal expected strings. Asserting that the
    two places agree with each other would pass for a format that is uniformly
    wrong, which is the shape of a mistake this project has made before.
    """

    def test_each_domain_produces_its_documented_key(self):
        from tokenizer_analysis.core.input_types import (
            CODE_CORPUS, MATH_CORPUS, PROSE_CORPUS, published_language_key,
        )

        assert published_language_key(PROSE_CORPUS, "eng_Latn") == "eng_Latn"
        assert published_language_key(CODE_CORPUS, "python") == "code_python"
        assert published_language_key(MATH_CORPUS, "math") == "math"
        # The label is ignored for maths: there is one maths domain, and the
        # label inside it is not part of the published key.
        assert published_language_key(MATH_CORPUS, "anything") == "math"

    def test_a_key_can_be_recognised_as_belonging_to_a_corpus(self):
        """The two places that read these keys back tested `startswith("code_")
        or == "math"` by hand. That is the same format knowledge as building
        the key, so it lives with it.
        """
        from tokenizer_analysis.core.input_types import is_corpus_domain_key

        assert is_corpus_domain_key("code_python") is True
        assert is_corpus_domain_key("math") is True
        assert is_corpus_domain_key("eng_Latn") is False
        assert is_corpus_domain_key("rus_Cyrl") is False

    def test_building_and_recognising_agree(self):
        """Not a substitute for the literal assertions above: this only checks
        the two halves are consistent, which they would be if both were wrong.
        """
        from tokenizer_analysis.core.input_types import (
            CODE_CORPUS, MATH_CORPUS, PROSE_CORPUS,
            is_corpus_domain_key, published_language_key,
        )

        for domain, lang in ((CODE_CORPUS, "rust"), (MATH_CORPUS, "math")):
            assert is_corpus_domain_key(published_language_key(domain, lang))
        assert not is_corpus_domain_key(
            published_language_key(PROSE_CORPUS, "deu_Latn"))
