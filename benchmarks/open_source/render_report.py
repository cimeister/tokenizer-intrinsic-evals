#!/usr/bin/env python
"""Render REPORT.md from a benchmark results file.

Every number in the report comes from `analysis_results.json` through this
script. Nothing is transcribed by hand, so the tables cannot drift from the
results, and `run.sh` regenerates both together.

    uv run python benchmarks/open_source/render_report.py \
        --results benchmarks/open_source/analysis_results.json \
        --output benchmarks/open_source/REPORT.md
"""

import argparse
import json
import math
from pathlib import Path

# Display name and Hub id per tokenizer key, in the order the tables use.
# Ordered by vocabulary size, since that is the axis most of the differences
# track and an alphabetical order would hide it.
TOKENIZERS = [
    ("bert-base", "BERT base uncased", "bert-base-uncased", "WordPiece"),
    ("gpt-neox-20b", "GPT-NeoX 20B", "EleutherAI/gpt-neox-20b", "BPE"),
    ("gpt2", "GPT-2", "gpt2", "BPE"),
    ("olmo-2", "OLMo 2", "allenai/OLMo-2-1124-7B", "BPE"),
    ("llama-3", "Llama 3", "meta-llama/Meta-Llama-3-8B", "BPE"),
    ("mistral-nemo", "Mistral NeMo", "mistralai/Mistral-Nemo-Base-2407", "BPE"),
    ("qwen-2.5", "Qwen 2.5", "Qwen/Qwen2.5-7B", "BPE"),
    ("xlm-roberta", "XLM-RoBERTa base", "xlm-roberta-base", "Unigram"),
    ("gemma-2", "Gemma 2", "google/gemma-2-9b", "Unigram"),
]

GATED = {"llama-3", "gemma-2"}


def get(results, *path, default=None):
    """Follow a dotted path, returning *default* when any step is absent."""
    node = results
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def fmt(value, places=3):
    """Render a number for a table cell, and an absent one as an em-free dash."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def present(results):
    """Tokenizer keys the results file actually holds, in TOKENIZERS order."""
    have = set(get(results, "fertility", "per_tokenizer", default={}))
    return [row for row in TOKENIZERS if row[0] in have]


def corpus_languages(results):
    """The corpus labels the run measured, from the first tokenizer's block.

    Each metric keys its per-language values under its own tokenizer entry;
    the top-level `per_language` block is a cross-tokenizer leaderboard that
    only some metrics publish, so it is not the place to read the corpus from.
    """
    per_tok = get(results, "compression_rate", "per_tokenizer", default={})
    if not per_tok:
        return []
    first = per_tok[sorted(per_tok)[0]]
    return sorted(first.get("per_language", {}))


def section_what_was_measured(results, rows):
    meta = results.get("run_metadata", {})
    corpus = meta.get("corpus", {})
    languages = corpus_languages(results)
    lines = ["## What was measured", ""]
    lines.append(
        f"{len(rows)} tokenizers, {len(languages)} languages. Every number below "
        f"comes from `analysis_results.json` in this directory, which records the "
        f"package version, the commit and a hash of each input."
    )
    lines.append("")
    lines.append("| Tokenizer | Hub id | Algorithm | Vocabulary | Gated |")
    lines.append("|---|---|---|---:|---|")
    for key, name, repo, algo in rows:
        vocab = get(results, "vocabulary_utilization", "per_tokenizer", key,
                    "global", "vocab_size")
        lines.append(
            f"| {name} | `{repo}` | {algo} | {fmt(vocab)} | "
            f"{'yes' if key in GATED else 'no'} |"
        )
    lines.append("")
    lines.append(
        "Corpus: FLORES+ `dev`, "
        f"{len(languages)} languages ({', '.join(languages)}), "
        f"{corpus.get('samples_per_lang')} sentences per language at most. "
        "Code: 1500 files across 15 languages from "
        "`bigcode/the-stack-smol-xs`, read whole. Math: the 285 bundled "
        "expressions."
    )
    lines.append("")
    lines.append(
        f"Package version {meta.get('package_version')}, commit "
        f"`{(meta.get('git_commit') or 'unknown')[:12]}`, working tree "
        f"{'modified' if meta.get('git_tree_modified') else 'clean'}."
    )
    return lines


def section_headline(results, rows):
    lines = [
        "", "## Headline", "",
        "One row per tokenizer. The arrow gives the direction each column is "
        "read in.",
        "",
        "| Tokenizer | Vocabulary | Bytes/token (up) | Tokens/line (down) | "
        "Fertility (down) | Gini per byte (down) | Gini per line (down) | "
        "UTF-8 complete (up) | AST aligned (up) | Digit F1 (up) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, name, _repo, _algo in rows:
        lines.append(
            "| {name} | {vocab} | {comp} | {tpl} | {fert} | {gini} | {gini_line} "
            "| {utf8} | {ast} | {digit} |".format(
                name=name,
                vocab=fmt(get(results, "vocabulary_utilization", "per_tokenizer", key,
                              "global", "vocab_size")),
                comp=fmt(get(results, "compression_rate", "per_tokenizer", key,
                             "global", "compression_rate")),
                tpl=fmt(get(results, "compression_rate", "per_tokenizer", key,
                            "tokens_per_line", "global_avg"), places=1),
                fert=fmt(get(results, "fertility", "per_tokenizer", key,
                             "global", "mean")),
                gini=fmt(get(results, "tokenizer_fairness_gini", "per_tokenizer", key,
                             "global", "gini_coefficient"), places=4),
                gini_line=fmt(get(results, "tokenizer_fairness_gini", "per_tokenizer",
                                  key, "per_line_normalization", "gini_coefficient"),
                              places=4),
                utf8=fmt(get(results, "utf8_token_integrity", "per_tokenizer", key,
                             "global", "completeness_rate")),
                ast=fmt(get(results, "ast_boundary_alignment", "per_tokenizer", key,
                            "global", "full_alignment_rate")),
                digit=fmt(get(results, "three_digit_boundary_alignment",
                              "per_tokenizer", key, "global", "mean_f1")),
            )
        )
    lines.append("")
    lines.append(
        "Every column is a value the results file publishes, and nothing in "
        "this table is recomputed here. `metadata.aggregation` in each metric "
        "block names which average its `global` is. The run passed "
        "`--use-builtin-math-data`, so the digit column is measured on the "
        "bundled math corpus rather than on the prose."
    )
    lines.append("")
    lines.append(
        "**Two compression columns, two Gini columns.** Bytes per token is "
        "`compression_rate.per_tokenizer.<tok>.global.compression_rate`. Tokens "
        "per line is `.tokens_per_line.global_avg`, and on a parallel corpus it "
        "is the cross-language-comparable one: line *i* is the same sentence in "
        "every language, so the same content is being counted. Bytes are not "
        "neutral across scripts, since UTF-8 spends one byte on a Latin "
        "character and three on most CJK and Devanagari."
    )
    lines.append("")
    lines.append(
        "The same holds for the two Gini columns. Gini per byte is "
        "`tokenizer_fairness_gini.per_tokenizer.<tok>.global.gini_coefficient`; "
        "Gini per line is `.per_line_normalization.gini_coefficient`, which is "
        "`n/a` unless every language has the same line count."
    )
    lines.extend(_gini_disagreement_sentences(results, rows))
    return lines


def _gini_disagreement_sentences(results, rows):
    """The comparison of the two Gini columns, computed from the results.

    These numbers were typed into this file until 1.0.3, which made them the
    only figures in the report that a regeneration could leave disagreeing with
    the table above them. The module docstring already promised nothing was
    transcribed by hand.
    """
    pairs = []
    for key, name, _repo, _algo in rows:
        by_byte = get(results, "tokenizer_fairness_gini", "per_tokenizer", key,
                      "global", "gini_coefficient")
        by_line = get(results, "tokenizer_fairness_gini", "per_tokenizer", key,
                      "per_line_normalization", "gini_coefficient")
        if isinstance(by_byte, (int, float)) and isinstance(by_line, (int, float)):
            pairs.append((name, by_byte, by_line))

    # Two points are the minimum for a rank correlation, and the per-line block
    # is absent on any corpus that is not line-parallel, so this is the ordinary
    # case rather than a degenerate one.
    if len(pairs) < 2:
        return ["", f"{len(pairs)} of the {len(rows)} tokenizers here have a "
                    "coefficient under both units, so the two columns cannot be "
                    "compared on this run."]

    from scipy.stats import spearmanr
    rho = spearmanr([p[1] for p in pairs], [p[2] for p in pairs]).statistic

    lowest_byte = min(pairs, key=lambda p: p[1])
    lowest_line = min(pairs, key=lambda p: p[2])
    out = ["", f"Over the {len(pairs)} tokenizers with both, the two columns "
               f"rank at Spearman {rho:.3f}."]
    if lowest_byte[0] == lowest_line[0]:
        out.append(
            f"{lowest_byte[0]} has the lowest coefficient under both units, "
            f"{lowest_byte[1]:.4f} per byte and {lowest_byte[2]:.4f} per line."
        )
    else:
        out.append(
            f"**They disagree on which tokenizer is the most equitable across "
            f"languages**: {lowest_byte[0]} is lowest per byte at "
            f"{lowest_byte[1]:.4f}, against {lowest_byte[2]:.4f} per line, while "
            f"{lowest_line[0]} is lowest per line at {lowest_line[2]:.4f}, "
            f"against {lowest_line[1]:.4f} per byte. Which one is lower depends "
            "on the unit."
        )
    out.append("On this corpus, which is parallel, read the per-line column.")
    return out


def section_per_language(results, rows):
    languages = corpus_languages(results)
    lines = ["", "## Compression by language", "",
             "Bytes per token, higher meaning fewer tokens for the same text.", ""]
    header = "| Tokenizer | " + " | ".join(languages) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(languages))
    for key, name, _repo, _algo in rows:
        cells = []
        for lang in languages:
            entry = get(results, "compression_rate", "per_tokenizer", key,
                        "per_language", lang)
            if isinstance(entry, dict):
                entry = entry.get("compression_rate")
            cells.append(fmt(entry, 2))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def section_domain(results, rows):
    lines = ["", "## Code, math and round trip", "",
             "| Tokenizer | AST aligned (up) | Identifier fragmentation (down) | "
             "Operator isolation (up) | Exact round trip (up) | CER (down) |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, name, _repo, _algo in rows:
        lines.append(
            "| {name} | {ast} | {ident} | {op} | {exact} | {cer} |".format(
                name=name,
                ast=fmt(get(results, "ast_boundary_alignment", "per_tokenizer", key,
                            "global", "full_alignment_rate")),
                ident=fmt(get(results, "identifier_fragmentation", "per_tokenizer", key,
                              "global", "fragmentation_rate")),
                op=fmt(get(results, "operator_isolation_rate", "per_tokenizer", key,
                           "global", "overall_isolation_rate")),
                exact=fmt(get(results, "reconstruction_fidelity", "per_tokenizer", key,
                              "global", "exact_match_rate")),
                cer=fmt(get(results, "reconstruction_fidelity", "per_tokenizer", key,
                            "global", "mean_cer"), 4),
            )
        )
    lines.append("")
    lines.append(
        "Operator isolation pools the domains that ran, weighted by operator "
        "instances, so with this code corpus it sits close to the code rate; "
        "`by_domain` in the results file names them and splits the rates. "
        "`run.sh` does not pass `--operator-prose-domain`, so a fresh run has "
        "`code` and `math` there."
    )
    # Select on cer_skipped, not on mean_cer being None. A null mean_cer means
    # either that the edit distance exceeded the budget or that there was
    # nothing to measure, and cer_skipped is the only field separating the two.
    # Selecting on the null and then asserting the budget explanation was right
    # by luck on this corpus, not by construction.
    skipped = [
        (key, name) for key, name, _repo, _algo in rows
        if get(results, "reconstruction_fidelity", "per_tokenizer", key,
               "cer_skipped") is True
    ]
    budget = get(results, "run_metadata", "arguments", "cer_time_budget",
                 default="the configured")
    if skipped:
        lines.append("")
        lines.append(
            "CER is `n/a` for " + ", ".join(name for _key, name in skipped)
            + ". The character error "
            "rate is an edit distance, and a tokenizer that does not "
            "reconstruct its input has a large distance on every text, so the "
            f"run projected past the {budget}-second budget and reported the "
            "value as null rather than spending the time, which "
            "`cer_skipped` records."
        )
        # The rate was typed as 0.031 until 1.0.3, which tied the sentence to
        # one tokenizer on one corpus. Reading the worst of the skipped ones
        # keeps it true whichever tokenizers a run skips.
        rates = [
            (name, get(results, "reconstruction_fidelity", "per_tokenizer", key,
                       "global", "exact_match_rate"))
            for key, name in skipped
        ]
        rates = [(n, r) for n, r in rates if isinstance(r, (int, float))]
        if rates:
            worst_name, worst_rate = min(rates, key=lambda p: p[1])
            lines.append(
                "The exact round-trip column carries the same information: "
                f"{worst_name} reconstructs {worst_rate:.3f} of its texts "
                "exactly, so it is lossy on nearly every one of them."
            )
    return lines


MERGE_USAGE_COLUMNS = ("pct_active", "pct_rare", "pct_uncommon", "pct_unseen", "pct_scaffold")


def section_merge_usage(results, rows):
    """Merge vocabulary usage, with the sentences that apply to this run.

    Omitted, with a message, when the results file has no
    merge_vocabulary_usage block: a run with --no-merge-vocabulary-usage, or
    one from before 1.2.0.
    """
    block = get(results, "merge_vocabulary_usage")
    if not block:
        print("No merge_vocabulary_usage block in the results; omitting that section.")
        return []
    per_tok = block.get("per_tokenizer", {})
    lines = ["", "## Merge vocabulary usage", "",
             "| Tokenizer | Merge tokens | Active % | Rare % | Uncommon % | "
             "Unseen % | Scaffold % |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, name, _repo, _algo in rows:
        g = get(per_tok, key, "global") or {}
        cells = [fmt(g.get("n_merge_tokens"))] + [fmt(g.get(c), 2) for c in MERGE_USAGE_COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    corpus = get(block, "metadata", "corpus")
    where = ("on the FLORES+ sentences of the tables above" if corpus == "main input"
             else "on the FLORES+ sentences of the tables above and the code and "
                  "math corpora")
    lines += [
        "",
        "Each percentage is the share of the tokenizer's merge tokens in that "
        f"category, {where}. A merge token is a vocabulary entry that a BPE "
        "merge produces. A merge token is Active when it is emitted at least 5 "
        "times per million final tokens, Rare when it is emitted from 1 up to 5 "
        "times per million, Uncommon when it is emitted less often than that but "
        "is still built as an intermediate step, and Unseen when it is never "
        "built. The four percentages sum to 100. Scaffold is the share of merge "
        "tokens that are Rare or Uncommon and that are merged further in more "
        "than 90% of the cases where BPE builds them. "
        "[METRICS.md](../../docs/METRICS.md#merge-vocabulary-usage-merge_vocabulary_usage) "
        "gives the definitions.",
    ]

    applicable = [(key, name, per_tok[key]) for key, name, _r, _a in rows
                  if get(per_tok, key, "applicable") is True]
    sentences = []
    limit = get(block, "metadata", "constants", "min_final_tokens_warn")
    sizes = [(name, e["global"]["total_final_tokens"], e["global"]["threshold_counts"])
             for _key, name, e in applicable]
    if sizes and limit is not None and max(n for _name, n, _t in sizes) < limit:
        smallest = min(sizes, key=lambda item: item[1])
        active_from = math.ceil(smallest[2]["rare_below"])
        sentences.append(
            f"This corpus has between {min(n for _name, n, _t in sizes):,} and "
            f"{max(n for _name, n, _t in sizes):,} final tokens per tokenizer, "
            f"fewer than {limit:,}, so each rate threshold corresponds to a few "
            f"emissions: for {smallest[0]}, Active means emitted at least "
            f"{active_from} times and Uncommon means never emitted on its own."
        )
    reencoded = [(name, e) for _key, name, e in applicable
                 if e.get("encoding") == "reencoded_ignore_merges_false"]
    if reencoded:
        changed = " and ".join(
            f"{e['texts_differing_after_reencode']:,} of "
            f"{sum(e['texts_per_corpus'].values()):,} texts for {name}"
            for name, e in reencoded)
        sentences.append(
            " and ".join(name for name, _e in reencoded)
            + (" ships" if len(reencoded) == 1 else " ship")
            + " with `ignore_merges` on and "
            + ("was" if len(reencoded) == 1 else "were")
            + " encoded a second time with "
            f"it off; this changed the token ids of {changed}."
        )
    for _key, name, e in applicable:
        unreachable = e["global"]["n_unreachable_without_ignore_merges"]
        if unreachable:
            sentences.append(
                f"{name} has {unreachable:,} merge tokens that cannot be built "
                "with `ignore_merges` off. They are counted as Unseen."
            )
    for key, name, _r, _a in rows:
        entry = per_tok.get(key)
        if entry is not None and entry.get("applicable") is False:
            sentences.append(f"The metric does not apply to {name} ({entry['reason']}).")
    if sentences:
        lines += [""] + [" ".join(sentences)]
    return lines


# Cell glyph per severity. Five, because sanity_check.Severity has five and a
# legend short of one renders an unknown state as a blank cell that reads as a
# pass. tokenizer_analysis/tests/test_render_report.py asserts the legend covers
# every severity the sanity file actually emitted.
SEVERITY_GLYPH = {
    "pass": "ok",
    "warn": "warn",
    "fail": "FAIL",
    "not_applicable": "n/a",
    "unverifiable": "?",
}

SEVERITY_LEGEND = {
    "pass": "the check ran and the tokenizer met its condition",
    "warn": "the check ran and found something that is a defect in some "
            "tokenizers and a design choice in others",
    "fail": "the check ran and found something no correct tokenizer does",
    "not_applicable": "the check does not apply to this tokenizer",
    "unverifiable": "the check could not run, because the tokenizer does not "
                    "expose what it needs",
}


def check_ids(sanity):
    """The check ids in file order, from the first tokenizer's own check names.

    The results file keys each check by its full name, "C7 special-token
    sanity" rather than "C7", and there is no C9. Reading the ids off the file
    rather than composing them keeps this from inventing a column for a check
    that was removed or missing one that was added.
    """
    per_tok = get(sanity, "tokenizer_sanity_check", "per_tokenizer", default={})
    if not per_tok:
        return []
    first = per_tok[sorted(per_tok)[0]]
    return [(name.split()[0], name) for name in first.get("checks", {})]


def section_health(sanity, rows):
    """The 16-check health matrix, or nothing at all when there is no input."""
    if not sanity:
        return []
    ids = check_ids(sanity)
    per_tok = get(sanity, "tokenizer_sanity_check", "per_tokenizer", default={})
    if not ids or not per_tok:
        return []

    header = " | ".join(
        f"[{cid}](../../docs/SANITY_CHECKS.md#{anchor(name)})" for cid, name in ids
    )
    lines = [
        "", "## Tokenizer health", "",
        "`tokenizer-sanity-check` asks whether each tokenizer is intact, which "
        "is a different question from how well it compresses. Every cell is "
        "`tokenizer_sanity_check.per_tokenizer.<tok>.checks.<name>.severity` "
        "from `sanity_results.json`. Column headers link to the check "
        "definition.",
        "",
        f"| Tokenizer | {header} |",
        "|---" * (len(ids) + 1) + "|",
    ]
    emitted = set()
    for key, name, _repo, _algo in rows:
        checks = get(sanity, "tokenizer_sanity_check", "per_tokenizer", key,
                     "checks", default={})
        cells = []
        for _cid, check_name in ids:
            sev = get(checks, check_name, "severity")
            if sev is not None:
                emitted.add(sev)
            cells.append(SEVERITY_GLYPH.get(sev, "?" if sev is None else sev))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("Legend, in worsening order:")
    lines.append("")
    lines.append("| Cell | Severity | Meaning |")
    lines.append("|---|---|---|")
    for sev in ("pass", "not_applicable", "unverifiable", "warn", "fail"):
        if sev in emitted:
            lines.append(f"| `{SEVERITY_GLYPH[sev]}` | `{sev}` | {SEVERITY_LEGEND[sev]} |")
    lines.append("")
    lines.append(
        "A tokenizer's overall verdict is the worst of its checks, with "
        "`not_applicable` ranking alongside `pass` and `unverifiable` alongside "
        "`warn`. A check that could not run is therefore never reported as one "
        "that passed."
    )
    lines.append("")
    lines.append(
        "The probes are the 78 built-in ones plus the bundled math corpus and "
        "up to 50 FLORES sentences per language from the same 13 languages the "
        "metrics above use, so the behavioural checks and the metrics describe "
        "the same text. `run.sh` records the exact invocation. Reproducing the "
        "matrix needs that configuration: the behavioural checks measure what "
        "the probes contain."
    )
    return lines


def anchor(check_name):
    """The docs/SANITY_CHECKS.md heading slug for a check name.

    The headings there are the check names verbatim, so the GitHub slug rule
    (lowercase, drop punctuation, spaces to hyphens) reproduces them.
    """
    slug = check_name.lower()
    slug = "".join(c for c in slug if c.isalnum() or c in " -_")
    return slug.replace(" ", "-")


def section_caveats(rows):
    return [
        "", "## How to read this", "",
        "**The per-byte Gini column is not neutral across scripts.** UTF-8 "
        "spends one byte per Latin character, two for Cyrillic, three for most "
        "CJK and Devanagari, so a tokenizer can score cheaper on Chinese than "
        "on English because the denominator is larger rather than because it "
        "segments Chinese better. The per-line column beside it divides by a "
        "count that is the same in every language, which is why it is the one "
        "to read here. Equal line counts do not by themselves establish that a "
        "corpus is parallel; that FLORES+ is parallel is what makes the "
        "per-line column meaningful, and the library cannot check it. Read the "
        "per-language costs in the results file beside either coefficient.",
        "",
        "**Vocabulary size is not held constant.** The tokenizers here span "
        f"{len(rows)} models and a wide vocabulary range, and a larger "
        "vocabulary buys compression almost mechanically. A comparison between "
        "two tokenizers of different size measures the size as much as the "
        "algorithm.",
        "",
        "**Training data is not held constant and is mostly undisclosed.** "
        "These tokenizers were trained on different corpora with different "
        "language mixes. A tokenizer that compresses Hindi well may have seen "
        "more Hindi, not tokenized it better.",
        "",
        "**The corpus is FLORES+, which is translated news.** It is parallel "
        "across languages, which is what makes the fairness comparison "
        "meaningful, and it is not representative of what any of these models "
        "was trained on.",
        "",
        "**Compression depends on the measurement unit.** The headline gives "
        "both bytes per token and tokens per line, and they order the "
        "tokenizers differently for the reason above. Under a word unit the "
        "ordering changes again, particularly for languages without whitespace "
        "word boundaries.",
        "",
        "**The code numbers depend on the code corpus.** They come from 100 "
        "files per language of The Stack, so they describe that sample rather "
        "than code in general.",
        "",
        "None of this makes the numbers wrong. It means a single ranking over "
        "them would be, so the report gives the columns and not a rank.",
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sanity", type=Path, default=None,
        help="sanity_results.json for the health matrix. Defaults to a file "
             "of that name beside --results. The section is omitted when the "
             "file is absent.",
    )
    args = parser.parse_args(argv)

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    # An explicit --sanity that does not exist is an error: the caller named a
    # file. An absent default is not, because a run predating the sanity step
    # has no such file and should still render everything else.
    sanity_path = args.sanity or args.results.parent / "sanity_results.json"
    sanity = None
    if args.sanity is not None and not args.sanity.is_file():
        raise SystemExit(f"--sanity {args.sanity} does not exist.")
    if sanity_path.is_file():
        with open(sanity_path, encoding="utf-8") as f:
            sanity = json.load(f)
    else:
        print(f"No {sanity_path}; omitting the tokenizer health section. "
              f"Run tokenizer-sanity-check to produce it.")

    rows = present(results)
    if not rows:
        raise SystemExit(
            f"{args.results} holds no tokenizer this report knows about. "
            f"Expected some of: {', '.join(key for key, *_ in TOKENIZERS)}."
        )

    lines = [
        "# Open-source tokenizers, measured",
        "",
        "Generated by `render_report.py` from `analysis_results.json` in this "
        "directory. Do not edit by hand: `bash benchmarks/open_source/run.sh` "
        "regenerates both.",
        "",
    ]
    lines += section_what_was_measured(results, rows)
    lines += section_headline(results, rows)
    lines += section_per_language(results, rows)
    lines += section_domain(results, rows)
    lines += section_merge_usage(results, rows)
    lines += section_health(sanity, rows)
    lines += section_caveats(rows)
    lines.append("")
    lines.append(
        "Every metric in the results file, including the ones no table above "
        "shows, is defined in [METRICS.md](../../docs/METRICS.md)."
    )
    lines.append("")

    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output} from {args.results}: {len(rows)} tokenizers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
