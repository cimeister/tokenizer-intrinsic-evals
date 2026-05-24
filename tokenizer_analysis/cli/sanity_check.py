"""CLI for the single-tokenizer sanity-check diagnostic.

    tokenizer-sanity-check --tokenizer-config configs/baseline.json [--only NAME]
    tokenizer-sanity-check custom_bpe:tokenizers/foo            # positional

Exit codes: 0 all pass, 1 >=1 warn, 2 >=1 fail, 3 execution error.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tokenizer_analysis.cli.run_analysis import load_config_from_file
from tokenizer_analysis.core.tokenizer_wrapper import (
    create_tokenizer_wrapper,
    TokenizerWrapper,
)
from tokenizer_analysis.diagnostics.probe_corpus import (
    builtin_probes,
    load_flores_probes,
    load_math_probes,
)
from tokenizer_analysis.diagnostics.sanity_check import (
    Severity,
    render_text,
    run_sanity_check,
    severity_to_exit_code,
)
from tokenizer_analysis.constants import SANITY_PROBE_SAMPLES_PER_LANG

logger = logging.getLogger(__name__)

# Distinct from the graded 0/1/2 health verdict: the tool itself failed to run.
EXIT_EXECUTION_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tokenizer-sanity-check",
        description="Single-tokenizer health diagnostic (faithful pipeline).",
    )
    p.add_argument("tokenizer", nargs="?",
                   help="positional CLASS:PATH shorthand for a one-off check")
    p.add_argument("--tokenizer-config",
                   help="JSON config {name: {class, path, ...}} (loops over all)")
    p.add_argument("--only", help="restrict a --tokenizer-config run to this name")
    p.add_argument("--output-dir",
                   help="write sanity_results.json here (mirrors run_analysis)")
    p.add_argument("--use-sample-data", action="store_true",
                   help="add FLORES breadth (requires --language-config)")
    p.add_argument("--language-config",
                   help="language metadata JSON for --use-sample-data")
    p.add_argument("--probe-samples-per-lang", type=int,
                   default=SANITY_PROBE_SAMPLES_PER_LANG)
    p.add_argument("--use-builtin-math-data", action="store_true")
    p.add_argument("--quiet", action="store_true",
                   help="collapse passing checks in the text report")
    p.add_argument("--exit-zero", action="store_true",
                   help="always exit 0 (report is informational)")
    return p


def _wrappers_from_args(args) -> Dict[str, TokenizerWrapper]:
    if args.tokenizer and args.tokenizer_config:
        raise ValueError("pass either a positional CLASS:PATH or "
                         "--tokenizer-config, not both")
    if args.tokenizer:
        if ":" not in args.tokenizer:
            raise ValueError(
                f"positional spec {args.tokenizer!r} must be CLASS:PATH")
        cls, path = args.tokenizer.split(":", 1)
        # create_tokenizer_wrapper reads config['class']/config['path'] and
        # raises ValueError listing valid classes for an unknown class.
        name = Path(path).name or path
        return {name: create_tokenizer_wrapper(name,
                                               {"class": cls, "path": path})}
    if args.tokenizer_config:
        cfg = load_config_from_file(args.tokenizer_config)
        if args.only:
            if args.only not in cfg:
                raise ValueError(
                    f"--only {args.only!r} not in config "
                    f"(have {list(cfg)})")
            cfg = {args.only: cfg[args.only]}
        return {name: create_tokenizer_wrapper(name, c)
                for name, c in cfg.items()}
    raise ValueError("provide a positional CLASS:PATH or --tokenizer-config")


def _build_probes(args) -> List:
    probes = builtin_probes()
    if args.use_sample_data:
        if not args.language_config:
            raise ValueError("--use-sample-data requires --language-config")
        probes += load_flores_probes(args.language_config,
                                     args.probe_samples_per_lang)
    if args.use_builtin_math_data:
        probes += load_math_probes()
    if not probes:
        raise ValueError("no probes assembled")
    return probes


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    try:
        wrappers = _wrappers_from_args(args)
        probes = _build_probes(args)
        report = run_sanity_check(wrappers, probes)
    except Exception as e:
        logger.error("execution error: %s", e)
        print(f"sanity-check execution error: {e}", file=sys.stderr)
        return 0 if args.exit_zero else EXIT_EXECUTION_ERROR

    print(render_text(report, quiet=args.quiet))

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        def conv(o):
            if hasattr(o, "tolist"):
                return o.tolist()
            if isinstance(o, dict):
                return {k: conv(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [conv(x) for x in o]
            return o

        with open(out / "sanity_results.json", "w") as f:
            json.dump(conv(report), f, indent=2)
        logger.info("wrote %s", out / "sanity_results.json")

    summary = report["tokenizer_sanity_check"]["summary"]
    worst = Severity.overall([s["overall_severity"] for s in summary.values()])
    code = severity_to_exit_code(worst)
    return 0 if args.exit_zero else code


if __name__ == "__main__":
    sys.exit(main())
