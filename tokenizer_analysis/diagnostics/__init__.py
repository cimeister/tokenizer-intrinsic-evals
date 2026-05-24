"""Single-tokenizer diagnostic tooling.

Distinct from :mod:`tokenizer_analysis.metrics` (the corpus-level batch
pipeline keyed by an ``InputProvider``).  The diagnostics subpackage answers
"is *this one* tokenizer sane / correctly trained?" with a pass/warn/fail
health report and a CI-gradable exit code.
"""

from .sanity_check import (
    TokenizerSanityChecker,
    Severity,
    run_sanity_check,
    render_text,
    severity_to_exit_code,
)

__all__ = [
    "TokenizerSanityChecker",
    "Severity",
    "run_sanity_check",
    "render_text",
    "severity_to_exit_code",
]
