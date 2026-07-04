#!/usr/bin/env python3
"""Visualize how tokenizers handle code, math, and multilingual text.

Usage:
    # Show built-in samples (code, math, multilingual) with all tokenizers
    tokenizer-visualize \\
        --tokenizer-config configs/sample_tokenizers.json

    # Only show a subset of tokenizers
    tokenizer-visualize \\
        --tokenizer-config configs/sample_tokenizers.json \\
        --tokenizers "bpe" "unigramlm"

    # Visualize a single text file
    tokenizer-visualize \\
        --tokenizer-config configs/sample_tokenizers.json \\
        --input my_script.py

    # Visualize all files in a directory (1 sample per file by default)
    tokenizer-visualize \\
        --tokenizer-config configs/sample_tokenizers.json \\
        --input data/samples/

    # Read up to 3 samples per file (separated by --- lines)
    tokenizer-visualize \\
        --tokenizer-config configs/sample_tokenizers.json \\
        --input data/samples/ --samples-per-file 3

    # Plain text (no colour escapes) for file output
    tokenizer-visualize \\
        --tokenizer-config configs/sample_tokenizers.json --no-color > out.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

from tokenizer_analysis.core.tokenizer_wrapper import create_tokenizer_wrapper

# ── Default samples ──────────────────────────────────────────────────────
# Each category has a list of (label, text) pairs.

DEFAULT_CODE = textwrap.dedent("""\
    import os
    from pathlib import Path

    def count_lines(path: str, include_empty: bool = True) -> int:
        \"\"\"Count lines in a file.\"\"\"
        total = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if include_empty or line.strip():
                    total += 1
        return total

    class FileProcessor:
        def __init__(self, root_dir: str):
            self.root = Path(root_dir)
            self._cache: dict[str, int] = {}

        def process(self):
            for f in self.root.rglob("*.py"):
                n = count_lines(str(f))
                self._cache[str(f)] = n
                if n >= 100:
                    print(f"  {f.name}: {n} lines")

    if __name__ == "__main__":
        proc = FileProcessor(os.getcwd())
        proc.process()
""")

DEFAULT_MULTILINGUAL = textwrap.dedent("""\
    The theory of relativity revolutionized our understanding of space, time, and gravity.
    Die Relativitätstheorie revolutionierte unser Verständnis von Raum, Zeit und Gravitation.
    La théorie de la relativité a révolutionné notre compréhension de l'espace, du temps et de la gravité.
    La teoría de la relatividad revolucionó nuestra comprensión del espacio, el tiempo y la gravedad.
    A teoria da relatividade revolucionou nossa compreensão do espaço, do tempo e da gravidade.
    La teoria della relatività ha rivoluzionato la nostra comprensione dello spazio, del tempo e della gravità.
    Теория относительности произвела революцию в нашем понимании пространства, времени и гравитации.
    相対性理論は、空間、時間、重力に対する我々の理解に革命をもたらした。
    相对论彻底改变了我们对空间、时间和引力的理解。
    상대성 이론은 공간, 시간, 중력에 대한 우리의 이해를 혁명적으로 변화시켰다.
    نظرية النسبية أحدثت ثورة في فهمنا للمكان والزمان والجاذبية.
    सापेक्षता के सिद्धांत ने अंतरिक्ष, समय और गुरुत्वाकर्षण की हमारी समझ में क्रांति ला दी।
    ทฤษฎีสัมพัทธภาพได้ปฏิวัติความเข้าใจของเราเกี่ยวกับอวกาศ เวลา และแรงโน้มถ่วง
    Η θεωρία της σχετικότητας έφερε επανάσταση στην κατανόησή μας για τον χώρο, τον χρόνο και τη βαρύτητα.
    Görelilik teorisi uzay, zaman ve kütleçekimi anlayışımızda devrim yarattı.
""").rstrip()

DEFAULT_MATH_UNICODE = textwrap.dedent("""\
    Definitions:
      \u2200\u03b5 > 0, \u2203\u03b4 > 0 such that |x \u2212 a| < \u03b4 \u27f9 |f(x) \u2212 L| < \u03b5
      f: \u211d\u207f \u2192 \u211d is a linear map if f(\u03b1x + \u03b2y) = \u03b1f(x) + \u03b2f(y)

    Key results:
      e^(i\u03c0) + 1 = 0                          (Euler's identity)
      \u2211(n=1 to \u221e) 1/n\u00b2 = \u03c0\u00b2/6                (Basel problem)
      \u222b(\u2212\u221e to \u221e) e^(\u2212x\u00b2) dx = \u221a\u03c0             (Gaussian integral)
      det(A) = \u220f\u1d62 \u03bb\u1d62                           (eigenvalue product)

    Set theory:
      A \u2282 B \u27fa (x \u2208 A \u27f9 x \u2208 B)
      |\u2115| = |\u2124| = |\u211a| = \u2135\u2080 < |\u211d|

    Logic:
      (P \u2227 Q) \u2228 R \u2261 (P \u2228 R) \u2227 (Q \u2228 R)
      \u00ac(\u2200x: P(x)) \u27fa \u2203x: \u00acP(x)
""").rstrip()

# Ordered list of (label, text) for the built-in demo.
_DEFAULT_SAMPLES: list[tuple[str, str]] = [
    ("Python code", DEFAULT_CODE),
    ("Unicode mathematics", DEFAULT_MATH_UNICODE),
    ("Multilingual text", DEFAULT_MULTILINGUAL),
]

# ── ANSI colour helpers ──────────────────────────────────────────────────
_BG_COLORS = [
    "\033[48;5;153m",  # light blue
    "\033[48;5;180m",  # tan
    "\033[48;5;151m",  # pale green
    "\033[48;5;182m",  # light purple
    "\033[48;5;216m",  # peach
    "\033[48;5;152m",  # pale cyan
]
_FG_BLACK = "\033[38;5;232m"
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_BG_SPLIT = "\033[41m"         # red background for sub-character splits
_FG_SPLIT = "\033[97m"         # bright white text on split background

_UNASSIGNED = -1  # sentinel for char_color positions not owned by any token

# ── Sample loading ──────────────────────────────────────────────────────

_SAMPLE_SEPARATOR = "\n---\n"


def _load_samples_from_file(
    path: Path, max_samples: int,
) -> list[tuple[str, str]]:
    """Load up to *max_samples* text samples from a single file.

    If the file contains ``---`` lines (on their own line), they are used
    as sample separators.  Otherwise the whole file is one sample.
    """
    text = path.read_text(encoding="utf-8", errors="replace").rstrip()
    if not text.strip():
        return []

    parts = text.split(_SAMPLE_SEPARATOR)
    samples: list[tuple[str, str]] = []
    for i, part in enumerate(parts):
        part = part.rstrip()
        if not part.strip():
            continue
        label = path.name if len(parts) == 1 else f"{path.name} [{i + 1}]"
        samples.append((label, part))
        if len(samples) >= max_samples:
            break
    return samples


def _load_samples_from_dir(
    dirpath: Path, max_samples_per_file: int,
) -> list[tuple[str, str]]:
    """Load samples from all text files in *dirpath* (non-recursive)."""
    samples: list[tuple[str, str]] = []
    for child in sorted(dirpath.iterdir()):
        if child.is_file() and not child.name.startswith("."):
            samples.extend(
                _load_samples_from_file(child, max_samples_per_file)
            )
    return samples


def collect_samples(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Return the list of ``(label, text)`` samples to visualize."""
    input_path = args.input or args.code_file
    if input_path is not None:
        p = Path(input_path)
        if not p.exists():
            print(f"Error: path does not exist: {p}", file=sys.stderr)
            sys.exit(1)
        if p.is_dir():
            samples = _load_samples_from_dir(p, args.samples_per_file)
        else:
            samples = _load_samples_from_file(p, args.samples_per_file)
        if not samples:
            print(f"Error: no text samples found in {p}", file=sys.stderr)
            sys.exit(1)
        return samples

    return list(_DEFAULT_SAMPLES)


# ── Offset extraction ────────────────────────────────────────────────────

def _get_offsets(wrapper, text: str, ids: list[int]) -> list[tuple[int, int]] | None:
    """Get (start, end) character offsets for each token via encode_with_offsets.

    Uses the wrapper's ``encode_with_offsets`` method, which guarantees that
    the returned offsets correspond to the same encoding path as ``encode``.

    Raises ``ValueError`` if offsets are returned but their length does not
    match *ids*.
    """
    token_ids, offsets = wrapper.encode_with_offsets(text)
    if offsets is None:
        return None
    if len(offsets) != len(ids):
        raise ValueError(
            f"Offset length mismatch for {wrapper.get_name()}: "
            f"encode() returned {len(ids)} tokens but encode_with_offsets() "
            f"returned {len(offsets)} offsets"
        )
    return offsets


# ── Visualisation ─────────────────────────────────────────────────────────

def _ws_visible(ch: str) -> str:
    """Replace a single whitespace char with a visible glyph."""
    if ch == " ":
        return "\u00b7"   # middle dot
    if ch == "\t":
        return "\u2192"   # right arrow
    if ch == "\n":
        return "\u21b5"   # down-left arrow (pilcrow-ish)
    if ch == "\r":
        return "\u240d"
    return ch


def _fill_offsets(
    offsets: list[tuple[int, int]],
    text_len: int,
) -> list[tuple[int, int]]:
    """Fill gaps in offsets and clamp overlaps.

    Pre-tokenizers (e.g. GPT-2 ByteLevel) may consume whitespace that
    doesn't appear in any token's offset.  Gaps are assigned to the next
    real token.  Zero-length offsets (``s == e``) are kept as-is — they
    mark special tokens (BOS/EOS, byte-fallback, etc.).
    """
    filled: list[tuple[int, int]] = []
    prev_end = 0
    for s, e in offsets:
        if s == e:
            # Zero-length span: special/synthetic token — keep unchanged
            filled.append((s, e))
            continue
        # Clamp start to avoid overlap; extend back to fill any gap
        real_start = max(prev_end, 0) if prev_end > s else (prev_end if prev_end < s else s)
        filled.append((real_start, e))
        prev_end = e
    return filled


def _build_char_owner(
    offsets: list[tuple[int, int]],
    text_len: int,
) -> tuple[list[int], list[int]]:
    """Map each character position to the token index that owns it.

    Returns ``(owner, token_count)`` where ``owner[i]`` is the last token
    index to claim position *i* (``_UNASSIGNED`` if none) and
    ``token_count[i]`` is how many tokens claim that position.

    When byte-level tokenizers split a multi-byte character across
    multiple tokens, the HuggingFace tokenizers library maps every
    byte-token back to the *same* character offset.  ``token_count``
    exposes these hidden splits so the visualisation can annotate them.
    """
    owner: list[int] = [_UNASSIGNED] * text_len
    token_count: list[int] = [0] * text_len
    for idx, (s, e) in enumerate(offsets):
        for c in range(s, min(e, text_len)):
            owner[c] = idx
            token_count[c] += 1
    return owner, token_count


def visualize_tokens(
    name: str,
    text: str,
    wrapper,
    use_color: bool,
    label: str = "",
) -> str:
    """Return a formatted string showing the tokenized text for one tokenizer."""
    ids = wrapper.encode(text)
    tokens = wrapper.convert_ids_to_tokens(ids)
    offsets = _get_offsets(wrapper, text, ids)

    # If offsets are available, use them for a character-span view.
    if offsets:
        raw_offsets = offsets
        offsets = _fill_offsets(raw_offsets, len(text))
        spans = [text[s:e] for s, e in offsets]
    else:
        raw_offsets = None
        offsets = None
        spans = None

    n_tokens = len(ids)
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────
    sep = "─" * 72
    header = f"{label}  ({n_tokens} tokens)" if label else f"{name}  ({n_tokens} tokens)"
    lines.append(f"\n{header}")
    lines.append(sep)

    # ── Token-coloured source view (line by line) ─────────────────────
    if spans is not None:
        char_color, _ = _build_char_owner(offsets, len(text))
        # Compute token counts from *raw* (pre-fill) offsets so that
        # overlapping byte-tokens sharing the same character offset
        # are counted before _fill_offsets clamps them to zero-length.
        _, char_tcount = _build_char_owner(raw_offsets, len(text))

        source_lines = text.split("\n")
        pos = 0  # character position in `text`
        for line_no, src_line in enumerate(source_lines):
            line_end = pos + len(src_line)
            buf: list[str] = []
            if use_color:
                buf.append(f"{_DIM}{line_no + 1:3d}{_RESET} ")
            else:
                buf.append(f"{line_no + 1:3d} ")

            prev_tok_idx = _UNASSIGNED - 1  # force first-char colour switch
            for ci in range(pos, line_end):
                tok_idx = char_color[ci]
                ch = text[ci]
                vis = _ws_visible(ch)
                is_split = char_tcount[ci] > 1
                is_boundary = tok_idx != prev_tok_idx

                if use_color:
                    if is_split:
                        # Red background makes split chars impossible to miss
                        buf.append(
                            f"{_BG_SPLIT}{_FG_SPLIT}{vis}{_RESET}"
                            f"{_BG_COLORS[tok_idx % len(_BG_COLORS)]}{_FG_BLACK}"
                        )
                    else:
                        if is_boundary:
                            buf.append(f"{_BG_COLORS[tok_idx % len(_BG_COLORS)]}{_FG_BLACK}")
                        buf.append(vis)
                else:
                    if is_boundary and prev_tok_idx != _UNASSIGNED - 1:
                        buf.append("|")
                    if is_split:
                        buf.append(f"{vis}({char_tcount[ci]})")
                    else:
                        buf.append(vis)

                if is_boundary:
                    prev_tok_idx = tok_idx

            if use_color:
                buf.append(_RESET)

            lines.append("".join(buf))
            pos = line_end + 1  # skip the \n

    else:
        # Fallback: show raw token strings (Ġ/Ċ encoding) delimited by |
        if use_color:
            buf: list[str] = []
            for i, tok_str in enumerate(tokens):
                bg = _BG_COLORS[i % len(_BG_COLORS)]
                buf.append(f"{bg}{_FG_BLACK}{tok_str}{_RESET}")
            lines.append("".join(buf))
        else:
            lines.append("|".join(tokens))

    # ── Whitespace statistics ─────────────────────────────────────────
    if spans is not None:
        # char_color / char_tcount already computed above for the
        # coloured source view.

        ws_only = 0          # tokens whose span is pure non-newline whitespace
        newline_toks = 0      # tokens whose span contains at least one \n
        special_toks = 0      # zero-length spans (BOS/EOS etc.)
        newline_indent_toks = 0  # tokens containing \n followed by spaces

        for sp in spans:
            if not sp:
                special_toks += 1
            elif "\n" in sp:
                newline_toks += 1
                # Check for merged newline+indent pattern
                after_last_nl = sp[sp.rfind("\n") + 1:]
                if after_last_nl and after_last_nl.isspace():
                    newline_indent_toks += 1
            elif sp.isspace():
                ws_only += 1

        # ── Indentation analysis (character-based) ────────────────────
        # For each indented line, find how many *distinct* tokens own
        # the leading-whitespace character positions.  This correctly
        # handles tokens that straddle \n boundaries (e.g. "\n    ")
        # because char_color assigns ownership per character.
        indent_patterns: dict[str, int] = {}
        indent_level_tokens: dict[int, list[int]] = {}
        total_indent_toks = 0

        source_lines = text.split("\n")
        pos = 0
        for src_line in source_lines:
            line_end = pos + len(src_line)
            leading_ws = len(src_line) - len(src_line.lstrip())
            if leading_ws > 0:
                # Collect the distinct tokens owning each indent char
                tok_ids_in_indent: list[int] = []
                for ci in range(pos, pos + leading_ws):
                    owner = char_color[ci]
                    if owner == _UNASSIGNED:
                        continue
                    if not tok_ids_in_indent or tok_ids_in_indent[-1] != owner:
                        tok_ids_in_indent.append(owner)

                n_indent_toks = len(tok_ids_in_indent)
                total_indent_toks += n_indent_toks
                indent_level_tokens.setdefault(leading_ws, []).append(n_indent_toks)

                # Build pattern: tuple of per-token space counts
                pattern_parts: list[int] = []
                for tid in tok_ids_in_indent:
                    count = sum(
                        1 for ci in range(pos, pos + leading_ws)
                        if char_color[ci] == tid
                    )
                    pattern_parts.append(count)
                pat_key = repr(tuple(pattern_parts)) if pattern_parts else "()"
                indent_patterns[pat_key] = indent_patterns.get(pat_key, 0) + 1

            pos = line_end + 1

        lines.append("")
        ws_summary = (
            f"  Whitespace tokens: {ws_only}/{n_tokens}"
            f"  |  Newline tokens: {newline_toks}"
            f"  |  Indentation tokens: {total_indent_toks}"
        )
        if newline_indent_toks:
            ws_summary += f" ({newline_indent_toks} merged with newline)"
        if special_toks:
            ws_summary += f"  |  Special: {len([sp for sp in spans if not sp])}"

        if use_color:
            lines.append(f"{_DIM}{ws_summary}{_RESET}")
        else:
            lines.append(ws_summary)

        # Sub-character split summary (byte-level tokenizers)
        split_chars = sum(1 for tc in char_tcount if tc > 1)
        hidden_tokens = sum(tc - 1 for tc in char_tcount if tc > 1)
        if split_chars:
            split_note = (
                f"  Sub-character splits: {split_chars} char(s) split "
                f"across multiple byte-tokens ({hidden_tokens} hidden "
                f"token(s) — red background in colour mode)"
            )
            if use_color:
                lines.append(f"{_DIM}{split_note}{_RESET}")
            else:
                lines.append(split_note)

        if indent_patterns:
            sorted_pats = sorted(indent_patterns.items(), key=lambda x: -x[1])
            pat_strs = [f"{pat} x{cnt}" for pat, cnt in sorted_pats]
            indent_detail = f"  Indent patterns (spaces per token): {', '.join(pat_strs)}"
            if use_color:
                lines.append(f"{_DIM}{indent_detail}{_RESET}")
            else:
                lines.append(indent_detail)

        if indent_level_tokens:
            depth_summary_parts = []
            for depth in sorted(indent_level_tokens):
                counts = indent_level_tokens[depth]
                avg = sum(counts) / len(counts)
                depth_summary_parts.append(f"{depth}sp={avg:.1f}tok")

            depth_detail = f"  Tokens per indent depth: {', '.join(depth_summary_parts)}"
            if use_color:
                lines.append(f"{_DIM}{depth_detail}{_RESET}")
            else:
                lines.append(depth_detail)

    return "\n".join(lines)


# ── Display helpers ──────────────────────────────────────────────────────

def _print_source(label: str, text: str, use_color: bool) -> None:
    """Print the source text with line numbers for reference."""
    n_chars = len(text)
    n_lines = len(text.splitlines())
    if use_color:
        print(f"\n{_BOLD}{label}{_RESET} ({n_chars} chars, {n_lines} lines):")
        print(f"{_DIM}{'─' * 72}{_RESET}")
        for i, line in enumerate(text.splitlines(), 1):
            print(f"{_DIM}{i:3d}{_RESET} {line}")
        print(f"{_DIM}{'─' * 72}{_RESET}")
    else:
        print(f"\n{label} ({n_chars} chars, {n_lines} lines):")
        print("─" * 72)
        for i, line in enumerate(text.splitlines(), 1):
            print(f"{i:3d} {line}")
        print("─" * 72)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize tokenization of code, math, and multilingual "
                    "text with emphasis on token boundaries and whitespace.",
    )
    parser.add_argument(
        "--tokenizer-config", required=True,
        help="JSON file with tokenizer configurations "
             "(same format as tokenizer-analysis)",
    )
    parser.add_argument(
        "--tokenizers", nargs="+", default=None,
        help="Subset of tokenizer names to show (default: all)",
    )
    parser.add_argument(
        "--input", default=None,
        help="Text file or directory of files to visualize.  "
             "Within a file, samples are separated by a line containing "
             "only '---'.  When omitted, built-in samples (code, math, "
             "multilingual) are shown.",
    )
    parser.add_argument(
        "--code-file", default=None,
        help="(Backward-compatible alias for --input with a single file.)",
    )
    parser.add_argument(
        "--samples-per-file", type=int, default=1,
        help="Max samples to read from each file when it contains '---' "
             "separators (default: 1, meaning the first sample only).",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colours (for piping to files)",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> None:

    # ── Collect samples ──────────────────────────────────────────────
    samples = collect_samples(args)

    # ── Load tokenizer configs ───────────────────────────────────────
    with open(args.tokenizer_config, encoding="utf-8") as f:
        all_configs = json.load(f)

    names = args.tokenizers or list(all_configs.keys())
    missing = [n for n in names if n not in all_configs]
    if missing:
        print(f"Error: tokenizer(s) not found in config: {missing}", file=sys.stderr)
        print(f"Available: {list(all_configs.keys())}", file=sys.stderr)
        sys.exit(1)

    use_color = not args.no_color and sys.stdout.isatty()

    # ── Create wrappers once ─────────────────────────────────────────
    wrappers: list[tuple[str, object]] = []
    for name in names:
        config = all_configs[name]
        try:
            wrapper = create_tokenizer_wrapper(name, config)
            wrappers.append((name, wrapper))
        except Exception as e:
            print(f"\nSkipping {name}: {e}", file=sys.stderr)

    # ── Show source texts once for reference ─────────────────────────
    for label, text in samples:
        _print_source(label, text, use_color)

    # ── Show tokenizations grouped by tokenizer ──────────────────────
    for name, wrapper in wrappers:
        if use_color:
            print(f"\n{'=' * 72}")
            print(f"{_BOLD}{name}{_RESET}")
            print(f"{'=' * 72}")
        else:
            print(f"\n{'=' * 72}")
            print(name)
            print(f"{'=' * 72}")

        for label, text in samples:
            output = visualize_tokens(name, text, wrapper, use_color, label=label)
            print(output)
        print()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
