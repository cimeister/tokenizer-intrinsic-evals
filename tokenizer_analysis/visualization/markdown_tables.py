"""
Markdown table generation for tokenizer analysis results.

Supports cumulative updates: new tokenizer rows are merged into an existing
results file so that a single table grows over successive runs.

Each combination of dataset and normalization method produces a separate
file, e.g. ``RESULTS_flores_bytes.md``.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import getpass
import logging
import os
import re

logger = logging.getLogger(__name__)

# Regex to detect old-format composite keys: "name (user, dataset)"
# Use greedy .+ so names containing parentheses (e.g. "Gemma 3 (512 codebook)")
# match correctly — the last "(user, dataset)" group is always the suffix.
_COMPOSITE_KEY_RE = re.compile(r'^(.+)\s*\(([^,]+),\s*([^)]+)\)$')

# Regex to detect new-format display names: "name [Nk]"
_DISPLAY_NAME_RE = re.compile(r'^(.+)\s*\[\d+k\]$')


def results_filename(
    dataset: str = "default",
    normalization_method: Optional[str] = None,
) -> str:
    """Return the markdown filename for a dataset / normalization-method pair.

    Examples
    --------
    >>> results_filename("flores", "bytes")
    'RESULTS_flores_bytes.md'
    >>> results_filename("flores")
    'RESULTS_flores.md'
    >>> results_filename()
    'RESULTS.md'
    """
    if dataset == "default" and normalization_method is None:
        return "RESULTS.md"
    parts = ["RESULTS"]
    if dataset and dataset != "default":
        parts.append(dataset)
    if normalization_method:
        parts.append(normalization_method)
    return "_".join(parts) + ".md"


def _format_vocab_tier(vocab_size: int) -> str:
    """Return a human-readable vocab-size tier label, e.g. ``'128k'``."""
    return f"{round(vocab_size / 1000)}k"


def _parse_float(s: str) -> Optional[float]:
    """Try to parse a formatted cell value to float, stripping bold markers and commas."""
    s = s.strip().replace('**', '').replace(',', '')
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _strip_arrow(header: str) -> str:
    """Remove trailing directional arrow (`` ↓`` or `` ↑``) from a header."""
    return re.sub(r'\s*[↓↑]$', '', header)


class MarkdownTableGenerator:
    """Generate and cumulatively update Markdown tables from tokenizer analysis results."""

    # Ordered list of metric configurations (determines column order).
    # ``lower_is_better``: True/False controls bolding direction;
    # None means informational (no bolding).
    # Defined at class level so standalone functions (e.g. generate_bar_plots_from_markdown)
    # can access them without a live instance.
    DEFAULT_METRIC_CONFIGS: List[Dict[str, Any]] = [
            {
                'key': 'vocab_size',
                'title': 'Vocab Size',
                'key_path': ['vocabulary_utilization', 'per_tokenizer'],
                'value_key': 'global_vocab_size',
                'stat_key': None,
                'format': '{:,}',
                'lower_is_better': None,
            },
            {
                'key': 'fertility',
                'title': 'Fertility',
                'key_path': ['fertility', 'per_tokenizer'],
                'value_key': 'global',
                'stat_key': 'mean',
                'format': '{:.3f}',
                'lower_is_better': True,
            },
            {
                'key': 'compression_rate',
                'title': 'Compression Rate',
                'key_path': ['compression_rate', 'per_tokenizer'],
                'value_key': 'global',
                'stat_key': 'compression_rate',
                'format': '{:.3f}',
                'lower_is_better': False,
            },
            {
                'key': 'vocabulary_utilization',
                'title': 'Vocab Util.',
                'key_path': ['vocabulary_utilization', 'per_tokenizer'],
                'value_key': 'global_utilization',
                'stat_key': None,
                'format': '{:.3f}',
                'lower_is_better': False,
            },
            {
                # Cross-lingual coefficient of variation of the per-language
                # vocabulary-utilization ratio (already computed in
                # basic.py; None for <2-language / mean<=0 tokenizers ->
                # rendered as the standard '---' placeholder). Lower = more
                # balanced vocabulary use across languages.
                'key': 'vocab_util_cross_lingual_cov',
                'title': 'Vocab Util. CoV',
                'key_path': ['vocabulary_utilization', 'per_tokenizer'],
                'value_key': 'per_language_cov',
                'stat_key': None,
                'format': '{:.3f}',
                'lower_is_better': True,
            },
            {
                'key': 'avg_token_rank',
                'title': 'Avg Token Rank',
                'key_path': ['unigram_distribution_metrics', 'per_tokenizer'],
                'value_key': 'global_avg_token_rank',
                'stat_key': None,
                'format': '{:.1f}',
                'lower_is_better': True,
            },
            {
                'key': 'tokenizer_fairness_gini',
                'title': 'Gini',
                'key_path': ['tokenizer_fairness_gini', 'per_tokenizer'],
                'value_key': 'gini_coefficient',
                'stat_key': None,
                'format': '{:.3f}',
                'lower_is_better': True,
            },
            {
                'key': 'bigram_entropy',
                'title': 'Bigram Ent.',
                'key_path': ['bigram_entropy', 'per_tokenizer'],
                'value_key': 'global_bigram_entropy',
                'stat_key': None,
                'format': '{:.3f}',
                'lower_is_better': False,
            },
            {
                'key': 'morphscore_recall',
                'title': 'MorphScore Recall',
                'key_path': ['morphscore', 'per_tokenizer'],
                'value_key': 'summary',
                'stat_key': 'avg_morphscore_recall',
                'format': '{:.3f}',
                'lower_is_better': False,
            },
            {
                'key': 'three_digit_boundary_f1',
                'title': '3-Digit Align. F1',
                'key_path': ['three_digit_boundary_alignment', 'summary'],
                'value_key': 'avg_f1',
                'stat_key': None,
                'format': '{:.3f}',
                'lower_is_better': False,
            },
            {
                'key': 'operator_isolation',
                'title': 'Op. Isolation',
                'key_path': ['operator_isolation_rate', 'summary'],
                'value_key': 'overall_isolation_rate',
                'stat_key': None,
                'format': '{:.3f}',
                'lower_is_better': False,
            },
            {
                'key': 'ast_full_alignment',
                'title': 'AST Align.',
                'key_path': ['ast_boundary_alignment', 'summary'],
                'value_key': 'avg_full_alignment_rate',
                'stat_key': None,
                'format': '{:.3f}',
                'lower_is_better': False,
            },
            {
                'key': 'ident_fragmentation',
                'title': 'Ident. Frag.',
                'key_path': ['identifier_fragmentation', 'summary'],
                'value_key': 'fragmentation_rate',
                'stat_key': None,
                'format': '{:.3f}',
                'lower_is_better': True,
            },
            {
                'key': 'indent_depth_corr',
                'title': 'Depth Corr.',
                'key_path': ['indentation_consistency', 'summary'],
                'value_key': 'avg_depth_proportionality_correlation',
                'stat_key': None,
                'format': '{:.3f}',
                'lower_is_better': False,
            },
            {
                'key': 'indent_pattern_stability',
                'title': 'Pat. Stability',
                'key_path': ['indentation_consistency', 'summary'],
                'value_key': 'avg_pattern_stability_rate',
                'stat_key': None,
                'format': '{:.3f}',
                'lower_is_better': False,
            },
            {
                'key': 'utf8_boundary_crossing',
                'title': 'Bound. Cross',
                'key_path': ['utf8_token_integrity', 'summary'],
                'value_key': 'boundary_crossing_rate',
                'stat_key': None,
                'format': '{:.4f}',
                'lower_is_better': True,
            },
            {
                'key': 'utf8_char_split',
                'title': 'Char Split',
                'key_path': ['utf8_char_split', 'summary'],
                'value_key': 'split_rate',
                'stat_key': None,
                'format': '{:.4f}',
                'lower_is_better': True,
            },
            {
                'key': 'mean_cer',
                'title': 'CER',
                'key_path': ['reconstruction_fidelity', 'summary'],
                'value_key': 'mean_cer',
                'stat_key': None,
                'format': '{:.4f}',
                'lower_is_better': True,
            },
            {
                'key': 'whitespace_fidelity',
                'title': 'WS Fidelity',
                'key_path': ['reconstruction_fidelity', 'summary'],
                'value_key': 'whitespace_fidelity',
                'stat_key': None,
                'format': '{:.3f}',
                'lower_is_better': False,
            },
            {
                'key': 'encode_time_ms',
                'title': 'Enc. ms/sample',
                'key_path': ['encoding_speed', 'per_tokenizer'],
                'value_key': 'mean_ms',
                'stat_key': None,
                'format': '{:.2f}',
                'lower_is_better': True,
            },
            {
                'key': 'num_languages',
                'title': 'Languages',
                'key_path': ['tokenizer_fairness_gini', 'per_tokenizer'],
                'value_key': 'num_languages',
                'stat_key': None,
                'format': '{:d}',
                'lower_is_better': None,
            },
        ]

    def __init__(self, results: Dict[str, Any], tokenizer_names: List[str]):
        self.results = results
        self.tokenizer_names = tokenizer_names
        self.metric_configs = list(self.DEFAULT_METRIC_CONFIGS)

    # ------------------------------------------------------------------
    # Value extraction / formatting
    # ------------------------------------------------------------------

    def _extract_metric_value(
        self, metric_config: Dict[str, Any], tokenizer_name: str
    ) -> Optional[Any]:
        """Navigate the results dict and return a single scalar value (or None)."""
        try:
            data = self.results
            for key in metric_config['key_path']:
                if key not in data:
                    return None
                data = data[key]

            if tokenizer_name not in data:
                return None

            tokenizer_data = data[tokenizer_name]

            if metric_config['stat_key']:
                value_data = tokenizer_data.get(metric_config['value_key'], {})
                if isinstance(value_data, dict):
                    return value_data.get(metric_config['stat_key'])
                return value_data
            else:
                return tokenizer_data.get(metric_config['value_key'])
        except Exception as e:
            logger.warning(
                f"Error extracting metric {metric_config['title']} "
                f"for {tokenizer_name}: {e}"
            )
            return None

    def _extract_vocab_size(self, tokenizer_name: str) -> Optional[int]:
        """Extract the vocab size for a tokenizer from the results."""
        cfg = next(
            (c for c in self.metric_configs if c['key'] == 'vocab_size'), None
        )
        if cfg is None:
            return None
        val = self._extract_metric_value(cfg, tokenizer_name)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _format_value(value: Any, format_str: str) -> str:
        """Format *value* with *format_str*, or return ``'---'`` when None."""
        if value is None:
            return '---'
        try:
            return format_str.format(value)
        except (ValueError, TypeError):
            return str(value)

    # ------------------------------------------------------------------
    # Best-value bolding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_best_row(
        rows: List[List[str]], col_index: int, lower_is_better: bool
    ) -> Optional[int]:
        """Return the row index with the best value in *col_index*, or None."""
        best_idx: Optional[int] = None
        best_val: Optional[float] = None
        for i, row in enumerate(rows):
            if col_index >= len(row):
                continue
            val = _parse_float(row[col_index])
            if val is None:
                continue
            if best_val is None:
                best_val = val
                best_idx = i
            elif lower_is_better and val < best_val:
                best_val = val
                best_idx = i
            elif not lower_is_better and val > best_val:
                best_val = val
                best_idx = i
        return best_idx

    @staticmethod
    def _apply_bolding_and_arrows(
        headers: List[str],
        rows: List[List[str]],
        metric_configs: List[Dict[str, Any]],
        active_titles: List[str],
    ) -> None:
        """Mutate *headers* and *rows* in-place to add bold markers and arrows.

        *active_titles* lists the metric titles that are currently present as
        columns (after any empty-column filtering).
        """
        # Build a title -> config lookup
        title_to_cfg = {c['title']: c for c in metric_configs}

        for col_idx, hdr in enumerate(headers):
            clean_hdr = _strip_arrow(hdr)
            cfg = title_to_cfg.get(clean_hdr)
            if cfg is None or cfg.get('lower_is_better') is None:
                continue
            lower = cfg['lower_is_better']

            # Add arrow to header
            arrow = '↓' if lower else '↑'
            headers[col_idx] = f"{clean_hdr} {arrow}"

            # Find and bold the best value
            best_idx = MarkdownTableGenerator._find_best_row(rows, col_idx, lower)
            if best_idx is not None:
                cell = rows[best_idx][col_idx]
                if not cell.startswith('**'):
                    rows[best_idx][col_idx] = f"**{cell}**"

    # ------------------------------------------------------------------
    # Sorting helper
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_rows(
        rows: List[List[str]],
        col_index: int,
        lower_is_better: bool,
    ) -> List[List[str]]:
        """Return *rows* sorted by the float value in *col_index*."""
        def sort_key(row):
            if col_index >= len(row):
                return (1, 0.0)  # sort to bottom
            val = _parse_float(row[col_index])
            if val is None:
                return (1, 0.0)
            return (0, val if lower_is_better else -val)
        return sorted(rows, key=sort_key)

    # ------------------------------------------------------------------
    # Empty-column filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_empty_columns(
        headers: List[str],
        rows: List[List[str]],
        metric_titles: List[str],
    ) -> Tuple[List[str], List[List[str]]]:
        """Remove metric columns where every row has ``'---'``.

        Only metric columns (those whose header is in *metric_titles*) are
        candidates for removal.  Meta columns (Tokenizer, Dataset, User, Date)
        are never removed.
        """
        # Find column indices that are entirely '---'
        cols_to_drop: set = set()
        for col_idx, hdr in enumerate(headers):
            clean = _strip_arrow(hdr)
            if clean not in metric_titles:
                continue
            all_empty = all(
                (col_idx >= len(row) or row[col_idx].strip().replace('**', '') == '---')
                for row in rows
            )
            if all_empty:
                cols_to_drop.add(col_idx)

        if not cols_to_drop:
            return headers, rows

        new_headers = [h for i, h in enumerate(headers) if i not in cols_to_drop]
        new_rows = [
            [cell for i, cell in enumerate(row) if i not in cols_to_drop]
            for row in rows
        ]
        return new_headers, new_rows

    # ------------------------------------------------------------------
    # Table generation
    # ------------------------------------------------------------------

    def generate_markdown_table(
        self,
        metrics: Optional[List[str]] = None,
        dataset: str = "default",
        normalization_method: Optional[str] = None,
        sort_by: Optional[str] = None,
    ) -> str:
        """Return a full Markdown document with one row per tokenizer.

        Parameters
        ----------
        metrics : list[str], optional
            Metric keys to include.  ``None`` means *all* configured metrics.
        dataset : str
            Dataset label for the composite key and Dataset column.
        normalization_method : str, optional
            Normalization method label (e.g. ``"bytes"``) included in the
            document title.
        sort_by : str, optional
            Metric key to sort rows by (e.g. ``"fertility"``).
        """
        configs = self._resolve_metrics(metrics)
        metric_titles = [c['title'] for c in configs]

        headers = ['Tokenizer'] + metric_titles + ['Dataset', 'User', 'Date']

        username = getpass.getuser()
        date_str = datetime.now().strftime('%Y-%m-%d')

        rows: List[List[str]] = []
        for tok_name in self.tokenizer_names:
            # Build display name with vocab tier
            vocab_size = self._extract_vocab_size(tok_name)
            if vocab_size is not None:
                display_name = f"{tok_name} [{_format_vocab_tier(vocab_size)}]"
            else:
                display_name = tok_name
            row = [display_name]
            for cfg in configs:
                value = self._extract_metric_value(cfg, tok_name)
                row.append(self._format_value(value, cfg['format']))
            row += [dataset, username, date_str]
            rows.append(row)

        # Filter empty metric columns
        headers, rows = self._filter_empty_columns(headers, rows, metric_titles)

        # Sort if requested
        if sort_by:
            sort_cfg = next(
                (c for c in configs if c['key'] == sort_by), None
            )
            if sort_cfg and sort_cfg.get('lower_is_better') is not None:
                col_idx = None
                for i, h in enumerate(headers):
                    if _strip_arrow(h) == sort_cfg['title']:
                        col_idx = i
                        break
                if col_idx is not None:
                    rows = self._sort_rows(
                        rows, col_idx, sort_cfg['lower_is_better']
                    )

        # Apply bolding and arrows
        self._apply_bolding_and_arrows(headers, rows, configs, metric_titles)

        separator = ['---'] * len(headers)
        title = self._build_title(dataset, normalization_method)
        return self._render_markdown(headers, separator, rows, title=title)

    # ------------------------------------------------------------------
    # Parsing an existing RESULTS.md
    # ------------------------------------------------------------------

    @staticmethod
    def parse_existing_markdown(
        filepath: str,
    ) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
        """Parse *filepath* and return ``(headers, rows_dict)``.

        ``rows_dict`` maps ``composite_key -> {column_title: cell_value}``.
        The composite key is ``"name (user, dataset)"`` regardless of the
        display format used in the Tokenizer column.

        Returns empty structures when the file doesn't exist or has no table.
        """
        path = Path(filepath)
        if not path.exists():
            return [], {}

        text = path.read_text(encoding='utf-8')
        lines = text.splitlines()

        # Find the header row (first line starting with '|')
        header_idx: Optional[int] = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('|') and '---' not in stripped:
                header_idx = i
                break

        if header_idx is None:
            return [], {}

        raw_headers = [
            h.strip() for h in lines[header_idx].strip().strip('|').split('|')
        ]
        # Strip arrow suffixes from headers for clean matching
        headers = [_strip_arrow(h) for h in raw_headers]

        # Skip separator line
        data_start = header_idx + 2

        rows_dict: Dict[str, Dict[str, str]] = {}
        for line in lines[data_start:]:
            stripped = line.strip()
            if not stripped.startswith('|'):
                break
            cells = [c.strip() for c in stripped.strip('|').split('|')]
            if not cells:
                continue
            tok_cell = cells[0]

            # Strip bold markers from all cell values
            clean_cells = [c.replace('**', '') for c in cells]

            # Determine composite key
            m = _COMPOSITE_KEY_RE.match(tok_cell)
            if m:
                # Old format: "name (user, dataset)" — use as-is
                composite_key = tok_cell.replace('**', '')
            else:
                # New format: "name [Nk]" — reconstruct composite key from
                # User and Dataset columns
                user_idx = None
                dataset_idx = None
                for j, hdr in enumerate(headers):
                    if hdr == 'User':
                        user_idx = j
                    elif hdr == 'Dataset':
                        dataset_idx = j

                if user_idx is not None and dataset_idx is not None and \
                   user_idx < len(clean_cells) and dataset_idx < len(clean_cells):
                    user_val = clean_cells[user_idx]
                    dataset_val = clean_cells[dataset_idx]
                    # Strip [Nk] suffix to get the base name
                    dm = _DISPLAY_NAME_RE.match(tok_cell.replace('**', ''))
                    base_name = dm.group(1).strip() if dm else tok_cell.replace('**', '').strip()
                    composite_key = f"{base_name} ({user_val}, {dataset_val})"
                else:
                    # Fallback: use the cell as-is
                    composite_key = tok_cell.replace('**', '')

            row_map: Dict[str, str] = {}
            for j, hdr in enumerate(headers):
                if j < len(clean_cells):
                    row_map[hdr] = clean_cells[j]
            rows_dict[composite_key] = row_map

        return headers, rows_dict

    # ------------------------------------------------------------------
    # Cumulative update
    # ------------------------------------------------------------------

    def update_markdown_file(
        self,
        filepath: str,
        metrics: Optional[List[str]] = None,
        dataset: str = "default",
        normalization_method: Optional[str] = None,
        sort_by: Optional[str] = None,
    ) -> str:
        """Merge current results into an existing results file (or create it).

        * Existing tokenizer rows not in the current run are preserved.
        * Existing tokenizer rows that *are* in the current run are updated.
        * New tokenizers are appended.
        * Column order follows the current metric config; extra columns from
          the old file that aren't in the current config are appended at the
          end.

        Parameters
        ----------
        filepath : str
            Path to the Markdown file.
        metrics : list[str], optional
            Metric keys to include.
        dataset : str
            Dataset label for the composite key and Dataset column.
        normalization_method : str, optional
            Normalization method label (e.g. ``"bytes"``) included in the
            document title.
        sort_by : str, optional
            Metric key to sort rows by (e.g. ``"fertility"``).

        Returns the rendered Markdown string.
        """
        configs = self._resolve_metrics(metrics)
        current_titles = [c['title'] for c in configs]

        old_headers, old_rows = self.parse_existing_markdown(filepath)

        # ----- Determine final column list -----
        # "Tokenizer" is always first; then current metric titles; then any
        # extra columns from the old file that we don't know about;
        # "Dataset", "User" and "Date" are always last.
        all_titles = current_titles + ['Dataset', 'User', 'Date']
        headers = ['Tokenizer'] + all_titles

        username = getpass.getuser()
        date_str = datetime.now().strftime('%Y-%m-%d')

        # ----- Build rows dict (old preserved, current overwritten) -----
        # Row key is "tokenizer_name (user, dataset)" so different users'
        # or different datasets' results coexist; same user + dataset
        # re-running updates in place.
        merged: Dict[str, Dict[str, str]] = {}

        # Start with old rows (keyed by composite key from parse)
        for composite_key, row_map in old_rows.items():
            merged[composite_key] = dict(row_map)

        # Overwrite / add current-run rows using composite key
        for tok_name in self.tokenizer_names:
            composite_key = f'{tok_name} ({username}, {dataset})'
            vocab_size = self._extract_vocab_size(tok_name)
            if vocab_size is not None:
                display_name = f"{tok_name} [{_format_vocab_tier(vocab_size)}]"
            else:
                display_name = tok_name
            if composite_key not in merged:
                merged[composite_key] = {}
            merged[composite_key]['Tokenizer'] = display_name
            for cfg in configs:
                value = self._extract_metric_value(cfg, tok_name)
                merged[composite_key][cfg['title']] = self._format_value(
                    value, cfg['format']
                )
            merged[composite_key]['Dataset'] = dataset
            merged[composite_key]['User'] = username
            merged[composite_key]['Date'] = date_str

        # ----- Determine row ordering -----
        # Current-run tokenizers first (preserving order), then old-only ones.
        ordered_names: List[str] = [
            f'{n} ({username}, {dataset})' for n in self.tokenizer_names
        ]
        for name in merged:
            if name not in ordered_names:
                ordered_names.append(name)

        rows: List[List[str]] = []
        for composite_key in ordered_names:
            row_map = merged.get(composite_key, {})
            # Use the display name from the Tokenizer field if available,
            # otherwise fall back to the composite key
            display = row_map.get('Tokenizer', composite_key)
            row = [display]
            for title in all_titles:
                row.append(row_map.get(title, '---'))
            rows.append(row)

        # Filter empty metric columns
        metric_titles = current_titles
        headers, rows = self._filter_empty_columns(headers, rows, metric_titles)

        # Sort if requested
        if sort_by:
            sort_cfg = next(
                (c for c in configs if c['key'] == sort_by), None
            )
            if sort_cfg and sort_cfg.get('lower_is_better') is not None:
                col_idx = None
                for i, h in enumerate(headers):
                    if _strip_arrow(h) == sort_cfg['title']:
                        col_idx = i
                        break
                if col_idx is not None:
                    rows = self._sort_rows(
                        rows, col_idx, sort_cfg['lower_is_better']
                    )

        # Apply bolding and arrows
        self._apply_bolding_and_arrows(headers, rows, configs, metric_titles)

        separator = ['---'] * len(headers)
        title = self._build_title(dataset, normalization_method)
        md = self._render_markdown(headers, separator, rows, title=title)

        # Write the file
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding='utf-8')
        logger.info(f"Markdown results table saved to {filepath}")

        # Generate bar plots
        try:
            generate_bar_plots_from_markdown(filepath)
        except Exception as e:
            logger.warning(f"Bar plot generation failed: {e}")

        return md

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_metrics(
        self, metrics: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Return the list of metric configs to use."""
        if metrics is None:
            return list(self.metric_configs)
        key_map = {c['key']: c for c in self.metric_configs}
        resolved = [key_map[m] for m in metrics if m in key_map]
        if not resolved:
            logger.warning("No valid metrics specified; using all defaults")
            return list(self.metric_configs)
        return resolved

    @staticmethod
    def _build_title(
        dataset: str = "default",
        normalization_method: Optional[str] = None,
    ) -> str:
        """Build a descriptive document title from dataset / method."""
        title = "Tokenizer Evaluation Results"
        parts: List[str] = []
        if dataset and dataset != "default":
            parts.append(dataset)
        if normalization_method:
            parts.append(normalization_method)
        if parts:
            title += " — " + " / ".join(parts)
        return title

    @staticmethod
    def _render_markdown(
        headers: List[str],
        separator: List[str],
        rows: List[List[str]],
        title: str = "Tokenizer Evaluation Results",
    ) -> str:
        """Render a complete Markdown document with header, timestamp, and table."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines = [
            f'# {title}',
            '',
            f'_Last updated: {timestamp}_',
            '',
            '| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join(separator) + ' |',
        ]
        for row in rows:
            lines.append('| ' + ' | '.join(row) + ' |')
        # Trailing newline
        lines.append('')
        return '\n'.join(lines)


def _plots_dir_for_results_file(md_filepath: str) -> str:
    """Return the plot directory path for a given results markdown file.

    ``RESULTS_flores_core_lines.md`` → ``…/flores_core_lines/``
    ``RESULTS.md`` → ``…/default/``
    """
    parent = str(Path(md_filepath).parent)
    stem = Path(md_filepath).stem  # e.g. "RESULTS_flores_core_lines"
    if stem == "RESULTS":
        folder_name = "default"
    else:
        folder_name = stem.replace("RESULTS_", "", 1)
    return os.path.join(parent, folder_name)


def _truncate_name(name: str, max_len: int = 30) -> str:
    """Truncate a tokenizer display name for plot labels."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "\u2026"


def generate_bar_plots_from_markdown(md_filepath: str) -> Optional[str]:
    """Parse a ``RESULTS*.md`` file and generate one horizontal bar plot per metric column.

    Returns the plot directory path, or ``None`` when no plots could be created.
    """
    import matplotlib
    matplotlib.use('Agg')

    headers, rows_dict = MarkdownTableGenerator.parse_existing_markdown(md_filepath)
    if not headers or not rows_dict:
        return None

    plot_dir = _plots_dir_for_results_file(md_filepath)
    os.makedirs(plot_dir, exist_ok=True)

    # Build config lookup for lower_is_better info
    title_to_cfg = {c['title']: c for c in MarkdownTableGenerator.DEFAULT_METRIC_CONFIGS}

    # Columns to skip (non-metric or informational)
    skip = {'Tokenizer', 'Dataset', 'User', 'Date'}

    # Collect tokenizer names and display names
    tok_names = list(rows_dict.keys())
    display_names = []
    for name in tok_names:
        row = rows_dict[name]
        display = row.get('Tokenizer', name)
        display_names.append(_truncate_name(display))

    for hdr in headers:
        clean_hdr = _strip_arrow(hdr)
        if clean_hdr in skip:
            continue
        cfg = title_to_cfg.get(clean_hdr)
        # Skip informational columns (lower_is_better is None)
        if cfg and cfg.get('lower_is_better') is None:
            continue

        # Extract values
        values = []
        labels = []
        for name, display in zip(tok_names, display_names):
            raw = rows_dict[name].get(clean_hdr, '---')
            val = _parse_float(raw)
            if val is not None:
                values.append(val)
                labels.append(display)

        if not values:
            continue

        _make_bar_plot(labels, values, clean_hdr, cfg, plot_dir)

    return plot_dir


def _make_bar_plot(
    labels: List[str],
    values: List[float],
    title: str,
    cfg: Optional[Dict[str, Any]],
    plot_dir: str,
) -> None:
    """Create and save a single horizontal bar plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tokenizer_analysis.visualization.plots import get_colors, setup_plot_style

    setup_plot_style()

    n = len(values)
    colors = list(get_colors(n))

    # Highlight best value
    lower = cfg['lower_is_better'] if cfg else None
    if lower is not None:
        best_idx = values.index(min(values) if lower else max(values))
        colors[best_idx] = '#009988'  # Teal highlight for best

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * n + 1)))
    y_pos = range(n)
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10)

    arrow = ''
    if lower is True:
        arrow = ' \u2193'
    elif lower is False:
        arrow = ' \u2191'
    ax.set_title(f'{title}{arrow}', fontsize=14)
    ax.invert_yaxis()  # Top-to-bottom matches table order

    plt.tight_layout()
    slug = title.lower().replace(' ', '_').replace('.', '').replace('-', '_').replace('/', '_')
    path = os.path.join(plot_dir, f'{slug}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
