import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon as scipy_wilcoxon  # type: ignore
from statsmodels.stats.multitest import multipletests  # type: ignore

from plot_commands.constants import (
    STRATEGY_EXCLUDE,
    STRATEGY_NAMES_LATEX as STRATEGY_NAMES,
)
from plot_commands.plotting_function.plot_auc_vs_cost import (
    VARIANT_LABELS,
    get_strategy_family,
)
from logging_config import logger

# reference strategy
# REFERENCE_KEY = "inferred_sparse_nn_opt" for statistical significance with SparseNN
REFERENCE_KEY = "rand_0_25"

_SIG_METRIC_KEY: dict[str, str | None] = {
    "train_cost_final_mean": None,
    "auc_performance_mean": "auc_performance",
    "auc_dto_mean": "auc_dto",
    "final_perf_mean": "final_perf",
}

_FAMILY_ORDER: list[str] = [
    "oracle_sparse",
    "oracle_ds",
    "inferred_sparse_nn_opt",
    "passive",
    "rand",
    "unc_mc_t3",
    "act_repr_min",
    "unc_t3",
    "repr_vmf_kap1",
]

ORACLE_FAMILIES: set[str] = {"oracle_sparse", "oracle_ds"}


def _cache_path(output_dir: str, benchmark: str) -> Path:
    """Return the path to the JSON stats cache file for a given benchmark."""
    return Path(output_dir) / "stats_cache" / f"{benchmark}.json"


def compute_significance(
    raw_per_k: dict[int, dict[str, Any]],
    raw_final_per_k: dict[int, dict[str, Any]],
    reference_key: str = REFERENCE_KEY,
    alpha: float = 0.05,
) -> dict:
    """
    Wilcoxon signed-rank tests with Holm-Bonferroni correction comparing reference_key
    against all other strategies, per (k_value, metric).


    Args:
        raw_per_k: {k: {strategy: {"auc_performance": [...], "train_cost_final": [...], "auc_dto": [...]}}}
        raw_final_per_k: {k: {strategy: [vals_per_seed]}}
        reference_key: Internal key of the reference strategy.
        alpha: Family-wise error rate.

    Returns:
        {k_value: {strategy: {metric_name: bool}}} — True means significantly different.
    """
    strategy_family = get_strategy_family(reference_key)[0]

    result: dict = {}
    all_k = set(raw_per_k.keys()) | set(raw_final_per_k.keys())

    for k_value in all_k:
        k_raw = raw_per_k.get(k_value, {})
        k_final = raw_final_per_k.get(k_value, {})

        all_present = set(k_raw.keys()) | set(k_final.keys())
        actual_reference_key = next(
            (s for s in all_present if get_strategy_family(s)[0] == strategy_family),
            reference_key,
        )

        reference_raw = k_raw.get(actual_reference_key, {})
        reference_final = list(k_final.get(actual_reference_key, []))

        if not reference_raw and not reference_final:
            continue

        other_strategies = sorted(all_present - {actual_reference_key})

        metrics_to_test: list[tuple[str, list, dict | None]] = []
        if reference_raw.get("auc_performance"):
            metrics_to_test.append(
                ("auc_performance", reference_raw["auc_performance"], k_raw)
            )
        if reference_raw.get("auc_dto"):
            metrics_to_test.append(("auc_dto", reference_raw["auc_dto"], k_raw))
        if reference_final:
            metrics_to_test.append(("final_perf", reference_final, None))

        sig_k: dict = {s: {} for s in other_strategies}

        for metric_name, reference_vals_list, source in metrics_to_test:
            reference_arr = np.array(reference_vals_list)
            entries: list[tuple[str, float]] = []

            for strategy in other_strategies:
                if source is not None:
                    other_arr = np.array(source.get(strategy, {}).get(metric_name, []))
                else:
                    other_arr = np.array(k_final.get(strategy, []))

                min_len = min(len(reference_arr), len(other_arr))
                if min_len < 2:
                    continue

                s_vals = reference_arr[:min_len]
                o_vals = other_arr[:min_len]

                if np.all(s_vals == o_vals):
                    entries.append((strategy, 1.0))
                    continue

                try:
                    _, p = scipy_wilcoxon(s_vals, o_vals, alternative="two-sided")
                    entries.append((strategy, float(p)))
                except Exception:
                    entries.append((strategy, 1.0))

            if not entries:
                continue

            strats_tested, p_vals = zip(*entries)
            reject, _, _, _ = multipletests(list(p_vals), alpha=alpha, method="holm")  # type: ignore[no-untyped-call]
            for strategy, is_sig in zip(strats_tested, reject):
                sig_k[strategy][metric_name] = bool(is_sig)

        result[k_value] = sig_k

    return result


def _save_stats_cache(
    stats_per_k: dict[int, dict[str, Any]],
    final_stats_per_k: dict[int, dict[str, Any]],
    benchmark: str,
    output_dir: str,
    raw_per_k: dict[int, dict[str, Any]] | None = None,
    raw_final_per_k: dict[int, dict[str, Any]] | None = None,
) -> None:
    """Persist stats for one benchmark to a JSON cache file under output_dir/stats_cache/.

    Args:
        stats_per_k: {k: {strategy: {metric_key: value}}} — aggregated metrics (mean, std, …).
        final_stats_per_k: {k: {strategy: {metric_key: value}}} — final-step aggregated metrics.
        benchmark: Benchmark name used as the cache file stem.
        output_dir: Root output directory; cache is written to output_dir/stats_cache/<benchmark>.json.
        raw_per_k: Optional {k: {strategy: {metric_key: [per-seed values]}}} — stored for significance tests.
        raw_final_per_k: Optional {k: {strategy: [per-seed values]}} — raw final-step values per seed.
    """
    path = _cache_path(output_dir, benchmark)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "stats_per_k": {
            str(k): {
                strategy: {
                    mk: (float(mv) if mv is not None else None)
                    for mk, mv in metrics.items()
                }
                for strategy, metrics in strats.items()
            }
            for k, strats in stats_per_k.items()
        },
        "final_stats_per_k": {
            str(k): {
                strategy: {
                    mk: (float(mv) if mv is not None else None)
                    for mk, mv in metrics.items()
                }
                for strategy, metrics in strats.items()
            }
            for k, strats in final_stats_per_k.items()
        },
    }
    if raw_per_k is not None:
        payload["raw_per_k"] = {
            str(k): {
                strategy: {
                    mk: ([float(v) for v in mv] if isinstance(mv, list) else mv)
                    for mk, mv in metrics.items()
                }
                for strategy, metrics in strats.items()
            }
            for k, strats in raw_per_k.items()
        }
    if raw_final_per_k is not None:
        payload["raw_final_per_k"] = {
            str(k): {
                strategy: [float(v) for v in vals] for strategy, vals in strats.items()
            }
            for k, strats in raw_final_per_k.items()
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Stats cache saved to {path}")


def _load_all_caches(output_dir: str) -> dict[str, dict[str, Any]]:
    """Return {benchmark: {stats_per_k, final_stats_per_k, raw_per_k, raw_final_per_k}} for all cached benchmarks."""
    cache_dir = Path(output_dir) / "stats_cache"
    result: dict[str, dict] = {}
    if not cache_dir.exists():
        return result
    for f in sorted(cache_dir.glob("*.json")):
        benchmark = f.stem
        payload = json.loads(f.read_text(encoding="utf-8"))
        result[benchmark] = {
            "stats_per_k": {int(k): v for k, v in payload["stats_per_k"].items()},
            "final_stats_per_k": {
                int(k): v for k, v in payload["final_stats_per_k"].items()
            },
            "raw_per_k": {int(k): v for k, v in payload.get("raw_per_k", {}).items()},
            "raw_final_per_k": {
                int(k): v for k, v in payload.get("raw_final_per_k", {}).items()
            },
        }
    return result


def _fmt_highlighted(
    mean: float | None,
    decimals: int,
    highlight: str | None,
    significant: bool = False,
    multiplier: float = 1.0,
) -> str:
    """Format a cell value with optional bold/underline highlight and significance marker.

    Args:
        mean: The value to format, or None to render as "---".
        decimals: Number of decimal places.
        highlight: ``"bold"`` for ``\\textbf``, ``"underline"`` for ``\\underline``, or None.
        significant: If True, appends ``$^{**}$`` to mark statistical significance.
        multiplier: Scalar applied to *mean* before formatting (e.g. 100 for percentages).

    Returns:
        A LaTeX-formatted string ready to embed in a table cell.
    """
    if mean is None:
        return "---"
    formatted = f"{mean * multiplier:.{decimals}f}"
    if highlight == "bold":
        formatted = r"\textbf{" + formatted + "}"
    elif highlight == "underline":
        formatted = r"\underline{" + formatted + "}"
    if significant:
        formatted += r"$^{**}$"
    return formatted


def _rank_non_oracle_highlights(
    strategies: list[str],
    values_raw: list,
    oracle_set: set[str],
    lower_is_better: bool = False,
) -> dict[int, str]:
    """Return {index: 'bold'|'underline'} for the top-2 non-oracle values."""
    non_oracle = [
        (i, v)
        for i, (s, v) in enumerate(zip(strategies, values_raw))
        if s not in oracle_set and v is not None
    ]
    non_oracle.sort(key=lambda x: x[1], reverse=not lower_is_better)
    result: dict[int, str] = {}
    if len(non_oracle) >= 1:
        result[non_oracle[0][0]] = "bold"
    if len(non_oracle) >= 2:
        result[non_oracle[1][0]] = "underline"
    return result


def _strategy_col_header(strategy: str) -> str:
    """Return the LaTeX column header label for a strategy.

    Tries an exact lookup in STRATEGY_NAMES first, then falls back to the
    family base name with the variant label appended (e.g. ``Rand (10%)``).
    """
    if strategy in STRATEGY_NAMES:
        return STRATEGY_NAMES[strategy]
    family, _ = get_strategy_family(strategy)
    base = STRATEGY_NAMES.get(family, family.replace("_", r"\_"))
    variant_label = VARIANT_LABELS.get(strategy, "").replace("%", r"\%")
    if variant_label and variant_label not in ("base", "Full Budget"):
        return f"{base} ({variant_label})"
    return base


def _has_dto(all_caches: dict[str, dict[str, Any]]) -> bool:
    """Return True if any cached benchmark contains an ``auc_dto_mean`` metric."""
    for cache in all_caches.values():
        for strats in cache["stats_per_k"].values():
            for metrics in strats.values():
                if "auc_dto_mean" in metrics:
                    return True
    return False


def _all_k_values(all_caches: dict[str, dict[str, Any]]) -> list[int]:
    """Return the sorted list of all k values present across every cached benchmark."""
    k_set: set[int] = set()
    for cache in all_caches.values():
        k_set |= set(cache["stats_per_k"].keys())
    return sorted(k_set)


def _best_per_family(
    all_caches: dict[str, dict[str, Any]], k_value: int, strategies: set[str]
) -> set[str]:
    """For each strategy family, keep only the variant with the highest mean
    auc_performance_mean averaged across all benchmarks. Strategies with no
    auc_performance_mean (e.g. only in final_stats) are kept as-is."""
    family_best: dict[str, tuple[str, float]] = {}  # family -> (strategy, score)
    scores: dict[str, list[float]] = {}

    for strategy in strategies:
        vals: list[float] = []
        for cache in all_caches.values():
            v = (
                cache["stats_per_k"]
                .get(k_value, {})
                .get(strategy, {})
                .get("auc_performance_mean")
            )
            if v is not None:
                vals.append(v)
        scores[strategy] = vals

    for strategy in strategies:
        family, _ = get_strategy_family(strategy)
        mean_score = float(np.mean(scores[strategy])) if scores[strategy] else None
        if mean_score is None:
            # No auc data — always keep (e.g. passive, oracles without variants)
            family_best.setdefault(family, (strategy, -1.0))
            continue
        if family not in family_best or mean_score > family_best[family][1]:
            family_best[family] = (strategy, mean_score)

    return {s for s, _ in family_best.values()}


def _all_strategies_sorted(
    all_caches: dict[str, dict[str, Any]], k_value: int
) -> list[str]:
    """Return strategies sorted by family order (_FAMILY_ORDER).
    Only the best-performing variant per family is kept.
    Families absent from _FAMILY_ORDER are appended alphabetically at the end."""
    present: set[str] = set()
    for cache in all_caches.values():
        present |= set(cache["stats_per_k"].get(k_value, {}).keys())
        present |= set(cache["final_stats_per_k"].get(k_value, {}).keys())

    present -= STRATEGY_EXCLUDE
    present = _best_per_family(all_caches, k_value, present)

    def _sort_key(s: str) -> tuple[int, str]:
        family, _ = get_strategy_family(s)
        pos = (
            _FAMILY_ORDER.index(family)
            if family in _FAMILY_ORDER
            else len(_FAMILY_ORDER)
        )
        return (pos, s)

    return sorted(present, key=_sort_key)


def _build_table_lines(
    all_caches: dict[str, dict[str, Any]],
    k_value: int,
    has_dto: bool,
    corpus_size_as_row: bool = True,
    sig_per_benchmark: dict[str, dict[int, dict]] | None = None,
) -> list[str]:
    """Build and return LaTeX lines for one combined table at k_value (does not write to disk).

    Layout: strategies as columns, metrics as rows, benchmarks as row groups.
    Cells significant vs. SparseNN (Wilcoxon + Holm-Bonferroni) are marked $^{**}$.
    """
    strategies = _all_strategies_sorted(all_caches, k_value)
    oracles = [s for s in strategies if get_strategy_family(s)[0] in ORACLE_FAMILIES]
    rest = [s for s in strategies if get_strategy_family(s)[0] not in ORACLE_FAMILIES]
    strategies = oracles + rest
    n_oracles = len(oracles)
    n_strats = len(strategies)

    metrics: list[tuple[str, str, int, bool, float]] = [
        *(
            ([("Corpus Size", "train_cost_final_mean", 0, True, 1.0)])
            if corpus_size_as_row
            else []
        ),
        ("AUC Perf", "auc_performance_mean", 1, False, 100.0),
        *(([("AUC DTO", "auc_dto_mean", 1, True, 100.0)]) if has_dto else []),
        ("Final Perf", "final_perf_mean", 1, False, 100.0),
    ]

    n_seeds = 0
    for cache in all_caches.values():
        for strats in cache["stats_per_k"].get(k_value, {}).values():
            n_seeds = max(n_seeds, int(strats.get("n_samples", 0)))

    label = f"tab:combined_k{k_value}"
    caption = (
        f"Combined results across all benchmarks (k={k_value}, {n_seeds} seeds). "
        r"$^{**}$ marks strategies significantly different from reference key "
        r"(Wilcoxon signed-rank, Holm-Bonferroni $\alpha=0.05$)."
    )

    n_cols = 1 + n_strats
    col_spec = "l" + "c" * n_oracles + "|" + "c" * (n_strats - n_oracles)

    strategy_headers = " & ".join(_strategy_col_header(s) for s in strategies)
    header = r"\textbf{Metric} & " + strategy_headers + r" \\"

    lines: list[str] = [
        r"\begin{table*}",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \hline",
        f"    {header}",
        r"    \hline",
    ]

    first_benchmark = True
    for benchmark, cache in all_caches.items():
        stats = cache["stats_per_k"].get(k_value, {})
        final_stats = cache["final_stats_per_k"].get(k_value, {})
        sig_data = (sig_per_benchmark or {}).get(benchmark, {}).get(k_value, {})

        if not stats and not final_stats:
            continue

        if not first_benchmark:
            lines.append(r"    \hline")
        first_benchmark = False

        if not corpus_size_as_row:
            corpus_sizes = [
                stats.get(s, {}).get("train_cost_final_mean")
                for s in strategies
                if stats.get(s, {}).get("train_cost_final_mean") is not None
            ]
            corpus_n = f"{np.mean(corpus_sizes):.0f}" if corpus_sizes else "?"
            sep = f"    \\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{benchmark.capitalize()}}} (N={corpus_n})}} \\\\"
        else:
            sep = f"    \\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{benchmark.capitalize()}}}}} \\\\"
        lines.append(sep)

        for metric_label, data_key, dec, lower_is_better, multiplier in metrics:
            sig_metric = _SIG_METRIC_KEY.get(data_key)
            values_raw = []
            for strategy in strategies:
                if data_key == "final_perf_mean":
                    val = final_stats.get(strategy, {}).get(data_key)
                else:
                    val = stats.get(strategy, {}).get(data_key)
                values_raw.append(val)

            highlights = _rank_non_oracle_highlights(
                strategies, values_raw, set(oracles), lower_is_better
            )

            reference_family = get_strategy_family(REFERENCE_KEY)[0]
            sig_flags: list[bool] = []
            for strategy in strategies:
                is_sparsenn = get_strategy_family(strategy)[0] == reference_family
                if sig_metric is not None and not is_sparsenn:
                    sig_flags.append(
                        bool(sig_data.get(strategy, {}).get(sig_metric, False))
                    )
                else:
                    sig_flags.append(False)

            formatted = [
                _fmt_highlighted(v, dec, highlights.get(i), sig_flags[i], multiplier)
                for i, v in enumerate(values_raw)
            ]
            row = f"    {metric_label} & " + " & ".join(formatted) + r" \\"
            lines.append(row)

    lines += [
        r"    \hline",
        r"  \end{tabular}",
        f"  \\caption{{\\label{{{label}}}",
        f"    {caption}",
        r"  }",
        r"\end{table*}",
        "",
    ]
    return lines


def _write_tex_file(
    all_caches: dict[str, dict[str, Any]],
    output_dir: str,
    corpus_size_as_row: bool = True,
    k_value_filter: int | None = None,
    reference_key: str = REFERENCE_KEY,
    alpha: float = 0.05,
) -> None:
    """Recompute significance tests and write the combined LaTeX table to disk.

    Iterates over all k values present in the caches (or just *k_value_filter*
    if provided), computes per-benchmark Wilcoxon significance against
    *REFERENCE_KEY*, and writes ``benchmark_table_review.tex`` to *output_dir*.

    Args:
        all_caches: Loaded benchmark caches as returned by ``_load_all_caches``.
        output_dir: Directory where the ``.tex`` file is written.
        corpus_size_as_row: When True, include a "Corpus Size" metric row.
        k_value_filter: If set, only generate the table for that k value.
        REFERENCE_KEY: Internal strategy key used as the significance reference.
        alpha: Family-wise error rate for Holm-Bonferroni correction.
    """
    has_dto = _has_dto(all_caches)
    all_k = _all_k_values(all_caches)
    k_values = (
        [k_value_filter]
        if k_value_filter is not None and k_value_filter in all_k
        else all_k
    )

    lines: list[str] = [
        r"% Auto-generated combined benchmark table",
        "",
    ]
    for k_value in k_values:
        strategy_set = set(_all_strategies_sorted(all_caches, k_value))
        sig_per_benchmark: dict[str, dict] = {}
        for benchmark, cache in all_caches.items():
            raw_per_k: dict[int, dict[str, Any]] = cache.get("raw_per_k", {})
            raw_final_per_k: dict[int, dict[str, Any]] = cache.get(
                "raw_final_per_k", {}
            )
            if raw_per_k or raw_final_per_k:
                filtered_raw: dict[int, dict[str, Any]] = {
                    k: {s: v for s, v in strats.items() if s in strategy_set}
                    for k, strats in raw_per_k.items()
                }
                filtered_raw_final: dict[int, dict[str, Any]] = {
                    k: {s: v for s, v in strats.items() if s in strategy_set}
                    for k, strats in raw_final_per_k.items()
                }
                sig_per_benchmark[benchmark] = compute_significance(
                    filtered_raw,
                    filtered_raw_final,
                    reference_key=reference_key,
                    alpha=alpha,
                )
            elif cache.get("sig_per_k"):
                sig_per_benchmark[benchmark] = {
                    int(k): v for k, v in cache["sig_per_k"].items()
                }
        lines += _build_table_lines(
            all_caches, k_value, has_dto, corpus_size_as_row, sig_per_benchmark
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tex_file = output_path / "benchmark_table_review.tex"
    tex_file.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Combined LaTeX table written to {tex_file}")


def generate_latex_table(
    stats_per_k: dict[int, dict[str, Any]],
    final_stats_per_k: dict[int, dict[str, Any]],
    benchmark: str,
    output_dir: str = "plots_benchmarks",
    raw_per_k: dict[int, dict[str, Any]] | None = None,
    raw_final_per_k: dict[int, dict[str, Any]] | None = None,
    reference_key: str = REFERENCE_KEY,
    alpha: float = 0.05,
    corpus_size_as_row: bool = True,
) -> None:
    """Save current benchmark stats to cache, then regenerate the combined table."""
    _save_stats_cache(
        stats_per_k,
        final_stats_per_k,
        benchmark,
        output_dir,
        raw_per_k,
        raw_final_per_k,
    )
    all_caches = _load_all_caches(output_dir)
    _write_tex_file(
        all_caches,
        output_dir,
        corpus_size_as_row,
        reference_key=reference_key,
        alpha=alpha,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate benchmark_table.tex from cached stats."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing stats_cache/ (e.g. plots_benchmarks/stationary)",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=0,
        help="KNN k value to include in the table (default: 0 = all k values).",
    )
    parser.add_argument(
        "--corpus-size",
        action="store_true",
        default=False,
        help="Include corpus size row in the table (default: omitted).",
    )
    args = parser.parse_args()

    all_caches = _load_all_caches(args.output_dir)
    if not all_caches:
        logger.info(f"No cached benchmarks found in {args.output_dir}/stats_cache/")
        raise SystemExit(1)
    logger.info(f"Found cached benchmarks: {list(all_caches.keys())}")
    k_filter = args.knn_k if args.knn_k != 0 else None
    _write_tex_file(
        all_caches,
        args.output_dir,
        corpus_size_as_row=args.corpus_size,
        k_value_filter=k_filter,
    )
