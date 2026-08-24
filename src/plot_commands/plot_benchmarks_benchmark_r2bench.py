import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from plot_commands.constants import (
    FAMILY_COLORS,
    FAMILY_LINESTYLES,
    STRATEGY_EXCLUDE as _BASE_STRATEGY_EXCLUDE,
    STRATEGY_NAMES,
    STRATEGY_ORDER,
)
from plot_commands.plotting_function.plot_auc_vs_cost import (
    compute_statistics,
    draw_family_lines_on_ax,
    get_strategy_family,
    load_all_data_with_k,
)
from plot_commands.plotting_function.plot_final_performance import (
    compute_final_performance_statistics,
    load_final_global_performance_with_k,
)
from plot_commands.plotting_function.plot_latex_table import generate_latex_table  # type: ignore
from plot_commands.plotting_function.utils import (
    apply_standard_filters,
    get_best_k_stats,
    get_colors_per_strategy,
    print_statistics_table,
    setup_paper_rcparams,
)
from plot_commands.plot_benchmarks_sparsity_exploration import (
    _cover_plot_data,
    _draw_donut_ax,
    _draw_tsne_cover_ax,
    _draw_tsne_scalar_ax,
    load_benchmark,
)
from logging_config import logger

BENCHMARK = "R2Bench"
OUT_DIR = Path("plots_benchmarks") / "r2bench"

setup_paper_rcparams()

# R2Bench-specific exclusions
EXTRA_EXCLUDE = {
    "oracle_ds",
    "oracle_ds_100",
    "oracle_ds_200",
    "oracle_ds_500",
}
STRATEGY_EXCLUDE = _BASE_STRATEGY_EXCLUDE | EXTRA_EXCLUDE

RESULTS_DIR_TEMPLATE = (
    "results/{penalty}/stationary/snowflake-arctic-embed-m-v2.0/R2Bench"
)

_POINT_SIZE = 5
_TITLE_KW: dict[str, Any] = {"fontsize": 10, "fontweight": "bold"}


def get_results_dir(penalty: float) -> str:
    """Return the R2Bench results directory for a given cost penalty."""
    return RESULTS_DIR_TEMPLATE.format(penalty=penalty)


def _load_filtered_stats(
    penalty: float, strategy_prefixes: list[str] | None = None
) -> dict[str, dict[str, float | int]]:
    """Load the best-k stats for one penalty, minus excluded strategies."""
    best_stats = get_best_k_stats(get_results_dir(penalty))
    return apply_standard_filters(
        best_stats, strategy_prefixes, extra_exclude=EXTRA_EXCLUDE
    )


def _draw_line_on_ax(
    ax: Axes,
    penalty: float,
    strategy_prefixes: list[str] | None = None,
) -> None:
    """Draw the line plot on a given axes."""
    family_order: list[str] = []
    for s in STRATEGY_ORDER:
        family, _ = get_strategy_family(s)
        if family not in family_order:
            family_order.append(family)

    best_stats = _load_filtered_stats(penalty, strategy_prefixes)

    all_handles = draw_family_lines_on_ax(
        ax,
        best_stats,
        family_order=family_order,
        colors={
            f: FAMILY_COLORS.get(f, get_colors_per_strategy([f])[0])
            for f in family_order
        },
        labels={f: STRATEGY_NAMES.get(f, f) for f in family_order},
        linestyles=FAMILY_LINESTYLES,
        markersize=8,
        annot_fontsize=9,
        annot_bold=True,
    )

    ax.set_xlabel("Size Corpus Annotated", fontsize=13, fontweight="bold")
    ax.set_ylabel("AUC Performance", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_title("Results on Stationary Stream", fontsize=10, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.3, linestyle="--")

    handles = [all_handles[f] for f in family_order if f in all_handles]
    labels = [STRATEGY_NAMES.get(f, f) for f in family_order if f in all_handles]
    ax.legend(handles, labels, fontsize=8, frameon=True, title="Strategy family")


def _draw_gain_ax(ax: Axes, data: dict[str, Any]) -> None:
    """t-SNE coloured by routing gain (best-vs-default gap)."""
    _draw_tsne_scalar_ax(
        ax,
        data["E_2d"],
        data["diff"],
        "Routing Gain",
        "Best-vs-Default Gap",
        s=_POINT_SIZE,
        title_kw=_TITLE_KW,
    )


def _draw_sparsity_ax(ax: Axes, data: dict[str, Any]) -> None:
    """t-SNE coloured by KL divergence from uniform Ψ(q)."""
    _draw_tsne_scalar_ax(
        ax,
        data["E_2d"],
        data["KL"],
        "Sparsity",
        "Ψ(q)",
        s=_POINT_SIZE,
        title_kw=_TITLE_KW,
    )


def _draw_cover_ax(ax: Axes, data: dict[str, Any], cd: dict[str, Any]) -> None:
    """t-SNE coloured by covering model."""
    _draw_tsne_cover_ax(
        ax,
        data["E_2d"],
        cd["qcm_sorted"],
        cd["colors"],
        "Cover Set Mapping",
        s=_POINT_SIZE,
        title_kw=_TITLE_KW,
    )


def _draw_composition_ax(ax: Axes, cd: dict[str, Any]) -> None:
    """Donut chart of cover-set composition."""
    _draw_donut_ax(
        ax,
        cd["pie_sizes_final"],
        cd["pie_colors"],
        cd["n_sel"],
        "Cover Set Composition",
        pct_fontsizes=(12, 9),
        center_fontsize=13,
        title_kw=_TITLE_KW,
    )


def _print_cover_counts(cd: dict[str, Any]) -> None:
    """Log the per-model cover-set counts and their share of the corpus."""
    logger.info("\n  Cover set model counts:")
    logger.info(f"  {'Model':<40s} {'Count':>6s} {'%':>6s}")
    logger.info(f"  {'-' * 52}")
    total = sum(cd["pie_sizes_sorted"])
    for label, count in zip(cd["pie_labels_sorted"], cd["pie_sizes_sorted"]):
        logger.info(f"  {label:<40s} {count:>6d} {count / total * 100:>5.1f}%")


def _save_fig(fig: Figure, out_file: Path) -> None:
    """Tighten, save at 300 dpi and close the figure."""
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    logger.info(f"  Saved: {out_file}")
    plt.close(fig)


def plot_combined(
    data: dict[str, Any],
    penalty: float,
    output_dir: Path,
    strategy_prefixes: list[str] | None = None,
) -> None:
    """2×3 figure: row 0 — line plot, routing gain t-SNE, KL t-SNE; row 1 — cover set t-SNE, donut (centred)."""
    cd = _cover_plot_data(data)

    fig = plt.figure(figsize=(18, 10))

    ax00 = fig.add_subplot(2, 3, 1)
    ax01 = fig.add_subplot(2, 3, 2)
    ax02 = fig.add_subplot(2, 3, 3)

    ax10 = fig.add_subplot(2, 3, 5)
    ax11 = fig.add_subplot(2, 3, 6)

    _draw_line_on_ax(ax00, penalty, strategy_prefixes)
    _draw_gain_ax(ax01, data)
    _draw_sparsity_ax(ax02, data)
    _draw_cover_ax(ax10, data, cd)
    _draw_composition_ax(ax11, cd)

    fig.add_subplot(2, 3, 4).set_visible(False)

    fig.suptitle("R2-Bench", fontsize=14, fontweight="bold", y=1.005)
    _save_fig(fig, output_dir / f"r2bench_combined_penalty{penalty}.png")
    _print_cover_counts(cd)


LATEX_TABLE_VARIANTS: set[str] = {
    "oracle_sparse_q25",
    "inferred_sparse_nn_opt_p50",
    "passive",
    "rand_0_25",
    "unc_mc_t3_q75",
    "act_repr_min_q25",
    "unc_t3_q75",
    "repr_vmf_kap1_q25",
}


def _filter_dict_by_strategies(
    data: dict[int, dict[str, Any]], allowed: set[str]
) -> dict[int, dict[str, Any]]:
    """Filter a {k: {strategy: ...}} dict to keep only allowed strategies."""
    return {
        k: {s: v for s, v in strats.items() if s in allowed}
        for k, strats in data.items()
    }


def print_latex_table(
    penalty: float,
    output_dir: Path,
) -> None:
    """Load experiment results for R2Bench, compute stats + significance, and generate LaTeX table."""
    results_dir = get_results_dir(penalty)
    logger.info(f"  Loading results from {results_dir}...")

    raw_per_k = _filter_dict_by_strategies(
        load_all_data_with_k(results_dir), LATEX_TABLE_VARIANTS
    )
    stats_per_k = {
        k: compute_statistics(strategies) for k, strategies in raw_per_k.items()
    }

    final_perf_data_per_k = _filter_dict_by_strategies(
        load_final_global_performance_with_k(results_dir), LATEX_TABLE_VARIANTS
    )
    final_stats_per_k = {
        k: compute_final_performance_statistics(data)
        for k, data in final_perf_data_per_k.items()
    }

    generate_latex_table(
        stats_per_k,
        final_stats_per_k,
        BENCHMARK,
        output_dir=str(output_dir),
        raw_per_k=raw_per_k,
        raw_final_per_k=final_perf_data_per_k,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot all R2Bench figures: line plot, cover set, t-SNE KL/gap, KL distribution."
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=0.0,
        help="Cost penalty value for the line plot (default: 0.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory to save figures (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--strategy-prefixes",
        type=str,
        nargs="+",
        default=None,
        help="Only include strategies starting with these prefixes in the line plot",
    )
    parser.add_argument(
        "--skip-line",
        action="store_true",
        help="Skip the line plot (requires experiment results)",
    )
    parser.add_argument(
        "--skip-sparsity",
        action="store_true",
        help="Skip sparsity plots (cover set, t-SNE, KL distribution) and only run the line plot",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip all figure generation and only regenerate the LaTeX table",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else OUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_figures:
        logger.info("\nGenerating LaTeX table …")
        print_latex_table(args.penalty, output_dir)
        logger.info(f"\nDone. Table saved to: {output_dir}")
        raise SystemExit(0)

    data = None
    if not args.skip_sparsity:
        logger.info("Loading R2Bench data …")
        data = load_benchmark(BENCHMARK)

    if not args.skip_line:
        print_statistics_table(
            _load_filtered_stats(args.penalty, args.strategy_prefixes), k_value=0
        )

    if not args.skip_line and not args.skip_sparsity and data is not None:
        logger.info("\nRendering combined figure …")
        plot_combined(
            data, args.penalty, output_dir, strategy_prefixes=args.strategy_prefixes
        )
    elif not args.skip_line and args.skip_sparsity:
        logger.info("\nRendering line plot only …")
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        _draw_line_on_ax(ax, args.penalty, strategy_prefixes=args.strategy_prefixes)
        _save_fig(fig, output_dir / f"r2bench_line_families_penalty{args.penalty}.png")
    elif args.skip_line and not args.skip_sparsity and data is not None:
        logger.info("\nRendering sparsity plots only …")
        cd = _cover_plot_data(data)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        _draw_gain_ax(axes[0], data)
        _draw_cover_ax(axes[1], data, cd)
        _draw_composition_ax(axes[2], cd)
        fig.suptitle("R2-Bench", fontsize=14, fontweight="bold", y=1.005)
        _save_fig(fig, output_dir / "r2bench_sparsity_only.png")
        _print_cover_counts(cd)

    if not args.skip_line:
        logger.info("\nGenerating LaTeX table …")
        print_latex_table(args.penalty, output_dir)

    logger.info(f"\nDone. All figures saved to: {output_dir}")
