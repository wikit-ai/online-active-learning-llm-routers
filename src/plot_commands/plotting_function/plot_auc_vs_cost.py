import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from sklearn.metrics import auc as sklearn_auc

from plot_commands.plotting_function.utils import (
    compute_mean_stderr,  # type: ignore
    get_colors_per_strategy,
    get_markers_per_strategy,
)
from logging_config import logger

plt.style.use("ggplot")

_KNOWN_FAMILIES = [
    "inferred_sparse_nn_opt",
    "oracle_sparse",
    "oracle_ds",
    "repr_vmf_kap1",
    "act_repr_min",
    "unc_mc_t3",
    "unc_t3",
    "rand",
    "passive",
]

VARIANT_LABELS: dict[str, str] = {
    # oracle_sparse family
    "oracle_sparse_q25": "25%",
    "oracle_sparse_q50": "50%",
    "oracle_sparse": "75%",
    "oracle_sparse_q90": "90%",
    # inferred_sparse_nn_opt family
    "inferred_sparse_nn_opt_p25": "25%",
    "inferred_sparse_nn_opt_p50": "50%",
    "inferred_sparse_nn_opt": "75%",
    "inferred_sparse_nn_opt_p90": "90%",
    # repr_vmf_kap1 family
    "repr_vmf_kap1": "5%",
    "repr_vmf_kap1_q15": "15%",
    "repr_vmf_kap1_q25": "25%",
    # act_repr_min family
    "act_repr_min": "5%",
    "act_repr_min_q15": "15%",
    "act_repr_min_q25": "25%",
    # unc_t3 family
    "unc_t3": "95%",
    "unc_t3_q75": "75%",
    "unc_t3_q85": "85%",
    # unc_mc_t3 family
    "unc_mc_t3": "95%",
    "unc_mc_t3_q75": "75%",
    "unc_mc_t3_q85": "85%",
    # randpm family
    "rand_0_05": "p=0.05",
    "rand_0_10": "p=0.10",
    "rand_0_15": "p=0.15",
    "rand_0_25": "p=0.25",
    # passive family
    "passive": "Full Budget",
    # oracle_ds family
    "oracle_ds_100": "n=100",
    "oracle_ds_200": "n=200",
    "oracle_ds_500": "n=500",
}


def get_variant_sort_key(strategy: str) -> float:
    """Numeric sort key from VARIANT_LABELS so line points follow 25->50->75->90 order."""
    label = VARIANT_LABELS.get(strategy, "")
    if label.endswith("%"):
        return float(label[:-1])
    if label.startswith("p="):
        return float(label[2:]) * 100
    if label.startswith("n="):
        return float(label[2:])
    return float("inf")


def get_strategy_family(strategy: str) -> tuple[str, str]:
    """Return (family_name, hyperparam_label) for a strategy name.

    The hyperparam label is the suffix after the family prefix, or "base"
    when the strategy name exactly matches the family (no suffix).
    """
    for family in _KNOWN_FAMILIES:
        if strategy == family:
            return family, "base"
        if strategy.startswith(family + "_"):
            suffix = strategy[len(family) + 1 :]
            return family, suffix
    return strategy, "base"


def load_all_data_with_k(results_dir: str) -> dict[int, dict[str, Any]]:
    """
    Load all run_*.json and auc_global_*.json files from the results directory,
    organized by k_value.

    Returns:
        dict: Nested dictionary structure: {k_value: {strategy: {metric: [values]}}}
    """

    results_path = Path(results_dir)
    data: dict[int, dict[str, Any]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # type: ignore

    run_files = list(results_path.rglob("run_*.json"))

    if not run_files:
        raise ValueError(f"No run files found in {results_dir}")

    logger.info(f"Found {len(run_files)} run files")

    for run_file in run_files:
        with open(run_file, "r", encoding="utf-8") as f:
            run_data = json.load(f)

        run_dir = run_file.parent
        timestamp = run_file.stem.replace("run_", "")

        auc_file = run_dir / f"run_{timestamp}/auc_global_{timestamp}.json"

        if not auc_file.exists():
            logger.error(
                f"Warning: No AUC file ({auc_file}) found for {run_file}, skipping..."
            )
            continue

        with open(auc_file, "r", encoding="utf-8") as f:
            auc_data = json.load(f)

        final_train_costs: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)  # type: ignore
        for result in run_data.get("streaming_results", []):
            k_value = result["k_value"]
            strategy = result["strategy"]
            step = result["step"]
            train_cost = result["train_cost"]

            if strategy not in final_train_costs[k_value]:
                final_train_costs[k_value][strategy] = {
                    "step": step,
                    "train_cost": train_cost,
                }
            elif step > final_train_costs[k_value][strategy]["step"]:
                final_train_costs[k_value][strategy] = {
                    "step": step,
                    "train_cost": train_cost,
                }

        for k_value_str, strategies_dict in auc_data.items():
            k_value = int(k_value_str)
            for strategy, auc_metrics in strategies_dict.items():
                if (
                    k_value in final_train_costs
                    and strategy in final_train_costs[k_value]
                ):
                    train_cost_final = final_train_costs[k_value][strategy]["train_cost"]  # type: ignore
                    auc_perf = auc_metrics["auc_performance"]
                    auc_dto = auc_metrics["auc_dto"]

                    data[k_value][strategy]["train_cost_final"].append(train_cost_final)  # type: ignore
                    data[k_value][strategy]["auc_performance"].append(auc_perf)  # type: ignore
                    if auc_dto is not None:
                        data[k_value][strategy]["auc_dto"].append(auc_dto)  # type: ignore

    return dict(data)  # type: ignore


def compute_statistics(
    data: dict[str, list[float]],
) -> dict[str, dict[str, float | int]]:
    """Compute mean and standard error for each strategy's metrics."""
    stats: dict[str, dict[str, float | int]] = {}

    for strategy, metrics in data.items():
        perf_mean, perf_se = compute_mean_stderr(metrics["auc_performance"])  # type: ignore
        cost_mean, cost_se = compute_mean_stderr(metrics["train_cost_final"])  # type: ignore

        stats[strategy] = {  # type: ignore
            "auc_performance_mean": perf_mean,
            "auc_performance_stderr": perf_se,
            "train_cost_final_mean": cost_mean,
            "train_cost_final_stderr": cost_se,
            "n_samples": len(metrics["auc_performance"]),  # type: ignore
        }

        if "auc_dto" in metrics and len(metrics["auc_dto"]) > 0:  # type: ignore
            dto_mean, dto_se = compute_mean_stderr(metrics["auc_dto"])  # type: ignore
            stats[strategy]["auc_dto_mean"] = dto_mean
            stats[strategy]["auc_dto_stderr"] = dto_se

    return stats


def scatter_plot_scoring_vs_count(
    stats: dict[str, dict[str, float | int]],
    scoring_function: str,
    output_path: str,
    title: str,
    X_label: str,
    figsize: tuple[int, int] = (12, 8),
):

    strategies: list[str] = []
    x_vals: list[float] = []
    y_vals: list[float] = []
    x_errs: list[float] = []
    y_errs: list[float] = []

    for strategy, data in stats.items():
        strategies.append(strategy)
        x_vals.append(data["train_cost_final_mean"])
        y_vals.append(data[f"{scoring_function}_mean"])
        x_errs.append(data["train_cost_final_stderr"])
        y_errs.append(data[f"{scoring_function}_stderr"])

    colors: list[str] = get_colors_per_strategy(strategies=strategies)
    markers: list[str] = get_markers_per_strategy(strategies=strategies)

    fig, ax = plt.subplots(figsize=figsize)  # type: ignore

    for i, strategy in enumerate(strategies):
        ax.errorbar(  # type: ignore
            x_vals[i],
            y_vals[i],
            xerr=x_errs[i],
            yerr=y_errs[i],
            fmt=markers[i],
            color=colors[i],
            markersize=10,
            capsize=5,
            capthick=2,
            label=strategy,
            alpha=0.7,
            linewidth=2,
        )

        ax.annotate(  # type: ignore
            strategy,
            (x_vals[i], y_vals[i]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=colors[i],
                alpha=0.2,
                edgecolor="none",
            ),
        )

    ax.set_xlabel("Size Corpus Annotated", fontsize=14, fontweight="bold")  # type: ignore
    ax.set_ylabel(X_label, fontsize=14, fontweight="bold")  # type: ignore
    ax.set_title(  # type: ignore
        title,
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle="--")  # type: ignore

    ax.legend(  # type: ignore
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        markerscale=0.8,
    )

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # type: ignore

    return fig, ax


def plot_auc_vs_cost_per_k(
    stats_per_k: dict[int, dict[str, dict[str, float | int]]],
    benchmark: str,
    n_seeds: int,
    output_dir: str = "plots_benchmarks/auc_perf_cost",
):
    """
    Create scatter plots for AUC performance vs cost, one per k_value.

    Args:
        stats_per_k: Dictionary with k_value -> strategy -> statistics
        benchmark: Benchmark name
        n_seeds: Number of seeds
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir) / benchmark
    output_path.mkdir(exist_ok=True, parents=True)

    for k_value, stats in stats_per_k.items():
        output_file = (
            output_path / f"auc_performance_vs_cost_k{k_value}_{benchmark}.png"
        )
        scatter_plot_scoring_vs_count(
            stats=stats,
            scoring_function="auc_performance",
            output_path=str(output_file),
            title=f"AUC Performance vs Size Corpus Annotated (k={k_value}, {benchmark.capitalize()})\n(Error bars show standard error across {n_seeds} seeds)",
            X_label="AUC Performance",
        )


def load_auc_from_steps_per_k(
    results_dir: str,
) -> dict[int, dict[str, dict[str, list[float]]]]:
    """
    Load run_*.json files and compute per-seed AUC (test_performance vs step,
    steps normalised to [0, 1]) and final train_cost for each (strategy, k_value).

    Returns:
        {k_value: {strategy: {"auc_performance": [...], "train_cost_final": [...]}}}
    """
    results_path = Path(results_dir)
    run_files = list(results_path.rglob("run_*.json"))
    if not run_files:
        raise ValueError(f"No run files found in {results_dir}")

    data: dict = defaultdict(
        lambda: defaultdict(lambda: {"auc_performance": [], "train_cost_final": []})
    )

    for run_file in run_files:
        with open(run_file, "r") as f:
            run_data = json.load(f)

        grouped: dict = defaultdict(list)
        for result in run_data.get("streaming_results", []):
            key = (result["strategy"], result["k_value"])
            grouped[key].append(
                (result["step"], result["test_performance"], result["train_cost"])
            )

        for (strategy, k_value), records in grouped.items():
            records.sort(key=lambda x: x[0])
            steps = np.array([r[0] for r in records])
            performances = np.array([r[1] for r in records])
            train_cost_final = float(records[-1][2])

            if len(steps) > 1 and steps.max() > steps.min():
                steps_norm = (steps - steps.min()) / (steps.max() - steps.min())
                auc_val = float(sklearn_auc(steps_norm, performances))
            else:
                auc_val = 0.0

            data[k_value][strategy]["auc_performance"].append(auc_val)
            data[k_value][strategy]["train_cost_final"].append(train_cost_final)

    return dict(data)


def plot_auc_vs_cost_best_k(
    results_dir: str,
    benchmark: str,
    n_seeds: int,
    output_dir: str = "plots_benchmarks/auc_perf_cost",
):
    """
    Scatter plot of AUC-under-performance-curve vs final train cost,
    keeping only the best k for each strategy.
    AUC is computed per seed from streaming results (steps normalised to [0, 1]).
    "Best k" = the k_value that maximises mean AUC for that strategy.

    Args:
        results_dir: Path to directory containing run_*.json files
        benchmark: Benchmark name
        n_seeds: Number of seeds (used in title)
        output_dir: Directory to save plots
    """
    raw = load_auc_from_steps_per_k(results_dir)
    stats_per_k = {k: compute_statistics(strategies) for k, strategies in raw.items()}

    all_strategies: set[str] = set()
    for stats in stats_per_k.values():
        all_strategies.update(stats.keys())

    best_stats: dict[str, dict[str, float | int]] = {}
    for strategy in sorted(all_strategies):
        best_k = max(
            (k for k, stats in stats_per_k.items() if strategy in stats),
            key=lambda k: stats_per_k[k][strategy]["auc_performance_mean"],  # type: ignore
        )
        best_stats[strategy] = stats_per_k[best_k][strategy]

    output_path = Path(output_dir) / benchmark
    output_path.mkdir(exist_ok=True, parents=True)
    output_file = output_path / f"auc_performance_vs_cost_best_k_{benchmark}.png"

    scatter_plot_scoring_vs_count(
        stats=best_stats,
        scoring_function="auc_performance",
        output_path=str(output_file),
        title=(
            f"AUC Performance vs Cost — Best k per Strategy ({benchmark.capitalize()})\n"
            f"(Error bars show standard error across {n_seeds} seeds)"
        ),
        X_label="AUC Performance (best k)",
    )


def plot_auc_dto_vs_cost_per_k(
    stats_per_k: dict[int, dict[str, dict[str, float | int]]],
    benchmark: str,
    n_seeds: int,
    output_dir: str = "plots_benchmarks/auc_dto_cost",
):
    """
    Create scatter plots for AUC DTO vs cost, one per k_value.

    Args:
        stats_per_k: Dictionary with k_value -> strategy -> statistics
        benchmark: Benchmark name
        n_seeds: Number of seeds
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir) / benchmark
    output_path.mkdir(exist_ok=True, parents=True)

    for k_value, stats in stats_per_k.items():
        output_file = output_path / f"auc_dto_vs_cost_k{k_value}_{benchmark}.png"
        scatter_plot_scoring_vs_count(
            stats=stats,
            scoring_function="auc_dto",
            output_path=str(output_file),
            title=f"AUC DTO vs Size Corpus Annotated (k={k_value}, {benchmark.capitalize()})\n(Error bars show standard error across {n_seeds} seeds)",
            X_label="AUC DTO",
        )


def draw_family_lines_on_ax(
    ax: Axes,
    stats: dict[str, dict[str, float | int]],
    family_order: list[str] | None = None,
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    linestyles: dict[str, str] | None = None,
    markersize: float = 7,
    annot_fontsize: float = 7,
    annot_bold: bool = False,
) -> dict[str, Line2D]:
    """Draw one annotated line per strategy family on `ax`, with a ±1 SE band.

    Strategies are grouped by family, ordered by ``get_variant_sort_key`` inside
    each family, and plotted as mean train cost vs mean AUC performance.

    Args:
        ax: Axes to draw on
        stats: {strategy: metric stats} for a single k value
        family_order: Families to draw, in order (default: sorted families present)
        colors: Per-family color (default: ``get_colors_per_strategy``)
        labels: Per-family legend label (default: the family name)
        linestyles: Per-family line style (default: solid)
        markersize: Marker size of the variant points
        annot_fontsize: Font size of the per-point hyperparameter labels
        annot_bold: Whether the point labels are bold

    Returns:
        {family: line handle} for the families actually drawn
    """
    families: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for strategy in stats:
        family, hyperparam = get_strategy_family(strategy)
        if family not in _KNOWN_FAMILIES:
            continue
        families[family].append((strategy, hyperparam))

    order = [f for f in family_order if f in families] if family_order else sorted(families)
    if colors is None:
        colors = dict(zip(order, get_colors_per_strategy(order)))

    handles: dict[str, Line2D] = {}
    for family in order:
        variants = sorted(families[family], key=lambda sv: get_variant_sort_key(sv[0]))
        color = colors[family]

        x_arr = np.array([stats[s]["train_cost_final_mean"] for s, _ in variants])
        y_arr = np.array([stats[s]["auc_performance_mean"] for s, _ in variants])
        y_err = np.array([stats[s]["auc_performance_stderr"] for s, _ in variants])

        (line,) = ax.plot(  # type: ignore
            x_arr,
            y_arr,
            color=color,
            linewidth=2,
            linestyle=(linestyles or {}).get(family, "-"),
            marker="o",
            markersize=markersize,
            label=(labels or {}).get(family, family),
            alpha=0.85,
            zorder=3,
        )
        handles[family] = line

        ax.fill_between(  # type: ignore
            x_arr, y_arr - y_err, y_arr + y_err, color=color, alpha=0.12, zorder=2
        )

        for (strategy_name, hyperparam), x, y in zip(variants, x_arr, y_arr):
            ax.annotate(  # type: ignore
                VARIANT_LABELS.get(strategy_name, hyperparam),
                (x, y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=annot_fontsize,
                fontweight="bold" if annot_bold else "normal",
                color=color,
                alpha=0.95,
            )

    return handles


def plot_auc_vs_cost_line_families(
    stats_per_k: dict[int, dict[str, dict[str, float | int]]],
    benchmark: str,
    n_seeds: int,
    output_dir: str = "plots_benchmarks/auc_vs_cost_line",
):
    """Line plot connecting hyperparameter variants within each strategy family.

    X axis: mean training cost (no error bars).
    Y axis: mean AUC performance with a colour band (±1 SE).
    Each point is labelled with its hyperparameter suffix; the legend shows
    only the family name.
    """
    output_path = Path(output_dir) / benchmark
    output_path.mkdir(exist_ok=True, parents=True)

    for k_value, stats in stats_per_k.items():
        fig, ax = plt.subplots(figsize=(12, 8))  # type: ignore

        draw_family_lines_on_ax(ax, stats)

        ax.set_xlabel("Training Cost (Size Corpus Annotated)", fontsize=13, fontweight="bold")  # type: ignore
        ax.set_ylabel("AUC Performance", fontsize=13, fontweight="bold")  # type: ignore
        ax.set_title(  # type: ignore
            f"AUC Performance vs Training Cost — Strategy Families (k={k_value}, {benchmark.capitalize()})\n"
            f"(Band = ±1 SE across {n_seeds} seeds, points = hyperparameter variants)",
            fontsize=14,
            fontweight="bold",
            pad=16,
        )
        ax.grid(True, alpha=0.3, linestyle="--")  # type: ignore
        ax.legend(  # type: ignore
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=9,
            framealpha=0.9,
            title="Strategy family",
        )

        plt.tight_layout()

        output_file = output_path / f"auc_vs_cost_line_k{k_value}_{benchmark}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")  # type: ignore
        plt.close(fig)
        print(f"Saved -> {output_file}")
