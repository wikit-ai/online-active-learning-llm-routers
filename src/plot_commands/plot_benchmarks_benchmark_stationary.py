import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_commands.plotting_function.plot_auc_vs_cost import (
    compute_statistics,
    load_auc_from_steps_per_k,
)
from plot_commands.plotting_function.utils import (
    get_colors_per_strategy,
    get_markers_per_strategy,
)

plt.style.use("ggplot")

STRATEGY_NAMES: dict[str, str] = {
    "inferred_sparse_nn_opt": "SparseNN",
    "oracle_sparse": "OracleSp",
    "oracle_ds_200": "OracleCov",
    "passive": "Passive",
    "rand_0_05": "Rand0.05",
    "rand_0_10": "Rand0.10",
    "rand_0_15": "Rand0.15",
    "rand_0_25": "Rand0.25",
    "unc_mc_t3": "a_rank",
    "act_repr_min": "a_min",
    "unc_t3": "a_var",
    "repr_vmf_kap1": "a_vMF",
}

STRATEGY_ORDER: list[str] = [
    "inferred_sparse_nn_opt",
    "oracle_sparse",
    "oracle_ds_200",
    "passive",
    "rand_0_05",
    "rand_0_10",
    "rand_0_15",
    "rand_0_25",
    "unc_mc_t3",
    "act_repr_min",
    "unc_t3",
    "repr_vmf_kap1",
]

STRATEGY_EXCLUDE: set[str] = {
    "passive_stop_at_115",
}

BENCHMARKS = ["sprout", "routerbench", "embedllm", "fusionbench"]
BENCHMARK_LABELS = {
    "sprout": "Sprout",
    "routerbench": "RouterBench",
    "embedllm": "EmbedLLM",
    "fusionbench": "FusionBench",
}


def get_results_dir(benchmark: str, penalty: float) -> str:
    dirs = {
        "sprout": f"results/{penalty}/stationary/snowflake-arctic-embed-m-v2.0/Sprout",
        "routerbench": f"results/{penalty}/stationary/snowflake-arctic-embed-m-v2.0/RouterBench",
        "embedllm": f"results/{penalty}/stationary/snowflake-arctic-embed-m-v2.0/EmbedLLM",
        "fusionbench": f"results/{penalty}/stationary/snowflake-arctic-embed-m-v2.0/FusionBench",
    }
    return dirs[benchmark]


def get_best_k_stats(results_dir: str) -> dict:
    """Load data and keep only best k per strategy (maximises mean AUC)."""
    raw = load_auc_from_steps_per_k(results_dir)
    stats_per_k = {k: compute_statistics(strategies) for k, strategies in raw.items()}

    all_strategies: set[str] = set()
    for stats in stats_per_k.values():
        all_strategies.update(stats.keys())

    best_stats: dict = {}
    for strategy in sorted(all_strategies):
        best_k = max(
            (k for k, stats in stats_per_k.items() if strategy in stats),
            key=lambda k: stats_per_k[k][strategy]["auc_performance_mean"],  # type: ignore
        )
        best_stats[strategy] = stats_per_k[best_k][strategy]

    return best_stats


def plot_all_benchmarks(
    penalty: float,
    output_dir: str = "plots_benchmarks/stationary",
    strategy_prefixes: list[str] | None = None,
):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), squeeze=False)
    axes_flat = axes.flatten()

    all_handles: dict[str, object] = {}

    for idx, benchmark in enumerate(BENCHMARKS):
        ax = axes_flat[idx]
        results_dir = get_results_dir(benchmark, penalty)

        print(f"Loading {benchmark} from {results_dir}...")
        try:
            best_stats = get_best_k_stats(results_dir)
        except ValueError as e:
            ax.text(
                0.5,
                0.5,
                f"No data\n{e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(BENCHMARK_LABELS[benchmark], fontsize=14, fontweight="bold")
            continue

        best_stats = {s: v for s, v in best_stats.items() if s not in STRATEGY_EXCLUDE}

        if strategy_prefixes:
            best_stats = {
                s: v
                for s, v in best_stats.items()
                if any(s.startswith(p) for p in strategy_prefixes)
            }

        ordered = [s for s in STRATEGY_ORDER if s in best_stats]
        remainder = sorted(s for s in best_stats if s not in STRATEGY_ORDER)
        strategies = ordered + remainder
        x_vals = [best_stats[s]["train_cost_final_mean"] for s in strategies]
        y_vals = [best_stats[s]["auc_performance_mean"] for s in strategies]
        x_errs = [best_stats[s]["train_cost_final_stderr"] for s in strategies]
        y_errs = [best_stats[s]["auc_performance_stderr"] for s in strategies]

        colors = get_colors_per_strategy(strategies)
        markers = get_markers_per_strategy(strategies)

        for i, strategy in enumerate(strategies):
            handle = ax.errorbar(
                x_vals[i],
                y_vals[i],
                xerr=x_errs[i],
                yerr=y_errs[i],
                fmt=markers[i],
                color=colors[i],
                markersize=13,
                capsize=5,
                capthick=2,
                label=strategy,
                alpha=0.8,
                linewidth=2,
            )
            if strategy not in all_handles:
                all_handles[strategy] = handle

            ax.annotate(
                STRATEGY_NAMES.get(strategy, strategy),
                (x_vals[i], y_vals[i]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
                alpha=0.95,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=colors[i],
                    alpha=0.2,
                    edgecolor="none",
                ),
            )

        ax.set_xlabel("Size Corpus Annotated", fontsize=13, fontweight="bold")
        ax.set_ylabel("AUC Performance", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(BENCHMARK_LABELS[benchmark], fontsize=16, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    out_file = output_path / f"all_benchmarks_auc_perf_vs_cost_penalty{penalty}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to {out_file}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot AUC Performance vs Corpus Annotated for all 4 stationary benchmarks in a 2x2 grid."
    )
    parser.add_argument(
        "--penalty",
        type=float,
        required=True,
        help="Cost penalty value (used to locate results directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots_benchmarks/stationary",
        help="Directory to save the figure (default: plots_benchmarks/stationary)",
    )
    parser.add_argument(
        "--strategy-prefixes",
        type=str,
        nargs="+",
        default=None,
        help="Only include strategies starting with these prefixes (e.g. 'rand_' 'oracle')",
    )

    args = parser.parse_args()

    plot_all_benchmarks(
        penalty=args.penalty,
        output_dir=args.output_dir,
        strategy_prefixes=args.strategy_prefixes,
    )

    print("\nDone!")
