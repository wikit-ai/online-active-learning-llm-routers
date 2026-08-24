"""Plot AUC Perf vs annotation cost: Wilcoxon partial annotation vs full baseline.

Full-annotation baselines are loaded from main_stationary results; their
annotation cost is computed as n_selected * sum(model_costs) using costs
loaded from the HuggingFace dataset.
Wilcoxon variants are loaded from partial_annotation results where the
cumulative annotation cost is tracked per step.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset  # type: ignore
from sklearn.metrics import auc as sklearn_auc

from datasets_management.stationary_dataset import DatasetManagement, min_max_normalize

from plot_commands.constants import BENCHMARKS, BENCHMARK_LABELS

plt.style.use("ggplot")

BENCHMARK_HF_SPLITS = BENCHMARK_LABELS  # same mapping

BASE_STRATEGIES = [
    "inferred_sparse_nn_opt_p25",
    "inferred_sparse_nn_opt_p50",
    "inferred_sparse_nn_opt",
    "inferred_sparse_nn_opt_p90",
]

VARIANT_LABELS = {
    "inferred_sparse_nn_opt": "75%",
    "inferred_sparse_nn_opt_p25": "25%",
    "inferred_sparse_nn_opt_p50": "50%",
    "inferred_sparse_nn_opt_p90": "90%",
}

COLORS = {
    "full": "C0",
    "wilcoxon": "C1",
    "heuristics": "gray",
}

# Heuristic strategies from main_stationary to show as grey reference points
HEURISTIC_PREFIXES = [
    "unc_mc_t3",
    "act_repr_min",
    "unc_t3",
    "repr_vmf_kap1",
]


# ── Cost from HuggingFace ────────────────────────────────────────────────────


def _get_full_annotation_cost_per_query(benchmark: str) -> float:
    """Sum of all model costs for one query, using the same pipeline as
    :class:`DatasetManagement` (get_cost_mapping -> clean_normalise_costs
    -> min_max_normalize).
    """
    split = BENCHMARK_HF_SPLITS[benchmark]
    print(f"  Loading cost dataset from HuggingFace (split={split})...")
    ds_cost = load_dataset("Wikit/RoutingCompendium-cost", split=split)

    # Reuse DatasetManagement methods to get normalised costs
    dm = object.__new__(DatasetManagement)  # lightweight — skip __init__
    dm.benchmark = split  # type: ignore[attr-defined]
    raw_mapping = dm.get_cost_mapping(dataset_cost=ds_cost)
    cost_mapping = dm.clean_normalise_costs(raw_mapping)
    normalised = min_max_normalize(list(cost_mapping.cost_map.values()))

    total = float(np.sum(normalised))
    print(f"  {len(cost_mapping.cost_map)} models:")
    for (name, dollar), norm in zip(cost_mapping.cost_map.items(), normalised):
        print(f"    {name:<45} dollar=${dollar:.4f}  normalised={norm:.4f}")
    print(f"  Full annotation cost per query (sum of normalised) = {total:.4f}")
    return total


# ── Result directory helpers ──────────────────────────────────────────────────


def _stationary_dir(benchmark: str, penalty: float) -> str:
    from plot_commands.constants import get_results_dir
    return get_results_dir(benchmark, penalty, "stationary")


def _partial_dir(benchmark: str, penalty: float) -> str:
    from plot_commands.constants import get_results_dir
    return get_results_dir(benchmark, penalty, "partial_annotation")


# ── Loaders ───────────────────────────────────────────────────────────────────


def _load_runs(results_dir: str, cost_key: str | None = None) -> dict:
    """Load per-seed AUC perf and cost from run JSONs.

    Returns {k: {strategy: {"auc_performance": [...], "cost_final": [...]}}}
    """
    results_path = Path(results_dir)
    run_files = list(results_path.rglob("run_*.json"))
    if not run_files:
        raise ValueError(f"No run files found in {results_dir}")

    print(f"  Found {len(run_files)} run files in {results_dir}")

    data: dict = defaultdict(
        lambda: defaultdict(lambda: {"auc_performance": [], "cost_final": []})
    )

    for run_file in run_files:
        with open(run_file, "r", encoding="utf-8") as f:
            run_data = json.load(f)

        grouped: dict = defaultdict(list)
        for r in run_data.get("streaming_results", []):
            grouped[(r["strategy"], r["k_value"])].append(r)

        for (strategy, k_value), records in grouped.items():
            records.sort(key=lambda x: x["step"])
            steps = np.array([r["step"] for r in records])
            perfs = np.array([r["test_performance"] for r in records])

            if len(steps) > 1 and steps.max() > steps.min():
                steps_norm = (steps - steps.min()) / (steps.max() - steps.min())
                auc_val = float(sklearn_auc(steps_norm, perfs))
            else:
                auc_val = 0.0

            if cost_key and cost_key in records[-1]:
                cost_final = float(records[-1][cost_key])
            else:
                cost_final = float(records[-1]["train_cost"])

            data[k_value][strategy]["auc_performance"].append(auc_val)
            data[k_value][strategy]["cost_final"].append(cost_final)

    # Summary log
    for k_val, strats in data.items():
        for strat, metrics in strats.items():
            n = len(metrics["auc_performance"])
            auc_m = float(np.mean(metrics["auc_performance"]))
            cost_m = float(np.mean(metrics["cost_final"]))
            print(
                f"    k={k_val}  {strat:<40} n_seeds={n}  AUC={auc_m:.4f}  cost_final={cost_m:.2f}"
            )

    return dict(data)


def _stats(values: list[float]) -> tuple[float, float]:
    arr = np.array(values)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, se


def _best_k_stats(raw: dict) -> dict:
    """Pick the k that maximises mean AUC for each strategy."""
    all_strats: set[str] = set()
    for kstats in raw.values():
        all_strats.update(kstats.keys())

    best: dict = {}
    for s in all_strats:
        bk = max(
            (k for k, ks in raw.items() if s in ks),
            key=lambda k, _s=s: float(np.mean(raw[k][_s]["auc_performance"])),
        )
        perf_m, perf_se = _stats(raw[bk][s]["auc_performance"])
        cost_m, cost_se = _stats(raw[bk][s]["cost_final"])
        best[s] = {
            "perf_mean": perf_m,
            "perf_se": perf_se,
            "cost_mean": cost_m,
            "cost_se": cost_se,
        }
    return best


# ── Plot ──────────────────────────────────────────────────────────────────────


def plot_all_benchmarks(
    penalty: float,
    output_dir: str = "plots_benchmarks/partial_annotation",
):
    """2x2 line plot: full vs Wilcoxon, each line connects strategy variants."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), squeeze=False)
    axes_flat = axes.flatten()

    full_line_handle = None
    wilc_line_handle = None
    heur_handle = None

    for idx, benchmark in enumerate(BENCHMARKS):
        ax = axes_flat[idx]

        # Load full annotation cost per query from HuggingFace
        try:
            full_cost_pq = _get_full_annotation_cost_per_query(benchmark)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Cost load error\n{e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(BENCHMARK_LABELS[benchmark], fontsize=14, fontweight="bold")
            continue

        # Load wilcoxon results (annotation_cost tracked per step)
        try:
            wilc_raw = _load_runs(
                _partial_dir(benchmark, penalty), cost_key="annotation_cost"
            )
            wilc_stats = _best_k_stats(wilc_raw)
        except ValueError:
            wilc_stats = {}

        # Load full baselines from main_stationary (train_cost = n_selected)
        try:
            full_raw = _load_runs(_stationary_dir(benchmark, penalty))
            full_stats = _best_k_stats(full_raw)
        except ValueError:
            full_stats = {}

        # Filter to BASE_STRATEGIES and sort
        present = [s for s in BASE_STRATEGIES if s in full_stats or s in wilc_stats]

        # --- Full annotation line ---
        full_xs, full_ys, full_yerrs, full_labels = [], [], [], []
        for s in present:
            if s in full_stats:
                # Convert n_selected to annotation cost
                full_xs.append(full_stats[s]["cost_mean"] * full_cost_pq)
                full_ys.append(full_stats[s]["perf_mean"])
                full_yerrs.append(full_stats[s]["perf_se"])
                full_labels.append(VARIANT_LABELS.get(s, s))

        if full_xs:
            (h,) = ax.plot(
                full_xs,
                full_ys,
                color=COLORS["full"],
                linewidth=2.5,
                marker="o",
                markersize=10,
                alpha=0.85,
                zorder=4,
                label="Full Model Annotation",
            )
            ax.fill_between(
                full_xs,
                np.array(full_ys) - np.array(full_yerrs),
                np.array(full_ys) + np.array(full_yerrs),
                color=COLORS["full"],
                alpha=0.12,
                zorder=2,
            )
            for x, y, lbl in zip(full_xs, full_ys, full_labels):
                ax.annotate(
                    lbl,
                    (x, y),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    color=COLORS["full"],
                    alpha=0.9,
                )
            if full_line_handle is None:
                full_line_handle = h

        # --- Wilcoxon line ---
        wilc_xs, wilc_ys, wilc_yerrs, wilc_xerrs, wilc_labels = [], [], [], [], []
        for s in present:
            if s in wilc_stats:
                wilc_xs.append(wilc_stats[s]["cost_mean"])
                wilc_ys.append(wilc_stats[s]["perf_mean"])
                wilc_yerrs.append(wilc_stats[s]["perf_se"])
                wilc_xerrs.append(wilc_stats[s]["cost_se"])
                wilc_labels.append(VARIANT_LABELS.get(s, s))

        if wilc_xs:
            (h,) = ax.plot(
                wilc_xs,
                wilc_ys,
                color=COLORS["wilcoxon"],
                linewidth=2.5,
                marker="X",
                markersize=11,
                alpha=0.85,
                zorder=4,
                label="Wilcoxon Partial Model Annotation",
            )
            ax.fill_between(
                wilc_xs,
                np.array(wilc_ys) - np.array(wilc_yerrs),
                np.array(wilc_ys) + np.array(wilc_yerrs),
                color=COLORS["wilcoxon"],
                alpha=0.12,
                zorder=2,
            )
            for x, y, lbl in zip(wilc_xs, wilc_ys, wilc_labels):
                ax.annotate(
                    lbl,
                    (x, y),
                    xytext=(8, -12),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    color=COLORS["wilcoxon"],
                    alpha=0.9,
                )
            if wilc_line_handle is None:
                wilc_line_handle = h

        # --- Heuristics as grey lines per family (from main_stationary) ---
        def _is_heuristic(name: str) -> bool:
            return any(name == p or name.startswith(p) for p in HEURISTIC_PREFIXES)

        from plot_commands.plotting_function.plot_auc_vs_cost import (
            get_strategy_family,
            get_variant_sort_key,
        )

        heur_families: dict[str, list[str]] = defaultdict(list)
        for s in full_stats:
            if _is_heuristic(s):
                family, _ = get_strategy_family(s)
                heur_families[family].append(s)

        for family, variants in heur_families.items():
            variants = sorted(variants, key=get_variant_sort_key)
            x_arr = np.array(
                [full_stats[s]["cost_mean"] * full_cost_pq for s in variants]
            )
            y_arr = np.array([full_stats[s]["perf_mean"] for s in variants])
            y_err = np.array([full_stats[s]["perf_se"] for s in variants])

            (h,) = ax.plot(
                x_arr,
                y_arr,
                color=COLORS["heuristics"],
                linewidth=1.5,
                linestyle="-",
                marker="o",
                markersize=6,
                alpha=0.45,
                zorder=3,
            )
            ax.fill_between(
                x_arr,
                y_arr - y_err,
                y_arr + y_err,
                color=COLORS["heuristics"],
                alpha=0.06,
                zorder=2,
            )
            if heur_handle is None:
                heur_handle = h

        ax.set_xlabel(
            "Total Annotation Cost (sum of model costs)", fontsize=13, fontweight="bold"
        )
        ax.set_ylabel("AUC Performance", fontsize=13, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        ax.set_title(BENCHMARK_LABELS[benchmark], fontsize=16, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.3, linestyle="--")

    # Global legend
    handles, labels = [], []
    if full_line_handle is not None:
        handles.append(full_line_handle)
        labels.append("Full Model Annotation")
    if wilc_line_handle is not None:
        handles.append(wilc_line_handle)
        labels.append("Wilcoxon Partial Model Annotation")
    if heur_handle is not None:
        handles.append(heur_handle)
        labels.append("Heuristics")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=13,
        bbox_to_anchor=(0.5, -0.01),
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    out_file = output_path / f"partial_annotation_auc_vs_cost_penalty{penalty}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to {out_file}")
    plt.close(fig)


def print_statistics_table(penalty: float) -> None:
    """Print detailed Mean±SE table for full and Wilcoxon strategies, per benchmark."""
    W = 115
    for benchmark in BENCHMARKS:
        label = BENCHMARK_LABELS[benchmark]
        print(f"\n{'=' * W}")
        print(f"{label} — penalty={penalty}")
        print("=" * W)
        print(
            f"{'Strategy':<40} {'Type':<12} {'AUC Perf (Mean±SE)':<25} {'Cost (Mean±SE)':<25} {'N Seeds':<8}"
        )
        print("-" * W)

        try:
            full_cost_pq = _get_full_annotation_cost_per_query(benchmark)
        except Exception as e:
            print(f"  Cost load error: {e}")
            continue

        try:
            wilc_raw = _load_runs(
                _partial_dir(benchmark, penalty), cost_key="annotation_cost"
            )
            wilc_stats_best = _best_k_stats(wilc_raw)
            # collect n_seeds from raw
            wilc_n: dict[str, int] = {}
            for kstrats in wilc_raw.values():
                for s, m in kstrats.items():
                    wilc_n[s] = len(m["auc_performance"])
        except ValueError:
            wilc_stats_best, wilc_n = {}, {}

        try:
            full_raw = _load_runs(_stationary_dir(benchmark, penalty))
            full_stats_best = _best_k_stats(full_raw)
            full_n: dict[str, int] = {}
            for kstrats in full_raw.values():
                for s, m in kstrats.items():
                    full_n[s] = len(m["auc_performance"])
        except ValueError:
            full_stats_best, full_n = {}, {}

        all_strategies = sorted(
            set(full_stats_best) | set(wilc_stats_best),
            key=lambda s: BASE_STRATEGIES.index(s) if s in BASE_STRATEGIES else 999,
        )

        for s in all_strategies:
            lbl = VARIANT_LABELS.get(s, s)
            if s in full_stats_best:
                st = full_stats_best[s]
                cost = st["cost_mean"] * full_cost_pq
                cost_se = st["cost_se"] * full_cost_pq
                perf_str = f"{st['perf_mean']:.4f} ± {st['perf_se']:.4f}"
                cost_str = f"{cost:.2f} ± {cost_se:.2f}"
                n = full_n.get(s, "?")
                print(f"{lbl:<40} {'full':<12} {perf_str:<25} {cost_str:<25} {n:<8}")
            if s in wilc_stats_best:
                st = wilc_stats_best[s]
                perf_str = f"{st['perf_mean']:.4f} ± {st['perf_se']:.4f}"
                cost_str = f"{st['cost_mean']:.2f} ± {st['cost_se']:.2f}"
                n = wilc_n.get(s, "?")
                print(f"{lbl:<40} {'wilcoxon':<12} {perf_str:<25} {cost_str:<25} {n:<8}")

        print("=" * W)


def print_summary_table(penalty: float):
    """Print delta table: Wilcoxon vs full for AUC Perf and annotation cost."""
    W = 85
    HDR = (
        f"{'Variant':<10} {'AUC Full':>10} {'AUC Wilc':>10} "
        f"{'dAUC':>8} {'Cost Full':>11} {'Cost Wilc':>11} {'Saved':>8}"
    )

    for benchmark in BENCHMARKS:
        try:
            full_cost_pq = _get_full_annotation_cost_per_query(benchmark)
        except Exception:
            continue
        try:
            wilc_stats = _best_k_stats(
                _load_runs(_partial_dir(benchmark, penalty), cost_key="annotation_cost")
            )
        except ValueError:
            continue
        try:
            full_stats = _best_k_stats(_load_runs(_stationary_dir(benchmark, penalty)))
        except ValueError:
            continue

        print(f"\n{'=' * W}")
        print(f"{BENCHMARK_LABELS[benchmark]} — penalty={penalty}")
        print(f"{'-' * W}")
        print(HDR)
        print(f"{'-' * W}")

        for s in BASE_STRATEGIES:
            if s not in full_stats or s not in wilc_stats:
                continue
            fp, wp = full_stats[s]["perf_mean"], wilc_stats[s]["perf_mean"]
            fc = full_stats[s]["cost_mean"] * full_cost_pq
            wc = wilc_stats[s]["cost_mean"]
            saved = (fc - wc) / fc * 100 if fc > 0 else 0.0
            lbl = VARIANT_LABELS.get(s, s)
            print(
                f"{lbl:<10} {fp:>10.4f} {wp:>10.4f} "
                f"{wp - fp:>+8.4f} {fc:>11.2f} {wc:>11.2f} {saved:>7.1f}%"
            )
        print(f"{'=' * W}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot AUC Perf vs annotation cost: full (main_stationary) vs Wilcoxon (partial_annotation)."
    )
    parser.add_argument("--penalty", type=float, required=True)
    parser.add_argument(
        "--output-dir", type=str, default="plots_benchmarks/partial_annotation"
    )

    args = parser.parse_args()

    plot_all_benchmarks(penalty=args.penalty, output_dir=args.output_dir)
    print_statistics_table(penalty=args.penalty)
    print_summary_table(penalty=args.penalty)
    print("\nDone!")
