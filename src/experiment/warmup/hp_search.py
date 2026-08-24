import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from itertools import product
from typing import Any

import numpy as np
from tqdm import tqdm

from datasets_management.non_stationary_dataset import DatasetManagement
from datasets_management.embedding import get_embedder
from experiment.evaluation import MainEvaluator
from experiment.results_tracker import ResultsTracker
from experiment.routers.knn_router import KNNRouter
from experiment.utils.metrics import (
    calculate_weighted_dto,
    extract_oracle_performance_numpy,
)
from sampling_strategies import InferredSparsePerformanceStrategy

DEFAULT_LEARNING_RATES: list[float] = [1e-4, 1e-3]
DEFAULT_WEIGHT_DECAYS: list[float] = [1e-3, 1e-2]
DEFAULT_HIDDEN_DIMS: list[int] = [16, 32, 64, 128]
DEFAULT_COLD_START_N: int = 115
DEFAULT_VAL_FRACTION: float = 0.2


def run_hp_search(
    seed: int,
    benchmark: str,
    embedding_model: str,
    budget: int,
    ds_loader: DatasetManagement,
    k_values: list[int],
    output_dir: str = "results/warmup",
    cost_penalty: float = 0.0,
    learning_rates: list[float] | None = None,
    weight_decays: list[float] | None = None,
    hidden_dims: list[int] | None = None,
    cold_start_n: int = DEFAULT_COLD_START_N,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> dict[str, Any]:
    """Run the HP grid search and evaluate on a held-out validation split.

    Returns a compact summary dict saved as results/warmup/<benchmark>/<model>/summary_seed<N>.json.
    """
    learning_rates = learning_rates or DEFAULT_LEARNING_RATES
    weight_decays = weight_decays or DEFAULT_WEIGHT_DECAYS
    hidden_dims = hidden_dims or DEFAULT_HIDDEN_DIMS

    train_ds = ds_loader.get_training_dataset()
    n = len(train_ds)
    n_val = int(n * val_fraction)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n).tolist()
    val_ds = train_ds.select(perm[:n_val])  # type: ignore
    stream_ds = train_ds.select(perm[n_val:])  # type: ignore

    o_cost_arr, o_perf_arr = extract_oracle_performance_numpy(
        np.array(val_ds["models_performance"]),  # type: ignore
        np.array(val_ds["cost"]),  # type: ignore
    )
    oracle_cost = float(np.mean(o_cost_arr))
    oracle_perf = float(np.mean(o_perf_arr))

    strategies: dict[str, InferredSparsePerformanceStrategy] = {
        f"lr{lr:.0e}_wd{wd:.0e}_hd{hd}": InferredSparsePerformanceStrategy(
            cold_start_n=cold_start_n,
            budget=budget,
            seed=seed,
            training_settings={"learning_rate": lr, "weight_decay": wd},
            model_settings={"hidden_dim": hd},
        )
        for lr, wd, hd in product(learning_rates, weight_decays, hidden_dims)
    }

    evaluators: dict[str, MainEvaluator] = {
        name: MainEvaluator(
            router=KNNRouter(),
            test_ds=val_ds,  # evaluator use val_ds as the routing evaluation set
            cost_penalty=cost_penalty,
        )
        for name in strategies
    }

    experiment_name = f"warmup/{embedding_model}/{benchmark}/seed{seed}"
    tracker = ResultsTracker(experiment_name, output_dir=output_dir)
    tracker.set_metadata(
        seed=seed,
        benchmark=benchmark,
        embedding_model=embedding_model,
        stream_size=len(stream_ds),
        val_size=n_val,
        cold_start_n=cold_start_n,
        val_fraction=val_fraction,
    )

    samples_selected = {name: 0 for name in strategies}
    previous_metrics: dict[str, dict[int, dict[str, float]]] = {
        name: {
            k: {"avg_perf": 0.0, "avg_cost": 0.0, "avg_mse": 0.0, "dto": 0.0}
            for k in k_values
        }
        for name in strategies
    }

    for step, item in enumerate(  # type: ignore
        tqdm(stream_ds, desc=f"warmup {benchmark}/seed{seed}", total=len(stream_ds))  # type: ignore
    ):
        for strategy_name, strategy in strategies.items():
            evaluator = evaluators[strategy_name]
            current_corpus = evaluator.train_df

            should_select = strategy.should_select(
                item=item, current_train_ds=current_corpus  # type: ignore
            )

            if should_select and (
                current_corpus is None or len(current_corpus) < budget
            ):
                results_k = evaluator.evaluate_on_test(train_item=item, k=k_values)  # type: ignore
                for k_value in results_k:
                    dto = calculate_weighted_dto(
                        oracle_cost=oracle_cost,
                        oracle_perf=oracle_perf,
                        test_cost=results_k[k_value]["avg_cost"],
                        test_perf=results_k[k_value]["avg_perf"],
                    )
                    results_k[k_value]["dto"] = float(dto)
                samples_selected[strategy_name] += 1
                previous_metrics[strategy_name] = results_k
            else:
                evaluator.redundant_step_no_annotation()
                results_k = previous_metrics[strategy_name]

            for k_value in results_k:
                tracker.log_step(
                    step=step,
                    k_value=k_value,
                    strategy=strategy_name,
                    test_performance=results_k[k_value]["avg_perf"],
                    test_cost=results_k[k_value]["avg_cost"],
                    test_mse=results_k[k_value]["avg_mse"],
                    train_cost=evaluator.cost_train,
                    samples_selected=samples_selected[strategy_name],
                    dto=results_k[k_value]["dto"],
                )

    auc_metrics = tracker.calculate_auc_metrics()
    summary = tracker.get_summary()

    results_per_k: dict[str, Any] = {}
    for k_value, strategies_auc in auc_metrics.items():
        results_per_k[str(k_value)] = {}
        for strategy_name, auc_vals in strategies_auc.items():
            final_support = (
                summary.get(k_value, {}).get(strategy_name, {}).get("samples_selected")
            )
            results_per_k[str(k_value)][strategy_name] = {
                "auc_performance": auc_vals["auc_performance"],
                "auc_dto": auc_vals["auc_dto"],
                "final_support_set_size": final_support,
            }

    compact = {  # type: ignore
        "benchmark": benchmark,
        "embedding_model": embedding_model,
        "seed": seed,
        "val_fraction": val_fraction,
        "oracle_cost": oracle_cost,
        "oracle_perf": oracle_perf,
        "results_per_k": results_per_k,
    }

    summary_dir = os.path.join(output_dir, benchmark, embedding_model)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"summary_seed{seed}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)

    print(f"[warmup] {benchmark}/seed={seed} saved to: {summary_path}")
    return compact  # type: ignore


def _load_existing_summaries(
    output_dir: str, benchmarks: list[str]
) -> list[dict[str, Any]]:
    """Load all previously saved per-seed summary JSON files for the given benchmarks."""
    existing: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        benchmark_dir = os.path.join(output_dir, benchmark)
        if not os.path.isdir(benchmark_dir):
            continue
        for root, _, files in os.walk(benchmark_dir):
            for fname in files:
                if fname.startswith("summary_seed") and fname.endswith(".json"):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, encoding="utf-8") as f:
                            existing.append(json.load(f))
                    except Exception as exc:
                        print(f"[warn] Could not load {fpath}: {exc}")
    return existing


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean ± SE across seeds for each (benchmark, k, strategy, metric)."""
    buckets: dict = defaultdict(  # type: ignore
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # type: ignore
    )
    for r in results:
        benchmark = r["benchmark"]
        for k, strategies in r["results_per_k"].items():
            for strategy, metrics in strategies.items():
                for metric, value in metrics.items():
                    if value is not None:
                        buckets[benchmark][k][strategy][metric].append(value)  # type: ignore

    aggregated: dict[str, Any] = {}
    for benchmark, ks in buckets.items():  # type: ignore
        aggregated[benchmark] = {}
        for k, strategies in ks.items():  # type: ignore
            aggregated[benchmark][k] = {}
            for strategy, metrics in strategies.items():  # type: ignore
                aggregated[benchmark][k][strategy] = {}
                for metric, values in metrics.items():  # type: ignore
                    n = len(values)  # type: ignore
                    mean = sum(values) / n  # type: ignore
                    sd = (
                        math.sqrt(sum((v - mean) ** 2 for v in values) / max(n - 1, 1))  # type: ignore
                        if n > 1
                        else 0.0
                    )
                    aggregated[benchmark][k][strategy][metric] = {
                        "mean": mean,
                        "se": sd / math.sqrt(n),
                        "n": n,
                    }
    return aggregated


def _sort_strategies(aggregated: dict[str, Any]) -> dict[str, Any]:
    """Sort strategies by combined rank: rank(auc_performance desc) + rank(support_set_size asc)."""
    sorted_agg: dict[str, Any] = {}
    for benchmark, ks in aggregated.items():
        sorted_agg[benchmark] = {}
        for k, strategies in ks.items():
            names = list(strategies.keys())
            perf_order = sorted(
                names,
                key=lambda s: -strategies[s]
                .get("auc_performance", {})
                .get("mean", 0.0),
            )
            size_order = sorted(
                names,
                key=lambda s: strategies[s]
                .get("final_support_set_size", {})
                .get("mean", float("inf")),
            )
            perf_rank = {s: i for i, s in enumerate(perf_order)}
            size_rank = {s: i for i, s in enumerate(size_order)}
            sorted_agg[benchmark][k] = {
                s: strategies[s]
                for s in sorted(names, key=lambda s: perf_rank[s] + size_rank[s])
            }
    return sorted_agg


def _best_per_benchmark(
    aggregated: dict[str, Any], metric: str = "auc_performance"
) -> dict[str, Any]:
    """Pick the strategy with the highest mean <metric> for each (benchmark, k)."""
    best: dict[str, Any] = {}
    for benchmark, ks in aggregated.items():
        best[benchmark] = {}
        for k, strategies in ks.items():
            best_strategy = max(
                strategies,
                key=lambda s: strategies[s].get(metric, {}).get("mean", -1),
            )
            best[benchmark][k] = {
                "strategy": best_strategy,
                metric: strategies[best_strategy][metric],
            }
    return best


def main() -> None:
    """Parse CLI arguments, run HP search across benchmarks/seeds, and save aggregated results."""
    parser = argparse.ArgumentParser(
        description="Warmup HP search evaluated on a validation split."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["EmbedLLM", "RouterBench", "Sprout", "FusionBench"],
        choices=["EmbedLLM", "RouterBench", "Sprout", "FusionBench"],
        help="Benchmarks to run (default: all four)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Single seed (overrides --seeds)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--embedding_model", type=str, default="snowflake-arctic-embed-m-v2.0"
    )
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--k_values", type=int, nargs="+", default=[40])
    parser.add_argument("--cost_penalty", type=float, default=0.0)
    parser.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--cold_start_n", type=int, default=DEFAULT_COLD_START_N)
    parser.add_argument("--output_dir", type=str, default="results/warmup")
    parser.add_argument("--learning_rates", type=float, nargs="+", default=None)
    parser.add_argument("--weight_decays", type=float, nargs="+", default=None)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=None)

    args = parser.parse_args()
    seeds = [args.seed] if args.seed is not None else args.seeds

    all_results: list[dict[str, Any]] = []

    for benchmark in args.benchmarks:
        for seed in seeds:
            print(f"\n{'=' * 60}")
            print(f"Warmup HP search | benchmark={benchmark} | seed={seed}")
            print(f"{'=' * 60}")

            embedder = (
                None
                if args.embedding_model == "snowflake-arctic-embed-m-v2.0"
                else get_embedder(embedding_model=args.embedding_model)
            )
            ds_loader = DatasetManagement(
                seed=seed,
                benchmark=benchmark,
                embedder=embedder,
                train_test_sizes=(4000, 5000),
            )

            result = run_hp_search(
                seed=seed,
                benchmark=benchmark,
                embedding_model=args.embedding_model,
                budget=args.budget,
                ds_loader=ds_loader,
                k_values=args.k_values,
                output_dir=args.output_dir,
                cost_penalty=args.cost_penalty,
                learning_rates=args.learning_rates,
                weight_decays=args.weight_decays,
                hidden_dims=args.hidden_dims,
                cold_start_n=args.cold_start_n,
                val_fraction=args.val_fraction,
            )
            all_results.append(result)

    all_benchmarks = ["EmbedLLM", "RouterBench", "Sprout", "FusionBench"]
    existing = _load_existing_summaries(args.output_dir, all_benchmarks)
    seen = {(r["benchmark"], r["embedding_model"], r["seed"]) for r in all_results}
    for r in existing:
        key = (r["benchmark"], r["embedding_model"], r["seed"])
        if key not in seen:
            all_results.append(r)
            seen.add(key)

    aggregated = _aggregate_results(all_results)
    aggregated = _sort_strategies(aggregated)
    best = _best_per_benchmark(aggregated, metric="auc_performance")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg_path = os.path.join(args.output_dir, f"aggregated_{timestamp}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_per_benchmark": best,
                "aggregated": aggregated,
                "per_seed": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nAggregated results saved to: {agg_path}")
    print("\nBest strategy per benchmark (auc_performance):")
    for benchmark, ks in best.items():
        for k, info in ks.items():
            print(
                f"  {benchmark} k={k}: {info['strategy']}  (mean={info['auc_performance']['mean']:.4f})"
            )


if __name__ == "__main__":
    main()
