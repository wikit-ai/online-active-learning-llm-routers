"""
Note on reproducibility:
Each ablation run executes several strategy variants in a single streaming loop.
Because PyTorch uses a global RNG (for Dropout and DataLoader shuffling), the
strategies interact with that global state in a different order than when a
single strategy runs alone (as in the main experiment).  This produces small
numerical differences in results and annotated corpus sizes between an ablation baseline and its
equivalent main-experiment result, even for the same seed.  These differences
are within the standard error across seeds and do not affect any conclusions.
"""

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Literal

from datasets_management.non_stationary_dataset import DatasetManagement
from datasets_management.embedding import get_embedder
from experiment.ablations import EXPERIMENTS  # type: ignore
from experiment.results_tracker import ResultsTracker

from logging_config import logger


def _build_ds_loader(
    seed: int,
    benchmark: str,
    embedding_model: Literal[
        "bge-m3",
        "snowflake-arctic-embed-l-v2.0",
        "snowflake-arctic-embed-m-v2.0",
        "potion-multilingual-128M",
        "text-embedding-3-large",
    ],
) -> DatasetManagement:
    """Instantiate a DatasetManagement loader, using the default embedder for snowflake-arctic."""
    embedder = (
        None
        if embedding_model == "snowflake-arctic-embed-m-v2.0"
        else get_embedder(embedding_model=embedding_model)
    )
    return DatasetManagement(
        seed=seed,
        benchmark=benchmark,  # type: ignore
        embedder=embedder,
        train_test_sizes=(4000, 5000),
    )


def _extract_summary(
    tracker: ResultsTracker,
    ablation_name: str,
    seed: int,
    benchmark: str,
    embedding_model: str,
) -> dict[str, Any]:
    """Extract a compact summary (AUC_P, AUC_DTO, final support set size) from the tracker."""
    auc_metrics = tracker.calculate_auc_metrics()
    experiment_summary = tracker.get_summary()

    results_per_k: dict[str, Any] = {}
    for k_value in auc_metrics:
        results_per_k[str(k_value)] = {}
        for strategy_name, auc_vals in auc_metrics[k_value].items():
            final_support_set_size = (
                experiment_summary.get(k_value, {})
                .get(strategy_name, {})
                .get("samples_selected", None)
            )
            results_per_k[str(k_value)][strategy_name] = {
                "auc_performance": auc_vals["auc_performance"],
                "auc_dto": auc_vals["auc_dto"],
                "final_support_set_size": final_support_set_size,
            }

    return {
        "experiment_type": ablation_name,
        "benchmark": benchmark,
        "embedding_model": embedding_model,
        "seed": seed,
        "results_per_k": results_per_k,
    }


def run_single(
    ablation_name: str,
    seed: int,
    benchmark: str,
    embedding_model: str,
    budget: int,
    k_values: list[int],
    base_output_dir: str,
    cost_penalty: float,
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Run one ablation experiment for a single seed and save its summary JSON."""
    output_dir = os.path.join(base_output_dir, ablation_name)

    ds_loader = _build_ds_loader(seed, benchmark, embedding_model)  # type: ignore

    run_fn = EXPERIMENTS[ablation_name]  # type: ignore
    tracker = run_fn(  # type: ignore
        seed=seed,
        benchmark=benchmark,
        embedding_model=embedding_model,
        budget=budget,
        ds_loader=ds_loader,
        k_values=k_values,
        output_dir=output_dir,
        cost_penalty=cost_penalty,
        **extra_kwargs.get(ablation_name, {}),
    )

    summary = _extract_summary(tracker, ablation_name, seed, benchmark, embedding_model)  # type: ignore
    summary_dir = os.path.join(output_dir, benchmark, embedding_model)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"summary_seed{seed}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[{ablation_name}] seed={seed} summary saved to: {summary_path}")
    return summary


def _load_existing_summaries(
    base_output_dir: str, ablations: list[str]
) -> list[dict[str, Any]]:
    """Load all summary_seed*.json files already saved under base_output_dir."""
    existing: list[dict[str, Any]] = []
    for ablation in ablations:
        ablation_dir = os.path.join(base_output_dir, ablation)
        if not os.path.isdir(ablation_dir):
            continue
        for root, _, files in os.walk(ablation_dir):
            for fname in files:
                if fname.startswith("summary_seed") and fname.endswith(".json"):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, encoding="utf-8") as f:
                            existing.append(json.load(f))
                    except Exception as exc:
                        logger.warning(f"[warn] Could not load {fpath}: {exc}")
    return existing


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean and SE across seeds for each (experiment, k, strategy, metric)."""
    buckets: dict[str, Any] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # type: ignore
    )
    for r in results:
        exp = r["experiment_type"]
        for k, strategies in r["results_per_k"].items():
            for strategy, metrics in strategies.items():
                for metric, value in metrics.items():
                    if value is not None:
                        buckets[exp][k][strategy][metric].append(value)

    aggregated: dict[str, Any] = {}
    for exp, ks in buckets.items():
        aggregated[exp] = {}
        for k, strategies in ks.items():
            aggregated[exp][k] = {}
            for strategy, metrics in strategies.items():
                aggregated[exp][k][strategy] = {}
                for metric, values in metrics.items():
                    n = len(values)
                    mean = sum(values) / n
                    sd = (
                        math.sqrt(sum((v - mean) ** 2 for v in values) / max(n - 1, 1))
                        if n > 1
                        else 0.0
                    )
                    se = sd / math.sqrt(n)
                    aggregated[exp][k][strategy][metric] = {
                        "mean": mean,
                        "se": se,
                        "n": n,
                    }
    return aggregated


def main() -> None:
    """Parse CLI arguments, run selected ablation experiments, and save aggregated results."""
    parser = argparse.ArgumentParser(
        description="Run secondary ablation experiments for InferredSparsePerformanceStrategy"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=[*EXPERIMENTS.keys(), "all"],
        required=True,
        help="Which ablations to run. Use 'all' for every experiment.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Single seed (overrides --seeds)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="List of random seeds (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="RouterBench",
        choices=["EmbedLLM", "RouterBench", "Sprout", "FusionBench"],
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="snowflake-arctic-embed-m-v2.0",
    )
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[40],
        help="k values for KNN (default: 40)",
    )
    parser.add_argument("--cost_penalty", type=float, default=0.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/secondary_experiments",
        help="Base output directory",
    )

    parser.add_argument(
        "--cold_start_values",
        type=int,
        nargs="+",
        default=None,
        help="cold_start ablation: n_0 values to sweep (default: 25 50 75 115 150 200)",
    )
    parser.add_argument(
        "--retrain_interval_values",
        type=int,
        nargs="+",
        default=None,
        help="retrain_interval ablation: delta values to sweep (default: 1 5 10 20 50)",
    )
    parser.add_argument(
        "--guardrail_thresholds",
        type=float,
        nargs="+",
        default=None,
        help="guardrail ablation: thresholds to sweep (omit a value to include disabled variant)",
    )
    parser.add_argument(
        "--guardrail_include_disabled",
        action="store_true",
        default=True,
        help="guardrail ablation: include the disabled-guardrail variant (default: True)",
    )

    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else args.seeds
    ablations = (
        list(EXPERIMENTS.keys()) if "all" in args.experiments else args.experiments  # type: ignore
    )

    extra_kwargs: dict[str, Any] = {}
    if args.cold_start_values:
        extra_kwargs["cold_start"] = {"cold_start_values": args.cold_start_values}
    if args.retrain_interval_values:
        extra_kwargs["retrain_interval"] = {
            "retrain_interval_values": args.retrain_interval_values
        }
    if args.guardrail_thresholds is not None:
        thresholds: list[Any] = list(args.guardrail_thresholds)
        if args.guardrail_include_disabled:
            thresholds = [None] + thresholds
        extra_kwargs["guardrail"] = {"guardrail_thresholds": thresholds}

    all_results: list[dict[str, Any]] = []
    for ablation_name in ablations:
        for seed in seeds:
            logger.info(f"\n{'=' * 60}")
            logger.info(
                f"Running: {ablation_name} | seed={seed} | benchmark={args.benchmark}"
            )
            logger.info(f"{'=' * 60}")
            result = run_single(
                ablation_name=ablation_name,
                seed=seed,
                benchmark=args.benchmark,
                embedding_model=args.embedding_model,
                budget=args.budget,
                k_values=args.k_values,
                base_output_dir=args.output_dir,
                cost_penalty=args.cost_penalty,
                extra_kwargs=extra_kwargs,
            )
            all_results.append(result)

    existing = _load_existing_summaries(args.output_dir, ablations)
    seen = {
        (r["experiment_type"], r["benchmark"], r["embedding_model"], r["seed"])
        for r in all_results
    }
    for r in existing:
        key = (r["experiment_type"], r["benchmark"], r["embedding_model"], r["seed"])
        if key not in seen:
            all_results.append(r)
            seen.add(key)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg_path = os.path.join(
        args.output_dir,
        f"aggregated_{args.benchmark}_{timestamp}.json",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    aggregated = _aggregate_results(all_results)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump({"aggregated": aggregated, "per_seed": all_results}, f, indent=2)

    logger.info(f"\nAll results aggregated at: {agg_path}")


if __name__ == "__main__":
    main()
