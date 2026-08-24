import argparse

from plot_commands.plotting_function.plot_regret import (
    compute_regret_stats,
    load_regret_data,
    plot_regret_over_time,
    _CHECKPOINT_PCTS,
)
from logging_config import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Line plot of regret (continual vs hindsight) over time"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Root results directory (default: results)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="snowflake-arctic-embed-m-v2.0",
        help="Embedding model subfolder name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots_benchmarks/model_switch_regret",
        help="Output directory for the plot",
    )
    args = parser.parse_args()

    logger.info(f"Loading regret data from: {args.results_dir}")
    logger.info(f"Embedding model: {args.embedding_model}")

    data_per_benchmark = load_regret_data(args.results_dir, args.embedding_model)

    if not data_per_benchmark:
        logger.info(
            "No regret_results files found. Check --results-dir and --embedding-model."
        )
        raise SystemExit(1)

    stats_per_benchmark: dict[str, dict[float, dict[str, dict[str, float]]]] = {}
    n_runs_per_benchmark: dict[str, int] = {}

    for benchmark, runs in sorted(data_per_benchmark.items()):
        logger.info(f"\nBenchmark: {benchmark} — {len(runs)} run(s)")
        stats_per_benchmark[benchmark] = compute_regret_stats(runs)
        n_runs_per_benchmark[benchmark] = len(runs)

        for pct in _CHECKPOINT_PCTS:
            pct_label = f"{int(pct * 100)}%"
            s = stats_per_benchmark[benchmark].get(pct, {})
            dp = s.get("delta_perf", {})
            dd = s.get("delta_dto", {})
            logger.info(
                f"  {pct_label:>4s}  "
                f"ΔPerf={dp.get('mean', 0):.4f}±{dp.get('stderr', 0):.4f}  "
                f"ΔDTO={dd.get('mean', 0):.4f}±{dd.get('stderr', 0):.4f}"
            )

    plot_regret_over_time(stats_per_benchmark, n_runs_per_benchmark, args.output_dir)
    logger.info("\nDone!")
