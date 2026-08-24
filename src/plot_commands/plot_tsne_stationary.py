import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from datasets.arrow_dataset import Dataset  # type: ignore
from datasets_management.non_stationary_dataset import DatasetManagement
from datasets_management.embedding import get_embedder
from logging_config import logger
from plot_commands.plotting_function.plot_tsne_selections import (
    plot_tsne_selections_per_seed,
)


def extract_datetime(filepath: str) -> datetime:
    """Extract datetime from run file name."""
    name = Path(filepath).stem
    try:
        return datetime.strptime(name.replace("run_", ""), "%Y%m%d_%H%M%S")
    except Exception:
        return datetime.min


def load_selections_from_seed(results_dir: str, seed: int) -> dict[str, list[int]]:
    """
    Load selection data from all run_*.json files for a specific seed.
    Merges dict_selected from multiple files, with later files overwriting earlier ones.

    Args:
        results_dir: Base results directory
        seed: Seed number

    Returns:
        dict: {strategy: [selected_indices]}
    """
    base_path = Path(results_dir) / str(seed)
    pattern = os.path.join(base_path, "run_*.json")
    files = glob.glob(pattern)

    files_sorted = sorted(files, key=extract_datetime)

    if not files_sorted:
        logger.warning(f"No run files found in {base_path}")
        return {}

    merged: dict[str, list[int]] = {}
    for filepath in files_sorted:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        dict_selected = data.get("dict_selected", {})
        merged.update(dict_selected)  # Later files overwrite earlier ones

    print(
        f"Seed {seed}: Loaded {len(files_sorted)} files, {len(merged)} strategies total"
    )
    return merged


def load_embeddings_from_dataset(
    benchmark: Literal["EmbedLLM", "RouterBench", "Sprout"],
    seed: int,
    embedding_model: Literal[
        "bge-m3",
        "snowflake-arctic-embed-l-v2.0",
        "snowflake-arctic-embed-m-v2.0",
        "potion-multilingual-128M",
        "text-embedding-3-large",
    ],
    train_size: int = 1000,
    test_size: int = 5000,
) -> np.ndarray:
    """
    Load embeddings from the dataset using DatasetManagement.

    Args:
        benchmark: Benchmark name
        seed: Seed number
        embedding_model: Name of the embedding model
        train_size: Training dataset size
        test_size: Test dataset size

    Returns:
        np.ndarray: Array of shape (N, D) containing embeddings
    """
    ds_loader = DatasetManagement(
        seed=seed,
        benchmark=benchmark,
        embedder=get_embedder(embedding_model=embedding_model),
        train_test_sizes=(train_size, test_size),
    )
    training_ds: Dataset = ds_loader.get_training_dataset()
    embeddings = np.array(training_ds["embeddings"]).squeeze(1)  # type: ignore
    return embeddings  # type: ignore


def filter_selections_by_prefix(
    selections_dict: dict[str, list[int]], prefixes: list[str] | None
) -> dict[str, list[int]]:
    """
    Filter selections by strategy prefixes.

    Args:
        selections_dict: Dictionary of strategy selections
        prefixes: List of prefixes to include

    Returns:
        dict: Filtered selections
    """
    if prefixes is None:
        return selections_dict

    filtered: dict[str, list[int]] = {}
    for strategy, indices in selections_dict.items():
        if any(strategy.startswith(prefix) for prefix in prefixes):
            filtered[strategy] = indices

    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate t-SNE visualizations of selected points per strategy"
    )
    parser.add_argument(
        "--penalty",
        type=str,
        required=True,
        help="Which cost penalty to use (e.g., 'main_0.23')",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["Sprout", "RouterBench", "EmbedLLM"],
        required=True,
        help="Which benchmark to use",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="List of seed numbers to process (e.g., 1 2 3)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="snowflake-arctic-embed-m-v2.0",
        help="Embedding model name. Default: snowflake-arctic-embed-m-v2.0",
    )
    parser.add_argument(
        "--strategy-prefixes",
        type=str,
        nargs="+",
        default=None,
        help="List of strategy prefixes to include (e.g., 'rand_' 'act_unc' 'oracle')",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="Perplexity parameter for t-SNE. Default: 30",
    )

    args = parser.parse_args()

    # Build results directory path
    base_results_dir = (
        f"results/{args.penalty}/stationary/{args.embedding_model}/{args.benchmark}"
    )

    print(f"Using benchmark: {args.benchmark}")
    print(f"Base results directory: {base_results_dir}")
    print(f"Seeds to process: {args.seeds}")
    print()

    # Process each seed
    for seed in args.seeds:
        print(f"\n{'=' * 80}")
        print(f"Processing Seed {seed}")
        print("=" * 80)

        # Load selections for this seed (merges all runs, last run overwrites)
        selections_raw = load_selections_from_seed(base_results_dir, seed)

        if not selections_raw:
            print(f"No selections found for seed {seed}, skipping...")
            continue

        # Filter strategies if requested
        if args.strategy_prefixes:
            print(f"Filtering strategies by prefixes: {args.strategy_prefixes}")
            selections_raw = filter_selections_by_prefix(
                selections_raw, args.strategy_prefixes
            )
            print(f"Strategies after filtering: {list(selections_raw.keys())}")

        print(f"Total strategies: {len(selections_raw)}")

        # Load embeddings from dataset
        print(f"\nLoading embeddings for {args.benchmark} (seed={seed})...")
        embeddings_data = load_embeddings_from_dataset(
            benchmark=args.benchmark,
            seed=seed,
            embedding_model=args.embedding_model,
        )

        # Generate one big t-SNE plot with all strategies
        print(f"\nGenerating t-SNE plot for seed {seed}...")
        output_dir = Path("plots_benchmarks/tsne_selections") / args.benchmark
        output_dir.mkdir(exist_ok=True, parents=True)

        output_file = output_dir / f"tsne_all_strategies_seed{seed}.png"
        plot_tsne_selections_per_seed(
            embeddings=embeddings_data,
            selections=selections_raw,
            seed=seed,
            output_path=str(output_file),
            perplexity=args.perplexity,
        )
        print(f"Generated t-SNE plot: {output_file}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
