"""Shared constants for plotting commands.

Centralises strategy names, ordering, exclusions, and benchmark definitions
that were previously duplicated across multiple plot_benchmarks_* files.
"""

STRATEGY_NAMES: dict[str, str] = {
    "inferred_sparse_nn_opt": "SparseNN",
    "oracle_sparse": "OracleSp",
    "oracle_ds": "OracleCov",
    "oracle_ds_200": "OracleCov",
    "passive": "Passive",
    "rand": "Random",
    "rand_0_05": "Rand0.05",
    "rand_0_10": "Rand0.10",
    "rand_0_15": "Rand0.15",
    "rand_0_25": "Rand0.25",
    "unc_mc_t3": "a_rank",
    "act_repr_min": "a_min",
    "unc_t3": "a_var",
    "repr_vmf_kap1": "a_vMF",
}

STRATEGY_NAMES_LATEX: dict[str, str] = {
    "inferred_sparse_nn_opt": "SparseNN",
    "oracle_sparse": "OracleSp",
    "oracle_ds_200": "OracleCov",
    "passive": "Passive",
    "rand_0_05": "Rand0.05",
    "rand_0_10": "Rand0.10",
    "rand_0_15": "Rand0.15",
    "rand_0_25": "Rand0.25",
    "unc_mc_t3": "a\\_rank",
    "act_repr_min": "a\\_min",
    "unc_t3": "a\\_var",
    "repr_vmf_kap1": "a\\_vMF",
    "oracle_ds": "OracleCov",
    "rand": "Rand",
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

FAMILY_LINESTYLES: dict[str, str] = {
    "oracle_sparse": "-",
    "oracle_ds": ":",
}

FAMILY_COLORS: dict[str, str] = {
    "rand": "#00159c",
    "unc_mc_t3": "#8B0000",
}

BENCHMARKS = ["sprout", "routerbench", "embedllm", "fusionbench"]

BENCHMARK_LABELS: dict[str, str] = {
    "sprout": "Sprout",
    "routerbench": "RouterBench",
    "embedllm": "EmbedLLM",
    "fusionbench": "FusionBench",
}

EMBEDDING_MODEL_DEFAULT = "snowflake-arctic-embed-m-v2.0"


def get_results_dir(
    benchmark: str, penalty: float, experiment_type: str = "stationary"
) -> str:
    """Build the results directory path for a given benchmark and experiment type."""
    benchmark_name = BENCHMARK_LABELS[benchmark]
    return f"results/{penalty}/{experiment_type}/{EMBEDDING_MODEL_DEFAULT}/{benchmark_name}"
