import argparse

from collections import Counter
from typing import Literal

from datasets_management.non_stationary_dataset import (
    DatasetManagement,
    DomainShiftDatasetManagement,
)
from datasets_management.embedding import get_embedder
from experiment.main_stationary import run_main_experiment
from experiment.main_no_stationary import run_domain_shift_experiment
from experiment.main_model_switch_regret import run_model_switch_regret_experiment
from experiment.main_partial_annotation import run_partial_annotation_experiment
from sampling_strategies import (
    HeuristicsStrategy,
    RandomStrategy,
    PassiveStrategy,
    OracleCoverageStrategy,
    OracleSparsePerformanceStrategy,
    InferredSparsePerformanceStrategy,
)


def run(
    type_experiment: Literal[
        "main_stationary",
        "main_domain_non_stationary",
        "model_switch_regret",
        "partial_annotation",
    ],
    embedding_model: Literal[
        "bge-m3",
        "snowflake-arctic-embed-l-v2.0",
        "snowflake-arctic-embed-m-v2.0",
        "potion-multilingual-128M",
        "text-embedding-3-large",
    ],
    benchmark: Literal["EmbedLLM", "RouterBench", "Sprout", "FusionBench", "R2Bench"],
    seed: int,
    budget: int,
    k_values: list[int],
    cost_penalty: float = 0.0,
):
    # snowflake-arctic-embed-m-v2.0 embeddings are pre-computed in the dataset
    embedder = (
        None
        if embedding_model == "snowflake-arctic-embed-m-v2.0"
        else get_embedder(embedding_model=embedding_model)
    )

    if type_experiment == "main_domain_non_stationary":
        ds_loader = DomainShiftDatasetManagement(
            seed=seed,
            benchmark=benchmark,
            embedder=embedder,
            cost_penalty=cost_penalty,
        )
    elif type_experiment == "main_stationary":
        ds_loader = DatasetManagement(
            seed=seed,
            benchmark=benchmark,
            embedder=embedder,
            train_test_sizes=(4000, 5000),
        )
    elif type_experiment == "model_switch_regret":
        ds_loader = DatasetManagement(
            seed=seed,
            benchmark=benchmark,
            embedder=embedder,
            train_test_sizes=(3000, 5000),
        )
    elif type_experiment == "partial_annotation":
        ds_loader = DatasetManagement(
            seed=seed,
            benchmark=benchmark,
            embedder=embedder,
            train_test_sizes=(4000, 5000),
        )
    else:
        raise NotImplementedError()

    training_ds = ds_loader.get_training_dataset()
    ds_names: list[str] = training_ds["dataset"]  # type: ignore
    counter = Counter(ds_names)

    strategies: dict[
        str,
        InferredSparsePerformanceStrategy
        | HeuristicsStrategy
        | PassiveStrategy
        | RandomStrategy
        | OracleCoverageStrategy
        | OracleSparsePerformanceStrategy,
    ] = {
        "passive": PassiveStrategy(n_max=budget),
        "inferred_sparse_nn_opt": InferredSparsePerformanceStrategy(
            cold_start_n=115,
            budget=budget,
            seed=seed,
            model_settings={"hidden_dim": 16},
            training_settings={
                "learning_rate": 1e-3,
                "weight_decay": 1e-3,
            },
        ),
        "inferred_sparse_nn_opt_p25": InferredSparsePerformanceStrategy(
            cold_start_n=115,
            budget=budget,
            seed=seed,
            percentile=25,
            model_settings={"hidden_dim": 16},
            training_settings={
                "learning_rate": 1e-3,
                "weight_decay": 1e-3,
            },
        ),
        "inferred_sparse_nn_opt_p50": InferredSparsePerformanceStrategy(
            cold_start_n=115,
            budget=budget,
            seed=seed,
            percentile=50,
            model_settings={"hidden_dim": 16},
            training_settings={
                "learning_rate": 1e-3,
                "weight_decay": 1e-3,
            },
        ),
        "inferred_sparse_nn_opt_p90": InferredSparsePerformanceStrategy(
            cold_start_n=115,
            budget=budget,
            seed=seed,
            percentile=90,
            model_settings={"hidden_dim": 16},
            training_settings={
                "learning_rate": 1e-3,
                "weight_decay": 1e-3,
            },
        ),
        "oracle_sparse": OracleSparsePerformanceStrategy(
            cold_start_n=98,  # 115 - 0.15 * 115 (current train proportion, removing validation, for inferred one)
        ),
        "oracle_sparse_q90": OracleSparsePerformanceStrategy(
            cold_start_n=98,
            kl_quantile=0.90,
        ),
        "oracle_sparse_q50": OracleSparsePerformanceStrategy(
            cold_start_n=98,
            kl_quantile=0.50,
        ),
        "oracle_sparse_q25": OracleSparsePerformanceStrategy(
            cold_start_n=98,
            kl_quantile=0.25,
        ),
        "rand_0_05": RandomStrategy(prob_true=0.05),
        "rand_0_10": RandomStrategy(prob_true=0.10),
        "rand_0_15": RandomStrategy(prob_true=0.15),
        "rand_0_25": RandomStrategy(prob_true=0.25),
        "oracle_ds_200": OracleCoverageStrategy(counter=counter, n_corpus=200),
        "oracle_ds_100": OracleCoverageStrategy(counter=counter, n_corpus=100),
        "oracle_ds_500": OracleCoverageStrategy(counter=counter, n_corpus=500),
        "repr_vmf_kap1": HeuristicsStrategy(
            dissimilarity=False,
            var_uncertainty=False,
            ranking_unc=False,
            vmf=True,
            vmf_kwargs={"strategy": "quantile", "fixed_quantile": 0.05, "kappa": 1},
        ),
        "repr_vmf_kap1_q15": HeuristicsStrategy(
            dissimilarity=False,
            var_uncertainty=False,
            ranking_unc=False,
            vmf=True,
            vmf_kwargs={"strategy": "quantile", "fixed_quantile": 0.15, "kappa": 1},
        ),
        "repr_vmf_kap1_q25": HeuristicsStrategy(
            dissimilarity=False,
            var_uncertainty=False,
            ranking_unc=False,
            vmf=True,
            vmf_kwargs={"strategy": "quantile", "fixed_quantile": 0.25, "kappa": 1},
        ),
        "act_repr_min": HeuristicsStrategy(
            dissimilarity=True,
            var_uncertainty=False,
            ranking_unc=False,
            vmf=False,
            diss_kwargs={
                "type_aggregation": "min",
                "fixed_quantile": 0.05,
            },
        ),
        "act_repr_min_q15": HeuristicsStrategy(
            dissimilarity=True,
            var_uncertainty=False,
            ranking_unc=False,
            vmf=False,
            diss_kwargs={
                "type_aggregation": "min",
                "fixed_quantile": 0.15,
            },
        ),
        "act_repr_min_q25": HeuristicsStrategy(
            dissimilarity=True,
            var_uncertainty=False,
            ranking_unc=False,
            vmf=False,
            diss_kwargs={
                "type_aggregation": "min",
                "fixed_quantile": 0.25,
            },
        ),
        "unc_t3": HeuristicsStrategy(
            dissimilarity=False,
            var_uncertainty=True,
            ranking_unc=False,
            vmf=False,
            unc_kwargs={
                "window_quantile": 0.95,
                "seed": seed,
                "uncertainty_top_k": 3,
            },
        ),
        "unc_t3_q85": HeuristicsStrategy(
            dissimilarity=False,
            var_uncertainty=True,
            ranking_unc=False,
            vmf=False,
            unc_kwargs={
                "window_quantile": 0.85,
                "seed": seed,
                "uncertainty_top_k": 3,
            },
        ),
        "unc_t3_q75": HeuristicsStrategy(
            dissimilarity=False,
            var_uncertainty=True,
            ranking_unc=False,
            vmf=False,
            unc_kwargs={
                "window_quantile": 0.75,
                "seed": seed,
                "uncertainty_top_k": 3,
            },
        ),
        "unc_mc_t3": HeuristicsStrategy(
            dissimilarity=False,
            var_uncertainty=False,
            ranking_unc=True,
            vmf=False,
            ranking_unc_kwargs={
                "window_quantile": 0.95,
                "seed": seed,
                "uncertainty_top_k": 3,
            },
        ),
        "unc_mc_t3_q85": HeuristicsStrategy(
            dissimilarity=False,
            var_uncertainty=False,
            ranking_unc=True,
            vmf=False,
            ranking_unc_kwargs={
                "window_quantile": 0.85,
                "seed": seed,
                "uncertainty_top_k": 3,
            },
        ),
        "unc_mc_t3_q75": HeuristicsStrategy(
            dissimilarity=False,
            var_uncertainty=False,
            ranking_unc=True,
            vmf=False,
            ranking_unc_kwargs={
                "window_quantile": 0.75,
                "seed": seed,
                "uncertainty_top_k": 3,
            },
        ),
    }

    if type_experiment == "main_stationary":
        run_main_experiment(
            budget=budget,
            seed=seed,
            benchmark=benchmark,
            embedding_model=embedding_model,
            ds_loader=ds_loader,
            strategies=strategies,
            max_step="all",
            cost_penalty=cost_penalty,
            k=k_values,
        )
    elif type_experiment == "main_domain_non_stationary":
        run_domain_shift_experiment(
            budget=budget,
            seed=seed,
            benchmark=benchmark,
            embedding_model=embedding_model,
            ds_loader=ds_loader,  # type: ignore
            strategies=strategies,
            k=k_values,
            max_step="all",
            cost_penalty=cost_penalty,
        )
    elif type_experiment == "partial_annotation":
        partial_strategies: dict[str, InferredSparsePerformanceStrategy] = {
            name: strategy
            for name, strategy in strategies.items()
            if isinstance(strategy, InferredSparsePerformanceStrategy)
        }
        run_partial_annotation_experiment(
            budget=budget,
            seed=seed,
            benchmark=benchmark,
            embedding_model=embedding_model,
            ds_loader=ds_loader,
            strategies=partial_strategies,
            k=k_values,
            max_step="all",
            cost_penalty=cost_penalty,
        )
    elif type_experiment == "model_switch_regret":
        regret_budget = 1000 if benchmark == "EmbedLLM" else budget
        regret_strategy = InferredSparsePerformanceStrategy(
            cold_start_n=115,
            budget=regret_budget,
            seed=seed,
            percentile=75,
            model_settings={"hidden_dim": 16},
            training_settings={
                "learning_rate": 1e-3,
                "weight_decay": 1e-3,
            },
        )
        run_model_switch_regret_experiment(
            budget=regret_budget,
            seed=seed,
            benchmark=benchmark,
            embedding_model=embedding_model,
            ds_loader=ds_loader,
            strategy=regret_strategy,
            cost_penalty=cost_penalty,
        )
    else:
        raise NotImplementedError(f"Unknown experiment type: {type_experiment}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run active learning routing experiments"
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=[
            "main_stationary",
            "main_domain_non_stationary",
            "model_switch_regret",
            "partial_annotation",
        ],
        help="Type of experiment to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Name of the embedding model to use",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["EmbedLLM", "RouterBench", "Sprout", "FusionBench", "R2Bench"],
        help="Name of the benchmark dataset to use",
    )
    parser.add_argument(
        "--cost_penalty",
        type=float,
        default=0.0,
        help="Cost penalty factor (default: 0.0)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        default=500,
        help="Budget for the experiment",
    )

    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[40],
        help="List of k values for KNN (default: [5]). Example: --k_values 3 5 7 10",
    )

    args = parser.parse_args()

    run(
        type_experiment=args.type,
        embedding_model=args.embedding_model,
        benchmark=args.benchmark,
        seed=args.seed,
        budget=args.budget,
        k_values=args.k_values,
        cost_penalty=args.cost_penalty,
    )
