import copy
import json
import os
from typing import Any, Literal

import numpy as np
import polars as pl
from scipy.stats import wilcoxon  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.multitest import multipletests  # type: ignore
from tqdm import tqdm

from datasets_management.stationary_dataset import DatasetManagement
from experiment.base_experiment import BaseExperiment
from experiment.results_tracker import ResultsTracker
from experiment.utils.metrics import calculate_weighted_dto
from logging_config import logger
from sampling_strategies import InferredSparsePerformanceStrategy
from sampling_strategies.oracle_sparse_performance import _kl_from_uniform  # type: ignore

from logging_config import logger


def _stack_embeddings(corpus: pl.DataFrame) -> np.ndarray:
    """Stack embedding vectors from a Polars DataFrame into a 2-D array."""
    emb = np.stack(corpus["embeddings"].to_list())
    if emb.ndim == 3:
        emb = emb.squeeze(1)
    return emb


def _knn_top_k_indices(
    item: dict[str, Any],
    train_emb: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return indices of the k nearest neighbors by cosine similarity."""
    item_emb = np.array(item["embeddings"]).reshape(1, -1)
    sims = cosine_similarity(item_emb, train_emb).squeeze(0)
    k = min(k, len(sims))
    return np.argpartition(sims, -k)[-k:]


def _wilcoxon_pvalue(perf_best: np.ndarray, perf_other: np.ndarray) -> float:
    """One-sided Wilcoxon signed-rank p-value testing perf_best > perf_other."""
    diff = perf_best - perf_other
    if np.all(diff == 0):
        return 1.0
    try:
        _, p = wilcoxon(diff, alternative="greater")
    except ValueError:
        return 1.0
    return float(p)


def _precompute_knn(
    item: dict[str, Any],
    train_corpus: pl.DataFrame | None,
    k: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Precompute train_perf and top-k indices once for both selection and imputation.

    Returns (train_perf[top_k], top_k_idx) or (None, None) if corpus too small.
    """
    if train_corpus is None or len(train_corpus) < k:
        return None, None
    train_emb = _stack_embeddings(train_corpus)
    train_perf = np.stack(train_corpus["models_performance"].to_list())
    top_k_idx = _knn_top_k_indices(item, train_emb, k)
    return train_perf, top_k_idx


def select_wilcoxon_models(
    item: dict[str, Any],
    neighbor_perf: np.ndarray | None,
    alpha: float = 0.05,
) -> np.ndarray:
    """Return a boolean mask over models: True = annotate, False = skip.

    Models whose KNN-neighbor performance is significantly worse than the best
    model (one-sided Wilcoxon + Holm correction) are masked out.
    """
    n_models = len(item["models_name"])
    mask = np.ones(n_models, dtype=bool)
    if neighbor_perf is None:
        return mask
    best_idx = int(np.argmax(neighbor_perf.mean(axis=0)))
    other_indices = [i for i in range(n_models) if i != best_idx]
    if not other_indices:
        return mask
    p_values = np.array(
        [
            _wilcoxon_pvalue(neighbor_perf[:, best_idx], neighbor_perf[:, i])
            for i in other_indices
        ]
    )
    reject, _, _, _ = multipletests(p_values, alpha=alpha, method="holm")
    for i, model_idx in enumerate(other_indices):
        if reject[i]:
            mask[model_idx] = False
    return mask


def impute_missing_performances(
    mask: np.ndarray,
    observed_perf: np.ndarray,
    train_perf: np.ndarray | None,
    top_k_idx: np.ndarray | None,
) -> np.ndarray:
    """Fill in performances for non-annotated models using KNN-neighbor mean.

    Falls back to the mean of observed values when precomputed KNN data is unavailable.
    """
    full_perf = observed_perf.copy()
    if not (~mask).any():
        return full_perf
    if train_perf is None or top_k_idx is None:
        full_perf[~mask] = float(np.nanmean(observed_perf))
        return full_perf
    full_perf[~mask] = train_perf[top_k_idx][:, ~mask].mean(axis=0)
    return full_perf


class DeferredUpdateStrategy(InferredSparsePerformanceStrategy):
    """Variant that separates the selection decision from the learner update.

    ``should_select`` only returns the decision without modifying internal state.
    After computing the (possibly imputed) item, call ``confirm_selection(item)``
    to update KL history and the MLP learner with the actual performances.
    """

    def __init__(self, source: InferredSparsePerformanceStrategy) -> None:
        super().__init__(
            budget=source.budget,
            model_settings={"hidden_dim": source._hidden_dim},
            training_settings={
                "learning_rate": source._learning_rate,
                "weight_decay": source._weight_decay,
                "loader_batch_size": source._loader_batch_size,
                "n_epochs": source._n_epochs,
                "trigger_every": source._trigger_every,
            },
            cold_start_n=source.cold_start_n,
            percentile=source.percentile,
            seed=source.seed,
        )
        self._pending_inp: Any = None

    def should_select(
        self,
        item: dict[str, str | float | int],
        current_train_ds: pl.DataFrame | None,
    ) -> bool:
        """Decide whether to select this item, without updating the learner.

        Budget enforcement is handled externally by the experiment loop
        (cost-based budget), so the internal query-count check is skipped.
        """
        inp = self._extract_input(item)

        if len(self._kl_history) < self.cold_start_n:
            self._pending_inp = inp
            return True

        if self._val_loss >= 0.20:
            logger.info(
                f"[Guardrail] val_loss={self._val_loss:.4f} >= 0.20 — "
                f"forcing selection and retraining (n_selected={len(self._kl_history)})"
            )
            self._pending_inp = inp
            return True

        # Normal: compare predicted KL to threshold
        predicted_kl = self._predict_kl(inp)
        threshold = float(np.percentile(self._kl_history, self.percentile))
        if predicted_kl > threshold:
            self._pending_inp = inp
            return True

        return False

    def is_in_cold_start(self) -> bool:
        """Whether the strategy is still in its cold-start phase."""
        return len(self._kl_history) < self.cold_start_n

    def confirm_selection(self, item: dict[str, Any]) -> None:
        """Update KL history and learner with the (possibly imputed) performances."""
        kl = _kl_from_uniform(np.array(item["models_performance"]))
        inp = self._pending_inp

        if len(self._kl_history) < self.cold_start_n:
            self._cold_start_buffer.append((inp, kl))  # type: ignore
            self._kl_history.append(kl)
            if len(self._kl_history) == self.cold_start_n:
                learner = self._get_learner(inp)
                inputs = [x for x, _ in self._cold_start_buffer]  # type: ignore
                kls = [k for _, k in self._cold_start_buffer]  # type: ignore
                self._val_loss = learner.fit(inputs, kls)
                self._cold_start_buffer = []
        else:
            self._kl_history.append(kl)
            learner = self._get_learner(inp)
            val_loss = learner.update(inp, kl)
            if val_loss is not None:
                self._val_loss = val_loss

        self._pending_inp = None


class PartialAnnotationExperiment(BaseExperiment):
    """Runs Wilcoxon partial-annotation variants of inferred_sparse strategies.

    Only the Wilcoxon branch is executed here; full-annotation baselines are
    loaded from main_stationary results at plot time.  Each strategy gets its
    own ``DeferredUpdateStrategy`` so the MLP learner trains on imputed (not
    true) KL values — no information leakage.
    """

    def __init__(
        self,
        budget: int,
        seed: int,
        embedding_model: str,
        benchmark: str,
        ds_loader: DatasetManagement,
        strategies: dict[str, InferredSparsePerformanceStrategy],
        k: list[int],
        alpha: float = 0.05,
        output_dir: str = "results",
        max_step: int | Literal["all"] = "all",
        cost_penalty: float = 0.0,
    ) -> None:
        self._base_strategy_names = list(strategies.keys())
        wilcoxon_strategies: dict[str, DeferredUpdateStrategy] = {}
        for name, strat in strategies.items():
            wilcoxon_strategies[name] = DeferredUpdateStrategy(source=strat)

        super().__init__(
            seed=seed,
            embedding_model=embedding_model,
            benchmark=benchmark,
            ds_loader=ds_loader,
            strategies=wilcoxon_strategies,  # type: ignore
            output_dir=output_dir,
            max_step=max_step,
            cost_penalty=cost_penalty,
            k=k,
        )
        self.budget = budget
        self.alpha = alpha
        self.k = k

    def preprocess_dataset(self) -> None:
        """No preprocessing needed."""
        pass

    def get_experiment_name(self) -> str:
        """Return experiment path: partial_annotation/<embedding>/<benchmark>/<seed>/run_<ts>."""
        return f"partial_annotation/{self.embedding_model}/{self.benchmark}/{self.seed}/run_{self.timestamp}"

    def run_streaming_loop(self) -> dict[str, list[int]]:
        """Stream through training items with Wilcoxon partial annotation.

        Each strategy uses a DeferredUpdateStrategy whose learner is updated
        with imputed (not true) KL values.  The budget is cost-based:
        ``budget * full_annotation_cost_per_query``.  Since Wilcoxon skips
        some models, each query costs less, allowing more queries to be
        annotated within the same total cost budget.
        """
        train_ds = self.ds_loader.get_training_dataset()
        first_item_costs = np.array(train_ds[0]["cost"], dtype=float)
        full_cost_per_query = float(first_item_costs.sum())
        cost_budget = self.budget * full_cost_per_query
        logger.info(
            f"Cost budget: {self.budget} queries * {full_cost_per_query:.4f} "
            f"cost/query = {cost_budget:.4f}"
        )

        samples_selected: dict[str, int] = {name: 0 for name in self.strategies}
        cumulative_ann_cost: dict[str, float] = {name: 0.0 for name in self.strategies}
        model_names = train_ds[0]["models_name"]  # type: ignore
        n_models = len(model_names)  # type: ignore
        post_cs_annotation_counts: dict[str, np.ndarray] = {
            name: np.zeros(n_models, dtype=int) for name in self.strategies
        }
        total_annotations: dict[str, int] = {name: 0 for name in self.strategies}
        post_cs_queries: dict[str, int] = {name: 0 for name in self.strategies}
        post_cs_annotations: dict[str, int] = {name: 0 for name in self.strategies}
        previous_metrics: dict[str, dict[int, dict[str, float]]] = {
            name: {
                kv: {"avg_perf": 0.0, "avg_cost": 0.0, "avg_mse": 0.0, "dto": 0.0}
                for kv in self.k
            }
            for name in self.strategies
        }

        length_stream = len(train_ds)
        dict_selected: dict[str, list[int]] = {s: [] for s in self.strategies}

        for step, item in enumerate(
            tqdm(
                self.ds_loader.get_training_generator(),
                desc=self.get_progress_description(),
                total=length_stream,
            )
        ):
            if step > length_stream:
                break

            model_costs = np.array(item["cost"], dtype=float)

            for name in self._base_strategy_names:
                strategy: DeferredUpdateStrategy = self.strategies[name]  # type: ignore
                evaluator = self.evaluators[name]

                should_select = strategy.should_select(
                    item=item,
                    current_train_ds=evaluator.train_df,
                )

                if should_select and cumulative_ann_cost[name] < cost_budget:
                    dict_selected[name].append(step)

                    # cold start: annotate all models (no filtering)
                    in_cold_start = strategy.is_in_cold_start()
                    true_perf = np.array(item["models_performance"], dtype=float)
                    k_select = self.k[0]

                    if in_cold_start:
                        annotated_item = item
                        mask = np.ones(len(true_perf), dtype=bool)
                    else:
                        train_perf, top_k_idx = _precompute_knn(
                            item, evaluator.train_df, k_select
                        )
                        neighbor_perf = (
                            train_perf[top_k_idx]
                            if train_perf is not None and top_k_idx is not None
                            else None
                        )
                        mask = select_wilcoxon_models(
                            item,
                            neighbor_perf,
                            alpha=self.alpha,
                        )
                        observed = np.where(mask, true_perf, np.nan)
                        imputed = impute_missing_performances(
                            mask, observed, train_perf, top_k_idx
                        )
                        annotated_item = copy.copy(item)
                        annotated_item["models_performance"] = imputed.tolist()

                    # update learner with (possibly imputed) performances
                    strategy.confirm_selection(annotated_item)

                    total_annotations[name] += int(mask.sum())
                    if not in_cold_start:
                        post_cs_annotation_counts[name] += mask.astype(int)
                        post_cs_queries[name] += 1
                        post_cs_annotations[name] += int(mask.sum())

                    # annotation cost: sum of costs for evaluated models
                    cumulative_ann_cost[name] += float(model_costs[mask].sum())

                    results = evaluator.evaluate_on_test(
                        train_item=annotated_item, k=self.k
                    )
                    for kv in results:
                        results[kv]["dto"] = float(
                            calculate_weighted_dto(
                                oracle_cost=self.o_cost,
                                oracle_perf=self.o_perf,
                                test_cost=results[kv]["avg_cost"],
                                test_perf=results[kv]["avg_perf"],
                            )
                        )
                    samples_selected[name] += 1
                    previous_metrics[name] = results
                else:
                    evaluator.redundant_step_no_annotation()
                    results = previous_metrics[name]

                for kv in results:
                    self.tracker.log_step(
                        step=step,
                        k_value=kv,
                        strategy=name,
                        test_performance=results[kv]["avg_perf"],
                        test_cost=results[kv]["avg_cost"],
                        test_mse=results[kv]["avg_mse"],
                        train_cost=evaluator.cost_train,
                        samples_selected=samples_selected[name],
                        dto=results[kv]["dto"],
                        annotation_cost=cumulative_ann_cost[name],
                    )

        annotation_stats: dict[str, dict[str, Any]] = {}
        for name in self._base_strategy_names:
            counts = post_cs_annotation_counts[name]
            annotation_stats[name] = {
                "per_model_annotation_count": dict(zip(model_names, counts.tolist())),  # type: ignore
                "mean_annotations_per_query": (
                    float(post_cs_annotations[name] / post_cs_queries[name])
                    if post_cs_queries[name] > 0
                    else 0.0
                ),
            }
        self.tracker.results["annotation_stats"] = annotation_stats  # type: ignore

        return dict_selected

    def save_additional_data(self) -> None:
        """Save global AUC metrics (perf + DTO) to JSON."""
        experiment_name = self.get_experiment_name()
        seed_dir = f"{self.output_dir}/{experiment_name}"
        os.makedirs(seed_dir, exist_ok=True)

        auc_global_file = os.path.join(seed_dir, f"auc_global_{self.timestamp}.json")
        auc_global_metrics = self.tracker.calculate_auc_metrics()
        with open(auc_global_file, "w", encoding="utf-8") as f:
            json.dump(auc_global_metrics, f, indent=2)
        logger.info(f"Global AUC metrics saved to: {auc_global_file}")


def run_partial_annotation_experiment(
    budget: int,
    seed: int,
    embedding_model: str,
    benchmark: str,
    ds_loader: DatasetManagement,
    strategies: dict[str, InferredSparsePerformanceStrategy],
    k: list[int],
    alpha: float = 0.05,
    output_dir: str = "results",
    max_step: int | Literal["all"] = "all",
    cost_penalty: float = 0.0,
) -> ResultsTracker:
    experiment = PartialAnnotationExperiment(
        budget=budget,
        seed=seed,
        embedding_model=embedding_model,
        benchmark=benchmark,
        ds_loader=ds_loader,
        strategies=strategies,
        k=k,
        alpha=alpha,
        output_dir=output_dir,
        max_step=max_step,
        cost_penalty=cost_penalty,
    )
    return experiment.run()
