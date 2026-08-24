import copy
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
from datasets import Dataset  # type: ignore
from tqdm import tqdm

from datasets_management.stationary_dataset import DatasetManagement
from experiment.evaluation import MainEvaluator
from experiment.routers.knn_router import KNNRouter
from experiment.utils.metrics import (
    calculate_weighted_dto,
    extract_oracle_performance_numpy,
)
from logging_config import logger
from sampling_strategies import InferredSparsePerformanceStrategy
from sampling_strategies.oracle_sparse_performance import _kl_from_uniform  # type: ignore

_CONDITIONS = ["continual", "hindsight"]
_CHECKPOINT_PCTS = [0.0] + [round(p * 0.1, 1) for p in range(1, 11)]


class ModelSwitchRegretExperiment:
    """Measure regret of pool-switch on routing: continual vs hindsight.

    Setup:
    - Start with half the models active.
    - Run pre-switch phase (first half of stream).
    - At switch: remove 25 % of active models, add 25 % from inactive.
    - Post-switch: stream remaining items, evaluate at every 10 % checkpoint.

    At each checkpoint we report Performance, DTO, sample size, and the
    delta (regret) between continual and hindsight for both metrics.
    """

    def __init__(
        self,
        budget: int,
        seed: int,
        embedding_model: str,
        benchmark: str,
        ds_loader: DatasetManagement,
        strategy: InferredSparsePerformanceStrategy,
        output_dir: str = "results",
        k: list[int] | None = None,
        cost_penalty: float = 0.0,
    ) -> None:
        """Store the experiment configuration and stamp the run timestamp."""
        self.budget = budget
        self.seed = seed
        self.embedding_model = embedding_model
        self.benchmark = benchmark
        self.ds_loader = ds_loader
        self.strategy = strategy
        self.output_dir = output_dir
        self.k = k or [1]
        self.cost_penalty = cost_penalty
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _filter_item(item: dict[str, Any], idx: list[int]) -> dict[str, Any]:
        """Restrict a stream item's performances and costs to the models in `idx`."""
        arr = np.array(idx)
        out = dict(item)
        out["models_performance"] = np.array(item["models_performance"])[arr].tolist()
        if isinstance(item.get("cost"), (list, np.ndarray)):
            out["cost"] = np.array(item["cost"])[arr].tolist()
        out["embeddings"] = np.array(item["embeddings"]).reshape(-1).tolist()
        return out

    @staticmethod
    def _filter_dataset(ds: Dataset, idx: list[int]) -> Dataset:
        """Same as `_filter_item` but applied to every example of a dataset."""
        arr = np.array(idx)

        def _map(ex: dict) -> dict:  # type: ignore
            ex["models_performance"] = np.array(ex["models_performance"])[arr].tolist()  # type: ignore
            if isinstance(ex.get("cost"), (list, np.ndarray)):  # type: ignore
                ex["cost"] = np.array(ex["cost"])[arr].tolist()  # type: ignore
            ex["embeddings"] = np.array(ex["embeddings"]).reshape(1, -1).tolist()  # type: ignore
            return ex  # type: ignore

        return ds.map(_map)  # type: ignore

    def _fresh_strategy(self) -> InferredSparsePerformanceStrategy:
        """Build an untrained strategy with the same hyperparameters as `self.strategy`."""
        return InferredSparsePerformanceStrategy(
            budget=self.strategy.budget,
            cold_start_n=self.strategy.cold_start_n,
            seed=self.strategy.seed,
            model_settings={"hidden_dim": self.strategy._hidden_dim},  # type: ignore
            training_settings={
                "learning_rate": self.strategy._learning_rate,  # type: ignore
                "weight_decay": self.strategy._weight_decay,  # type: ignore
                "loader_batch_size": self.strategy._loader_batch_size,  # type: ignore
                "n_epochs": self.strategy._n_epochs,  # type: ignore
                "trigger_every": self.strategy._trigger_every,  # type: ignore
            },
        )

    @staticmethod
    def _bulk_rekl_fit(
        strat: InferredSparsePerformanceStrategy,
        items_raw: list[dict[str, Any]],
        active_idx: list[int],
    ) -> None:
        """Recompute the KL targets of past annotations under the new pool and refit.

        Used by the continual condition: history is kept but relabelled for `active_idx`.
        """
        if not items_raw:
            return
        arr = np.array(active_idx)
        embeddings = [np.array(it["embeddings"]).reshape(-1) for it in items_raw]
        kl_values = [
            float(_kl_from_uniform(np.array(it["models_performance"])[arr]))
            for it in items_raw
        ]
        learner = strat._get_learner(embeddings[0])  # type: ignore
        strat._val_loss = learner.fit(embeddings, kl_values)  # type: ignore
        strat._kl_history = kl_values[:]  # type: ignore
        strat._cold_start_buffer = []  # type: ignore

    def run(self) -> dict[str, Any]:
        """Run the pre-switch phase, apply the pool switch, then stream both conditions.

        Returns the saved checkpoint log and its output directory.
        """
        training_ds = self.ds_loader.get_training_dataset()
        test_ds = self.ds_loader.get_test_dataset()
        all_items: list[dict[str, Any]] = list(training_ds)  # type: ignore
        length = len(all_items)

        n_models = len(all_items[0]["models_performance"])
        rng = np.random.default_rng(self.seed)

        # Initial active set: half the models are randomly selected
        n_initial = max(1, n_models // 2)
        active = sorted(
            int(x) for x in rng.choice(n_models, size=n_initial, replace=False)
        )

        switch_step = length // 2

        logger.info(
            f"[MSRegret] {n_models} models | active={active} | switch@{switch_step}"
        )

        selected_raw: list[dict[str, Any]] = []
        for step in tqdm(range(switch_step), desc="MSRegret pre-switch"):
            item = all_items[step]
            filtered = self._filter_item(item, active)
            if (
                self.strategy.should_select(filtered, None)
                and len(selected_raw) < self.budget
            ):
                selected_raw.append(dict(item))

        # Switch: remove randomly 25 % of model pool, add randomly 25 % of previously discarded models
        n_active = len(active)
        n_swap = max(1, n_models // 4)
        inactive = sorted(set(range(n_models)) - set(active))

        to_remove = sorted(
            int(x)
            for x in rng.choice(active, size=min(n_swap, n_active - 1), replace=False)
        )
        n_add = min(n_swap, len(inactive))
        to_add = (
            sorted(int(x) for x in rng.choice(inactive, size=n_add, replace=False))
            if n_add > 0
            else []
        )
        new_active = sorted((set(active) | set(to_add)) - set(to_remove))

        logger.info(
            f"[MSRegret] Switch: -{to_remove} +{to_add} -> active={new_active} "
            f"(n_annotated={len(selected_raw)})"
        )

        filtered_test = self._filter_dataset(test_ds, new_active)
        o_cost, o_perf = extract_oracle_performance_numpy(
            np.array(filtered_test["models_performance"]),  # type: ignore
            np.array(filtered_test["cost"]),  # type: ignore
        )
        oracle_cost = float(np.mean(o_cost))
        oracle_perf = float(np.mean(o_perf))

        # continual: warm-start refit on re-KL'd history
        continual_strat = copy.deepcopy(self.strategy)
        if continual_strat.learner is not None:
            continual_strat.learner.dataset._embeddings.clear()
            continual_strat.learner.dataset._kls.clear()
        self._bulk_rekl_fit(continual_strat, selected_raw, new_active)

        # hindsight: strategy which would have the new pool since beginning. Replay stream
        hindsight_strat = self._fresh_strategy()
        hindsight_selected: list[dict[str, Any]] = []
        for item in all_items[:switch_step]:
            filtered = self._filter_item(item, new_active)
            if (
                hindsight_strat.should_select(filtered, None)
                and len(hindsight_selected) < self.budget
            ):
                hindsight_selected.append(dict(item))

        conditions: dict[str, dict[str, Any]] = {
            "continual": {
                "strategy": continual_strat,
                "evaluator": MainEvaluator(
                    router=KNNRouter(),
                    test_ds=filtered_test,
                    cost_penalty=self.cost_penalty,
                ),
                "selected_raw": list(selected_raw),
            },
            "hindsight": {
                "strategy": hindsight_strat,
                "evaluator": MainEvaluator(
                    router=KNNRouter(),
                    test_ds=filtered_test,
                    cost_penalty=self.cost_penalty,
                ),
                "selected_raw": list(hindsight_selected),
            },
        }

        last_metrics: dict[str, dict[int, dict[str, float]]] = {}
        for name, cond in conditions.items():
            for item in cond["selected_raw"]:
                filtered = self._filter_item(item, new_active)
                last_metrics[name] = cond["evaluator"].evaluate_on_test(
                    train_item=filtered, k=self.k
                )

        post_len = length - switch_step
        checkpoint_steps: dict[float, int] = {}
        for pct in _CHECKPOINT_PCTS:
            cp = switch_step + max(1, int(pct * post_len)) - 1
            checkpoint_steps[pct] = min(cp, length - 1)
        checkpoint_steps[1.0] = length - 1

        results_log: list[dict[str, Any]] = [
            {
                "type": "metadata",
                "switch_step": switch_step,
                "to_remove": to_remove,
                "to_add": to_add,
                "initial_active": active,
                "new_active": new_active,
                "n_annotated_pre_switch": len(selected_raw),
                "oracle_cost": oracle_cost,
                "oracle_perf": oracle_perf,
                "checkpoint_steps": {str(k): v for k, v in checkpoint_steps.items()},
                "k": self.k,
            }
        ]

        recorded: set[float] = set()

        entry_0: dict[str, Any] = {
            "type": "checkpoint",
            "checkpoint_pct": 0.0,
            "step": switch_step,
        }
        for name in _CONDITIONS:
            entry_0[f"{name}_n_selected"] = len(conditions[name]["selected_raw"])
            for k_val in self.k:
                m = last_metrics.get(name, {})
                km = m.get(k_val, {"avg_perf": 0.0, "avg_cost": 0.0})
                p = km["avg_perf"]
                dto = float(
                    calculate_weighted_dto(oracle_cost, oracle_perf, km["avg_cost"], p)
                )
                entry_0[f"{name}_perf_k{k_val}"] = p
                entry_0[f"{name}_dto_k{k_val}"] = dto
        for k_val in self.k:
            entry_0[f"delta_perf_k{k_val}"] = (
                entry_0[f"continual_perf_k{k_val}"]
                - entry_0[f"hindsight_perf_k{k_val}"]
            )
            entry_0[f"delta_dto_k{k_val}"] = (
                entry_0[f"continual_dto_k{k_val}"] - entry_0[f"hindsight_dto_k{k_val}"]
            )
        results_log.append(entry_0)
        recorded.add(0.0)

        for step in tqdm(range(switch_step, length), desc="MSRegret post-switch"):
            item = all_items[step]

            for name, cond in conditions.items():
                filtered = self._filter_item(item, new_active)
                should = cond["strategy"].should_select(filtered, None)
                if should and len(cond["selected_raw"]) < self.budget:
                    cond["selected_raw"].append(dict(item))
                    metrics = cond["evaluator"].evaluate_on_test(
                        train_item=filtered, k=self.k
                    )
                    last_metrics[name] = metrics
                else:
                    cond["evaluator"].redundant_step_no_annotation()

            for pct, cp_step in checkpoint_steps.items():
                if step == cp_step and pct not in recorded:
                    recorded.add(pct)
                    entry: dict[str, Any] = {
                        "type": "checkpoint",
                        "checkpoint_pct": pct,
                        "step": step,
                    }
                    for name in _CONDITIONS:
                        entry[f"{name}_n_selected"] = len(
                            conditions[name]["selected_raw"]
                        )
                        for k_val in self.k:
                            m = last_metrics.get(name, {})
                            km = m.get(k_val, {"avg_perf": 0.0, "avg_cost": 0.0})
                            p = km["avg_perf"]
                            dto = float(
                                calculate_weighted_dto(
                                    oracle_cost, oracle_perf, km["avg_cost"], p
                                )
                            )
                            entry[f"{name}_perf_k{k_val}"] = p
                            entry[f"{name}_dto_k{k_val}"] = dto

                    # Deltas between continual and hindsight
                    for k_val in self.k:
                        entry[f"delta_perf_k{k_val}"] = (
                            entry[f"continual_perf_k{k_val}"]
                            - entry[f"hindsight_perf_k{k_val}"]
                        )
                        entry[f"delta_dto_k{k_val}"] = (
                            entry[f"continual_dto_k{k_val}"]
                            - entry[f"hindsight_dto_k{k_val}"]
                        )
                    results_log.append(entry)

        return self._save(results_log)

    def _save(self, results_log: list[dict[str, Any]]) -> dict[str, Any]:
        """Dump the results log as JSON under a run-specific directory."""
        name = (
            f"model_switch_regret/{self.embedding_model}/{self.benchmark}/"
            f"{self.seed}/run_{self.timestamp}"
        )
        out_dir = os.path.join(self.output_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        out_file = os.path.join(out_dir, f"regret_results_{self.timestamp}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results_log, f, indent=2)

        logger.info(f"[MSRegret] Results saved -> {out_file}")
        return {"results": results_log, "output_dir": out_dir}


def run_model_switch_regret_experiment(
    budget: int,
    seed: int,
    embedding_model: str,
    benchmark: str,
    ds_loader: DatasetManagement,
    strategy: InferredSparsePerformanceStrategy,
    output_dir: str = "results",
    k: list[int] | None = None,
    cost_penalty: float = 0.0,
) -> dict[str, Any]:
    """Run the model-switch regret experiment (continual vs hindsight)."""
    exp = ModelSwitchRegretExperiment(
        budget=budget,
        seed=seed,
        embedding_model=embedding_model,
        benchmark=benchmark,
        ds_loader=ds_loader,
        strategy=strategy,
        output_dir=output_dir,
        k=k,
        cost_penalty=cost_penalty,
    )
    return exp.run()
