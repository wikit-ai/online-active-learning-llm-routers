"""
When val_loss >= threshold, f_theta forces selection and retrains.
A threshold at None is equivalent to an ablation of this process: 'None' means guardrail disabled
"""

from typing import Any


from datasets_management.non_stationary_dataset import DatasetManagement
from experiment.main_stationary import run_main_experiment  # type: ignore
from experiment.results_tracker import ResultsTracker
from sampling_strategies import InferredSparsePerformanceStrategy

DEFAULT_GUARDRAIL_THRESHOLDS: list[float | None] = [None, 0.10, 0.20, 0.30, 0.50]
DEFAULT_COLD_START_N = 115


class _InferredSparseNoGuardrail(InferredSparsePerformanceStrategy):
    """Variant with the validation-loss guardrail fully disabled."""

    def _guardrail_degraded_performance(self, inp: Any, true_kl: float) -> bool | None:
        return None


class _InferredSparseCustomGuardrail(InferredSparsePerformanceStrategy):
    """Variant with a configurable validation-loss guardrail threshold."""

    def __init__(self, *args: Any, guardrail_threshold: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._guardrail_threshold = guardrail_threshold

    def _guardrail_degraded_performance(self, inp: Any, true_kl: float) -> bool | None:
        if self._val_loss < self._guardrail_threshold:  # type: ignore
            return None

        print(
            f"[Guardrail] val_loss={self._val_loss:.4f} >= {self._guardrail_threshold} — "  # type: ignore
            f"forcing selection and retraining (n_selected={len(self._kl_history)})"
        )
        learner = self._get_learner(inp)
        self._kl_history.append(true_kl)
        val_loss = learner.update(inp, true_kl)
        if val_loss is not None:
            self._val_loss = val_loss
        return True


def _make_strategy(
    threshold: float | None,
    cold_start_n: int,
    budget: int,
    seed: int,
) -> InferredSparsePerformanceStrategy:
    if threshold is None:
        return _InferredSparseNoGuardrail(
            cold_start_n=cold_start_n, budget=budget, seed=seed
        )
    return _InferredSparseCustomGuardrail(
        cold_start_n=cold_start_n,
        budget=budget,
        seed=seed,
        guardrail_threshold=threshold,
    )


def _threshold_key(threshold: float | None) -> str:
    return "guardrail_off" if threshold is None else f"guardrail_{threshold:.2f}"


def run_guardrail_ablation(
    seed: int,
    benchmark: str,
    embedding_model: str,
    budget: int,
    ds_loader: DatasetManagement,
    k_values: list[int],
    output_dir: str = "results/secondary_experiments/guardrail",
    guardrail_thresholds: list[float | None] | None = None,
    cold_start_n: int = DEFAULT_COLD_START_N,
    cost_penalty: float = 0.0,
) -> ResultsTracker:
    """Run InferredSparsePerformanceStrategy with varying guardrail thresholds."""
    guardrail_thresholds = guardrail_thresholds or DEFAULT_GUARDRAIL_THRESHOLDS

    strategies = {
        _threshold_key(t): _make_strategy(t, cold_start_n, budget, seed)
        for t in guardrail_thresholds
    }

    return run_main_experiment(
        budget=budget,
        seed=seed,
        benchmark=benchmark,
        embedding_model=embedding_model,
        ds_loader=ds_loader,
        strategies=strategies,  # type: ignore
        k=k_values,
        output_dir=output_dir,
        cost_penalty=cost_penalty,
    )
