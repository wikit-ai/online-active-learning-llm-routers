from typing import Callable

from experiment.ablations.cold_start import run_cold_start_ablation
from experiment.ablations.retrain_interval import run_retrain_interval_ablation
from experiment.ablations.guardrail import run_guardrail_ablation

EXPERIMENTS: dict[str, Callable] = {  # type: ignore
    "cold_start": run_cold_start_ablation,
    "retrain_interval": run_retrain_interval_ablation,
    "guardrail": run_guardrail_ablation,
}

__all__ = [
    "run_cold_start_ablation",
    "run_retrain_interval_ablation",
    "run_guardrail_ablation",
    "EXPERIMENTS",
]
