import numpy as np
import numpy.typing as npt


def calculate_weighted_dto(
    oracle_cost: float,
    oracle_perf: float,
    test_cost: float,
    test_perf: float,
    weight_cost: float = 0.25,
    weight_perf: float = 0.75,
) -> npt.NDArray[np.float64]:
    """Calculate weighted DTO metric using weighted euclidean distance."""
    weighted_cost = weight_cost * ((100 - oracle_cost) - (100 - test_cost)) ** 2
    weighted_perf = weight_perf * ((oracle_perf) - (test_perf)) ** 2
    return np.sqrt(weighted_cost + weighted_perf)


def extract_oracle_performance_numpy(
    targets: npt.NDArray[np.float64], costs: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Extract the oracle performance: best performance with minimal cost among best performers.

    For each sample, finds models with the maximum performance, then returns the
    minimum cost among those models.
    """

    row_max = np.max(targets, axis=1, keepdims=True)
    mask = targets == row_max
    min_costs = np.array(
        [
            costs[i][mask[i]].min() if np.any(mask[i]) else 0.0
            for i in range(mask.shape[0])
        ]
    )

    return min_costs, row_max.squeeze(1)
