from typing import Literal, Protocol, Any
from numpy.typing import ArrayLike

import polars as pl

from active_learning.heuristics.diversity import MinCosineEvaluation, VmFCoverage
from active_learning.heuristics.uncertainty import (
    VarianceUncertaintyEvaluation,
    RankingUncertaintyEvaluation,
)

from .base_strategy import BaseSelectionStrategy


class AcquisitionFunction(Protocol):
    """Protocol defining the interface for acquisition functions in active learning.

    All acquisition functions must implement __call__ with at least embedding_query
    and training_dataset parameters.
    """

    def __call__(
        self,
        embedding_query: ArrayLike,
        training_dataset: pl.DataFrame | None,
    ) -> bool:
        """Evaluate if a sample should be selected.

        Args:
            embedding_query: Query embedding to evaluate
            training_dataset: Current training dataset

        Returns:
            True if sample should be selected, False otherwise
        """
        ...

    def get_params(self) -> dict[str, Any]:
        """Get the configuration parameters of the acquisition function.

        Returns:
            Dictionary of parameter names and values
        """
        ...


class HeuristicsStrategy(BaseSelectionStrategy):
    """Active learning sample selection strategy.

    Supports multiple modes:
    - dissimilarity: Select samples dissimilar to the training set (MinCosineEvaluation)
    - uncertainty: Select samples with high weighted-variance uncertainty
    - ranking_unc: Select samples with high Monte Carlo ranking uncertainty
    - vmf: Select samples based on vMF KDE log-likelihood coverage
    """

    def __init__(
        self,
        dissimilarity: bool = True,
        var_uncertainty: bool = False,
        ranking_unc: bool = False,
        vmf: bool = False,
        threshold_strategy: Literal["quantile", "fixed"] = "quantile",
        diss_kwargs: dict[str, str | int | float] | None = None,
        unc_kwargs: dict[str, str | int | float] | None = None,
        ranking_unc_kwargs: dict[str, Any] | None = None,
        vmf_kwargs: dict[str, str | int | float] | None = None,
    ):
        """Initialize active learning strategy.

        Args:
            dissimilarity: Enable dissimilarity mode (cosine dissimilarity)
            uncertainty: Enable weighted-variance uncertainty mode
            ranking_unc: Enable Monte Carlo ranking uncertainty mode
            vmf: Enable vMF KDE coverage mode
            threshold_strategy: Default threshold strategy for repr/unc modes
            diss_kwargs: Additional kwargs for MinCosineEvaluation
            unc_kwargs: Additional kwargs for VarianceUncertaintyEvaluation
            ranking_unc_kwargs: Additional kwargs for RankingUncertaintyEvaluation
            vmf_kwargs: Additional kwargs for VmFCoverage
        """
        self.diss_enabled = dissimilarity
        self.unc_enabled = var_uncertainty
        self.ranking_unc_enabled = ranking_unc
        self.vmf_enabled = vmf

        diss_config = diss_kwargs or {}
        if self.diss_enabled:
            self.representativeness_strategy = MinCosineEvaluation(
                strategy=threshold_strategy, **diss_config  # type: ignore
            )

        unc_config = unc_kwargs or {}
        if self.unc_enabled:
            self.uncertainty_strategy = VarianceUncertaintyEvaluation(
                strategy=threshold_strategy, **unc_config  # type: ignore
            )

        ranking_unc_config = ranking_unc_kwargs or {}
        if self.ranking_unc_enabled:
            self.ranking_unc_strategy = RankingUncertaintyEvaluation(
                strategy=threshold_strategy, **ranking_unc_config  # type: ignore
            )

        vmf_config = vmf_kwargs or {}
        if self.vmf_enabled:
            self.vmf_strategy = VmFCoverage(**vmf_config)  # type: ignore

    def get_params(self) -> dict[str, Any]:
        """Get parameters from the active acquisition function.

        Returns:
            Dictionary of acquisition function parameters
        """
        return self.get_acquisition_function().get_params()

    def get_acquisition_function(self) -> AcquisitionFunction:
        """Get the active acquisition function based on enabled mode.

        Returns:
            The configured acquisition function instance

        Raises:
            NotImplementedError: If no acquisition mode is enabled
        """
        if self.diss_enabled:
            self.name_acquisition = "diss"
            return self.representativeness_strategy
        if self.unc_enabled:
            self.name_acquisition = "unc"
            return self.uncertainty_strategy
        if self.ranking_unc_enabled:
            self.name_acquisition = "ranking_unc"
            return self.ranking_unc_strategy
        if self.vmf_enabled:
            self.name_acquisition = "vmf"
            return self.vmf_strategy
        raise NotImplementedError()

    def should_select(
        self,
        item: dict[str, str | float],
        current_train_ds: pl.DataFrame,
    ) -> bool:
        """Use active learning filtering to determine if sample should be selected.

        Args:
            item: The candidate sample to evaluate
            current_train_ds: The current training dataset
            budget: Budget parameter (unused, kept for interface compatibility)

        Returns:
            Indicates if sample should be selected
        """
        acquisition_function = self.get_acquisition_function()
        return acquisition_function(item["embeddings"], current_train_ds)
