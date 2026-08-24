import random
from typing import Any, Dict, Literal, List, Generator
import ast

from datasets.formatting.formatting import LazyRow  # type: ignore
from datasets import load_dataset, DatasetDict, Dataset  # type: ignore
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from datasets import concatenate_datasets  # type: ignore
from unidecode import unidecode

from datasets_management.embedding import EmbeddingModel, get_embeddings
from logging_config import logger


def min_max_normalize(arr: list[float]) -> np.ndarray:
    """
    Apply min-max normalization to scale values to [0, 1] range.

    Args:
        values: List of values to normalize

    Returns:
        Normalized array with values in [0, 1] range

    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)


class CostMapping(BaseModel):
    cost_map: dict[str, float]


class DatasetManagement:
    def __init__(
        self,
        seed: int,
        benchmark: Literal[
            "EmbedLLM", "RouterBench", "Sprout", "FusionBench", "R2Bench"
        ],
        embedder: EmbeddingModel | None,
        train_test_sizes: tuple[int, int] = (1000, 5000),
        use_all_remaining_as_train: bool = False,
    ):
        self.ds: Dataset = load_dataset("Wikit/RoutingCompendium-perf", split=benchmark)
        ds_cost: Dataset = load_dataset("Wikit/RoutingCompendium-cost", split=benchmark)
        self.cost_mapping = self.get_cost_mapping(dataset_cost=ds_cost)
        self.benchmark = benchmark
        self.embedder = embedder
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.final_ds = self._get_sampled_dataset(
            train_test_sizes[0], train_test_sizes[1], use_all_remaining_as_train
        )

    def get_cost_mapping(
        self,
        dataset_cost: Dataset,
    ) -> dict[str, int | float | dict[str, float]]:
        return {
            item["model"]: ast.literal_eval(item["cost"])  # type: ignore
            for item in dataset_cost.to_list()  # type: ignore
        }

    @staticmethod
    def translate_parameters_to_dollars(n_parameters: int | float) -> float:
        """
        Returns the price per 1 million tokens based on the number of parameters in billions.
        The values are extracted from TogetherAI pricing page, accessed on 05/06/2025.

        https://www.together.ai/pricing

        Args:
            n_parameters (int): The number of parameters in billions.

        Returns:
            float: The price per 1 million tokens in USD.
        """
        if n_parameters <= 4:
            return 0.10
        elif n_parameters <= 8:
            return 0.20
        elif n_parameters <= 21:
            return 0.30
        elif n_parameters <= 41:
            return 0.80
        elif n_parameters <= 80:
            return 0.90
        elif n_parameters <= 110:
            return 1.80
        else:
            # Interpolate beyond 110B using last known interval
            # From 80B to 110B: (110 - 80) = 30B increase leads to (1.80 - 0.90) = $0.90 increase
            # So slope = 0.90 / 30 = $0.03 per billion parameters
            base_price = 1.80
            extra_parameters = n_parameters - 110
            additional_cost = extra_parameters * 0.03
            return base_price + additional_cost

    def get_cost_into_dollars_strategy(
        self,
        cost_mapping: dict[str, float | int | dict[str, float]],
        translate_parameters: bool,
    ) -> CostMapping:
        for model in cost_mapping:
            cost_model = cost_mapping[model]
            if (
                isinstance(cost_model, int)
                or isinstance(cost_model, float)
                and translate_parameters
            ):
                cost_mapping[model] = self.translate_parameters_to_dollars(
                    n_parameters=cost_model
                )
            elif (
                isinstance(cost_model, int)
                or isinstance(cost_model, float)
                and not translate_parameters
            ):
                cost_mapping[model] = cost_model
            elif isinstance(cost_model, dict) and "input" in cost_model.keys():
                cost_mapping[model] = cost_model["input"]

            else:
                raise ValueError(
                    f"Something wrong with the cost of {model} for the benchmark: {self.benchmark}"
                )
        return CostMapping(cost_map=cost_mapping)  # type: ignore

    def normalize_performance(
        self, scaler: MinMaxScaler, dataset_dict: DatasetDict
    ) -> DatasetDict:
        """Normalise the models performances between 0 and 1"""
        try:
            required_splits = ["train", "test"]
            for split in required_splits:
                if split not in dataset_dict:
                    raise KeyError(f"Dataset must contain '{split}' split")
                if "models_performance" not in dataset_dict[split].column_names:
                    raise KeyError(
                        f"'{split}' split must contain 'models_performance' column"
                    )

            train = dataset_dict["train"]
            train_performance = dataset_dict["train"]["models_performance"]  # type: ignore
            if not train_performance:
                raise ValueError("Training performance data cannot be empty")

            normalized_train_perf = scaler.fit_transform(train_performance)  # type: ignore
            train = train.remove_columns("models_performance").add_column(  # type: ignore
                "models_performance", list(normalized_train_perf)
            )

            test = dataset_dict["test"]
            test_performance = dataset_dict["test"]["models_performance"]  # type: ignore
            normalized_test_perf = scaler.transform(test_performance)  # type: ignore
            test = test.remove_columns("models_performance").add_column(  # type: ignore
                "models_performance", list(normalized_test_perf)
            )

            return DatasetDict({"train": train, "test": test})

        except Exception as e:
            print(f"Error normalizing performance: {str(e)}")
            raise

    def clean_normalise_costs(
        self, cost_mapping: dict[str, float | int | dict[str, float]]
    ) -> CostMapping:
        translate_parameters = (
            True if self.benchmark in ["Sprout", "EmbedLLM"] else False
        )
        cost_mapping_dollars = self.get_cost_into_dollars_strategy(
            cost_mapping, translate_parameters=translate_parameters
        )
        return cost_mapping_dollars

    def remove_indices(
        self, example: Dict[str, Any], indices: List[int]
    ) -> Dict[str, Any]:
        """
        Filter model performance and names based on specified indices.

        Args:
            example: Dataset example containing models_performance and models_name
            indices: List of indices to keep from the original arrays

        Returns:
            Dict[str, Any]: Modified example with filtered model data

        Raises:
            KeyError: If required keys are missing from the example
            IndexError: If indices are out of bounds
        """
        try:
            if "models_performance" in example:
                example["models_performance"] = np.array(example["models_performance"])[
                    indices
                ]
            if "models_name" in example:
                example["models_name"] = np.array(example["models_name"])[indices]
            if "cost" in example:
                example["cost"] = np.array(example["cost"])[indices]
            return example
        except (KeyError, IndexError) as e:
            print(f"Error filtering indices in example: {str(e)}")
            return example

    def keep_models_with_costs(self, dataset: Dataset, cost: CostMapping) -> Dataset:
        """
        Filter dataset to keep only models that have associated cost information.

        This function identifies models that exist in the cost mapping and removes
        all other models from the dataset across all splits (train/validation/test).

        Args:
            dataset: HuggingFace Dataset containing model data
            cost: CostMapping object with cost information for models

        Returns:
            Dataset: Filtered dataset containing only models with cost data

        Raises:
            KeyError: If required dataset splits or columns are missing
            ValueError: If no models have cost information
        """
        try:
            models_name: list[str] = dataset["models_name"][0]  # type: ignore
            implemented_set = set(cost.cost_map.keys())

            # Find indices of models that have cost information
            keep_indices: List[int] = [
                i for i, model in enumerate(models_name) if model in implemented_set  # type: ignore
            ]

            if not keep_indices:
                raise ValueError("No models found with cost information")

            logger.info(f"Keep indices: {keep_indices}")
            logger.info(
                f"Indices kept: {len(keep_indices)} out of {len(models_name)} total models"  # type: ignore
            )

            return dataset.map(self.remove_indices, fn_kwargs={"indices": keep_indices})  # type: ignore

        except Exception as e:
            print(f"Error filtering models with costs: {str(e)}")
            raise

    def add_cost(self, example: Dict[str, Any], cost: List[float]) -> Dict[str, Any]:
        """
        Add cost information to a dataset example.

        Args:
            example: Dataset example to modify
            cost: List of cost values for each model

        Returns:
            Dict[str, Any]: Example with added cost information
        """
        example["cost"] = cost
        return example

    @staticmethod
    def remove_special_characters(example: LazyRow, prompt_variable: str):
        example[prompt_variable] = unidecode(example[prompt_variable])  # type: ignore

        return example

    def _get_sampled_dataset(
        self,
        train_size_available: int = 1000,
        test_size_available: int = 4000,
        use_all_remaining_as_train: bool = False,
    ) -> DatasetDict:
        """
        Create a sampled, embedded, and preprocessed dataset with train/test splits.

        This method performs the following steps:
        1. Samples a subset of the original dataset
        2. Removes special characters from prompts
        3. Generates embeddings for all prompts
        4. Cleans and normalizes cost information
        5. Filters models based on cost availability (for EmbedLLM benchmark)
        6. Adds normalized cost data to each example
        7. Splits data into train and test sets
        8. Normalizes model performance scores

        Args:
            train_size_available: Number of samples to include in the training set. Defaults to 1000.
            test_size_available: Number of samples to include in the test set. Defaults to 4000.

        Returns:
            DatasetDict: A dictionary containing 'train' and 'test' splits with the following fields:
                - embeddings: Encoded prompt embeddings
                - models_performance: Normalized performance scores for each model
                - models_name: Names of available models
                - cost: Normalized cost values for each model
                - Other original fields from the source dataset
        """
        n_available = len(self.ds)

        if use_all_remaining_as_train:
            ds_sampled = self.ds.class_encode_column("dataset").shuffle(seed=self.seed)  # type: ignore
            test_split_size: int | float = test_size_available
            logger.info(
                f"use_all_remaining_as_train=True: test={test_size_available}, "
                f"train={n_available - test_size_available} (full corpus minus test)."
            )
        else:
            size_to_get = train_size_available + test_size_available
            if size_to_get < n_available:
                ds_sampled = self.ds.class_encode_column("dataset").train_test_split(  # type: ignore
                    test_size=size_to_get,
                    stratify_by_column="dataset",
                    seed=self.seed,
                )[
                    "test"
                ]
                test_split_size = test_size_available
            else:
                logger.warning(
                    f"Requested {size_to_get} samples but dataset only has {n_available}. "
                    f"Using full dataset with proportional split "
                    f"({train_size_available / size_to_get:.1%} train / "
                    f"{test_size_available / size_to_get:.1%} test)."
                )
                ds_sampled = self.ds.class_encode_column("dataset").shuffle(seed=self.seed)  # type: ignore
                test_split_size = test_size_available / size_to_get
        ds_sampled_cleaned = ds_sampled.map(  # type: ignore
            self.remove_special_characters, fn_kwargs={"prompt_variable": "prompt"}
        )
        if self.embedder is not None:
            ds_sampled_embedded = ds_sampled_cleaned.map(  # type: ignore
                get_embeddings,
                fn_kwargs={"embedder": self.embedder},
                batch_size=32,
                load_from_cache_file=False,
            )
        else:
            ds_sampled_embedded = ds_sampled_cleaned

        cost_mapping = self.clean_normalise_costs(self.cost_mapping)

        if self.benchmark == "EmbedLLM":
            ds_sampled_embedded = self.keep_models_with_costs(
                dataset=ds_sampled_embedded,
                cost=cost_mapping,
            )

        models_name: list[str] = ds_sampled_embedded["models_name"][0]  # type: ignore
        cost_mapping.cost_map = {
            model: cost_mapping.cost_map[model] for model in models_name  # type: ignore
        }

        ds_sampled_embedded = ds_sampled_embedded.map(  # type: ignore
            self.add_cost,
            fn_kwargs={"cost": min_max_normalize(list(cost_mapping.cost_map.values()))},
        )
        counts = Counter(ds_sampled_embedded["dataset"])  # type: ignore
        rare = ds_sampled_embedded.filter(lambda x: counts[x["dataset"]] < 2)  # type: ignore
        stratifiable = ds_sampled_embedded.filter(lambda x: counts[x["dataset"]] >= 2)  # type: ignore
        split_ds = stratifiable.train_test_split(
            test_size=test_split_size,
            stratify_by_column="dataset",
            seed=self.seed,
        )
        if len(rare) > 0:
            split_ds["test"] = concatenate_datasets([split_ds["test"], rare])

        split_ds = self.normalize_performance(
            scaler=MinMaxScaler(), dataset_dict=split_ds
        )
        return split_ds

    def get_training_generator(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generate training data items one at a time from the training dataset.

        Yields:
            Dict[str, Any]: Training data items containing embeddings, models_performance,
                models_name, cost, and other fields from the training split.
        """
        for item in self.final_ds["train"]:  # type: ignore
            yield item  # type: ignore

    def get_training_dataset(self) -> Dataset:
        """
        Get the complete training dataset.

        Returns:
            Dataset: The training split containing all training examples with embeddings,
                models_performance, models_name, cost, and other preprocessed fields.
        """
        return self.final_ds["train"]

    def get_test_dataset(self) -> Dataset:
        """
        Get the complete test dataset.

        Returns:
            Dataset: The test split containing all test examples with embeddings,
                models_performance, models_name, cost, and other preprocessed fields.
        """
        return self.final_ds["test"]
