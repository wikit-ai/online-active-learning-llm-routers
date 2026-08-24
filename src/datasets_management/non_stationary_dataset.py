import random
from typing import Any, Generator, Literal
from datasets import load_dataset, DatasetDict, Dataset  # type: ignore
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datasets import concatenate_datasets  # type: ignore

from datasets_management.embedding import EmbeddingModel, get_embeddings
from datasets_management.stationary_dataset import DatasetManagement, min_max_normalize
from logging_config import logger

DOMAIN_GROUPS: dict[str, dict[str, list[str]]] = {
    "Sprout": {
        "mathematics": [
            "lighteval/MATH/all",
            "lighteval/MATH/all/test",
        ],
        "rag_task_reasoning": [
            "rungalileo/ragbench/covidqa",
            "rungalileo/ragbench/pubmedqa",
            "rungalileo/ragbench/hotpotqa",
            "rungalileo/ragbench/msmarco",
            "rungalileo/ragbench/hagrid",
            "rungalileo/ragbench/expertqa",
            "rungalileo/ragbench/delucionqa",
            "rungalileo/ragbench/cuad",
            "rungalileo/ragbench/emanual",
            "rungalileo/ragbench/finqa",
            "rungalileo/ragbench/tatqa",
            "rungalileo/ragbench/techqa",
            "TAUR-Lab/MuSR",
            "TAUR-Lab/MuSR/object_placements",
            "TAUR-Lab/MuSR/team_allocation",
        ],
        "question_answering": [
            "TIGER-Lab/MMLU-Pro",
        ],
    },
    "RouterBench": {
        "mathematics": [
            "grade-school-math",
            "mmlu-elementary-mathematics",
            "mmlu-high-school-mathematics",
            "mmlu-college-mathematics",
            "mmlu-abstract-algebra",
            "mmlu-high-school-statistics",
            "mmlu-econometrics",
            "mtbench-math",
        ],
        "science": [
            "mmlu-conceptual-physics",
            "mmlu-high-school-physics",
            "mmlu-college-physics",
            "mmlu-astronomy",
            "mmlu-electrical-engineering",
            "mmlu-high-school-chemistry",
            "mmlu-college-chemistry",
            "mmlu-high-school-biology",
            "mmlu-college-biology",
        ],
        "medicine_health": [
            "mmlu-professional-medicine",
            "mmlu-clinical-knowledge",
            "mmlu-college-medicine",
            "mmlu-virology",
            "mmlu-anatomy",
            "mmlu-medical-genetics",
            "mmlu-nutrition",
            "mmlu-human-aging",
            "mmlu-human-sexuality",
        ],
        "law_ethics": [
            "mmlu-professional-law",
            "mmlu-moral-scenarios",
            "mmlu-moral-disputes",
            "mmlu-philosophy",
            "mmlu-logical-fallacies",
            "mmlu-formal-logic",
            "mmlu-international-law",
            "mmlu-jurisprudence",
            "mmlu-business-ethics",
            "bias_detection",
        ],
    },
    "EmbedLLM": {
        "medicine_health": [
            "medmcqa",
            "mmlu_professional_medicine",
            "mmlu_clinical_knowledge",
            "mmlu_college_medicine",
            "mmlu_virology",
            "mmlu_anatomy",
            "mmlu_medical_genetics",
            "mmlu_nutrition",
            "mmlu_human_aging",
            "mmlu_human_sexuality",
        ],
        "mathematics": [
            "mathqa",
            "asdiv",
            "gsm8k",
            "mmlu_elementary_mathematics",
            "mmlu_high_school_mathematics",
            "mmlu_high_school_statistics",
            "mmlu_college_mathematics",
            "mmlu_abstract_algebra",
        ],
        "social_sciences": [
            "social_iqa",
            "mmlu_professional_psychology",
            "mmlu_high_school_psychology",
            "mmlu_sociology",
            "mmlu_high_school_world_history",
            "mmlu_high_school_us_history",
            "mmlu_high_school_european_history",
            "mmlu_high_school_geography",
            "mmlu_high_school_government_and_politics",
            "mmlu_world_religions",
            "mmlu_prehistory",
        ],
        "science": [
            "mmlu_conceptual_physics",
            "mmlu_high_school_physics",
            "mmlu_college_physics",
            "mmlu_astronomy",
            "mmlu_electrical_engineering",
            "mmlu_high_school_chemistry",
            "mmlu_college_chemistry",
            "mmlu_high_school_biology",
            "mmlu_college_biology",
            "gpqa_extended_zeroshot",
            "gpqa_extended_n_shot",
            "gpqa_extended_cot_zeroshot",
            "gpqa_extended_cot_n_shot",
            "gpqa_extended_generative_n_shot",
            "gpqa_main_cot_n_shot",
            "gpqa_main_n_shot",
            "gpqa_main_zeroshot",
            "gpqa_main_generative_n_shot",
            "gpqa_main_cot_zeroshot",
            "gpqa_diamond_cot_zeroshot",
            "gpqa_diamond_generative_n_shot",
            "gpqa_diamond_cot_n_shot",
            "gpqa_diamond_zeroshot",
            "gpqa_diamond_n_shot",
        ],
    },
    "FusionBench": {
        "reasoning": [
            "math",
            "gsm8k",
            "arc_challenge",
            "hellaswag",
            "mbpp",
            "human_eval",
        ],
        "question_answering": [
            "gpqa",
            "trivia_qa",
            "natural_qa",
            "squad",
            "openbook_qa",
            "commonsense_qa",
            "boolq",
            "mmlu",
        ],
    },
}


class DomainShiftDatasetManagement(DatasetManagement):
    """Dataset management for domain-shift experiments.

    Builds one training split per topic domain (from DOMAIN_GROUPS), each containing
    up to TRAIN_SIZE_PER_DOMAIN samples, and a single test split pooled from the
    remaining samples of every domain (up to TEST_SIZE_PER_DOMAIN per domain).
    Domains are visited in declaration order; empty domains are skipped.
    At least two non-empty domains are required.

    DatasetDict layout:
        "train_0", "train_1", ..., "train_N" — one split per domain (≤500 each)
        "test"                                — pooled remainder (≤500 per domain)
    """

    TRAIN_SIZE_PER_DOMAIN = 500
    TEST_SIZE_PER_DOMAIN = 500
    TOTAL_SAMPLE_SIZE = 6000

    def __init__(
        self,
        seed: int,
        benchmark: Literal[
            "EmbedLLM", "RouterBench", "Sprout", "FusionBench", "R2Bench"
        ],
        embedder: EmbeddingModel | None,
        cost_penalty: float = 0.0,
    ):
        domain_key = benchmark
        if domain_key not in DOMAIN_GROUPS:
            raise ValueError(
                f"No domain groups defined for benchmark '{domain_key}'. "
                f"Available: {list(DOMAIN_GROUPS.keys())}"
            )
        self.domain_groups = DOMAIN_GROUPS[domain_key]

        split_name = benchmark
        self.ds = load_dataset("Wikit/RoutingCompendium-perf", split=split_name)
        ds_cost: Dataset = load_dataset(
            "Wikit/RoutingCompendium-cost", split=split_name
        )
        self.cost_mapping = self.get_cost_mapping(dataset_cost=ds_cost)
        self.benchmark = benchmark
        self.embedder = embedder
        self.seed = seed
        self.cost_penalty = cost_penalty
        random.seed(seed)
        np.random.seed(seed)

        self.domain_names: list[str] = []
        self.final_ds = self._get_sampled_domain_shift_dataset()

    def _embed_dataset(self, dataset: Dataset) -> Dataset:
        """Apply text cleaning and embedding to an arbitrary dataset."""
        ds_cleaned = dataset.map(  # type: ignore
            self.remove_special_characters, fn_kwargs={"prompt_variable": "prompt"}
        )
        if self.embedder is not None:
            return ds_cleaned.map(  # type: ignore
                get_embeddings,
                fn_kwargs={"embedder": self.embedder},
                batch_size=32,
                load_from_cache_file=False,
            )
        return ds_cleaned

    def _attach_costs(self, dataset: Dataset) -> Dataset:
        """Filter models by cost availability (EmbedLLM only) and add normalized cost column."""
        cost_mapping = self.clean_normalise_costs(self.cost_mapping)
        if self.benchmark == "EmbedLLM":
            dataset = self.keep_models_with_costs(dataset=dataset, cost=cost_mapping)
        models_name: list[str] = dataset["models_name"][0]  # type: ignore
        cost_mapping.cost_map = {m: cost_mapping.cost_map[m] for m in models_name}  # type: ignore
        return dataset.map(  # type: ignore
            self.add_cost,
            fn_kwargs={"cost": min_max_normalize(list(cost_mapping.cost_map.values()))},
        )

    def _get_sampled_domain_shift_dataset(self) -> DatasetDict:
        """
        Build train/test splits by filtering self.ds per domain directly,
        avoiding any prior random sampling that could exclude domain rows.
        """
        train_raws: list[Dataset] = []
        test_raws: list[Dataset] = []
        self.domain_names = []

        for domain_name, ds_list in self.domain_groups.items():
            ds_set = set(ds_list)
            domain_ds: Dataset = self.ds.filter(  # type: ignore
                lambda x, s=ds_set: x["dataset"] in s  # type: ignore
            )
            n_available = len(domain_ds)
            logger.info(
                f"Domain '{domain_name}': {n_available} samples in full dataset"
            )

            if n_available == 0:
                logger.info(f"Domain '{domain_name}' has no samples — skipping.")
                continue

            domain_ds = domain_ds.shuffle(seed=self.seed)  # type: ignore
            n_train = min(self.TRAIN_SIZE_PER_DOMAIN, n_available)
            if n_train < self.TRAIN_SIZE_PER_DOMAIN:
                logger.info(
                    f"Domain '{domain_name}' has only {n_train} samples "
                    f"(wanted {self.TRAIN_SIZE_PER_DOMAIN}) — using all."
                )

            train_raws.append(domain_ds.select(range(n_train)))  # type: ignore

            remaining = domain_ds.select(range(n_train, n_available))  # type: ignore
            n_test_domain = min(self.TEST_SIZE_PER_DOMAIN, len(remaining))
            if n_test_domain < self.TEST_SIZE_PER_DOMAIN:
                logger.info(
                    f"Domain '{domain_name}' has only {n_test_domain} samples for test "
                    f"(wanted {self.TEST_SIZE_PER_DOMAIN})."
                )
            test_raws.append(remaining.select(range(n_test_domain)))  # type: ignore
            self.domain_names.append(domain_name)

        if len(self.domain_names) < 2:
            raise ValueError(
                f"Need at least 2 non-empty domains, got {len(self.domain_names)}: "
                f"{self.domain_names}"
            )

        test_raw: Dataset = concatenate_datasets(test_raws)  # type: ignore

        combined_raw = concatenate_datasets([*train_raws, test_raw])  # type: ignore
        combined_embedded = self._embed_dataset(combined_raw)
        combined_with_costs = self._attach_costs(combined_embedded)

        n_train_total = sum(len(t) for t in train_raws)
        combined_train = combined_with_costs.select(range(n_train_total))  # type: ignore
        test_ds = combined_with_costs.select(range(n_train_total, len(combined_with_costs)))  # type: ignore

        normalized = self.normalize_performance(
            scaler=MinMaxScaler(),
            dataset_dict=DatasetDict({"train": combined_train, "test": test_ds}),
        )

        splits: dict[str, Dataset] = {"test": normalized["test"]}
        offset = 0
        for i, train_raw in enumerate(train_raws):
            n = len(train_raw)
            splits[f"train_{i}"] = normalized["train"].select(range(offset, offset + n))  # type: ignore
            offset += n

        logger.info(
            f"DomainShift dataset — "
            + ", ".join(
                f"train_{i} ({self.domain_names[i]}): {len(splits[f'train_{i}'])}"
                for i in range(len(self.domain_names))
            )
            + f", test: {len(splits['test'])}"
        )
        return DatasetDict(splits)  # type: ignore

    def get_training_generator(self) -> Generator[dict[str, Any], None, None]:
        """Yield all training items sequentially: domain_0 -> domain_1 -> … -> domain_N."""
        for i in range(len(self.domain_names)):
            for item in self.final_ds[f"train_{i}"]:  # type: ignore
                yield item  # type: ignore

    def get_training_dataset(self) -> Dataset:
        """Return all training splits concatenated."""
        return concatenate_datasets(
            [self.final_ds[f"train_{i}"] for i in range(len(self.domain_names))]  # type: ignore
        )

    def get_domain_dataset(self, i: int) -> Dataset:
        """Return the training split for domain index i."""
        return self.final_ds[f"train_{i}"]

    def get_test_dataset(self) -> Dataset:
        return self.final_ds["test"]
