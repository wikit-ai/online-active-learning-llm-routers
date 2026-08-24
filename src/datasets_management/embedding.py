from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol, runtime_checkable

from datasets.formatting.formatting import LazyRow  # type: ignore
from model2vec import StaticModel
import numpy as np
from numpy.typing import NDArray

from chromacache import ChromaCache  # type: ignore
from chromacache.embedding_functions import (  # type: ignore
    OpenAIEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)

from logging_config import logger

# Maximum character length for prompt truncation when encoding fails
MAX_PROMPT_LENGTH = 20000


def get_embeddings(item: LazyRow, embedder: "EmbeddingModel") -> LazyRow:
    """
    Generate embeddings for prompts using the provided embedding model.

    Attempts three strategies in order:
    1. Direct encoding of prompts
    2. Encoding with truncated prompts (if prompts are too long)
    3. Fallback to accessing the underlying model directly

    Args:
        item: Dataset row containing a 'prompt' field
        embedder: Embedding model to use for encoding

    Returns:
        LazyRow: The input item with added 'embeddings' field

    Raises:
        RuntimeError: If all encoding strategies fail
    """
    prompts = item["prompt"] if isinstance(item["prompt"], list) else [item["prompt"]]  # type: ignore
    embeddings = None

    if prompts:
        # Strategy 1: Direct encoding
        try:
            embeddings = embedder.encode(prompts)  # type: ignore
        except Exception:
            logger.warning("Direct encoding failed. Attempting truncation.")

            # Strategy 2: Truncate and retry
            try:
                truncated_prompts: list[str] = [str(s)[:MAX_PROMPT_LENGTH] for s in prompts]  # type: ignore[misc]
                embeddings = embedder.encode(truncated_prompts)
            except Exception:
                logger.warning(
                    "Truncated encoding failed. Attempting direct model access."
                )

                # Strategy 3: Direct model access fallback
                try:
                    embeddings = embedder.model.embedding_function.model.encode(prompts)  # type: ignore
                except Exception as e:
                    logger.error(f"All encoding strategies failed for prompts: {e}")
                    raise RuntimeError(
                        "Failed to encode prompts after all strategies"
                    ) from e

        item["embeddings"] = np.array(embeddings, dtype=np.float32)
        return item
    else:
        raise RuntimeError("No prompts in the item")


@runtime_checkable
class EmbeddingModel(Protocol):
    """
    Protocol for text embedding models.

    This defines the minimal interface that all embedding models should implement,
    allowing for flexible typing without inheritance constraints.
    """

    @abstractmethod
    def encode(self, sequences: str | list[str]) -> NDArray[np.float32]:
        """
        Generate embeddings for the given text sequence(s).

        Args:
            sequences: A single text string or list of text strings to embed

        Returns:
            Embedding vector(s) as numpy array with shape:
            - (embedding_dim,) for single string input
            - (n_sequences, embedding_dim) for list input
        """
        pass


class AbstractEmbeddingModel(ABC):
    """
    Abstract base class for text embedding models.

    This class defines a common interface for all embedding models,
    regardless of their underlying implementation (OpenAI, Fast, etc.).

    Implementations should provide at minimum:
    1. A constructor that initializes the embedding model
    2. A get_embedding method that generates embeddings for input text

    Type parameters:
        EmbeddingOutput: The specific return type of embeddings (NDArray or List[NDArray])
    """

    def __init__(self, model_name: str):
        """
        Initialize EmbeddingModel with the specified model.

        Args:
            model_name:  Name of the pretrained model to use
        """
        self.model_name = model_name

    @abstractmethod
    def encode(self, sequences: str | list[str]) -> NDArray[np.float32]:
        """
        Generate embeddings for the given text sequence(s).

        Args:
            sequences: A single text string or list of text strings to embed

        Returns:
            Embedding vector(s) with the appropriate output type for this model

        Raises:
            ValueError: If sequences is empty or contains empty strings
            RuntimeError: If embedding generation fails
        """
        pass


class StaticEmbedding(AbstractEmbeddingModel):
    def __init__(
        self,
        model_name: Literal[
            "minishlab/potion-base-32M", "minishlab/potion-multilingual-128M"
        ] = "minishlab/potion-multilingual-128M",
    ) -> None:
        """
        Initialize StaticEmbedding with the specified model.

        Args:
            model_name: Name of the pretrained model to use
        """
        self.model_name = model_name
        self.model = StaticModel.from_pretrained(model_name)

    def encode(self, sequences: str | list[str]) -> NDArray[np.float32]:  # type: ignore
        """
        Generate embeddings for the given text sequence(s).

        Args:
            sequences: A single text string or list of text strings to embed

        Returns:
            Embedding vector(s) as numpy array
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        return self.model.encode(sequences)  # type: ignore -- original function not properly types (np.ndarray)


class OpenAIEmbedding(AbstractEmbeddingModel):
    def __init__(
        self, dimensions: int | None = None, model_name: str = "text-embedding-3-large"
    ) -> None:
        """
        Initialize OpenAI embeddings with optional dimension truncation.

        Args:
            model_name: Default - "text-embedding-3-large"
            dimensions: If provided, embeddings will be truncated to this length
        """
        self.model_name = model_name
        self.model = ChromaCache(OpenAIEmbeddingFunction(model_name=model_name))
        self.dimensions = dimensions

    def encode(self, sequences: str | list[str]) -> NDArray[np.float32]:  # type: ignore
        """
        Generate embeddings using OpenAI's model.

        Args:
            sequences: Text string or list of strings to embed

        Returns:
            Embedding vector(s) as numpy array,
            optionally truncated to the specified dimensions
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        embeddings = self.model.encode(sequences)  # type: ignore -- original function not properly types (np.ndarray)
        if self.dimensions:
            return np.array([x[: self.dimensions] for x in embeddings])  # type: ignore
        return np.array(embeddings)  # type: ignore


class STEmbedding(AbstractEmbeddingModel):
    def __init__(
        self,
        model_name: Literal[
            "BAAI/bge-m3",
            "Snowflake/snowflake-arctic-embed-m-v2.0",
            "Snowflake/snowflake-arctic-embed-l-v2.0",
        ],
    ) -> None:
        """
        Initialize sentence_transormers-based embeddings.


        Args:
            model_name: Name of the pretrained model to use
            Literal[
                "BAAI/bge-m3",
                "Snowflake/snowflake-arctic-embed-m-v2.0",
                "Snowflake/snowflake-arctic-embed-l-v2.0",
                ]
        """
        self.model = ChromaCache(
            SentenceTransformerEmbeddingFunction(
                model_name=model_name,
            )
        )

    def encode(self, sequences: str | list[str]) -> NDArray[np.float32]:
        """
        Generate embeddings using OpenAI's model.

        Args:
            sequences: Text string or list of strings to embed

        Returns:
            Embedding vector(s) as numpy array,
            optionally truncated to the specified dimensions
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        return np.array(self.model.encode(sequences))  # type: ignore -- original function not properly types (np.ndarray)


def get_embedder(
    embedding_model: Literal[
        "bge-m3",
        "snowflake-arctic-embed-l-v2.0",
        "snowflake-arctic-embed-m-v2.0",
        "potion-multilingual-128M",
        "text-embedding-3-large",
    ],
) -> EmbeddingModel:
    mapping_dict: dict[str, Any] = {
        "bge-m3": lambda: STEmbedding("BAAI/bge-m3"),
        "snowflake-arctic-embed-l-v2.0": lambda: STEmbedding(
            "Snowflake/snowflake-arctic-embed-l-v2.0"
        ),
        "snowflake-arctic-embed-m-v2.0": lambda: STEmbedding(
            "Snowflake/snowflake-arctic-embed-m-v2.0"
        ),
        "potion-multilingual-128M": lambda: StaticEmbedding(
            "minishlab/potion-multilingual-128M"
        ),
        "text-embedding-3-large": lambda: OpenAIEmbedding(
            model_name="text-embedding-3-large"
        ),
    }
    return mapping_dict[embedding_model]()  # type: ignore
