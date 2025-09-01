"""Abstract base class for all candidate retrievers.

All retrieval models must implement this interface to be composable
in the multi-stage pipeline. The Protocol-style interface enforces:
    1. Fit on training data
    2. Generate top-K candidate items per user
    3. Report model size and latency characteristics

Design rationale: Having a common interface lets the pipeline treat
all retrievers uniformly, enables ensemble fusion, and makes A/B
testing trivial — just swap retrievers behind the interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Structured output from a candidate retriever.

    Attributes:
        user_idx: Integer user index.
        item_indices: Ranked list of candidate item indices (best first).
        scores: Corresponding relevance scores (higher = more relevant).
        retriever_name: Name of the retriever that produced this result.
    """

    user_idx: int
    item_indices: List[int]
    scores: List[float]
    retriever_name: str

    def __post_init__(self) -> None:
        assert len(self.item_indices) == len(self.scores), (
            f"item_indices and scores must have equal length, got "
            f"{len(self.item_indices)} vs {len(self.scores)}"
        )

    def to_dict(self) -> Dict:
        """Serialize to flat dictionary for DataFrame construction.

        Returns:
            Dictionary with scalar and list fields.
        """
        return {
            "user_idx": self.user_idx,
            "item_indices": self.item_indices,
            "scores": self.scores,
            "retriever_name": self.retriever_name,
        }


@dataclass
class RetrieverMetrics:
    """Runtime and model metrics for a retriever.

    Attributes:
        name: Retriever name.
        fit_time_sec: Wall-clock training time.
        predict_time_sec: Total inference time for all users.
        peak_memory_mb: Peak RAM during training.
        model_size_mb: Serialized model size on disk.
        n_users: Number of users in training.
        n_items: Number of items in catalog.
    """

    name: str
    fit_time_sec: float = 0.0
    predict_time_sec: float = 0.0
    peak_memory_mb: float = 0.0
    model_size_mb: float = 0.0
    n_users: int = 0
    n_items: int = 0
    extra: Dict = field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract base class for candidate retrieval models.

    All retrievers share a common fit/predict interface. Subclasses
    must implement `fit`, `get_candidates_for_users`, and `save`/`load`.

    Args:
        name: Human-readable retriever name (used for logging and artifacts).
        top_k: Maximum number of candidates to return per user.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        name: str,
        top_k: int = 100,
        seed: int = 42,
    ) -> None:
        self.name = name
        self.top_k = top_k
        self.seed = seed
        self.is_fitted: bool = False
        self.metrics: RetrieverMetrics = RetrieverMetrics(name=name)
        self._n_users: int = 0
        self._n_items: int = 0

    @abstractmethod
    def fit(
        self,
        train: pl.DataFrame,
        n_users: int,
        n_items: int,
    ) -> BaseRetriever:
        """Train the retrieval model on interaction data.

        Args:
            train: Training interactions with columns:
                [user_idx, item_idx, t_dat, price].
            n_users: Total number of users in the catalog.
            n_items: Total number of items in the catalog.

        Returns:
            self (for method chaining).
        """

    @abstractmethod
    def get_candidates(
        self,
        user_indices: List[int],
        exclude_seen: bool = True,
        seen_items: Optional[Dict[int, List[int]]] = None,
    ) -> pl.DataFrame:
        """Generate top-K candidates for a batch of users.

        Args:
            user_indices: List of user integer indices.
            exclude_seen: If True, remove already-purchased items.
            seen_items: Dict mapping user_idx → list of seen item_indices.

        Returns:
            DataFrame with columns:
                [user_idx, item_idx, score, rank, retriever_name].
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model state to disk.

        Args:
            path: Directory to save model artifacts.
        """

    @abstractmethod
    def load(self, path: Path) -> BaseRetriever:
        """Load model state from disk.

        Args:
            path: Directory containing saved model artifacts.

        Returns:
            self with loaded state.
        """

    def _build_seen_items(self, train: pl.DataFrame) -> Dict[int, List[int]]:
        """Build mapping from user → set of seen item indices.

        Used to exclude already-purchased items during inference.

        Args:
            train: Training DataFrame with user_idx and item_idx columns.

        Returns:
            Dict mapping user_idx (int) → list of item_idx (int).
        """
        return (
            train.group_by("user_idx")
            .agg(pl.col("item_idx").unique().alias("seen_items"))
            .to_pandas()
            .set_index("user_idx")["seen_items"]
            .to_dict()
        )

    def _check_fitted(self) -> None:
        """Raise if model has not been fitted yet.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.name} has not been fitted. Call fit() before get_candidates()."
            )

    def _candidates_to_df(
        self,
        results: List[RetrievalResult],
    ) -> pl.DataFrame:
        """Convert list of RetrievalResult objects to a flat DataFrame.

        Args:
            results: List of per-user retrieval results.

        Returns:
            DataFrame with columns: [user_idx, item_idx, score, rank, retriever_name].
        """
        rows = []
        for r in results:
            for rank, (item_idx, score) in enumerate(zip(r.item_indices, r.scores)):
                rows.append(
                    {
                        "user_idx": r.user_idx,
                        "item_idx": item_idx,
                        "score": float(score),
                        "rank": rank + 1,  # 1-indexed rank
                        "retriever_name": r.retriever_name,
                    }
                )

        if not rows:
            return pl.DataFrame(
                schema={
                    "user_idx": pl.Int32,
                    "item_idx": pl.Int32,
                    "score": pl.Float32,
                    "rank": pl.Int32,
                    "retriever_name": pl.Utf8,
                }
            )

        return pl.DataFrame(rows).with_columns(
            pl.col("user_idx").cast(pl.Int32),
            pl.col("item_idx").cast(pl.Int32),
            pl.col("score").cast(pl.Float32),
            pl.col("rank").cast(pl.Int32),
        )
