"""Popularity-based candidate retriever with optional temporal decay.

Popularity is the strongest non-personalized baseline in recommendation.
Despite its simplicity, it captures global purchase trends and serves as
the floor for any personalized model to surpass.

Implementation details:
    - Supports pure purchase count and time-decayed popularity
    - Temporal decay uses exponential weighting: score(t) = exp(-λ * age_days)
    - Fully vectorized via Polars for sub-millisecond inference
    - No training state beyond the sorted item ranking

Reference: Cremonesi et al. "Performance of Recommender Algorithms
on Top-N Recommendation Tasks." RecSys 2010.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from src.retrievers.base import BaseRetriever, RetrievalResult
from src.utils.logger import get_logger
from src.utils.timer import timer

log = get_logger(__name__)


class PopularityRetriever(BaseRetriever):
    """Non-personalized popularity-based candidate retriever.

    Ranks items by aggregate purchase frequency, optionally weighted
    by recency to capture current fashion trends. All users receive
    the same candidate list (minus already-seen items).

    Popularity serves two purposes in the pipeline:
    1. As a standalone baseline to quantify the lift from personalization
    2. As a retriever in the ensemble to catch globally trendy items
       that personalized models might miss (especially for cold users)

    Args:
        top_k: Number of top items to retrieve per user.
        time_decay: If True, weight purchases by recency.
        decay_factor: Exponential decay rate (λ). Larger = faster decay.
        seed: Random seed (unused for pure popularity, kept for interface).

    Example:
        >>> retriever = PopularityRetriever(top_k=100, time_decay=True)
        >>> retriever.fit(train_df, n_users=5000, n_items=20000)
        >>> candidates = retriever.get_candidates([0, 1, 2])
    """

    def __init__(
        self,
        top_k: int = 100,
        time_decay: bool = True,
        decay_factor: float = 0.95,
        seed: int = 42,
    ) -> None:
        super().__init__(name="popularity", top_k=top_k, seed=seed)
        self.time_decay = time_decay
        self.decay_factor = decay_factor

        self._item_scores: Optional[np.ndarray] = None
        self._top_items: Optional[np.ndarray] = None
        self._top_scores: Optional[np.ndarray] = None

    def fit(
        self,
        train: pl.DataFrame,
        n_users: int,
        n_items: int,
    ) -> PopularityRetriever:
        """Compute item popularity scores from training interactions.

        If time_decay=True, each purchase contributes exp(-λ * age) to
        the item score, where age is in days from the most recent date.

        Args:
            train: Training interactions with [user_idx, item_idx, t_dat].
            n_users: Total catalog users (not used, included for interface).
            n_items: Total catalog items.

        Returns:
            self (fitted).
        """
        self._n_users = n_users
        self._n_items = n_items

        with timer("PopularityRetriever.fit"):
            if self.time_decay:
                item_scores = self._compute_decayed_scores(train, n_items)
            else:
                item_scores = self._compute_count_scores(train, n_items)

        self._item_scores = item_scores

        # Pre-sort for O(1) retrieval
        sorted_indices = np.argsort(item_scores)[::-1]
        self._top_items = sorted_indices[: self.top_k].astype(np.int32)
        self._top_scores = item_scores[sorted_indices[: self.top_k]].astype(np.float32)

        self.is_fitted = True
        log.info(
            f"PopularityRetriever fitted: top item score={self._top_scores[0]:.4f}, "
            f"time_decay={self.time_decay}"
        )
        return self

    def _compute_count_scores(self, train: pl.DataFrame, n_items: int) -> np.ndarray:
        """Pure purchase count popularity.

        Args:
            train: Training transactions.
            n_items: Catalog size.

        Returns:
            Float array of shape (n_items,) with purchase counts.
        """
        item_counts = (
            train.group_by("item_idx")
            .agg(pl.len().alias("count"))
        )
        scores = np.zeros(n_items, dtype=np.float32)
        for row in item_counts.to_dicts():
            scores[row["item_idx"]] = row["count"]
        return scores

    def _compute_decayed_scores(self, train: pl.DataFrame, n_items: int) -> np.ndarray:
        """Recency-weighted popularity with exponential decay.

        Each purchase contributes: exp(-λ * age_days) to the item score.
        Recent purchases are weighted higher, capturing current trends.

        Args:
            train: Training transactions (must have t_dat column).
            n_items: Catalog size.

        Returns:
            Float array of shape (n_items,) with decayed scores.
        """
        max_date = train["t_dat"].max()

        decayed = train.with_columns(
            (
                (pl.lit(max_date).cast(pl.Date) - pl.col("t_dat"))
                .dt.total_days()
                .cast(pl.Float32)
            ).alias("age_days")
        ).with_columns(
            ((-self.decay_factor * pl.col("age_days") / 30.0).exp()).alias("weight")
        )

        item_scores = (
            decayed.group_by("item_idx")
            .agg(pl.col("weight").sum().alias("score"))
        )

        scores = np.zeros(n_items, dtype=np.float32)
        for row in item_scores.to_dicts():
            if row["item_idx"] < n_items:
                scores[row["item_idx"]] = row["score"]
        return scores

    def get_candidates(
        self,
        user_indices: List[int],
        exclude_seen: bool = True,
        seen_items: Optional[Dict[int, List[int]]] = None,
    ) -> pl.DataFrame:
        """Retrieve top-K popular items for each user.

        All users receive the same global popularity ranking,
        filtered by their personal purchase history.

        Args:
            user_indices: List of integer user indices.
            exclude_seen: If True, remove items already purchased.
            seen_items: Dict of {user_idx: [item_idx, ...]} for exclusion.

        Returns:
            DataFrame with [user_idx, item_idx, score, rank, retriever_name].
        """
        self._check_fitted()

        results: List[RetrievalResult] = []
        seen_items = seen_items or {}

        for user_idx in user_indices:
            seen = set(seen_items.get(user_idx, []))
            candidates = []
            cand_scores = []

            for item_idx, score in zip(
                self._top_items.tolist(), self._top_scores.tolist()
            ):
                if exclude_seen and item_idx in seen:
                    continue
                candidates.append(item_idx)
                cand_scores.append(score)
                if len(candidates) >= self.top_k:
                    break

            results.append(
                RetrievalResult(
                    user_idx=user_idx,
                    item_indices=candidates,
                    scores=cand_scores,
                    retriever_name=self.name,
                )
            )

        return self._candidates_to_df(results)

    def save(self, path: Path) -> None:
        """Save popularity scores and configuration to disk.

        Args:
            path: Directory to save artifacts.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "item_scores.npy", self._item_scores)
        np.save(path / "top_items.npy", self._top_items)
        np.save(path / "top_scores.npy", self._top_scores)

        config = {
            "name": self.name,
            "top_k": self.top_k,
            "time_decay": self.time_decay,
            "decay_factor": self.decay_factor,
            "n_users": self._n_users,
            "n_items": self._n_items,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        log.info(f"PopularityRetriever saved to {path}")

    def load(self, path: Path) -> PopularityRetriever:
        """Load popularity model from disk.

        Args:
            path: Directory containing saved artifacts.

        Returns:
            self with loaded state.
        """
        path = Path(path)
        self._item_scores = np.load(path / "item_scores.npy")
        self._top_items = np.load(path / "top_items.npy")
        self._top_scores = np.load(path / "top_scores.npy")

        with open(path / "config.json") as f:
            config = json.load(f)
        self._n_users = config["n_users"]
        self._n_items = config["n_items"]
        self.is_fitted = True

        log.info(f"PopularityRetriever loaded from {path}")
        return self
