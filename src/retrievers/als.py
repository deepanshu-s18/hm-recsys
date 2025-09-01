"""ALS (Alternating Least Squares) implicit feedback retriever.

Implements collaborative filtering via matrix factorization on implicit
feedback (purchases). Uses the `implicit` library which provides a
highly optimized ALS implementation with Cython/BLAS acceleration.

The confidence matrix C[u,i] = 1 + α * r[u,i] follows Hu et al. (2008),
where r[u,i] is the interaction count (here: 1 for purchase observed).
Higher α means more confidence in observed interactions vs unobserved.

Key design decisions:
    - ALS over SVD: ALS handles sparse implicit data without explicit negatives
    - FAISS index for fast approximate nearest-neighbor retrieval
    - Users/items represented as dense embedding vectors for downstream features

Reference: Hu, Koren, Volinsky. "Collaborative Filtering for Implicit
Feedback Datasets." ICDM 2008.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import implicit
import numpy as np
import polars as pl
import scipy.sparse as sp

from src.retrievers.base import BaseRetriever, RetrievalResult
from src.utils.logger import get_logger
from src.utils.timer import timer

log = get_logger(__name__)


class ALSRetriever(BaseRetriever):
    """Implicit ALS collaborative filtering retriever.

    Factorizes the implicit feedback matrix into user and item
    embedding matrices, then uses FAISS MIPS (Maximum Inner Product
    Search) for efficient top-K retrieval.

    Args:
        factors: Embedding dimensionality. Higher = more expressive but slower.
        regularization: L2 regularization strength. Controls overfitting.
        iterations: Number of ALS optimization iterations.
        alpha: Confidence weighting parameter (see Hu et al. 2008).
        top_k: Number of candidates to retrieve per user.
        num_threads: CPU threads for parallel ALS updates.
        seed: Random seed.

    Example:
        >>> retriever = ALSRetriever(factors=128, iterations=30)
        >>> retriever.fit(train_df, n_users=5000, n_items=20000)
        >>> candidates = retriever.get_candidates([0, 1, 2])
    """

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.01,
        iterations: int = 30,
        alpha: float = 40.0,
        top_k: int = 100,
        num_threads: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__(name="als", top_k=top_k, seed=seed)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.num_threads = num_threads

        self._model: Optional[implicit.als.AlternatingLeastSquares] = None
        self._user_factors: Optional[np.ndarray] = None
        self._item_factors: Optional[np.ndarray] = None
        self._faiss_index: Optional[faiss.Index] = None

    def fit(
        self,
        train: pl.DataFrame,
        n_users: int,
        n_items: int,
    ) -> ALSRetriever:
        """Build sparse interaction matrix and train ALS model.

        Constructs the user-item confidence matrix and runs ALS
        to decompose it into latent factor matrices.

        Args:
            train: Training interactions [user_idx, item_idx, t_dat].
            n_users: Total number of users.
            n_items: Total number of items.

        Returns:
            self (fitted).
        """
        self._n_users = n_users
        self._n_items = n_items

        with timer("ALSRetriever.build_matrix"):
            interaction_matrix = self._build_sparse_matrix(train, n_users, n_items)

        with timer("ALSRetriever.fit"):
            self._model = implicit.als.AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                alpha=self.alpha,
                num_threads=self.num_threads,
                random_state=self.seed,
                use_gpu=False,
            )
            # ALS expects item x user sparse matrix for training
            self._model.fit(interaction_matrix)

        self._user_factors = self._model.user_factors.astype(np.float32)
        self._item_factors = self._model.item_factors.astype(np.float32)

        with timer("ALSRetriever.build_faiss_index"):
            self._build_faiss_index()

        self.is_fitted = True
        log.info(
            f"ALS fitted: factors={self.factors}, "
            f"user_matrix={self._user_factors.shape}, "
            f"item_matrix={self._item_factors.shape}"
        )
        return self

    def _build_sparse_matrix(
        self, train: pl.DataFrame, n_users: int, n_items: int
    ) -> sp.csr_matrix:
        """Build user × item implicit feedback sparse matrix.

        Interaction strength = purchase count (summing duplicates).
        The matrix is later converted to a confidence matrix inside ALS.

        Args:
            train: Transaction DataFrame.
            n_users: Matrix row dimension.
            n_items: Matrix column dimension.

        Returns:
            Sparse CSR matrix of shape (n_users, n_items).
        """
        # Aggregate: multiple purchases of same item → count
        agg = (
            train.group_by(["user_idx", "item_idx"])
            .agg(pl.len().alias("count"))
        )

        rows = agg["user_idx"].to_numpy().astype(np.int32)
        cols = agg["item_idx"].to_numpy().astype(np.int32)
        data = agg["count"].to_numpy().astype(np.float32)

        matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )
        log.info(
            f"Sparse matrix: {matrix.shape}, "
            f"nnz={matrix.nnz:,}, "
            f"density={matrix.nnz / (n_users * n_items):.6f}"
        )
        return matrix

    def _build_faiss_index(self) -> None:
        """Build FAISS flat inner product index for exact MIPS retrieval.

        Preserves item factor norms to accurately reflect ALS confidence/popularity.
        """
        item_factors = np.ascontiguousarray(self._item_factors, dtype=np.float32)
        self._faiss_index = faiss.IndexFlatIP(self.factors)
        self._faiss_index.add(item_factors)
        log.info(f"FAISS MIPS index built with {self._faiss_index.ntotal:,} items")

    def get_candidates(
        self,
        user_indices: List[int],
        exclude_seen: bool = True,
        seen_items: Optional[Dict[int, List[int]]] = None,
    ) -> pl.DataFrame:
        """Retrieve top-K candidates via ALS user-item inner product (MIPS).

        Args:
            user_indices: List of user integer indices.
            exclude_seen: Whether to remove already-purchased items.
            seen_items: Dict mapping user_idx → list of seen item indices.

        Returns:
            DataFrame with [user_idx, item_idx, score, rank, retriever_name].
        """
        self._check_fitted()
        seen_items = seen_items or {}

        results: List[RetrievalResult] = []
        valid_uids = []

        for uid in user_indices:
            if 0 <= uid < len(self._user_factors):
                valid_uids.append(uid)
            else:
                results.append(
                    RetrievalResult(
                        user_idx=uid,
                        item_indices=[],
                        scores=[],
                        retriever_name=self.name,
                    )
                )

        if valid_uids:
            user_vecs = np.ascontiguousarray(self._user_factors[valid_uids], dtype=np.float32)
            n_retrieve = min(self.top_k * 2, self._n_items)
            scores_batch, indices_batch = self._faiss_index.search(user_vecs, n_retrieve)

            for i, user_idx in enumerate(valid_uids):
                seen = set(seen_items.get(user_idx, []))
                candidates = []
                cand_scores = []

                for item_idx, score in zip(
                    indices_batch[i].tolist(), scores_batch[i].tolist()
                ):
                    if item_idx < 0 or item_idx >= self._n_items:
                        continue
                    if exclude_seen and item_idx in seen:
                        continue
                    candidates.append(item_idx)
                    cand_scores.append(float(score))
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

    def get_user_embeddings(self) -> np.ndarray:
        """Return user embedding matrix for downstream feature engineering.

        Returns:
            Array of shape (n_users, factors).

        Raises:
            RuntimeError: If model not fitted.
        """
        self._check_fitted()
        return self._user_factors

    def get_item_embeddings(self) -> np.ndarray:
        """Return item embedding matrix for downstream feature engineering.

        Returns:
            Array of shape (n_items, factors).

        Raises:
            RuntimeError: If model not fitted.
        """
        self._check_fitted()
        return self._item_factors

    def save(self, path: Path) -> None:
        """Save ALS model, embeddings, and FAISS index to disk.

        Args:
            path: Target directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "user_factors.npy", self._user_factors)
        np.save(path / "item_factors.npy", self._item_factors)

        faiss.write_index(self._faiss_index, str(path / "faiss.index"))

        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self._model, f, protocol=4)

        config = {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "top_k": self.top_k,
            "n_users": self._n_users,
            "n_items": self._n_items,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        log.info(f"ALS model saved to {path}")

    def load(self, path: Path) -> ALSRetriever:
        """Load ALS model from disk.

        Args:
            path: Directory containing saved artifacts.

        Returns:
            self with loaded state.
        """
        path = Path(path)
        self._user_factors = np.load(path / "user_factors.npy")
        self._item_factors = np.load(path / "item_factors.npy")
        self._faiss_index = faiss.read_index(str(path / "faiss.index"))

        with open(path / "model.pkl", "rb") as f:
            self._model = pickle.load(f)

        with open(path / "config.json") as f:
            config = json.load(f)
        self._n_users = config["n_users"]
        self._n_items = config["n_items"]
        self.is_fitted = True

        log.info(f"ALS model loaded from {path}")
        return self
