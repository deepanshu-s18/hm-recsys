"""Candidate Fusion: merges outputs from multiple retrievers.

In a multi-stage recommendation pipeline, the fusion layer combines
candidates from diverse retrieval strategies to maximize recall ceiling
before the ranking stage.

Strategy: Reciprocal Rank Fusion (RRF)
    RRF was chosen over score normalization because:
    1. Scores from different models are not directly comparable
    2. RRF is robust to outliers and score distribution differences
    3. Simple and effective - consistently outperforms linear combination
    4. No tuning required beyond the k parameter

    score_rrf(d) = Σ_r 1 / (k + rank_r(d))

Reference: Cormack, Clarke, Buettcher. "Reciprocal Rank Fusion Outperforms
Condorcet and Individual Rank Learning Methods." SIGIR 2009.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import polars as pl

from src.utils.logger import get_logger
from src.utils.timer import timer

log = get_logger(__name__)


class CandidateFusion:
    """Fuses multiple retriever candidate sets into a unified ranked list.

    Implements Reciprocal Rank Fusion (RRF) with optional retriever
    weighting. Automatically deduplicates candidates and computes
    complementarity statistics for analysis.

    Args:
        rrf_k: RRF smoothing constant (default=60 is standard).
        retriever_weights: Optional dict mapping retriever_name → weight.
            If provided, RRF scores are multiplied by weights.
        max_candidates: Maximum candidates to return per user.

    Example:
        >>> fusion = CandidateFusion(max_candidates=200)
        >>> merged = fusion.fuse([pop_candidates, als_candidates, tt_candidates])
    """

    def __init__(
        self,
        rrf_k: int = 60,
        retriever_weights: Optional[Dict[str, float]] = None,
        max_candidates: int = 200,
    ) -> None:
        self.rrf_k = rrf_k
        self.retriever_weights = retriever_weights or {}
        self.max_candidates = max_candidates
        self._fusion_stats: Dict = {}

    def fuse(self, candidate_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        """Merge and rank candidates from multiple retrievers.

        Algorithm:
        1. For each (user, item) pair, collect its rank from each retriever
        2. Compute RRF score: sum(weight / (k + rank)) across retrievers
        3. Sort by RRF score descending, take top max_candidates
        4. Add metadata: source retrievers, is_multi_retriever flag

        Args:
            candidate_dfs: List of DataFrames from individual retrievers,
                each with [user_idx, item_idx, score, rank, retriever_name].

        Returns:
            Merged DataFrame with [user_idx, item_idx, rrf_score,
                retriever_name, rank, n_retrievers_src, ...].
        """
        if not candidate_dfs:
            return pl.DataFrame()

        # Filter out empty DataFrames
        candidate_dfs = [df for df in candidate_dfs if len(df) > 0]
        if not candidate_dfs:
            return pl.DataFrame()

        with timer("CandidateFusion.fuse", samples=sum(len(d) for d in candidate_dfs)):
            # Combine all candidates
            all_candidates = pl.concat(candidate_dfs, how="diagonal")

            # Compute RRF scores per (user, item)
            fused = self._compute_rrf_scores(all_candidates)

            # Track retriever sources per (user, item)
            fused = self._add_source_metadata(fused, all_candidates)

        self._compute_stats(fused)
        log.info(
            f"Fused {len(all_candidates):,} candidates → "
            f"{len(fused):,} unique (user, item) pairs | "
            f"retrievers={[df['retriever_name'][0] for df in candidate_dfs if len(df) > 0]}"
        )
        return fused

    def _compute_rrf_scores(self, all_candidates: pl.DataFrame) -> pl.DataFrame:
        """Apply Reciprocal Rank Fusion scoring.

        Args:
            all_candidates: Concatenated candidate DataFrame.

        Returns:
            DataFrame with rrf_score aggregated per (user_idx, item_idx).
        """
        k = float(self.rrf_k)

        # Apply retriever weights if provided (vectorized without map_elements)
        if self.retriever_weights:
            weighted = all_candidates.with_columns(
                pl.col("retriever_name")
                .cast(pl.Utf8)
                .replace(self.retriever_weights, default=1.0)
                .cast(pl.Float64)
                .alias("retriever_weight")
            )
        else:
            weighted = all_candidates.with_columns(
                pl.lit(1.0).alias("retriever_weight")
            )

        weighted = weighted.with_columns(
            (pl.col("retriever_weight") / (k + pl.col("rank").cast(pl.Float64))).alias("rrf_contribution")
        )

        # Aggregate RRF scores per (user, item)
        fused = (
            weighted.group_by(["user_idx", "item_idx"])
            .agg(
                pl.col("rrf_contribution").sum().alias("rrf_score"),
                pl.col("retriever_name").n_unique().alias("n_retrievers_src"),
                pl.col("score").max().alias("max_retriever_score"),
                pl.col("rank").min().alias("best_rank"),
            )
            .sort(["user_idx", "rrf_score"], descending=[False, True])
        )

        # Take top max_candidates per user
        fused = (
            fused.with_columns(
                pl.col("rrf_score")
                .rank(method="ordinal", descending=True)
                .over("user_idx")
                .alias("fusion_rank")
            )
            .filter(pl.col("fusion_rank") <= self.max_candidates)
            .sort(["user_idx", "fusion_rank"])
        )

        return fused

    def _add_source_metadata(self, fused: pl.DataFrame, all_candidates: pl.DataFrame) -> pl.DataFrame:
        """Add source metadata - which retrievers found each candidate."""
        sources = (
            all_candidates
            .select(["user_idx", "item_idx", "retriever_name"])
            .group_by(["user_idx", "item_idx"])
            .agg([
                pl.col("retriever_name")
                .unique()
                .sort()
                .str.concat(",")
                .alias("retriever_sources"),
            ])
        )
        return fused.join(sources, on=["user_idx", "item_idx"], how="left", coalesce=True)

    def _compute_stats(self, fused: pl.DataFrame) -> None:
        """Compute and cache fusion statistics for analysis.

        Args:
            fused: Final fused candidate DataFrame.
        """
        total_candidates = len(fused)
        multi_source = (fused["n_retrievers_src"] > 1).sum()
        avg_candidates_per_user = (
            fused.group_by("user_idx").agg(pl.len().alias("n")).mean()["n"][0]
        )

        self._fusion_stats = {
            "total_candidates": total_candidates,
            "multi_source_ratio": float(multi_source / max(total_candidates, 1)),
            "avg_candidates_per_user": float(avg_candidates_per_user or 0),
            "max_candidates_limit": self.max_candidates,
        }
        log.info(
            f"Fusion stats: {total_candidates:,} candidates, "
            f"{self._fusion_stats['multi_source_ratio']:.2%} multi-source, "
            f"{avg_candidates_per_user:.1f} avg per user"
        )

    @property
    def fusion_stats(self) -> Dict:
        """Return last fusion run statistics.

        Returns:
            Dict with fusion statistics.
        """
        return self._fusion_stats

    def compute_retriever_overlap(
        self, candidate_dfs: List[pl.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """Compute pairwise Jaccard overlap between retrievers.

        Used to analyze retriever complementarity — low overlap
        between retrievers is desirable as it means diverse candidates.

        Args:
            candidate_dfs: List of per-retriever candidate DataFrames.

        Returns:
            Nested dict: {retriever_i: {retriever_j: jaccard_score}}.
        """
        retriever_sets: Dict[str, Dict[int, set]] = defaultdict(
            lambda: defaultdict(set)
        )

        for df in candidate_dfs:
            if len(df) == 0:
                continue
            name = df["retriever_name"][0]
            for row in df.select(["user_idx", "item_idx"]).to_dicts():
                retriever_sets[name][row["user_idx"]].add(row["item_idx"])

        names = list(retriever_sets.keys())
        overlap: Dict[str, Dict[str, float]] = {}

        for i, name_i in enumerate(names):
            overlap[name_i] = {}
            for j, name_j in enumerate(names):
                if i == j:
                    overlap[name_i][name_j] = 1.0
                    continue

                # Average pairwise Jaccard across all users
                jaccards = []
                users = set(retriever_sets[name_i].keys()) | set(
                    retriever_sets[name_j].keys()
                )
                for user in users:
                    set_i = retriever_sets[name_i].get(user, set())
                    set_j = retriever_sets[name_j].get(user, set())
                    union = set_i | set_j
                    if union:
                        jaccards.append(len(set_i & set_j) / len(union))

                import numpy as np
                overlap[name_i][name_j] = float(np.mean(jaccards)) if jaccards else 0.0

        return overlap
