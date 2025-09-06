"""Ground truth label construction for ranking and evaluation.

Builds binary relevance labels by joining candidate (user, item) pairs
against actual future purchases. Labels are used for:
    1. LightGBM LambdaMART training (purchase=1, non-purchase=0)
    2. Recall ceiling analysis (how many purchases are in candidates)
    3. Post-hoc evaluation of model-predicted rankings
"""

from __future__ import annotations

from typing import Dict

import polars as pl

from src.utils.logger import get_logger

log = get_logger(__name__)


def build_ground_truth(
    interactions: pl.DataFrame,
    min_purchases: int = 1,
) -> pl.DataFrame:
    """Extract ground truth purchases from an interaction split.

    Args:
        interactions: DataFrame with [user_idx, item_idx, t_dat].
        min_purchases: Minimum purchases for a (user, item) pair to count.

    Returns:
        DataFrame with [user_idx, item_idx] representing target purchases.
    """
    gt = (
        interactions.group_by(["user_idx", "item_idx"])
        .agg(pl.len().alias("n_purchases"))
        .filter(pl.col("n_purchases") >= min_purchases)
        .select(["user_idx", "item_idx"])
        .with_columns([
            pl.col("user_idx").cast(pl.Int64),
            pl.col("item_idx").cast(pl.Int64),
        ])
    )
    log.info(
        f"Ground truth: {len(gt):,} (user, item) pairs, "
        f"{gt['user_idx'].n_unique():,} users"
    )
    return gt


def build_ranking_labels(
    candidates: pl.DataFrame,
    ground_truth: pl.DataFrame,
) -> pl.DataFrame:
    """Assign binary relevance labels to candidate pairs.

    Joins candidate (user, item) pairs against ground truth purchases.
    A candidate gets label=1 if the user actually bought that item.

    Args:
        candidates: DataFrame with [user_idx, item_idx, features...].
        ground_truth: DataFrame with [user_idx, item_idx] (actual purchases).

    Returns:
        Candidates DataFrame with added 'label' column (0 or 1).
    """
    candidates = candidates.with_columns([
        pl.col("user_idx").cast(pl.Int64),
        pl.col("item_idx").cast(pl.Int64),
    ])
    ground_truth = ground_truth.with_columns([
        pl.col("user_idx").cast(pl.Int64),
        pl.col("item_idx").cast(pl.Int64),
    ])

    gt_with_label = ground_truth.with_columns(
        pl.lit(1).cast(pl.Int8).alias("label")
    )

    labeled = candidates.join(
        gt_with_label, on=["user_idx", "item_idx"], how="left", coalesce=True
    ).with_columns(
        pl.col("label").fill_null(0).cast(pl.Int8)
    )

    n_positive = (labeled["label"] == 1).sum()
    n_negative = (labeled["label"] == 0).sum()
    pos_rate = n_positive / max(len(labeled), 1)

    log.info(
        f"Ranking labels: {len(labeled):,} total | "
        f"+={n_positive:,} ({pos_rate:.2%}) | "
        f"-={n_negative:,}"
    )

    if n_positive == 0:
        log.warning(
            "No positive labels found! Check that candidates and ground truth "
            "are from the correct splits."
        )

    return labeled


def compute_recall_ceiling(
    candidates: pl.DataFrame,
    ground_truth: pl.DataFrame,
) -> Dict[str, float]:
    """Compute the maximum achievable recall from the candidate set.

    The recall ceiling is the fraction of ground-truth purchases that
    appear in the candidate set (before ranking). This tells us how
    well the retrieval stage is doing — the ranker cannot recover
    items missing from candidates.

    A ceiling of 0.85 means we can never exceed 85% recall regardless
    of how good our ranker is.

    Args:
        candidates: Candidate DataFrame with [user_idx, item_idx].
        ground_truth: Ground truth with [user_idx, item_idx].

    Returns:
        Dict with per-user ceiling statistics.
    """
    candidates = candidates.with_columns([
        pl.col("user_idx").cast(pl.Int64),
        pl.col("item_idx").cast(pl.Int64),
    ])
    ground_truth = ground_truth.with_columns([
        pl.col("user_idx").cast(pl.Int64),
        pl.col("item_idx").cast(pl.Int64),
    ])

    candidates_set = (
        candidates.select(["user_idx", "item_idx"])
        .with_columns(pl.lit(1).alias("in_candidates"))
    )

    gt_with_coverage = ground_truth.join(
        candidates_set, on=["user_idx", "item_idx"], how="left", coalesce=True
    ).with_columns(
        pl.col("in_candidates").fill_null(0)
    )

    per_user_ceiling = (
        gt_with_coverage.group_by("user_idx")
        .agg([
            pl.col("in_candidates").mean().alias("recall_ceiling"),
            pl.col("in_candidates").sum().alias("hits"),
            pl.len().alias("total_relevant"),
        ])
    )

    import numpy as np
    ceilings = per_user_ceiling["recall_ceiling"].to_numpy()

    stats = {
        "mean_recall_ceiling": float(np.mean(ceilings)),
        "median_recall_ceiling": float(np.median(ceilings)),
        "p10_recall_ceiling": float(np.percentile(ceilings, 10)),
        "p90_recall_ceiling": float(np.percentile(ceilings, 90)),
        "n_users": len(ceilings),
    }
    log.info(
        f"Recall ceiling: mean={stats['mean_recall_ceiling']:.4f}, "
        f"median={stats['median_recall_ceiling']:.4f}"
    )
    return stats
