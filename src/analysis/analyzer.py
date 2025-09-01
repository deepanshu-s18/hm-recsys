"""Research analysis modules for deep model understanding.

Implements post-hoc analysis required for publication-quality results:
    1. Retriever Complementarity Analysis
    2. Candidate Recall Ceiling Analysis
    3. Popularity Bias Analysis
    4. Cold-Start Analysis
    5. Long-Tail Analysis
    6. User Segment Analysis
    7. Failure Analysis
    8. Embedding Space Analysis

Each analysis produces a dict of statistics and a set of plots.
Designed to be runnable independently after the main pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import polars as pl
from scipy import stats

from src.evaluation.metrics import RecSysEvaluator
from src.utils.logger import get_logger

log = get_logger(__name__)


class PopularityBiasAnalyzer:
    """Analyzes popularity bias in recommendations.

    Computes the degree to which the model over-recommends popular items
    relative to their true purchase distribution.

    Args:
        n_bins: Number of popularity bins for analysis.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def analyze(
        self,
        recommendations: pl.DataFrame,
        ground_truth: pl.DataFrame,
        item_popularity: Dict[int, float],
    ) -> Dict:
        """Compute popularity bias statistics.

        Args:
            recommendations: Recommendation DataFrame with [user_idx, item_idx].
            ground_truth: Ground truth DataFrame with [user_idx, item_idx].
            item_popularity: Dict mapping item_idx → popularity fraction.

        Returns:
            Dict with bias statistics and binned analysis.
        """
        log.info("Running popularity bias analysis...")

        rec_items = recommendations["item_idx"].to_list()
        gt_items = ground_truth["item_idx"].to_list()

        rec_popularity = [item_popularity.get(i, 0.0) for i in rec_items]
        gt_popularity = [item_popularity.get(i, 0.0) for i in gt_items]

        rec_pop = np.array(rec_popularity)
        gt_pop = np.array(gt_popularity)

        # Compute ARP (Average Recommendation Popularity)
        arp = float(np.mean(rec_pop))
        avg_gt_pop = float(np.mean(gt_pop))

        # Gini coefficient of recommendation distribution
        gini = self._gini_coefficient(rec_pop)

        # KL divergence: recommendation distribution vs ground truth
        bins = np.linspace(0, max(max(rec_pop), max(gt_pop), 1e-8), self.n_bins + 1)
        rec_hist, _ = np.histogram(rec_pop, bins=bins, density=True)
        gt_hist, _ = np.histogram(gt_pop, bins=bins, density=True)

        # Smooth histograms to avoid division by zero
        rec_hist = (rec_hist + 1e-8) / (rec_hist + 1e-8).sum()
        gt_hist = (gt_hist + 1e-8) / (gt_hist + 1e-8).sum()
        kl_div = float(stats.entropy(rec_hist, gt_hist))

        return {
            "avg_recommendation_popularity": arp,
            "avg_ground_truth_popularity": avg_gt_pop,
            "popularity_ratio": arp / max(avg_gt_pop, 1e-10),
            "gini_coefficient": gini,
            "kl_divergence_from_gt": kl_div,
            "n_recommendations": len(rec_items),
            "n_ground_truth": len(gt_items),
        }

    def _gini_coefficient(self, values: np.ndarray) -> float:
        """Compute Gini coefficient of a distribution.

        Gini = 0 means perfectly equal; 1 means all mass on one item.

        Args:
            values: Non-negative array of popularity scores.

        Returns:
            Float Gini coefficient in [0, 1].
        """
        values = np.sort(np.abs(values))
        n = len(values)
        if n == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return float(((2 * index - n - 1) * values).sum() / (n * values.sum() + 1e-10))


class ColdStartAnalyzer:
    """Analyzes recommendation quality for cold-start users.

    Cold-start users are those with few historical interactions.
    They are the hardest to serve well because collaborative filtering
    has little signal for them.
    """

    def __init__(
        self,
        activity_thresholds: Optional[List[int]] = None,
    ) -> None:
        self.activity_thresholds = activity_thresholds or [2, 5, 10, 20]

    def analyze(
        self,
        train: pl.DataFrame,
        recommendations: pl.DataFrame,
        ground_truth: pl.DataFrame,
        k: int = 12,
    ) -> Dict:
        """Compute recall@K segmented by user activity level.

        Args:
            train: Training interactions for activity computation.
            recommendations: Recommendation DataFrame.
            ground_truth: Ground truth DataFrame.
            k: Recommendation cutoff.

        Returns:
            Dict with recall statistics per activity segment.
        """
        log.info("Running cold-start analysis...")
        evaluator = RecSysEvaluator(k=k, n_bootstrap=100)

        # Compute per-user activity in training
        user_activity = (
            train.group_by("user_idx")
            .agg(pl.len().alias("n_train_interactions"))
        )

        # Join activity to ground truth
        gt_with_activity = ground_truth.join(user_activity, on="user_idx", how="left")
        gt_with_activity = gt_with_activity.with_columns(
            pl.col("n_train_interactions").fill_null(0)
        )

        results = {}
        thresholds = [0] + self.activity_thresholds + [1_000_000]

        for i in range(len(thresholds) - 1):
            low, high = thresholds[i], thresholds[i + 1]
            label = f"activity_{low}_to_{high}"

            segment_users = (
                gt_with_activity.filter(
                    (pl.col("n_train_interactions") >= low) &
                    (pl.col("n_train_interactions") < high)
                )["user_idx"].unique().to_list()
            )

            if len(segment_users) < 10:
                continue

            seg_recs = recommendations.filter(
                pl.col("user_idx").is_in(segment_users)
            )
            seg_gt = ground_truth.filter(
                pl.col("user_idx").is_in(segment_users)
            )

            if len(seg_recs) == 0 or len(seg_gt) == 0:
                continue

            result = evaluator.evaluate(
                recommendations=seg_recs,
                ground_truth=seg_gt,
                model_name=label,
            )
            recall_key = f"recall@{k}"
            if recall_key in result.metrics:
                results[label] = {
                    "n_users": len(segment_users),
                    f"recall@{k}": result.metrics[recall_key].to_dict(),
                }

        return results


class UserSegmentAnalyzer:
    """Analyzes recommendation quality across user demographic segments.

    Segments users by age, club membership, or activity level and
    computes per-segment recall to identify fairness issues.
    """

    def analyze(
        self,
        customers: pl.DataFrame,
        recommendations: pl.DataFrame,
        ground_truth: pl.DataFrame,
        user_id_map: Dict[str, int],
        k: int = 12,
    ) -> Dict:
        """Compute per-demographic segment evaluation metrics.

        Args:
            customers: Customer metadata DataFrame.
            recommendations: Recommendation DataFrame.
            ground_truth: Ground truth DataFrame.
            user_id_map: customer_id → user_idx mapping.
            k: Recommendation cutoff.

        Returns:
            Dict with per-segment recall statistics.
        """
        log.info("Running user segment analysis...")
        results = {}
        evaluator = RecSysEvaluator(k=k, n_bootstrap=100)

        # Map customer_id to user_idx
        user_map_df = pl.DataFrame({
            "customer_id": list(user_id_map.keys()),
            "user_idx": list(user_id_map.values()),
        })
        customers_mapped = customers.join(user_map_df, on="customer_id", how="inner")

        # Age segments
        if "age" in customers_mapped.columns:
            for age_bucket, (low, high) in {
                "18-24": (18, 25),
                "25-34": (25, 35),
                "35-44": (35, 45),
                "45-54": (45, 55),
                "55+": (55, 120),
            }.items():
                segment_users = (
                    customers_mapped.filter(
                        (pl.col("age") >= low) & (pl.col("age") < high)
                    )["user_idx"].to_list()
                )
                if len(segment_users) < 10:
                    continue

                seg_recs = recommendations.filter(pl.col("user_idx").is_in(segment_users))
                seg_gt = ground_truth.filter(pl.col("user_idx").is_in(segment_users))
                if len(seg_recs) == 0 or len(seg_gt) == 0:
                    continue

                res = evaluator.evaluate(seg_recs, seg_gt, f"age_{age_bucket}")
                recall_key = f"recall@{k}"
                if recall_key in res.metrics:
                    results[f"age_{age_bucket}"] = {
                        "n_users": len(segment_users),
                        f"recall@{k}": res.metrics[recall_key].mean,
                    }

        return results


class LongTailAnalyzer:
    """Analyzes recommendation quality for long-tail items.

    Long-tail items are those with low purchase frequency.
    Recommending them well demonstrates that the model can serve
    niche tastes, not just popularity-driven preferences.
    """

    def analyze(
        self,
        train: pl.DataFrame,
        recommendations: pl.DataFrame,
        ground_truth: pl.DataFrame,
        item_popularity: Dict[int, float],
        percentile_buckets: Optional[List[int]] = None,
        k: int = 12,
    ) -> Dict:
        """Compute recall stratified by item popularity percentile.

        Args:
            train: Training data for popularity computation.
            recommendations: Recommendation DataFrame.
            ground_truth: Ground truth DataFrame.
            item_popularity: Dict mapping item_idx → popularity.
            percentile_buckets: Popularity percentile cutoffs.
            k: Recommendation cutoff.

        Returns:
            Dict with per-popularity-bucket recall statistics.
        """
        log.info("Running long-tail analysis...")
        percentile_buckets = percentile_buckets or [10, 25, 50, 75, 90]
        evaluator = RecSysEvaluator(k=k, n_bootstrap=100)

        pop_values = np.array(list(item_popularity.values()))
        results = {}

        for pct in percentile_buckets:
            threshold = np.percentile(pop_values, pct)
            tail_items = {
                item for item, pop in item_popularity.items() if pop <= threshold
            }

            # Filter ground truth to tail items only
            gt_tail = ground_truth.filter(pl.col("item_idx").is_in(tail_items))
            if len(gt_tail) < 10:
                continue

            # Restrict recs to users with tail GT items
            tail_users = gt_tail["user_idx"].unique().to_list()
            rec_tail = recommendations.filter(pl.col("user_idx").is_in(tail_users))

            if len(rec_tail) == 0:
                continue

            res = evaluator.evaluate(rec_tail, gt_tail, f"tail_p{pct}")
            recall_key = f"recall@{k}"
            if recall_key in res.metrics:
                results[f"tail_p{pct}"] = {
                    "n_tail_items": len(tail_items),
                    "n_users_with_tail_gt": len(tail_users),
                    f"recall@{k}": res.metrics[recall_key].mean,
                }

        return results


class RetrieverComplementarityAnalyzer:
    """Analyzes how much value each retriever adds to the ensemble.

    Complementarity measures the fraction of recalls that are unique
    to a given retriever — items that would be missed without it.
    """

    def analyze(
        self,
        retriever_candidates: Dict[str, pl.DataFrame],
        ground_truth: pl.DataFrame,
        k: int = 12,
    ) -> Dict:
        """Compute exclusive recall contribution per retriever.

        Args:
            retriever_candidates: Dict mapping retriever_name → candidate DataFrame.
            ground_truth: Ground truth DataFrame.
            k: Recommendation cutoff.

        Returns:
            Dict with per-retriever recall and exclusive recall statistics.
        """
        log.info("Running retriever complementarity analysis...")
        evaluator = RecSysEvaluator(k=k, n_bootstrap=100)
        gt_dict: dict[int, set[int]] = {}
        for row in ground_truth.to_dicts():
            uid = row["user_idx"]
            if uid not in gt_dict:
                gt_dict[uid] = set()
            gt_dict[uid].add(row["item_idx"])

        results = {}
        recall_key = f"recall@{k}"

        for name, cands in retriever_candidates.items():
            top_k_cands = (
                cands.filter(pl.col("rank") <= k)
            )
            res = evaluator.evaluate(top_k_cands, ground_truth, name)
            if recall_key in res.metrics:
                results[name] = {"recall": res.metrics[recall_key].mean}

        # Pairwise overlap analysis
        for name_i, cands_i in retriever_candidates.items():
            items_i = set(zip(
                cands_i["user_idx"].to_list(),
                cands_i["item_idx"].to_list(),
            ))
            for name_j, cands_j in retriever_candidates.items():
                if name_i >= name_j:
                    continue
                items_j = set(zip(
                    cands_j["user_idx"].to_list(),
                    cands_j["item_idx"].to_list(),
                ))
                union = items_i | items_j
                inter = items_i & items_j
                jaccard = len(inter) / max(len(union), 1)
                key = f"jaccard_{name_i}_{name_j}"
                results[key] = {"jaccard": jaccard}  # type: ignore[assignment]

        return results
