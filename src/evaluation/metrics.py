"""Recommendation evaluation metrics with bootstrap confidence intervals.

Implements all standard RecSys metrics with proper statistical rigor:
    - Every metric has 95% bootstrap CI and standard deviation
    - Supports per-user metric distributions for segment analysis
    - Includes beyond-accuracy metrics: coverage, novelty, diversity

Metric definitions:
    Recall@K: Fraction of relevant items retrieved in top-K
    Precision@K: Fraction of top-K items that are relevant
    MAP@K: Mean Average Precision at K
    MRR: Mean Reciprocal Rank of first relevant item
    NDCG@K: Normalized Discounted Cumulative Gain at K
    HitRate@K: Fraction of users with at least 1 hit in top-K
    Coverage: Fraction of catalog items ever recommended
    Novelty: -log2(popularity) of recommended items (higher = more novel)
    Diversity: Mean pairwise Jaccard distance between recommendation lists
    Personalization: 1 - mean pairwise Jaccard similarity (higher = more personalized)

All metrics return point estimates + bootstrap statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import polars as pl

from src.utils.logger import get_logger
from src.utils.timer import timer

log = get_logger(__name__)


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval result for a single metric.

    Attributes:
        mean: Point estimate of the metric.
        std: Bootstrap standard deviation.
        ci_lower: Lower bound of (1-α) confidence interval.
        ci_upper: Upper bound of (1-α) confidence interval.
        bootstrap_samples: Full array of bootstrap estimates.
        confidence_level: Confidence level used (e.g., 0.95).
    """

    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    bootstrap_samples: np.ndarray
    confidence_level: float = 0.95

    def to_dict(self) -> Dict:
        """Serialize to dictionary (excluding large bootstrap array).

        Returns:
            Dict with scalar statistics.
        """
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "confidence_level": self.confidence_level,
            "n_bootstrap": len(self.bootstrap_samples),
        }

    def __str__(self) -> str:
        return (
            f"{self.mean:.4f} ± {self.std:.4f} "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )


@dataclass
class EvaluationResult:
    """Full evaluation result container for one model.

    Attributes:
        model_name: Name of the evaluated model.
        k: Recommendation cutoff.
        metrics: Dict mapping metric_name → BootstrapResult.
        per_user_metrics: Optional DataFrame with per-user metric values.
        runtime: Dict with timing information.
    """

    model_name: str
    k: int
    metrics: Dict[str, BootstrapResult] = field(default_factory=dict)
    per_user_metrics: Optional[pl.DataFrame] = None
    runtime: Dict[str, float] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict:
        """Produce a serializable summary of all metrics.

        Returns:
            Nested dict: {metric_name: {mean, std, ci_lower, ci_upper}}.
        """
        return {
            "model_name": self.model_name,
            "k": self.k,
            "metrics": {name: r.to_dict() for name, r in self.metrics.items()},
            "runtime": self.runtime,
        }

    def print_summary(self) -> None:
        """Print a formatted metric table to stdout."""
        print(f"\n{'='*60}")
        print(f"Model: {self.model_name} | K={self.k}")
        print(f"{'='*60}")
        print(f"{'Metric':<25} {'Mean':>8} {'±Std':>8} {'95% CI':>20}")
        print(f"{'-'*61}")
        for name, result in self.metrics.items():
            ci = f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
            print(f"{name:<25} {result.mean:>8.4f} {result.std:>8.4f} {ci:>20}")
        print(f"{'='*60}\n")


def recall_at_k(
    recommended: List[int],
    relevant: Set[int],
    k: int,
) -> float:
    """Compute Recall@K for a single user.

    Recall@K = |hits in top-K| / |all relevant items|

    Args:
        recommended: Ordered list of recommended item indices.
        relevant: Set of ground-truth relevant items.
        k: Cutoff position.

    Returns:
        Float in [0, 1]. Returns 0.0 if no relevant items.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def precision_at_k(
    recommended: List[int],
    relevant: Set[int],
    k: int,
) -> float:
    """Compute Precision@K for a single user.

    Precision@K = |hits in top-K| / K

    Args:
        recommended: Ordered list of recommended item indices.
        relevant: Set of ground-truth relevant items.
        k: Cutoff position.

    Returns:
        Float in [0, 1].
    """
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def average_precision_at_k(
    recommended: List[int],
    relevant: Set[int],
    k: int,
) -> float:
    """Compute Average Precision@K for a single user.

    AP@K = (1/|R|) * sum_{i=1}^{K} P(i) * rel(i)
    where rel(i)=1 if position i is a hit.

    Args:
        recommended: Ordered list of recommended item indices.
        relevant: Set of ground-truth relevant items.
        k: Cutoff position.

    Returns:
        Float in [0, 1]. Returns 0.0 if no relevant items.
    """
    if not relevant:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / min(len(relevant), k)


def reciprocal_rank(
    recommended: List[int],
    relevant: Set[int],
) -> float:
    """Compute Reciprocal Rank for a single user.

    RR = 1 / rank_of_first_hit (0 if no hit)

    Args:
        recommended: Ordered list of recommended item indices.
        relevant: Set of ground-truth relevant items.

    Returns:
        Float in [0, 1].
    """
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    recommended: List[int],
    relevant: Set[int],
    k: int,
) -> float:
    """Compute NDCG@K for a single user.

    DCG@K = sum_{i=1}^{K} (2^rel(i) - 1) / log2(i+1)
    NDCG@K = DCG@K / IDCG@K

    Binary relevance: rel(i) ∈ {0, 1}.

    Args:
        recommended: Ordered list of recommended item indices.
        relevant: Set of ground-truth relevant items.
        k: Cutoff position.

    Returns:
        Float in [0, 1]. Returns 0.0 if no relevant items.
    """
    if not relevant:
        return 0.0

    # DCG
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )

    # IDCG (ideal: relevant items at top positions)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(
    recommended: List[int],
    relevant: Set[int],
    k: int,
) -> float:
    """Compute HitRate@K for a single user.

    HitRate@K = 1 if any top-K item is relevant, else 0.

    Args:
        recommended: Ordered list of recommended item indices.
        relevant: Set of ground-truth relevant items.
        k: Cutoff position.

    Returns:
        Float: 1.0 or 0.0.
    """
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def novelty(
    recommended: List[int],
    item_popularity: Dict[int, float],
    k: int,
) -> float:
    """Compute mean self-information novelty of top-K recommendations.

    Novelty = mean(-log2(popularity(i))) for i in recommended[:K]
    Higher novelty means recommending less popular (more surprising) items.

    Args:
        recommended: Ordered list of recommended item indices.
        item_popularity: Dict mapping item_idx → popularity fraction.
        k: Cutoff position.

    Returns:
        Float novelty score.
    """
    scores = [
        -np.log2(item_popularity.get(item, 1e-10) + 1e-10)
        for item in recommended[:k]
    ]
    return float(np.mean(scores)) if scores else 0.0


class RecSysEvaluator:
    """Comprehensive evaluator with bootstrap confidence intervals.

    Evaluates recommendations against ground truth using all standard
    RecSys metrics. Bootstrap resampling is applied over users.

    Args:
        k: Recommendation cutoff (default=12 for H&M).
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence interval level (default=0.95).
        n_jobs: Parallel workers for bootstrap (set to 1 to avoid issues).

    Example:
        >>> evaluator = RecSysEvaluator(k=12)
        >>> result = evaluator.evaluate(recommendations, ground_truth, "my_model")
    """

    def __init__(
        self,
        k: int = 12,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        n_jobs: int = 1,
    ) -> None:
        self.k = k
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.n_jobs = n_jobs

    def evaluate(
        self,
        recommendations: pl.DataFrame,
        ground_truth: pl.DataFrame,
        model_name: str,
        item_popularity: Optional[Dict[int, float]] = None,
        all_items: Optional[Set[int]] = None,
    ) -> EvaluationResult:
        """Run full evaluation suite on recommendation outputs.

        Args:
            recommendations: DataFrame with [user_idx, item_idx, rank].
                Expected: top-K ranked items per user.
            ground_truth: DataFrame with [user_idx, item_idx].
                Expected: actual purchased items in the evaluation period.
            model_name: Label for this evaluation run.
            item_popularity: Optional dict for novelty computation.
            all_items: Full catalog item set for coverage computation.

        Returns:
            EvaluationResult with all metrics and bootstrap statistics.
        """
        with timer(f"Evaluation.{model_name}"):
            # Build per-user structures
            recs_dict = self._build_recs_dict(recommendations)
            gt_dict = self._build_gt_dict(ground_truth)

            # Evaluate all users with ground truth to avoid selection bias on model dropouts
            eval_users = sorted(gt_dict.keys())
            if not eval_users:
                log.warning("No ground truth users found for evaluation!")
                return EvaluationResult(model_name=model_name, k=self.k)

            log.info(
                f"Evaluating {model_name} on {len(eval_users):,} ground-truth users "
                f"({len(recs_dict):,} users received recommendations)"
            )

            # Compute per-user metrics
            per_user = self._compute_per_user_metrics(
                eval_users, recs_dict, gt_dict, item_popularity
            )

            # Bootstrap confidence intervals
            metrics = {}
            metric_cols = [
                f"recall@{self.k}", f"precision@{self.k}", f"map@{self.k}",
                "mrr", f"ndcg@{self.k}", f"hit_rate@{self.k}",
            ]
            if item_popularity:
                metric_cols.append(f"novelty@{self.k}")

            for col in metric_cols:
                if col in per_user.columns:
                    values = per_user[col].to_numpy()
                    metrics[col] = self._bootstrap(values)

            # Beyond-accuracy metrics (computed once, not bootstrapped per user)
            if all_items:
                metrics[f"coverage@{self.k}"] = self._compute_coverage(
                    recommendations, all_items
                )
                metrics[f"catalog_coverage@{self.k}"] = metrics[f"coverage@{self.k}"]

            if item_popularity:
                metrics[f"popularity_bias@{self.k}"] = self._compute_popularity_bias(
                    recommendations, item_popularity
                )

            metrics["diversity"] = self._compute_diversity(recommendations)
            metrics["personalization"] = self._compute_personalization(recommendations)

            # Long-tail and cold-start recall
            if item_popularity:
                metrics[f"long_tail_recall@{self.k}"] = self._compute_long_tail_recall(
                    eval_users, recs_dict, gt_dict, item_popularity
                )

        result = EvaluationResult(
            model_name=model_name,
            k=self.k,
            metrics=metrics,
            per_user_metrics=per_user,
        )
        result.print_summary()
        return result

    def _build_recs_dict(
        self, recommendations: pl.DataFrame
    ) -> Dict[int, List[int]]:
        """Build {user_idx: [item_idx, ...]} from recommendations DataFrame.

        Items are ordered by rank (ascending) to preserve ranking order.

        Args:
            recommendations: DataFrame with user_idx, item_idx, rank.

        Returns:
            Dict mapping user_idx → ordered list of recommended item indices.
        """
        sorted_recs = recommendations.sort(["user_idx", "rank"])
        return (
            sorted_recs.group_by("user_idx", maintain_order=True)
            .agg(pl.col("item_idx").alias("items"))
            .to_pandas()
            .set_index("user_idx")["items"]
            .apply(list)
            .to_dict()
        )

    def _build_gt_dict(
        self, ground_truth: pl.DataFrame
    ) -> Dict[int, Set[int]]:
        """Build {user_idx: {item_idx, ...}} from ground truth DataFrame.

        Args:
            ground_truth: DataFrame with user_idx, item_idx.

        Returns:
            Dict mapping user_idx → set of ground truth item indices.
        """
        return (
            ground_truth.group_by("user_idx")
            .agg(pl.col("item_idx").unique().alias("items"))
            .to_pandas()
            .set_index("user_idx")["items"]
            .apply(set)
            .to_dict()
        )

    def _compute_per_user_metrics(
        self,
        eval_users: List[int],
        recs_dict: Dict[int, List[int]],
        gt_dict: Dict[int, Set[int]],
        item_popularity: Optional[Dict[int, float]],
    ) -> pl.DataFrame:
        """Compute all metrics for each individual user.

        Args:
            eval_users: Users to evaluate.
            recs_dict: User → recommended items.
            gt_dict: User → ground truth items.
            item_popularity: Optional popularity dict for novelty.

        Returns:
            DataFrame with one row per user and one column per metric.
        """
        records = []
        k = self.k

        for user_idx in eval_users:
            recs = recs_dict.get(user_idx, [])
            gt = gt_dict.get(user_idx, set())

            row: Dict = {"user_idx": user_idx, "n_relevant": len(gt)}
            row[f"recall@{k}"] = recall_at_k(recs, gt, k)
            row[f"precision@{k}"] = precision_at_k(recs, gt, k)
            row[f"map@{k}"] = average_precision_at_k(recs, gt, k)
            row["mrr"] = reciprocal_rank(recs, gt)
            row[f"ndcg@{k}"] = ndcg_at_k(recs, gt, k)
            row[f"hit_rate@{k}"] = hit_rate_at_k(recs, gt, k)

            if item_popularity:
                row[f"novelty@{k}"] = novelty(recs, item_popularity, k)

            records.append(row)

        return pl.DataFrame(records)

    def _bootstrap(
        self,
        values: np.ndarray,
    ) -> BootstrapResult:
        """Compute bootstrap confidence interval for a metric.

        Uses basic percentile bootstrap (non-parametric, no normality assumption).

        Args:
            values: Per-user metric values.

        Returns:
            BootstrapResult with mean, std, CI.
        """
        rng = np.random.default_rng(42)
        n = len(values)
        point_estimate = float(np.mean(values))

        boot_means = np.array([
            np.mean(rng.choice(values, size=n, replace=True))
            for _ in range(self.n_bootstrap)
        ])

        alpha = 1.0 - self.confidence_level
        return BootstrapResult(
            mean=point_estimate,
            std=float(np.std(boot_means)),
            ci_lower=float(np.percentile(boot_means, 100 * alpha / 2)),
            ci_upper=float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
            bootstrap_samples=boot_means,
            confidence_level=self.confidence_level,
        )

    def _compute_coverage(
        self,
        recommendations: pl.DataFrame,
        all_items: Set[int],
    ) -> BootstrapResult:
        """Compute catalog coverage as fraction of items ever recommended.

        Args:
            recommendations: Recommendation DataFrame.
            all_items: Full catalog item set.

        Returns:
            BootstrapResult (coverage is a scalar, not per-user).
        """
        recommended_items = set(recommendations["item_idx"].unique().to_list())
        coverage_val = len(recommended_items) / max(len(all_items), 1)

        return BootstrapResult(
            mean=coverage_val,
            std=0.0,
            ci_lower=coverage_val,
            ci_upper=coverage_val,
            bootstrap_samples=np.array([coverage_val]),
        )

    def _compute_popularity_bias(
        self,
        recommendations: pl.DataFrame,
        item_popularity: Dict[int, float],
    ) -> BootstrapResult:
        """Compute mean popularity of recommended items.

        High popularity bias means the model over-recommends popular items.

        Args:
            recommendations: Recommendation DataFrame.
            item_popularity: Dict mapping item → popularity fraction.

        Returns:
            BootstrapResult.
        """
        pop_scores = np.array([
            item_popularity.get(iid, 0.0)
            for iid in recommendations["item_idx"].to_list()
        ])
        return self._bootstrap(pop_scores)

    def _compute_diversity(
        self,
        recommendations: pl.DataFrame,
        sample_users: int = 1000,
    ) -> BootstrapResult:
        """Compute intra-list diversity via mean pairwise Jaccard distance.

        For each user, computes pairwise Jaccard distances between their
        recommendation lists (treated as item sets). Low diversity = same
        items keep appearing across positions.

        Args:
            recommendations: Recommendation DataFrame.
            sample_users: Number of users to sample for efficiency.

        Returns:
            BootstrapResult.
        """
        user_recs = (
            recommendations.group_by("user_idx")
            .agg(pl.col("item_idx").alias("items"))
            .to_pandas()
            .set_index("user_idx")["items"]
            .apply(set)
        )

        if len(user_recs) > sample_users:
            user_recs = user_recs.sample(n=sample_users, random_state=42)

        # For diversity: compare recommendation sets across users
        users = list(user_recs.index)
        n = len(users)
        if n < 2:
            return BootstrapResult(mean=0.0, std=0.0, ci_lower=0.0,
                                   ci_upper=0.0, bootstrap_samples=np.array([0.0]))

        pairwise = []
        sample_pairs = min(5000, n * (n - 1) // 2)
        rng = np.random.default_rng(42)
        pairs = rng.choice(n, size=(sample_pairs, 2), replace=True)

        for i, j in pairs:
            if i == j:
                continue
            s_i = user_recs.iloc[i]
            s_j = user_recs.iloc[j]
            union = len(s_i | s_j)
            if union > 0:
                pairwise.append(1.0 - len(s_i & s_j) / union)  # Jaccard distance

        return self._bootstrap(np.array(pairwise)) if pairwise else BootstrapResult(
            mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0,
            bootstrap_samples=np.array([0.0])
        )

    def _compute_personalization(
        self,
        recommendations: pl.DataFrame,
        sample_pairs: int = 5000,
    ) -> BootstrapResult:
        """Compute personalization as 1 - mean Jaccard similarity between user lists.

        High personalization means users receive different recommendations.
        Low personalization = popularity bias (everyone gets the same list).

        Args:
            recommendations: Recommendation DataFrame.
            sample_pairs: Number of user pairs to sample.

        Returns:
            BootstrapResult.
        """
        user_recs = (
            recommendations.group_by("user_idx")
            .agg(pl.col("item_idx").alias("items"))
            .to_pandas()
            .set_index("user_idx")["items"]
            .apply(set)
        )

        users = list(user_recs.index)
        n = len(users)
        if n < 2:
            return BootstrapResult(mean=0.0, std=0.0, ci_lower=0.0,
                                   ci_upper=0.0, bootstrap_samples=np.array([0.0]))

        rng = np.random.default_rng(42)
        # replace=True avoids ValueError when n_pairs exceeds n*(n-1)//2
        n_pairs = min(sample_pairs, n * (n - 1) // 2)
        # Use indices [0..n-1] per pair column to avoid out-of-bounds
        i_idx = rng.choice(n, size=n_pairs, replace=True)
        j_idx = rng.choice(n, size=n_pairs, replace=True)

        similarities = []
        for i, j in zip(i_idx, j_idx):
            if i == j:
                continue
            s_i = user_recs.iloc[i]
            s_j = user_recs.iloc[j]
            union = len(s_i | s_j)
            if union > 0:
                similarities.append(len(s_i & s_j) / union)

        personalization = 1.0 - np.mean(similarities) if similarities else 0.0
        return BootstrapResult(
            mean=float(personalization),
            std=0.0,
            ci_lower=float(personalization),
            ci_upper=float(personalization),
            bootstrap_samples=np.array([personalization]),
        )

    def _compute_long_tail_recall(
        self,
        eval_users: List[int],
        recs_dict: Dict[int, List[int]],
        gt_dict: Dict[int, Set[int]],
        item_popularity: Dict[int, float],
        tail_threshold: float = 0.2,
    ) -> BootstrapResult:
        """Compute recall restricted to long-tail (niche) items.

        Long-tail items are defined as those in the bottom X% by popularity.
        This metric captures whether the model serves niche taste preferences.

        Args:
            eval_users: User list.
            recs_dict: Recommendations dict.
            gt_dict: Ground truth dict.
            item_popularity: Popularity scores.
            tail_threshold: Popularity percentile cutoff for "long tail".

        Returns:
            BootstrapResult for long-tail recall.
        """
        threshold = np.percentile(
            list(item_popularity.values()), tail_threshold * 100
        )
        tail_items = {item for item, pop in item_popularity.items() if pop <= threshold}

        recalls = []
        for user_idx in eval_users:
            gt = gt_dict.get(user_idx, set()) & tail_items
            if not gt:
                continue
            recs = recs_dict.get(user_idx, [])
            rec_set = set(recs[: self.k])
            recalls.append(len(rec_set & gt) / len(gt))

        return (
            self._bootstrap(np.array(recalls))
            if recalls
            else BootstrapResult(
                mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0,
                bootstrap_samples=np.array([0.0])
            )
        )


def paired_bootstrap_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap significance test between two models.

    Tests H0: E[A] = E[B] using bootstrap resampling of the difference.
    Preferred over t-test because it makes no normality assumption.

    Args:
        values_a: Per-user metric values for model A.
        values_b: Per-user metric values for model B.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Dict with: mean_a, mean_b, mean_diff, p_value, significant.
    """
    assert len(values_a) == len(values_b), "Must have same number of users"
    rng = np.random.default_rng(seed)
    n = len(values_a)

    observed_diff = np.mean(values_a) - np.mean(values_b)

    # Bootstrap distribution of difference under H0
    boot_diffs = []
    boot_diffs_list: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs_list.append(float(np.mean(values_a[idx]) - np.mean(values_b[idx])))

    boot_diffs = np.array(boot_diffs_list)

    # Two-sided p-value
    p_value = float(np.mean(np.abs(boot_diffs - np.mean(boot_diffs)) >= abs(observed_diff)))

    return {
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
        "mean_diff": float(observed_diff),
        "p_value": p_value,
        "significant_at_0.05": bool(p_value < 0.05),
        "significant_at_0.01": bool(p_value < 0.01),
    }
