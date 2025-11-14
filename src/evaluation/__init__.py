"""Evaluation module for recommendation system metrics.

Provides:
    - RecommendationMetrics: 13-metric evaluation with bootstrap CIs
    - build_ranking_labels: Ground-truth label construction from interactions
    - build_ground_truth: Positive item set construction per user
"""

from src.evaluation.labels import build_ground_truth, build_ranking_labels
from src.evaluation.metrics import RecommendationMetrics

__all__ = [
    "RecommendationMetrics",
    "build_ranking_labels",
    "build_ground_truth",
]
