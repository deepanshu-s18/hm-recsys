"""Evaluation module for recommendation system metrics.

Provides:
    - RecSysEvaluator: 13-metric evaluation with bootstrap CIs
    - BootstrapResult: Container for bootstrap confidence intervals
    - EvaluationResult: Container for point estimates and aggregated metrics
    - build_ranking_labels: Ground-truth label construction from interactions
    - build_ground_truth: Positive item set construction per user
"""

from src.evaluation.labels import build_ground_truth, build_ranking_labels
from src.evaluation.metrics import BootstrapResult, EvaluationResult, RecSysEvaluator

__all__ = [
    "RecSysEvaluator",
    "BootstrapResult",
    "EvaluationResult",
    "build_ranking_labels",
    "build_ground_truth",
]
