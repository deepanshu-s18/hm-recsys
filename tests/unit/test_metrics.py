"""Unit tests for recommendation evaluation metrics.

Tests metric correctness with known ground truth examples
where the expected values can be hand-computed.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestRecallAtK:
    """Tests for recall@K metric."""

    def test_perfect_recall(self) -> None:
        """All relevant items are retrieved."""
        from src.evaluation.metrics import recall_at_k

        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        assert recall_at_k(recommended, relevant, k=5) == pytest.approx(1.0)

    def test_zero_recall(self) -> None:
        """No relevant items retrieved."""
        from src.evaluation.metrics import recall_at_k

        recommended = [6, 7, 8]
        relevant = {1, 2, 3}
        assert recall_at_k(recommended, relevant, k=3) == pytest.approx(0.0)

    def test_partial_recall(self) -> None:
        """1 of 3 relevant items retrieved in top-2."""
        from src.evaluation.metrics import recall_at_k

        recommended = [1, 5, 6, 7]
        relevant = {1, 2, 3}
        # 1 hit out of 3 relevant = 1/3
        assert recall_at_k(recommended, relevant, k=2) == pytest.approx(1 / 3)

    def test_empty_relevant(self) -> None:
        """Returns 0 when there are no relevant items."""
        from src.evaluation.metrics import recall_at_k

        assert recall_at_k([1, 2, 3], set(), k=3) == 0.0

    def test_k_limits_recommendations(self) -> None:
        """Only items within top-K are considered."""
        from src.evaluation.metrics import recall_at_k

        recommended = [10, 1, 2, 3]  # 1,2,3 are relevant but beyond K=1
        relevant = {1, 2, 3}
        assert recall_at_k(recommended, relevant, k=1) == pytest.approx(0.0)


class TestNDCGAtK:
    """Tests for NDCG@K metric."""

    def test_perfect_ndcg(self) -> None:
        """Relevant items at top positions → NDCG=1."""
        from src.evaluation.metrics import ndcg_at_k

        recommended = [1, 2, 3]
        relevant = {1, 2, 3}
        assert ndcg_at_k(recommended, relevant, k=3) == pytest.approx(1.0)

    def test_zero_ndcg(self) -> None:
        """No relevant items → NDCG=0."""
        from src.evaluation.metrics import ndcg_at_k

        recommended = [4, 5, 6]
        relevant = {1, 2, 3}
        assert ndcg_at_k(recommended, relevant, k=3) == pytest.approx(0.0)

    def test_ndcg_penalizes_lower_rank(self) -> None:
        """Relevant item at position 1 > position 2."""
        from src.evaluation.metrics import ndcg_at_k

        # Hit at position 1
        ndcg_early = ndcg_at_k([1, 2, 3], {1}, k=3)
        # Hit at position 2
        ndcg_late = ndcg_at_k([2, 1, 3], {1}, k=3)
        assert ndcg_early > ndcg_late

    def test_empty_relevant(self) -> None:
        """Returns 0 for empty relevant set."""
        from src.evaluation.metrics import ndcg_at_k

        assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


class TestAveragePrecision:
    """Tests for MAP@K metric."""

    def test_perfect_ap(self) -> None:
        """All items relevant → AP=1."""
        from src.evaluation.metrics import average_precision_at_k

        recommended = [1, 2, 3]
        relevant = {1, 2, 3}
        assert average_precision_at_k(recommended, relevant, k=3) == pytest.approx(1.0)

    def test_ap_known_value(self) -> None:
        """Hand-computed AP for partial overlap."""
        from src.evaluation.metrics import average_precision_at_k

        # Hits at positions 1 and 3 out of {1, 2}
        recommended = [1, 4, 2, 5]
        relevant = {1, 2}
        # P@1=1/1, P@3=2/3, AP = (1 + 2/3) / 2 = 5/6
        result = average_precision_at_k(recommended, relevant, k=4)
        assert result == pytest.approx(5 / 6, abs=1e-4)


class TestMRR:
    """Tests for Mean Reciprocal Rank."""

    def test_first_hit(self) -> None:
        """Relevant item at position 1 → RR=1."""
        from src.evaluation.metrics import reciprocal_rank

        assert reciprocal_rank([1, 2, 3], {1}) == pytest.approx(1.0)

    def test_second_hit(self) -> None:
        """Relevant item at position 2 → RR=0.5."""
        from src.evaluation.metrics import reciprocal_rank

        assert reciprocal_rank([2, 1, 3], {1}) == pytest.approx(0.5)

    def test_no_hit(self) -> None:
        """No relevant items → RR=0."""
        from src.evaluation.metrics import reciprocal_rank

        assert reciprocal_rank([4, 5, 6], {1, 2, 3}) == pytest.approx(0.0)


class TestBootstrap:
    """Tests for bootstrap confidence interval computation."""

    def test_bootstrap_ci_coverage(self) -> None:
        """Bootstrap CI should contain true mean for a simple case."""
        from src.evaluation.metrics import RecSysEvaluator

        evaluator = RecSysEvaluator(k=12, n_bootstrap=500)
        values = np.array([0.3, 0.4, 0.5, 0.35, 0.45, 0.38, 0.42])
        result = evaluator._bootstrap(values)

        assert result.ci_lower <= result.mean <= result.ci_upper
        assert result.std >= 0.0
        assert len(result.bootstrap_samples) == 500

    def test_bootstrap_consistency(self) -> None:
        """Same data with same seed → identical bootstrap results."""
        from src.evaluation.metrics import RecSysEvaluator

        evaluator = RecSysEvaluator(k=12, n_bootstrap=200)
        values = np.array([0.1, 0.2, 0.3, 0.25, 0.15])

        r1 = evaluator._bootstrap(values)
        r2 = evaluator._bootstrap(values)

        # With same seed in both calls, results should be identical
        assert r1.mean == r2.mean
        assert r1.std == pytest.approx(r2.std, abs=1e-6)


class TestPairedBootstrapTest:
    """Tests for paired bootstrap significance test."""

    def test_identical_models_not_significant(self) -> None:
        """Two identical models should not be significantly different."""
        from src.evaluation.metrics import paired_bootstrap_test

        values = np.array([0.3, 0.4, 0.5, 0.35, 0.45, 0.38, 0.42, 0.47])
        result = paired_bootstrap_test(values, values, n_bootstrap=500)

        assert result["mean_diff"] == pytest.approx(0.0)
        assert not result["significant_at_0.05"]

    def test_clearly_better_model_significant(self) -> None:
        """Model A consistently better than B should show significance."""
        from src.evaluation.metrics import paired_bootstrap_test

        rng = np.random.default_rng(42)
        a_values = rng.uniform(0.4, 0.6, size=200)
        b_values = rng.uniform(0.1, 0.3, size=200)

        result = paired_bootstrap_test(a_values, b_values, n_bootstrap=1000)

        assert result["mean_diff"] > 0.0
        assert result["significant_at_0.05"]

    def test_mismatched_lengths_raise_error(self) -> None:
        """Different-length arrays should raise AssertionError."""
        from src.evaluation.metrics import paired_bootstrap_test

        with pytest.raises(AssertionError):
            paired_bootstrap_test(
                np.array([0.1, 0.2, 0.3]),
                np.array([0.1, 0.2]),
            )
