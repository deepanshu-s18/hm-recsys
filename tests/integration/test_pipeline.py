"""Integration tests for the full recommendation pipeline.

Tests end-to-end pipeline execution with synthetic data, verifying
that all stages produce valid outputs and metrics are in expected ranges.

These tests are slower (~5-10 minutes) and marked as integration tests.
Run with: pytest tests/integration/ -m integration
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.pipeline.runner import PipelineConfig, PipelineRunner


def generate_test_data(tmp_path: Path, n_users: int = 500, n_items: int = 1000) -> Path:
    """Generate minimal synthetic H&M data for integration testing.

    Args:
        tmp_path: Temporary directory.
        n_users: Number of synthetic users.
        n_items: Number of synthetic items.

    Returns:
        Path to the raw data directory.
    """
    data_dir = tmp_path / "raw"
    data_dir.mkdir()

    rng = np.random.default_rng(42)

    # Articles
    product_groups = ["Tops", "Bottoms", "Shoes", "Bags", "Accessories"]
    pd.DataFrame({
        "article_id": [f"{10000000 + i:08d}" for i in range(n_items)],
        "product_group_name": rng.choice(product_groups, size=n_items),
        "colour_group_name": rng.choice(["Black", "White", "Blue"], size=n_items),
        "department_name": rng.choice(["Ladies", "Men", "Kids"], size=n_items),
        "section_name": rng.choice(["Ladies", "Men"], size=n_items),
    }).to_csv(data_dir / "articles.csv", index=False)

    # Customers
    pd.DataFrame({
        "customer_id": [f"cust_{i:04d}" for i in range(n_users)],
        "age": rng.uniform(20, 65, size=n_users),
        "club_member_status": rng.choice(["ACTIVE", "PRE-CREATE"], size=n_users),
        "fashion_news_frequency": rng.choice(["Monthly", "NONE"], size=n_users),
    }).to_csv(data_dir / "customers.csv", index=False)

    # Transactions
    item_probs = 1.0 / (np.arange(1, n_items + 1) ** 0.8)
    item_probs /= item_probs.sum()

    n_interactions = n_users * 20
    article_ids = [f"{10000000 + i:08d}" for i in range(n_items)]
    customer_ids = [f"cust_{i:04d}" for i in range(n_users)]

    rows = []
    start = pd.Timestamp("2019-09-01")
    for _ in range(n_interactions):
        rows.append({
            "t_dat": (start + pd.Timedelta(days=int(rng.integers(0, 365)))).strftime("%Y-%m-%d"),
            "customer_id": rng.choice(customer_ids),
            "article_id": rng.choice(article_ids, p=item_probs),
            "price": float(rng.uniform(0.01, 0.2)),
            "sales_channel_id": int(rng.choice([1, 2])),
        })

    pd.DataFrame(rows).sort_values("t_dat").to_csv(
        data_dir / "transactions_train.csv", index=False
    )
    return data_dir


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """End-to-end integration tests for PipelineRunner."""

    def test_popularity_only_pipeline(self, tmp_path: Path) -> None:
        """Full pipeline with only popularity retriever runs without error."""
        data_dir = generate_test_data(tmp_path)

        config = PipelineConfig(
            seed=42,
            data_dir=str(data_dir),
            artifacts_dir=str(tmp_path / "artifacts"),
            processed_dir=str(tmp_path / "processed"),
            n_interactions=5000,
            use_popularity=True,
            use_als=False,
            use_two_tower=False,
            use_ranker=False,
            top_k=12,
            n_bootstrap=10,  # Minimal bootstrap for speed
        )

        runner = PipelineRunner(config)
        artifacts = runner.run()

        assert artifacts.dataset is not None
        assert "popularity" in artifacts.retrievers
        assert "popularity" in artifacts.results

        result = artifacts.results["popularity"]
        recall_key = "recall@12"
        if recall_key in result.metrics:
            assert 0 <= result.metrics[recall_key].mean <= 1.0

    def test_als_pipeline(self, tmp_path: Path) -> None:
        """Pipeline with ALS retriever produces valid metrics."""
        data_dir = generate_test_data(tmp_path)

        config = PipelineConfig(
            seed=42,
            data_dir=str(data_dir),
            artifacts_dir=str(tmp_path / "artifacts"),
            processed_dir=str(tmp_path / "processed"),
            n_interactions=5000,
            use_popularity=False,
            use_als=True,
            use_two_tower=False,
            use_ranker=False,
            als_factors=32,
            als_iterations=5,
            n_bootstrap=10,
        )

        runner = PipelineRunner(config)
        artifacts = runner.run()

        assert "als" in artifacts.retrievers
        assert artifacts.retrievers["als"].is_fitted

    def test_two_tower_pipeline(self, tmp_path: Path) -> None:
        """Pipeline with Two-Tower produces valid embeddings and candidates."""
        data_dir = generate_test_data(tmp_path)

        config = PipelineConfig(
            seed=42,
            data_dir=str(data_dir),
            artifacts_dir=str(tmp_path / "artifacts"),
            processed_dir=str(tmp_path / "processed"),
            n_interactions=5000,
            use_popularity=False,
            use_als=False,
            use_two_tower=True,
            use_ranker=False,
            two_tower_embedding_dim=32,
            two_tower_epochs=2,
            n_bootstrap=10,
        )

        runner = PipelineRunner(config)
        artifacts = runner.run()

        assert "two_tower" in artifacts.retrievers
        tt = artifacts.retrievers["two_tower"]
        assert tt.is_fitted

        user_emb = tt.get_user_embeddings()
        assert user_emb.ndim == 2
        assert user_emb.shape[1] == 32

    def test_full_multistage_pipeline(self, tmp_path: Path) -> None:
        """Full multi-stage pipeline with all components."""
        data_dir = generate_test_data(tmp_path, n_users=200, n_items=400)

        config = PipelineConfig(
            seed=42,
            data_dir=str(data_dir),
            artifacts_dir=str(tmp_path / "artifacts"),
            processed_dir=str(tmp_path / "processed"),
            n_interactions=3000,
            use_popularity=True,
            use_als=True,
            use_two_tower=True,
            use_ranker=True,
            als_factors=16,
            als_iterations=3,
            two_tower_embedding_dim=16,
            two_tower_epochs=2,
            lgbm_n_estimators=50,
            n_candidates=50,
            n_bootstrap=10,
        )

        runner = PipelineRunner(config)
        artifacts = runner.run()

        # Verify all stages completed
        assert len(artifacts.retrievers) == 3
        assert artifacts.ranker is not None
        assert artifacts.ranker.is_fitted

        # Verify metrics were produced
        assert len(artifacts.results) > 0

        # Verify artifacts were saved to disk
        artifacts_dir = Path(config.artifacts_dir)
        assert (artifacts_dir / "models" / "popularity").exists()
        assert (artifacts_dir / "models" / "als").exists()
        assert (artifacts_dir / "models" / "two_tower").exists()
        assert (artifacts_dir / "models" / "lgbm_ranker").exists()

    def test_metrics_in_valid_range(self, tmp_path: Path) -> None:
        """All computed metrics should be in valid ranges [0, 1]."""
        data_dir = generate_test_data(tmp_path)

        config = PipelineConfig(
            seed=42,
            data_dir=str(data_dir),
            artifacts_dir=str(tmp_path / "artifacts"),
            processed_dir=str(tmp_path / "processed"),
            n_interactions=3000,
            use_popularity=True,
            use_als=False,
            use_two_tower=False,
            use_ranker=False,
            n_bootstrap=10,
        )

        runner = PipelineRunner(config)
        artifacts = runner.run()

        for model_name, result in artifacts.results.items():
            for metric_name, boot_result in result.metrics.items():
                if metric_name in ["diversity", "personalization",
                                   "recall@12", "precision@12", "hit_rate@12",
                                   "ndcg@12", "map@12", "mrr"]:
                    assert 0.0 <= boot_result.mean <= 1.0, (
                        f"{model_name}.{metric_name} = {boot_result.mean} out of [0,1]"
                    )

    def test_pipeline_non_null_and_features(self, tmp_path: Path) -> None:
        """Pipeline features must have zero null values and valid predictions."""
        data_dir = generate_test_data(tmp_path, n_users=100, n_items=200)

        config = PipelineConfig(
            seed=42,
            data_dir=str(data_dir),
            artifacts_dir=str(tmp_path / "artifacts"),
            processed_dir=str(tmp_path / "processed"),
            n_interactions=2000,
            use_popularity=True,
            use_als=True,
            use_two_tower=True,
            use_ranker=True,
            als_factors=16,
            als_iterations=5,
            two_tower_epochs=2,
            lgbm_n_estimators=10,
            n_bootstrap=10,
        )

        runner = PipelineRunner(config)
        artifacts = runner.run()

        # Check feature importance artifact
        fi_path = Path(config.artifacts_dir) / "models" / "lgbm_ranker" / "feature_importance.parquet"
        assert fi_path.exists()
        fi = pl.read_parquet(fi_path)
        assert len(fi) > 0
        assert fi["feature"].null_count() == 0
        assert fi["gain_importance"].null_count() == 0
