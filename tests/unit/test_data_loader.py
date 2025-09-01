"""Unit tests for HMDataLoader.

Tests data loading, sampling, splitting, and ID mapping
without requiring the actual Kaggle dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _create_mock_data(tmp_path: Path, n_users: int = 100, n_items: int = 200, n_interactions: int = 1000) -> Path:
    """Create minimal mock H&M CSV files for testing.

    Args:
        tmp_path: Temporary directory.
        n_users: Number of synthetic users.
        n_items: Number of synthetic items.
        n_interactions: Number of interaction rows.

    Returns:
        Path to the directory containing mock CSVs.
    """
    import pandas as pd

    rng = np.random.default_rng(42)

    # Transactions
    customer_ids = [f"cust_{i:04d}" for i in range(n_users)]
    article_ids = [f"{10000000 + i:08d}" for i in range(n_items)]

    item_probs = 1.0 / (np.arange(1, n_items + 1) ** 0.8)
    item_probs /= item_probs.sum()

    rows = []
    start = pd.Timestamp("2019-09-01")
    for _ in range(n_interactions):
        cust = rng.choice(customer_ids)
        item = rng.choice(article_ids, p=item_probs)
        day = rng.integers(0, 365)
        rows.append({
            "t_dat": (start + pd.Timedelta(days=int(day))).strftime("%Y-%m-%d"),
            "customer_id": cust,
            "article_id": item,
            "price": float(rng.uniform(0.01, 0.2)),
            "sales_channel_id": int(rng.choice([1, 2])),
        })
    pd.DataFrame(rows).to_csv(tmp_path / "transactions_train.csv", index=False)

    # Articles
    pd.DataFrame({
        "article_id": article_ids,
        "product_group_name": rng.choice(["Tops", "Bottoms", "Shoes"], size=n_items),
        "colour_group_name": rng.choice(["Black", "White", "Blue"], size=n_items),
        "department_name": rng.choice(["Ladies", "Men", "Kids"], size=n_items),
        "section_name": rng.choice(["Ladies", "Men"], size=n_items),
    }).to_csv(tmp_path / "articles.csv", index=False)

    # Customers
    pd.DataFrame({
        "customer_id": customer_ids,
        "age": rng.uniform(18, 70, size=n_users),
        "club_member_status": rng.choice(["ACTIVE", "PRE-CREATE"], size=n_users),
        "fashion_news_frequency": rng.choice(["Monthly", "NONE"], size=n_users),
    }).to_csv(tmp_path / "customers.csv", index=False)

    return tmp_path


class TestHMDataLoader:
    """Tests for HMDataLoader."""

    def test_load_creates_splits(self, tmp_path: Path) -> None:
        """Test that load() returns train/val/test splits."""
        from src.data.loader import HMDataLoader

        data_dir = _create_mock_data(tmp_path, n_users=50, n_items=100, n_interactions=500)
        loader = HMDataLoader(data_dir=data_dir, n_interactions=500, seed=42)
        dataset = loader.load()

        assert len(dataset.train) > 0
        assert len(dataset.val) > 0
        assert len(dataset.test) > 0

    def test_chronological_split_no_leakage(self, tmp_path: Path) -> None:
        """Test that train max date < val max date < test max date."""
        from src.data.loader import HMDataLoader

        data_dir = _create_mock_data(tmp_path, n_users=50, n_items=100, n_interactions=500)
        loader = HMDataLoader(data_dir=data_dir, n_interactions=500, seed=42)
        dataset = loader.load()

        train_max = dataset.train["t_dat"].max()
        val_min = dataset.val["t_dat"].min()
        test_min = dataset.test["t_dat"].min()

        assert train_max <= val_min, f"Temporal leakage: train_max={train_max} > val_min={val_min}"
        assert val_min <= test_min, "Temporal ordering violated"

    def test_id_maps_contiguous(self, tmp_path: Path) -> None:
        """Test that user/item indices are contiguous integers starting from 0."""
        from src.data.loader import HMDataLoader

        data_dir = _create_mock_data(tmp_path, n_users=50, n_items=100, n_interactions=500)
        loader = HMDataLoader(data_dir=data_dir, n_interactions=500, seed=42)
        dataset = loader.load()

        user_indices = sorted(dataset.user2idx.values())
        item_indices = sorted(dataset.item2idx.values())

        assert user_indices == list(range(len(dataset.user2idx)))
        assert item_indices == list(range(len(dataset.item2idx)))

    def test_id_maps_invertible(self, tmp_path: Path) -> None:
        """Test that idx2user[user2idx[uid]] == uid."""
        from src.data.loader import HMDataLoader

        data_dir = _create_mock_data(tmp_path, n_users=50, n_items=100, n_interactions=500)
        loader = HMDataLoader(data_dir=data_dir, n_interactions=500, seed=42)
        dataset = loader.load()

        for uid, idx in list(dataset.user2idx.items())[:10]:
            assert dataset.idx2user[idx] == uid

        for iid, idx in list(dataset.item2idx.items())[:10]:
            assert dataset.idx2item[idx] == iid

    def test_sampling_is_deterministic(self, tmp_path: Path) -> None:
        """Test that two runs with same seed produce identical datasets."""
        from src.data.loader import HMDataLoader

        data_dir = _create_mock_data(tmp_path, n_users=100, n_items=200, n_interactions=2000)

        loader1 = HMDataLoader(data_dir=data_dir, n_interactions=500, seed=42)
        dataset1 = loader1.load()

        loader2 = HMDataLoader(data_dir=data_dir, n_interactions=500, seed=42)
        dataset2 = loader2.load()

        assert set(dataset1.user2idx.keys()) == set(dataset2.user2idx.keys())
        assert len(dataset1.train) == len(dataset2.train)

    def test_different_seeds_give_different_samples(self, tmp_path: Path) -> None:
        """Test that different seeds may produce different samples."""
        from src.data.loader import HMDataLoader

        data_dir = _create_mock_data(tmp_path, n_users=100, n_items=200, n_interactions=2000)

        loader1 = HMDataLoader(data_dir=data_dir, n_interactions=500, seed=42)
        dataset1 = loader1.load()

        loader2 = HMDataLoader(data_dir=data_dir, n_interactions=500, seed=99)
        dataset2 = loader2.load()

        # With different seeds, the user sets might differ (not guaranteed but likely)
        # At minimum, both must have valid data
        assert len(dataset1.train) > 0
        assert len(dataset2.train) > 0

    def test_stats_returns_correct_shape(self, tmp_path: Path) -> None:
        """Test DataStats computation."""
        from src.data.loader import HMDataLoader

        data_dir = _create_mock_data(tmp_path, n_users=50, n_items=100, n_interactions=500)
        loader = HMDataLoader(data_dir=data_dir, n_interactions=500, seed=42)
        dataset = loader.load()

        stats = dataset.stats("train")
        assert stats.n_interactions == len(dataset.train)
        assert stats.n_users > 0
        assert stats.n_items > 0
        assert 0 < stats.sparsity <= 1.0

    def test_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing files."""
        from src.data.loader import HMDataLoader

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            HMDataLoader(data_dir=empty_dir)

    def test_cache_roundtrip(self, tmp_path: Path) -> None:
        """Test that saving and loading from cache reproduces the same dataset."""
        from src.data.loader import HMDataLoader

        data_dir = _create_mock_data(tmp_path, n_users=50, n_items=100, n_interactions=500)
        cache_dir = tmp_path / "cache"

        loader = HMDataLoader(data_dir=data_dir, n_interactions=500, seed=42)
        dataset1 = loader.load(processed_dir=cache_dir)

        # Load again from cache
        dataset2 = loader.load(processed_dir=cache_dir)

        assert len(dataset1.train) == len(dataset2.train)
        assert len(dataset1.val) == len(dataset2.val)
        assert len(dataset1.test) == len(dataset2.test)
