"""H&M dataset loading and preprocessing pipeline.

Handles deterministic subsampling, chronological train/val/test splitting,
and feature engineering for the H&M Personalized Fashion Recommendations dataset.

Dataset structure:
    - transactions_train.csv: purchase history with (customer_id, article_id, t_dat)
    - articles.csv: article metadata (product group, color, etc.)
    - customers.csv: customer demographics (age, club_member_status, etc.)

Design decisions:
    - Polars for fast, lazy, columnar data processing
    - Chronological split to prevent temporal leakage
    - Fixed seed sampling for reproducibility
    - Core ID integer mapping for efficient matrix operations
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import polars as pl

from src.utils.logger import get_logger
from src.utils.timer import timer

log = get_logger(__name__)


@dataclass
class DataStats:
    """Summary statistics for a dataset split.

    Attributes:
        n_interactions: Total number of user-item interactions.
        n_users: Number of unique users.
        n_items: Number of unique items.
        sparsity: Fraction of possible interactions that are observed.
        date_range: (min_date, max_date) strings.
    """

    n_interactions: int
    n_users: int
    n_items: int
    sparsity: float
    date_range: Tuple[str, str]

    def __str__(self) -> str:
        return (
            f"interactions={self.n_interactions:,} | "
            f"users={self.n_users:,} | "
            f"items={self.n_items:,} | "
            f"sparsity={self.sparsity:.6f} | "
            f"dates={self.date_range[0]} → {self.date_range[1]}"
        )


@dataclass
class HMDataset:
    """Container for the full H&M dataset splits.

    Attributes:
        train: Training interactions DataFrame.
        val: Validation interactions DataFrame.
        test: Test interactions DataFrame.
        articles: Article metadata DataFrame.
        customers: Customer metadata DataFrame.
        user2idx: Mapping from customer_id string to integer index.
        item2idx: Mapping from article_id string to integer index.
        idx2user: Reverse mapping.
        idx2item: Reverse mapping.
    """

    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    articles: pl.DataFrame
    customers: pl.DataFrame
    user2idx: Dict[str, int]
    item2idx: Dict[str, int]
    idx2user: Dict[int, str]
    idx2item: Dict[int, str]

    @property
    def n_users(self) -> int:
        """Number of unique users in the mapped space."""
        return len(self.user2idx)

    @property
    def n_items(self) -> int:
        """Number of unique items in the mapped space."""
        return len(self.item2idx)

    def stats(self, split: str = "train") -> DataStats:
        """Compute summary statistics for a given split.

        Args:
            split: One of 'train', 'val', 'test'.

        Returns:
            DataStats instance with computed summary.

        Raises:
            ValueError: If split name is not recognized.
        """
        split_map = {"train": self.train, "val": self.val, "test": self.test}
        if split not in split_map:
            raise ValueError(f"Unknown split '{split}'. Choose from {list(split_map)}")
        df = split_map[split]

        n_users = df["user_idx"].n_unique()
        n_items = df["item_idx"].n_unique()
        n = len(df)
        sparsity = 1.0 - (n / (n_users * n_items)) if n_users * n_items > 0 else 1.0

        dates = (
            str(df["t_dat"].min()),
            str(df["t_dat"].max()),
        )
        return DataStats(
            n_interactions=n,
            n_users=n_users,
            n_items=n_items,
            sparsity=sparsity,
            date_range=dates,
        )


class HMDataLoader:
    """End-to-end data pipeline for the H&M recommendation dataset.

    Implements deterministic subsampling, filtering, ID mapping,
    and chronological splitting without any temporal leakage.

    The sampling strategy selects users by a deterministic hash-based
    approach seeded by a fixed random state, ensuring full reproducibility
    without any dependence on Pandas/NumPy global state.

    Args:
        data_dir: Path to directory containing raw CSV files.
        n_interactions: Target number of interactions in the subset.
        seed: Random seed for all stochastic operations.
        min_user_interactions: Minimum purchases per user to include.
        min_item_interactions: Minimum purchases per item to include.
        train_ratio: Fraction of chronological data for training.
        val_ratio: Fraction for validation.
    """

    TRANSACTIONS_FILE = "transactions_train.csv"
    ARTICLES_FILE = "articles.csv"
    CUSTOMERS_FILE = "customers.csv"

    def __init__(
        self,
        data_dir: Path,
        n_interactions: int = 100_000,
        seed: int = 42,
        min_user_interactions: int = 5,
        min_item_interactions: int = 3,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.n_interactions = n_interactions
        self.seed = seed
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self._validate_paths()

    def _validate_paths(self) -> None:
        """Check all required files exist before loading.

        Raises:
            FileNotFoundError: If any required CSV is missing.
        """
        for fname in [self.TRANSACTIONS_FILE, self.ARTICLES_FILE, self.CUSTOMERS_FILE]:
            fpath = self.data_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Required data file not found: {fpath}\n"
                    f"Download from: https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations"
                )

    def load(self, processed_dir: Optional[Path] = None) -> HMDataset:
        """Load, process, and split the H&M dataset.

        Attempts to load from cache if processed_dir is provided and
        the processed files exist. Otherwise runs the full pipeline.

        Args:
            processed_dir: Optional directory to cache processed data.

        Returns:
            HMDataset with train/val/test splits and metadata.
        """
        if processed_dir is not None:
            cached = self._try_load_cache(Path(processed_dir))
            if cached is not None:
                log.info("Loaded dataset from cache")
                return cached

        with timer("Full data pipeline"):
            transactions = self._load_transactions()
            articles = self._load_articles()
            customers = self._load_customers()

            log.info(f"Raw transactions: {len(transactions):,}")

            transactions = self._filter_core_users(transactions)
            log.info(f"After core filtering: {len(transactions):,}")

            transactions = self._deterministic_sample(transactions)
            log.info(f"After sampling: {len(transactions):,}")

            transactions, articles, customers = self._align_metadata(
                transactions, articles, customers
            )

            train_raw, val_raw, test_raw = self._chronological_split(transactions)

            user2idx, item2idx = self._build_id_maps(train_raw, full_df=transactions)
            train = self._apply_id_maps(train_raw, user2idx, item2idx)
            val = self._apply_id_maps(val_raw, user2idx, item2idx)
            test = self._apply_id_maps(test_raw, user2idx, item2idx)

            dataset = HMDataset(
                train=train,
                val=val,
                test=test,
                articles=articles,
                customers=customers,
                user2idx=user2idx,
                item2idx=item2idx,
                idx2user={v: k for k, v in user2idx.items()},
                idx2item={v: k for k, v in item2idx.items()},
            )

        # Log split statistics
        for split in ["train", "val", "test"]:
            stats = dataset.stats(split)
            log.info(f"[{split.upper():5s}] {stats}")

        if processed_dir is not None:
            self._save_cache(dataset, Path(processed_dir))

        return dataset

    def _load_transactions(self) -> pl.DataFrame:
        """Load and minimally clean the transactions file.

        Returns:
            Polars DataFrame with columns: customer_id, article_id, t_dat, price.
        """
        log.info("Loading transactions...")
        df = pl.read_csv(
            self.data_dir / self.TRANSACTIONS_FILE,
            columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
            schema_overrides={
                "customer_id": pl.Utf8,
                "article_id": pl.Utf8,
                "price": pl.Float32,
                "sales_channel_id": pl.Int8,
            },
        )

        df = df.with_columns(
            pl.col("t_dat").str.to_date("%Y-%m-%d"),
            pl.col("article_id").cast(pl.Utf8),
        ).sort("t_dat")

        log.info(f"Loaded {len(df):,} transactions spanning {df['t_dat'].min()} → {df['t_dat'].max()}")
        return df

    def _load_articles(self) -> pl.DataFrame:
        """Load article metadata and engineer features.

        Returns:
            Polars DataFrame with article features.
        """
        log.info("Loading articles metadata...")
        df = pl.read_csv(
            self.data_dir / self.ARTICLES_FILE,
            schema_overrides={"article_id": pl.Utf8},
        )
        # Normalize text columns
        text_cols = [c for c in df.columns if df[c].dtype == pl.Utf8 and c != "article_id"]
        df = df.with_columns(
            [pl.col(c).str.to_lowercase().str.strip_chars() for c in text_cols]
        )
        return df

    def _load_customers(self) -> pl.DataFrame:
        """Load customer demographics and engineer features.

        Returns:
            Polars DataFrame with customer features.
        """
        log.info("Loading customers metadata...")
        df = pl.read_csv(
            self.data_dir / self.CUSTOMERS_FILE,
            schema_overrides={"customer_id": pl.Utf8},
        )
        # Fill missing ages with median
        if "age" in df.columns:
            median_age = df["age"].median()
            df = df.with_columns(pl.col("age").fill_null(median_age))
        return df

    def _filter_core_users(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply k-core filtering to remove cold users and items.

        Iteratively removes users/items with fewer than the minimum
        interaction thresholds until convergence. This is the standard
        k-core filtering used in production RecSys pipelines.

        Args:
            df: Raw transactions DataFrame.

        Returns:
            Filtered DataFrame satisfying both min interaction constraints.
        """
        log.info(
            f"Applying k-core filter (min_user={self.min_user_interactions}, "
            f"min_item={self.min_item_interactions})..."
        )
        prev_len = len(df) + 1
        iteration = 0

        while len(df) < prev_len:
            prev_len = len(df)
            iteration += 1

            # Filter users
            user_counts = df.group_by("customer_id").agg(pl.len().alias("n"))
            valid_users = user_counts.filter(
                pl.col("n") >= self.min_user_interactions
            )["customer_id"]
            df = df.filter(pl.col("customer_id").is_in(valid_users))

            # Filter items
            item_counts = df.group_by("article_id").agg(pl.len().alias("n"))
            valid_items = item_counts.filter(
                pl.col("n") >= self.min_item_interactions
            )["article_id"]
            df = df.filter(pl.col("article_id").is_in(valid_items))

        log.info(
            f"K-core converged in {iteration} iterations: "
            f"{len(df):,} interactions remaining"
        )
        return df

    def _deterministic_sample(self, df: pl.DataFrame) -> pl.DataFrame:
        """Sample exactly n_interactions interactions deterministically.

        Selects a stratified subset of users such that the total
        interactions approximate n_interactions. Uses a hash-based
        shuffle over user IDs to ensure reproducibility without
        relying on global NumPy state.

        The approach:
        1. Hash each user_id with the seed to get a deterministic rank
        2. Sort users by their hash rank
        3. Greedily select users until n_interactions is reached

        Args:
            df: Filtered transactions DataFrame.

        Returns:
            Subsampled DataFrame with approximately n_interactions rows.
        """
        if len(df) <= self.n_interactions:
            log.info("Dataset smaller than target — no sampling needed")
            return df

        log.info(f"Sampling {self.n_interactions:,} from {len(df):,} interactions...")

        # Hash-based deterministic user ordering
        user_counts = (
            df.group_by("customer_id")
            .agg(pl.len().alias("n_interactions"))
            .sort("customer_id")  # Deterministic base order
        )

        # Compute SHA-256 hash of (seed + user_id) for deterministic shuffle
        def _hash_user(uid: str) -> int:
            """Return deterministic 32-bit integer hash of seed+user_id."""
            h = hashlib.sha256(f"{self.seed}:{uid}".encode()).hexdigest()
            return int(h[:8], 16)

        user_df = user_counts.to_pandas()
        user_df["hash_rank"] = user_df["customer_id"].apply(_hash_user)
        user_df = user_df.sort_values("hash_rank").reset_index(drop=True)

        # Greedy selection until we reach target
        cumsum = user_df["n_interactions"].cumsum()
        cutoff = (cumsum <= self.n_interactions).sum()
        cutoff = max(cutoff, 1)
        selected_users = set(user_df.iloc[:cutoff]["customer_id"].tolist())

        sampled = df.filter(pl.col("customer_id").is_in(selected_users))
        log.info(
            f"Sampled {len(sampled):,} interactions from "
            f"{len(selected_users):,} users"
        )
        return sampled

    def _align_metadata(
        self,
        transactions: pl.DataFrame,
        articles: pl.DataFrame,
        customers: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Restrict metadata to items/users present in transactions.

        Args:
            transactions: Sampled transaction DataFrame.
            articles: Full articles metadata.
            customers: Full customers metadata.

        Returns:
            Tuple of (transactions, aligned_articles, aligned_customers).
        """
        article_ids = transactions["article_id"].unique()
        customer_ids = transactions["customer_id"].unique()

        articles = articles.filter(pl.col("article_id").is_in(article_ids))
        customers = customers.filter(pl.col("customer_id").is_in(customer_ids))

        log.info(
            f"Aligned metadata: {len(articles):,} articles, "
            f"{len(customers):,} customers"
        )
        return transactions, articles, customers

    def _build_id_maps(
        self, train_df: pl.DataFrame, full_df: Optional[pl.DataFrame] = None
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Build contiguous integer ID mappings for matrix operations.

        Integer IDs are sorted by training frequency (most popular = lowest index)
        to encourage cache locality in embedding lookups without leaking future frequencies.
        Any items/users appearing only in validation/test are appended to preserve catalog coverage.

        Args:
            train_df: Training interactions DataFrame with string IDs.
            full_df: Optional full transactions DataFrame to include all entities.

        Returns:
            Tuple of (user2idx, item2idx) dictionaries.
        """
        user_freq = (
            train_df.group_by("customer_id")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
        )
        item_freq = (
            train_df.group_by("article_id")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
        )

        user_list = user_freq["customer_id"].to_list()
        item_list = item_freq["article_id"].to_list()

        if full_df is not None:
            extra_users = sorted(set(full_df["customer_id"].unique().to_list()) - set(user_list))
            extra_items = sorted(set(full_df["article_id"].unique().to_list()) - set(item_list))
            user_list.extend(extra_users)
            item_list.extend(extra_items)

        user2idx = {uid: i for i, uid in enumerate(user_list)}
        item2idx = {iid: i for i, iid in enumerate(item_list)}

        log.info(f"Built ID maps: {len(user2idx):,} users, {len(item2idx):,} items")
        return user2idx, item2idx

    def _apply_id_maps(
        self,
        df: pl.DataFrame,
        user2idx: Dict[str, int],
        item2idx: Dict[str, int],
    ) -> pl.DataFrame:
        """Apply integer ID mappings to transaction DataFrame.

        Args:
            df: Transactions DataFrame with string IDs.
            user2idx: Customer ID to integer index mapping.
            item2idx: Article ID to integer index mapping.

        Returns:
            DataFrame with added user_idx and item_idx columns.
        """
        user_map_df = pl.DataFrame(
            {"customer_id": list(user2idx.keys()), "user_idx": list(user2idx.values())},
            schema={"customer_id": pl.Utf8, "user_idx": pl.Int32}
        )
        item_map_df = pl.DataFrame(
            {"article_id": list(item2idx.keys()), "item_idx": list(item2idx.values())},
            schema={"article_id": pl.Utf8, "item_idx": pl.Int32}
        )

        df = (
            df.join(user_map_df, on="customer_id", how="left")
            .join(item_map_df, on="article_id", how="left")
        )
        return df

    def _chronological_split(
        self, df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split interactions by time to prevent leakage.

        Uses global date quantiles so that train/val/test cover
        non-overlapping chronological periods. This mirrors real
        production evaluation where the model sees historical data
        and predicts future behavior.

        IMPORTANT: Never use random splitting for temporal data —
        it leaks future purchase signals into training.

        Args:
            df: Transactions with t_dat date column.

        Returns:
            Tuple of (train, val, test) DataFrames sorted by date.
        """
        dates = df["t_dat"].sort()
        n = len(dates)

        train_end_idx = int(n * self.train_ratio)
        val_end_idx = int(n * (self.train_ratio + self.val_ratio))

        train_end_date = dates[train_end_idx]
        val_end_date = dates[val_end_idx]

        train = df.filter(pl.col("t_dat") < train_end_date)
        val = df.filter(
            (pl.col("t_dat") >= train_end_date) & (pl.col("t_dat") < val_end_date)
        )
        test = df.filter(pl.col("t_dat") >= val_end_date)

        log.info(
            f"Chronological split: "
            f"train={len(train):,} (<{train_end_date}), "
            f"val={len(val):,} ({train_end_date}–{val_end_date}), "
            f"test={len(test):,} (≥{val_end_date})"
        )
        return train, val, test

    def _try_load_cache(self, processed_dir: Path) -> Optional[HMDataset]:
        """Attempt to load preprocessed data from Parquet cache.

        Args:
            processed_dir: Directory containing cached Parquet files.

        Returns:
            Loaded HMDataset if cache exists and is valid, else None.
        """
        import json as _json

        required = [
            "train.parquet", "val.parquet", "test.parquet",
            "articles.parquet", "customers.parquet",
            "user2idx.json", "item2idx.json",
        ]
        if not all((processed_dir / f).exists() for f in required):
            return None

        try:
            log.info(f"Loading cached data from {processed_dir}")
            train = pl.read_parquet(processed_dir / "train.parquet")
            val = pl.read_parquet(processed_dir / "val.parquet")
            test = pl.read_parquet(processed_dir / "test.parquet")
            articles = pl.read_parquet(processed_dir / "articles.parquet")
            customers = pl.read_parquet(processed_dir / "customers.parquet")

            # Load full ID maps (preserves all items, not just training ones)
            with open(processed_dir / "user2idx.json") as f:
                user2idx: Dict[str, int] = _json.load(f)
            with open(processed_dir / "item2idx.json") as f:
                item2idx: Dict[str, int] = _json.load(f)

            return HMDataset(
                train=train, val=val, test=test,
                articles=articles, customers=customers,
                user2idx=user2idx, item2idx=item2idx,
                idx2user={v: k for k, v in user2idx.items()},
                idx2item={v: k for k, v in item2idx.items()},
            )
        except Exception as e:
            log.warning(f"Cache load failed ({e}), reprocessing...")
            return None

    def _save_cache(self, dataset: HMDataset, processed_dir: Path) -> None:
        """Persist processed DataFrames to Parquet for fast reloading.

        Also saves ID maps as JSON to preserve the complete catalog size,
        including items that only appear in val/test splits.

        Args:
            dataset: Loaded HMDataset to persist.
            processed_dir: Target directory for Parquet files.
        """
        import json as _json

        processed_dir.mkdir(parents=True, exist_ok=True)
        dataset.train.write_parquet(processed_dir / "train.parquet")
        dataset.val.write_parquet(processed_dir / "val.parquet")
        dataset.test.write_parquet(processed_dir / "test.parquet")
        dataset.articles.write_parquet(processed_dir / "articles.parquet")
        dataset.customers.write_parquet(processed_dir / "customers.parquet")

        # Persist ID maps so cache reload has full catalog knowledge
        with open(processed_dir / "user2idx.json", "w") as f:
            _json.dump(dataset.user2idx, f)
        with open(processed_dir / "item2idx.json", "w") as f:
            _json.dump(dataset.item2idx, f)

        log.info(f"Cached processed data to {processed_dir}")


def build_loo_validation(train_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Leave-one-out validation for LightGBM ranker.

    Holds out each user's single most recent purchase from training.
    This guarantees exactly 1 positive label per user in validation,
    giving LightGBM a stable gradient signal for early stopping.
    Critically, only the exact held-out row is removed, preserving repeat purchases.

    Args:
        train_df: full training DataFrame with [user_idx, item_idx, t_dat]

    Returns:
        train_reduced: training data with LOO rows removed
        val_loo:       the held-out rows (1 per user)
    """
    # Add temporary unique row ID to ensure exact single-row holdout
    df_with_id = train_df.with_row_index("__row_id")

    # Get each user's single most recent purchase (sorted by t_dat descending)
    val_loo_indexed = (
        df_with_id
        .sort(["user_idx", "t_dat"], descending=[False, True])
        .group_by("user_idx", maintain_order=True)
        .first()
    )

    held_out_ids = set(val_loo_indexed["__row_id"].to_list())

    train_reduced = df_with_id.filter(~pl.col("__row_id").is_in(held_out_ids)).drop("__row_id")
    val_loo = val_loo_indexed.drop("__row_id")

    return train_reduced, val_loo

