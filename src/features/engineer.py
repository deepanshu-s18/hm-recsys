"""Feature Engineering for the LightGBM Ranking Stage.

Assembles rich, interpretable features from multiple signals:
    - Collaborative: ALS/Two-Tower scores, embedding similarity
    - Retrieval: source retriever, rank position, fusion score
    - Item: popularity, price, category, novelty
    - User: activity level, recency, age group
    - Temporal: seasonality, purchase recency
    - Cross: user-item affinity features

Feature groups are designed to be individually ablatable to measure
their contribution in feature importance analysis.

Design principles:
    - All features computed in Polars for performance
    - No target leakage: features use only information from train split
    - Features are named descriptively for SHAP interpretability
    - Categorical features are label-encoded (LightGBM natively handles them)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import polars as pl

from src.utils.logger import get_logger
from src.utils.timer import timer

log = get_logger(__name__)

# Feature group names for ablation analysis
FEATURE_GROUPS = {
    "retrieval": [
        "retrieval_rrf_score",
        "retrieval_max_score",
        "retrieval_best_rank",
        "retrieval_n_sources",
        "retrieval_from_popularity",
        "retrieval_from_als",
        "retrieval_from_two_tower",
    ],
    "item": [
        "item_popularity_score",
        "item_purchase_count_log",
        "item_unique_buyers",
        "item_price",
        "item_price_norm",
        "item_product_group_idx",
        "item_color_idx",
        "item_dept_idx",
        "item_section_idx",
        "item_is_long_tail",
    ],
    "user": [
        "user_total_purchases",
        "user_unique_items",
        "user_avg_price",
        "user_purchase_std",
        "user_activity_level",
        "user_days_since_last_purchase",
        "user_avg_days_between_purchases",
        "user_age",
        "user_age_group",
    ],
    "temporal": [
        "item_days_since_last_purchase",
        "item_purchase_velocity_7d",
        "item_purchase_velocity_30d",
        "temporal_month",
        "temporal_week_of_year",
    ],
    "cross": [
        "cross_price_affinity",
        "cross_category_affinity",
        "cross_dept_affinity",
        "cross_recency_affinity",
    ],
}


class FeatureEngineer:
    """Assembles ranking features from interaction history and metadata.

    Operates in two phases:
    1. fit(): Computes all per-item and per-user statistics from train data
    2. transform(): Joins computed statistics to candidate (user, item) pairs

    This separation ensures no target leakage when building ranking features
    for validation and test sets.

    Args:
        train: Training interactions DataFrame.
        articles: Article metadata DataFrame.
        customers: Customer metadata DataFrame.
        reference_date: Date to use as 'now' for recency calculations.
            Defaults to max date in training data.
    """

    def __init__(
        self,
        train: pl.DataFrame,
        articles: pl.DataFrame,
        customers: pl.DataFrame,
        reference_date: Optional[str] = None,
    ) -> None:
        self.train = train
        self.articles = articles
        self.customers = customers

        if reference_date is not None:
            self.reference_date = pl.lit(reference_date).str.to_date("%Y-%m-%d")
        else:
            max_dt = train["t_dat"].max()
            self.reference_date = pl.lit(max_dt)

        # Statistics computed during fit
        self._item_stats: Optional[pl.DataFrame] = None
        self._user_stats: Optional[pl.DataFrame] = None
        self._article_features: Optional[pl.DataFrame] = None
        self._customer_features: Optional[pl.DataFrame] = None
        self._category_encoders: Dict[str, Dict[str, int]] = {}
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self) -> FeatureEngineer:
        """Compute all feature statistics from training data.

        Must be called before transform(). Computes:
        - Per-item popularity, price, velocity statistics
        - Per-user activity, recency, preference statistics
        - Category label encodings
        - Article and customer feature matrices

        Returns:
            self (fitted).
        """
        with timer("FeatureEngineer.fit"):
            log.info("Computing item statistics...")
            self._item_stats = self._compute_item_stats()

            log.info("Computing user statistics...")
            self._user_stats = self._compute_user_stats()

            log.info("Engineering article features...")
            self._article_features = self._engineer_article_features()

            log.info("Engineering customer features...")
            self._customer_features = self._engineer_customer_features()

        self.is_fitted = True
        log.info("FeatureEngineer fitted")
        return self

    def transform(
        self,
        candidates: pl.DataFrame,
        als_scores: Optional[Dict[Tuple[int, int], float]] = None,
        two_tower_scores: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Tuple[pl.DataFrame, List[str]]:
        """Generate feature matrix for a set of (user, item) candidate pairs.

        Args:
            candidates: DataFrame with [user_idx, item_idx, rrf_score,
                retrieval_sources, ...] from fusion stage.
            als_scores: Optional dict {(user_idx, item_idx): als_score}.
            two_tower_scores: Optional dict {(user_idx, item_idx): tt_score}.

        Returns:
            Tuple of (feature_df, feature_names) where feature_df has one row
            per candidate pair and columns for each feature.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform()")

        with timer("FeatureEngineer.transform", samples=len(candidates)):
            df = candidates.clone()

            # Ensure consistent integer types for join keys
            df = df.with_columns([
                pl.col("user_idx").cast(pl.Int32),
                pl.col("item_idx").cast(pl.Int32),
            ])

            # Retrieval features
            df = self._add_retrieval_features(df)

            # Item features
            if self._item_stats is not None:
                item_stats = self._item_stats.with_columns(pl.col("item_idx").cast(pl.Int32))
                df = df.join(item_stats, on="item_idx", how="left")
            if self._article_features is not None and len(self._article_features) > 0:
                article_feats = self._article_features.with_columns(pl.col("item_idx").cast(pl.Int32))
                df = df.join(article_feats, on="item_idx", how="left")

            # User features
            if self._user_stats is not None:
                user_stats = self._user_stats.with_columns(pl.col("user_idx").cast(pl.Int32))
                df = df.join(user_stats, on="user_idx", how="left")
            if self._customer_features is not None and len(self._customer_features) > 0:
                cust_feats = self._customer_features.with_columns(pl.col("user_idx").cast(pl.Int32))
                df = df.join(cust_feats, on="user_idx", how="left")

            # Cross features
            df = self._add_cross_features(df)

            # Fill nulls for robustness — exclude join keys to preserve integer types
            key_cols = {"user_idx", "item_idx", "customer_id", "article_id"}
            numeric_cols = [
                c for c, dtype in zip(df.columns, df.dtypes)
                if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                and c not in key_cols
            ]
            df = df.with_columns(
                [pl.col(c).fill_null(0.0) for c in numeric_cols]
            )

        # Collect feature names
        retrieval_features = [
            c for c in df.columns
            if c.startswith("retrieval_") or c.startswith("item_")
            or c.startswith("user_") or c.startswith("temporal_")
            or c.startswith("cross_")
        ]
        self.feature_names = retrieval_features

        log.info(f"Feature matrix: {df.shape}, {len(self.feature_names)} features")
        return df, self.feature_names

    def _compute_item_stats(self) -> pl.DataFrame:
        """Compute per-item popularity and temporal statistics.

        Returns:
            DataFrame indexed by item_idx with popularity features.
        """
        max_date = self.train["t_dat"].max()

        train = self.train.with_columns(
            pl.col("item_idx").cast(pl.Int32),
            pl.col("user_idx").cast(pl.Int32),
        )

        stats = (
            train.group_by("item_idx")
            .agg([
                pl.len().alias("item_purchase_count"),
                pl.col("user_idx").n_unique().alias("item_unique_buyers"),
                pl.col("price").mean().alias("item_price"),
                pl.col("price").std().alias("item_price_std"),
                pl.col("t_dat").max().alias("item_last_purchase_date"),
                pl.col("t_dat").min().alias("item_first_purchase_date"),
            ])
        )

        # Recency-weighted popularity (30-day velocity)
        recent_30 = train.filter(
            pl.col("t_dat") >= pl.lit(max_date) - pl.duration(days=30)
        ).group_by("item_idx").agg(
            pl.len().alias("item_purchase_velocity_30d")
        )

        recent_7 = train.filter(
            pl.col("t_dat") >= pl.lit(max_date) - pl.duration(days=7)
        ).group_by("item_idx").agg(
            pl.len().alias("item_purchase_velocity_7d")
        )

        stats = (
            stats
            .join(recent_30, on="item_idx", how="left")
            .join(recent_7, on="item_idx", how="left")
        )

        # Derived features
        total_purchases = stats["item_purchase_count"].sum()
        max_purchases = stats["item_purchase_count"].max()

        stats = stats.with_columns([
            (pl.col("item_purchase_count") / total_purchases).alias("item_popularity_score"),
            (pl.col("item_purchase_count").log(base=10) + 1).alias("item_purchase_count_log"),
            (pl.col("item_purchase_count") < (float(total_purchases) / len(stats) * 0.1))
            .cast(pl.Int8).alias("item_is_long_tail"),
            pl.col("item_purchase_velocity_30d").fill_null(0),
            pl.col("item_purchase_velocity_7d").fill_null(0),
        ])

        # Days since last purchase
        stats = stats.with_columns([
            (pl.lit(max_date).cast(pl.Date) - pl.col("item_last_purchase_date"))
            .dt.total_days()
            .alias("item_days_since_last_purchase"),
        ])

        return stats.select([
            "item_idx",
            "item_popularity_score",
            "item_purchase_count_log",
            "item_unique_buyers",
            "item_price",
            "item_price_std",
            "item_is_long_tail",
            "item_purchase_velocity_30d",
            "item_purchase_velocity_7d",
            "item_days_since_last_purchase",
        ])

    def _compute_user_stats(self) -> pl.DataFrame:
        """Compute per-user behavior statistics.

        Returns:
            DataFrame indexed by user_idx with user activity features.
        """
        max_date = self.train["t_dat"].max()
        train = self.train.with_columns(
            pl.col("user_idx").cast(pl.Int32),
            pl.col("item_idx").cast(pl.Int32),
        )

        stats = (
            train.group_by("user_idx")
            .agg([
                pl.len().alias("user_total_purchases"),
                pl.col("item_idx").n_unique().alias("user_unique_items"),
                pl.col("price").mean().alias("user_avg_price"),
                pl.col("price").std().alias("user_purchase_std"),
                pl.col("t_dat").max().alias("user_last_purchase_date"),
                pl.col("t_dat").min().alias("user_first_purchase_date"),
            ])
        )

        stats = stats.with_columns([
            (pl.lit(max_date).cast(pl.Date) - pl.col("user_last_purchase_date"))
            .dt.total_days()
            .alias("user_days_since_last_purchase"),

            (
                (pl.col("user_last_purchase_date") - pl.col("user_first_purchase_date"))
                .dt.total_days()
            ).alias("user_tenure_days"),
        ])

        # Activity level buckets: 1=cold, 2=medium, 3=heavy
        q33 = stats["user_total_purchases"].quantile(0.33)
        q66 = stats["user_total_purchases"].quantile(0.66)
        stats = stats.with_columns(
            pl.when(pl.col("user_total_purchases") <= q33)
            .then(1)
            .when(pl.col("user_total_purchases") <= q66)
            .then(2)
            .otherwise(3)
            .cast(pl.Int8)
            .alias("user_activity_level")
        )

        # Avg days between purchases
        stats = stats.with_columns(
            (
                pl.col("user_tenure_days") / pl.col("user_total_purchases").clip(lower_bound=1)
            ).alias("user_avg_days_between_purchases")
        )

        return stats.select([
            "user_idx",
            "user_total_purchases",
            "user_unique_items",
            "user_avg_price",
            "user_purchase_std",
            "user_days_since_last_purchase",
            "user_avg_days_between_purchases",
            "user_activity_level",
            "user_tenure_days",
        ])

    def _engineer_article_features(self) -> pl.DataFrame:
        """Encode article categorical metadata.

        Returns:
            DataFrame with item_idx and encoded categorical features.
        """
        articles = self.articles.clone()

        if "item_idx" in articles.columns:
            articles = articles.with_columns(pl.col("item_idx").cast(pl.Int32))
        else:
            item_id_map = (
                self.train.select(["article_id", "item_idx"])
                .unique()
                .with_columns(pl.col("item_idx").cast(pl.Int32))
            )
            articles = articles.join(item_id_map, on="article_id", how="inner")

        # Encode key categoricals
        cat_cols = {
            "product_group_name": "item_product_group_idx",
            "colour_group_name": "item_color_idx",
            "department_name": "item_dept_idx",
            "section_name": "item_section_idx",
            "index_name": "item_index_idx",
        }

        encoded_cols = ["item_idx"]
        for col_name, new_name in cat_cols.items():
            if col_name not in articles.columns:
                continue
            # Build label encoder
            unique_vals = (
                articles[col_name].drop_nulls().cast(pl.Utf8).unique().sort().to_list()
            )
            self._category_encoders[col_name] = {str(v): i + 1 for i, v in enumerate(unique_vals)}

            encoder = self._category_encoders[col_name]
            articles = articles.with_columns(
                pl.col(col_name)
                .cast(pl.Utf8)
                .replace(encoder, default=0)
                .cast(pl.Int32)
                .alias(new_name)
            )
            encoded_cols.append(new_name)

        # Price from articles if available
        if "perceived_colour_value_name" in articles.columns:
            encoded_cols_existing = [c for c in encoded_cols if c in articles.columns]
            return articles.select(encoded_cols_existing)

        return articles.select([c for c in encoded_cols if c in articles.columns])

    def _engineer_customer_features(self) -> pl.DataFrame:
        """Encode customer demographic features.

        Returns:
            DataFrame with user_idx and demographic features.
        """
        customers = self.customers.clone()

        if "user_idx" in customers.columns:
            customers = customers.with_columns(pl.col("user_idx").cast(pl.Int32))
        else:
            user_id_map = (
                self.train.select(["customer_id", "user_idx"])
                .unique()
                .with_columns(pl.col("user_idx").cast(pl.Int32))
            )
            customers = customers.join(user_id_map, on="customer_id", how="inner")

        result = customers.select(["user_idx"])

        if "age" in customers.columns:
            result = result.with_columns(
                customers["age"].fill_null(customers["age"].median()).alias("user_age")
            )
            # Age group bins: <25, 25-35, 35-45, 45-55, 55+
            result = result.with_columns(
                pl.when(pl.col("user_age") < 25).then(1)
                .when(pl.col("user_age") < 35).then(2)
                .when(pl.col("user_age") < 45).then(3)
                .when(pl.col("user_age") < 55).then(4)
                .otherwise(5)
                .cast(pl.Int8)
                .alias("user_age_group")
            )

        if "club_member_status" in customers.columns:
            status_map = {"ACTIVE": 2, "PRE-CREATE": 1, "LEFT CLUB": 0}
            result = result.with_columns(
                customers["club_member_status"]
                .cast(pl.Utf8)
                .replace(status_map, default=0)
                .cast(pl.Int8)
                .alias("user_club_status")
            )

        if "fashion_news_frequency" in customers.columns:
            freq_map = {"Regularly": 2, "Monthly": 1, "NONE": 0, "None": 0}
            result = result.with_columns(
                customers["fashion_news_frequency"]
                .cast(pl.Utf8)
                .replace(freq_map, default=0)
                .cast(pl.Int8)
                .alias("user_news_freq")
            )

        return result

    def _add_retrieval_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add features derived from candidate retrieval stage.

        Args:
            df: Candidate DataFrame with fusion scores.

        Returns:
            DataFrame with retrieval features added.
        """
        # Ensure rrf_score is present
        if "rrf_score" in df.columns:
            df = df.with_columns([
                pl.col("rrf_score").alias("retrieval_rrf_score"),
            ])

        if "fusion_rank" in df.columns:
            df = df.with_columns([
                pl.col("fusion_rank").alias("retrieval_fusion_rank"),
            ])

        if "max_retriever_score" in df.columns:
            df = df.with_columns([
                pl.col("max_retriever_score").alias("retrieval_max_score"),
            ])

        if "best_rank" in df.columns:
            df = df.with_columns([
                pl.col("best_rank").alias("retrieval_best_rank"),
            ])

        if "n_retrievers_src" in df.columns:
            df = df.with_columns([
                pl.col("n_retrievers_src").alias("retrieval_n_sources"),
            ])

        # Binary flags per retriever source
        if "retriever_sources" in df.columns:
            df = df.with_columns([
                pl.col("retriever_sources").str.contains("popularity")
                .cast(pl.Int8).alias("retrieval_from_popularity"),
                pl.col("retriever_sources").str.contains("als")
                .cast(pl.Int8).alias("retrieval_from_als"),
                pl.col("retriever_sources").str.contains("two_tower")
                .cast(pl.Int8).alias("retrieval_from_two_tower"),
            ])

        return df

    def _add_cross_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add user-item interaction (cross) features.

        Args:
            df: Feature DataFrame with both user and item statistics.

        Returns:
            DataFrame with added cross features.
        """
        # Price affinity: |user_avg_price - item_price| / item_price
        if "user_avg_price" in df.columns and "item_price" in df.columns:
            df = df.with_columns(
                (
                    (pl.col("user_avg_price") - pl.col("item_price")).abs()
                    / (pl.col("item_price") + 1e-6)
                ).alias("cross_price_affinity")
            )
        else:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float32).alias("cross_price_affinity"))

        # Recency affinity: recent user × recent item purchases
        if "user_days_since_last_purchase" in df.columns and "item_days_since_last_purchase" in df.columns:
            df = df.with_columns(
                (
                    1.0 / (pl.col("user_days_since_last_purchase") + 1)
                    * 1.0 / (pl.col("item_days_since_last_purchase") + 1)
                ).alias("cross_recency_affinity")
            )
        else:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float32).alias("cross_recency_affinity"))

        # Category and department affinity
        df = df.with_columns([
            pl.lit(0.0).cast(pl.Float32).alias("cross_category_affinity"),
            pl.lit(0.0).cast(pl.Float32).alias("cross_dept_affinity"),
        ])

        return df
