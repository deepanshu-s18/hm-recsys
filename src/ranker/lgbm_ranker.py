"""LightGBM LambdaMART Ranker for Stage 4 of the recommendation pipeline.

LambdaMART is the state-of-the-art learning-to-rank algorithm used in
production at Amazon, Microsoft (Bing), and major e-commerce platforms.
It directly optimizes a ranking metric (NDCG) rather than pointwise loss.

Training formulation:
    - For each candidate (user, item) pair, the relevance label = 1 if
      the item was actually purchased, 0 otherwise (binary relevance)
    - LambdaMART computes lambda gradients from NDCG to update the model
    - Query groups = users (all candidates for one user form one group)

Key design decisions:
    1. LambdaMART over pointwise LR: directly optimizes ranking metrics
    2. LightGBM over XGBoost: faster, lower memory, native categorical support
    3. Binary labels (purchase=1): clean signal, no label noise
    4. Feature groups enable ablation studies
    5. SHAP for interpretability (required for production sign-off)

Reference: Burges et al. "Learning to Rank using Gradient Descent." ICML 2005.
           Ke et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS 2017.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl
import shap
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger
from src.utils.timer import timer

log = get_logger(__name__)


class LGBMRanker:
    """LightGBM LambdaMART ranker for final candidate re-ranking.

    Takes the ~200 candidates from fusion, joins them with rich features,
    and re-ranks them using a gradient-boosted tree ensemble trained to
    predict purchase probability.

    Args:
        n_estimators: Number of boosting iterations (trees).
        num_leaves: Max leaves per tree (controls model capacity).
        learning_rate: Gradient descent step size.
        feature_fraction: Fraction of features per tree.
        bagging_fraction: Fraction of data per tree.
        bagging_freq: Bagging frequency (every N rounds).
        min_child_samples: Minimum data in leaves.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.
        early_stopping_rounds: Stop if val metric doesn't improve.
        num_threads: CPU thread count.
        seed: Random seed.

    Example:
        >>> ranker = LGBMRanker(n_estimators=500)
        >>> ranker.fit(train_features, val_features, feature_names)
        >>> predictions = ranker.predict(test_features)
    """

    def __init__(
        self,
        n_estimators: int = 500,
        num_leaves: int = 127,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        min_child_samples: int = 20,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        early_stopping_rounds: int = 30,
        num_threads: int = 4,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.num_threads = num_threads
        self.seed = seed

        self._model: Optional[lgb.Booster] = None
        self._feature_names: List[str] = []
        self._cat_encoders: Dict[str, LabelEncoder] = {}
        self.training_history: Dict[str, List[float]] = {}
        self.feature_importance: Optional[pl.DataFrame] = None
        self.is_fitted: bool = False

    def fit(
        self,
        train_features: pl.DataFrame,
        val_features: pl.DataFrame,
        feature_names: List[str],
        label_col: str = "label",
        group_col: str = "user_idx",
    ) -> LGBMRanker:
        """Train LambdaMART ranker on feature-label pairs.

        Constructs LightGBM Dataset objects with proper query group
        information required for listwise ranking loss computation.

        Args:
            train_features: DataFrame with features, labels, and user groups.
            val_features: Validation DataFrame for early stopping.
            feature_names: List of feature column names to use.
            label_col: Column name for binary relevance labels.
            group_col: Column name for query group IDs (user_idx).

        Returns:
            self (fitted).
        """
        self._feature_names = feature_names
        log.info(f"Training LGBMRanker with {len(feature_names)} features")

        with timer("LGBMRanker.fit"):
            # Prepare matrices
            X_train, y_train, groups_train = self._prepare_dataset(
                train_features, feature_names, label_col, group_col
            )
            X_val, y_val, groups_val = self._prepare_dataset(
                val_features, feature_names, label_col, group_col
            )

            cat_features = [
                c for c in [
                    "item_product_group_idx",
                    "item_color_idx",
                    "item_dept_idx",
                    "item_section_idx",
                    "item_index_idx",
                    "user_activity_level",
                    "user_age_group",
                    "user_club_status",
                    "user_news_freq",
                    "temporal_month",
                    "temporal_week_of_year",
                ] if c in feature_names
            ]

            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                group=groups_train,
                feature_name=feature_names,
                categorical_feature=cat_features if cat_features else "auto",
            )
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                group=groups_val,
                feature_name=feature_names,
                categorical_feature=cat_features if cat_features else "auto",
                reference=train_data,
            )

            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [12],
                "num_leaves": self.num_leaves,
                "learning_rate": self.learning_rate,
                "feature_fraction": self.feature_fraction,
                "bagging_fraction": self.bagging_fraction,
                "bagging_freq": self.bagging_freq,
                "min_child_samples": self.min_child_samples,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
                "num_threads": self.num_threads,
                "seed": self.seed,
                "verbose": -1,
                "label_gain": list(range(max(int(y_train.max()) + 1, 2))),
            }

            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds, verbose=True
                ),
                lgb.log_evaluation(period=50),
                lgb.record_evaluation(self.training_history),
            ]

            self._model = lgb.train(
                params=params,
                train_set=train_data,
                num_boost_round=self.n_estimators,
                valid_sets=[val_data],
                valid_names=["val"],
                callbacks=callbacks,
            )

        # Compute feature importance
        self.feature_importance = self._compute_feature_importance()

        self.is_fitted = True
        best_iter = self._model.best_iteration
        log.info(
            f"LGBMRanker trained: best_iter={best_iter}, "
            f"n_features={len(feature_names)}"
        )
        return self

    def _prepare_dataset(
        self,
        df: pl.DataFrame,
        feature_names: List[str],
        label_col: str,
        group_col: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert polars DataFrame to numpy arrays for LightGBM.

        Ensures proper sorting by user groups (required for LambdaMART)
        and extracts group sizes.

        Args:
            df: Feature DataFrame.
            feature_names: Feature column names.
            label_col: Target column name.
            group_col: User group column name.

        Returns:
            Tuple of (X, y, group_sizes).

        Raises:
            ValueError: If required columns are missing.
        """
        missing = set(feature_names) - set(df.columns)
        if missing:
            log.warning(f"Missing features (filling with 0): {missing}")
            for col in missing:
                df = df.with_columns(pl.lit(0.0).alias(col))

        # Sort by user group for correct group counting
        df = df.sort(group_col)

        # Extract feature matrix
        available_features = [f for f in feature_names if f in df.columns]
        X = df.select(available_features).to_numpy().astype(np.float32)

        # Labels
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found")
        y = df[label_col].to_numpy().astype(np.float32)

        # Group sizes (sorted consecutive user blocks)
        group_counts = (
            df.group_by(group_col, maintain_order=True)
            .agg(pl.len().alias("n"))["n"]
            .to_numpy()
            .astype(np.int32)
        )

        log.debug(
            f"Dataset prepared: X={X.shape}, y_pos={int(y.sum())}, "
            f"n_groups={len(group_counts)}, avg_group={group_counts.mean():.1f}"
        )
        return X, y, group_counts

    def predict(
        self,
        features: pl.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Predict ranking scores for candidate (user, item) pairs.

        Args:
            features: DataFrame with feature columns.
            feature_names: Optional override for feature list.

        Returns:
            Float array of predicted ranking scores.

        Raises:
            RuntimeError: If model is not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")

        names = feature_names or self._feature_names
        available = [f for f in names if f in features.columns]
        X = features.select(available).to_numpy().astype(np.float32)

        return self._model.predict(X, num_iteration=self._model.best_iteration)

    def rerank(
        self,
        candidates: pl.DataFrame,
        feature_names: Optional[List[str]] = None,
        top_k: int = 12,
    ) -> pl.DataFrame:
        """Score candidates and return top-K per user.

        Args:
            candidates: Feature DataFrame with [user_idx, item_idx, features...].
            feature_names: Feature columns to use.
            top_k: Number of items to recommend per user.

        Returns:
            DataFrame with [user_idx, item_idx, ranker_score, rank].
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before rerank()")

        with timer("LGBMRanker.rerank", samples=len(candidates)):
            scores = self.predict(candidates, feature_names)
            result = candidates.select(["user_idx", "item_idx"]).with_columns(
                pl.Series("ranker_score", scores.astype(np.float32))
            )

            # Rank within each user group
            result = (
                result.sort(["user_idx", "ranker_score"], descending=[False, True])
                .with_columns([
                    pl.col("ranker_score")
                    .rank(method="ordinal", descending=True)
                    .over("user_idx")
                    .alias("rank"),
                    pl.col("ranker_score").alias("score"),
                ])
                .filter(pl.col("rank") <= top_k)
            )

        log.info(
            f"Re-ranked {len(candidates):,} candidates → "
            f"{len(result):,} top-{top_k} recommendations"
        )
        return result

    def _compute_feature_importance(self) -> pl.DataFrame:
        """Build feature importance DataFrame from gain and split counts.

        Returns:
            DataFrame with [feature, gain_importance, split_importance].
        """
        assert self._model is not None
        names = self._model.feature_name()
        gain = self._model.feature_importance(importance_type="gain")
        split = self._model.feature_importance(importance_type="split")

        total_gain = gain.sum()
        total_split = split.sum()

        return pl.DataFrame({
            "feature": names,
            "gain_importance": gain / max(total_gain, 1),
            "split_importance": split / max(total_split, 1),
        }).sort("gain_importance", descending=True)

    def compute_shap_values(
        self,
        features: pl.DataFrame,
        n_samples: int = 1000,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Compute SHAP values for model interpretation.

        Uses TreeExplainer for exact Shapley values on tree models.
        SHAP values quantify each feature's marginal contribution
        to each prediction, enabling per-user debugging.

        Args:
            features: Feature DataFrame.
            n_samples: Subsample size for efficiency.
            feature_names: Feature columns to use.

        Returns:
            SHAP values array of shape (n_samples, n_features).
        """
        self._check_fitted()
        names = feature_names or self._feature_names
        available = [f for f in names if f in features.columns]

        # Subsample for speed
        sample_size = min(n_samples, len(features))
        features_sample = features.sample(n=sample_size, seed=self.seed)
        X = features_sample.select(available).to_numpy().astype(np.float32)

        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(X)
        log.info(f"SHAP values computed for {sample_size} samples")
        return shap_values

    def _check_fitted(self) -> None:
        """Verify model is fitted before inference.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self.is_fitted:
            raise RuntimeError("LGBMRanker is not fitted")

    def save(self, path: Path) -> None:
        """Save model, encoders, and metadata.

        Args:
            path: Target directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save LightGBM model (natively to text format for portability)
        self._model.save_model(str(path / "model.lgb"))

        with open(path / "cat_encoders.pkl", "wb") as f:
            pickle.dump(self._cat_encoders, f)

        if self.feature_importance is not None:
            self.feature_importance.write_parquet(path / "feature_importance.parquet")

        config = {
            "feature_names": self._feature_names,
            "training_history": self.training_history,
            "best_iteration": self._model.best_iteration,
            "n_estimators": self.n_estimators,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        log.info(f"LGBMRanker saved to {path}")

    def load(self, path: Path) -> LGBMRanker:
        """Load model from disk.

        Args:
            path: Directory containing saved artifacts.

        Returns:
            self with loaded state.
        """
        path = Path(path)
        self._model = lgb.Booster(model_file=str(path / "model.lgb"))

        with open(path / "cat_encoders.pkl", "rb") as f:
            self._cat_encoders = pickle.load(f)

        fi_path = path / "feature_importance.parquet"
        if fi_path.exists():
            self.feature_importance = pl.read_parquet(fi_path)

        with open(path / "config.json") as f:
            config = json.load(f)
        self._feature_names = config["feature_names"]
        self.training_history = config.get("training_history", {})
        self.is_fitted = True

        log.info(f"LGBMRanker loaded from {path}")
        return self
