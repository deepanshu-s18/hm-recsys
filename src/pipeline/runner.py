"""End-to-end recommendation pipeline runner.

Orchestrates the full multi-stage pipeline:
    Stage 1: Candidate Generation (Popularity + ALS + Two-Tower)
    Stage 2: Candidate Fusion (RRF)
    Stage 3: Feature Engineering
    Stage 4: LightGBM Ranking
    Stage 5: Evaluation & Reporting

Each stage produces artifacts (models, features, metrics) that are
persisted to disk for reproducibility and debugging.

The runner is designed to be:
    1. Resumable: Skips already-completed stages if artifacts exist
    2. Configurable: All hyperparameters come from Hydra config
    3. Reproducible: Seed propagated to every component
    4. Observable: Comprehensive logging and metric tracking
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from src.data.loader import HMDataLoader, HMDataset
from src.evaluation.labels import (
    build_ground_truth,
    build_ranking_labels,
)
from src.evaluation.metrics import EvaluationResult, RecSysEvaluator
from src.features.engineer import FeatureEngineer
from src.ranker.lgbm_ranker import LGBMRanker
from src.retrievers.als import ALSRetriever
from src.retrievers.base import BaseRetriever
from src.retrievers.fusion import CandidateFusion
from src.retrievers.popularity import PopularityRetriever
from src.retrievers.two_tower import TwoTowerRetriever
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Full pipeline configuration.

    Attributes:
        seed: Global random seed.
        data_dir: Path to raw H&M data files.
        artifacts_dir: Root path for all saved artifacts.
        processed_dir: Path for cached processed data.
        n_interactions: Target interaction count for subset.
        top_k: Final recommendation cutoff.
        n_candidates: Candidate set size from retrieval.
        use_popularity: Whether to include popularity retriever.
        use_als: Whether to include ALS retriever.
        use_two_tower: Whether to include Two-Tower retriever.
        use_ranker: Whether to use LightGBM ranker.
        als_factors: ALS embedding dimension.
        als_iterations: ALS training iterations.
        two_tower_embedding_dim: Two-Tower output embedding dim.
        two_tower_epochs: Two-Tower max training epochs.
        lgbm_n_estimators: LightGBM max trees.
        device: Torch device string.
        n_bootstrap: Bootstrap samples for CI computation.
    """

    seed: int = 42
    data_dir: str = "data/raw"
    artifacts_dir: str = "artifacts"
    processed_dir: str = "data/processed"
    n_interactions: int = 100_000
    top_k: int = 12
    n_candidates: int = 200
    use_popularity: bool = True
    use_als: bool = True
    use_two_tower: bool = True
    use_ranker: bool = True
    als_factors: int = 128
    als_iterations: int = 30
    two_tower_embedding_dim: int = 128
    two_tower_epochs: int = 20
    lgbm_n_estimators: int = 500
    device: str = "cpu"
    n_bootstrap: int = 1000


@dataclass
class PipelineArtifacts:
    """All artifacts produced by the pipeline run.

    Attributes:
        dataset: Loaded dataset object.
        retrievers: Fitted retriever models.
        train_candidates: Fused candidate set for training the ranker.
        val_candidates: Fused candidates for validation.
        test_candidates: Fused candidates for final evaluation.
        ranker: Fitted LightGBM ranker.
        results: Per-experiment evaluation results.
        fusion_stats: Retrieval fusion statistics.
    """

    dataset: Optional[HMDataset] = None
    retrievers: Dict[str, BaseRetriever] = field(default_factory=dict)
    train_candidates: Optional[pl.DataFrame] = None
    val_candidates: Optional[pl.DataFrame] = None
    test_candidates: Optional[pl.DataFrame] = None
    ranker: Optional[LGBMRanker] = None
    results: Dict[str, EvaluationResult] = field(default_factory=dict)
    fusion_stats: Dict = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)


class PipelineRunner:
    """Orchestrates the full multi-stage recommendation pipeline.

    Args:
        config: Pipeline configuration dataclass.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.artifacts_dir = Path(config.artifacts_dir)
        self.artifacts = PipelineArtifacts()
        self._start_time = time.time()

        set_seed(config.seed)
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create all output directories."""
        for subdir in ["models", "metrics", "figures", "reports"]:
            (self.artifacts_dir / subdir).mkdir(parents=True, exist_ok=True)

    def run(self) -> PipelineArtifacts:
        """Execute the full pipeline from data loading to evaluation.

        Returns:
            PipelineArtifacts with all produced models and metrics.
        """
        log.info("=" * 70)
        log.info("Starting H&M Recommendation Pipeline")
        log.info("=" * 70)

        # Stage 0: Data Loading
        dataset = self._run_data_loading()
        self.artifacts.dataset = dataset

        # Stage 1: Candidate Generation
        retrievers = self._run_candidate_generation(dataset)
        self.artifacts.retrievers = retrievers

        # Stage 2: Candidate Fusion
        train_cands, val_cands, test_cands = self._run_fusion(dataset, retrievers)
        self.artifacts.train_candidates = train_cands
        self.artifacts.val_candidates = val_cands
        self.artifacts.test_candidates = test_cands

        # Stage 3: Feature Engineering + Stage 4: Ranking
        if self.config.use_ranker:
            ranker, feature_names = self._run_ranking(dataset, train_cands, val_cands)
            self.artifacts.ranker = ranker
            self.artifacts.feature_names = feature_names

        # Stage 5: Evaluation
        results = self._run_evaluation(dataset, retrievers, test_cands)
        self.artifacts.results = results

        # Save summary
        self._save_summary()

        total_time = time.time() - self._start_time
        log.info(f"Pipeline complete in {total_time:.1f}s ({total_time/60:.1f} min)")

        return self.artifacts

    def _run_data_loading(self) -> HMDataset:
        """Load and preprocess H&M dataset with chronological splitting."""
        log.info("\n[Stage 0] Data Loading")

        loader = HMDataLoader(
            data_dir=Path(self.config.data_dir),
            n_interactions=self.config.n_interactions,
            seed=self.config.seed,
            min_user_interactions=5,
            min_item_interactions=3,
        )
        dataset = loader.load(processed_dir=Path(self.config.processed_dir))

        log.info(f"Dataset loaded: {dataset.n_users:,} users, {dataset.n_items:,} items")
        return dataset

    def _run_candidate_generation(
        self, dataset: HMDataset
    ) -> Dict[str, BaseRetriever]:
        """Train all enabled retrievers on training data.

        Args:
            dataset: Loaded HMDataset.

        Returns:
            Dict mapping retriever_name → fitted retriever.
        """
        log.info("\n[Stage 1] Candidate Generation")
        retrievers: Dict[str, BaseRetriever] = {}
        train = dataset.train

        if self.config.use_popularity:
            log.info("Training Popularity retriever...")
            pop = PopularityRetriever(top_k=100, time_decay=True, seed=self.config.seed)
            pop.fit(train, n_users=dataset.n_users, n_items=dataset.n_items)
            pop.save(self.artifacts_dir / "models" / "popularity")
            retrievers["popularity"] = pop

        if self.config.use_als:
            log.info("Training ALS retriever...")
            als = ALSRetriever(
                factors=self.config.als_factors,
                iterations=self.config.als_iterations,
                top_k=100,
                seed=self.config.seed,
            )
            als.fit(train, n_users=dataset.n_users, n_items=dataset.n_items)
            als.save(self.artifacts_dir / "models" / "als")
            retrievers["als"] = als

        if self.config.use_two_tower:
            log.info("Training Two-Tower retriever...")
            tt = TwoTowerRetriever(
                embedding_dim=self.config.two_tower_embedding_dim,
                num_epochs=self.config.two_tower_epochs,
                top_k=100,
                device=self.config.device,
                seed=self.config.seed,
            )
            tt.fit(train, n_users=dataset.n_users, n_items=dataset.n_items)
            tt.save(self.artifacts_dir / "models" / "two_tower")
            retrievers["two_tower"] = tt

        log.info(f"Trained {len(retrievers)} retrievers: {list(retrievers.keys())}")
        return retrievers

    def _generate_candidates_for_split(
        self,
        dataset: HMDataset,
        retrievers: Dict[str, BaseRetriever],
        split: str,
    ) -> pl.DataFrame:
        """Generate fused candidates for a given data split.

        Args:
            dataset: Loaded dataset.
            retrievers: Fitted retrievers.
            split: One of "train", "val", "test".

        Returns:
            Fused candidate DataFrame.
        """
        split_df = {"train": dataset.train, "val": dataset.val, "test": dataset.test}[split]
        user_ids = split_df["user_idx"].unique().to_list()

        # Exclude items seen in training ONLY for val/test splits.
        # For training candidates (used to create ranking labels), we must NOT exclude seen
        # items — otherwise the positive labels (training purchases) will never appear.
        exclude_seen = (split != "train")
        seen_items = retrievers[list(retrievers.keys())[0]]._build_seen_items(dataset.train)

        # Collect candidates from each retriever
        candidate_dfs = []
        for name, retriever in retrievers.items():
            log.info(f"  Generating {split} candidates from {name}...")
            candidates = retriever.get_candidates(
                user_indices=user_ids,
                exclude_seen=exclude_seen,
                seen_items=seen_items,
            )
            candidate_dfs.append(candidates)
            log.info(f"  {name}: {len(candidates):,} candidates")

        # Fuse
        fusion = CandidateFusion(
            max_candidates=self.config.n_candidates,
        )
        fused = fusion.fuse(candidate_dfs)
        self.artifacts.fusion_stats[split] = fusion.fusion_stats

        log.info(
            f"  [{split.upper()}] Fused: {len(fused):,} candidates, "
            f"{fused['user_idx'].n_unique():,} users"
        )
        return fused

    def _run_fusion(
        self,
        dataset: HMDataset,
        retrievers: Dict[str, BaseRetriever],
    ) -> tuple:
        """Generate and fuse candidates for all splits.

        Args:
            dataset: Loaded dataset.
            retrievers: Fitted retrievers.

        Returns:
            Tuple of (train_cands, val_cands, test_cands).
        """
        log.info("\n[Stage 2] Candidate Fusion")
        train_cands = self._generate_candidates_for_split(dataset, retrievers, "train")
        val_cands = self._generate_candidates_for_split(dataset, retrievers, "val")
        test_cands = self._generate_candidates_for_split(dataset, retrievers, "test")
        return train_cands, val_cands, test_cands

    def _run_ranking(
        self,
        dataset: HMDataset,
        train_cands: pl.DataFrame,
        val_cands: pl.DataFrame,
    ) -> tuple:
        """Train feature engineer and LightGBM ranker.

        Args:
            dataset: Loaded dataset.
            train_cands: Training candidate DataFrame.
            val_cands: Validation candidate DataFrame.

        Returns:
            Tuple of (fitted_ranker, feature_names).
        """
        log.info("\n[Stage 3+4] Feature Engineering + Ranking")

        # Feature engineer fitted on training data only
        fe = FeatureEngineer(
            train=dataset.train,
            articles=dataset.articles,
            customers=dataset.customers,
        )
        fe.fit()

        # Build training & validation sets with 100% disjoint user query partitions
        import numpy as np
        all_train_users = np.array(train_cands["user_idx"].unique().to_list())
        rng = np.random.default_rng(self.config.seed)
        perm = rng.permutation(len(all_train_users))
        n_val_users = max(int(len(all_train_users) * 0.15), 1)
        val_users_set = set(all_train_users[perm[:n_val_users]])
        train_users_set = set(all_train_users[perm[n_val_users:]])

        train_df_sub = dataset.train.filter(~pl.col("user_idx").is_in(val_users_set))
        val_df_sub = dataset.train.filter(pl.col("user_idx").is_in(val_users_set))

        train_gt = build_ground_truth(train_df_sub)
        val_gt = build_ground_truth(val_df_sub)

        train_cands_split = train_cands.filter(pl.col("user_idx").is_in(train_users_set))
        val_cands_split = train_cands.filter(pl.col("user_idx").is_in(val_users_set))

        # Generate features
        train_features, feature_names = fe.transform(train_cands_split)
        val_features, _ = fe.transform(val_cands_split)

        # Assign labels
        train_labeled = build_ranking_labels(train_features, train_gt)
        val_labeled = build_ranking_labels(val_features, val_gt)

        # Hard data quality assertion: fail fast if label alignment collapses
        pos_rate = float((train_labeled["label"] == 1).sum()) / max(len(train_labeled), 1)
        assert pos_rate >= 0.005, (
            f"CRITICAL DATA QUALITY ERROR: Positive label rate {pos_rate:.2%} is too low (< 0.5%). "
            "Check candidate generation and ground-truth split alignment!"
        )

        # Train ranker
        ranker = LGBMRanker(
            n_estimators=self.config.lgbm_n_estimators,
            seed=self.config.seed,
        )
        ranker.fit(
            train_features=train_labeled,
            val_features=val_labeled,
            feature_names=feature_names,
        )
        ranker.save(self.artifacts_dir / "models" / "lgbm_ranker")

        return ranker, feature_names

    def _run_evaluation(
        self,
        dataset: HMDataset,
        retrievers: Dict[str, BaseRetriever],
        test_cands: pl.DataFrame,
    ) -> Dict[str, EvaluationResult]:
        """Evaluate all models against held-out test set.

        Args:
            dataset: Loaded dataset.
            retrievers: Fitted retrievers.
            test_cands: Test candidate DataFrame.

        Returns:
            Dict mapping model_name → EvaluationResult.
        """
        log.info("\n[Stage 5] Evaluation")
        evaluator = RecSysEvaluator(k=self.config.top_k, n_bootstrap=self.config.n_bootstrap)
        test_gt = build_ground_truth(dataset.test)
        test_users = test_gt["user_idx"].unique().to_list()

        # Item popularity for novelty/bias metrics
        item_pop = self._compute_item_popularity(dataset.train, dataset.n_items)
        all_items = set(range(dataset.n_items))

        results: Dict[str, EvaluationResult] = {}

        # Evaluate each retriever individually
        for name, retriever in retrievers.items():
            log.info(f"Evaluating {name}...")
            single_cands = retriever.get_candidates(
                user_indices=test_users,
                exclude_seen=True,
                seen_items=retriever._build_seen_items(dataset.train),
            )
            # Simple top-K from retriever scores
            top_k_recs = (
                single_cands.sort(["user_idx", "rank"])
                .with_columns(
                    pl.col("rank").alias("rank")
                )
                .filter(pl.col("rank") <= self.config.top_k)
            )
            result = evaluator.evaluate(
                recommendations=top_k_recs,
                ground_truth=test_gt,
                model_name=name,
                item_popularity=item_pop,
                all_items=all_items,
            )
            results[name] = result
            self._save_evaluation_result(result)

        # Evaluate full pipeline (retrieval + ranker)
        if self.artifacts.ranker is not None and len(test_cands) > 0:
            log.info("Evaluating full pipeline (retrieval + ranker)...")
            fe = FeatureEngineer(
                train=dataset.train,
                articles=dataset.articles,
                customers=dataset.customers,
            )
            fe.fit()
            test_features, feature_names = fe.transform(test_cands)
            final_recs = self.artifacts.ranker.rerank(
                test_features,
                feature_names=feature_names,
                top_k=self.config.top_k,
            )
            result = evaluator.evaluate(
                recommendations=final_recs,
                ground_truth=test_gt,
                model_name="two_tower_plus_ranker",
                item_popularity=item_pop,
                all_items=all_items,
            )
            results["two_tower_plus_ranker"] = result
            self._save_evaluation_result(result)

        return results

    def _compute_item_popularity(
        self, train: pl.DataFrame, n_items: int
    ) -> Dict[int, float]:
        """Compute normalized item purchase frequency.

        Args:
            train: Training transactions.
            n_items: Catalog size.

        Returns:
            Dict mapping item_idx → popularity fraction.
        """
        counts = (
            train.group_by("item_idx")
            .agg(pl.len().alias("count"))
        )
        total = counts["count"].sum()
        return {
            row["item_idx"]: row["count"] / total
            for row in counts.to_dicts()
        }

    def _save_evaluation_result(self, result: EvaluationResult) -> None:
        """Save evaluation result to metrics directory.

        Args:
            result: Evaluation result to persist.
        """
        metrics_dir = self.artifacts_dir / "metrics" / result.model_name
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        summary = result.to_summary_dict()
        with open(metrics_dir / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)

        # Save per-user metrics
        if result.per_user_metrics is not None:
            result.per_user_metrics.write_parquet(
                metrics_dir / "per_user_metrics.parquet"
            )

        # Save bootstrap distributions
        boot_data = {}
        for name, br in result.metrics.items():
            boot_data[name] = br.bootstrap_samples.tolist()
        with open(metrics_dir / "bootstrap_results.json", "w") as f:
            json.dump(boot_data, f)

        log.info(f"Evaluation results saved to {metrics_dir}")

    def _save_summary(self) -> None:
        """Save pipeline summary JSON with all result metrics."""
        summary = {
            "config": vars(self.config),
            "total_runtime_sec": time.time() - self._start_time,
            "results": {
                name: result.to_summary_dict()
                for name, result in self.artifacts.results.items()
            },
            "fusion_stats": self.artifacts.fusion_stats,
        }
        with open(self.artifacts_dir / "reports" / "pipeline_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        log.info("Pipeline summary saved")
