#!/usr/bin/env python3
"""Main training script for H&M multi-stage recommendation system.

Entry point for the full pipeline. Reads configuration via Hydra
and orchestrates all stages: data loading, retrieval, feature engineering,
ranking, and evaluation.

Usage:
    python scripts/train.py                           # Default config
    python scripts/train.py use_ranker=false          # No ranker
    python scripts/train.py als_factors=256           # Override ALS factors
    python scripts/train.py data_dir=/path/to/hm_data # Custom data path

Output artifacts:
    artifacts/models/          - Saved model files
    artifacts/metrics/         - Evaluation metrics JSON + parquet
    artifacts/figures/         - Publication-quality plots
    artifacts/reports/         - Summary reports
"""

import argparse
import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.analysis.visualizer import generate_all_figures
from src.pipeline.runner import PipelineConfig, PipelineRunner
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def str2bool(v: str | bool) -> bool:
    """Convert string or boolean value to bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="H&M Recommendation System Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw H&M CSV files")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Output artifacts directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Processed data cache")
    parser.add_argument("--n-interactions", type=int, default=100_000, help="Target interaction count")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--use-popularity", action=argparse.BooleanOptionalAction, default=True, help="Include popularity retriever")
    parser.add_argument("--use-als", action=argparse.BooleanOptionalAction, default=True, help="Include ALS retriever")
    parser.add_argument("--use-two-tower", action=argparse.BooleanOptionalAction, default=True, help="Include Two-Tower retriever")
    parser.add_argument("--use-ranker", action=argparse.BooleanOptionalAction, default=True, help="Include LightGBM ranker")
    parser.add_argument("--als-factors", type=int, default=128, help="ALS embedding factors")
    parser.add_argument("--als-iterations", type=int, default=30, help="ALS training iterations")
    parser.add_argument("--two-tower-epochs", type=int, default=20, help="Two-Tower training epochs")
    parser.add_argument("--two-tower-dim", type=int, default=128, help="Two-Tower embedding dimension")
    parser.add_argument("--lgbm-estimators", type=int, default=500, help="LightGBM max trees")
    parser.add_argument("--n-candidates", type=int, default=200, help="Fusion candidate set size")
    parser.add_argument("--top-k", type=int, default=12, help="Final recommendation cutoff")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu/mps)")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap samples for CI")
    parser.add_argument("--generate-plots", action=argparse.BooleanOptionalAction, default=True, help="Generate figures after training")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    """Run the full recommendation system training pipeline."""
    args = parse_args()

    setup_logger(level=args.log_level, log_file=Path(args.artifacts_dir) / "logs" / "train.log")
    set_seed(args.seed)

    logger.info("=" * 70)
    logger.info("H&M Personalized Fashion Recommendation System")
    logger.info("Production-Grade Multi-Stage Pipeline")
    logger.info("=" * 70)

    config = PipelineConfig(
        seed=args.seed,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        processed_dir=args.processed_dir,
        n_interactions=args.n_interactions,
        top_k=args.top_k,
        n_candidates=args.n_candidates,
        use_popularity=args.use_popularity,
        use_als=args.use_als,
        use_two_tower=args.use_two_tower,
        use_ranker=args.use_ranker,
        als_factors=args.als_factors,
        als_iterations=args.als_iterations,
        two_tower_embedding_dim=args.two_tower_dim,
        two_tower_epochs=args.two_tower_epochs,
        lgbm_n_estimators=args.lgbm_estimators,
        device=args.device,
        n_bootstrap=args.n_bootstrap,
    )

    runner = PipelineRunner(config)

    try:
        artifacts = runner.run()
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error(
            "Please download the H&M dataset from Kaggle:\n"
            "  kaggle competitions download -c h-and-m-personalized-fashion-recommendations\n"
            f"  and extract to: {args.data_dir}"
        )
        sys.exit(1)

    # Generate visualization figures
    if args.generate_plots:
        logger.info("Generating visualization figures...")
        try:
            generate_all_figures(
                results_dir=Path(args.artifacts_dir) / "metrics",
                figures_dir=Path(args.artifacts_dir) / "figures",
            )
        except Exception as e:
            logger.warning(f"Figure generation failed (non-fatal): {e}")

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)
    for model_name, result in artifacts.results.items():
        recall_key = f"recall@{args.top_k}"
        ndcg_key = f"ndcg@{args.top_k}"
        if recall_key in result.metrics:
            r = result.metrics[recall_key]
            n = result.metrics.get(ndcg_key)
            ndcg_str = f"  NDCG@{args.top_k}={n.mean:.4f}" if n else ""
            logger.info(
                f"  {model_name:30s} | "
                f"Recall@{args.top_k}={r.mean:.4f} ± {r.std:.4f} "
                f"[{r.ci_lower:.4f}, {r.ci_upper:.4f}]"
                f"{ndcg_str}"
            )

    logger.info(f"\nArtifacts saved to: {args.artifacts_dir}/")
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
