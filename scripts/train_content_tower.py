#!/usr/bin/env python3
"""Content-aware Two-Tower experiment.

Trains the ContentTwoTowerRetriever using pre-computed text and/or
CLIP embeddings, then evaluates against the baseline Two-Tower.

Run after:
    python scripts/encode_text.py         # Option A
    python scripts/encode_images.py       # Option B (optional)

Usage:
    # Text only
    python scripts/train_content_tower.py --use-text

    # Text + CLIP
    python scripts/train_content_tower.py --use-text --use-clip

    # Full run with all options
    python scripts/train_content_tower.py \
        --use-text \
        --use-clip \
        --n-interactions 3000000 \
        --epochs 20
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
import typer
from loguru import logger

from src.data.loader import HMDataLoader
from src.evaluation.labels import build_ground_truth, build_ranking_labels
from src.evaluation.metrics import RecSysEvaluator
from src.retrievers.content_two_tower import ContentTwoTowerRetriever
from src.retrievers.fusion import CandidateFusion
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

app = typer.Typer()


@app.command()
def train(
    data_dir: str = typer.Option("data/raw", help="Raw data directory"),
    processed_dir: str = typer.Option("data/processed", help="Processed cache directory"),
    text_emb_path: str = typer.Option("data/text_embeddings.npy", help="Text embeddings"),
    text_ids_path: str = typer.Option("data/text_article_ids.npy", help="Text article IDs"),
    clip_emb_path: str = typer.Option("data/clip_embeddings.npy", help="CLIP embeddings"),
    clip_ids_path: str = typer.Option("data/clip_article_ids.npy", help="CLIP article IDs"),
    artifacts_dir: str = typer.Option("artifacts", help="Artifacts output directory"),
    n_interactions: int = typer.Option(3_000_000, help="Interaction count"),
    epochs: int = typer.Option(20, help="Training epochs"),
    embedding_dim: int = typer.Option(128, help="Output embedding dimension"),
    batch_size: int = typer.Option(512, help="Training batch size"),
    use_text: bool = typer.Option(False, help="Use text embeddings"),
    use_clip: bool = typer.Option(False, help="Use CLIP image embeddings"),
    top_k: int = typer.Option(12, help="Evaluation cutoff"),
    n_bootstrap: int = typer.Option(1000, help="Bootstrap samples"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Train and evaluate content-aware Two-Tower retriever."""
    setup_logger(level="INFO")
    set_seed(seed)

    if not use_text and not use_clip:
        logger.error("At least one of --use-text or --use-clip must be enabled")
        raise typer.Exit(code=1)

    logger.info("=" * 70)
    logger.info("Content-Aware Two-Tower Experiment")
    modalities = []
    if use_text:
        modalities.append("Text (sentence-transformers)")
    if use_clip:
        modalities.append("CLIP ViT-B/32")
    logger.info(f"Modalities: {' + '.join(modalities)}")
    logger.info("=" * 70)

    # ─── Load Data ────────────────────────────────────────────────────────
    loader = HMDataLoader(data_dir=data_dir, n_interactions=n_interactions, seed=seed)
    dataset = loader.load(processed_dir=Path(processed_dir))
    logger.info(f"Dataset: {dataset.n_users} users, {dataset.n_items} items")

    # ─── Load Content Embeddings ──────────────────────────────────────────
    text_embeddings = None
    text_item_ids = None
    if use_text:
        if not Path(text_emb_path).exists():
            logger.error(
                f"Text embeddings not found at {text_emb_path}\n"
                "Run: python scripts/encode_text.py"
            )
            raise typer.Exit(code=1)
        text_embeddings = np.load(text_emb_path)
        text_item_ids = np.load(text_ids_path, allow_pickle=True)
        logger.info(f"Text embeddings loaded: {text_embeddings.shape}")

    clip_embeddings = None
    clip_item_ids = None
    if use_clip:
        if not Path(clip_emb_path).exists():
            logger.error(
                f"CLIP embeddings not found at {clip_emb_path}\n"
                "Run: python scripts/encode_images.py"
            )
            raise typer.Exit(code=1)
        clip_embeddings = np.load(clip_emb_path)
        clip_item_ids = np.load(clip_ids_path, allow_pickle=True)
        logger.info(f"CLIP embeddings loaded: {clip_embeddings.shape}")

    # ─── Train Retriever ──────────────────────────────────────────────────
    experiment_name = "content_" + "_".join(
        (["text"] if use_text else []) + (["clip"] if use_clip else [])
    )

    retriever = ContentTwoTowerRetriever(
        text_embeddings=text_embeddings,
        text_item_ids=text_item_ids,
        clip_embeddings=clip_embeddings,
        clip_item_ids=clip_item_ids,
        embedding_dim=embedding_dim,
        hidden_dims=[256, 128],
        num_epochs=epochs,
        batch_size=batch_size,
        top_k=100,
        device="cpu",
        seed=seed,
    )

    retriever.fit(
        interactions=dataset.train,
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        item2idx=dataset.item2idx,
    )

    # Save model
    model_path = Path(artifacts_dir) / "models" / experiment_name
    retriever.save(model_path)

    # ─── Generate Candidates ─────────────────────────────────────────────
    seen_items = retriever._build_seen_items(dataset.train)
    test_users = dataset.test["user_idx"].unique().to_list()

    logger.info(f"Generating candidates for {len(test_users)} test users...")
    candidates = retriever.get_candidates(
        user_indices=test_users,
        exclude_seen=True,
        seen_items=seen_items,
    )
    logger.info(f"Candidates: {len(candidates):,}")

    # ─── Evaluate ─────────────────────────────────────────────────────────
    test_gt = build_ground_truth(dataset.test)
    evaluator = RecSysEvaluator(k=top_k, n_bootstrap=n_bootstrap)
    result = evaluator.evaluate(candidates, test_gt, experiment_name)

    # ─── Report ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info(f"RESULTS: {experiment_name}")
    logger.info("=" * 70)

    metrics_of_interest = ["recall@12", "ndcg@12", "mrr", "coverage@12", "diversity"]
    for metric in metrics_of_interest:
        if metric in result.metrics:
            m = result.metrics[metric]
            logger.info(
                f"  {metric:<25} {m.mean:.4f} ± {m.std:.4f} "
                f"[{m.ci_lower:.4f}, {m.ci_upper:.4f}]"
            )

    # ─── Compare vs Baseline ──────────────────────────────────────────────
    baseline_path = Path(artifacts_dir) / "metrics" / "two_tower" / "metrics.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)["metrics"]

        logger.info("\n  Delta vs baseline Two-Tower:")
        for metric in ["recall@12", "ndcg@12", "mrr"]:
            if metric in result.metrics and metric in baseline:
                delta = result.metrics[metric].mean - baseline[metric]["mean"]
                pct = 100 * delta / max(baseline[metric]["mean"], 1e-6)
                direction = "↑" if delta > 0 else "↓"
                logger.info(f"  {metric:<25} {direction} {abs(delta):.4f} ({pct:+.1f}%)")

    # Save results
    metrics_path = Path(artifacts_dir) / "metrics" / experiment_name
    metrics_path.mkdir(parents=True, exist_ok=True)

    metrics_dict = {}
    for name, boot_result in result.metrics.items():
        metrics_dict[name] = {
            "mean": boot_result.mean,
            "std": boot_result.std,
            "ci_lower": boot_result.ci_lower,
            "ci_upper": boot_result.ci_upper,
            "n_bootstrap": n_bootstrap,
        }

    with open(metrics_path / "metrics.json", "w") as f:
        json.dump({
            "model_name": experiment_name,
            "k": top_k,
            "metrics": metrics_dict,
            "modalities": modalities,
        }, f, indent=2)

    logger.info(f"\nResults saved to {metrics_path}")
    logger.info("Done.")


if __name__ == "__main__":
    app()
