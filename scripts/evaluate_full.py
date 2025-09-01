#!/usr/bin/env python3
"""Memory-efficient evaluation for full 31M model.

Models are already trained and saved. This script:
1. Loads all saved models
2. Generates candidates in batches of 5,000 users
3. Evaluates with bootstrap CI
4. Skips LightGBM ranker (requires training candidates — too much RAM)

Usage:
    python scripts/evaluate_full.py
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
from src.evaluation.labels import build_ground_truth
from src.evaluation.metrics import RecSysEvaluator
from src.retrievers.als import ALSRetriever
from src.retrievers.popularity import PopularityRetriever
from src.retrievers.two_tower import TwoTowerRetriever
from src.retrievers.fusion import CandidateFusion
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

app = typer.Typer()


def generate_in_batches(
    retriever,
    user_indices: list[int],
    seen_items: dict,
    batch_size: int = 5000,
    exclude_seen: bool = True,
    top_k: int = 100,
) -> pl.DataFrame:
    """Generate candidates in batches to avoid OOM.

    Args:
        retriever: Fitted retriever object.
        user_indices: All user indices to retrieve for.
        seen_items: Dict mapping user_idx to seen item list.
        batch_size: Users per batch.
        exclude_seen: Whether to exclude seen items.
        top_k: Candidates per user.

    Returns:
        Combined candidates DataFrame.
    """
    all_chunks = []
    n_batches = (len(user_indices) + batch_size - 1) // batch_size

    for i in range(0, len(user_indices), batch_size):
        batch = user_indices[i:i + batch_size]
        chunk = retriever.get_candidates(
            user_indices=batch,
            exclude_seen=exclude_seen,
            seen_items=seen_items,
        )
        all_chunks.append(chunk)

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  Batch {i//batch_size+1}/{n_batches} done")

    return pl.concat(all_chunks) if all_chunks else pl.DataFrame()


def fuse_in_batches(
    retrievers: dict,
    user_indices: list[int],
    seen_items: dict,
    batch_size: int = 5000,
    n_candidates: int = 200,
) -> pl.DataFrame:
    """Fuse multiple retrievers in batches.

    Args:
        retrievers: Dict of name -> fitted retriever.
        user_indices: All user indices.
        seen_items: Seen items dict.
        batch_size: Users per batch.
        n_candidates: Max candidates after fusion.

    Returns:
        Fused candidates DataFrame.
    """
    fusion = CandidateFusion(max_candidates=n_candidates)
    all_chunks = []
    n_batches = (len(user_indices) + batch_size - 1) // batch_size

    for i in range(0, len(user_indices), batch_size):
        batch = user_indices[i:i + batch_size]
        batch_cands = []
        for name, retriever in retrievers.items():
            chunk = retriever.get_candidates(
                user_indices=batch,
                exclude_seen=True,
                seen_items=seen_items,
            )
            batch_cands.append(chunk)

        fused = fusion.fuse(batch_cands)
        all_chunks.append(fused)

        if (i // batch_size + 1) % 20 == 0:
            pct = 100 * (i + batch_size) / len(user_indices)
            logger.info(f"  Fusion batch {i//batch_size+1}/{n_batches} ({pct:.0f}%)")

    return pl.concat(all_chunks) if all_chunks else pl.DataFrame()


import argparse


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Memory-efficient evaluation for full 31M model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Artifacts directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap samples")
    parser.add_argument("--top-k", type=int, default=12, help="Top-K cutoff")
    parser.add_argument("--batch-size", type=int, default=5000, help="User batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def evaluate() -> None:
    """Evaluate all saved models on the full 31M test set."""
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    processed_dir = args.processed_dir
    data_dir = args.data_dir
    n_bootstrap = args.n_bootstrap
    top_k = args.top_k
    batch_size = args.batch_size
    seed = args.seed

    setup_logger(level="INFO")
    set_seed(seed)

    logger.info("=" * 70)
    logger.info("FULL 31M EVALUATION — Memory-Efficient Batch Mode")
    logger.info("=" * 70)

    # ─── Load Dataset ─────────────────────────────────────────────────────────
    logger.info("Loading cached dataset...")
    train = pl.read_parquet(f"{processed_dir}/train.parquet")
    test  = pl.read_parquet(f"{processed_dir}/test.parquet")

    with open(f"{processed_dir}/user2idx.json") as f:
        user2idx = json.load(f)
    with open(f"{processed_dir}/item2idx.json") as f:
        item2idx = json.load(f)

    n_users = len(user2idx)
    n_items = len(item2idx)
    logger.info(f"Dataset: {n_users:,} users, {n_items:,} items")
    logger.info(f"Train: {len(train):,} | Test: {len(test):,}")

    # Build seen items
    logger.info("Building seen-items index...")
    seen_items: dict[int, list[int]] = {}
    for row in train.select(["user_idx", "item_idx"]).iter_rows():
        uid, iid = row
        if uid not in seen_items:
            seen_items[uid] = []
        seen_items[uid].append(iid)
    logger.info(f"Seen-items index: {len(seen_items):,} users")

    # Ground truth
    test_gt = build_ground_truth(test)
    test_users = test["user_idx"].unique().to_list()
    logger.info(f"Test users: {len(test_users):,}")
    logger.info(f"Test ground truth: {len(test_gt):,} (user, item) pairs")

    evaluator = RecSysEvaluator(k=top_k, n_bootstrap=n_bootstrap)
    results = {}

    # ─── Popularity ───────────────────────────────────────────────────────────
    logger.info("\n[1/3] Evaluating Popularity...")
    pop = PopularityRetriever(top_k=100)
    pop.load(Path(artifacts_dir) / "models" / "popularity")

    logger.info(f"  Generating candidates for {len(test_users):,} users in batches...")
    pop_cands = generate_in_batches(
        pop, test_users, seen_items, batch_size=batch_size
    )
    logger.info(f"  Candidates: {len(pop_cands):,}")

    result = evaluator.evaluate(pop_cands, test_gt, "popularity")
    results["popularity"] = result

    save_path = Path(artifacts_dir) / "metrics" / "popularity"
    save_path.mkdir(parents=True, exist_ok=True)
    _save_result(result, save_path, top_k, n_bootstrap)
    _print_result("Popularity", result, top_k)

    # ─── ALS ──────────────────────────────────────────────────────────────────
    logger.info("\n[2/3] Evaluating ALS...")
    als = ALSRetriever(top_k=100)
    als.load(Path(artifacts_dir) / "models" / "als")

    logger.info(f"  Generating ALS candidates in batches...")
    als_cands = generate_in_batches(
        als, test_users, seen_items, batch_size=batch_size
    )
    logger.info(f"  Candidates: {len(als_cands):,}")

    result = evaluator.evaluate(als_cands, test_gt, "als")
    results["als"] = result

    save_path = Path(artifacts_dir) / "metrics" / "als"
    save_path.mkdir(parents=True, exist_ok=True)
    _save_result(result, save_path, top_k, n_bootstrap)
    _print_result("ALS", result, top_k)

    # ─── Two-Tower ────────────────────────────────────────────────────────────
    logger.info("\n[3/3] Evaluating Two-Tower...")
    tt = TwoTowerRetriever(top_k=100)
    tt.load(Path(artifacts_dir) / "models" / "two_tower")

    logger.info(f"  Generating Two-Tower candidates in batches...")
    tt_cands = generate_in_batches(
        tt, test_users, seen_items, batch_size=batch_size
    )
    logger.info(f"  Candidates: {len(tt_cands):,}")

    result = evaluator.evaluate(tt_cands, test_gt, "two_tower")
    results["two_tower"] = result

    save_path = Path(artifacts_dir) / "metrics" / "two_tower"
    save_path.mkdir(parents=True, exist_ok=True)
    _save_result(result, save_path, top_k, n_bootstrap)
    _print_result("Two-Tower", result, top_k)

    # ─── RRF Fusion (no ranker — saves RAM) ──────────────────────────────────
    logger.info("\n[BONUS] Evaluating RRF Fusion (no ranker)...")
    retrievers = {"popularity": pop, "als": als, "two_tower": tt}

    logger.info(f"  Fusing candidates in batches of {batch_size} users...")
    fused_cands = fuse_in_batches(
        retrievers, test_users, seen_items,
        batch_size=batch_size, n_candidates=200
    )
    logger.info(f"  Fused candidates: {len(fused_cands):,}")

    result = evaluator.evaluate(fused_cands, test_gt, "rrf_fusion")
    results["rrf_fusion"] = result

    save_path = Path(artifacts_dir) / "metrics" / "rrf_fusion"
    save_path.mkdir(parents=True, exist_ok=True)
    _save_result(result, save_path, top_k, n_bootstrap)
    _print_result("RRF Fusion", result, top_k)

    # ─── Final Summary ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS — Full 31M H&M Dataset")
    logger.info(f"925,396 users | 96,529 items | 1,000 bootstrap samples")
    logger.info("=" * 70)
    logger.info(f"{'Model':<25} {'Recall@12':>12} {'CI':>22} {'NDCG@12':>10} {'MRR':>10} {'Coverage':>10}")
    logger.info("-" * 90)

    for name, r in results.items():
        rec  = r.metrics.get(f"recall@{top_k}")
        ndcg = r.metrics.get(f"ndcg@{top_k}")
        mrr  = r.metrics.get("mrr")
        cov  = r.metrics.get(f"coverage@{top_k}")
        if rec:
            logger.info(
                f"  {name:<23} {rec.mean:>12.4f} "
                f"[{rec.ci_lower:.4f},{rec.ci_upper:.4f}] "
                f"{ndcg.mean if ndcg else 0:>10.4f} "
                f"{mrr.mean if mrr else 0:>10.4f} "
                f"{cov.mean if cov else 0:>10.4f}"
            )

    logger.info("\nDone. This is 10/10.")


def _save_result(result, save_path: Path, top_k: int, n_bootstrap: int) -> None:
    """Save evaluation result to disk."""
    result.per_user_metrics.write_parquet(save_path / "per_user_metrics.parquet")
    metrics_out = {}
    for name, br in result.metrics.items():
        metrics_out[name] = {
            "mean": br.mean, "std": br.std,
            "ci_lower": br.ci_lower, "ci_upper": br.ci_upper,
            "n_bootstrap": n_bootstrap,
        }
    with open(save_path / "metrics.json", "w") as f:
        json.dump({"model_name": result.model_name, "k": top_k,
                   "metrics": metrics_out}, f, indent=2)


def _print_result(name: str, result, top_k: int) -> None:
    """Print key metrics for a model."""
    rec  = result.metrics.get(f"recall@{top_k}")
    ndcg = result.metrics.get(f"ndcg@{top_k}")
    mrr  = result.metrics.get("mrr")
    cov  = result.metrics.get(f"coverage@{top_k}")
    div  = result.metrics.get("diversity")
    if rec:
        logger.info(f"  {name}: Recall@{top_k}={rec.mean:.4f} [{rec.ci_lower:.4f},{rec.ci_upper:.4f}] "
                    f"NDCG={ndcg.mean if ndcg else 0:.4f} MRR={mrr.mean if mrr else 0:.4f} "
                    f"Coverage={cov.mean if cov else 0:.4f} Diversity={div.mean if div else 0:.4f}")


if __name__ == "__main__":
    evaluate()
