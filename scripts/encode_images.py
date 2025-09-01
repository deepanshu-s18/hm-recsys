#!/usr/bin/env python3
"""Encode H&M product images using CLIP ViT-B/32.

Downloads the CLIP model (~340MB, one-time), then encodes the top-N
most popular product images into 512-dim visual embeddings.

Images are expected at: data/raw/images/{article_prefix}/{article_id}.jpg
(standard H&M Kaggle dataset structure)

Saves:
    data/clip_embeddings.npy      -- shape (n_encoded, 512)
    data/clip_article_ids.npy     -- article_id strings in same order
    data/clip_coverage.json       -- stats on how many items were encoded

Usage:
    python scripts/encode_images.py
    python scripts/encode_images.py --top-n 15000
    python scripts/encode_images.py --image-dir data/raw/images --top-n 15000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
import torch
import typer
from loguru import logger
from PIL import Image
from tqdm import tqdm

app = typer.Typer()


def load_clip_model(device: str):
    """Load CLIP ViT-B/32 model.

    Args:
        device: Torch device string.

    Returns:
        Tuple of (model, preprocess).
    """
    import clip
    logger.info("Loading CLIP ViT-B/32 (~340MB, downloading if needed)...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    logger.info(f"CLIP loaded on {device}")
    return model, preprocess


def get_popular_article_ids(
    articles_path: str,
    transactions_path: str,
    top_n: int,
) -> list[str]:
    """Get top-N most purchased article IDs.

    Args:
        articles_path: Path to articles.csv.
        transactions_path: Path to transactions_train.csv.
        top_n: Number of top articles to return.

    Returns:
        List of article_id strings.
    """
    logger.info(f"Finding top-{top_n} popular articles...")
    trans = pl.scan_csv(transactions_path).select("article_id").collect()
    counts = (
        trans.group_by("article_id")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(top_n)
    )
    ids = counts["article_id"].cast(pl.Utf8).to_list()
    logger.info(f"Selected {len(ids)} popular articles")
    return ids


def find_image_path(image_dir: Path, article_id: str) -> Path | None:
    """Find image file for an article ID.

    H&M images are stored as:
        images/010/0108775015.jpg   (first 3 digits = subfolder)

    Args:
        image_dir: Root images directory.
        article_id: Article ID string (may be zero-padded).

    Returns:
        Path to image file, or None if not found.
    """
    # Ensure 10-digit zero-padded format
    article_str = str(article_id).zfill(10)
    prefix = article_str[:3]

    candidates = [
        image_dir / prefix / f"{article_str}.jpg",
        image_dir / f"{article_str}.jpg",
        image_dir / prefix / f"{article_id}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@app.command()
def encode(
    image_dir: str = typer.Option("data/raw/images", help="Root image directory"),
    articles_path: str = typer.Option("data/raw/articles.csv", help="Articles CSV"),
    transactions_path: str = typer.Option(
        "data/raw/transactions_train.csv", help="Transactions CSV"
    ),
    output_dir: str = typer.Option("data", help="Output directory for embeddings"),
    top_n: int = typer.Option(15000, help="Encode top-N most popular items"),
    batch_size: int = typer.Option(64, help="Image batch size"),
    device: str = typer.Option("cpu", help="Torch device (cpu/mps)"),
) -> None:
    """Encode top-N H&M product images with CLIP ViT-B/32."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_dir_path = Path(image_dir)

    if not image_dir_path.exists():
        logger.error(
            f"Image directory not found: {image_dir_path}\n"
            "Download images from Kaggle H&M competition data tab.\n"
            "Only the 'images' folder is needed (~2GB for top items)."
        )
        raise typer.Exit(code=1)

    # Get top-N popular articles
    article_ids = get_popular_article_ids(articles_path, transactions_path, top_n)

    # Find which ones have images
    logger.info("Scanning for available images...")
    found = []
    missing = []
    for aid in article_ids:
        p = find_image_path(image_dir_path, aid)
        if p is not None:
            found.append((aid, p))
        else:
            missing.append(aid)

    logger.info(f"Images found: {len(found)}/{len(article_ids)}")
    logger.info(f"Images missing: {len(missing)}")

    if len(found) == 0:
        logger.error("No images found. Check image directory structure.")
        raise typer.Exit(code=1)

    # Load CLIP
    model, preprocess = load_clip_model(device)

    # Encode in batches
    all_embeddings = []
    all_ids = []

    logger.info(f"Encoding {len(found)} images with CLIP ViT-B/32...")
    batch_ids = []
    batch_images = []

    def process_batch(batch_ids, batch_images):
        """Encode one batch and return embeddings."""
        if not batch_images:
            return np.zeros((0, 512), dtype=np.float32), []
        image_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            features = model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize
        return features.cpu().numpy().astype(np.float32), batch_ids

    for aid, img_path in tqdm(found, desc="Encoding images"):
        try:
            img = preprocess(Image.open(img_path).convert("RGB"))
            batch_images.append(img)
            batch_ids.append(aid)
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            continue

        if len(batch_images) >= batch_size:
            embs, ids = process_batch(batch_ids, batch_images)
            all_embeddings.append(embs)
            all_ids.extend(ids)
            batch_ids = []
            batch_images = []

    # Flush remaining
    if batch_images:
        embs, ids = process_batch(batch_ids, batch_images)
        all_embeddings.append(embs)
        all_ids.extend(ids)

    if not all_embeddings:
        logger.error("No embeddings produced. Check image files.")
        raise typer.Exit(code=1)

    embeddings = np.vstack(all_embeddings)
    article_ids_arr = np.array(all_ids)

    logger.info(f"Final embeddings shape: {embeddings.shape}")
    logger.info(f"Norm check (first 5): {np.linalg.norm(embeddings[:5], axis=1)}")

    # Save
    np.save(output_path / "clip_embeddings.npy", embeddings)
    np.save(output_path / "clip_article_ids.npy", article_ids_arr)

    coverage = {
        "total_requested": top_n,
        "images_found": len(found),
        "images_encoded": len(all_ids),
        "images_missing": len(missing),
        "coverage_pct": round(100 * len(all_ids) / top_n, 1),
        "embedding_dim": 512,
        "model": "ViT-B/32",
    }
    with open(output_path / "clip_coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)

    logger.info(f"Coverage: {coverage['coverage_pct']}%")
    logger.info(f"Saved CLIP embeddings → {output_path}/clip_embeddings.npy")
    logger.info(f"Saved article IDs → {output_path}/clip_article_ids.npy")
    logger.info("Done.")


if __name__ == "__main__":
    app()
