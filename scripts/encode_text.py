#!/usr/bin/env python3
"""Encode H&M article text descriptions using sentence-transformers.

Creates a 384-dim embedding for each article by concatenating:
    prod_name + product_type_name + colour_group_name +
    department_name + garment_group_name + detail_desc

Saves:
    data/text_embeddings.npy      -- shape (n_articles, 384)
    data/text_article_ids.npy     -- article_id strings in same order

Usage:
    python scripts/encode_text.py
    python scripts/encode_text.py --model all-MiniLM-L6-v2  # faster
    python scripts/encode_text.py --model all-mpnet-base-v2  # stronger
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
import typer
from loguru import logger
from sentence_transformers import SentenceTransformer

app = typer.Typer()


def build_article_text(df: pl.DataFrame) -> list[str]:
    """Concatenate article fields into a single descriptive sentence.

    Args:
        df: Articles DataFrame with H&M columns.

    Returns:
        List of text strings, one per article.
    """
    texts = []
    for row in df.iter_rows(named=True):
        parts = []

        name = row.get("prod_name", "") or ""
        if name:
            parts.append(name)

        ptype = row.get("product_type_name", "") or ""
        if ptype and ptype != name:
            parts.append(ptype)

        colour = row.get("colour_group_name", "") or ""
        if colour:
            parts.append(colour)

        dept = row.get("department_name", "") or ""
        if dept:
            parts.append(dept)

        garment = row.get("garment_group_name", "") or ""
        if garment:
            parts.append(garment)

        desc = row.get("detail_desc", "") or ""
        if desc:
            parts.append(desc)

        text = ". ".join(parts) if parts else "fashion item"
        texts.append(text)

    return texts


@app.command()
def encode(
    articles_path: str = typer.Option("data/raw/articles.csv", help="Path to articles.csv"),
    output_dir: str = typer.Option("data", help="Output directory"),
    model_name: str = typer.Option("all-MiniLM-L6-v2", help="SentenceTransformer model"),
    batch_size: int = typer.Option(256, help="Encoding batch size"),
    device: str = typer.Option("cpu", help="Torch device"),
) -> None:
    """Encode all article descriptions into dense text embeddings."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading articles from {articles_path}...")
    df = pl.read_csv(articles_path)
    logger.info(f"Loaded {len(df)} articles")

    logger.info("Building text representations...")
    texts = build_article_text(df)

    sample_texts = texts[:3]
    for i, t in enumerate(sample_texts):
        logger.info(f"  Sample {i+1}: {t[:120]}")

    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {embedding_dim}")

    logger.info(f"Encoding {len(texts)} articles (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
        device=device,
    )

    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Norm check (first 5): {np.linalg.norm(embeddings[:5], axis=1)}")

    # Save embeddings and corresponding article IDs
    emb_path = output_path / "text_embeddings.npy"
    ids_path = output_path / "text_article_ids.npy"

    np.save(emb_path, embeddings.astype(np.float32))
    article_ids = df["article_id"].cast(pl.Utf8).to_numpy()
    np.save(ids_path, article_ids)

    logger.info(f"Saved embeddings → {emb_path}")
    logger.info(f"Saved article IDs → {ids_path}")
    logger.info(f"Done. Shape: {embeddings.shape}, dtype: {embeddings.dtype}")


if __name__ == "__main__":
    app()
