#!/usr/bin/env python3
"""Identify which Kaggle image subfolders to download for top-15K items.

H&M images are stored as:
    images/010/0108775015.jpg   (first 3 digits = subfolder name)

This script:
1. Finds the top-15K most popular article IDs from training data
2. Identifies which subfolders contain those images
3. Prints exact download instructions

Usage:
    python scripts/identify_image_folders.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np


def main():
    # Load training data
    print("Loading training data...")
    train = pl.read_parquet("data/processed/train.parquet")

    # Get top-15K most popular article IDs
    pop = (
        train.group_by("article_id")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(15000)
    )
    top_ids = pop["article_id"].cast(pl.Utf8).to_list()
    print(f"Top-15K article IDs identified: {len(top_ids)}")

    # Determine which subfolders are needed
    subfolders = set()
    for aid in top_ids:
        aid_str = str(aid).zfill(10)
        prefix = aid_str[:3]
        subfolders.add(prefix)

    subfolders = sorted(subfolders)
    print(f"\nImage subfolders needed: {len(subfolders)} folders")
    print(f"Folders: {subfolders}")

    # Estimate size
    avg_images_per_folder = 15000 / len(subfolders)
    est_size_gb = len(subfolders) * avg_images_per_folder * 0.05 / 1000  # ~50KB per image
    print(f"\nEstimated download size: ~{est_size_gb:.1f} GB")

    # Save top IDs for reference
    np.save("data/top_15k_article_ids.npy", np.array(top_ids))
    print(f"\nSaved top IDs to data/top_15k_article_ids.npy")

    print("\n" + "="*60)
    print("DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\n1. Go to Kaggle in your browser:")
    print("   https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data")
    print("\n2. Click on the 'images' folder")
    print("\n3. Download ONLY these subfolders (click each, then download):")
    for folder in subfolders:
        print(f"   images/{folder}/")
    print(f"\n4. Create the images directory:")
    print(f"   mkdir -p ~/hm-recsys/data/raw/images")
    print(f"\n5. Move downloaded folders there:")
    print(f"   mv ~/Downloads/images/* ~/hm-recsys/data/raw/images/")
    print("\n6. Then run:")
    print("   python scripts/encode_images.py --top-n 15000")
    print("\n" + "="*60)

    # Also try kaggle API if available
    print("\nAlternatively, if Kaggle API works, run:")
    for folder in subfolders:
        print(f"  kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p images/{folder}")


if __name__ == "__main__":
    main()
