#!/usr/bin/env python3
"""Generate synthetic H&M-format data for testing and CI.

Creates realistic synthetic data with the same schema as the H&M dataset,
enabling full pipeline testing without requiring the Kaggle download.

The synthetic data follows the H&M CSV schema exactly:
    transactions_train.csv: t_dat, customer_id, article_id, price, sales_channel_id
    articles.csv: article_id, product_group_name, colour_group_name, ...
    customers.csv: customer_id, age, club_member_status, ...

Generated data has realistic properties:
    - Power-law item popularity distribution
    - User activity follows log-normal distribution
    - Prices correlated with product group
    - Temporal activity with seasonal patterns

Usage:
    python scripts/generate_synthetic_data.py --n-users 2000 --n-items 5000
    python scripts/generate_synthetic_data.py --output data/raw
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import typer
from loguru import logger

app = typer.Typer()


@app.command()
def generate(
    n_users: int = typer.Option(2000, help="Number of synthetic users"),
    n_items: int = typer.Option(5000, help="Number of synthetic items"),
    n_interactions: int = typer.Option(120_000, help="Number of interactions"),
    output_dir: str = typer.Option("data/raw", help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Generate synthetic H&M-format data."""
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic data: {n_users} users, {n_items} items, {n_interactions} interactions")

    # ─── Articles ────────────────────────────────────────────────────────────
    logger.info("Generating articles.csv...")

    product_groups = [
        "Garment Upper body", "Garment Lower body", "Garment Full body",
        "Accessories", "Underwear", "Shoes", "Bags", "Cosmetics"
    ]
    colours = [
        "Black", "White", "Blue", "Red", "Green", "Yellow", "Pink",
        "Grey", "Brown", "Orange", "Purple", "Navy", "Beige"
    ]
    departments = [
        "Jersey Fancy", "Knitwear", "Trousers", "Shorts", "Skirts",
        "Dresses", "Blouses", "Shirts", "Jackets", "Shoes Adult"
    ]
    sections = ["Ladies", "Men", "Divided", "Kids", "Sport", "H&M Home"]

    articles_df = pd.DataFrame({
        "article_id": [f"{10000000 + i:08d}" for i in range(n_items)],
        "product_group_name": rng.choice(product_groups, size=n_items),
        "colour_group_name": rng.choice(colours, size=n_items),
        "department_name": rng.choice(departments, size=n_items),
        "section_name": rng.choice(sections, size=n_items),
        "index_name": rng.choice(["Ladieswear", "Menswear", "Sport", "Divided"], size=n_items),
        "index_group_name": rng.choice(["Ladieswear", "Menswear", "Baby Sizes 50-98", "Sport"], size=n_items),
    })

    # Correlated prices with product group
    base_prices = {
        "Garment Upper body": 0.05, "Garment Lower body": 0.06,
        "Shoes": 0.10, "Bags": 0.08, "Cosmetics": 0.03,
        "Accessories": 0.04, "Underwear": 0.02, "Garment Full body": 0.07,
    }
    articles_df["price"] = articles_df["product_group_name"].map(
        lambda x: float(base_prices.get(x, 0.05)) * (1 + rng.standard_normal() * 0.3)
    ).clip(lower=0.005, upper=0.3)

    articles_df.to_csv(output_path / "articles.csv", index=False)
    logger.info(f"  articles.csv: {len(articles_df)} rows")

    # ─── Customers ───────────────────────────────────────────────────────────
    logger.info("Generating customers.csv...")

    customers_df = pd.DataFrame({
        "customer_id": [f"cust_{i:06d}" for i in range(n_users)],
        "age": np.clip(rng.normal(35, 12, n_users), 16, 75),
        "club_member_status": rng.choice(
            ["ACTIVE", "PRE-CREATE", "LEFT CLUB"],
            size=n_users,
            p=[0.6, 0.3, 0.1],
        ),
        "fashion_news_frequency": rng.choice(
            ["Regularly", "Monthly", "NONE"],
            size=n_users,
            p=[0.3, 0.3, 0.4],
        ),
        "postal_code": rng.choice([f"PC{i:04d}" for i in range(500)], size=n_users),
    })
    customers_df.to_csv(output_path / "customers.csv", index=False)
    logger.info(f"  customers.csv: {len(customers_df)} rows")

    # ─── Transactions ─────────────────────────────────────────────────────────
    logger.info("Generating transactions_train.csv...")

    # Power-law item popularity (Zipf distribution)
    item_probs = 1.0 / np.arange(1, n_items + 1) ** 0.7
    item_probs /= item_probs.sum()

    # User activity follows log-normal: heavy-tailed (some super users)
    user_activity = np.exp(rng.normal(2.5, 1.2, n_users)).astype(int)
    user_activity = np.clip(user_activity, 3, 200)
    total_interactions_possible = user_activity.sum()

    # Scale to target n_interactions
    scale_factor = n_interactions / total_interactions_possible
    user_activity = np.ceil(user_activity * scale_factor).astype(int)
    user_activity = np.clip(user_activity, 2, 100)

    # Date range: 2 years of data
    start_date = pd.Timestamp("2019-09-01")
    end_date = pd.Timestamp("2020-09-22")
    date_range_days = (end_date - start_date).days

    rows = []
    all_customers = customers_df["customer_id"].tolist()
    all_articles = articles_df["article_id"].tolist()

    for user_i, (cust_id, n_act) in enumerate(zip(all_customers, user_activity)):
        # User-specific item bias: sample favorite categories
        if rng.random() < 0.3:  # 30% focused users
            category_items = rng.choice(n_items, size=min(50, n_items), replace=False)
            user_item_probs = item_probs.copy()
            user_item_probs[category_items] *= 5
            user_item_probs /= user_item_probs.sum()
        else:
            user_item_probs = item_probs

        # Sample items for this user
        n_unique_items = min(int(n_act * rng.uniform(0.7, 1.3)), n_items)
        sampled_items = rng.choice(
            n_items, size=n_unique_items, replace=False, p=user_item_probs
        )

        # Temporal pattern: random days for this user
        user_dates = sorted(
            rng.integers(0, date_range_days, size=n_unique_items)
        )

        for item_idx, day_offset in zip(sampled_items, user_dates):
            purchase_date = start_date + pd.Timedelta(days=int(day_offset))
            article_id = all_articles[item_idx]
            price_noise = 1 + rng.normal(0, 0.05)
            rows.append({
                "t_dat": purchase_date.strftime("%Y-%m-%d"),
                "customer_id": cust_id,
                "article_id": article_id,
                "price": float(articles_df.iloc[item_idx]["price"]) * price_noise,
                "sales_channel_id": int(rng.choice([1, 2], p=[0.7, 0.3])),
            })

    transactions_df = pd.DataFrame(rows).sort_values("t_dat").reset_index(drop=True)

    # Trim to target size
    if len(transactions_df) > n_interactions:
        transactions_df = transactions_df.head(n_interactions)

    transactions_df.to_csv(output_path / "transactions_train.csv", index=False)

    logger.info(f"  transactions_train.csv: {len(transactions_df):,} rows")
    logger.info(
        f"  Date range: {transactions_df['t_dat'].min()} → {transactions_df['t_dat'].max()}"
    )
    logger.info(
        f"  Unique users: {transactions_df['customer_id'].nunique():,}, "
        f"Unique items: {transactions_df['article_id'].nunique():,}"
    )
    logger.info(f"\nSynthetic data written to: {output_path}")
    logger.info("Ready to run: python scripts/train.py")


if __name__ == "__main__":
    app()
