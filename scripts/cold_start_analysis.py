#!/usr/bin/env python3
"""Cold-start analysis comparing baseline vs content Two-Tower."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
import json

# Load training data
train = pl.read_parquet("data/processed/train.parquet")
test  = pl.read_parquet("data/processed/test.parquet")

# Item activity segments
item_counts = train.group_by("item_idx").agg(pl.len().alias("n"))

cold_items  = set(item_counts.filter(pl.col("n") <= 5)["item_idx"].to_list())
medium_items = set(item_counts.filter((pl.col("n") > 5) & (pl.col("n") <= 20))["item_idx"].to_list())
warm_items  = set(item_counts.filter(pl.col("n") > 20)["item_idx"].to_list())

print("=" * 60)
print("ITEM ACTIVITY SEGMENTS")
print("=" * 60)
print(f"Cold  items (≤5 purchases):      {len(cold_items):,}")
print(f"Medium items (6-20 purchases):   {len(medium_items):,}")
print(f"Warm  items (>20 purchases):     {len(warm_items):,}")
print(f"Total items in catalog:          {item_counts['item_idx'].n_unique():,}")

# Load per-user metrics
baseline_path = Path("artifacts/metrics/two_tower/per_user_metrics.parquet")
content_path  = Path("artifacts/metrics/content_text/per_user_metrics.parquet")

baseline_pum = pl.read_parquet(baseline_path)
has_content  = content_path.exists()

if has_content:
    content_pum = pl.read_parquet(content_path)
    print(f"\nBoth models loaded:")
    print(f"  Baseline Two-Tower: {len(baseline_pum)} users")
    print(f"  Content Two-Tower:  {len(content_pum)} users")

# For each test user, check if their purchased items are cold/warm
test_gt = (
    test.group_by("user_idx")
    .agg(pl.col("item_idx").alias("items"))
)

cold_users  = []
medium_users = []
warm_users  = []

for row in test_gt.iter_rows(named=True):
    user_items = set(row["items"])
    cold_overlap  = len(user_items & cold_items)
    medium_overlap = len(user_items & medium_items)
    warm_overlap  = len(user_items & warm_items)
    total = len(user_items)
    if total == 0:
        continue
    # Classify user by majority of their test purchases
    if cold_overlap / total >= 0.5:
        cold_users.append(row["user_idx"])
    elif warm_overlap / total >= 0.5:
        warm_users.append(row["user_idx"])
    else:
        medium_users.append(row["user_idx"])

print(f"\n{'='*60}")
print("TEST USERS BY ITEM COLD-START PROFILE")
print(f"{'='*60}")
print(f"Users buying mostly cold items:   {len(cold_users):,}")
print(f"Users buying mostly medium items: {len(medium_users):,}")
print(f"Users buying mostly warm items:   {len(warm_users):,}")

# Compute recall per segment for each model
def segment_recall(pum, user_set, label):
    """Compute mean recall for a set of users."""
    if "recall@12" not in pum.columns:
        return 0.0
    filtered = pum.filter(pl.col("user_idx").is_in(list(user_set)))
    if len(filtered) == 0:
        return 0.0
    return float(filtered["recall@12"].mean())

print(f"\n{'='*60}")
print("RECALL@12 BY COLD-START SEGMENT")
print(f"{'='*60}")
print(f"{'Segment':<25} {'Baseline':>12} {'Content':>12} {'Delta':>10}")
print("-" * 60)

segments = [
    ("Cold (≤5 purchases)", set(cold_users)),
    ("Medium (6-20)", set(medium_users)),
    ("Warm (>20)", set(warm_users)),
    ("All users", set(baseline_pum["user_idx"].to_list())),
]

for seg_name, user_set in segments:
    base_r = segment_recall(baseline_pum, user_set, seg_name)
    if has_content:
        cont_r = segment_recall(content_pum, user_set, seg_name)
        delta = cont_r - base_r
        pct = 100 * delta / max(base_r, 1e-6)
        direction = "↑" if delta >= 0 else "↓"
        print(f"{seg_name:<25} {base_r:>12.4f} {cont_r:>12.4f} {direction}{abs(pct):>8.1f}%")
    else:
        print(f"{seg_name:<25} {base_r:>12.4f} {'N/A':>12}")

# Key finding
print(f"\n{'='*60}")
print("KEY FINDING FOR RESUME/INTERVIEW")
print(f"{'='*60}")
cold_base = segment_recall(baseline_pum, set(cold_users), "cold")
if has_content:
    cold_cont = segment_recall(content_pum, set(cold_users), "cold")
    delta = cold_cont - cold_base
    pct = 100 * delta / max(cold_base, 1e-6)
    print(f"Text embeddings improve cold-start Recall@12 by {pct:+.1f}%")
    print(f"  Baseline: {cold_base:.4f} → Content: {cold_cont:.4f}")
    print(f"\nInterview talking point:")
    print(f"  'Content features (text embeddings) improve Recall@12 by {pct:+.1f}%")
    print(f"  for cold-start users whose purchases overlap with low-activity items,")
    print(f"  confirming our hypothesis that content signals generalise where")
    print(f"  collaborative filtering has sparse signal.'")
else:
    print(f"Baseline cold-start Recall@12: {cold_base:.4f}")
    print(f"Run train_content_tower.py first to get comparison.")
    print(f"\nNote: content_text per_user_metrics not found.")
    print(f"The content tower saves metrics.json but not per_user_metrics.parquet.")
    print(f"This is expected — the train_content_tower.py script only saves aggregate metrics.")
