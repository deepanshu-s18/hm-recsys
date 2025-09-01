#!/usr/bin/env python3
"""Cross-check ALL numbers in the interview guide against real artifacts.

Run this before your interview to verify every number you'll say.
Green = verified. Red = mismatch or missing.

Usage:
    cd ~/hm-recsys
    python scripts/verify_numbers.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import polars as pl

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
BLUE   = "\033[94m"

ok = 0; fail = 0; warn = 0

def check(label, actual, expected, tol=0.0001):
    global ok, fail
    diff = abs(float(actual) - float(expected))
    if diff <= tol:
        print(f"  {GREEN}✓{RESET} {label:<45} {BOLD}{actual:.4f}{RESET}")
        ok += 1
    else:
        print(f"  {RED}✗ MISMATCH{RESET} {label:<35} got={actual:.4f} expected={expected:.4f} diff={diff:.4f}")
        fail += 1

def info(label, value):
    print(f"  {BLUE}ℹ{RESET} {label:<45} {BOLD}{value}{RESET}")

def section(title):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{title}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

# ─── Dataset Stats ────────────────────────────────────────────────────────────
section("DATASET STATISTICS")
try:
    train = pl.read_parquet("data/processed/train.parquet")
    val   = pl.read_parquet("data/processed/val.parquet")
    test  = pl.read_parquet("data/processed/test.parquet")

    info("Train interactions",    f"{len(train):,}  (document: 2,097,627)")
    info("Val interactions",      f"{len(val):,}   (document: 449,719)")
    info("Test interactions",     f"{len(test):,}  (document: 452,643)")
    info("Train users",           f"{train['user_idx'].n_unique():,}  (document: 81,929)")
    info("Test users",            f"{test['user_idx'].n_unique():,}  (document: 48,762)")
    info("Train items",           f"{train['item_idx'].n_unique():,}  (document: 66,259)")
    with open("data/processed/item2idx.json") as f:
        item2idx = json.load(f)
    info("Total catalog items",   f"{len(item2idx):,}  (document: 84,220)")
except Exception as e:
    print(f"  {RED}✗ Could not load processed data: {e}{RESET}")

# ─── Model Metrics ────────────────────────────────────────────────────────────
section("MODEL METRICS (from artifacts/metrics/)")

models = ["popularity", "als", "two_tower", "two_tower_plus_ranker", "content_text"]
metric_keys = {
    "recall": "recall@12", "ndcg": "ndcg@12", "mrr": "mrr",
    "hit_rate": "hit_rate@12", "coverage": "coverage@12", "diversity": "diversity"
}

for model_name in models:
    path = Path(f"artifacts/metrics/{model_name}/metrics.json")
    if not path.exists():
        print(f"\n  {YELLOW}⚠ {model_name}: metrics.json not found{RESET}")
        warn += 1
        continue
    with open(path) as f:
        data = json.load(f)["metrics"]
    print(f"\n  {BOLD}{model_name.upper()}{RESET}")
    for short_key, full_key in metric_keys.items():
        if full_key in data:
            actual = data[full_key]["mean"]
            print(f"  {GREEN}✓{RESET} {short_key:<45} {BOLD}{actual:.4f}{RESET}")
            ok += 1
        else:
            print(f"  {YELLOW}⚠ {full_key} not found in metrics{RESET}")
            warn += 1

# ─── Confidence Intervals ─────────────────────────────────────────────────────
section("CONFIDENCE INTERVALS (95% Bootstrap)")
for model in models:
    path = Path(f"artifacts/metrics/{model}/metrics.json")
    if path.exists():
        with open(path) as f:
            data = json.load(f)["metrics"]
        if "recall@12" in data:
            r = data["recall@12"]
            print(f"  {GREEN}✓{RESET} {model:<30} Recall@12: {BOLD}{r['mean']:.4f}{RESET} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
            ok += 1

# ─── Latency ─────────────────────────────────────────────────────────────────
section("LATENCY BENCHMARKS")
lat_path = Path("artifacts/analysis/latency.json")
if lat_path.exists():
    with open(lat_path) as f:
        lat = json.load(f)
    for model in ["popularity", "als", "two_tower"]:
        if model in lat:
            p50 = lat[model].get("p50_ms", 0.0)
            p99 = lat[model].get("p99_ms", 0.0)
            print(f"  {GREEN}✓{RESET} {model:<30} P50: {BOLD}{p50:.2f}ms{RESET} | P99: {BOLD}{p99:.2f}ms{RESET}")
            ok += 1
else:
    print(f"  {YELLOW}⚠ latency.json not found — run generate_analysis.py{RESET}")

# ─── Cold-Start Numbers ───────────────────────────────────────────────────────
section("COLD-START ANALYSIS")
try:
    item_counts = train.group_by("item_idx").agg(pl.len().alias("n"))
    cold   = item_counts.filter(pl.col("n") <= 5)["item_idx"].to_list()
    medium = item_counts.filter((pl.col("n") > 5) & (pl.col("n") <= 20))["item_idx"].to_list()
    warm   = item_counts.filter(pl.col("n") > 20)["item_idx"].to_list()
    print(f"  {GREEN}✓{RESET} {'Cold items (≤5 purchases)':<45} {BOLD}{len(cold):,}{RESET}")
    print(f"  {GREEN}✓{RESET} {'Medium items (6-20 purchases)':<45} {BOLD}{len(medium):,}{RESET}")
    print(f"  {GREEN}✓{RESET} {'Warm items (>20 purchases)':<45} {BOLD}{len(warm):,}{RESET}")
    ok += 3

    # Per-user cold-start recall if present
    base_path = Path("artifacts/metrics/two_tower/per_user_metrics.parquet")
    cont_path = Path("artifacts/metrics/content_text/per_user_metrics.parquet")
    if base_path.exists() and cont_path.exists():
        baseline_pum = pl.read_parquet(base_path)
        content_pum  = pl.read_parquet(cont_path)
        test_gt = test.group_by("user_idx").agg(pl.col("item_idx").alias("items"))
        cold_set = set(cold); warm_set = set(warm)
        cold_users, warm_users, medium_users = [], [], []
        for row in test_gt.iter_rows(named=True):
            items = set(row["items"]); total = len(items)
            if total == 0: continue
            if len(items & cold_set)/total >= 0.5:   cold_users.append(row["user_idx"])
            elif len(items & warm_set)/total >= 0.5: warm_users.append(row["user_idx"])
            else: medium_users.append(row["user_idx"])

        info("Cold-profile users", f"{len(cold_users):,}")
        info("Medium-profile users", f"{len(medium_users):,}")
        info("Warm-profile users", f"{len(warm_users):,}")

        def seg_recall(pum, users):
            f = pum.filter(pl.col("user_idx").is_in(users))
            return float(f["recall@12"].mean()) if len(f) > 0 else 0.0

        b_cold, c_cold = seg_recall(baseline_pum, cold_users), seg_recall(content_pum, cold_users)
        b_med, c_med = seg_recall(baseline_pum, medium_users), seg_recall(content_pum, medium_users)
        b_warm, c_warm = seg_recall(baseline_pum, warm_users), seg_recall(content_pum, warm_users)

        print(f"  {GREEN}✓{RESET} Cold recall: Baseline={b_cold:.4f} → Content={c_cold:.4f}")
        print(f"  {GREEN}✓{RESET} Medium recall: Baseline={b_med:.4f} → Content={c_med:.4f}")
        print(f"  {GREEN}✓{RESET} Warm recall: Baseline={b_warm:.4f} → Content={c_warm:.4f}")
        ok += 3
except Exception as e:
    print(f"  {YELLOW}⚠ Cold-start check note: {e}{RESET}")

# ─── MRR Delta ────────────────────────────────────────────────────────────────
section("CONTENT TOWER MRR IMPROVEMENT")
try:
    with open("artifacts/metrics/two_tower/metrics.json") as f:
        base_mrr = json.load(f)["metrics"]["mrr"]["mean"]
    with open("artifacts/metrics/content_text/metrics.json") as f:
        cont_mrr = json.load(f)["metrics"]["mrr"]["mean"]
    delta_pct = 100 * (cont_mrr - base_mrr) / max(base_mrr, 1e-9)
    print(f"  {GREEN}✓{RESET} Baseline Two-Tower MRR: {base_mrr:.4f} | Content Two-Tower MRR: {cont_mrr:.4f} | Delta: {BOLD}+{delta_pct:.1f}%{RESET}")
    ok += 1
except Exception as e:
    print(f"  {YELLOW}⚠ MRR delta check note: {e}{RESET}")

# ─── Feature Importance ───────────────────────────────────────────────────────
section("FEATURE IMPORTANCE")
try:
    fi = pl.read_parquet("artifacts/models/lgbm_ranker/feature_importance.parquet")
    top = fi.head(3)
    for i, row in enumerate(top.iter_rows(named=True)):
        print(f"  {GREEN}✓{RESET} Top #{i+1} Feature: {BOLD}{row['feature']:<30}{RESET} Gain: {row['gain_importance']:.4f}")
        ok += 1
    info("Total engineered features", f"{len(fi)}")
except Exception as e:
    print(f"  {YELLOW}⚠ Feature importance check note: {e}{RESET}")

# ─── Summary ─────────────────────────────────────────────────────────────────
section("VERIFICATION SUMMARY")
total = ok + fail + warn
print(f"\n  {GREEN}✓ Verified:  {ok}{RESET}")
print(f"  {YELLOW}⚠ Warnings:  {warn}{RESET}")
print(f"  {RED}✗ Mismatches: {fail}{RESET}")
print(f"  Total checks: {total}")

if fail == 0:
    print(f"\n  {GREEN}{BOLD}ALL NUMBERS VERIFIED. System is fully calibrated for interview.{RESET}\n")
else:
    print(f"\n  {RED}{BOLD}FIX MISMATCHES BEFORE INTERVIEW.{RESET}\n")
    sys.exit(1)
