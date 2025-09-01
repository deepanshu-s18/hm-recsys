#!/usr/bin/env python3
"""
Cross-check every number in the interview guide against real artifact files.

Run this from your hm-recsys directory:
    cd ~/hm-recsys
    python scripts/crosscheck_guide.py

Green  ✓ = number verified against artifact
Yellow ⚠ = number cannot be verified (no artifact) — use with caution
Red    ✗ = number WRONG — do not say this in the interview
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

# ── Colour helpers ────────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
B = "\033[94m"; W = "\033[1m";  X = "\033[0m"

ok = fail = warn = 0

def check(label, actual, expected, tol=0.0001, pct=False):
    global ok, fail
    a, e = float(actual), float(expected)
    diff = abs(a - e)
    threshold = abs(e) * tol if pct else tol
    if diff <= threshold:
        print(f"  {G}✓{X} {label:<55} {W}{a:.4f}{X}")
        ok += 1
    else:
        print(f"  {R}✗ WRONG{X}  {label:<47} got={a:.4f}  expected={e:.4f}  diff={diff:.4f}")
        fail += 1

def info(label, value, expected_note=""):
    print(f"  {B}ℹ{X}  {label:<55} {W}{value}{X}  {expected_note}")

def skip(label, reason):
    global warn
    print(f"  {Y}⚠{X}  {label:<55} {Y}SKIP — {reason}{X}")
    warn += 1

def section(title):
    print(f"\n{W}{'═'*70}{X}")
    print(f"{W}{B}{title}{X}")
    print(f"{W}{'═'*70}{X}")

# ── Load data ─────────────────────────────────────────────────────────────────
section("LOADING CACHED DATA")

try:
    train = pl.read_parquet("data/processed/train.parquet")
    val   = pl.read_parquet("data/processed/val.parquet")
    test  = pl.read_parquet("data/processed/test.parquet")
    with open("data/processed/user2idx.json") as f:
        user2idx = json.load(f)
    with open("data/processed/item2idx.json") as f:
        item2idx = json.load(f)
    print(f"  {G}✓{X}  Data loaded successfully")
except Exception as e:
    print(f"  {R}✗  Could not load data: {e}{X}")
    print(f"  Run the full pipeline first: python scripts/train.py --n-interactions 3000000")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
# 1. DATASET STATISTICS
# ════════════════════════════════════════════════════════════════════════════
section("1. DATASET STATISTICS  (guide claims these numbers)")

n_train_users = train["user_idx"].n_unique()
n_train_items = train["item_idx"].n_unique()
n_catalog     = len(item2idx)
n_users_total = len(user2idx)
n_test_users  = test["user_idx"].n_unique()

print(f"\n  {'Label':<55} {'Actual':>12}  {'Guide says':>12}")
print(f"  {'-'*80}")

rows = [
    ("Train interactions",        len(train),    2_097_627),
    ("Val interactions",          len(val),      15_014,   ),
    ("Test interactions",         len(test),     15_027,   ),
    ("Unique users (user2idx)",   n_users_total, 2_630     ),
    ("Unique catalog items",      n_catalog,     21_422    ),   # guide says 84,220 for 3M
    ("Test users",                n_test_users,  2_379     ),
]

for label, actual, expected in rows:
    diff = abs(actual - expected)
    pct  = 100 * diff / max(expected, 1)
    col  = G if pct < 5 else (Y if pct < 20 else R)
    sym  = "✓" if pct < 5 else ("⚠" if pct < 20 else "✗")
    print(f"  {col}{sym}{X}  {label:<53} {actual:>12,}  {expected:>12,}  (diff={pct:.1f}%)")

print(f"""
  {Y}NOTE:{X} The guide quotes 3M-run numbers:
    90,246 users | 84,220 items | 2,097,627 train | 48,762 test users
  These are from a larger sample run. Check which run your artifacts are from.
  The numbers above are from the currently loaded cache.
""")

# ════════════════════════════════════════════════════════════════════════════
# 2. MODEL METRICS
# ════════════════════════════════════════════════════════════════════════════
section("2. MODEL METRICS  (guide's Master Numbers table)")

GUIDE_METRICS = {
    "popularity": {
        "recall@12": 0.0050, "ndcg@12": 0.0048, "mrr": 0.0115,
        "coverage@12": 0.0002, "diversity": 0.045,
        "ci_lower_recall": 0.0046, "ci_upper_recall": 0.0054,
    },
    "als": {
        "recall@12": 0.0027, "ndcg@12": 0.0021, "mrr": 0.0040,
        "coverage@12": 0.527, "diversity": 0.990,
        "ci_lower_recall": 0.0023, "ci_upper_recall": 0.0030,
    },
    "two_tower": {
        "recall@12": 0.0008, "ndcg@12": 0.0006, "mrr": 0.0011,
        "coverage@12": 0.932, "diversity": 1.000,
        "ci_lower_recall": 0.0007, "ci_upper_recall": 0.0010,
    },
    "two_tower_plus_ranker": {
        "recall@12": 0.0037, "ndcg@12": 0.0028, "mrr": 0.0052,
        "coverage@12": 0.594, "diversity": 0.993,
        "ci_lower_recall": 0.0033, "ci_upper_recall": 0.0040,
    },
    "content_text": {
        "recall@12": 0.0007, "mrr": 0.0018,
        "ci_lower_recall": 0.0005, "ci_upper_recall": 0.0008,
    },
    "gated_content_two_tower": {
        "recall@12": 0.0005, "mrr": 0.0013,
        "ci_lower_recall": 0.0004, "ci_upper_recall": 0.0007,
    },
}

METRIC_MAP = {
    "recall@12":      "recall@12",
    "ndcg@12":        "ndcg@12",
    "mrr":            "mrr",
    "coverage@12":    "coverage@12",
    "diversity":      "diversity",
    "ci_lower_recall":"ci_lower",
    "ci_upper_recall":"ci_upper",
}

for model_name, guide_vals in GUIDE_METRICS.items():
    path = Path(f"artifacts/metrics/{model_name}/metrics.json")
    if not path.exists():
        skip(f"Model: {model_name}", f"metrics.json not found at {path}")
        continue

    with open(path) as f:
        data = json.load(f)["metrics"]

    print(f"\n  {W}{model_name.upper()}{X}")
    for guide_key, guide_val in guide_vals.items():
        artifact_key = METRIC_MAP.get(guide_key, guide_key)
        if guide_key in ("ci_lower_recall", "ci_upper_recall"):
            metric_data = data.get("recall@12", {})
            artifact_val = metric_data.get(artifact_key)
        else:
            metric_data = data.get(guide_key, {})
            artifact_val = metric_data.get("mean") if isinstance(metric_data, dict) else metric_data

        if artifact_val is None:
            skip(f"  {guide_key}", "not in metrics.json")
        else:
            check(f"  {guide_key}", artifact_val, guide_val)

# ════════════════════════════════════════════════════════════════════════════
# 3. COLD-START NUMBERS
# ════════════════════════════════════════════════════════════════════════════
section("3. COLD-START NUMBERS  (from content experiment)")

try:
    item_counts = train.group_by("item_idx").agg(pl.len().alias("n"))
    cold   = item_counts.filter(pl.col("n") <= 5)["item_idx"].to_list()
    medium = item_counts.filter((pl.col("n") > 5) & (pl.col("n") <= 20))["item_idx"].to_list()
    warm   = item_counts.filter(pl.col("n") > 20)["item_idx"].to_list()

    print(f"\n  {'Label':<55} {'Actual':>10}  {'Guide says':>10}")
    print(f"  {'-'*78}")

    for label, actual, expected in [
        ("Cold items (≤5 purchases)",   len(cold),   25_344),
        ("Medium items (6-20)",         len(medium), 18_551),
        ("Warm items (>20)",            len(warm),   22_364),
    ]:
        diff = abs(actual - expected)
        pct  = 100 * diff / max(expected, 1)
        col  = G if pct < 5 else (Y if pct < 20 else R)
        sym  = "✓" if pct < 5 else ("⚠" if pct < 20 else "✗")
        print(f"  {col}{sym}{X}  {label:<53} {actual:>10,}  {expected:>10,}  (diff={pct:.1f}%)")

    # Cold-start recall comparison
    baseline_pum_path = Path("artifacts/metrics/two_tower/per_user_metrics.parquet")
    content_pum_path  = Path("artifacts/metrics/content_text/per_user_metrics.parquet")

    if baseline_pum_path.exists() and content_pum_path.exists():
        baseline_pum = pl.read_parquet(baseline_pum_path)
        content_pum  = pl.read_parquet(content_pum_path)

        test_gt = test.group_by("user_idx").agg(pl.col("item_idx").alias("items"))
        cold_set = set(cold); medium_set = set(medium); warm_set = set(warm)

        cold_users=[]; medium_users=[]; warm_users=[]
        for row in test_gt.iter_rows(named=True):
            items = set(row["items"]); total = len(items)
            if total == 0: continue
            if len(items & cold_set) / total >= 0.5:   cold_users.append(row["user_idx"])
            elif len(items & warm_set) / total >= 0.5:  warm_users.append(row["user_idx"])
            else:                                        medium_users.append(row["user_idx"])

        def seg_recall(pum, users):
            f = pum.filter(pl.col("user_idx").is_in(users))
            return float(f["recall@12"].mean()) if len(f) > 0 else 0.0

        b_cold   = seg_recall(baseline_pum, cold_users)
        c_cold   = seg_recall(content_pum,  cold_users)
        b_medium = seg_recall(baseline_pum, medium_users)
        c_medium = seg_recall(content_pum,  medium_users)
        b_warm   = seg_recall(baseline_pum, warm_users)
        c_warm   = seg_recall(content_pum,  warm_users)

        print(f"\n  {'Cold-start recall comparison':<55} {'baseline':>10}  {'content':>10}  {'delta%':>8}")
        print(f"  {'-'*88}")
        for label, base, cont, exp_pct in [
            ("Cold  (≤5)",  b_cold,   c_cold,   6.2),
            ("Medium (6-20)", b_medium, c_medium, 15.3),
            ("Warm  (>20)",  b_warm,   c_warm,  -39.9),
        ]:
            actual_pct = 100 * (cont - base) / max(base, 1e-9)
            diff = abs(actual_pct - exp_pct)
            col  = G if diff < 3 else (Y if diff < 10 else R)
            sym  = "✓" if diff < 3 else ("⚠" if diff < 10 else "✗")
            print(f"  {col}{sym}{X}  {label:<53} {base:>10.4f}  {cont:>10.4f}  {actual_pct:>+7.1f}%  (guide:{exp_pct:+.1f}%)")
    else:
        skip("Cold-start per_user_metrics", "run train_content_tower.py first")

except Exception as e:
    skip("Cold-start analysis", str(e))

# ════════════════════════════════════════════════════════════════════════════
# 4. LATENCY
# ════════════════════════════════════════════════════════════════════════════
section("4. LATENCY  (M3 CPU, batch=100 users)")

lat_path = Path("artifacts/analysis/latency.json")
if lat_path.exists():
    with open(lat_path) as f:
        lat = json.load(f)

    GUIDE_LATENCY = {
        "popularity":  {"p50_ms": 0.12, "p99_ms": 1.21},
        "als":         {"p50_ms": 2.02, "p99_ms": 2.76},
        "two_tower":   {"p50_ms": 3.55, "p99_ms": 8.84},
    }

    for model, expected in GUIDE_LATENCY.items():
        if model in lat:
            for metric, exp_val in expected.items():
                actual_val = lat[model].get(metric)
                if actual_val is not None:
                    check(f"{model} {metric}", actual_val, exp_val, tol=0.3)
                else:
                    skip(f"{model} {metric}", "key not in latency.json")
        else:
            skip(model, "not in latency.json")
else:
    skip("Latency", "artifacts/analysis/latency.json not found — run generate_analysis.py")

# ════════════════════════════════════════════════════════════════════════════
# 5. ABLATION NUMBERS
# ════════════════════════════════════════════════════════════════════════════
section("5. ABLATION NUMBERS  (guide claims these deltas)")

ablation_path = Path("artifacts/ablation/ablation_summary.json")
if ablation_path.exists():
    with open(ablation_path) as f:
        ab = json.load(f)

    baseline = ab.get("full_pipeline", {}).get("recall", 0)
    GUIDE_ABLATION = {
        "ablation_no_als":          -0.333,
        "ablation_no_ranker":       -0.303,
        "ablation_no_popularity":   -0.125,
        "ablation_no_two_tower":    -0.045,
        "ablation_100_candidates":  -0.030,
    }
    GUIDE_LABELS = {
        "ablation_no_als":          "Without ALS          (guide: −33.3%)",
        "ablation_no_ranker":       "Without Ranker       (guide: −30.3%)",
        "ablation_no_popularity":   "Without Popularity   (guide: −12.5%)",
        "ablation_no_two_tower":    "Without Two-Tower    (guide: −4.5%)",
        "ablation_100_candidates":  "100 Candidates       (guide: −3.0%)",
    }

    if baseline > 0:
        print(f"\n  Baseline recall (full pipeline): {baseline:.4f}  (guide: 0.0038)")
        for key, exp_delta_frac in GUIDE_ABLATION.items():
            exp_label = GUIDE_LABELS.get(key, key)
            actual_recall = ab.get(key, {}).get("recall", None)
            if actual_recall is not None:
                actual_delta = (actual_recall - baseline) / baseline
                diff = abs(actual_delta - exp_delta_frac)
                col  = G if diff < 0.05 else (Y if diff < 0.15 else R)
                sym  = "✓" if diff < 0.05 else ("⚠" if diff < 0.15 else "✗")
                print(f"  {col}{sym}{X}  {exp_label:<55} actual={100*actual_delta:+.1f}%")
            else:
                skip(exp_label, "not in ablation_summary.json")
    else:
        skip("Ablation baseline", "full_pipeline recall is 0")
else:
    skip("Ablation", "artifacts/ablation/ablation_summary.json not found — run run_ablation.py")

# ════════════════════════════════════════════════════════════════════════════
# 6. MODEL CONFIG
# ════════════════════════════════════════════════════════════════════════════
section("6. MODEL CONFIG  (guide claims these hyperparameters)")

# ALS config
als_cfg_path = Path("artifacts/models/als/config.json")
if als_cfg_path.exists():
    with open(als_cfg_path) as f:
        als_cfg = json.load(f)
    print(f"\n  ALS config:")
    for key, expected in [("factors", 128), ("iterations", 30), ("regularization", 0.01), ("alpha", 40.0)]:
        actual = als_cfg.get(key)
        if actual is not None:
            check(f"  als.{key}", actual, expected, tol=0.001)
        else:
            skip(f"  als.{key}", "not in config.json")
else:
    skip("ALS config", "artifacts/models/als/config.json not found")

# Two-Tower config
tt_cfg_path = Path("artifacts/models/two_tower/config.json")
if tt_cfg_path.exists():
    with open(tt_cfg_path) as f:
        tt_cfg = json.load(f)
    print(f"\n  Two-Tower config:")
    for key, expected in [("embedding_dim", 128), ("num_epochs", 20), ("temperature", 0.07),
                           ("batch_size", 512), ("dropout", 0.2)]:
        actual = tt_cfg.get(key)
        if actual is not None:
            check(f"  two_tower.{key}", actual, expected, tol=0.001)
        else:
            skip(f"  two_tower.{key}", "not in config.json")
else:
    skip("Two-Tower config", "artifacts/models/two_tower/config.json not found")

# LightGBM feature importance
lgbm_fi_path = Path("artifacts/models/lgbm_ranker/feature_importance.parquet")
if lgbm_fi_path.exists():
    fi = pl.read_parquet(lgbm_fi_path)
    top_feature = fi.head(1)["feature"][0]
    top_gain    = fi.head(1)["gain_importance"][0]
    total_gain  = fi["gain_importance"].sum()
    top_pct     = 100 * top_gain / total_gain if total_gain > 0 else 0

    print(f"\n  LightGBM feature importance:")
    info("Top feature name",        top_feature, "(guide: rrf_score)")
    info("Top feature gain %",      f"{top_pct:.1f}%", "(guide: 94.1%)")
    info("Total features",          str(len(fi)), "(guide: 38 features)")

    col = G if top_feature == "retrieval_rrf_score" else R
    sym = "✓" if top_feature == "retrieval_rrf_score" else "✗"
    print(f"  {col}{sym}{X}  Top feature is retrieval_rrf_score: {top_feature == 'retrieval_rrf_score'}")

    diff_pct = abs(top_pct - 94.1)
    col = G if diff_pct < 5 else (Y if diff_pct < 15 else R)
    sym = "✓" if diff_pct < 5 else ("⚠" if diff_pct < 15 else "✗")
    print(f"  {col}{sym}{X}  Top feature gain % = {top_pct:.1f}%  (guide: 94.1%,  diff={diff_pct:.1f}%)")
else:
    skip("LightGBM feature importance", "artifacts/models/lgbm_ranker/feature_importance.parquet not found")

# ════════════════════════════════════════════════════════════════════════════
# 7. EMBEDDING SHAPES
# ════════════════════════════════════════════════════════════════════════════
section("7. EMBEDDING SHAPES")

for name, path, exp_shape in [
    ("ALS user_factors",     "artifacts/models/als/user_factors.npy",          None),
    ("ALS item_factors",     "artifacts/models/als/item_factors.npy",          None),
    ("TT user_embeddings",   "artifacts/models/two_tower/user_embeddings.npy", None),
    ("TT item_embeddings",   "artifacts/models/two_tower/item_embeddings.npy", None),
]:
    p = Path(path)
    if p.exists():
        arr = np.load(p)
        norm_check = np.allclose(np.linalg.norm(arr[:5], axis=1), 1.0, atol=0.01)
        norm_str   = f"L2-norm≈1.0 ✓" if "embeddings" in name.lower() and norm_check else \
                     f"L2-norm≈1.0 ✗ (raw, not normalised)" if "embeddings" in name.lower() else \
                     "raw (not L2-normalised)"
        dim_ok = arr.shape[1] == 128
        col    = G if dim_ok else R
        sym    = "✓" if dim_ok else "✗"
        print(f"  {col}{sym}{X}  {name:<40} shape={arr.shape}  dim128={dim_ok}  {norm_str}")
    else:
        skip(name, f"{path} not found")

# ════════════════════════════════════════════════════════════════════════════
# 8. MRR IMPROVEMENT (content experiment)
# ════════════════════════════════════════════════════════════════════════════
section("8. CONTENT EXPERIMENT — MRR +68.2% claim")

tt_path      = Path("artifacts/metrics/two_tower/metrics.json")
content_path = Path("artifacts/metrics/content_text/metrics.json")

if tt_path.exists() and content_path.exists():
    with open(tt_path) as f:
        tt_mrr = json.load(f)["metrics"].get("mrr", {}).get("mean", 0)
    with open(content_path) as f:
        ct_mrr = json.load(f)["metrics"].get("mrr", {}).get("mean", 0)
    if tt_mrr > 0:
        actual_pct = 100 * (ct_mrr - tt_mrr) / tt_mrr
        check("  MRR improvement %", actual_pct, 68.2, tol=10.0)
        info("  Baseline TT MRR",  f"{tt_mrr:.4f}",  "(guide: 0.0011)")
        info("  Content TT MRR",   f"{ct_mrr:.4f}",  "(guide: 0.0018)")
    else:
        skip("MRR improvement", "two_tower MRR is 0")
else:
    skip("MRR improvement", "metrics.json not found for two_tower or content_text")

# ════════════════════════════════════════════════════════════════════════════
# 9. 31M SCALING CLAIMS
# ════════════════════════════════════════════════════════════════════════════
section("9. 31M SCALING CLAIMS")

print(f"""
  {Y}NOTE:{X} 31M metrics require running evaluate_full.py on the 31M model.
  These are loaded from artifacts/metrics/ — which may be your 3M run results.
  To verify 31M numbers: cd ~/hm-recsys && python scripts/evaluate_full.py
""")

GUIDE_31M = {
    "popularity":  {"recall@12": 0.0045, "mrr": 0.0148},
    "als":         {"recall@12": 0.0034, "mrr": 0.0071},
    "two_tower":   {"recall@12": 0.0012, "mrr": 0.0026},
}
for model, guide_vals in GUIDE_31M.items():
    mp = Path(f"artifacts/metrics/{model}/metrics.json")
    if mp.exists():
        with open(mp) as f:
            m = json.load(f)["metrics"]
        for key, exp_val in guide_vals.items():
            actual = m.get(key, {}).get("mean", 0)
            diff   = abs(actual - exp_val)
            tol    = 0.002
            col    = G if diff < tol else (Y if diff < 0.01 else R)
            sym    = "✓" if diff < tol else ("⚠" if diff < 0.01 else "✗")
            print(f"  {col}{sym}{X}  31M {model}.{key:<20} actual={actual:.4f}  guide={exp_val:.4f}  diff={diff:.4f}")
    else:
        skip(f"31M {model}", "metrics.json not found")

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
section("SUMMARY")

total = ok + fail + warn
print(f"""
  {G}✓  Verified:   {ok}{X}
  {R}✗  Wrong:      {fail}{X}
  {Y}⚠  Skipped:    {warn}{X}
  {'─'*30}
  Total checks:   {total}
""")

if fail == 0 and warn == 0:
    print(f"  {G}{W}ALL NUMBERS VERIFIED. Every claim in the guide matches your artifacts.{X}")
elif fail == 0:
    print(f"  {Y}{W}No wrong numbers found. {warn} checks skipped (missing artifacts).{X}")
    print(f"  {Y}Run the missing scripts above to complete verification.{X}")
else:
    print(f"  {R}{W}⚠ {fail} WRONG NUMBER(S) FOUND. Do not say these in the interview.{X}")
    print(f"  Fix the failing checks before your interview date.")

print(f"""
HOW TO RUN:
  cd ~/hm-recsys
  python scripts/crosscheck_guide.py
""")
