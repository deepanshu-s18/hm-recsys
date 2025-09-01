#!/usr/bin/env python3
"""
Cross-check every 3M number in the interview guide against real artifacts.

Run from your hm-recsys directory AFTER running the 3M pipeline:
    cd ~/hm-recsys
    python scripts/check_3m.py

Green  ✓ = verified
Yellow ⚠ = artifact missing
Red    ✗ = WRONG — do not say this number in the interview
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
B = "\033[94m"; W = "\033[1m";  X = "\033[0m"

ok = fail = warn = 0

def chk(label, actual, expected, tol=0.0002):
    global ok, fail
    diff = abs(float(actual) - float(expected))
    if diff <= tol:
        print(f"  {G}✓{X} {label:<52} {W}{float(actual):.4f}{X}  (guide={expected})")
        ok += 1
    else:
        print(f"  {R}✗ WRONG{X}  {label:<44} got={float(actual):.4f}  expected={expected}  diff={diff:.4f}")
        fail += 1

def skip(label, reason):
    global warn
    print(f"  {Y}⚠{X}  {label:<52} {Y}{reason}{X}")
    warn += 1

def info(label, val, note=""):
    print(f"  {B}ℹ{X}  {label:<52} {W}{val}{X}  {note}")

def section(t):
    print(f"\n{W}{'═'*68}{X}\n{W}{B}{t}{X}\n{W}{'═'*68}{X}")

# ── Load ──────────────────────────────────────────────────────────
section("LOADING DATA")
try:
    train = pl.read_parquet("data/processed/train.parquet")
    val   = pl.read_parquet("data/processed/val.parquet")
    test  = pl.read_parquet("data/processed/test.parquet")
    with open("data/processed/item2idx.json") as f:
        item2idx = json.load(f)
    with open("data/processed/user2idx.json") as f:
        user2idx = json.load(f)
    print(f"  {G}✓{X}  Data loaded")
except Exception as e:
    print(f"  {R}✗  Cannot load data: {e}{X}")
    sys.exit(1)

# ── 1. Dataset stats ──────────────────────────────────────────────
section("1. DATASET STATISTICS")
chk("Train interactions",        len(train),               2_097_627, tol=100)
chk("Val interactions",          len(val),                   449_719, tol=100)
chk("Test interactions",         len(test),                  452_643, tol=100)
chk("Total users (user2idx)",    len(user2idx),               90_246, tol=10)
chk("Total catalog items",       len(item2idx),               84_220, tol=10)
chk("Test users",                test["user_idx"].n_unique(),  48_762, tol=10)

# ── 2. Model metrics ──────────────────────────────────────────────
section("2. MODEL METRICS (3M run)")

MODELS = {
    "popularity": {
        "recall@12": 0.0050, "ndcg@12": 0.0048, "mrr": 0.0115,
        "diversity": 0.045,
        "ci_lower_recall": 0.0046, "ci_upper_recall": 0.0054,
    },
    "als": {
        "recall@12": 0.0027, "ndcg@12": 0.0021, "mrr": 0.0040,
        "diversity": 0.990,
        "ci_lower_recall": 0.0023, "ci_upper_recall": 0.0030,
    },
    "two_tower": {
        "recall@12": 0.0008, "ndcg@12": 0.0006, "mrr": 0.0011,
        "diversity": 1.000,
        "ci_lower_recall": 0.0007, "ci_upper_recall": 0.0010,
    },
    "two_tower_plus_ranker": {
        "recall@12": 0.0037, "ndcg@12": 0.0028, "mrr": 0.0052,
        "diversity": 0.993,
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

for model, guide in MODELS.items():
    p = Path(f"artifacts/metrics/{model}/metrics.json")
    if not p.exists():
        skip(f"Model: {model}", f"metrics.json not found")
        continue
    print(f"\n  {W}{model.upper()}{X}")
    with open(p) as f:
        m = json.load(f)["metrics"]
    for key, exp in guide.items():
        if key in ("ci_lower_recall", "ci_upper_recall"):
            bound = key.split("_")[1]  # lower or upper
            val = m.get("recall@12", {}).get(f"ci_{bound}")
        else:
            val = m.get(key, {})
            val = val.get("mean") if isinstance(val, dict) else val
        if val is None:
            skip(f"  {key}", "not in metrics.json")
        else:
            chk(f"  {key}", val, exp)

# ── 3. Coverage ───────────────────────────────────────────────────
section("3. COVERAGE (3M run)")
COV = {
    "popularity":             0.0002,
    "als":                    0.527,
    "two_tower":              0.932,
    "two_tower_plus_ranker":  0.594,
}
for model, exp in COV.items():
    p = Path(f"artifacts/metrics/{model}/metrics.json")
    if p.exists():
        with open(p) as f:
            m = json.load(f)["metrics"]
        val = m.get("coverage@12", {})
        val = val.get("mean") if isinstance(val, dict) else val
        if val is not None:
            chk(f"{model} coverage@12", val, exp, tol=0.02)
        else:
            skip(f"{model} coverage@12", "not in metrics.json")
    else:
        skip(f"{model} coverage@12", "metrics.json not found")

# ── 4. Ablation ───────────────────────────────────────────────────
section("4. ABLATION DELTAS (3M run)")
ab = Path("artifacts/ablation/ablation_summary.json")
if ab.exists():
    with open(ab) as f:
        data = json.load(f)
    base = data.get("full_pipeline", {}).get("recall", 0)
    info("Full pipeline baseline", f"{base:.4f}", "(guide: 0.0038)")
    for key, exp_pct, label in [
        ("ablation_no_als",         -33.3, "Without ALS         (guide: -33.3%)"),
        ("ablation_no_ranker",      -30.3, "Without Ranker      (guide: -30.3%)"),
        ("ablation_no_popularity",  -12.5, "Without Popularity  (guide: -12.5%)"),
        ("ablation_no_two_tower",    -4.5, "Without Two-Tower   (guide: -4.5%)"),
        ("ablation_100_candidates",  -3.0, "100 Candidates      (guide: -3.0%)"),
    ]:
        r = data.get(key, {}).get("recall")
        if r and base > 0:
            actual_pct = 100 * (r - base) / base
            diff = abs(actual_pct - exp_pct)
            col = G if diff < 3 else (Y if diff < 8 else R)
            sym = "✓" if diff < 3 else ("⚠" if diff < 8 else "✗")
            print(f"  {col}{sym}{X}  {label:<48} actual={actual_pct:+.1f}%")
            if diff < 3: ok += 1
            elif diff < 8: warn += 1
            else: fail += 1
        else:
            skip(label, "not in ablation_summary.json")
else:
    skip("Ablation", "run: python scripts/run_ablation.py")

# ── 5. Cold-start ─────────────────────────────────────────────────
section("5. COLD-START (3M run)")
ic = train.group_by("item_idx").agg(pl.len().alias("n"))
cold   = ic.filter(pl.col("n") <= 5)["item_idx"].to_list()
medium = ic.filter((pl.col("n") > 5) & (pl.col("n") <= 20))["item_idx"].to_list()
warm   = ic.filter(pl.col("n") > 20)["item_idx"].to_list()

for label, actual, exp in [
    ("Cold items (≤5 purchases)",    len(cold),   25_344),
    ("Medium items (6-20)",          len(medium), 18_551),
    ("Warm items (>20 purchases)",   len(warm),   22_364),
]:
    diff = abs(actual - exp)
    pct  = 100 * diff / max(exp, 1)
    col  = G if pct < 5 else (Y if pct < 15 else R)
    sym  = "✓" if pct < 5 else ("⚠" if pct < 15 else "✗")
    print(f"  {col}{sym}{X}  {label:<52} actual={actual:,}  guide={exp:,}  ({pct:.1f}%)")
    if pct < 5: ok += 1
    elif pct < 15: warn += 1
    else: fail += 1

# Cold-start recall deltas
bp = Path("artifacts/metrics/two_tower/per_user_metrics.parquet")
cp = Path("artifacts/metrics/content_text/per_user_metrics.parquet")
if bp.exists() and cp.exists():
    bpum = pl.read_parquet(bp)
    cpum = pl.read_parquet(cp)
    tgt  = test.group_by("user_idx").agg(pl.col("item_idx").alias("items"))
    cs, ms, ws = set(cold), set(medium), set(warm)
    cu, mu, wu = [], [], []
    for row in tgt.iter_rows(named=True):
        items = set(row["items"]); total = len(items)
        if total == 0: continue
        if len(items & cs) / total >= 0.5:   cu.append(row["user_idx"])
        elif len(items & ws) / total >= 0.5:  wu.append(row["user_idx"])
        else:                                  mu.append(row["user_idx"])

    def sr(pum, users):
        f = pum.filter(pl.col("user_idx").is_in(users))
        return float(f["recall@12"].mean()) if len(f) > 0 else 0.0

    print()
    for label, base, cont, exp_pct in [
        ("Cold  recall delta  (guide: +6.2%)",  sr(bpum, cu), sr(cpum, cu),   6.2),
        ("Medium recall delta (guide: +15.3%)", sr(bpum, mu), sr(cpum, mu),  15.3),
        ("Warm  recall delta  (guide: -39.9%)", sr(bpum, wu), sr(cpum, wu), -39.9),
    ]:
        actual_pct = 100 * (cont - base) / max(base, 1e-9)
        diff = abs(actual_pct - exp_pct)
        col = G if diff < 5 else (Y if diff < 15 else R)
        sym = "✓" if diff < 5 else ("⚠" if diff < 15 else "✗")
        print(f"  {col}{sym}{X}  {label:<52} actual={actual_pct:+.1f}%")
        if diff < 5: ok += 1
        elif diff < 15: warn += 1
        else: fail += 1

    # MRR improvement
    tt_p = Path("artifacts/metrics/two_tower/metrics.json")
    ct_p = Path("artifacts/metrics/content_text/metrics.json")
    if tt_p.exists() and ct_p.exists():
        tt_mrr = json.load(open(tt_p))["metrics"]["mrr"]["mean"]
        ct_mrr = json.load(open(ct_p))["metrics"]["mrr"]["mean"]
        if tt_mrr > 0:
            actual_pct = 100 * (ct_mrr - tt_mrr) / tt_mrr
            diff = abs(actual_pct - 68.2)
            col = G if diff < 10 else (Y if diff < 20 else R)
            sym = "✓" if diff < 10 else ("⚠" if diff < 20 else "✗")
            print(f"  {col}{sym}{X}  MRR improvement content vs baseline  (guide: +68.2%)     actual={actual_pct:+.1f}%")
            if diff < 10: ok += 1
            elif diff < 20: warn += 1
            else: fail += 1
else:
    skip("Cold-start recall deltas", "run: python scripts/train_content_tower.py --use-text")

# ── 6. Latency ────────────────────────────────────────────────────
section("6. LATENCY (M3 CPU, batch=100 users)")
lat_p = Path("artifacts/analysis/latency.json")
if lat_p.exists():
    with open(lat_p) as f:
        lat = json.load(f)
    for model, p50_exp, p99_exp in [
        ("popularity", 0.12, 1.21),
        ("als",        2.02, 2.76),
        ("two_tower",  3.55, 8.84),
    ]:
        if model in lat:
            chk(f"{model} P50 ms/user", lat[model].get("p50_ms", 0), p50_exp, tol=0.5)
            chk(f"{model} P99 ms/user", lat[model].get("p99_ms", 0), p99_exp, tol=2.0)
        else:
            skip(f"{model} latency", "not in latency.json")
else:
    skip("Latency", "run: python scripts/generate_analysis.py")

# ── 7. Model config ───────────────────────────────────────────────
section("7. MODEL CONFIG")
als_p = Path("artifacts/models/als/config.json")
if als_p.exists():
    c = json.load(open(als_p))
    print(f"  {W}ALS{X}")
    for k, v in [("factors",128),("iterations",30),("regularization",0.01),("alpha",40.0)]:
        a = c.get(k)
        if a is not None: chk(f"  als.{k}", a, v, tol=0.001)
        else: skip(f"  als.{k}", "not in config.json")
else:
    skip("ALS config", "not found")

tt_p = Path("artifacts/models/two_tower/config.json")
if tt_p.exists():
    c = json.load(open(tt_p))
    print(f"  {W}Two-Tower{X}")
    for k, v in [("embedding_dim",128),("temperature",0.07),("dropout",0.2)]:
        a = c.get(k)
        if a is not None: chk(f"  tt.{k}", a, v, tol=0.001)
        else: skip(f"  tt.{k}", "not in config.json")
else:
    skip("Two-Tower config", "not found")

# ── 8. Feature importance ─────────────────────────────────────────
section("8. LIGHTGBM FEATURE IMPORTANCE")
fi_p = Path("artifacts/models/lgbm_ranker/feature_importance.parquet")
if fi_p.exists():
    fi    = pl.read_parquet(fi_p)
    top   = fi.head(1)["feature"][0]
    total = fi["gain_importance"].sum()
    pct   = 100 * fi.head(1)["gain_importance"][0] / total if total > 0 else 0
    col   = G if top == "retrieval_rrf_score" else R
    sym   = "✓" if top == "retrieval_rrf_score" else "✗"
    print(f"  {col}{sym}{X}  Top feature: {top}  (guide: retrieval_rrf_score)")
    if top == "retrieval_rrf_score": ok += 1
    else: fail += 1
    diff = abs(pct - 94.1)
    col  = G if diff < 5 else (Y if diff < 15 else R)
    sym  = "✓" if diff < 5 else ("⚠" if diff < 15 else "✗")
    print(f"  {col}{sym}{X}  Top feature gain%: {pct:.1f}%  (guide: 94.1%)")
    if diff < 5: ok += 1
    elif diff < 15: warn += 1
    else: fail += 1
    info("Total features", str(len(fi)), "(guide: 38)")
else:
    skip("Feature importance", "run full pipeline first")

# ── 9. Embeddings ─────────────────────────────────────────────────
section("9. EMBEDDING SHAPES")
for name, path in [
    ("ALS user_factors",    "artifacts/models/als/user_factors.npy"),
    ("ALS item_factors",    "artifacts/models/als/item_factors.npy"),
    ("TT user_embeddings",  "artifacts/models/two_tower/user_embeddings.npy"),
    ("TT item_embeddings",  "artifacts/models/two_tower/item_embeddings.npy"),
]:
    p = Path(path)
    if p.exists():
        arr  = np.load(p)
        d128 = arr.shape[1] == 128
        norm = np.allclose(np.linalg.norm(arr[:5], axis=1), 1.0, atol=0.01) if "emb" in name.lower() else None
        col  = G if d128 else R
        sym  = "✓" if d128 else "✗"
        norm_str = f"  L2-norm=1.0 {'✓' if norm else '✗'}" if norm is not None else "  (raw, not L2-norm)"
        print(f"  {col}{sym}{X}  {name:<40} shape={arr.shape}  dim128={d128}{norm_str}")
        if d128: ok += 1
        else: fail += 1
    else:
        skip(name, f"{path} not found")

# ── Summary ───────────────────────────────────────────────────────
section("SUMMARY — 3M RUN")
total = ok + fail + warn
print(f"""
  {G}✓  Verified:   {ok}{X}
  {R}✗  Wrong:      {fail}{X}
  {Y}⚠  Skipped:    {warn}{X}
  {'─'*30}
  Total checks:   {total}
""")
if fail == 0 and warn == 0:
    print(f"  {G}{W}ALL 3M NUMBERS VERIFIED. Safe to say every number in the guide.{X}")
elif fail == 0:
    print(f"  {Y}{W}No wrong numbers. {warn} artifact(s) missing — run the scripts above.{X}")
else:
    print(f"  {R}{W}{fail} WRONG NUMBER(S). Fix before interview.{X}")
