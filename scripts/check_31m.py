#!/usr/bin/env python3
"""
Cross-check every 31M number in the interview guide against real artifacts.

Run from your hm-recsys directory AFTER running evaluate_full.py:
    cd ~/hm-recsys
    python scripts/check_31m.py

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
    test  = pl.read_parquet("data/processed/test.parquet")
    with open("data/processed/item2idx.json") as f:
        item2idx = json.load(f)
    with open("data/processed/user2idx.json") as f:
        user2idx = json.load(f)
    print(f"  {G}✓{X}  Data loaded")
    print(f"  {B}ℹ{X}  Train rows: {len(train):,}  (guide: 21,516,428)")
    print(f"  {B}ℹ{X}  Users:      {len(user2idx):,}  (guide: 925,396)")
    print(f"  {B}ℹ{X}  Items:      {len(item2idx):,}  (guide: 96,529)")
    print(f"  {B}ℹ{X}  Test rows:  {len(test):,}  (guide: 4,632,533)")
    is_31m = len(user2idx) > 500_000
    if not is_31m:
        print(f"\n  {R}⚠ WARNING: This looks like the 3M cache, not 31M.{X}")
        print(f"  {R}  Use check_3m.py for 3M numbers.{X}")
        print(f"  {R}  To get 31M artifacts: run evaluate_full.py on the 31M model.{X}")
except Exception as e:
    print(f"  {R}✗  Cannot load data: {e}{X}")
    sys.exit(1)

# ── 1. Dataset stats ──────────────────────────────────────────────
section("1. DATASET STATISTICS (31M run)")
for label, actual, exp, tol in [
    ("Total users",      len(user2idx),            925_396, 1000),
    ("Total items",      len(item2idx),              96_529,  100),
    ("Train interactions", len(train),           21_516_428, 50000),
    ("Test interactions",  len(test),             4_632_533, 10000),
    ("Test users",       test["user_idx"].n_unique(), 503_072, 1000),
]:
    diff = abs(actual - exp)
    pct  = 100 * diff / max(exp, 1)
    col  = G if pct < 1 else (Y if pct < 5 else R)
    sym  = "✓" if pct < 1 else ("⚠" if pct < 5 else "✗")
    print(f"  {col}{sym}{X}  {label:<52} actual={actual:,}  guide={exp:,}  ({pct:.2f}%)")
    if pct < 1: ok += 1
    elif pct < 5: warn += 1
    else: fail += 1

# ── 2. 31M model metrics ──────────────────────────────────────────
section("2. MODEL METRICS (31M run)")

GUIDE_31M = {
    "popularity": {
        "recall@12": 0.0045, "ndcg@12": 0.0046, "mrr": 0.0148,
        "ci_lower_recall": 0.0044, "ci_upper_recall": 0.0046,
    },
    "als": {
        "recall@12": 0.0034, "ndcg@12": 0.0028, "mrr": 0.0071,
        "ci_lower_recall": 0.0033, "ci_upper_recall": 0.0035,
    },
    "two_tower": {
        "recall@12": 0.0012, "ndcg@12": 0.0010, "mrr": 0.0026,
        "ci_lower_recall": 0.0012, "ci_upper_recall": 0.0013,
    },
}

for model, guide in GUIDE_31M.items():
    p = Path(f"artifacts/metrics/{model}/metrics.json")
    if not p.exists():
        skip(f"Model: {model}", "metrics.json not found — run evaluate_full.py")
        continue
    print(f"\n  {W}{model.upper()}{X}")
    with open(p) as f:
        m = json.load(f)["metrics"]
    for key, exp in guide.items():
        if key in ("ci_lower_recall", "ci_upper_recall"):
            bound = "ci_lower" if "lower" in key else "ci_upper"
            val = m.get("recall@12", {}).get(bound)
        else:
            val = m.get(key, {})
            val = val.get("mean") if isinstance(val, dict) else val
        if val is None:
            skip(f"  {key}", "not in metrics.json")
        else:
            # 31M CIs are very tight (±0.0001) — use tighter tolerance
            tol = 0.0001 if "ci_" in key else 0.0003
            chk(f"  {key}", val, exp, tol=tol)

# ── 3. Scaling improvements ───────────────────────────────────────
section("3. SCALING IMPROVEMENTS (3M → 31M, guide claims)")

print(f"\n  {'Claim':<55} {'Actual':>10}  {'Guide':>10}")
print(f"  {'-'*78}")

# Need both 3M and 31M metrics — check if we have them
# Since 31M overwrote 3M, we compare against guide 3M values
guide_3m = {"als": 0.0027, "two_tower": 0.0008}
guide_31m = {"als": 0.0034, "two_tower": 0.0012}
guide_recall_pct = {"als": 25.9, "two_tower": 50.0}
guide_mrr_3m  = {"als": 0.0040, "two_tower": 0.0011}
guide_mrr_31m = {"als": 0.0071, "two_tower": 0.0026}
guide_mrr_pct = {"als": 77.5,   "two_tower": 136.0}

for model in ["als", "two_tower"]:
    p = Path(f"artifacts/metrics/{model}/metrics.json")
    if p.exists():
        with open(p) as f:
            m = json.load(f)["metrics"]
        actual_31m_r = m.get("recall@12", {}).get("mean", 0)
        actual_31m_mrr = m.get("mrr", {}).get("mean", 0)

        # Recall improvement vs guide 3M baseline
        act_r_pct   = 100*(actual_31m_r   - guide_3m[model])   / max(guide_3m[model],   1e-9)
        act_mrr_pct = 100*(actual_31m_mrr - guide_mrr_3m[model]) / max(guide_mrr_3m[model], 1e-9)

        for label, actual_pct, exp_pct in [
            (f"{model} Recall improvement vs 3M", act_r_pct,   guide_recall_pct[model]),
            (f"{model} MRR improvement vs 3M",    act_mrr_pct, guide_mrr_pct[model]),
        ]:
            diff = abs(actual_pct - exp_pct)
            col  = G if diff < 5 else (Y if diff < 15 else R)
            sym  = "✓" if diff < 5 else ("⚠" if diff < 15 else "✗")
            print(f"  {col}{sym}{X}  {label:<53} actual={actual_pct:+.1f}%  guide={exp_pct:+.1f}%")
            if diff < 5: ok += 1
            elif diff < 15: warn += 1
            else: fail += 1
    else:
        skip(f"{model} scaling", "run evaluate_full.py first")

# ── 4. Model config ───────────────────────────────────────────────
section("4. MODEL CONFIG (same for both runs)")
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

# ── 5. Embedding shapes ───────────────────────────────────────────
section("5. EMBEDDING SHAPES (31M run)")

EXPECTED_SHAPES = {
    "ALS user_factors":   ("artifacts/models/als/user_factors.npy",          925_396, 128),
    "ALS item_factors":   ("artifacts/models/als/item_factors.npy",            96_529, 128),
    "TT user_embeddings": ("artifacts/models/two_tower/user_embeddings.npy",  925_396, 128),
    "TT item_embeddings": ("artifacts/models/two_tower/item_embeddings.npy",   96_529, 128),
}

for name, (path, exp_rows, exp_dim) in EXPECTED_SHAPES.items():
    p = Path(path)
    if p.exists():
        arr  = np.load(p)
        rows_ok = abs(arr.shape[0] - exp_rows) < 1000
        dim_ok  = arr.shape[1] == exp_dim
        norm    = np.allclose(np.linalg.norm(arr[:5], axis=1), 1.0, atol=0.01) if "emb" in name.lower() else None
        col     = G if (rows_ok and dim_ok) else R
        sym     = "✓" if (rows_ok and dim_ok) else "✗"
        norm_str = f"  L2-norm={'✓' if norm else '✗'}" if norm is not None else "  (raw)"
        print(f"  {col}{sym}{X}  {name:<40} shape={arr.shape}  "
              f"rows={'✓' if rows_ok else '✗'}  dim={'✓' if dim_ok else '✗'}{norm_str}")
        if rows_ok and dim_ok: ok += 1
        else: fail += 1
    else:
        skip(name, f"{path} not found")

# ── 6. Training time ──────────────────────────────────────────────
section("6. TRAINING TIMES (31M run)")
rt_p = Path("artifacts/runtime.json")
if rt_p.exists():
    with open(rt_p) as f:
        rt = json.load(f)
    als_time = rt.get("als", {}).get("seconds", 0)
    tt_time  = rt.get("two_tower", {}).get("seconds", 0)
    info("ALS training time",       f"{als_time:.0f}s ({als_time/60:.1f} min)", "(guide: 822s / 13.7 min)")
    info("Two-Tower training time", f"{tt_time:.0f}s ({tt_time/3600:.1f} hr)",  "(guide: 247,773s / 69 hr)")
else:
    skip("Training times", "artifacts/runtime.json not found")

# ── Summary ───────────────────────────────────────────────────────
section("SUMMARY — 31M RUN")
total = ok + fail + warn
print(f"""
  {G}✓  Verified:   {ok}{X}
  {R}✗  Wrong:      {fail}{X}
  {Y}⚠  Skipped:    {warn}{X}
  {'─'*30}
  Total checks:   {total}
""")
if fail == 0 and warn == 0:
    print(f"  {G}{W}ALL 31M NUMBERS VERIFIED. Safe to say every number in the guide.{X}")
elif fail == 0:
    print(f"  {Y}{W}No wrong numbers. {warn} artifact(s) missing.{X}")
    print(f"  {Y}  Run: python scripts/evaluate_full.py{X}")
else:
    print(f"  {R}{W}{fail} WRONG NUMBER(S). Fix before interview.{X}")
