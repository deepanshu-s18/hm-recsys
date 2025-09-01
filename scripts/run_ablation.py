#!/usr/bin/env python3
"""Ablation study — measures contribution of every pipeline component.

Runs 7 experiments by disabling one component at a time, then
prints a comparison table showing exactly what each component contributes.

This produces the table a Principal Applied Scientist wants to see:
    Full pipeline      → Recall@12 = 0.0037 (baseline)
    Without ALS        → Recall@12 = ?      (ALS contribution)
    Without Two-Tower  → Recall@12 = ?      (TT contribution)
    Without Popularity → Recall@12 = ?      (Pop contribution)
    Without Ranker     → Recall@12 = ?      (Ranker contribution)
    100 candidates     → Recall@12 = ?      (Fusion ceiling effect)
    200 candidates     → Recall@12 = ?      (Your actual setup — sanity check)

Runtime: ~20 min per run × 7 = ~2.5 hours total.
Data is cached — no reloading. Models use reduced iterations for speed.

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --fast   # 10 min per run, less accurate
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import argparse

EXPERIMENTS = [
    {
        "name": "full_pipeline",
        "label": "Full Pipeline (baseline)",
        "extra_args": [],
        "description": "All 3 retrievers + LightGBM ranker",
    },
    {
        "name": "ablation_no_als",
        "label": "Without ALS",
        "extra_args": ["--no-use-als"],
        "description": "Popularity + Two-Tower + Ranker",
    },
    {
        "name": "ablation_no_two_tower",
        "label": "Without Two-Tower",
        "extra_args": ["--no-use-two-tower"],
        "description": "Popularity + ALS + Ranker",
    },
    {
        "name": "ablation_no_popularity",
        "label": "Without Popularity",
        "extra_args": ["--no-use-popularity"],
        "description": "ALS + Two-Tower + Ranker",
    },
    {
        "name": "ablation_no_ranker",
        "label": "Without Ranker (RRF only)",
        "extra_args": ["--no-use-ranker"],
        "description": "All 3 retrievers, no re-ranking",
    },
    {
        "name": "ablation_100_candidates",
        "label": "100 Candidates (half)",
        "extra_args": ["--n-candidates", "100"],
        "description": "Full pipeline with smaller candidate set",
    },
    {
        "name": "ablation_popularity_only",
        "label": "Popularity Only (pure baseline)",
        "extra_args": ["--no-use-als", "--no-use-two-tower", "--no-use-ranker"],
        "description": "Just popularity retriever — absolute baseline",
    },
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Ablation Study — Component Contribution Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fast", action="store_true", default=False, help="Faster but less accurate")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Processed cache directory")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/ablation", help="Ablation output directory")
    parser.add_argument("--n-interactions", type=int, default=3_000_000, help="Interaction limit")
    return parser.parse_args()


def run() -> None:
    """Run the full ablation study."""
    args = parse_args()
    fast = args.fast
    data_dir = args.data_dir
    processed_dir = args.processed_dir
    artifacts_dir = args.artifacts_dir
    n_interactions = args.n_interactions

    # Reduced iterations for ablation speed
    als_iter     = 15 if fast else 25
    tt_epochs    = 8  if fast else 15
    lgbm_est     = 100 if fast else 300
    n_bootstrap  = 200 if fast else 500

    print("=" * 70)
    print("ABLATION STUDY — Component Contribution Analysis")
    print(f"Mode: {'FAST (less accurate)' if fast else 'STANDARD'}")
    print(f"Estimated runtime: ~{10 if fast else 20} min × {len(EXPERIMENTS)} = "
          f"~{10*len(EXPERIMENTS) if fast else 20*len(EXPERIMENTS)} min total")
    print("=" * 70)

    results = {}
    start_total = time.time()

    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] {exp['label']}")
        print(f"    {exp['description']}")
        print(f"    Extra args: {exp['extra_args'] or 'none'}")

        exp_artifacts = f"{artifacts_dir}/{exp['name']}"
        start = time.time()

        cmd = [
            sys.executable, "scripts/train.py",
            "--data-dir", data_dir,
            "--processed-dir", processed_dir,
            "--artifacts-dir", exp_artifacts,
            "--n-interactions", str(n_interactions),
            "--als-factors", "128",
            "--als-iterations", str(als_iter),
            "--two-tower-epochs", str(tt_epochs),
            "--two-tower-dim", "128",
            "--lgbm-estimators", str(lgbm_est),
            "--n-candidates", "200",
            "--n-bootstrap", str(n_bootstrap),
            "--log-level", "WARNING",
        ] + exp["extra_args"]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=9000
            )
            elapsed = time.time() - start

            # Parse results from output
            recall = ndcg = mrr = cov = div = None
            for line in result.stdout.split("\n"):
                if "two_tower_plus_ranker" in line and "Recall" in line:
                    # Full pipeline result
                    try:
                        recall = float(line.split("Recall@12=")[1].split(" ")[0])
                        ndcg   = float(line.split("NDCG@12=")[1].split()[0])
                    except Exception:
                        pass
                elif "popularity" in line and "Recall" in line and recall is None:
                    # Fallback to best model result
                    try:
                        recall = float(line.split("Recall@12=")[1].split(" ")[0])
                        ndcg   = float(line.split("NDCG@12=")[1].split()[0])
                    except Exception:
                        pass

            # Load from saved metrics if parsing failed
            if recall is None:
                for model in ["two_tower_plus_ranker", "als", "popularity", "two_tower"]:
                    mp = Path(exp_artifacts) / "metrics" / model / "metrics.json"
                    if mp.exists():
                        with open(mp) as f:
                            m = json.load(f)["metrics"]
                        recall = m.get("recall@12", {}).get("mean", 0)
                        ndcg   = m.get("ndcg@12", {}).get("mean", 0)
                        mrr    = m.get("mrr", {}).get("mean", 0)
                        cov    = m.get("coverage@12", {}).get("mean", 0)
                        div    = m.get("diversity", {}).get("mean", 0)
                        break

            results[exp["name"]] = {
                "label": exp["label"],
                "description": exp["description"],
                "recall": recall or 0,
                "ndcg":   ndcg or 0,
                "mrr":    mrr or 0,
                "coverage": cov or 0,
                "diversity": div or 0,
                "elapsed_min": round(elapsed / 60, 1),
                "returncode": result.returncode,
            }

            status = "✓" if result.returncode == 0 else "✗"
            print(f"    {status} Done in {elapsed/60:.1f} min | "
                  f"Recall@12={recall:.4f}" if recall else f"    {status} Done")

        except subprocess.TimeoutExpired:
            print(f"    ✗ TIMEOUT after 60 min — skipping")
            results[exp["name"]] = {"label": exp["label"], "recall": 0, "timed_out": True}

        except Exception as e:
            print(f"    ✗ FAILED: {e}")
            results[exp["name"]] = {"label": exp["label"], "recall": 0, "error": str(e)}

    # ─── Print Results Table ──────────────────────────────────────────────────
    total_time = (time.time() - start_total) / 60
    baseline_recall = results.get("full_pipeline", {}).get("recall", 0)

    print("\n\n" + "=" * 90)
    print("ABLATION STUDY RESULTS")
    print("=" * 90)
    print(f"{'Experiment':<35} {'R@12':>8} {'NDCG':>8} {'Delta':>8} {'%Chg':>8} {'Coverage':>10} {'Diversity':>10}")
    print("-" * 90)

    for exp in EXPERIMENTS:
        r = results.get(exp["name"], {})
        recall   = r.get("recall", 0)
        ndcg     = r.get("ndcg", 0)
        cov      = r.get("coverage", 0)
        div      = r.get("diversity", 0)
        delta    = recall - baseline_recall
        pct      = 100 * delta / max(baseline_recall, 1e-9)
        is_base  = exp["name"] == "full_pipeline"
        marker   = " ← baseline" if is_base else f" ({pct:+.1f}%)"

        print(f"  {exp['label']:<33} {recall:>8.4f} {ndcg:>8.4f} "
              f"{delta:>+8.4f} {pct:>+7.1f}% {cov:>10.4f} {div:>10.4f}{marker}")

    print("-" * 90)
    print(f"\nTotal runtime: {total_time:.0f} min")

    # ─── Key Insights ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("KEY INSIGHTS FOR INTERVIEW")
    print("=" * 90)

    for exp_name, r in results.items():
        if exp_name == "full_pipeline":
            continue
        recall = r.get("recall", 0)
        delta  = recall - baseline_recall
        pct    = 100 * delta / max(baseline_recall, 1e-9)
        label  = r.get("label", exp_name)
        desc   = r.get("description", "")
        direction = "HURTS" if delta < 0 else "IMPROVES"
        print(f"  {label}: {direction} recall by {abs(pct):.1f}% ({delta:+.4f})")

    # ─── Save Results ─────────────────────────────────────────────────────────
    out = Path(artifacts_dir) / "ablation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out}")
    print("\nRun 'python scripts/show_ablation.py' to see the resume-ready table.")


if __name__ == "__main__":
    run()
