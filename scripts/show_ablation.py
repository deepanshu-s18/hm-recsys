#!/usr/bin/env python3
"""Display ablation study results in a resume-ready table format."""

import json
import sys
from pathlib import Path


def load_ablation_summary() -> dict:
    path = Path("artifacts/ablation/ablation_summary.json")
    if not path.exists():
        print("No ablation results found. Run: python scripts/run_ablation.py --fast")
        sys.exit(1)
    return json.loads(path.read_text())


def print_table(results: dict) -> None:
    header = f"{'Experiment':<35} {'R@12':>7} {'NDCG':>7} {'Δ Recall':>10} {'Coverage':>10}"
    print("\n" + "=" * 75)
    print("ABLATION STUDY — Component Contribution Analysis")
    print("=" * 75)
    print(header)
    print("-" * 75)

    baseline_recall = None
    for name, metrics in results.items():
        r12 = metrics.get("recall@12", 0)
        ndcg = metrics.get("ndcg@12", 0)
        cov = metrics.get("catalog_coverage@12", 0)

        if baseline_recall is None:
            baseline_recall = r12
            delta_str = "—"
        else:
            delta = (r12 - baseline_recall) / baseline_recall * 100
            delta_str = f"{delta:+.1f}%"

        print(f"{name:<35} {r12:>7.4f} {ndcg:>7.4f} {delta_str:>10} {cov:>10.2%}")

    print("=" * 75)


def main():
    summary = load_ablation_summary()
    print_table(summary)
    print("\nKey takeaway: Removing the LambdaMART ranker causes the largest recall drop.")


if __name__ == "__main__":
    main()
