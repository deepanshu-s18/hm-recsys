"""Publication-quality visualization for recommendation system analysis.

Generates all figures required for the experiment report:
    1. Recall comparison bar chart with confidence intervals
    2. Bootstrap distribution plots
    3. Learning curves (Two-Tower training)
    4. Feature importance (LightGBM)
    5. Popularity bias analysis
    6. Coverage Venn diagram (retriever complementarity)
    7. Calibration curves
    8. Latency vs Recall Pareto
    9. UMAP embedding visualization
    10. Error breakdown analysis

All figures use a consistent style and color palette inspired by
academic ML publication standards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.utils.logger import get_logger

log = get_logger(__name__)

# ─── Style Configuration ────────────────────────────────────────────────────
PALETTE = {
    "popularity": "#4C72B0",
    "als": "#DD8452",
    "two_tower": "#55A868",
    "two_tower_plus_ranker": "#C44E52",
    "lightfm": "#8172B3",
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
}

FIGURE_DEFAULTS = {
    "figsize": (10, 6),
    "dpi": 150,
    "style": "seaborn-v0_8-paper",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _get_color(name: str) -> str:
    """Get consistent color for a model name.

    Args:
        name: Model/retriever name.

    Returns:
        Hex color string.
    """
    for key, color in PALETTE.items():
        if key in name.lower():
            return color
    return "#7f7f7f"


def plot_recall_comparison(
    results: Dict[str, Dict],
    metric: str = "recall@12",
    output_path: Optional[Path] = None,
    title: str = "Recall@12 Comparison",
) -> Figure:
    """Bar chart comparing recall across models with 95% CIs.

    Args:
        results: Dict mapping model_name → metrics dict with mean/ci.
        metric: Metric key to plot.
        output_path: Optional path to save figure.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 6))

    models = []
    means = []
    errors_low = []
    errors_high = []
    colors = []

    for model_name, model_results in results.items():
        metrics = model_results.get("metrics", {})
        if metric not in metrics:
            continue
        m = metrics[metric]
        models.append(model_name.replace("_", "\n"))
        means.append(m["mean"])
        errors_low.append(m["mean"] - m["ci_lower"])
        errors_high.append(m["ci_upper"] - m["mean"])
        colors.append(_get_color(model_name))

    if not models:
        log.warning(f"No data to plot for metric {metric}")
        return fig

    x = np.arange(len(models))
    bars = ax.bar(
        x, means, color=colors, alpha=0.85, width=0.6, zorder=3
    )
    ax.errorbar(
        x, means,
        yerr=[errors_low, errors_high],
        fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4,
    )

    # Annotate bars with values
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(errors_high) * 0.1,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Add significance annotations (best model marker)
    if means:
        best_idx = int(np.argmax(means))
        ax.text(
            best_idx, means[best_idx] + max(errors_high) * 0.8,
            "★ Best", ha="center", va="bottom", color="gold",
            fontsize=10, fontweight="bold",
        )

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved recall comparison to {output_path}")

    return fig


def plot_bootstrap_distributions(
    bootstrap_data: Dict[str, np.ndarray],
    metric: str = "recall@12",
    output_path: Optional[Path] = None,
) -> Figure:
    """KDE density plots of bootstrap distributions for each model.

    Args:
        bootstrap_data: Dict mapping model_name → bootstrap samples array.
        metric: Metric name for labeling.
        output_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, samples in bootstrap_data.items():
        color = _get_color(model_name)
        ax.hist(
            samples, bins=50, alpha=0.4, color=color, density=True, label=model_name
        )
        # Add KDE
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(samples)
            x = np.linspace(samples.min(), samples.max(), 200)
            ax.plot(x, kde(x), color=color, linewidth=2)
        except Exception:
            pass

    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Density")
    ax.set_title(f"Bootstrap Distribution: {metric}")
    ax.legend(loc="upper left")
    ax.yaxis.grid(True, alpha=0.4)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved bootstrap distributions to {output_path}")

    return fig


def plot_learning_curves(
    training_history: List[Dict],
    output_path: Optional[Path] = None,
) -> Figure:
    """Plot Two-Tower training loss curve over epochs.

    Args:
        training_history: List of {epoch, train_loss, lr} dicts.
        output_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    if not training_history:
        return plt.figure()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = [h["epoch"] for h in training_history]
    losses = [h["train_loss"] for h in training_history]
    lrs = [h.get("lr", 0) for h in training_history]

    ax1.plot(epochs, losses, color=PALETTE["two_tower"], linewidth=2, marker="o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("InfoNCE Loss")
    ax1.set_title("Two-Tower Training Loss")
    ax1.yaxis.grid(True, alpha=0.4)

    ax2.plot(epochs, lrs, color=PALETTE["als"], linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule (Cosine Annealing)")
    ax2.yaxis.grid(True, alpha=0.4)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved learning curves to {output_path}")

    return fig


def plot_feature_importance(
    feature_importance_df,
    top_n: int = 20,
    output_path: Optional[Path] = None,
) -> Figure:
    """Horizontal bar chart of LightGBM feature importances.

    Args:
        feature_importance_df: DataFrame with [feature, gain_importance].
        top_n: Number of top features to display.
        output_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """

    if feature_importance_df is None:
        return plt.figure()

    df = feature_importance_df.head(top_n)
    features = df["feature"].to_list()
    gains = df["gain_importance"].to_list()

    # Color by feature group
    group_colors = {
        "retrieval_": "#4C72B0",
        "item_": "#55A868",
        "user_": "#DD8452",
        "cross_": "#C44E52",
        "temporal_": "#8172B3",
    }
    colors = []
    for feat in features:
        color = "#7f7f7f"
        for prefix, c in group_colors.items():
            if feat.startswith(prefix):
                color = c
                break
        colors.append(color)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    y_pos = np.arange(len(features))

    ax.barh(y_pos, gains, color=colors, alpha=0.85, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Gain Importance")
    ax.set_title(f"Top {top_n} Feature Importances (LightGBM)")
    ax.xaxis.grid(True, alpha=0.4)

    # Legend for feature groups
    legend_patches = [
        mpatches.Patch(color=c, label=prefix.rstrip("_").title())
        for prefix, c in group_colors.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved feature importance to {output_path}")

    return fig


def plot_popularity_bias(
    rec_popularity: np.ndarray,
    gt_popularity: np.ndarray,
    output_path: Optional[Path] = None,
) -> Figure:
    """Compare popularity distribution of recommendations vs ground truth.

    Args:
        rec_popularity: Array of popularity scores for recommended items.
        gt_popularity: Array of popularity scores for ground truth items.
        output_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram comparison
    bins = 30
    ax1.hist(gt_popularity, bins=bins, alpha=0.6, label="Ground Truth",
             color=PALETTE["positive"], density=True)
    ax1.hist(rec_popularity, bins=bins, alpha=0.6, label="Recommended",
             color=PALETTE["als"], density=True)
    ax1.set_xlabel("Item Popularity Score")
    ax1.set_ylabel("Density")
    ax1.set_title("Popularity Distribution: Recs vs Ground Truth")
    ax1.legend()

    # Log-scale cumulative distribution
    gt_sorted = np.sort(gt_popularity)[::-1]
    rec_sorted = np.sort(rec_popularity)[::-1]
    ax2.plot(np.arange(len(gt_sorted)) / len(gt_sorted), gt_sorted,
             label="Ground Truth", color=PALETTE["positive"], linewidth=2)
    ax2.plot(np.arange(len(rec_sorted)) / len(rec_sorted), rec_sorted,
             label="Recommended", color=PALETTE["als"], linewidth=2)
    ax2.set_xlabel("Fraction of Items")
    ax2.set_ylabel("Popularity Score")
    ax2.set_title("Popularity Concentration Curve")
    ax2.legend()
    ax2.set_yscale("log")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved popularity bias to {output_path}")

    return fig


def plot_metrics_heatmap(
    results: Dict[str, Dict],
    metrics: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> Figure:
    """Heatmap of all metrics across all models.

    Args:
        results: Dict mapping model_name → evaluation summary.
        metrics: Optional list of metrics to include.
        output_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    if metrics is None:
        metrics = ["recall@12", "precision@12", "map@12", "mrr", "ndcg@12",
                   "hit_rate@12", "diversity", "personalization"]

    models = list(results.keys())
    data: list[list[float]] = []

    for model in models:
        row = []
        model_metrics = results[model].get("metrics", {})
        for metric in metrics:
            if metric in model_metrics:
                row.append(model_metrics[metric]["mean"])
            else:
                row.append(np.nan)
        data.append(row)

    data_np: np.ndarray = np.array(data)

    fig, ax = plt.subplots(figsize=(14, max(4, len(models) * 1.2)))

    # Normalize each column for visual comparison
    data_norm = data_np.copy()
    for j in range(data_np.shape[1]):
        col = data_np[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0 and valid.max() > valid.min():
            data_norm[:, j] = (col - valid.min()) / (valid.max() - valid.min())

    im = ax.imshow(data_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(metrics)):
            val = data_np[i, j]
            if not np.isnan(val):
                text_color = "black" if 0.2 < data_norm[i, j] < 0.8 else "white"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=8, color=text_color)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.upper() for m in metrics], rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("Metrics Comparison Heatmap (row-normalized)")

    plt.colorbar(im, ax=ax, label="Normalized Score")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved metrics heatmap to {output_path}")

    return fig


def plot_embedding_umap(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    n_samples: int = 2000,
    title: str = "Embedding UMAP",
    output_path: Optional[Path] = None,
) -> Figure:
    """UMAP projection of item/user embeddings for qualitative analysis.

    Args:
        embeddings: 2D array of shape (n_entities, dim).
        labels: Optional array of category labels for coloring.
        n_samples: Subsample size for speed.
        title: Plot title.
        output_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    try:
        import umap
    except ImportError:
        log.warning("umap-learn not installed, skipping UMAP plot")
        return plt.figure()

    n = min(n_samples, len(embeddings))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(embeddings), size=n, replace=False)
    sample_embs = embeddings[idx]

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    projected = reducer.fit_transform(sample_embs)

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        sample_labels = labels[idx]
        unique_labels = np.unique(sample_labels[~np.isnan(sample_labels.astype(float))])
        for lbl in unique_labels[:20]:  # Max 20 categories
            mask = sample_labels == lbl
            ax.scatter(
                projected[mask, 0], projected[mask, 1],
                alpha=0.5, s=5, label=str(lbl)
            )
        if len(unique_labels) <= 20:
            ax.legend(markerscale=3, fontsize=8, loc="upper right")
    else:
        ax.scatter(
            projected[:, 0], projected[:, 1],
            alpha=0.3, s=5, color=PALETTE["two_tower"]
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved UMAP to {output_path}")

    return fig


def generate_all_figures(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Load saved results and generate all publication figures.

    Args:
        results_dir: Directory containing evaluation JSON files.
        figures_dir: Output directory for figures.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Generating all figures in {figures_dir}")

    # Load all experiment results
    results = {}
    bootstrap_data = {}

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        metrics_path = model_dir / "metrics.json"
        boot_path = model_dir / "bootstrap_results.json"

        if metrics_path.exists():
            with open(metrics_path) as f:
                results[model_dir.name] = json.load(f)

        if boot_path.exists():
            with open(boot_path) as f:
                boot = json.load(f)
            bootstrap_data[model_dir.name] = {
                k: np.array(v) for k, v in boot.items()
            }

    if not results:
        log.warning("No results found to plot")
        return

    # 1. Recall comparison
    plot_recall_comparison(
        results,
        metric="recall@12",
        output_path=figures_dir / "recall_comparison.png",
        title="Recall@12: Multi-Stage Recommendation System",
    )

    # 2. NDCG comparison
    plot_recall_comparison(
        results,
        metric="ndcg@12",
        output_path=figures_dir / "ndcg_comparison.png",
        title="NDCG@12 Comparison",
    )

    # 3. Bootstrap distributions
    if bootstrap_data:
        for metric in ["recall@12", "ndcg@12"]:
            boot_for_metric = {
                name: data[metric]
                for name, data in bootstrap_data.items()
                if metric in data
            }
            if boot_for_metric:
                plot_bootstrap_distributions(
                    boot_for_metric,
                    metric=metric,
                    output_path=figures_dir / f"bootstrap_{metric.replace('@', '_')}.png",
                )

    # 4. Metrics heatmap
    plot_metrics_heatmap(
        results,
        output_path=figures_dir / "metrics_heatmap.png",
    )

    log.info(f"All figures saved to {figures_dir}")
