# H&M Personalized Fashion Recommendation System

A production-grade, multi-stage recommendation system built on the
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
Kaggle dataset. Designed as an Amazon Applied Scientist portfolio project.

---

## Architecture Overview

```
Raw Interactions
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 0 · Data Loading & Preprocessing                     │
│  K-core filtering → Deterministic sampling → Chrono split   │
└───────────────────────────┬─────────────────────────────────┘
                            │  ~50K interactions
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 · Candidate Generation (Retrieval)                 │
│                                                             │
│   ┌──────────────────┐  ┌──────────┐  ┌────────────────┐  │
│   │ Popularity       │  │  ALS     │  │  Two-Tower     │  │
│   │ Time-decayed     │  │ 64 factors│  │ InfoNCE + FAISS│  │
│   │ O(1) inference   │  │ FAISS    │  │ 64-dim embed.  │  │
│   └──────────────────┘  └──────────┘  └────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │  ~100 candidates/retriever
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 · Candidate Fusion                                 │
│  Reciprocal Rank Fusion → 200 unique (user, item) pairs     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 · Feature Engineering                              │
│  Retrieval features + Item stats + User stats + Cross feats │
└───────────────────────────┬─────────────────────────────────┘
                            │  38 features
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4 · Ranking                                          │
│  LightGBM LambdaMART · NDCG@12 optimization · SHAP values  │
└───────────────────────────┬─────────────────────────────────┘
                            │  Top-12 recommendations
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5 · Evaluation                                       │
│  95% Bootstrap CI · Paired significance test · Per-user     │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Data splitting** | Chronological (70/15/15) | Prevents temporal leakage; simulates production deployment |
| **User sampling** | SHA-256 hash-based | Reproducible across runs without storing user lists |
| **k-core filtering** | min 5 user / 3 item interactions | Removes cold-start noise; iterative convergence |
| **Candidate fusion** | Reciprocal Rank Fusion (k=60) | Scores not comparable across models; RRF is parameter-free |
| **Two-Tower loss** | InfoNCE with in-batch negatives | Scales efficiently; no explicit negative sampling needed |
| **Ranking objective** | LambdaMART (NDCG@12) | Directly optimizes the evaluation metric |
| **ALS index** | FAISS FlatIP (exact) | Normalized embeddings → cosine = inner product; exact search |
| **Feature types** | 38 features across 5 groups | See Feature Engineering section |

---

## Repository Structure

```
hm-recsys/
├── configs/                    # Hydra configuration
│   ├── config.yaml             # Root config
│   ├── data/hm.yaml            # Dataset parameters
│   ├── model/default.yaml      # Model hyperparameters
│   ├── pipeline/default.yaml   # Stage toggles
│   └── experiment/             # Experiment variants
├── scripts/
│   ├── train.py                # Main training script (argparse CLI)
│   ├── run_ablation.py         # Component ablation study
│   ├── evaluate_full.py        # Memory-efficient full dataset evaluation
│   ├── cold_start_analysis.py  # User activity segment evaluation
│   ├── verify_numbers.py       # Automated metric verification test
│   └── generate_synthetic_data.py  # Synthetic data generator
├── src/
│   ├── data/
│   │   └── loader.py           # HMDataLoader (sampling, filtering, splits)
│   ├── retrievers/
│   │   ├── base.py             # BaseRetriever ABC
│   │   ├── popularity.py       # Time-decayed popularity
│   │   ├── als.py              # Implicit ALS + unnormalized MIPS FAISS
│   │   ├── two_tower.py        # InfoNCE Two-Tower + Log-Q debiasing + FAISS
│   │   ├── content_two_tower.py# Gated Multi-Modal Content Dual-Encoder
│   │   └── fusion.py           # Reciprocal Rank Fusion (RRF)
│   ├── features/
│   │   └── engineer.py         # 38-feature engineering pipeline
│   ├── ranker/
│   │   └── lgbm_ranker.py      # LightGBM LambdaMART + SHAP attribution
│   ├── evaluation/
│   │   ├── metrics.py          # RecSysEvaluator with bootstrap CI & paired tests
│   │   └── labels.py           # Ground truth & ranking label generation
│   ├── pipeline/
│   │   └── runner.py           # PipelineRunner (5-stage orchestration)
│   └── analysis/
│       ├── analyzer.py         # Popularity bias, cold-start, segmentation
│       └── visualizer.py       # Publication-quality figures
└── tests/
    ├── unit/                   # 46 unit tests (all passing)
    └── integration/            # End-to-end multi-stage pipeline tests (all passing)
```

---

## Getting Started

### 1. Install Dependencies

```bash
pip install polars lightgbm implicit faiss-cpu torch shap umap-learn \
            loguru typer scipy scikit-learn numpy pandas matplotlib seaborn \
            pyarrow
```

### 2. Prepare Data

**Option A — Real H&M data:**
```bash
kaggle competitions download -c h-and-m-personalized-fashion-recommendations
unzip h-and-m-personalized-fashion-recommendations.zip -d data/raw/
```

**Option B — Synthetic data (no Kaggle account needed):**
```bash
python scripts/generate_synthetic_data.py --n-users 2000 --n-items 5000
```

### 3. Train

```bash
# Full pipeline with defaults
python scripts/train.py

# Custom hyperparameters
python scripts/train.py \
    --als-factors 128 \
    --two-tower-epochs 20 \
    --lgbm-estimators 500 \
    --n-interactions 100000

# CPU-only (no CUDA)
python scripts/train.py --device cpu
```

### 4. Test

```bash
# Unit tests (fast, ~10s)
pytest tests/unit/ -v

# Integration tests (slow, ~5 min)
pytest tests/integration/ -m integration
```

---

## Evaluation Results (Synthetic Data, 50K interactions)

| Model | Recall@12 | NDCG@12 | MRR | HitRate@12 |
|---|---|---|---|---|
| Popularity | 0.0826 ± 0.0045 | 0.0744 | 0.0965 | 0.3880 |
| ALS | 0.0061 ± 0.0015 | 0.0048 | 0.0065 | 0.0437 |
| Two-Tower | 0.0009 ± 0.0004 | 0.0006 | 0.0009 | 0.0080 |
| **Two-Tower + Ranker** | **0.0385** ± 0.0030 | **0.0307** | 0.0616 | 0.2046 |

> Note: Evaluated on synthetic data; ALS and Two-Tower are warm-up limited (8 epochs, 64 factors).
> On the real H&M dataset with full training, ALS and Two-Tower typically outperform popularity.
> All CIs computed via 200 bootstrap samples.

---

## Feature Engineering

38 features across 5 groups:

| Group | Features | Description |
|---|---|---|
| **Retrieval** (8) | rrf_score, n_sources, best_rank, source flags | How well each retriever ranked this item |
| **Item** (10) | purchase_count_log, unique_buyers, price, velocity_7d/30d, days_since_purchase, is_long_tail | Item popularity and recency signals |
| **User** (9) | total_purchases, unique_items, avg_price, tenure, activity_level, days_since_purchase | User behavior profile |
| **Article Metadata** (5) | product_group, color, department, section, index (label-encoded) | Product catalog attributes |
| **Cross** (6) | price_affinity, recency_affinity, inter-retriever agreement | User–item interaction signals |

---

## Model Components

### Popularity Retriever
- Time-decayed purchase frequency: `score = count × exp(-λ × days_ago)`
- Decay half-life: 30 days
- O(1) inference (same list for all users, minus seen items)

### ALS Retriever
- `implicit` library: Alternating Least Squares with confidence matrix `C = 1 + α × count`
- FAISS FlatIP index for exact unnormalized Maximum Inner Product Search (MIPS)
- Unnormalized factor vectors preserve implicit feedback interaction confidence and item popularity weights

### Two-Tower Retriever
- Dual-encoder architecture with user/item embedding towers and MLP projectors
- **InfoNCE loss with in-batch negatives & Log-Q sampling bias correction** (Yi et al. Google RecSys 2019)
- Cosine annealing learning rate schedule with warmup ($\eta_{\text{min}} = 10^{-5}$)
- FAISS FlatIP index over L2-normalized item embeddings

### LightGBM Ranker
- **LambdaMART**: directly optimizes NDCG@12 via gradient boosted decision trees
- Native categorical splitting for nominal article and customer attributes (`item_product_group_idx`, `item_dept_idx`, `user_age_group`, etc.)
- Group-aware training: queries grouped by user_idx with disjoint validation partitioning
- SHAP values computed for feature attribution

---

## Interview Talking Points

**Why multi-stage?**  
Exact scoring over all 100K items per user is O(n) per user × O(n) items = O(n²). The two-stage design reduces this: retrieval narrows to 200 candidates in O(log n), ranking re-scores those 200 in O(k × n_features).

**Why RRF over score normalization?**  
Popularity scores (purchase counts) and ALS cosine similarities live on different scales and have different distributions. Normalizing them introduces bias. RRF uses only the rank ordering, which is distribution-agnostic.

**Why InfoNCE?**  
For a Two-Tower model with large item catalogs, explicit negative sampling is expensive. InfoNCE uses all other items in the batch as negatives, making it O(batch_size²) instead of O(n_items) per gradient step.

**Why LambdaMART instead of logistic regression?**  
LambdaMART directly optimizes NDCG (a non-differentiable IR metric) via gradient approximation. Logistic regression minimizes pointwise logloss, which doesn't correlate well with ranking quality.

**How do you prevent data leakage?**  
Chronological splitting — train on purchases before date T, evaluate on purchases after T. Random splitting would let future interactions contaminate training, inflating all metrics.

---

## Reproducibility

- Global seed set via `set_seed(42)` (Python, NumPy, PyTorch, CUDA)
- User sampling: SHA-256 hash of `seed|user_id` → deterministic subset selection
- Data loading: k-core convergence is deterministic given seed
- All models: seeds propagated to ALS, Two-Tower, LightGBM, FAISS

```bash
# Identical results across runs
python scripts/train.py --seed 42
python scripts/train.py --seed 42  # same output
```
