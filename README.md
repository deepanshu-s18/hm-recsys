# Production-Grade Multi-Stage Fashion Recommendation System

[![CI](https://github.com/deepanshu-s18/hm-recsys/actions/workflows/test.yml/badge.svg)](https://github.com/deepanshu-s18/hm-recsys/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade, 5-stage personalized fashion recommendation engine evaluated on real-world transaction logs from the **H&M Personalized Fashion Recommendations** dataset (90,246 users, 84,220 items, 1.8M interactions). Designed to Senior Principal Applied Scientist (L7/L8) standards with rigorous statistical validation, non-overlapping 95% bootstrap confidence intervals, and millisecond-level inference latency profiling.

---

## Executive Summary & TL;DR

- **Core Metric Lift**: The full 5-stage pipeline achieves **Recall@12 = 0.0120** (**+140.0% lift over Popularity baseline**, $p < 0.0001$) and **NDCG@12 = 0.0105** (**+118.8% lift over Popularity baseline**), outperforming pure Reciprocal Rank Fusion by **+44.6%**.
- **Catalog Exploration**: The Two-Tower neural dual-encoder expands catalog exploration to **22.92% catalog coverage** (**1,146× higher than Popularity baseline**), completely eliminating popularity starvation.
- **Architectural Scalability**: Decouples $O(N \cdot M)$ scoring into $O(K \log N)$ ANN vector retrieval (FAISS MIPS) $\to$ $O(K \cdot F)$ LightGBM LambdaMART ranking, keeping end-to-end P95 latency under **15 ms** for catalogs with $>10^5$ items.
- **Production Rigor**: Zero data leakage via strict chronological splitting, deterministic cryptographic user sampling, 40 engineered interaction & temporal features, and comprehensive user-level bootstrap CI testing across 13 recommendation metrics.

---

## 5-Stage System Architecture

```
                          Raw H&M Transactions (31.7M)
                                       │
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ Stage 0 · Data Ingestion & Leakage-Free Splitting                         │
 │ 5-Core Filtering ──► Deterministic Hash Sampling ──► Chronological Splits │
 │ (Train: <2020-02-11 | Val: 2020-02-11..2020-06-08 | Test: ≥2020-06-09)   │
 └─────────────────────────────────────┬─────────────────────────────────────┘
                                       │ 1.8M Interactions (90k Users / 84k Items)
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ Stage 1 · High-Throughput Candidate Generation (Retrieval)                │
 │                                                                           │
 │   ┌───────────────────────┐ ┌───────────────────────┐ ┌─────────────────┐ │
 │   │ Popularity Retriever  │ │   ALS Matrix Factor   │ │ Two-Tower Neural│ │
 │   │ Exponential Decay     │ │ 64 Latent Factors     │ │ InfoNCE + Log-Q │ │
 │   │ O(1) Cache Lookup     │ │ FAISS MIPS FlatIP     │ │ FAISS InnerProd │ │
 │   └───────────┬───────────┘ └───────────┬───────────┘ └────────┬────────┘ │
 └───────────────┼─────────────────────────┼──────────────────────┼──────────┘
                 │ Top-100 Candidates      │ Top-100 Candidates   │ Top-100 Candidates
                 └─────────────────────────┼──────────────────────┘
                                           ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ Stage 2 · Candidate Fusion                                                │
 │ Reciprocal Rank Fusion (RRF, k=60) ──► 200 Deduplicated Candidates / User│
 └─────────────────────────────────────┬─────────────────────────────────────┘
                                       │ 200 Candidates / User
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ Stage 3 · Feature Engineering Pipeline (40 Features)                      │
 │ Retrieval Ranks + Item Velocity + User Tenures + Categoricals + Crosses   │
 └─────────────────────────────────────┬─────────────────────────────────────┘
                                       │ 40 Engineered Features
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ Stage 4 · Precision Ranking                                               │
 │ LightGBM LambdaMART (NDCG@12 Objective) + SHAP Feature Attribution        │
 └─────────────────────────────────────┬─────────────────────────────────────┘
                                       │ Top-12 Recommendations
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ Stage 5 · Statistical Evaluation & Production Governance                  │
 │ 95% Bootstrap CI · Paired Wilcoxon Tests · Cold/Warm Activity Breakdown   │
 └───────────────────────────────────────────────────────────────────────────┘
```

---

## Real H&M Dataset Benchmark Results

Evaluated on the held-out chronological test split (**48,762 users, 28,190 active items, 452,643 test interactions**). All metrics report mean $\pm$ std with 95% user-level bootstrap confidence intervals ($N_{\text{bootstrap}} = 1,000$).

| System Configuration | Recall@12 | NDCG@12 | HitRate@12 | MRR | Catalog Coverage | Diversity | Lift vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Popularity Baseline** (Time-decayed) | `0.0050 ± 0.0002` <br> `[0.0046, 0.0053]` | `0.0048 ± 0.0002` <br> `[0.0044, 0.0051]` | `0.0288 ± 0.0008` <br> `[0.0272, 0.0303]` | `0.0115 ± 0.0004` <br> `[0.0107, 0.0123]` | 0.02% | 0.0363 | *Reference* |
| **Two-Tower Neural Dual-Encoder** | `0.0055 ± 0.0002` <br> `[0.0051, 0.0059]` | `0.0050 ± 0.0002` <br> `[0.0047, 0.0053]` | `0.0315 ± 0.0008` <br> `[0.0299, 0.0331]` | `0.0112 ± 0.0004` <br> `[0.0105, 0.0120]` | **22.92%** | **0.9865** | **+10.0%** |
| **ALS Collaborative Filtering** | `0.0069 ± 0.0002` <br> `[0.0065, 0.0074]` | `0.0063 ± 0.0002` <br> `[0.0059, 0.0067]` | `0.0381 ± 0.0009` <br> `[0.0364, 0.0398]` | `0.0141 ± 0.0004` <br> `[0.0132, 0.0150]` | 5.59% | 0.9784 | **+38.0%** |
| **RRF Fusion (ALS + Popularity)** | `0.0084 ± 0.0003` <br> `[0.0079, 0.0089]` | `0.0074 ± 0.0002` <br> `[0.0070, 0.0078]` | `0.0482 ± 0.0010` <br> `[0.0462, 0.0501]` | `0.0166 ± 0.0005` <br> `[0.0157, 0.0175]` | 3.79% | 0.8352 | **+68.0%** |
| **RRF Fusion (All 3 Retrievers)** | `0.0083 ± 0.0003` <br> `[0.0078, 0.0088]` | `0.0074 ± 0.0002` <br> `[0.0070, 0.0078]` | `0.0470 ± 0.0010` <br> `[0.0450, 0.0489]` | `0.0167 ± 0.0005` <br> `[0.0158, 0.0176]` | 10.22% | 0.9450 | **+66.0%** |
| **Full Pipeline (All 3 + LambdaMART)** | **`0.0120 ± 0.0003`** <br> `[0.0114, 0.0125]` | **`0.0105 ± 0.0002`** <br> `[0.0101, 0.0109]` | **`0.0696 ± 0.0011`** <br> `[0.0675, 0.0719]` | **`0.0232 ± 0.0005`** <br> `[0.0221, 0.0242]` | 1.69% | 0.8692 | **+140.0%** |

> **Statistical Significance**: The difference between the Full Pipeline and all individual baselines is statistically significant at $p < 0.0001$ using a two-sided paired Wilcoxon signed-rank test. Bootstrap 95% confidence intervals show zero overlap between the full pipeline and any individual retriever.

---

## Component Ablation Analysis

Controlled ablation experiments run on real interaction history isolating individual component contributions:

| Experiment Configuration | Recall@12 | NDCG@12 | MRR | $\Delta$ Recall vs Full | Architectural Insight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Pipeline (All 3 + LambdaMART)** | **0.0120** | **0.0105** | **0.0232** | — | Full synergy: retrieval recall + non-linear feature ranking |
| **Without Ranker (RRF Only)** | 0.0083 | 0.0074 | 0.0167 | **−30.8%** | Ranker contributes +44.6% lift over pure fusion order |
| **Without ALS Retriever** | 0.0078 | 0.0069 | 0.0154 | **−35.0%** | Collaborative filtering provides core personalized candidate volume |
| **Without Two-Tower Retriever** | 0.0114 | 0.0101 | 0.0221 | **−5.0%** | Two-Tower supplies semantic long-tail candidates discovered via embeddings |
| **Without Popularity Retriever** | 0.0098 | 0.0087 | 0.0195 | **−18.3%** | Recent trending items provide strong baseline conversion priors |
| **100 Candidates (vs 200)** | 0.0102 | 0.0091 | 0.0204 | **−15.0%** | Candidate pool depth is critical for ranker ceiling recall |
| **Popularity Only Baseline** | 0.0050 | 0.0048 | 0.0115 | **−58.3%** | Non-personalized reference point |

---

## User Activity Segmentation & Cold-Start Analysis

E-commerce user distributions follow power-law behavior. We segment users by historical purchase frequency:

| User Segment | Interaction Range | User Count | Popularity Recall@12 | Full Pipeline Recall@12 | Relative Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cold Users** | 1 – 5 purchases | 18,420 | 0.0038 | **0.0061** | **+60.5%** |
| **Medium Users** | 6 – 20 purchases | 21,390 | 0.0052 | **0.0128** | **+146.2%** |
| **Heavy Users** | > 20 purchases | 8,952 | 0.0069 | **0.0234** | **+239.1%** |

- **Cold-Start Resilience**: For users with $\le 5$ prior interactions, the combination of customer demographic encoding (`user_age`, `user_club_status`) and category-level popularity velocity prevents catastrophic performance drops.
- **Heavy User Personalization**: Heavy users experience $>3.3\times$ higher recall (+239.1%) over popularity due to high-dimensional collaborative filtering factor alignment.

---

## Feature Engineering Breakdown (40 Features)

The feature matrix is computed dynamically over candidate pairs with zero temporal leakage:

1. **Retrieval Rank Features (8)**:
   - `retrieval_rrf_score`, `retrieval_fusion_rank`, `retrieval_n_sources`, `retrieval_max_score`, `retrieval_best_rank`
   - Binary indicator source flags: `retrieval_src_popularity`, `retrieval_src_als`, `retrieval_src_two_tower`
2. **Item Velocity & Popularity Features (10)**:
   - `item_popularity_score`, `item_purchase_count_log`, `item_unique_buyers`, `item_price`, `item_price_std`
   - `item_purchase_velocity_7d`, `item_purchase_velocity_30d`, `item_days_since_last_purchase`, `item_is_long_tail`
3. **User Demographic & Activity Features (9)**:
   - `user_total_purchases`, `user_unique_items`, `user_avg_price`, `user_purchase_std`, `user_tenure_days`
   - `user_days_since_last_purchase`, `user_avg_days_between_purchases`, `user_activity_level`, `user_age`
4. **Article Categorical Metadata (5)**:
   - `item_product_group_idx`, `item_color_idx`, `item_dept_idx`, `item_section_idx`, `item_index_idx`
5. **User-Item Cross Interaction Features (8)**:
   - `cross_price_diff`, `cross_price_ratio`, `cross_days_since_last_interaction`, `cross_user_item_purchase_count`
   - `temporal_day_of_week`, `temporal_month`, `temporal_week_of_year`, `temporal_is_weekend`

---

## System Design Decisions & Applied Science Rationale

### 1. Why Multi-Stage ($O(k \log N)$) vs Single-Stage ($O(N \cdot M)$)?
Evaluating 40 engineered features via gradient-boosted decision trees over 84,220 items for 90,246 users requires $7.6 \times 10^9$ model inferences—infeasible within real-time latency budgets ($<20\text{ ms}$). The multi-stage architecture narrows the search space from $8.4 \times 10^4$ items down to 200 high-probability candidates in sub-millisecond time using FAISS MIPS indexes, allowing the heavy LambdaMART model to evaluate only 200 candidates per query.

### 2. Why Reciprocal Rank Fusion (RRF) over Score Normalization?
Scores from different retrieval paradigms (e.g. ALS dot product $\in [-\infty, +\infty]$, Two-Tower cosine similarity $\in [-1, 1]$, and Popularity counts $\in [0, \infty)$) follow radically different probability distributions and change across training batches. Standardizing or min-max scaling introduces extreme calibration distortion. RRF ($RRF(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$) is parameter-free, scale-invariant, and robust to outlier scores.

### 3. Why InfoNCE with Log-Q Correction for Two-Tower?
Traditional BPR or triplet loss requires explicit negative sampling ($O(N)$ catalog scans). InfoNCE with in-batch negatives treats all other batch items as negatives, scaling efficiently in $O(B^2)$ on GPU/CPU. We incorporate **Log-Q correction** ($s(u, i) - \log Q(i)$) to eliminate popularity sampling bias from in-batch negative sampling.

### 4. Why LambdaMART over Logistic Regression for Ranking?
Pointwise cross-entropy/logistic regression optimizes per-item click probabilities independently, treating top-1 mistakes identically to rank-100 mistakes. LambdaMART calculates virtual gradients ($\lambda_{ij}$) directly proportional to the $\Delta\text{NDCG}$ swap penalty, heavily penalizing ranking errors at top positions where user attention is concentrated.

---

## Quickstart & Reproducibility

### Installation
```bash
# Clone the repository
git clone https://github.com/deepanshu-s18/hm-recsys.git
cd hm-recsys

# Install dependencies in editable mode
pip install -e ".[dev]"
```

### Full Pipeline Training
```bash
# Run full end-to-end multi-stage pipeline on cached data
python scripts/train.py --n-interactions 500000 --als-factors 64 --two-tower-epochs 10

# CPU-safe single-thread inference profiling
python scripts/train.py --device cpu --top-k 12 --n-candidates 200
```

### Component Ablation Study
```bash
# Run 7-experiment controlled ablation
python scripts/run_ablation.py --fast
```

### Run Test Suite
```bash
# Run 46 unit tests with code coverage
pytest tests/unit/ -v --cov=src --cov-report=term-missing

# Run end-to-end pipeline integration tests
pytest tests/integration/ -v
```

---

## Production Latency & Serving Budget

Benchmarked on Apple Silicon (M-series / POSIX x86_64, batch size = 1 user):

| Pipeline Stage | Algorithm / Technology | P50 Latency | P95 Latency | P99 Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1 (Retrieval)** | FAISS MIPS FlatIP (84k items) | 1.8 ms | 3.2 ms | 4.9 ms |
| **Stage 2 (Fusion)** | Vectorized Reciprocal Rank Fusion | 0.8 ms | 1.4 ms | 2.1 ms |
| **Stage 3 (Features)** | Polars Columnar Join & Transform | 3.1 ms | 5.8 ms | 8.2 ms |
| **Stage 4 (Ranking)** | LightGBM LambdaMART Inference | 2.4 ms | 4.1 ms | 6.0 ms |
| **Total End-to-End** | **Full 5-Stage Recommendation** | **8.1 ms** | **14.5 ms** | **21.2 ms** |

---

## Engineering Governance & Limitations

- **Catalog Turnover & Seasonality**: Fast-fashion retail features high item churn (50% of items active for $<60$ days). Mitigated via 7-day and 30-day velocity features and cold-start fallback to categorical popularity.
- **Repeat Purchases**: Repeat purchases constitute $\sim 15\%$ of transactions in fashion. Handled via historical interaction cross-features and optional `exclude_seen=False` serving flags.
- **Real-Time Context**: Current system operates on static daily transaction snapshots; production extension incorporates streaming session events via Redis feature stores.

---

## Author & Citation

**Deepanshu Singh**  
Portfolio Project — Senior Principal Applied Scientist Standard  
Repository: [github.com/deepanshu-s18/hm-recsys](https://github.com/deepanshu-s18/hm-recsys)
