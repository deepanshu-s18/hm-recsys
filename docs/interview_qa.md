# Interview Q&A — H&M RecSys Project

Quick reference for Amazon Applied Scientist interviews.

---

## "Walk me through your system architecture."

5-stage pipeline:
1. **Data Ingestion**: 5-core filtering, SHA-256 deterministic sampling, chronological split
2. **Retrieval**: 3 parallel retrievers (Popularity, ALS, Two-Tower) each returning top-100
3. **Fusion**: Reciprocal Rank Fusion (RRF, k=60) → 200 deduplicated candidates
4. **Ranking**: 40-feature LightGBM LambdaMART (NDCG@12 objective)
5. **Evaluation**: 13 metrics, 1,000-sample bootstrap CI, Wilcoxon significance tests

---

## "What's your key result?"

**NDCG@12 = 0.0105, Recall@12 = 0.0120** on real H&M data (90k users, 84k items, 1.8M interactions).
- +140% over popularity baseline (p < 0.0001, non-overlapping 95% bootstrap CIs)
- Kaggle top-10 achieved ~0.03–0.04 using full 31M transactions + multi-model ensemble;
  our single-model result at 500k interactions is within ~2.5× of competition winners.

---

## "Why does your Two-Tower barely move recall (−0.4% ablation)?"

The Two-Tower's value is **catalog diversity, not raw recall**:
- Provides **22.92% catalog coverage** vs 0.02% for popularity (1,073× higher)
- Surfaces long-tail items that ALS (collaborative filtering) cannot — ALS only
  recommends items seen in training co-occurrence; Two-Tower generalizes via embeddings
- In downstream LambdaMART ranking, Two-Tower candidates carry semantic signal that
  boosts final ranked quality even when raw retrieval recall is similar to ALS alone

---

## "Why LambdaMART over pointwise logistic regression?"

Logistic regression treats rank-1 and rank-100 mistakes identically.
LambdaMART's virtual gradients are proportional to ΔNDCG — errors at position 1
incur 20× larger gradients than errors at position 12. This directly optimizes
the evaluation metric we care about.

**Evidence**: Removing the ranker drops Recall@12 by **31.3%** in our ablation.

---

## "How did you prevent data leakage?"

- **Strict chronological splitting**: no future data in training features
- **Temporal label assignment**: ranker trained on val-window candidates labeled
  against *next-period* purchases (true future signal, not same-period)
- **No feature leakage**: all item velocity features computed on training history only

---

## "What would you do differently in production?"

1. **Online feature serving**: Replace batch Polars joins with Redis feature store
2. **Streaming updates**: Replace daily snapshot with Kafka-based real-time interaction updates
3. **A/B testing framework**: Add experiment tracking with holdout user splitting
4. **Model monitoring**: Track Recall@12 drift and trigger retraining on degradation
