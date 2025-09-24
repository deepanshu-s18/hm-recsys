# Contributing

## Development Setup

```bash
git clone https://github.com/deepanshu-s18/hm-recsys.git
cd hm-recsys
pip install -e ".[dev]"
pre-commit install
```

## Code Style

- Formatter: `black` (line length 100)
- Linter: `ruff`
- Type checking: `mypy`

Run all checks: `make lint`

## Adding a New Retriever

1. Subclass `src.retrievers.base.BaseRetriever`
2. Implement `fit(dataset: HMDataset) -> None`
3. Implement `retrieve(user_ids: np.ndarray, top_k: int) -> pl.DataFrame`
4. Return a DataFrame with columns: `[user_idx, item_idx, score, retriever]`
5. Add unit tests in `tests/unit/test_retrievers.py`

## Running Experiments

```bash
# Full pipeline
python scripts/train.py --n-interactions 500000

# Ablation study
python scripts/run_ablation.py --fast
```

## Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — code restructuring without behavior change
- `perf:` — performance improvement
- `test:` — adding or updating tests
- `docs:` — documentation changes
- `ci:` — CI/CD configuration
