# Test Fixtures

All shared fixtures are defined in `tests/conftest.py`.

## Key Fixtures

- `tiny_dataset` — 200-user synthetic dataset for fast unit tests
- `sample_candidates` — pre-generated candidate DataFrame (50 users × 20 items)
- `trained_als` — pre-fitted ALSRetriever on tiny_dataset
- `trained_two_tower` — pre-fitted TwoTowerRetriever on tiny_dataset

## Running Tests

```bash
# Unit tests only (fast, ~9s)
pytest tests/unit/ -v

# Integration tests (slow, ~25s, runs full pipeline)
pytest tests/integration/ -v

# All tests with coverage
pytest tests/ --cov=src --cov-report=term-missing
```
