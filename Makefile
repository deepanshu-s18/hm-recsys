.PHONY: help install lint test test-unit test-integration generate-data train clean

# ─── Config ──────────────────────────────────────────────────────────────────
PYTHON := python
DATA_DIR := data/raw
ARTIFACTS_DIR := artifacts
N_INTERACTIONS := 100000

# ─── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo "H&M Recommendation System"
	@echo ""
	@echo "Usage:"
	@echo "  make install              Install Python dependencies"
	@echo "  make generate-data        Generate synthetic H&M data (no Kaggle needed)"
	@echo "  make train                Run full training pipeline"
	@echo "  make train-quick          Quick test run (small model)"
	@echo "  make lint                 Run ruff + black + mypy"
	@echo "  make test                 Run unit tests"
	@echo "  make test-integration     Run integration tests (slow)"
	@echo "  make clean                Remove artifacts and cache"
	@echo ""

# ─── Install ─────────────────────────────────────────────────────────────────
install:
	pip install polars lightgbm implicit faiss-cpu torch shap umap-learn \
	            loguru typer scipy scikit-learn numpy pandas matplotlib seaborn \
	            pyarrow pytest pytest-cov ruff black mypy

# ─── Data ─────────────────────────────────────────────────────────────────────
generate-data:
	$(PYTHON) scripts/generate_synthetic_data.py \
	    --n-users 2000 --n-items 5000 --n-interactions 120000 \
	    --output-dir $(DATA_DIR)

# ─── Training ─────────────────────────────────────────────────────────────────
train:
	$(PYTHON) scripts/train.py \
	    --data-dir $(DATA_DIR) \
	    --artifacts-dir $(ARTIFACTS_DIR) \
	    --n-interactions $(N_INTERACTIONS) \
	    --als-factors 128 \
	    --als-iterations 30 \
	    --two-tower-epochs 20 \
	    --two-tower-dim 128 \
	    --lgbm-estimators 500 \
	    --n-bootstrap 1000

train-quick:
	$(PYTHON) scripts/train.py \
	    --data-dir $(DATA_DIR) \
	    --artifacts-dir $(ARTIFACTS_DIR) \
	    --n-interactions 10000 \
	    --als-factors 32 \
	    --als-iterations 5 \
	    --two-tower-epochs 3 \
	    --two-tower-dim 32 \
	    --lgbm-estimators 50 \
	    --n-bootstrap 100 \
	    --no-generate-plots

# ─── Code Quality ─────────────────────────────────────────────────────────────
lint:
	ruff check src/ scripts/ tests/ --fix
	black src/ scripts/ tests/
	mypy src/ --ignore-missing-imports

# ─── Tests ────────────────────────────────────────────────────────────────────
test:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short -m integration

test-all: test test-integration

# ─── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	rm -rf $(ARTIFACTS_DIR) data/processed __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
