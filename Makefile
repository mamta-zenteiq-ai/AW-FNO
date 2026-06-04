# =============================================================================
# AW-FNO Makefile
# =============================================================================
# Common shortcuts for development and paper reproduction.
#
# Usage:
#   make train-awfno   DATA_PATH=/path/to/ns/data
#   make benchmark     DATA_PATH=/path/to/ns/data
#   make test
#   make figures       DATA_PATH=/path/to/ns/data
#   make clean

DATA_PATH  ?= data/ns2d
DEVICE     ?= auto
EPOCHS     ?= 500
SEED       ?= 42
PYTHON     ?= python

.PHONY: help install test lint train-all train-awfno train-fno train-wno ablation \
        benchmark figures paper-ready clean

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "AW-FNO Makefile"
	@echo "==============="
	@echo "  make install          Install Python dependencies"
	@echo "  make test             Run unit tests"
	@echo "  make lint             Run flake8 / black --check"
	@echo ""
	@echo "  make download         Download FNO benchmark datasets"
	@echo "  make train-awfno      Train AW-FNO on NS2D"
	@echo "  make train-fno        Train FNO baseline"
	@echo "  make train-wno        Train WNO baseline"
	@echo "  make train-all        Train all three models sequentially"
	@echo "  make ablation         Run ablation study (no-gate variant)"
	@echo "  make benchmark        Compare all trained models"
	@echo "  make figures          Generate paper figures"
	@echo "  make paper-ready      Run full reproduction pipeline"
	@echo "  make clean            Remove compiled Python files"
	@echo ""
	@echo "Override defaults: make train-awfno DATA_PATH=/path EPOCHS=200"
	@echo ""

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install:
	pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	flake8 awfno/ utils/ datasets/ experiments/ trainers/ --max-line-length 100 --extend-ignore E203,W503
	black --check awfno/ utils/ datasets/ experiments/ trainers/ --line-length 100

format:
	black awfno/ utils/ datasets/ experiments/ trainers/ --line-length 100

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
download:
	$(PYTHON) datasets/download_fno_data.py --dataset all

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
train-awfno:
	$(PYTHON) experiments/train.py \
		--config configs/experiment/train_awfno_ns.yaml \
		--data_path $(DATA_PATH) \
		--epochs $(EPOCHS) \
		--seed $(SEED) \
		--device $(DEVICE)

train-fno:
	$(PYTHON) experiments/train.py \
		--config configs/experiment/train_fno_ns.yaml \
		--data_path $(DATA_PATH) \
		--epochs $(EPOCHS) \
		--seed $(SEED) \
		--device $(DEVICE)

train-wno:
	$(PYTHON) experiments/train.py \
		--config configs/experiment/train_wno_ns.yaml \
		--data_path $(DATA_PATH) \
		--epochs $(EPOCHS) \
		--seed $(SEED) \
		--device $(DEVICE)

train-all: train-fno train-wno train-awfno

ablation:
	$(PYTHON) experiments/train.py \
		--config configs/experiment/ablation_no_gate.yaml \
		--data_path $(DATA_PATH) \
		--epochs $(EPOCHS) \
		--seed $(SEED) \
		--device $(DEVICE)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
benchmark:
	$(PYTHON) experiments/benchmark.py \
		--dataset ns2d \
		--data_path $(DATA_PATH) \
		--device $(DEVICE) \
		--save_figures

figures:
	$(PYTHON) scripts/generate_paper_figures.py \
		--data_path $(DATA_PATH) \
		--device $(DEVICE)

paper-ready:
	bash scripts/reproduce_all.sh

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
