.PHONY: all data apoe gwas coloc ml test clean install

install:
	pip install -e ".[dev]"

data:
	python scripts/00_simulate.py

apoe: data
	python scripts/01_apoe_stratification.py

gwas: data
	python scripts/02_gwas.py

coloc:
	python scripts/03_coloc_mr.py

ml: data
	python scripts/04_ml_integration.py

all: data apoe gwas coloc ml

test:
	pytest -q

clean:
	rm -rf data/processed results/*.csv results/figures/*.png
	find . -type d -name __pycache__ -exec rm -rf {} +
