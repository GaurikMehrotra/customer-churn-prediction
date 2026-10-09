
.PHONY: lint test quality pipeline

lint:
	ruff check src scripts tests

test:
	python -m pytest -q

quality: lint test

pipeline:
	python -m scripts.run_pipeline
