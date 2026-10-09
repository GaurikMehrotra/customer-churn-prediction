.PHONY: setup lint test quality pipeline validate

setup:
	python -m pip install -r requirements.txt

lint:
	ruff check src scripts tests

test:
	python -m pytest -q

quality: lint test

pipeline:
	python -m scripts.run_pipeline

validate:
	$(MAKE) quality
	$(MAKE) pipeline
