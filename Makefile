.PHONY: lint test quality

lint:
	ruff check src scripts tests

test:
	python -m pytest -q

quality: lint test
