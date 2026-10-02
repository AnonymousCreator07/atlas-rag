.PHONY: install dev test lint serve evaluate
install:
	python -m pip install -e ".[dev]"
dev:
	uvicorn atlas_rag.api:app --reload
test:
	pytest
lint:
	ruff check .
serve:
	uvicorn atlas_rag.api:app --host 0.0.0.0 --port 8000
evaluate:
	python scripts/evaluate.py --dataset data/eval.sample.jsonl
