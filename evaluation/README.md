# Workflow evaluation

Run from the repository root after installing the project and optional development dependencies:

```powershell
python -m pip install -e ".[dev]"
python scripts/run_workflow_eval.py --k 5
```

The runner ingests the Markdown corpus in `data/eval_corpus/` into the configured Chroma collection, then exercises the configured embedding model, dense retrieval, cross-encoder reranking, and bounded agent retrieval. It also probes SQLite conversation memory and deterministic citation validation. Retrieval Recall@k and MRR are calculated from the labeled questions in `data/eval.workflow.jsonl`.

The out-of-corpus prompt is included to inspect abstention behavior but is excluded from retrieval relevance scoring. This script does not claim semantic answer faithfulness from retrieval metrics.

With `OPENAI_API_KEY` configured, it also runs generation, memory-conditioned answers, citation validation, and optional LLM grounding via a separate full-query pass. If the key is absent, those external calls are explicitly reported as unavailable; do not replace them with invented scores.

The script writes a machine-readable result to `evaluation/latest.json`. Treat results as specific to the corpus, model versions, configuration, and runtime recorded alongside them. Remove the evaluation session from SQLite after inspecting results if desired.
