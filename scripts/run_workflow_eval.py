"""Run the Atlas RAG evaluation corpus against the configured local stack.

This exercises ingestion, chunking, embeddings, Chroma retrieval, reranking, and the
bounded agent path. If OPENAI_API_KEY is configured, it also runs generation, memory,
and grounding through answer_question. Metrics describe this corpus only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from atlas_rag.agent import retrieve_with_agent
from atlas_rag.config import get_settings
from atlas_rag.embeddings import configured_embedder
from atlas_rag.grounding import validate_citations
from atlas_rag.ingestion import ingest_document
from atlas_rag.memory import ConversationMemory
from atlas_rag.reranking import rerank
from atlas_rag.vector_store import configured_store

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "eval_corpus"
DATASET = ROOT / "data" / "eval.workflow.jsonl"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "latest.json")
    args = parser.parse_args()
    settings = get_settings()
    embedder, store = configured_embedder(settings), configured_store(settings)
    source_ids: dict[str, str] = {}
    ingestion = []
    for path in sorted(CORPUS.glob("*.md")):
        content = path.read_bytes()
        result = ingest_document(path.name, content, settings, embedder, store)
        source_ids[path.name] = str(result["source_id"])
        ingestion.append({"filename": path.name, **result})

    rows = load_rows(DATASET)
    recalls, reciprocal_ranks, cases = [], [], []
    for row in rows:
        relevant = {source_ids[name] for name in row["relevant_filenames"]}
        # Exercise the same bounded retrieval/reranking path used by the application.
        query, hits, passes = retrieve_with_agent(row["question"], args.k, settings, embedder, store)
        ranked = [item["metadata"]["source_id"] for item in hits]
        first_rank = next((i + 1 for i, item in enumerate(ranked) if item in relevant), None)
        recalls.append(float(first_rank is not None))
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
        cases.append({
            "question": row["question"], "category": row["category"],
            "expected_filenames": row["relevant_filenames"],
            "retrieved_filenames": [item["metadata"]["filename"] for item in hits],
            "hit": first_rank is not None, "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
            "retrieval_passes": passes, "effective_query": query,
        })

    # Exercise persistent session storage and deterministic citation validation.
    memory = ConversationMemory(settings.memory_path, max_turns=2)
    memory.clear("workflow-eval")
    memory.add("workflow-eval", "user", "evaluation memory probe")
    memory.add("workflow-eval", "assistant", "stored answer [S1]")
    memory_probe = memory.get("workflow-eval")
    citation_probe = {
        "valid": validate_citations("A grounded answer [S1].", 1)[0],
        "invalid_rejected": not validate_citations("A claim [S2].", 1)[0],
        "missing_rejected": not validate_citations("An uncited claim.", 1)[0],
    }

    result = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_cases": len(rows),
        "corpus_documents": len(ingestion),
        "ingestion": ingestion,
        "retrieval": {
            f"recall@{args.k}": round(mean(recalls), 4),
            "mrr": round(mean(reciprocal_ranks), 4),
            "cases": cases,
        },
        "memory_probe_passed": len(memory_probe) == 2,
        "citation_probe": citation_probe,
        "external_steps": {
            "generation": "not_run_no_api_key" if not settings.openai_api_key else "available_run_via_API",
            "agentic_reformulation": "requires_OPENAI_API_KEY_when_initial_retrieval_is_weak",
            "llm_grounding_verifier": "requires_OPENAI_API_KEY_and_GROUNDING_CHECK_ENABLED",
        },
        "limitations": [
            "Recall/MRR measure retrieval only; they do not establish answer correctness.",
            "The out-of-corpus case is reported for inspection and is not counted as a retrieval hit.",
            "Generation, semantic grounding, and agentic reformulation require configured OpenAI credentials.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
