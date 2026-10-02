"""Run the labeled Atlas RAG corpus against the configured application stack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from atlas_rag.agent import retrieve_with_agent
from atlas_rag.config import get_settings
from atlas_rag.embeddings import configured_embedder
from atlas_rag.grounding import validate_citations
from atlas_rag.ingestion import ingest_document
from atlas_rag.memory import ConversationMemory
from atlas_rag.retrieval import answer_question
from atlas_rag.vector_store import configured_store

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "eval_corpus"
DATASET = ROOT / "data" / "eval.workflow.jsonl"


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "latest.json")
    args = parser.parse_args()
    if args.k < 1:
        raise SystemExit("--k must be at least 1")

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
    memory = ConversationMemory(settings.memory_path, max_turns=settings.memory_max_turns)
    session_id = "workflow-eval"
    memory.clear(session_id)
    recalls, reciprocal_ranks, cases, answer_checks = [], [], [], []

    for row in rows:
        relevant = {source_ids[name] for name in row["relevant_filenames"]}
        answer_result = None
        if settings.openai_api_key:
            # Uses the application's full query path: bounded agent, reranker,
            # conversation history, generation, deterministic citation validation,
            # and optional LLM grounding check.
            answer_result = answer_question(
                row["question"], args.k, settings, embedder, store,
                memory=memory, session_id=session_id, use_agentic=True,
            )
            hits = [
                {
                    "metadata": {"source_id": item["source_id"], "filename": item["filename"]},
                    "score": item["score"],
                }
                for item in answer_result["sources"]
            ]
            passes = answer_result["retrieval_passes"]
            effective_query = row["question"]
        else:
            # Offline path runs ingestion, chunking, local embeddings, Chroma,
            # dense retrieval, reranking, and one bounded (non-reformulating) pass.
            effective_query, hits, passes = retrieve_with_agent(
                row["question"], args.k, settings, embedder, store
            )

        ranked = [item["metadata"]["source_id"] for item in hits]
        first_rank = next((i + 1 for i, item in enumerate(hits) if item["metadata"]["source_id"] in relevant), None)
        reciprocal_rank = 1 / first_rank if first_rank else 0.0
        is_scored = bool(relevant)
        if is_scored:
            recalls.append(float(first_rank is not None))
            reciprocal_ranks.append(reciprocal_rank)

        case = {
            "question": row["question"],
            "category": row["category"],
            "expected_filenames": row["relevant_filenames"],
            "retrieved_filenames": [item["metadata"]["filename"] for item in hits],
            "retrieval_hit": first_rank is not None if is_scored else None,
            "reciprocal_rank": reciprocal_rank if is_scored else None,
            "retrieval_passes": passes,
            "effective_query": effective_query,
            "should_abstain": row.get("should_abstain", False),
            "expected_facts_for_human_review": row.get("expected_facts", []),
        }
        if answer_result is not None:
            valid_citations, citation_reason = validate_citations(
                answer_result["answer"], len(answer_result["sources"])
            )
            case.update({
                "answer": answer_result["answer"],
                "grounded": answer_result["grounded"],
                "citations_valid": valid_citations,
                "citation_reason": citation_reason,
            })
            answer_checks.append({
                "grounded": bool(answer_result["grounded"]),
                "citations_valid": valid_citations,
            })
        cases.append(case)

    memory_probe = memory.get(session_id)
    citation_probe = {
        "valid": validate_citations("A grounded answer [S1].", 1)[0],
        "invalid_rejected": not validate_citations("A claim [S2].", 1)[0],
        "missing_rejected": not validate_citations("An uncited claim.", 1)[0],
    }
    scored_count = len(recalls)
    result = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_cases": len(rows),
        "retrieval_scored_cases": scored_count,
        "unscored_out_of_corpus_cases": len(rows) - scored_count,
        "corpus_documents": len(ingestion),
        "ingestion": ingestion,
        "retrieval": {
            f"recall@{args.k}": round(mean(recalls), 4) if recalls else None,
            "mrr": round(mean(reciprocal_ranks), 4) if reciprocal_ranks else None,
            "cases": cases,
        },
        "memory_probe_passed": len(memory_probe) >= 2,
        "citation_probe": citation_probe,
        "generation_cases": len(answer_checks),
        "generation_citation_validity": (
            round(mean(float(item["citations_valid"]) for item in answer_checks), 4)
            if answer_checks else None
        ),
        "generation_grounded_rate": (
            round(mean(float(item["grounded"]) for item in answer_checks), 4)
            if answer_checks else None
        ),
        "external_steps": {
            "generation": "ran" if settings.openai_api_key else "not_run_no_api_key",
            "agentic_reformulation": (
                "eligible_when_retrieval_is_weak" if settings.openai_api_key
                else "not_run_no_api_key"
            ),
            "llm_grounding_verifier": (
                "ran_if_enabled" if settings.openai_api_key and settings.grounding_check_enabled
                else "not_run_disabled_or_no_api_key"
            ),
        },
        "limitations": [
            "Retrieval Recall/MRR measure source retrieval, not answer factuality.",
            "The out-of-corpus case is excluded from retrieval relevance metrics.",
            "Expected answer facts are presented for human review; no automatic semantic answer score is claimed.",
            "Generation and agentic reformulation require a configured OpenAI API key.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
