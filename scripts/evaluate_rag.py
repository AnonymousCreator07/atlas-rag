"""Evaluate retrieval quality plus optional answer citation/grounding behavior."""

import argparse
import json
from pathlib import Path

from atlas_rag.config import get_settings
from atlas_rag.embeddings import configured_embedder
from atlas_rag.grounding import validate_citations
from atlas_rag.vector_store import configured_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--answers",
        type=Path,
        help="Optional JSONL with answer and expected source count for citation checks.",
    )
    args = parser.parse_args()

    settings = get_settings()
    embedder, store = configured_embedder(settings), configured_store(settings)
    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("Evaluation dataset is empty.")

    recalls, reciprocal_ranks = [], []
    for row in rows:
        hits = store.query(embedder.embed([row["question"]])[0], args.k)
        relevant = set(row["relevant_source_ids"])
        ranked = [hit["metadata"]["source_id"] for hit in hits]
        recalls.append(float(bool(relevant.intersection(ranked))))
        reciprocal_ranks.append(
            next((1 / (i + 1) for i, source_id in enumerate(ranked) if source_id in relevant), 0.0)
        )

    result = {
        "queries": len(rows),
        f"recall@{args.k}": round(sum(recalls) / len(rows), 4),
        "mrr": round(sum(reciprocal_ranks) / len(rows), 4),
    }

    if args.answers:
        answer_rows = [
            json.loads(line)
            for line in args.answers.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        citation_checks = [
            validate_citations(row["answer"], int(row["source_count"]))[0] for row in answer_rows
        ]
        result["citation_validity"] = (
            round(sum(citation_checks) / len(citation_checks), 4) if citation_checks else 0.0
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
