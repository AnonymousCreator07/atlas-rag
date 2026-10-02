# Atlas RAG

An end-to-end, inspectable Retrieval-Augmented Generation system with production-oriented retrieval, reranking, conversational memory, grounding, evaluation, deployment, monitoring, and bounded agentic workflows.

## Architecture

~~~text
Documents -> ingestion -> chunking -> embeddings -> Chroma
Question -> dense retrieval -> cross-encoder reranking
         -> bounded agentic reformulation when retrieval is weak
         -> prompt construction + conversation memory
         -> OpenAI API generation
         -> citation validation + optional grounding verification
         -> answer + source excerpts + metrics
~~~

## Requirements implemented

| Requirement | Implementation |
|---|---|
| Document ingestion | loaders.py + ingestion.py |
| Chunking | chunking.py |
| Embeddings | Sentence Transformers locally or OpenAI |
| Vector database | Chroma |
| Retrieval | Dense semantic retrieval |
| Reranking | CrossEncoder |
| Prompt construction | generation.py |
| OpenAI API | generation.py, agent.py, grounding.py |
| Conversational memory | SQLite session memory |
| Evaluation | evaluate.py + evaluate_rag.py |
| Hallucination handling | evidence-only prompting + grounding verifier + fallback |
| Citations | [S1], [S2] markers + source metadata |
| Deployment | Dockerfile + docker-compose.yml |
| Monitoring | structured logs + /health + /metrics |
| Agentic workflows | bounded retrieval/query-reformulation agent |

## Quick start

Requires Python 3.11+.

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
~~~

Set OPENAI_API_KEY in .env to enable answer generation, then:

~~~powershell
uvicorn atlas_rag.api:app --reload
~~~

Open http://127.0.0.1:8000/docs.

Upload a document with POST /documents, then query POST /query:

~~~json
{
  "question": "Explain the key concept in my lecture notes",
  "top_k": 5,
  "session_id": "demo-session",
  "use_agentic": true
}
~~~

The response contains the generated answer, source excerpts, page/document metadata, dense scores, optional reranker scores, timings, grounding status, and retrieval-pass count.

## Conversational memory

Use the same session_id for follow-up questions. Memory is persisted in SQLite and bounded by MEMORY_MAX_TURNS.

Clear a session with:

~~~text
DELETE /sessions/demo-session
~~~

## Retrieval and reranking

The system retrieves RETRIEVAL_CANDIDATES dense candidates and then optionally reranks them using:

~~~text
cross-encoder/ms-marco-MiniLM-L-6-v2
~~~

The final TOP_K passages become the evidence context.

## Agentic RAG

The included agent is deliberately bounded. It does not execute arbitrary tools or privileged actions.

When the first retrieval pass is weak, the agent can use the OpenAI model to reformulate the question and perform one additional retrieval/reranking pass. This makes the project demonstrate an agentic workflow without turning document text into executable instructions.

## Hallucination and citation handling

The generation prompt requires evidence-only answers and [S#] citations. Citation IDs are checked deterministically.

Set GROUNDING_CHECK_ENABLED=true to perform an additional OpenAI verification call. If the verifier determines that the answer is not supported by the supplied evidence, Atlas returns a safe grounded fallback instead of presenting the unsupported answer.

This is a mitigation layer, not a complete guarantee against hallucinations or prompt injection.

## Monitoring

The service provides:

- GET /health for service/provider/index configuration
- GET /metrics for request counts and latency
- structured JSON log events for completed queries
- retrieval-pass and grounding information per query

For a distributed production deployment, replace the in-memory metrics with Prometheus/OpenTelemetry or another persistent observability backend.

## Evaluation

A labeled workflow corpus and reproducible runner are included:

~~~powershell
python scripts/run_workflow_eval.py --k 5
~~~

The runner ingests three evaluation documents, chunks and embeds them, indexes them in Chroma, then measures retrieval Recall@k/MRR after reranking and the bounded agent retrieval path. It also probes SQLite memory and deterministic citation validation. With OPENAI_API_KEY configured it exercises the full query path and saves answers, citation validity, and grounding status for human review. Without credentials it records generation and LLM-assisted reformulation as unavailable; it does not invent answer scores.

Nine questions cover retrieval, ingestion, deployment, memory, citation/grounding, and an out-of-corpus abstention case. The out-of-corpus item is excluded from retrieval relevance metrics. Expected facts are review guidance, not automatically scored answer truth.

Results are written to evaluation/latest.json with per-question details. See [evaluation/README.md](evaluation/README.md) for setup and interpretation. The corpus is intentionally small and is a reproducible smoke evaluation, not a production benchmark.

## Docker deployment

~~~powershell
Copy-Item .env.example .env
docker compose up --build
~~~

The Compose configuration persists Chroma and SQLite memory in a named volume.

For public deployment, add TLS, authentication/authorization, rate limiting, secret management, restricted CORS, encrypted storage, backups, malware scanning, tenant isolation, and persistent observability.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| OPENAI_API_KEY | unset | OpenAI generation and optional agent/grounding calls |
| OPENAI_CHAT_MODEL | gpt-4o-mini | Generation/verifier model |
| EMBEDDING_PROVIDER | local | local or openai embeddings |
| LOCAL_EMBEDDING_MODEL | sentence-transformers/all-MiniLM-L6-v2 | Local embedding model |
| OPENAI_EMBEDDING_MODEL | text-embedding-3-small | OpenAI embedding model |
| CHROMA_PATH | .data/chroma | Vector store |
| MEMORY_PATH | .data/memory.sqlite3 | Conversation memory |
| CHUNK_SIZE / CHUNK_OVERLAP | 900 / 150 | Chunking |
| TOP_K | 5 | Final evidence count |
| RETRIEVAL_CANDIDATES | 20 | Candidates before reranking |
| MAX_CONTEXT_CHARS | 12000 | LLM context budget |
| RERANKER_ENABLED | true | Cross-encoder reranking |
| RERANKER_MODEL | cross-encoder/ms-marco-MiniLM-L-6-v2 | Reranker model |
| MEMORY_ENABLED | true | Session memory |
| MEMORY_MAX_TURNS | 8 | Memory limit |
| GROUNDING_CHECK_ENABLED | false | Optional LLM grounding verification |
| AGENTIC_RAG_ENABLED | true | Bounded query reformulation |
| AGENTIC_MIN_RETRIEVAL_SCORE | 0.25 | Agent trigger threshold |

Changing embedding models requires re-indexing the corpus.

## Repository layout

~~~text
src/atlas_rag/
  api.py             FastAPI routes
  agent.py           bounded agentic retrieval
  chunking.py        deterministic chunking
  embeddings.py      local/OpenAI embedding providers
  generation.py      OpenAI generation and prompt construction
  grounding.py       citation and grounding verification
  ingestion.py       ingestion orchestration
  loaders.py         TXT/MD/PDF/DOCX extraction
  memory.py          SQLite conversational memory
  monitoring.py      structured logs and metrics
  reranking.py       cross-encoder reranking
  retrieval.py       end-to-end query orchestration
  schemas.py         API contracts
  vector_store.py    Chroma adapter

scripts/
  evaluate.py
  evaluate_rag.py

tests/
  chunking, loaders, memory, grounding, and reranking tests
~~~

Run the test suite and linter with:

~~~powershell
pytest
ruff check .
~~~

MIT License.
