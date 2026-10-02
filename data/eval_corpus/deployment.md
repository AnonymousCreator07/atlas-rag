# Atlas RAG deployment and API

The FastAPI service exposes GET /health, GET /metrics, POST /documents, POST /query, and DELETE /sessions/{session_id}. Document uploads accept TXT, Markdown, PDF, and DOCX files up to 25 MB. Scanned PDFs need OCR before ingestion because the loader extracts text rather than images.

Docker Compose persists Chroma and SQLite session memory in a named volume. Before public deployment, the project recommends TLS, authentication and authorization, rate limiting, secret management, restricted CORS, encrypted storage, backups, malware scanning, tenant isolation, and persistent observability.