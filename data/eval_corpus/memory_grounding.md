# Atlas RAG memory and grounded answers

Conversation memory is stored in SQLite and bounded by MEMORY_MAX_TURNS. A client reuses the same session_id for follow-up questions and can clear that session with DELETE /sessions/{session_id}. Conversation history provides context but is not factual evidence.

The generation prompt asks the model to answer only from retrieved excerpts and cite factual claims with markers such as [S1]. Citation IDs are checked deterministically. An optional OpenAI grounding verifier can check whether the answer is supported; if it rejects the answer, Atlas returns a grounded fallback. This mitigation does not guarantee that every answer is correct or resist every prompt injection.