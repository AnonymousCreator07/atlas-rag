# Atlas RAG retrieval behavior

Atlas RAG turns an uploaded document into text chunks, embeds each chunk, and stores the vectors in Chroma. At query time it embeds the question, retrieves dense candidates, and can rerank them with a cross-encoder. The default local embedding model is sentence-transformers/all-MiniLM-L6-v2. The default cross-encoder is cross-encoder/ms-marco-MiniLM-L-6-v2. The default final result count is five passages.

The similarity score reported by the vector store is one minus cosine distance. Changing embedding models requires re-indexing every document because vectors from different embedding spaces cannot be compared safely.