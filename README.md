# RAG pipeline: LangChain + Ollama + Qdrant

Fully local: Ollama serves the chat model and embeddings through its OpenAI-compatible API (`/v1`), so `langchain-openai` drives both. Qdrant stores the vectors.

## Setup

```bash
ollama pull llama3.2          # chat model (or qwen3.5, etc.)
ollama pull nomic-embed-text  # embedding model (768-dim)
uv sync
cp .env.example .env          # edit models / Qdrant URL as needed
```

Qdrant: run a server (for example `docker run -p 6333:6333 qdrant/qdrant`). If `QDRANT_URL` is unreachable, the app falls back to an embedded on-disk store at `./qdrant_data`.

## Usage

```bash
uv run main.py ingest data/ https://example.com/page   # URLs, files, or folders
uv run main.py ask "your question"
uv run main.py chat
uv run main.py reset        # needed if you change EMBED_MODEL (vector size changes)
```

## Layout

- `rag_pipe/loaders.py`: URL (requests + BeautifulSoup) and local file loading
- `rag_pipe/store.py`: Ollama dense embeddings (with nomic task prefixes) + BM25 sparse vectors (fastembed), Qdrant hybrid collection
- `rag_pipe/pipeline.py`: chunking, ingest with deterministic IDs (re-ingesting doesn't duplicate), multi-query retrieval with reciprocal rank fusion, and the answer chain

## Retrieval

Each question is rewritten by the LLM into `MULTI_QUERY` extra search queries (set `0` to disable). Every query runs a hybrid dense + BM25 search in Qdrant, the results are merged by reciprocal rank fusion, and the top `TOP_K` chunks go to the LLM in document order. Add `-v` to see the rewritten queries, e.g. `uv run main.py -v chat`.

The embedded Qdrant store (`./qdrant_data`) can only be opened by one process at a time, so close any running `chat` before running `ask` or `ingest`.
