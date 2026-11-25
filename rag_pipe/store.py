"""Embeddings (Ollama via OpenAI-compatible API) + Qdrant vector store."""

import atexit
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, Modifier, SparseVectorParams, VectorParams

from .config import settings


class PrefixedEmbeddings(OpenAIEmbeddings):
    """Some embedding models (nomic-embed-text) are trained with task prefixes and
    retrieve much worse without them."""

    query_prefix: str = ""
    doc_prefix: str = ""

    def embed_documents(self, texts, chunk_size=None, **kwargs):
        return super().embed_documents([self.doc_prefix + t for t in texts], chunk_size, **kwargs)

    def embed_query(self, text, **kwargs):
        # Parent's embed_query delegates to embed_documents; bypass ours to avoid double-prefixing.
        return super().embed_documents([self.query_prefix + text], **kwargs)[0]


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    nomic = "nomic-embed" in settings.embed_model
    return PrefixedEmbeddings(
        query_prefix="search_query: " if nomic else "",
        doc_prefix="search_document: " if nomic else "",
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
        api_key="ollama",  # required by the client, ignored by Ollama
        # Ollama wants raw strings; skip OpenAI's tiktoken pre-tokenization.
        check_embedding_ctx_length=False,
    )


@lru_cache
def get_client() -> QdrantClient:
    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5)
        client.get_collections()
        print(f"[qdrant] server at {settings.qdrant_url}")
        return client
    except Exception:
        print(f"[qdrant] server unreachable, using embedded store at {settings.qdrant_path}")
        try:
            client = QdrantClient(path=settings.qdrant_path)
        except RuntimeError as e:
            if "already accessed" in str(e):
                raise SystemExit(
                    f"[qdrant] {settings.qdrant_path} is locked by another running process "
                    "(embedded mode allows one at a time). Close the other chat/ask, or run a Qdrant server."
                ) from None
            raise
        atexit.register(client.close)  # release the file lock before interpreter teardown
        return client


DENSE, SPARSE = "dense", "sparse"


@lru_cache
def get_sparse_embeddings() -> FastEmbedSparse:
    # BM25 keyword vectors: catches exact names/terms that dense embeddings blur.
    return FastEmbedSparse(model_name="Qdrant/bm25")


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    client = get_client()
    if not client.collection_exists(settings.collection):
        dim = len(get_embeddings().embed_query("dimension probe"))
        client.create_collection(
            settings.collection,
            vectors_config={DENSE: VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE: SparseVectorParams(modifier=Modifier.IDF)},
        )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE,
        sparse_vector_name=SPARSE,
    )


def reset_collection() -> None:
    client = get_client()
    if client.collection_exists(settings.collection):
        client.delete_collection(settings.collection)
    get_vector_store.cache_clear()
