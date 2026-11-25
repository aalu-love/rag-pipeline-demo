"""Ingestion (load -> split -> embed -> store) and retrieval-augmented answering."""

import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .chunking import split_documents
from .config import settings
from .loaders import load_sources
from .store import get_vector_store

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about a document using the excerpts below. The excerpts may be "
            "from a screenplay: character names in CAPS introduce their dialogue, and other lines "
            "describe the action. Read every excerpt carefully; the answer is often spread across "
            "dialogue and action lines. Answer directly and concretely, citing excerpts like [2]. "
            "Only say you don't know if none of the excerpts are relevant.\n\nExcerpts:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def ingest(sources: list[str]) -> int:
    docs = load_sources(sources)
    chunks = split_documents(docs)
    # Deterministic IDs: re-ingesting the same source overwrites instead of duplicating.
    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.metadata['source']}#{c.metadata['start_index']}"))
        for c in chunks
    ]
    get_vector_store().add_documents(chunks, ids=ids)
    print(f"[ingest] {len(docs)} document(s) -> {len(chunks)} chunk(s)")
    return len(chunks)


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {d.metadata.get('source')})\n{d.page_content}" for i, d in enumerate(docs, 1)
    )


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        api_key="ollama",
        temperature=0,
    )


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write search queries for a document index. Given a question, write {n} short "
            "alternative search queries that use the words the source text itself would likely "
            "contain (names, places, actions, concrete nouns) rather than abstract wording. "
            "Output one query per line, no numbering, nothing else.",
        ),
        ("human", "{question}"),
    ]
)


def rewrite_queries(question: str) -> list[str]:
    if settings.multi_query <= 0:
        return [question]
    out = (REWRITE_PROMPT | get_llm() | StrOutputParser()).invoke(
        {"question": question, "n": settings.multi_query}
    )
    alts = [q.strip(" -*\t\"'") for q in out.splitlines() if q.strip()]
    return [question, *alts[: settings.multi_query]]


def _doc_key(d: Document) -> tuple:
    return d.metadata.get("source"), d.metadata.get("start_index")


def retrieve(question: str, verbose: bool = False) -> list[Document]:
    """Hybrid (dense + BM25) search for the question and its rewrites, merged by
    reciprocal rank fusion so chunks found by several queries rise to the top."""
    store = get_vector_store()
    queries = rewrite_queries(question)
    if verbose:
        print("[retrieve] queries:", *queries, sep="\n  ")
    scores: dict[tuple, float] = {}
    docs: dict[tuple, Document] = {}
    for q in queries:
        for rank, d in enumerate(store.similarity_search(q, k=settings.fetch_k)):
            key = _doc_key(d)
            docs[key] = d
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
    best = sorted(scores, key=scores.get, reverse=True)[: settings.top_k]
    # Present in document order so adjacent scenes read as a continuous story.
    best.sort(key=lambda k: (str(k[0]), k[1] or 0))
    return [docs[k] for k in best]


def answer(question: str, stream: bool = True, verbose: bool = False) -> tuple[str, list[Document]]:
    docs = retrieve(question, verbose=verbose)
    if not docs:
        return "No documents in the index yet. Run `ingest` first.", []
    chain = PROMPT | get_llm() | StrOutputParser()
    inputs = {"context": _format_context(docs), "question": question}
    if not stream:
        return chain.invoke(inputs), docs
    parts = []
    for token in chain.stream(inputs):
        print(token, end="", flush=True)
        parts.append(token)
    print()
    return "".join(parts), docs
