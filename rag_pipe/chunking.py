"""Chunking strategies, built on langchain-text-splitters.

- recursive: split on paragraph -> line -> word, CHUNK_SIZE characters.
- scene: same splitter, but scene headings (INT/EXT ...) are the first separator, so
  chunks break at scene boundaries; only oversized scenes fall back to line/word cuts.
- auto: scene if the document has enough scene headings, else recursive.

add_start_index records each chunk's offset in the source, used for deterministic IDs
and for putting retrieved chunks back in story order.
"""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

SCENE_HEADING = re.compile(r"^(?:INT|EXT|I/E)\b", re.M)
MIN_SCENE_HEADINGS = 5  # below this, the doc probably isn't a screenplay


def _splitter(scene: bool) -> RecursiveCharacterTextSplitter:
    if scene:
        return RecursiveCharacterTextSplitter(
            # Lookahead keeps the heading at the start of the next chunk.
            separators=[r"\n(?=(?:INT|EXT|I/E)\b)", r"\n\n", r"\n", " ", ""],
            is_separator_regex=True,
            chunk_size=settings.scene_max_chars,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )


def split_documents(docs: list[Document]) -> list[Document]:
    out: list[Document] = []
    for doc in docs:
        n_headings = len(SCENE_HEADING.findall(doc.page_content))
        scene = settings.splitter == "scene" or (
            settings.splitter == "auto" and n_headings >= MIN_SCENE_HEADINGS
        )
        chunks = _splitter(scene).split_documents([doc])
        print(f"[chunk] {doc.metadata.get('source')}: {'scene' if scene else 'recursive'} -> {len(chunks)} chunks")
        out.extend(chunks)
    return out
