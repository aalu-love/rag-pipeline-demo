import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    chat_model: str = os.getenv("CHAT_MODEL", "llama3.2")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_path: str = os.getenv("QDRANT_PATH", "./qdrant_data")
    collection: str = os.getenv("COLLECTION", "rag_docs")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    splitter: str = os.getenv("SPLITTER", "recursive")  # recursive | scene | auto
    scene_max_chars: int = int(os.getenv("SCENE_MAX_CHARS", "1200"))
    top_k: int = int(os.getenv("TOP_K", "6"))  # chunks sent to the LLM
    fetch_k: int = int(os.getenv("FETCH_K", "10"))  # candidates per query before fusion
    multi_query: int = int(os.getenv("MULTI_QUERY", "3"))  # LLM rewrites per question; 0 = off


settings = Settings()
