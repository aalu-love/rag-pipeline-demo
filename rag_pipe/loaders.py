"""Load web pages and local files into LangChain Documents."""

from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

TEXT_SUFFIXES = {".txt", ".md", ".html", ".htm", ".rst", ".py", ".json", ".csv"}


def _html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    root = soup.find("main") or soup.find("article") or soup.body or soup
    lines = (line.strip() for line in root.get_text("\n").splitlines())
    return title, "\n".join(line for line in lines if line)


def load_url(url: str) -> Document:
    resp = requests.get(url, timeout=30, headers={"User-Agent": "rag-pipe/0.1"})
    resp.raise_for_status()
    title, text = _html_to_text(resp.text)
    return Document(page_content=text, metadata={"source": url, "title": title})


def load_file(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".html", ".htm"}:
        title, text = _html_to_text(raw)
    else:
        title, text = path.stem, raw
    return Document(page_content=text, metadata={"source": str(path), "title": title})


def load_sources(sources: list[str]) -> list[Document]:
    """Each source may be a URL, a file, or a directory (scanned recursively)."""
    docs: list[Document] = []
    for src in sources:
        if src.startswith(("http://", "https://")):
            docs.append(load_url(src))
            continue
        p = Path(src)
        if p.is_dir():
            files = [f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES]
            docs.extend(load_file(f) for f in files)
        elif p.is_file():
            docs.append(load_file(p))
        else:
            raise FileNotFoundError(f"Not a URL, file, or directory: {src}")
    return [d for d in docs if d.page_content.strip()]
