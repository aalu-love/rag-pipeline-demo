"""RAG pipeline CLI.

  uv run main.py ingest <url|file|dir> [...]   # add sources to Qdrant
  uv run main.py ask "your question"           # one-shot question
  uv run main.py chat                          # interactive loop
  uv run main.py reset                         # drop the collection
"""

import argparse

from rag_pipe.pipeline import answer, ingest
from rag_pipe.store import reset_collection


def print_sources(docs) -> None:
    print("\nSources:")
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] {d.metadata.get('source')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG with Ollama + Qdrant")
    parser.add_argument("-v", "--verbose", action="store_true", help="show rewritten search queries")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="Load, chunk, embed and store sources")
    p_ing.add_argument("sources", nargs="+")
    p_ask = sub.add_parser("ask", help="Ask one question")
    p_ask.add_argument("question")
    sub.add_parser("chat", help="Interactive Q&A")
    sub.add_parser("reset", help="Delete the collection")
    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest(args.sources)
    elif args.cmd == "ask":
        _, docs = answer(args.question, verbose=args.verbose)
        print_sources(docs)
    elif args.cmd == "chat":
        print("Ask a question (empty line or Ctrl+C to quit).")
        try:
            while q := input("\n> ").strip():
                _, docs = answer(q, verbose=args.verbose)
                print_sources(docs)
        except (KeyboardInterrupt, EOFError):
            pass
    elif args.cmd == "reset":
        reset_collection()
        print("Collection deleted.")


if __name__ == "__main__":
    main()
