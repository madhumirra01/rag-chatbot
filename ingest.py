"""Run this whenever your documents change:  python ingest.py"""
from collections import Counter

from src.chunking import chunk_documents
from src.config import DOCS_DIR
from src.loaders import load_documents
from src.vectorstore import build_vectorstore

DOCS_DIR.mkdir(parents=True, exist_ok=True)
docs = load_documents(DOCS_DIR)
if not docs:
    raise SystemExit(f"No documents found. Put PDF/TXT/CSV files in {DOCS_DIR}")

chunks = chunk_documents(docs)
print(f"\n{len(docs)} raw units -> {len(chunks)} chunks")
print("chunks by type:  ", dict(Counter(c.metadata["type"] for c in chunks)))
print("chunks by source:", dict(Counter(c.metadata["source"] for c in chunks)))

build_vectorstore(chunks)
print("\nDone. Vector database saved to chroma_db/")