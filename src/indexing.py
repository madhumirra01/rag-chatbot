"""Adds ONE file to the existing knowledge base (used by the chat app's upload button)."""
from pathlib import Path

import pandas as pd

from .chunking import chunk_documents
from .config import MAX_CSV_ROWS
from .loaders import LOADERS
from .vectorstore import add_chunks, remove_source


def index_file(store, path: Path, progress=None) -> dict:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    chunks = chunk_documents(loader(path))
    remove_source(store, path.name)          # uploading the same file again replaces the old copy
    add_chunks(store, chunks, progress)

    note = ""
    if path.suffix.lower() == ".csv":
        total_rows = len(pd.read_csv(path, usecols=[0]))
        if total_rows > MAX_CSV_ROWS:
            note = f"Only the first {MAX_CSV_ROWS} of {total_rows} rows were indexed."
    return {"file": path.name, "chunks": len(chunks), "note": note}