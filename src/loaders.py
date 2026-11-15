"""Step 1: read files and turn them into LangChain Document objects.
A Document = text + metadata (where the text came from). The metadata is what
lets us show citations later."""
from pathlib import Path

import pymupdf  # PyMuPDF
import pandas as pd
from langchain_core.documents import Document

from .config import MAX_CSV_ROWS


def load_pdf(path: Path) -> list[Document]:
    """One Document per PDF page, so we can cite page numbers."""
    docs = []
    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if text:  # skip blank pages
                docs.append(Document(
                    page_content=text,
                    metadata={"source": path.name, "page": page_number, "type": "pdf"},
                ))
    return docs


def load_txt(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [Document(page_content=text,
                     metadata={"source": path.name, "page": 1, "type": "txt"})]


def load_csv(path: Path) -> list[Document]:
    """One Document per ROW, written as 'column: value; column: value'.
    Keeping the column name next to each value lets the embedding model
    understand what each number means."""
    df = pd.read_csv(path, nrows=MAX_CSV_ROWS)     # cap very large files
    docs = []
    for row_number, row in enumerate(df.to_dict("records"), start=1):
        parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        if parts:
            docs.append(Document(
                page_content="; ".join(parts),
                metadata={"source": path.name, "row": row_number, "type": "csv"},
            ))
    return docs


LOADERS = {".pdf": load_pdf, ".txt": load_txt, ".csv": load_csv}


def load_documents(folder: Path) -> list[Document]:
    docs = []
    for path in sorted(folder.iterdir()):
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            print(f"skipping unsupported file: {path.name}")
            continue
        loaded = loader(path)
        print(f"loaded {path.name}: {len(loaded)} units")
        docs.extend(loaded)
    return docs