"""Step 2: type-aware chunking.
- PDF / TXT  -> split by TOKENS (the unit LLMs and embedding models actually count)
- CSV        -> one row = one chunk (cutting a row in half would destroy its meaning)"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_TOKENS, MIN_CHUNK_CHARS


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    # Tries to cut at paragraph, then line, then sentence, then word boundaries,
    # while measuring chunk length in tokens.
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
    )


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = get_text_splitter()
    chunks: list[Document] = []
    for doc in docs:
        if doc.metadata["type"] == "csv":
            chunks.append(doc)                       # row-level chunk, keep as is
        else:
            for piece in splitter.split_documents([doc]):    # token-based chunks
                if len(piece.page_content.strip()) >= MIN_CHUNK_CHARS:
                    chunks.append(piece)             # drop tiny fragments like "21  Week 4:"

    # give every chunk a unique id like "report.pdf::12"
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        n = counters.get(source, 0)
        counters[source] = n + 1
        chunk.metadata["chunk_id"] = f"{source}::{n}"
    return chunks