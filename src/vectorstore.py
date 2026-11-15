"""Step 3: embeddings + ChromaDB.
An embedding turns text into a list of numbers so that texts with similar MEANING
end up close together. ChromaDB stores those numbers and finds the closest ones."""
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import CHROMA_DIR, COLLECTION, EMBED_MODEL


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings  # imported here: slow import
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},   # needed for cosine similarity
    )


def _store(embeddings) -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},   # score = 1 - distance = similarity
    )


def build_vectorstore(chunks: list[Document], embeddings=None) -> Chroma:
    """Rebuilds the database from scratch each time (simple and repeatable)."""
    embeddings = embeddings or get_embeddings()
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    store = _store(embeddings)
    ids = [c.metadata["chunk_id"] for c in chunks]
    for i in range(0, len(chunks), 500):                 # add in batches
        store.add_documents(chunks[i:i + 500], ids=ids[i:i + 500])
    return store


def load_vectorstore(embeddings=None) -> Chroma:
    return _store(embeddings or get_embeddings())


# ---- helpers used by the chat app (add / remove / list documents) ----

def count(store: Chroma) -> int:
    """Number of chunks currently stored."""
    return store._collection.count()


def list_sources(store: Chroma) -> dict[str, int]:
    """{'report.pdf': 36, 'sales.csv': 200} - which files are indexed and how many chunks each."""
    metadatas = store.get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    for meta in metadatas:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return counts


def remove_source(store: Chroma, source: str) -> None:
    ids = store.get(where={"source": source})["ids"]
    if ids:
        store.delete(ids=ids)


def add_chunks(store: Chroma, chunks: list[Document], progress=None) -> None:
    """Add chunks in small batches; progress(fraction) lets the UI show a progress bar."""
    ids = [c.metadata["chunk_id"] for c in chunks]
    step = 200
    for i in range(0, len(chunks), step):
        store.add_documents(chunks[i:i + step], ids=ids[i:i + step])
        if progress:
            progress(min(1.0, (i + step) / max(len(chunks), 1)))


def clear_vectorstore(store: Chroma) -> None:
    store.delete_collection()