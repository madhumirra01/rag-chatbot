"""Step 4: find the chunks most similar to the question, and apply the confidence threshold."""
from .config import SIM_THRESHOLD, TOP_K


def retrieve(store, query: str, k: int = TOP_K) -> list[dict]:
    """Returns [{'doc': Document, 'score': similarity}] sorted best-first.
    The store uses cosine distance, so similarity = 1 - distance."""
    results = store.similarity_search_with_score(query, k=k)
    hits = [{"doc": doc, "score": 1 - distance} for doc, distance in results]
    return sorted(hits, key=lambda h: h["score"], reverse=True)


def filter_by_threshold(hits: list[dict], threshold: float = SIM_THRESHOLD) -> list[dict]:
    """Keep only chunks that are similar enough. If nothing survives, we refuse to answer
    instead of letting the LLM guess - this is what reduces hallucinations."""
    return [h for h in hits if h["score"] >= threshold]