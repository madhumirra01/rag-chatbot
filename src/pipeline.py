"""One function that runs the whole RAG flow. The CLI, the Streamlit app and the
evaluation script all call ask(), so they always behave the same."""
from .config import SIM_THRESHOLD
from .generation import REFUSAL, answer_question
from .query_rewrite import rewrite_question
from .retrieval import filter_by_threshold, retrieve


def ask(store, question: str, threshold: float = SIM_THRESHOLD, history: list[dict] | None = None) -> dict:
    """history = earlier chat messages [{'role': 'user'|'assistant', 'content': str}, ...]"""
    standalone = rewrite_question(question, history or [])     # resolve "it", "that", ...

    hits = retrieve(store, standalone)
    top_score = hits[0]["score"] if hits else 0.0
    good_hits = filter_by_threshold(hits, threshold)

    if not good_hits:                      # nothing similar enough: refuse, don't call the LLM
        return {"answer": REFUSAL, "citations": [], "refused": True,
                "top_score": top_score, "rewritten": standalone}

    result = answer_question(standalone, good_hits)
    result["refused"] = REFUSAL in result["answer"]
    result["top_score"] = top_score
    result["rewritten"] = standalone
    return result