"""Step 6: make follow-up questions searchable.
The search engine only sees ONE question at a time. "What about its cost?" means nothing on
its own, so before searching we ask the LLM to rewrite it using the chat history:
    "What about its cost?"  ->  "What is the cost of the volunteer training program?"
"""
from . import generation

REWRITE_SYSTEM = """You rewrite follow-up questions so they can be understood without the chat.
Rules:
- Use the conversation to replace pronouns and vague references (it, that, they, the second one).
- If the question is already clear on its own, return it unchanged.
- Output ONLY the rewritten question, nothing else. Do not answer it."""

MAX_TURNS = 6   # how many recent messages (user + assistant) the rewriter gets to see


def rewrite_question(question: str, history: list[dict]) -> str:
    """Returns a standalone version of the question. Falls back to the original on any problem."""
    if not history:                                   # first question: nothing to resolve
        return question

    transcript = "\n".join(f"{m['role'].capitalize()}: {m['content']}"
                           for m in history[-MAX_TURNS:])
    user = f"Conversation so far:\n{transcript}\n\nFollow-up question: {question}"
    rewritten = generation.call_llm(REWRITE_SYSTEM, user).strip().strip('"').strip()

    failed = (not rewritten
              or rewritten in (generation.MISSING_KEY_MSG, generation.RATE_LIMIT_MSG)
              or len(rewritten) > 400)                # a rewrite should be one short question
    return question if failed else rewritten