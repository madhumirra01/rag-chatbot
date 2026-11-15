"""Step 5: ask the LLM to answer using ONLY the retrieved chunks, and extract citations.
All LLM calls live in call_llm(), so switching provider later means editing one function."""
import re

from groq import Groq, RateLimitError

from .config import GROQ_API_KEY, LLM_MODEL

REFUSAL = "I could not find this in the documents."
MISSING_KEY_MSG = "GROQ_API_KEY is missing. Add it to your .env file and restart."
RATE_LIMIT_MSG = "The free LLM limit was reached. Wait a minute and try again."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered context chunks provided.
Rules:
- If the context does not contain the answer, reply exactly: {REFUSAL}
- Cite the chunks you used with their numbers in square brackets, like [1] or [2][3].
- Never use outside knowledge. Be concise."""


def call_llm(system: str, user: str) -> str:
    if not GROQ_API_KEY:
        return MISSING_KEY_MSG
    client = Groq(api_key=GROQ_API_KEY)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0,          # same question -> same answer (needed for evaluation)
        )
    except RateLimitError:
        return RATE_LIMIT_MSG
    return (response.choices[0].message.content or "").strip()


def location(meta: dict) -> str:
    return f"page {meta['page']}" if "page" in meta else f"row {meta['row']}"


def build_context(hits: list[dict]) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        meta = hit["doc"].metadata
        blocks.append(f"[{number}] (source: {meta['source']}, {location(meta)})\n"
                      f"{hit['doc'].page_content}")
    return "\n\n".join(blocks)


def answer_question(question: str, hits: list[dict]) -> dict:
    user = f"Context:\n{build_context(hits)}\n\nQuestion: {question}"
    text = call_llm(SYSTEM_PROMPT, user)

    # find the [1] [2] markers the model used and map them back to real chunks
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    citations = []
    for n in used:
        hit = hits[n - 1]
        meta = hit["doc"].metadata
        citations.append({
            "number": n,
            "source": meta["source"],
            "location": location(meta),
            "chunk_id": meta["chunk_id"],
            "score": round(hit["score"], 3),
            "snippet": hit["doc"].page_content[:200].replace("\n", " "),
        })
    return {"answer": text, "citations": citations}