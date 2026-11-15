"""Chat with your documents in the terminal:  python ask.py"""
from src.pipeline import ask
from src.vectorstore import load_vectorstore

store = load_vectorstore()
print("Ask questions about your documents. Type 'quit' to exit.")
while True:
    question = input("\nYou: ").strip()
    if question.lower() in {"", "quit", "exit"}:
        break
    result = ask(store, question)
    print(f"\n(top similarity {result['top_score']:.3f}, refused={result['refused']})")
    print("Answer:", result["answer"])
    for c in result["citations"]:
        print(f"  [{c['number']}] {c['source']} | {c['location']} | similarity {c['score']}")
        print(f"      {c['snippet']}")