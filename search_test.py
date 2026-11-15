"""Check that retrieval works:  python search_test.py "your question" """
import sys

from src.vectorstore import load_vectorstore

query = " ".join(sys.argv[1:]) or input("Question: ")
store = load_vectorstore()
for doc, distance in store.similarity_search_with_score(query, k=5):
    similarity = 1 - distance          # cosine space: distance = 1 - similarity
    where = doc.metadata.get("page") or f"row {doc.metadata.get('row')}"
    print(f"\nsimilarity={similarity:.3f} | {doc.metadata['source']} | {where} | {doc.metadata['type']}")
    print("   ", doc.page_content[:220].replace("\n", " "))