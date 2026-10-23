
from retrieval.hybrid_search import HybridSearch
from llm.groq_client import generate_answer

search_engine = HybridSearch()

question = "What techniques can attackers use to steal credentials?"

results = search_engine.search(question, top_k=5)

context_parts = []

for result in results:
    context_parts.append(
        f"Source ID: {result['id']}\n"
        f"Technique: {result['name']}\n"
        f"Tactics: {', '.join(result['tactics'])}\n"
        f"Description: {result['description']}\n"
        f"Source URL: {result['url']}"
    )

retrieved_context = "\n\n---\n\n".join(context_parts)

print("\n" + "=" * 70)
print("RETRIEVED EVIDENCE")
print("=" * 70)
print(retrieved_context)

print("\n" + "=" * 70)
print("THREATLENS RAG ANSWER")
print("=" * 70)

answer = generate_answer(question, retrieved_context)
print(answer)
