
from retrieval.hybrid_search import HybridSearch

search_engine = HybridSearch()

queries = [
    "T1001",
    "What techniques can attackers use to steal credentials?",
    "command and control obfuscation",
]

for query in queries:
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    results = search_engine.search(query, top_k=5)

    for rank, result in enumerate(results, start=1):
        print(
            f"\n{rank}. {result['id']} — {result['name']}"
        )
        print(f"   Tactics: {', '.join(result['tactics'])}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Source: {result['url']}")
        print(f"   Description: {result['description'][:250]}...")
