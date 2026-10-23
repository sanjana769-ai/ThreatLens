
from retrieval.hybrid_search import HybridSearch
from llm.groq_client import generate_answer


def retrieve_context(search_engine, question, top_k=5):
    results = search_engine.search(question, top_k=top_k)

    context_parts = []
    for result in results:
        context_parts.append(
            f"Source ID: {result['id']}\n"
            f"Technique: {result['name']}\n"
            f"Tactics: {', '.join(result['tactics'])}\n"
            f"Description: {result['description']}\n"
            f"Source URL: {result['url']}"
        )

    return "\n\n---\n\n".join(context_parts), results



def run_test(search_engine, question):
    context, results = retrieve_context(search_engine, question)

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    print("\nRETRIEVED SOURCES:")
    for result in results:
        print(f"- {result['id']}: {result['name']}")

    answer = generate_answer(question, context)

    print("\nANSWER:")
    print(answer)

    valid_ids = {result["id"] for result in results}
    cited_ids = set()

    for result in results:
        technique_id = result["id"]
        if f"[{technique_id}]" in answer:
            cited_ids.add(technique_id)

    invalid_citations = []
    import re

    for citation in re.findall(r"\[([^\]]+)\]", answer):
        if citation not in valid_ids:
            invalid_citations.append(citation)

    print("\nCITATION CHECK:")
    print("Retrieved IDs:", sorted(valid_ids))
    print("Matching retrieved IDs:", sorted(cited_ids))
    print("Invalid or unsupported citations:", invalid_citations)
    print("Has matching citation:", bool(cited_ids))
    print("Citation check:", "PASS" if cited_ids and not invalid_citations else "FAIL")



def main():
    search_engine = HybridSearch()

    run_test(
        search_engine,
        "What techniques can attackers use to steal credentials?"
    )

    run_test(
        search_engine,
        "What is the official ThreatLens detection signature "
        "for the fictional Zorblax-9000 attack?"
    )


if __name__ == "__main__":
    main()
