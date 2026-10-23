
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_retrieval import get_chroma_client, search_document

COLLECTION_NAME = "doc_07868648f3f2b07475b3"
TOP_K = 4

# Relevant pages are based on the evidence we've already inspected.
TEST_CASES = [
    {
        "question": (
            "What are Command and Control, Exfiltration, "
            "and Impact according to this document?"
        ),
        "relevant_pages": {24},
    },
    {
        "question": (
            "Which tactics are listed in the MITRE ATT&CK framework?"
        ),
        "relevant_pages": {21},
    },
    {
        "question": (
            "How can adversary behavior be identified in a narrative report?"
        ),
        "relevant_pages": {12},
    },

    {
        "question": "What are the key strategies for identifying techniques and sub-techniques?",
        "relevant_pages": {31, 32, 33},
    },
    {
        "question": "What techniques are associated with the behavior of using email attachments and creating scheduled tasks?",
        "relevant_pages": {35},
    },

]


def semantic_search(collection, question, top_k):
    result = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    return [
        metadata["page_number"]
        for metadata in result["metadatas"][0]
    ]


def recall_at_k(retrieved_pages, relevant_pages):
    if not relevant_pages:
        return 0.0

    hits = set(retrieved_pages) & set(relevant_pages)
    return len(hits) / len(relevant_pages)

def reciprocal_rank(retrieved_pages, relevant_pages):
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant_pages:
            return 1.0 / rank
    return 0.0



def main():
    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)
    semantic_scores = []
    hybrid_scores = []
    semantic_mrr_scores = []
    hybrid_mrr_scores = []

    print(f"Evaluating collection: {COLLECTION_NAME}")
    print(f"Total chunks: {collection.count()}")
    print(f"Top K: {TOP_K}\n")

    for index, case in enumerate(TEST_CASES, start=1):
        question = case["question"]
        relevant_pages = case["relevant_pages"]

        semantic_pages = semantic_search(
            collection, question, TOP_K
        )

        hybrid_results = search_document(
            COLLECTION_NAME, question, top_k=TOP_K
        )
        hybrid_pages = [
            result["page_number"]
            for result in hybrid_results
        ]

        semantic_score = recall_at_k(
            semantic_pages, relevant_pages
        )
        hybrid_score = recall_at_k(
            hybrid_pages, relevant_pages
        )

        semantic_scores.append(semantic_score)
        hybrid_scores.append(hybrid_score)

        semantic_mrr = reciprocal_rank(
            semantic_pages, relevant_pages
        )
        hybrid_mrr = reciprocal_rank(
            hybrid_pages, relevant_pages
        )

        semantic_mrr_scores.append(semantic_mrr)
        hybrid_mrr_scores.append(hybrid_mrr)


        print(f"Test {index}: {question}")
        print(f"Expected relevant pages: {sorted(relevant_pages)}")
        print(f"Semantic top pages: {semantic_pages}")
        print(f"Hybrid top pages:   {hybrid_pages}")
        print(f"Semantic Recall@{TOP_K}: {semantic_score:.2f}")
        print(f"Semantic MRR: {semantic_mrr:.2f}")
        print(f"Hybrid MRR:   {hybrid_mrr:.2f}")
        print(f"Hybrid Recall@{TOP_K}:   {hybrid_score:.2f}")
        print("-" * 60)

    print("\nSUMMARY")
    print(
        "Average semantic Recall@{}: {:.2f}".format(
            TOP_K, sum(semantic_scores) / len(semantic_scores)
        )
    )
    print(
        "Average hybrid Recall@{}:   {:.2f}".format(
            TOP_K, sum(hybrid_scores) / len(hybrid_scores)
        )
    )
    print(
    f"Average semantic MRR: "
    f"{sum(semantic_mrr_scores) / len(semantic_mrr_scores):.2f}"
    )
    print(
        f"Average hybrid MRR:   "
        f"{sum(hybrid_mrr_scores) / len(hybrid_mrr_scores):.2f}"
    )


if __name__ == "__main__":
    main()
