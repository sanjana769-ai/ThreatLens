
import hashlib
import re
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Load reusable local components
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the embedding model once per Python process."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_chroma_client():
    """Create a persistent local vector database."""
    return chromadb.PersistentClient(path="./chroma_db")


# --------------------------------------------------
# 2. Tokenize text for BM25 keyword retrieval
# --------------------------------------------------

def tokenize(text):
    """Tokenize text while preserving cybersecurity identifiers."""
    return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", text.lower())


def clean_pdf_text(text):
    """Normalize extracted PDF text before chunking."""

    if not isinstance(text, str):
        return ""

    # Normalize whitespace and common PDF extraction artifacts.
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[\u25a0-\u25ff\u2610-\u2612]", " ", text)

    # Remove repeated copyright notices.
    text = re.sub(
        r"©\s*2024\s+THE MITRE CORPORATION\.?\s*ALL RIGHTS RESERVED\.?",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize spaces and line breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# 3. Split documents into overlapping chunks
# --------------------------------------------------

def chunk_pages(pages, chunk_size=180, overlap=40):
    """
    Split extracted PDF pages into overlapping word chunks.
    Preserve each chunk's original page number.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size or overlap.")

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        text = page.get("text", "")
        text = clean_pdf_text(page.get("text", ""))

        if not text:
            continue

        words = re.findall(r"\S+", text)

        for start in range(0, len(words), step):
            section = words[start:start + chunk_size]

            if not section:
                continue

            chunks.append({
                "text": " ".join(section),
                "page_number": page["page_number"],
            })

            if start + chunk_size >= len(words):
                break

    return chunks


# --------------------------------------------------
# 4. Index chunks in ChromaDB
# --------------------------------------------------

def index_document(pages, source_name):
    """Chunk, embed, and index one uploaded document."""

    if not isinstance(source_name, str) or not source_name.strip():
        raise ValueError("A valid source name is required.")

    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("No text available to index.")

    model = get_embedding_model()
    client = get_chroma_client()

    content_hash = hashlib.sha256(
        (
            source_name
            + "\n"
            + "\n".join(
                page.get("text", "")
                for page in pages
                if isinstance(page.get("text", ""), str)
            )
        ).encode("utf-8")
    ).hexdigest()[:20]

    collection_name = f"doc_{content_hash}"

    existing = {
        collection.name
        for collection in client.list_collections()
    }

    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    documents = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {
                "source": source_name,
                "page_number": chunk["page_number"],
            }
            for chunk in chunks
        ],
    )

    return collection_name, len(chunks)


# --------------------------------------------------
# 5. Retrieve evidence using semantic + BM25 search
# --------------------------------------------------

def search_document(collection_name, question, top_k=4, candidate_k=12):
    """
    Retrieve PDF chunks with semantic search and BM25,
    then combine their rankings using Reciprocal Rank Fusion.

    Returns the same core fields as before:
    text, source, page_number, distance.

    Also includes hybrid_score and retrieval_rank.
    """

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer.")

    if (
        not isinstance(candidate_k, int)
        or isinstance(candidate_k, bool)
        or candidate_k <= 0
    ):
        raise ValueError("candidate_k must be a positive integer.")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a valid question to search.")

    client = get_chroma_client()
    collection = client.get_collection(name=collection_name)
    chunk_count = collection.count()

    if chunk_count == 0:
        return []

    # Retrieve a larger candidate pool before fusion.
    fetch_k = min(max(top_k, candidate_k), chunk_count)

    model = get_embedding_model()
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    semantic_result = collection.query(
        query_embeddings=query_embedding,
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    semantic_documents = semantic_result["documents"][0]
    semantic_metadatas = semantic_result["metadatas"][0]
    semantic_distances = semantic_result["distances"][0]

    # Load all indexed chunks so BM25 searches the full PDF.
    all_records = collection.get(
        include=["documents", "metadatas"],
    )

    all_documents = all_records["documents"] or []
    all_metadatas = all_records["metadatas"] or []
    all_ids = all_records["ids"]

    if not all_documents:
        return []

    tokenized_documents = [
        tokenize(document)
        for document in all_documents
    ]

    bm25 = BM25Okapi(tokenized_documents)
    keyword_scores = bm25.get_scores(tokenize(question))

    keyword_ranked_indices = sorted(
        range(len(keyword_scores)),
        key=lambda i: keyword_scores[i],
        reverse=True,
    )[:fetch_k]

    # RRF combines rankings rather than incomparable raw scores.
    rrf_constant = 60
    fused_scores = {}
    record_lookup = {}

    # Add semantic results.
    for rank, (document, metadata, distance) in enumerate(
        zip(
            semantic_documents,
            semantic_metadatas,
            semantic_distances,
        ),
        start=1,
    ):
        key = (
            metadata["source"],
            metadata["page_number"],
            document,
        )

        fused_scores[key] = fused_scores.get(key, 0.0) + (
            1.0 / (rrf_constant + rank)
        )

        record_lookup[key] = {
            "text": document,
            "source": metadata["source"],
            "page_number": metadata["page_number"],
            "distance": distance,
        }

    # Add BM25 results, preserving source/page metadata.
    for rank, index in enumerate(keyword_ranked_indices, start=1):
        document = all_documents[index]
        metadata = all_metadatas[index]

        key = (
            metadata["source"],
            metadata["page_number"],
            document,
        )

        fused_scores[key] = fused_scores.get(key, 0.0) + (
            1.0 / (rrf_constant + rank)
        )

        if key not in record_lookup:
            record_lookup[key] = {
                "text": document,
                "source": metadata["source"],
                "page_number": metadata["page_number"],
                "distance": None,
            }

    ranked_keys = sorted(
        fused_scores,
        key=lambda key: fused_scores[key],
        reverse=True,
    )

    matches = []

    for rank, key in enumerate(ranked_keys[:top_k], start=1):
        match = record_lookup[key].copy()
        match["hybrid_score"] = fused_scores[key]
        match["retrieval_rank"] = rank
        matches.append(match)

    return matches
