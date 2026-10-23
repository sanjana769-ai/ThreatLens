
import json
import re
from pathlib import Path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "techniques.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridSearch:
    def __init__(self):
        if not DATA_FILE.exists():
            raise FileNotFoundError(
                f"Dataset not found: {DATA_FILE}"
            )

        with DATA_FILE.open("r", encoding="utf-8") as file:
            self.records = json.load(file)

        if not self.records:
            raise ValueError("No technique records found.")

        # Include IDs and metadata to improve keyword retrieval.
        self.documents = [
            " ".join([
                record["id"],
                record["name"],
                record["description"],
                " ".join(record.get("tactics", [])),
                " ".join(record.get("platforms", [])),
                " ".join(record.get("data_sources", [])),
            ])
            for record in self.records
        ]

        self.tokenized_documents = [
            tokenize(document) for document in self.documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_documents)

        print("Loading local embedding model...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)

        print("Embedding technique records...")
        self.document_embeddings = self.encoder.encode(
            self.documents,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        print(
            f"Indexed {len(self.records)} technique records."
        )

    def search(self, query, top_k=5, alpha=0.5):
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")

        # Keyword scores.
        keyword_scores = np.asarray(
            self.bm25.get_scores(tokenize(query)),
            dtype=float,
        )

        # Semantic cosine similarity; vectors are normalized.
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        semantic_scores = (
            self.document_embeddings @ query_embedding
        )

        # Min-max normalize each score list before combining.
        def normalize(scores):
            low = scores.min()
            high = scores.max()

            if high - low < 1e-12:
                return np.zeros_like(scores)

            return (scores - low) / (high - low)

        keyword_norm = normalize(keyword_scores)
        semantic_norm = normalize(semantic_scores)

        combined_scores = (
            alpha * keyword_norm
            + (1 - alpha) * semantic_norm
        )

        # Explicitly favor an exact technique ID when supplied.
        query_ids = set(
            re.findall(r"\bT\d{4}(?:\.\d{3})?\b", query.upper())
        )

        ranked_indices = np.argsort(combined_scores)[::-1]

        results = []

        for index in ranked_indices:
            record = self.records[int(index)]

            results.append({
                **record,
                "score": float(combined_scores[index]),
                "keyword_score": float(keyword_scores[index]),
                "semantic_score": float(semantic_scores[index]),
            })

            if len(results) >= top_k:
                break

        if query_ids:
            exact_matches = [
                result for result in results
                if result["id"] in query_ids
            ]

            remaining = [
                result for result in results
                if result["id"] not in query_ids
            ]

            # Check the full corpus, not only the top-k results.
            for record in self.records:
                if record["id"] in query_ids and not any(
                    item["id"] == record["id"]
                    for item in exact_matches
                ):
                    exact_matches.append({
                        **record,
                        "score": 1.0,
                        "keyword_score": float(
                            keyword_scores[
                                self.records.index(record)
                            ]
                        ),
                        "semantic_score": float(
                            semantic_scores[
                                self.records.index(record)
                            ]
                        ),
                    })

            results = (exact_matches + remaining)[:top_k]

        return results
