
# src/search.py

from typing import Any, List

from sentence_transformers import SentenceTransformer
from src.vectorstore import VectorStore


class DocumentSearch:
    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.vector_store = vector_store

        # Load the same model used for document embeddings
        self.model = SentenceTransformer(model_name)

        # Access the stored ChromaDB collection
        self.collection = vector_store.get_collection()

        print("[INFO] Document search initialized.")

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Any]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Convert the user's question into a vector
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        # Find the closest document chunks
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        # Convert the search results into readable dictionaries
        retrieved_documents = []

        if results["documents"] and results["documents"][0]:
            for i, text in enumerate(results["documents"][0]):
                retrieved_documents.append(
                    {
                        "content": text,
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )

        print(
            f"[INFO] Retrieved {len(retrieved_documents)} chunks."
        )

        return retrieved_documents
