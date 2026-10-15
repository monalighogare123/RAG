
# src/vectorstore.py

from pathlib import Path
from typing import Any, List

import chromadb


class VectorStore:
    def __init__(
        self,
        persist_directory: str = "data/vector_store",
        collection_name: str = "pdf_documents",
    ):
        # Create the directory if it does not exist
        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        # Create a persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(Path(persist_directory).resolve())
        )

        # Create or retrieve the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        print(
            f"[INFO] Vector store ready: {collection_name}"
        )

    def add_documents(
        self,
        chunks: List[Any],
        embeddings,
    ) -> None:
        if not chunks:
            print("[WARNING] No document chunks to store.")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        # Generate unique IDs for each chunk
        existing_count = self.collection.count()
        ids = [
            f"chunk_{existing_count + i}"
            for i in range(len(chunks))
        ]

        # Extract the text from each chunk
        texts = [
            chunk.page_content for chunk in chunks
        ]

        # Chroma metadata supports simple scalar values.
        # Convert unsupported metadata values to strings.
        metadatas = []

        for chunk in chunks:
            metadata = {}

            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                elif value is not None:
                    metadata[key] = str(value)

            metadatas.append(metadata)

        # Store text, embeddings, IDs, and metadata
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        print(
            f"[INFO] Stored {len(chunks)} chunks successfully."
        )

    def get_collection(self):
        # Return the collection for searching
        return self.collection
