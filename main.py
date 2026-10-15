
# main.py

from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import VectorStore
from src.search import DocumentSearch


def main():
    # Step 1: Load documents from the data folder
    print("\n--- Step 1: Loading Documents ---")

    documents = load_all_documents("data")

    if not documents:
        print("No PDF documents found in the data folder.")
        return

    print(f"Loaded {len(documents)} document pages.")


    # Step 2: Split documents and generate embeddings
    print("\n--- Step 2: Creating Embeddings ---")

    pipeline = EmbeddingPipeline(
        model_name="all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = pipeline.chunk_documents(documents)

    if not chunks:
        print("No document chunks were generated.")
        return

    embeddings = pipeline.embed_chunks(chunks)


    # Step 3: Store chunks in ChromaDB
    print("\n--- Step 3: Creating Vector Store ---")

    vector_store = VectorStore(
        persist_directory="data/vector_store",
        collection_name="pdf_documents",
    )

    # Avoid adding duplicate chunks on every run:
    # Rebuild the collection from the current documents.
    vector_store.client.delete_collection("pdf_documents")

    vector_store = VectorStore(
        persist_directory="data/vector_store",
        collection_name="pdf_documents",
    )

    vector_store.add_documents(chunks, embeddings)


    # Step 4: Initialize document search
    print("\n--- Step 4: Initializing Search ---")

    search_engine = DocumentSearch(
        vector_store=vector_store,
        model_name="all-MiniLM-L6-v2",
    )


    # Step 5: Ask questions about your documents
    print("\n--- Step 5: Ask Questions ---")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nEnter your question: ").strip()

        if question.lower() == "exit":
            print("RAG search stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        results = search_engine.search(
            query=question,
            top_k=3,
        )

        if not results:
            print("No relevant document chunks found.")
            continue

        print("\n--- Retrieved Results ---")

        for index, result in enumerate(results, start=1):
            print(f"\nResult {index}")
            print("Content:", result["content"])
            print("Metadata:", result["metadata"])
            print("Cosine distance:", result["distance"])


if __name__ == "__main__":
    main()
