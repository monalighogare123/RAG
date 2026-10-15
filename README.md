# RAG Document Search System

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) document retrieval pipeline using Python. It loads PDF documents, splits their content into smaller chunks, generates semantic embeddings using the `all-MiniLM-L6-v2` model, and stores them in a persistent ChromaDB vector database.

Users can ask questions about their documents and retrieve the most relevant text chunks using vector similarity search.

## Tech Stack
- Language: Python
- Document Loading: PyPDFLoader
- Text Splitting: RecursiveCharacterTextSplitter
- Embeddings: Sentence Transformers (`all-MiniLM-L6-v2`)
- Vector Database: ChromaDB
- Search: Cosine similarity

## Key Features
- PDF document loading
- Text chunking with overlapping sections
- Semantic embedding generation
- Persistent vector storage
- Similarity-based document search
- Interactive question-based retrieval

## How to Run
1. Install the required dependencies.
2. Place PDF files inside the `data/` directory.
3. Run the application:

   by using command
   python main.py
   

4. Enter a question to retrieve relevant document chunks.

## Future Enhancements
- Integrate an LLM to generate contextual answers.
- Support additional file formats.
- Build a web interface for document-based question answering.
