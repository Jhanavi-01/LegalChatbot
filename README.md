# LegalChatbot


LegalChatbot is an AI-powered system that helps users analyze legal case documents. It extracts meaningful summaries, identifies legal provisions, and retrieves similar past cases using Retrieval-Augmented Generation (RAG).

## Features

Converts user-uploaded legal case PDFs into embeddings and stores them in ChromaDB.

Summarizes cases in a structured format (case name, jurisdiction, legal provisions, key facts, etc.).

Identifies IPC sections and key legal terms.

Retrieves similar previous cases from the IL-TUR dataset.

Uses Mistral (via Ollama) for text generation.

Uses nomic-embed-text for embedding generation.

Supports OCR to extract text from images in PDFs.

## Tech Stack

Python (Backend Logic)

Streamlit (UI Framework)

ChromaDB (Vector Database for Retrieval)

Ollama (Locally Running LLM & Embedding Model)

Docker (Containerization)

