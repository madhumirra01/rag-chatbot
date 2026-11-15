# RAG Chatbot for Multi-Format Documents

A Retrieval-Augmented Generation (RAG) application that enables users to interact with their documents through natural language questions. It supports PDF, TXT, and CSV files, using semantic search and context-grounded responses to retrieve relevant information.

## Features

- **Multi-Format Document Support:** Process PDF, TXT, and CSV files.
- **Type-Aware Chunking:** Uses token-based text splitting for PDF and TXT documents and row-level chunking for CSV files.
- **Semantic Search:** Converts document chunks into vector embeddings for meaning-based retrieval.
- **Vector Database:** Uses ChromaDB to store and retrieve document embeddings.
- **Context-Grounded Question Answering:** Retrieves relevant document content to provide answers based on the available context.
- **Similarity Filtering:** Uses a similarity threshold to filter potentially irrelevant retrieval results.

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| LLM Framework | LangChain |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Database | ChromaDB |
| LLM | Groq API (openai/gpt-oss-120b) |
| Document Formats | PDF, TXT, CSV |

## How It Works

1. **Document Loading:** Documents are loaded from PDF, TXT, and CSV files.
2. **Text Processing:** PDF and TXT content is split into smaller token-based chunks, while CSV files are processed row by row.
3. **Embedding Generation:** Document chunks are converted into vector representations using the BGE embedding model.
4. **Vector Storage:** Embeddings are stored in ChromaDB for efficient similarity-based retrieval.
5. **Semantic Retrieval:** When a user asks a question, the system retrieves the most relevant document chunks.
6. **Response Generation:** Retrieved context is passed to the language model to generate a context-grounded response.

## Chunking Strategy

The project uses different chunking strategies depending on the document format:

- **PDF and TXT:** Token-based splitting with a chunk size of 300 tokens and an overlap of 50 tokens.
- **CSV:** Each row is treated as an individual chunk to preserve tabular information.
- Small text fragments below 40 characters are filtered out.


## Future Improvements

- Support for additional document formats.
- Improved retrieval using hybrid search.
- Conversation history and multi-turn question answering.
- Enhanced document filtering and retrieval evaluation.

---
