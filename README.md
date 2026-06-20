# Document-Based Chatbot

A Streamlit app that lets you upload PDF documents and ask questions grounded strictly in your uploaded content.

## Features

- Upload one or more PDF files via the sidebar
- Semantic search over document chunks using ChromaDB and Sentence Transformers
- Multi-part question decomposition with Ollama (llama3.2)
- Streaming answers with source citations

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed with the `llama3.2` model pulled locally

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.2
```

## Run

```bash
streamlit run app.py
```

## Usage

1. Upload PDFs from the sidebar.
2. Ask questions in the chat input.
3. The assistant answers only from your uploaded documents and reports when information is missing.
