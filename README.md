# Local RAG Assistant

A command-line document Q&A application built with Microsoft Foundry Local. It searches local text documents using embeddings and cosine similarity, then passes the retrieved passages to a local language model to generate an answer.

## Overview

The application has two stages:

1. **Ingestion:** Read text files, split them into passages, generate embeddings, and save the passages and vectors in SQLite.
2. **Question answering:** Embed the question, retrieve the three most similar passages, and use them as context for the chat model.

Document embeddings are stored between sessions, so they only need to be rebuilt when documents change.

## Features

- Local embedding generation and language-model inference with Foundry Local.
- Multiple UTF-8 text documents and paragraph-based chunking.
- SQLite storage for text, source filenames, and embedding vectors.
- Cosine similarity search implemented with Python's standard library.
- A relevance threshold that rejects questions with weak document matches.
- Source filenames and response timings displayed with each answer.
- Selection of an available model variant when the preferred execution provider is unavailable.

## Technologies

| Component | Implementation |
| --- | --- |
| Language | Python |
| Model runtime | Microsoft Foundry Local SDK 2.0.1 |
| Embedding model | `qwen3-embedding-0.6b` |
| Chat model | `qwen2.5-0.5b` |
| Storage | SQLite through Python's `sqlite3` module |
| Search | Cosine similarity, top 3 passages |
| Interface | Terminal |

## Project Structure

```text
local-rag-assistant/
├── documents/
│   ├── embeddings.txt
│   ├── foundry_local.txt
│   ├── prompt_engineering.txt
│   ├── rag.txt
│   └── sqlite.txt
├── data/
│   └── .gitkeep
├── ingest.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

Ingestion creates `data/knowledge.db`. The generated database and virtual environment are excluded from version control.

## Installation

Use Python 3.11 or newer on a platform supported by Foundry Local. Initial package and model downloads require internet access and enough disk space for both models.

Download or clone this repository and open a terminal in its directory.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux

On a platform supported by the installed SDK distribution:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

These commands use the virtual environment directly; activation is optional.

## Usage

### 1. Build the knowledge base

On Windows:

```powershell
.\.venv\Scripts\python.exe ingest.py
```

The script reads `documents/*.txt`, groups up to two paragraphs per passage, generates embeddings, and writes them to SQLite. Running it again replaces the indexed contents with the current documents.

### 2. Start the assistant

```powershell
.\.venv\Scripts\python.exe main.py
```

On macOS/Linux, use `./.venv/bin/python` in place of `.\.venv\Scripts\python.exe` for both commands.

The application downloads models if needed and loads them before accepting questions. At the `Question:` prompt, enter a question and press Enter. Type `quit` to exit.

Example questions for the included documents:

```text
What are the three basic steps of RAG?
What does a text embedding represent?
Why is SQLite suitable for small local applications?
```

For the first question, the expected information is **retrieve, augment, and generate**. The wording of the generated answer may vary. Retrieved source filenames appear below the answer.

If the highest similarity score is below `0.60`, the application skips chat generation and returns:

```text
The provided documents do not contain enough information to answer this question.
```

### Adding documents

Save your documents as UTF-8 `.txt` files in `documents/`. Separate paragraphs with a blank line. Run ingestion again after adding, changing, or removing documents, then restart the assistant.

## How It Works

```text
Text documents → Paragraph chunks → Local embeddings → SQLite
                                                        ↓
Question → Query embedding → Cosine similarity → Top 3 passages
                                                        ↓
                              Relevance check → Context + question
                                                        ↓
                                              Local chat model
                                                        ↓
                                              Answer + sources
```

Each stored passage has an ID, source filename, text, and a JSON embedding vector. At startup, the assistant reads the stored passages into memory. Each question uses the same embedding model as ingestion, and the application ranks all passages by cosine similarity.

The system prompt instructs the chat model to answer from the retrieved context and acknowledge missing information. Source names come from the retrieval results. Both models stay loaded during the question loop and are unloaded on normal exit.

## Limitations

- Intended for small collections of text documents. PDF and Word files are not supported.
- Paragraph size is not limited by token count, so long paragraphs should be shortened before ingestion.
- Search compares every stored vector and is not intended for large datasets.
- The similarity threshold can reject a relevant question and does not guarantee that every generated statement is supported by the documents.
- Source filenames identify retrieved documents; they are not citations for individual sentences.
- Inference runs locally. Initial downloads and catalog access may require a connection; operation with networking completely disabled has not been verified.

## Learning Objectives

This project covers document preparation, embeddings, SQLite storage, semantic search, prompt construction, and local model integration in a complete RAG workflow.


