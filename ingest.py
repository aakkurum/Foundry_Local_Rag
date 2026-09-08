"""Create the local SQLite knowledge base from text files in documents/."""

import json
import sqlite3
from pathlib import Path

from foundry_local_sdk import (
    Configuration,
    EmbeddingsSession,
    FoundryLocalManager,
    Request,
    TensorItem,
    TextItem,
)

APP_NAME = "local_rag_assistant"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
DOCUMENTS_DIR = Path(__file__).parent / "documents"
DATABASE_PATH = Path(__file__).parent / "data" / "knowledge.db"
PARAGRAPHS_PER_CHUNK = 2


def get_available_model(manager, alias: str):
    """Keep the model alias, but avoid variants whose provider is unavailable."""
    model = manager.catalog.get_model(alias)
    if model is None:
        raise RuntimeError(f"Model not found in the Foundry Local catalog: {alias}")
    providers = {"CPUExecutionProvider"}
    providers.update(ep.name for ep in manager.discover_eps() if ep.is_registered)
    provider = model.info.runtime.execution_provider
    if provider not in providers:
        variants = [v for v in model.variants if v.info.runtime.execution_provider in providers]
        if not variants:
            raise RuntimeError(f"No available execution provider for model: {alias}")
        selected = next((v for v in variants if v.is_cached), variants[0])
        print(f"{provider} unavailable; using {selected.id}.")
        model.select_variant(selected)
    return model


def chunk_text(text: str, paragraphs_per_chunk: int = PARAGRAPHS_PER_CHUNK) -> list[str]:
    """Split text into small chunks containing one or more paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [
        "\n\n".join(paragraphs[i : i + paragraphs_per_chunk])
        for i in range(0, len(paragraphs), paragraphs_per_chunk)
    ]


def load_document_chunks() -> list[tuple[str, str]]:
    """Return (source, content) pairs for all .txt documents."""
    if not DOCUMENTS_DIR.exists():
        raise RuntimeError(f"Documents directory not found: {DOCUMENTS_DIR}")

    files = sorted(DOCUMENTS_DIR.glob("*.txt"))
    if not files:
        raise RuntimeError(f"No .txt documents found in {DOCUMENTS_DIR}")

    chunks: list[tuple[str, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        chunks.extend((path.name, chunk) for chunk in chunk_text(text))

    if not chunks:
        raise RuntimeError("The documents folder does not contain any usable text.")

    return chunks


def create_embeddings(session: EmbeddingsSession, texts: list[str]) -> list[list[float]]:
    """Generate one embedding for each input text."""
    with Request() as request:
        for text in texts:
            request.add_item(TextItem(text))

        with session.process_request(request) as response:
            # SDK 2.x exposes float32 tensors as raw bytes, not scalar values.
            embeddings = [list(memoryview(item.data).cast("f")) for item in response if isinstance(item, TensorItem)]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(texts)}, received {len(embeddings)}."
        )

    return embeddings


def save_to_sqlite(chunks: list[tuple[str, str]], embeddings: list[list[float]]) -> None:
    """Replace the SQLite knowledge base with the newly generated chunks."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )
        connection.execute("DELETE FROM chunks")
        connection.executemany(
            "INSERT INTO chunks (source, content, embedding) VALUES (?, ?, ?)",
            [
                (source, content, json.dumps(embedding))
                for (source, content), embedding in zip(chunks, embeddings)
            ],
        )


def progress(label: str):
    """Create a small model-download progress callback."""
    return lambda percent: print(
        f"\r{label}: {percent:5.1f}%", end="", flush=True
    )


def main() -> None:
    chunks = load_document_chunks()
    print(f"Found {len(chunks)} document chunks.")

    FoundryLocalManager.initialize(Configuration(app_name=APP_NAME))
    manager = FoundryLocalManager.instance
    model = get_available_model(manager, EMBEDDING_MODEL)

    try:
        print(f"Using embedding model: {EMBEDDING_MODEL}")
        model.download(progress("Downloading embedding model"))
        print()
        model.load()

        with EmbeddingsSession(model) as session:
            embeddings = create_embeddings(session, [content for _, content in chunks])

        save_to_sqlite(chunks, embeddings)
        print(f"Saved {len(chunks)} chunks to {DATABASE_PATH}")
    finally:
        if model.is_loaded:
            model.unload()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
