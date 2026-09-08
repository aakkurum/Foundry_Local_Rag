"""Simple local RAG question-answering application using Foundry Local and SQLite."""

import json
import math
import sqlite3
import time
from pathlib import Path

from ingest import get_available_model

from foundry_local_sdk import (
    ChatSession,
    Configuration,
    EmbeddingsSession,
    FoundryLocalManager,
    MessageItem,
    Request,
    RequestOptions,
    SearchOptions,
    TensorItem,
    TextItem,
)

APP_NAME = "local_rag_assistant"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "qwen2.5-0.5b"
DATABASE_PATH = Path(__file__).parent / "data" / "knowledge.db"
TOP_K = 3
MIN_RELEVANCE_SCORE = 0.60
FALLBACK_ANSWER = "The provided documents do not contain enough information to answer this question."

SYSTEM_PROMPT = """You are a local document question-answering assistant.
Answer the user's question using only the provided context.
Do not use outside knowledge and do not invent facts.
If the context does not contain enough information, say: "The provided documents do not contain enough information to answer this question."
Keep the answer concise and clear.
"""


def load_knowledge_base() -> list[dict]:
    """Load all stored chunks and embeddings from SQLite."""
    if not DATABASE_PATH.exists():
        raise RuntimeError(
            "Knowledge database not found. Run 'python ingest.py' before starting the app."
        )

    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute(
            "SELECT id, source, content, embedding FROM chunks ORDER BY id"
        ).fetchall()

    if not rows:
        raise RuntimeError(
            "Knowledge database is empty. Add documents and run 'python ingest.py' again."
        )

    return [
        {
            "id": row[0],
            "source": row[1],
            "content": row[2],
            "embedding": json.loads(row[3]),
        }
        for row in rows
    ]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def embed_query(session: EmbeddingsSession, query: str) -> list[float]:
    """Generate the embedding for one user query."""
    with Request().add_item(TextItem(query)) as request:
        with session.process_request(request) as response:
            for item in response:
                if isinstance(item, TensorItem):
                    # SDK 2.x exposes the float32 tensor as raw bytes.
                    return list(memoryview(item.data).cast("f"))

    raise RuntimeError("The embedding model did not return a query embedding.")


def retrieve(query_embedding: list[float], chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Return the top-k chunks ranked by cosine similarity."""
    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append({**chunk, "score": score})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[: min(top_k, len(scored))]


def build_context(results: list[dict]) -> str:
    """Format retrieved chunks as grounded context for the chat model."""
    if not results or max(item.get("score", 1.0) for item in results) < MIN_RELEVANCE_SCORE:
        return ""
    return "\n\n".join(
        f"[Source: {item['source']}]\n{item['content']}" for item in results
    )


def generate_answer(chat_model, question: str, context: str) -> str:
    """Generate a grounded answer with a fresh chat session for each question."""
    if not context.strip():
        return FALLBACK_ANSWER
    system_message = f"{SYSTEM_PROMPT}\nContext:\n{context}"

    with ChatSession(chat_model) as session:
        session.set_options(
            RequestOptions(
                search=SearchOptions(
                    temperature=0.0,
                    max_output_tokens=256,
                )
            )
        )

        with Request() as request:
            request.add_item(MessageItem.system(system_message))
            request.add_item(MessageItem.user(question))

            with session.process_request(request) as response:
                parts = []
                for item in response:
                    if isinstance(item, TextItem):
                        parts.append(item.text)
                    elif isinstance(item, MessageItem):
                        parts.extend(part.text for part in item.parts if isinstance(part, TextItem))

    answer = "".join(parts).strip()
    if not answer:
        raise RuntimeError("The chat model returned an empty answer.")
    return answer


def print_sources(results: list[dict]) -> None:
    """Print unique source names from the actual retrieval results."""
    sources = list(dict.fromkeys(item["source"] for item in results))
    print("\nSources:")
    for source in sources:
        print(f"- {source}")


def progress(label: str):
    """Create a small model-download progress callback."""
    return lambda percent: print(
        f"\r{label}: {percent:5.1f}%", end="", flush=True
    )


def main() -> None:
    chunks = load_knowledge_base()
    print(f"Loaded {len(chunks)} chunks from SQLite.")

    FoundryLocalManager.initialize(Configuration(app_name=APP_NAME))
    manager = FoundryLocalManager.instance
    embedding_model = get_available_model(manager, EMBEDDING_MODEL)
    chat_model = get_available_model(manager, CHAT_MODEL)

    try:
        print(f"Using embedding model: {EMBEDDING_MODEL}")
        embedding_model.download(progress("Downloading embedding model"))
        print()
        embedding_model.load()

        print(f"Using chat model: {CHAT_MODEL}")
        chat_model.download(progress("Downloading chat model"))
        print()
        chat_model.load()

        print("\nLocal RAG Assistant is ready.")
        print('Type "quit" to exit.\n')

        with EmbeddingsSession(embedding_model) as embedding_session:
            while True:
                question = input("Question: ").strip()

                if question.lower() == "quit":
                    break
                if not question:
                    print("Please enter a question.\n")
                    continue

                started = time.perf_counter()
                query_embedding = embed_query(embedding_session, question)
                results = retrieve(query_embedding, chunks)
                retrieval_seconds = time.perf_counter() - started

                answer = generate_answer(chat_model, question, build_context(results))
                total_seconds = time.perf_counter() - started

                print(f"\nAnswer:\n{answer}")
                print_sources(results)
                print(
                    f"\nTime: retrieval {retrieval_seconds:.2f}s | total {total_seconds:.2f}s\n"
                )
    finally:
        if chat_model.is_loaded:
            chat_model.unload()
        if embedding_model.is_loaded:
            embedding_model.unload()
        print("Models unloaded. Goodbye.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
