"""RAG module: retrieval + strict grounded generation using Google Gemini."""

import time
from typing import Any
import chromadb
from google import genai
from google.genai import types
from google.genai.errors import APIError
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TOP_K,
    DISTANCE_THRESHOLD,
)
from src.ingest import ingest_documents

# Module-level singletons for performance across queries
_embedder: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None
_chroma_collection = None
_gemini_client: genai.Client | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def get_collection():
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        existing = [c.name for c in _chroma_client.list_collections()]
        if COLLECTION_NAME not in existing or _chroma_client.get_collection(COLLECTION_NAME).count() == 0:
            print(f"Collection '{COLLECTION_NAME}' not found or empty. Running initial ingestion...")
            _chroma_collection = ingest_documents(rebuild=False)
        else:
            _chroma_collection = _chroma_client.get_collection(COLLECTION_NAME)
    return _chroma_collection


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def retrieve_context(
    query: str, top_k: int = TOP_K, distance_threshold: float = DISTANCE_THRESHOLD
) -> list[dict]:
    """Embeds the query and retrieves top-k chunks from ChromaDB with a distance cutoff."""
    embedder = get_embedder()
    collection = get_collection()

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    if not results or not results["documents"] or not results["documents"][0]:
        return retrieved

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc_text, meta, dist in zip(docs, metas, distances):
        # Only retain chunks within the distance threshold
        if dist <= distance_threshold:
            retrieved.append(
                {
                    "chunk_id": meta["chunk_id"],
                    "page": meta["page"],
                    "text": doc_text,
                    "distance": dist,
                }
            )

    return retrieved


def call_gemini_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Calls Gemini API with exponential backoff on transient errors."""
    client = get_gemini_client()

    models_to_try = [GEMINI_MODEL]
    # If the user specified gemini-2.0-flash, automatically fallback to gemini-3.8-flash if retired
    if GEMINI_MODEL not in ["gemini-3.8-flash", "gemini-2.5-flash"]:
        models_to_try.append("gemini-3.8-flash")

    last_error = None
    for model_name in models_to_try:
        backoff = 1.0
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                    ),
                )
                if response and response.text:
                    return response.text.strip()
                return "[not found]"
            except APIError as e:
                last_error = e
                # If model is deprecated or not found (404), try alternative model immediately
                if "404" in str(e) or "NOT_FOUND" in str(e):
                    break
                # Transient error: backoff and retry
                time.sleep(backoff)
                backoff *= 2
            except Exception as e:
                last_error = e
                time.sleep(backoff)
                backoff *= 2

    raise RuntimeError(f"Gemini API generation failed after retries: {last_error}")


def answer(question: str) -> dict:
    """Returns {"answer": <str>, "sources": [<page_or_chunk_id>, ...]}

    Behavior:
    1. Embed query & retrieve top-k chunks from ChromaDB with distance cutoff.
    2. If no chunks pass threshold, return {"answer": "[not found]", "sources": []}.
    3. Send retrieved chunks with strict grounding system prompt to Gemini (temp 0).
    4. If context lacks answer, model returns exactly '[not found]'.
    5. Return cleaned answer and list of source page numbers / chunk IDs.
    """
    retrieved_chunks = retrieve_context(question, top_k=TOP_K, distance_threshold=DISTANCE_THRESHOLD)

    if not retrieved_chunks:
        return {"answer": "[not found]", "sources": []}

    # Format context with chunk_id and page number
    context_blocks = []
    for c in retrieved_chunks:
        context_blocks.append(
            f"--- START CHUNK [{c['chunk_id']} | Page {c['page']}] ---\n"
            f"{c['text']}\n"
            f"--- END CHUNK [{c['chunk_id']}] ---"
        )
    formatted_context = "\n\n".join(context_blocks)

    strict_prompt = f"""You are a strict, factual University Academic & Hostel Policy Advisor for Mirai School of Technology.
Your task is to answer student questions based EXCLUSIVELY on the provided Handbook context chunks below.

STRICT GROUNDING RULES:
1. Answer ONLY from the provided context. Do NOT use external knowledge, assumptions, guessing, or general university policy conventions.
2. If the context does not contain the answer, respond with exactly "[not found]" and nothing else (no punctuation, no explanation, no apologies).
3. If a question has multiple parts (e.g., marks AND eligibility), answer every part that is supported by the context. If any required part is missing from the context, answer only the parts present and do NOT invent or guess the missing part.
4. Quote all numbers, percentages, days, fees, thresholds, and authority names exactly as written in the context.
5. Be concise, direct, and factual. Do not add filler or conversational preamble.
6. At the end of your response, on a new line, list the chunk IDs that you directly used to answer the question in the format:
SOURCES: [chunk_id_1, chunk_id_2, ...]
If your answer is [not found], do not include any SOURCES line.

CONTEXT:
{formatted_context}

QUESTION:
{question}

ANSWER:"""

    raw_output = call_gemini_with_retry(strict_prompt)

    # Process response
    if raw_output == "[not found]" or raw_output.lower().startswith("[not found]"):
        return {"answer": "[not found]", "sources": []}

    # Parse SOURCES if generated by the model, otherwise use the retrieved chunk pages/ids
    answer_text = raw_output
    used_sources = []

    if "SOURCES:" in raw_output:
        parts = raw_output.rsplit("SOURCES:", 1)
        answer_text = parts[0].strip()
        sources_str = parts[1].strip()
        # Extract chunk ids like p7_c1 or pages
        import re
        extracted = re.findall(r"p\d+_c\d+", sources_str)
        if extracted:
            used_sources = list(dict.fromkeys(extracted))
        else:
            pages = re.findall(r"\d+", sources_str)
            used_sources = [int(p) for p in dict.fromkeys(pages)]

    # Fallback for sources: if answer was produced but sources not explicitly parsed,
    # map to unique pages of the retrieved chunks
    if not used_sources and answer_text != "[not found]":
        used_sources = sorted(list(set(c["page"] for c in retrieved_chunks)))

    # Final sanity check: if the answer contains [not found]
    if answer_text == "[not found]":
        return {"answer": "[not found]", "sources": []}

    return {"answer": answer_text, "sources": used_sources}
