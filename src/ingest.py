"""PDF Ingestion pipeline: loads handbook PDF, performs page-aware chunking,
computes embeddings, and persists to ChromaDB.
"""

import sys
import argparse
from pathlib import Path
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Support both pymupdf and pypdf
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        import pypdf

from src.config import (
    PDF_PATH,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extracts text page by page from the PDF, keeping page numbers (1-indexed)."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    pages = []
    # Try pymupdf / fitz first
    try:
        doc = fitz.open(str(pdf_path))
        for page_idx in range(len(doc)):
            text = doc[page_idx].get_text("text")
            pages.append({"page": page_idx + 1, "text": text})
        doc.close()
        return pages
    except Exception:
        pass

    # Fallback to pypdf
    import pypdf

    reader = pypdf.PdfReader(str(pdf_path))
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": page_idx + 1, "text": text})
    return pages


def chunk_pages(
    pages: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """Page-aware chunking: splits text strictly within each page, assigning

    metadata with page (1-indexed) and chunk_id (format p{page}_c{index}).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page_info in pages:
        page_num = page_info["page"]
        page_text = page_info["text"]
        splits = splitter.split_text(page_text)

        for idx, split_text in enumerate(splits, start=1):
            chunk_id = f"p{page_num}_c{idx}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page_num,
                    "text": split_text,
                }
            )

    return chunks


def ingest_documents(rebuild: bool = False):
    """Embeds and loads the handbook into ChromaDB if not already present."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Check if collection already exists
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections and not rebuild:
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        if count > 0:
            print(f"ChromaDB collection '{COLLECTION_NAME}' already contains {count} chunks. Skipping re-embedding.")
            print("Use --rebuild to force re-ingestion.")
            return collection

    if COLLECTION_NAME in existing_collections and rebuild:
        print(f"Deleting existing collection '{COLLECTION_NAME}' for --rebuild...")
        client.delete_collection(name=COLLECTION_NAME)

    print(f"Loading PDF from: {PDF_PATH}")
    pages = load_pdf_pages(PDF_PATH)
    total_pages = len(pages)

    print(f"Chunking {total_pages} pages (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})...")
    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    total_chunks = len(chunks)

    print(f"Generated {total_chunks} chunks across {total_pages} pages.")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"page": c["page"], "chunk_id": c["chunk_id"]} for c in chunks]

    print("Computing embeddings...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    print(f"Storing chunks in ChromaDB (cosine distance)...")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Add in batches
    batch_size = 64
    for i in range(0, total_chunks, batch_size):
        end = min(i + batch_size, total_chunks)
        collection.add(
            ids=ids[i:end],
            documents=texts[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )

    print("=" * 60)
    print("Ingestion Summary:")
    print(f"  - Total Pages Ingested: {total_pages}")
    print(f"  - Total Chunks Created: {total_chunks}")
    print(f"  - Chunk Size: {CHUNK_SIZE} characters")
    print(f"  - Chunk Overlap: {CHUNK_OVERLAP} characters")
    print(f"  - Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"  - Vector Store: ChromaDB at {CHROMA_DIR} (cosine distance)")
    print("=" * 60)

    return collection


def main():
    parser = argparse.ArgumentParser(description="Ingest handbook PDF into ChromaDB.")
    parser.add_argument("--rebuild", action="store_true", help="Force re-ingestion of the PDF.")
    args = parser.parse_args()

    ingest_documents(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
