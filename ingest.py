"""Ingestion + Embedding + Vector store — Stages 1, 3 of the pipeline.

Reads every document in ``documents/``, chunks each one (chunk.py), embeds the
chunks with all-MiniLM-L6-v2, and writes them to a persistent ChromaDB collection.
Run this once before querying:

    python ingest.py
"""

import os
import glob

import chromadb
from chromadb.utils import embedding_functions

from chunk import chunk_text

DOCUMENTS_DIR = "documents"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "unofficial_guide"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_documents(directory: str = DOCUMENTS_DIR) -> dict[str, str]:
    """Return {filename: text} for every .md/.txt/.pdf in the documents directory."""
    docs: dict[str, str] = {}

    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        name = os.path.basename(path)
        if name == ".gitkeep":
            continue
        ext = os.path.splitext(name)[1].lower()

        if ext in (".md", ".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs[name] = f.read()
        elif ext == ".pdf":
            docs[name] = _read_pdf(path)
        # Silently skip anything else (images, etc.).

    return docs


def _read_pdf(path: str) -> str:
    """Extract text from a PDF using pdfplumber (optional dependency)."""
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"{path} is a PDF but pdfplumber is not installed. "
            "Add pdfplumber to requirements.txt and reinstall."
        ) from exc
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n\n".join(text_parts)


def get_embedding_function():
    """Local sentence-transformers embedding function for Chroma."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )


def build_index() -> int:
    """Load, chunk, embed, and store all documents. Returns the total chunk count."""
    docs = load_documents()
    if not docs:
        raise SystemExit(
            f"No documents found in '{DOCUMENTS_DIR}/'. Add .md/.txt/.pdf files first."
        )

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Rebuild from scratch so re-running ingest reflects the current documents.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for source, text in docs.items():
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{source}::chunk{i}")
            texts.append(chunk)
            metadatas.append({"source": source, "chunk_index": i})
        print(f"  {source}: {len(chunks)} chunks")

    # Add in batches (embeddings computed by the embedding function).
    collection.add(ids=ids, documents=texts, metadatas=metadatas)

    print(f"\nIndexed {len(texts)} chunks from {len(docs)} documents into '{CHROMA_DIR}'.")
    return len(texts)


if __name__ == "__main__":
    print(f"Loading documents from '{DOCUMENTS_DIR}/' and building the index...\n")
    build_index()
