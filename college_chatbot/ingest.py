"""
ingest.py
---------
Reads all documents from the DATA_DIR folder (.pdf and .txt files),
splits them into overlapping chunks, embeds each chunk locally using
sentence-transformers, and stores everything in a persistent ChromaDB
collection so it can be searched later at query time.

Run this once (and again any time your source documents change):
    python ingest.py
"""

import os
import glob
import uuid

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config


def load_text_from_file(filepath: str) -> str:
    """Extract raw text from a .pdf or .txt file."""
    if filepath.lower().endswith(".pdf"):
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif filepath.lower().endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return ""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_knowledge_base():
    print("Loading embedding model...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    # Start fresh each time ingest.py is run
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(config.COLLECTION_NAME)

    filepaths = glob.glob(os.path.join(config.DATA_DIR, "*.pdf")) + \
        glob.glob(os.path.join(config.DATA_DIR, "*.txt"))

    if not filepaths:
        print(f"No files found in '{config.DATA_DIR}/'. Add PDFs or .txt files and re-run.")
        return

    total_chunks = 0
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")
        text = load_text_from_file(filepath)
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

        if not chunks:
            print(f"  Warning: no extractable text in {filename}")
            continue

        embeddings = embedder.encode(chunks).tolist()
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        print(f"  Added {len(chunks)} chunks.")

    print(f"\nDone. Knowledge base built with {total_chunks} chunks from {len(filepaths)} file(s).")
    print(f"Stored at: {config.CHROMA_PATH}/")


if __name__ == "__main__":
    os.makedirs(config.DATA_DIR, exist_ok=True)
    build_knowledge_base()
