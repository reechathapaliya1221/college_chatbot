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

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".txt")


def load_text_from_file(filepath: Path) -> str:
    """Extract raw text from a .pdf or .txt file."""
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(filepath))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".txt":
        return filepath.read_text(encoding="utf-8")
    return ""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)  # guard against overlap >= chunk_size
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += step
    return chunks


def find_source_files(data_dir: Path) -> list[Path]:
    """Return all supported files in data_dir, sorted for reproducible runs."""
    files = [
        f for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def build_knowledge_base(data_dir: Path, chroma_path: Path) -> None:
    logger.info("Loading embedding model...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(chroma_path))
    # Start fresh each time ingest.py is run
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet — nothing to clean up
    collection = client.create_collection(config.COLLECTION_NAME)

    source_files = find_source_files(data_dir)
    if not source_files:
        logger.warning(
            "No files found in '%s/'. Add PDFs or .txt files and re-run.", data_dir
        )
        return

    total_chunks = 0
    processed_files = 0
    for filepath in source_files:
        logger.info("Processing %s...", filepath.name)
        try:
            text = load_text_from_file(filepath)
        except Exception as exc:
            logger.error("  Skipped %s — could not read file (%s)", filepath.name, exc)
            continue

        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if not chunks:
            logger.warning("  Skipped %s — no extractable text.", filepath.name)
            continue

        embeddings = embedder.encode(chunks).tolist()
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"source": filepath.name, "chunk_index": i} for i in range(len(chunks))
        ]

        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        total_chunks += len(chunks)
        processed_files += 1
        logger.info("  Added %d chunks.", len(chunks))

    logger.info(
        "\nDone. Knowledge base built with %d chunks from %d file(s).",
        total_chunks, processed_files,
    )
    logger.info("Stored at: %s/", chroma_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the RAG knowledge base.")
    parser.add_argument(
        "--data-dir", default=config.DATA_DIR,
        help=f"Folder containing source documents (default: {config.DATA_DIR})",
    )
    parser.add_argument(
        "--chroma-path", default=config.CHROMA_PATH,
        help=f"Folder to store the vector database (default: {config.CHROMA_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    build_knowledge_base(data_dir, Path(args.chroma_path))


if __name__ == "__main__":
    main()
