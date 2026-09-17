"""
rag.py
------
Core Retrieval-Augmented Generation logic:
1. Embed the user's question.
2. Retrieve the most relevant chunks from ChromaDB.
3. Build a grounded prompt and call Gemini (free tier) to generate the answer.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

import config

# Load these once at import time (not per-request) so the app stays fast.
_embedder = SentenceTransformer(config.EMBEDDING_MODEL)
_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
_collection = _client.get_or_create_collection(config.COLLECTION_NAME)

_genai_client = genai.Client(api_key=config.GEMINI_API_KEY)


def retrieve_chunks(query: str, top_k: int = config.TOP_K):
    """Return the top_k most relevant chunks (with sources) for a query."""
    query_embedding = _embedder.encode([query]).tolist()
    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(documents, metadatas):
        chunks.append({"text": doc, "source": meta.get("source", "unknown")})
    return chunks


SYSTEM_PROMPT = """You are a helpful college assistant chatbot. Answer the
student's question using ONLY the context provided below. Be concise and
friendly.

If the answer is not contained in the context, say clearly that you don't
have that information and suggest the student contact the college office,
rather than guessing or making something up."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"""Context:
{context_text}

Student's question: {question}

Answer the question using only the context above."""


def make_excerpt(text: str, max_chars: int = 160) -> str:
    """Shorten a chunk of text into a clean, readable excerpt."""
    text = " ".join(text.split())  # collapse whitespace/newlines
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def generate_answer(question: str, history: list[dict] | None = None) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks, then ask Gemini to answer
    using only that context. Returns the answer plus the source excerpts used.
    """
    chunks = retrieve_chunks(question)

    if not chunks:
        return {
            "answer": "I don't have any information loaded yet. Please make sure "
                      "the knowledge base has been built (run ingest.py).",
            "sources": [],
        }

    prompt = build_prompt(question, chunks)

    response = _genai_client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=config.MAX_TOKENS,
        ),
    )

    answer_text = response.text

    # One excerpt per unique source file (first chunk seen for that file)
    seen_sources = set()
    sources = []
    for c in chunks:
        if c["source"] not in seen_sources:
            seen_sources.add(c["source"])
            sources.append({
                "source": c["source"],
                "excerpt": make_excerpt(c["text"]),
            })

    return {"answer": answer_text, "sources": sources}


if __name__ == "__main__":
    # Quick manual test from the command line
    print("College Chatbot (RAG) — type 'quit' to exit\n")
    while True:
        q = input("You: ")
        if q.lower() in ("quit", "exit"):
            break
        result = generate_answer(q)
        print(f"\nBot: {result['answer']}")
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                print(f"  📄 {s['source']}: \"{s['excerpt']}\"")
        print()
