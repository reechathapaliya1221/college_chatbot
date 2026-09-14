"""
Central configuration for the College Chatbot (RAG) project.

Your ANTHROPIC_API_KEY is loaded from a local .env file (see .env.example).
Never type your real key into a terminal command or commit it to a file
that gets shared or screenshotted — that exposes it to anyone who sees it.

Setup:
    1. Copy .env.example to a new file named .env
    2. Open .env and paste your real key after ANTHROPIC_API_KEY=
    3. Save. The .env file is already excluded from version control via .gitignore.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file, if present

# --- Folders ---
DATA_DIR = "data"                  # put your college PDFs / txt files here
CHROMA_PATH = "chroma_db"          # local vector database storage folder
COLLECTION_NAME = "college_kb"

# --- Embedding model (runs locally, free, no API key needed) ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Chunking ---
CHUNK_SIZE = 400        # words per chunk
CHUNK_OVERLAP = 60      # words of overlap between chunks

# --- Retrieval ---
TOP_K = 4               # number of chunks to retrieve per query

# --- LLM (Google Gemini - free tier) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.6-flash"   # current stable Flash model
MAX_TOKENS = 1024