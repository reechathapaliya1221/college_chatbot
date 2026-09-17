# College Chatbot (RAG-powered)

A Retrieval-Augmented Generation chatbot that answers student questions
using your college's own documents (handbook, fee structure, timetable,
FAQs, etc.) grounded through Google Gemini (free tier).

## How it works
1. **ingest.py** — loads your PDFs/text files, splits them into chunks,
   embeds them locally (sentence-transformers), and stores them in a
   ChromaDB vector database.
2. **rag.py** — given a question, retrieves the most relevant chunks and
   asks Claude to answer using only that context (prevents hallucination).
3. **app.py** — FastAPI backend exposing a `/chat` endpoint.
4. **streamlit_app.py** — a ready-to-use chat UI.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key (free — get one at https://aistudio.google.com)
# Copy .env.example to .env, then open .env and paste your real key in.
# Never type your key directly into the terminal or a screenshot-able file.
cp .env.example .env      # Windows: copy .env.example .env

# 4. Add your college documents
#    Put PDFs or .txt files into the data/ folder
mkdir -p data
cp /path/to/handbook.pdf data/
cp /path/to/fee_structure.pdf data/

# 5. Build the knowledge base (run this once, and again whenever docs change)
python ingest.py
```

## Ideas to extend this for your project report
- Add multi-turn conversation memory (pass `history` into `generate_answer`)
- Show citation snippets, not just filenames
- Add a feedback (👍/👎) button per answer
- Build an admin page to upload new documents without re-running ingest.py manually
- Deploy the FastAPI backend + connect a Telegram/WhatsApp bot
- Add authentication so it only works for verified students
- Evaluate answer quality with a small test set of Q&A pairs

