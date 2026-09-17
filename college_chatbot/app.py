"""
app.py
------
FastAPI backend that exposes the RAG chatbot as a REST API.

Run with:
    uvicorn app:app --reload
Then open http://127.0.0.1:8000/docs for interactive API docs.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import generate_answer

app = FastAPI(title="College Chatbot API")

# Allow the Streamlit / React frontend to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
def health_check():
    return {"status": "ok", "message": "College Chatbot API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = generate_answer(request.question)
    return ChatResponse(answer=result["answer"], sources=result["sources"])
