"""
streamlit_app.py
----------------
Simple chat UI for the college chatbot.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
from rag import generate_answer

st.set_page_config(page_title="College Chatbot", page_icon="🎓")
st.title("🎓 College Assistant Chatbot")
st.caption("Ask me about admissions, fees, timetables, faculty, and more.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    """Show an expandable list of the document excerpts used for an answer."""
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['source']}**")
            st.caption(f"\"{s['excerpt']}\"")


# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

# Chat input
if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = generate_answer(question)
        st.markdown(result["answer"])
        render_sources(result["sources"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
