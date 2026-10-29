"""Ask questions about your profile. Retrieval is local; answers come from a Groq model."""
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()  # reads GROQ_API_KEY / GROQ_MODEL from .env

MODEL = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TOP_K = 3

st.set_page_config(page_title="Syed Daud Ali", page_icon="💬")
st.title("Profile Chatbot")


def get_key():
    """.env first (local), then Streamlit secrets (deployed)."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


@st.cache_resource
def load():
    vectors = np.load(Path("index/vectors.npy"))
    chunks = json.loads(Path("index/chunks.json").read_text(encoding="utf-8"))
    return TextEmbedding(model_name=MODEL), vectors, chunks


def retrieve(question, model, vectors, chunks):
    q = np.array(list(model.embed([question])), dtype="float32")[0]
    q /= np.linalg.norm(q)
    scores = vectors @ q
    top = np.argsort(scores)[::-1][:TOP_K]
    return [(chunks[i], float(scores[i])) for i in top]


def answer(question, context):
    key = get_key()
    if not key:
        return None
    from groq import Groq

    client = Groq(api_key=key)
    reply = client.chat.completions.create(
        model=GROQ_MODEL,
        max_completion_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the context given. If the answer is not in "
                    "the context, say you don't know. Be concise."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return reply.choices[0].message.content


model, vectors, chunks = load()
question = st.text_input("Your question", placeholder="Questions about my profile!")

if question:
    hits = retrieve(question, model, vectors, chunks)
    context = "\n\n".join(c for c, _ in hits)
    with st.spinner("Thinking..."):
        text = answer(question, context)
    if text:
        st.write(text)
    else:
        st.info("No GROQ_API_KEY set, so here are the best matching passages instead.")
    with st.expander("Sources used"):
        for c, s in hits:
            st.caption(f"similarity {s:.2f}")
            st.text(c)
