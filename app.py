from pathlib import Path

import streamlit as st

from rag_engine import ask_rag, load_base


st.set_page_config(page_title="SIET Chatbot", layout="centered")
st.title("Shridevi Institute of Engineering and Technology — Chatbot")
st.caption("Ask anything about the college.")
st.caption("Created by AMEER KHAN")

PDF_PATH = Path(__file__).resolve().parent / "01_College_Information.pdf"


@st.cache_resource
def initialize_rag(pdf_path: str):
    return load_base(pdf_path)


if not PDF_PATH.is_file():
    st.error(
        "The college information PDF is missing. Add "
        f"`{PDF_PATH.name}` to the same GitHub folder as `app.py`, "
        "then commit and push it so Streamlit Cloud can access it."
    )
    st.stop()

try:
    index, embed_model, chunks = initialize_rag(str(PDF_PATH))
except Exception as exc:
    st.error(f"Could not load the college information PDF: {exc}")
    st.stop()

if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

for message in st.session_state.chat_hist:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask anything about SIET...")

if question:
    st.session_state.chat_hist.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding an answer..."):
            try:
                api_key = st.secrets["GROQ_API_KEY"]
                answer = ask_rag(
                    question,
                    embed_model,
                    chunks,
                    index,
                    api_key=api_key,
                )
            except KeyError:
                answer = "GROQ_API_KEY is missing from the app's Streamlit secrets."
            except Exception as exc:
                answer = f"I couldn't get an answer: {exc}"
            st.write(answer)
    st.session_state.chat_hist.append(
        {"role": "assistant", "content": answer}
    )

if st.button("Clear chat"):
    st.session_state.chat_hist = []
    st.rerun()
