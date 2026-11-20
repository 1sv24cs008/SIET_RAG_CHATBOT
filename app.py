import streamlit as st
from rag_engine import load_base, ask_rag

st.set_page_config(page_title="SIET Chatbot", layout="centered")
st.title("Shridevi Institute of Engineering and Technology — Chatbot")
st.caption("Ask anything about the college.")
st.caption("Created by AMEER KHAN")

my_key = st.secrets["GROQ_API_KEY"]


@st.cache_resource
def initialize_rag():
    return load_base("01_College_Information")


index, embed_model, chunks = initialize_rag()

if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

for message in st.session_state.chat_hist:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask anything about SIET...")

if question:
    st.session_state.chat_hist.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Getting an answer..."):
            answer = ask_rag(
                question,
                embed_model,
                chunks,
                index,
                api_key=my_key,
            )
            st.write(answer)

    st.session_state.chat_hist.append(
        {"role": "assistant", "content": answer}
    )

if st.button("Clear chat"):
    st.session_state.chat_hist = []
    st.rerun()
