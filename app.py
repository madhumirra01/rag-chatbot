"""Chat with your documents.  Start it with:  streamlit run app.py"""
import streamlit as st

from src.config import SIM_THRESHOLD, UPLOAD_DIR
from src.indexing import index_file
from src.pipeline import ask
from src.vectorstore import (clear_vectorstore, count, get_embeddings,
                             list_sources, load_vectorstore)

st.set_page_config(page_title="Chat with your documents", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading the embedding model (the first time takes a minute)...")
def get_embedder():
    return get_embeddings()


@st.cache_resource
def get_store():
    return load_vectorstore(get_embedder())


def show_sources(citations):
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            st.markdown(f"**[{c['number']}]** {c['source']} · {c['location']} · similarity {c['score']}")
            st.caption(c["snippet"])


st.session_state.setdefault("messages", [])
store = get_store()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF, TXT or CSV files", type=["pdf", "txt", "csv"],
                               accept_multiple_files=True)
    if st.button("Add to knowledge base", type="primary", disabled=not uploads):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        bar = st.progress(0.0, text="Starting...")
        for i, upload in enumerate(uploads):
            path = UPLOAD_DIR / upload.name
            path.write_bytes(upload.getbuffer())

            def update_bar(fraction, i=i, name=upload.name):
                bar.progress((i + fraction) / len(uploads), text=f"Indexing {name}...")

            info = index_file(store, path, update_bar)
            st.success(f"{info['file']}: {info['chunks']} chunks")
            if info["note"]:
                st.warning(info["note"])
        bar.empty()

    st.subheader("Indexed files")
    sources = list_sources(store)
    if sources:
        for name, n in sources.items():
            st.write(f"- {name} ({n} chunks)")
    else:
        st.caption("Nothing yet. Upload a file above.")

    st.subheader("Settings")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, float(SIM_THRESHOLD), 0.01,
                          help="If the best matching passage scores below this, the bot says it "
                               "could not find the answer instead of guessing.")

    left, right = st.columns(2)
    if left.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if right.button("Delete docs"):
        clear_vectorstore(store)
        get_store.clear()
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------ main chat
st.title("📄 Chat with your documents")
st.caption("Answers come only from the files you uploaded, with sources you can check.")

has_docs = count(store) > 0
if not has_docs:
    st.info("Upload a PDF, TXT or CSV in the sidebar and click **Add to knowledge base** to begin.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("citations"))

question = st.chat_input("Ask a question about your documents", disabled=not has_docs)
if question:
    history = list(st.session_state.messages)         # earlier turns, before this question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            result = ask(store, question, threshold, history)
        st.markdown(result["answer"])
        if result["rewritten"] != question:
            st.caption(f"Searched for: *{result['rewritten']}*")
        if result["refused"]:
            st.caption(f"Best match similarity was {result['top_score']:.2f} "
                       f"(threshold {threshold:.2f}).")
        show_sources(result["citations"])
    st.session_state.messages.append({"role": "assistant", "content": result["answer"],
                                      "citations": result["citations"]})

