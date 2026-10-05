import time

import streamlit as st

from app.clients import MAX_FILES, MAX_PAGES
from app.rag_workflow import graph
from app.vector_store import build_vector_store, get_retriever, validate_pdfs

def stream_text(text: str):
    # stream the finished response word by word (UI only, the graph is untouched)
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)


st.set_page_config(
    page_title="PageWise | Ask your PDFs anything",
    page_icon="📚",
    layout="wide",
)

# ── Bootstrap ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = None


# ── Sidebar: upload PDFs ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Your documents")
    files = st.file_uploader(
        f"Upload up to {MAX_FILES} PDFs ({MAX_PAGES} pages each)",
        type="pdf",
        accept_multiple_files=True,
    )

    if files:
        error = validate_pdfs(files)
        if error:
            st.error(error)
            st.session_state.retriever = None
            st.session_state.indexed_files = None
        else:
            # rebuild the vector store only when the uploaded files change
            file_key = [(file.name, file.size) for file in files]
            if st.session_state.indexed_files != file_key:
                with st.spinner("Indexing your documents…"):
                    vs = build_vector_store(files)
                st.session_state.retriever = get_retriever(vs)
                st.session_state.indexed_files = file_key
                st.session_state.messages = []
            st.success(f"{len(files)} file(s) indexed")
    else:
        st.session_state.retriever = None
        st.session_state.indexed_files = None
        st.session_state.messages = []


# ── Main page ────────────────────────────────────────────────────────────────
st.title("📚 PageWise")
st.markdown("🔍 **Ask your PDFs anything.** Upload up to 2 PDFs in the sidebar and chat with them.")
st.divider()

# ── Chat display ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents…",
                           disabled=st.session_state.retriever is None):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = graph.invoke({"query": prompt,
                                   "retriever": st.session_state.retriever})
        response = st.write_stream(stream_text(result["response"]))

    st.session_state.messages.append({"role": "assistant", "content": response})
