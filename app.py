from pathlib import Path

import streamlit as st

from RAG.rag import LocalRAG, read_documents


BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"


@st.cache_resource
def load_rag() -> LocalRAG:
    rag = LocalRAG()
    rag.add_documents(read_documents(DOCUMENTS_DIR))
    return rag


def main() -> None:
    st.set_page_config(
        page_title="Local RAG",
        page_icon="📚",
        layout="centered",
    )
    st.title("Ask your documents")
    st.caption("Search the local document index for a grounded answer.")

    documents = read_documents(DOCUMENTS_DIR)
    if not documents:
        st.error(f"No .md or .txt files found in {DOCUMENTS_DIR}.")
        return

    rag = load_rag()
    st.sidebar.header("Index")
    st.sidebar.write(f"{len(documents)} document(s) available")
    st.sidebar.write(f"{len(rag.chunks)} chunk(s) indexed")

    with st.form("query_form"):
        query = st.text_area(
            "Your question",
            placeholder="What does this project do?",
            height=110,
        )
        top_k = st.slider("Context chunks", min_value=1, max_value=5, value=3)
        submitted = st.form_submit_button("Search documents", type="primary")

    if submitted:
        if not query.strip():
            st.warning("Enter a question to search the documents.")
            return

        st.subheader("Answer")
        st.write(rag.answer(query, top_k=top_k))

        sources = rag.retrieve(query, top_k=top_k)
        sources = [(chunk, score) for chunk, score in sources if score > 0]
        if sources:
            st.subheader("Sources")
            for chunk, score in sources:
                with st.expander(f"{chunk.source} | relevance {score:.2f}"):
                    st.write(chunk.text)


if __name__ == "__main__":
    main()
