from pathlib import Path

import streamlit as st

from banking_tower import InvestmentBankingTower
from rag import LocalRAG, read_documents


BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"


@st.cache_resource
def load_rag() -> LocalRAG:
    rag = LocalRAG()
    rag.add_documents(read_documents(DOCUMENTS_DIR))
    return rag


@st.cache_resource
def load_tower() -> InvestmentBankingTower:
    return InvestmentBankingTower(DOCUMENTS_DIR)


def main() -> None:
    st.set_page_config(
        page_title="Local RAG",
        page_icon="📚",
        layout="centered",
    )
    st.title("Investment Banking Assistant Tower")
    st.caption("Grounded research and coordinated transaction-analysis workflows.")

    documents = read_documents(DOCUMENTS_DIR)
    if not documents:
        st.error(f"No .md or .txt files found in {DOCUMENTS_DIR}.")
        return

    rag = load_rag()
    st.sidebar.header("Index")
    st.sidebar.write(f"{len(documents)} document(s) available")
    st.sidebar.write(f"{len(rag.chunks)} chunk(s) indexed")

    research_tab, tower_tab = st.tabs(["Document research", "A2A deal team"])
    with research_tab:
        with st.form("query_form"):
            query = st.text_area("Your question", placeholder="What does this project do?", height=110)
            top_k = st.slider("Context chunks", min_value=1, max_value=5, value=3)
            submitted = st.form_submit_button("Search documents", type="primary")

        if submitted:
            if not query.strip():
                st.warning("Enter a question to search the documents.")
            else:
                st.subheader("Answer")
                st.write(rag.answer(query, top_k=top_k))
                sources = [(chunk, score) for chunk, score in rag.retrieve(query, top_k) if score > 0]
                if sources:
                    st.subheader("Sources")
                    for chunk, score in sources:
                        with st.expander(f"{chunk.source} | relevance {score:.2f}"):
                            st.write(chunk.text)

    with tower_tab:
        st.write("The coordinator routes requests to research, market, valuation, capital-structure, and risk/compliance specialists.")
        with st.form("tower_form"):
            request = st.text_area("Deal-team request", placeholder="Prepare a DCF framework and review leverage for the proposed acquisition.", height=110)
            run_tower = st.form_submit_button("Run specialist review", type="primary")
        if run_tower:
            if not request.strip():
                st.warning("Enter a deal-team request.")
            else:
                task = load_tower().send_message(request)
                st.caption(f"A2A task: {task.task_id} · {task.status}")
                st.markdown(task.artifacts[0].text)


if __name__ == "__main__":
    main()
