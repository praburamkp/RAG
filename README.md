# Local RAG Operations

A small Python implementation of retrieval-augmented generation (RAG) operations. It indexes Markdown and text files, retrieves relevant chunks with TF-IDF and cosine similarity, and returns the retrieved context as an extractive answer.

## Run

```bash
python3 -m unittest -v
python3 rag.py ingest ./documents
python3 rag.py query "What does the project do?"
```

## Streamlit frontend

Install the frontend dependency and start the web app:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

The app indexes the `.txt` and `.md` files in `documents/` when it starts, then displays the retrieved answer and matching source chunks.

Use `--index path/to/index.json` to choose an index location, `--top-k 5` to retrieve more context, or `--chunk-size 120 --overlap 20` to tune chunking.

The implementation uses only the Python standard library, so it works offline and does not need an API key. To connect a generative model, pass the retrieved text from `LocalRAG.retrieve()` to the model client of your choice and keep the returned source chunks as citations.

## Document layout

Put `.txt` and `.md` files below a folder such as `documents/`. The source path is stored with each chunk so callers can display citations.
