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

## Investment-banking A2A tower

The repository also includes a local, multi-agent investment-banking assistant
tower.  It uses the Agent2Agent (A2A) concepts of an Agent Card, messages,
tasks and artifacts. A coordinator routes a request to company research, market
intelligence, valuation, capital structure, and risk/compliance specialists.
The company-research specialist is grounded in the local document room; the
other specialists provide workflow frameworks and do not invent market data.

Start the JSON-RPC service:

```bash
python3 a2a_server.py --documents ./documents --port 8080
```

Discover its card at `http://localhost:8080/.well-known/agent-card.json` and
send `message/send` JSON-RPC requests to `http://localhost:8080/a2a`. The
response is an A2A-style task with a report artifact and complete transcript.
This is research/workflow support only, not investment, legal, tax, or
compliance advice; validate live data and apply the firm's approval controls.

## MCP from the terminal or VS Code

The same tower is available as a dependency-free MCP stdio server. Start it:

```bash
python3 mcp_server.py --documents ./documents
```

It accepts newline-delimited JSON-RPC on standard input. Paste this request to
list the available MCP tools:

```json
{"jsonrpc":"2.0","id":1,"method":"tools/list"}
```

Call the multi-agent workflow directly:

```json
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"run_investment_banking_review","arguments":{"request":"Prepare a DCF valuation and assess leverage."}}}
```

For VS Code, register `python3` as the command and `mcp_server.py --documents ./documents`
as its arguments in your MCP server configuration. The server exposes
`run_investment_banking_review`, `search_research_room`, and `get_a2a_agent_card`.

Use `--index path/to/index.json` to choose an index location, `--top-k 5` to retrieve more context, or `--chunk-size 120 --overlap 20` to tune chunking.

The implementation uses only the Python standard library, so it works offline and does not need an API key. To connect a generative model, pass the retrieved text from `LocalRAG.retrieve()` to the model client of your choice and keep the returned source chunks as citations.

## Document layout

Put `.txt` and `.md` files below a folder such as `documents/`. The source path is stored with each chunk so callers can display citations.
