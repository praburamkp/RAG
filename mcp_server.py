"""MCP stdio server exposing the investment-banking assistant tower as tools.

Run this process from an MCP client (such as VS Code) or send one JSON-RPC
message per line from a terminal. Logs are intentionally sent only to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from banking_tower import InvestmentBankingTower


class InvestmentBankingMCPServer:
    def __init__(self, documents_dir: Path) -> None:
        self.tower = InvestmentBankingTower(documents_dir)

    @staticmethod
    def _response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    @staticmethod
    def tools() -> list[dict[str, Any]]:
        return [
            {
                "name": "run_investment_banking_review",
                "description": "Route a deal-team request through the A2A investment-banking specialist tower.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"request": {"type": "string", "description": "The deal-team question or work request."}},
                    "required": ["request"],
                },
            },
            {
                "name": "search_research_room",
                "description": "Search approved local research documents and return cited source excerpts.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_a2a_agent_card",
                "description": "Return the A2A Agent Card describing the specialist tower.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "run_investment_banking_review":
                request = arguments["request"]
                if not isinstance(request, str) or not request.strip():
                    raise ValueError("request must be a non-empty string")
                task = self.tower.send_message(request)
                return {"content": [{"type": "text", "text": task.artifacts[0].text}], "structuredContent": task.to_dict()}
            if name == "search_research_room":
                query = arguments["query"]
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("query must be a non-empty string")
                top_k = arguments.get("top_k", 3)
                if not isinstance(top_k, int) or not 1 <= top_k <= 10:
                    raise ValueError("top_k must be an integer between 1 and 10")
                sources = [
                    {"source": chunk.source, "relevance": round(score, 4), "text": chunk.text}
                    for chunk, score in self.tower.rag.retrieve(query, top_k) if score > 0
                ]
                text = "\n".join(f"- {item['text']} [Source: {item['source']}]" for item in sources) or "No matching research was found."
                return {"content": [{"type": "text", "text": text}], "structuredContent": {"sources": sources}}
            if name == "get_a2a_agent_card":
                card = self.tower.agent_card().to_dict()
                return {"content": [{"type": "text", "text": json.dumps(card, indent=2)}], "structuredContent": card}
            return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}
        except (KeyError, ValueError) as error:
            return {"content": [{"type": "text", "text": str(error)}], "isError": True}

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")
        if method == "notifications/initialized":
            return None
        if request.get("jsonrpc") != "2.0":
            return self._error(request_id, -32600, "JSON-RPC version must be 2.0")
        if method == "initialize":
            return self._response(request_id, {
                "protocolVersion": request.get("params", {}).get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "investment-banking-tower", "version": "0.1.0"},
            })
        if method == "tools/list":
            return self._response(request_id, {"tools": self.tools()})
        if method == "tools/call":
            params = request.get("params", {})
            name = params.get("name")
            if not isinstance(name, str):
                return self._error(request_id, -32602, "tools/call requires a tool name")
            arguments = params.get("arguments", {})
            if not isinstance(arguments, dict):
                return self._error(request_id, -32602, "arguments must be an object")
            return self._response(request_id, self._call_tool(name, arguments))
        return self._error(request_id, -32601, f"Method not found: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Investment Banking Tower MCP stdio server")
    parser.add_argument("--documents", type=Path, default=Path("documents"))
    args = parser.parse_args()
    server = InvestmentBankingMCPServer(args.documents)
    for line in sys.stdin:
        try:
            response = server.handle(json.loads(line))
            if response is not None:
                print(json.dumps(response), flush=True)
        except json.JSONDecodeError as error:
            print(json.dumps(server._error(None, -32700, f"Parse error: {error.msg}")), flush=True)


if __name__ == "__main__":
    main()
