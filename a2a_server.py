"""Minimal JSON-RPC HTTP binding for the local investment-banking A2A tower."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from banking_tower import InvestmentBankingTower


class A2AHandler(BaseHTTPRequestHandler):
    tower: InvestmentBankingTower

    def _json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/.well-known/agent-card.json":
            self._json(HTTPStatus.OK, self.tower.agent_card().to_dict())
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/a2a":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            params = request.get("params", {})
            if request.get("method") in {"message/send", "sendMessage"}:
                message = params.get("message", {})
                text = "\n".join(part.get("text", "") for part in message.get("parts", []) if part.get("kind") == "text")
                if not text:
                    raise ValueError("A text message part is required")
                result = self.tower.send_message(text, params.get("contextId")).to_dict()
            elif request.get("method") in {"tasks/get", "getTask"}:
                result = self.tower.tasks[params["id"]].to_dict()
            else:
                raise ValueError("Unsupported method")
            self._json(HTTPStatus.OK, {"jsonrpc": "2.0", "id": request.get("id"), "result": result})
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"jsonrpc": "2.0", "id": None, "error": {"code": -32602, "message": str(error)}})

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Investment Banking A2A tower")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--documents", type=Path, default=Path("documents"))
    args = parser.parse_args()
    endpoint = f"http://localhost:{args.port}/a2a"
    A2AHandler.tower = InvestmentBankingTower(args.documents, endpoint)
    server = ThreadingHTTPServer(("", args.port), A2AHandler)
    print(f"A2A card: http://localhost:{args.port}/.well-known/agent-card.json")
    server.serve_forever()


if __name__ == "__main__":
    main()
