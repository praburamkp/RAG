import unittest
from pathlib import Path

from mcp_server import InvestmentBankingMCPServer


class InvestmentBankingMCPServerTests(unittest.TestCase):
    def setUp(self):
        self.server = InvestmentBankingMCPServer(Path("documents"))

    def test_lists_tools_after_initialization(self):
        initialized = self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
        tools = self.server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual(initialized["result"]["serverInfo"]["name"], "investment-banking-tower")
        self.assertIn("run_investment_banking_review", [tool["name"] for tool in tools["result"]["tools"]])

    def test_calls_tower_tool(self):
        response = self.server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "run_investment_banking_review", "arguments": {"request": "Prepare a DCF valuation."}}})
        self.assertFalse(response["result"].get("isError", False))
        self.assertIn("Valuation", response["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
