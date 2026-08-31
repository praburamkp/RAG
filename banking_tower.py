"""An investment-banking research tower coordinated through local A2A tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from a2a import AgentCard, AgentSkill, Artifact, Message, Task
from rag import LocalRAG, read_documents


@dataclass
class Specialist:
    name: str
    skill: AgentSkill

    def execute(self, prompt: str, rag: LocalRAG) -> str:
        if self.name == "company-research":
            matches = [(chunk, score) for chunk, score in rag.retrieve(prompt, 3) if score > 0]
            if not matches:
                return "No supporting information was found in the approved research room."
            return "\n".join(f"- {chunk.text} [Source: {chunk.source}]" for chunk, _ in matches)
        if self.name == "valuation":
            return ("Build the valuation bridge from approved assumptions: comparable-company "
                    "multiples, precedent transactions, DCF/WACC and sensitivity cases. "
                    "No market data or valuation conclusion is invented by this agent.")
        if self.name == "capital-structure":
            return ("Assess leverage, liquidity, debt maturities, covenant headroom and financing "
                    "alternatives using supplied financial statements and transaction assumptions.")
        if self.name == "market-intelligence":
            return ("Assess sector catalysts, competitive positioning and transaction comparables from "
                    "the approved research room; validate time-sensitive data before use.")
        return ("Flag diligence gaps, conflicts, material non-public information controls and the need "
                "for qualified legal, tax and compliance review.")


class InvestmentBankingTower:
    """Routes a request to specialist agents and records an auditable task transcript."""

    def __init__(self, documents_dir: Path | None = None, endpoint: str = "http://localhost:8080/a2a") -> None:
        self.rag = LocalRAG()
        if documents_dir and documents_dir.exists():
            self.rag.add_documents(read_documents(documents_dir))
        self.specialists = [
            Specialist("company-research", AgentSkill("company-research", "Company research", "Grounded company and filing research", ["research", "filings"])),
            Specialist("market-intelligence", AgentSkill("market-intelligence", "Market intelligence", "Sector and transaction context", ["markets", "sector"])),
            Specialist("valuation", AgentSkill("valuation", "Valuation", "Valuation framework and sensitivities", ["valuation", "DCF", "multiples"])),
            Specialist("capital-structure", AgentSkill("capital-structure", "Capital structure", "Leverage and financing analysis", ["debt", "financing"])),
            Specialist("risk-compliance", AgentSkill("risk-compliance", "Risk and compliance", "Diligence and escalation controls", ["risk", "compliance"])),
        ]
        self.tasks: dict[str, Task] = {}
        self.endpoint = endpoint

    def agent_card(self) -> AgentCard:
        return AgentCard("Investment Banking Assistant Tower", "Routes research and transaction-analysis requests to specialist agents.", self.endpoint, skills=[agent.skill for agent in self.specialists])

    def _select_agents(self, prompt: str) -> list[Specialist]:
        text = prompt.lower()
        names = {"company-research", "risk-compliance"}
        if any(word in text for word in ("valuation", "dcf", "multiple", "ev/", "enterprise value")):
            names.add("valuation")
        if any(word in text for word in ("debt", "leverage", "liquidity", "financing", "covenant")):
            names.add("capital-structure")
        if any(word in text for word in ("market", "sector", "competitor", "transaction", "m&a")):
            names.add("market-intelligence")
        return [agent for agent in self.specialists if agent.name in names]

    def send_message(self, text: str, context_id: str | None = None) -> Task:
        task = Task()
        if context_id:
            task.context_id = context_id
        task.history.append(Message("user", text))
        task.status = "working"
        reports = []
        for agent in self._select_agents(text):
            report = agent.execute(text, self.rag)
            reports.append(f"## {agent.skill.name}\n{report}")
            task.history.append(Message("agent", report))
        output = ("# Investment Banking Assistant Tower\n\n" + "\n\n".join(reports) +
                  "\n\n---\nFor research and workflow support only; not investment, legal, tax, or compliance advice. "
                  "Validate all time-sensitive information and obtain required approvals.")
        task.artifacts.append(Artifact("tower-report.md", output, "Specialist research and analysis handoff"))
        task.status = "completed"
        self.tasks[task.task_id] = task
        return task
