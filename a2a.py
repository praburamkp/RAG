"""Small, transport-neutral building blocks for A2A task delegation.

The objects deliberately mirror the public Agent2Agent vocabulary (Agent Card,
Message, Task and Artifact) while keeping the local demo dependency-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentSkill:
    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)


@dataclass
class AgentCard:
    name: str
    description: str
    url: str
    version: str = "0.1.0"
    protocol_version: str = "0.3"
    skills: list[AgentSkill] = field(default_factory=list)
    default_input_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    default_output_modes: list[str] = field(default_factory=lambda: ["text/plain"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "protocolVersion": self.protocol_version,
            "capabilities": {"streaming": False, "pushNotifications": False},
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "skills": [asdict(skill) for skill in self.skills],
        }


@dataclass
class Message:
    role: str
    text: str
    message_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "messageId": self.message_id,
            "role": self.role,
            "parts": [{"kind": "text", "text": self.text}],
        }


@dataclass
class Artifact:
    name: str
    text: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifactId": str(uuid4()),
            "name": self.name,
            "description": self.description,
            "parts": [{"kind": "text", "text": self.text}],
        }


@dataclass
class Task:
    context_id: str = field(default_factory=lambda: str(uuid4()))
    task_id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "submitted"
    history: list[Message] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "contextId": self.context_id,
            "status": {"state": self.status, "timestamp": utc_now()},
            "history": [message.to_dict() for message in self.history],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }
