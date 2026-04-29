"""Base agent abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentTrace:
    agent_name: str
    messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.trace = AgentTrace(agent_name=name)

    def log(self, message: str, **metadata: Any) -> None:
        self.trace.messages.append(message)
        if metadata:
            self.trace.metadata.update(metadata)
