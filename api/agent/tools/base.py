from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class AgentTool:
    name: str
    action_tag: str
    description: str
    keywords: Iterable[str]


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[AgentTool]:
        return self._tools.get(name)

    def all(self) -> Iterable[AgentTool]:
        return self._tools.values()
