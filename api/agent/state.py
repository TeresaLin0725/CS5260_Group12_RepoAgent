from dataclasses import dataclass, field
from typing import List, Optional

from api.agent.events import AgentEvent


@dataclass
class AgentRunState:
    query: str
    language: str = "en"
    selected_tool: Optional[str] = None
    events: List[AgentEvent] = field(default_factory=list)
