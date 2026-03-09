from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AgentEventType(str, Enum):
    PLAN_CREATED = "plan_created"
    TOOL_SELECTED = "tool_selected"
    TOOL_SKIPPED = "tool_skipped"
    REACT_STEP = "react_step"
    REACT_TOOL_CALL = "react_tool_call"
    REACT_FINISHED = "react_finished"


@dataclass
class AgentEvent:
    event_type: AgentEventType
    message: str
    tool_name: Optional[str] = None
