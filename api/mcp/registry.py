"""
MCP Tool Registry - Standardized tool registration and execution layer.

Implements the Model Context Protocol tool registration pattern to provide
a unified interface for all tools (RAG, file operations, memory, etc).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, List
from enum import Enum
import json


class ToolCategory(str, Enum):
    """Tool categories for organization and access control."""
    SEARCH = "search"        # Information retrieval
    READ = "read"           # File/data reading
    WRITE = "write"         # File/data writing
    MEMORY = "memory"       # Memory operations
    ANALYSIS = "analysis"   # Code analysis
    EXPORT = "export"       # Data export
    CUSTOM = "custom"       # Custom tools


class ToolStatus(str, Enum):
    """Tool availability status."""
    AVAILABLE = "available"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    DISABLED = "disabled"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = False
    default: Any = None
    enum_values: Optional[List[str]] = None


@dataclass
class ToolSchema:
    """JSON Schema-like definition of tool interface."""
    #input_schema: Dict[str, Any]
    #output_schema: Dict[str, Any]
    parameters: List[ToolParameter] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum_values,
                }
                for p in self.parameters
            ]
        }


@dataclass
class MCPTool:
    """MCP Tool definition."""
    name: str
    description: str
    category: ToolCategory
    schema: ToolSchema
    handler: Callable[[Dict[str, Any]], Awaitable[str]]
    status: ToolStatus = ToolStatus.AVAILABLE
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    requires_auth: bool = False
    rate_limit: Optional[int] = None  # requests per minute

    async def execute(self, input_data: Dict[str, Any]) -> str:
        """Execute the tool with given input."""
        # Validate input parameters
        self._validate_input(input_data)
        
        # Call handler
        result = await self.handler(input_data)
        return result

    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input against schema."""
        for param in self.schema.parameters:
            if param.required and param.name not in input_data:
                raise ValueError(f"Missing required parameter: {param.name}")
            
            if param.name in input_data:
                value = input_data[param.name]
                # Type validation
                if not self._check_type(value, param.type):
                    raise TypeError(
                        f"Parameter '{param.name}' has wrong type. "
                        f"Expected {param.type}, got {type(value).__name__}"
                    )
                # Enum validation
                if param.enum_values and value not in param.enum_values:
                    raise ValueError(
                        f"Parameter '{param.name}' value '{value}' not in allowed values: "
                        f"{param.enum_values}"
                    )

    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True  # Unknown type, skip validation
        return isinstance(value, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "status": self.status.value,
            "version": self.version,
            "schema": self.schema.to_dict(),
            "requires_auth": self.requires_auth,
            "rate_limit": self.rate_limit,
            "tags": self.tags,
            "metadata": self.metadata,
        }


class MCPToolRegistry:
    """
    Unified tool registry implementing MCP pattern.
    
    Manages registration, discovery, and execution of all tools
    (RAG, file operations, memory, etc) through a standard interface.
    """

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, MCPTool] = {}
        self._by_category: Dict[ToolCategory, List[str]] = {cat: [] for cat in ToolCategory}
        self._by_tag: Dict[str, List[str]] = {}
        self._auth_required: set = set()

    def register(self, tool: MCPTool) -> None:
        """Register a new tool."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        
        self._tools[tool.name] = tool
        self._by_category[tool.category].append(tool.name)
        
        # Index by tags
        for tag in tool.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(tool.name)
        
        # Track auth requirements
        if tool.requires_auth:
            self._auth_required.add(tool.name)

    def get(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_by_category(self, category: ToolCategory) -> List[MCPTool]:
        """Get all tools in a category."""
        tool_names = self._by_category.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_by_tag(self, tag: str) -> List[MCPTool]:
        """Get all tools with a specific tag."""
        tool_names = self._by_tag.get(tag, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def list_all(self) -> List[MCPTool]:
        """List all registered tools."""
        return list(self._tools.values())

    async def execute(self, tool_name: str, input_data: Dict[str, Any]) -> str:
        """Execute a tool."""
        tool = self.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        if tool.status == ToolStatus.DISABLED:
            raise ValueError(f"Tool '{tool_name}' is disabled")
        
        return await tool.execute(input_data)

    def get_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool schema for documentation."""
        tool = self.get(tool_name)
        if tool is None:
            return None
        return tool.to_dict()

    def list_schemas(self, category: Optional[ToolCategory] = None) -> Dict[str, Any]:
        """Get schemas for all tools, optionally filtered by category."""
        tools = (
            self.get_by_category(category) if category 
            else self.list_all()
        )
        return {
            tool.name: tool.to_dict()
            for tool in tools
        }

    def is_enabled(self, tool_name: str) -> bool:
        """Check if tool is enabled."""
        tool = self.get(tool_name)
        return tool is not None and tool.status == ToolStatus.AVAILABLE

    def disable(self, tool_name: str) -> bool:
        """Disable a tool."""
        tool = self.get(tool_name)
        if tool is None:
            return False
        tool.status = ToolStatus.DISABLED
        return True

    def enable(self, tool_name: str) -> bool:
        """Enable a previously disabled tool."""
        tool = self.get(tool_name)
        if tool is None:
            return False
        tool.status = ToolStatus.AVAILABLE
        return True


# Global instance
_mcp_registry: Optional[MCPToolRegistry] = None


def get_mcp_registry() -> MCPToolRegistry:
    """Get or create the global MCP registry."""
    global _mcp_registry
    if _mcp_registry is None:
        _mcp_registry = MCPToolRegistry()
    return _mcp_registry
