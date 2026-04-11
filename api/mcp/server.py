"""
MCP Server — Bootstraps the MCP tool registry with built-in tools.

Provides:
  - MCPServer: Thin wrapper around MCPToolRegistry for lifecycle management.
  - initialize_mcp: One-call bootstrap that registers all default tools.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from api.mcp.registry import (
    MCPToolRegistry,
    MCPTool,
    get_mcp_registry,
)

logger = logging.getLogger(__name__)


class MCPServer:
    """Lightweight MCP server that owns a tool registry."""

    def __init__(self, registry: Optional[MCPToolRegistry] = None):
        self.registry = registry or get_mcp_registry()
        self._initialized = False

    def initialize(self, extra_tools: Optional[List[MCPTool]] = None) -> None:
        """Register any extra tools supplied by the caller."""
        if self._initialized:
            logger.debug("MCP server already initialized — skipping.")
            return

        for tool in extra_tools or []:
            try:
                self.registry.register(tool)
                logger.debug("Registered MCP tool: %s", tool.name)
            except ValueError:
                pass  # already registered

        self._initialized = True
        logger.info("MCP server initialized with %d tool(s).", len(self.registry.list_all()))

    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.registry.list_all()]

    async def execute(self, tool_name: str, input_data: Dict[str, Any]) -> str:
        return await self.registry.execute(tool_name, input_data)


_mcp_server: Optional[MCPServer] = None


def initialize_mcp(
    extra_tools: Optional[List[MCPTool]] = None,
) -> MCPServer:
    """Get (or create) the global MCPServer and initialize it.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer()
    _mcp_server.initialize(extra_tools=extra_tools)
    return _mcp_server
