"""
Integration Guide: Memory & MCP Integration for ReAct Agent

This module provides integration patterns and examples for incorporating
the memory system and MCP tools into the existing agent architecture.
"""

# ============================================================================
# 1. INITIALIZATION IN main.py
# ============================================================================

"""
Add this to api/main.py in the app initialization section:

```python
# At the top of main.py, after other imports
from api.mcp.server import initialize_mcp
from api.memory.manager import get_memory_manager

# In the app startup (before serving requests)
# Initialize memory manager
memory_manager = get_memory_manager()
logger.info("Memory manager initialized")

# Initialize MCP server
mcp_server = initialize_mcp()
logger.info("MCP server initialized")
```
"""

# ============================================================================
# 2. UPDATING websocket_wiki.py
# ============================================================================

"""
Update the WebSocket handler to pass RAG tools to MCP initialization:

```python
# In websocket_wiki.py, in your WebSocket endpoint handler

from api.agent.tools.search_tools import build_react_tools
from api.mcp.server import initialize_mcp

async def handle_chat_message(request_data, ...):
    # Build RAG tools as before
    tools = build_react_tools(rag_instance, ...)
    
    # Initialize MCP with RAG tools (only once per session/app)
    if not mcp_initialized:
        mcp_server = initialize_mcp(tools)
        mcp_initialized = True
    
    # Get MCP registry for tool execution
    registry = get_mcp_registry()
    
    # ... rest of your handler
```
"""

# ============================================================================
# 3. UPDATING ReAct LOOP
# ============================================================================

"""
Update the ReAct runner to use MCP tools:

```python
# In api/agent/react.py

import json
from api.mcp.registry import get_mcp_registry
from api.memory.manager import get_memory_manager

class ReActRunner:
    def __init__(self, ...):
        # ... existing init code
        self.mcp_registry = get_mcp_registry()
        self.memory = get_memory_manager()
    
    async def execute_tool(self, tool_name: str, tool_input: str) -> str:
        '''Execute a tool using MCP registry.'''
        try:
            # Try to parse input as JSON
            try:
                input_data = json.loads(tool_input)
            except:
                # Fall back to simple string input
                input_data = {"input": tool_input}
            
            # Execute via MCP registry
            return await self.mcp_registry.execute(tool_name, input_data)
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Tool execution failed: {e}"
    
    async def run(self, llm_fn, ...):
        '''Enhanced ReAct loop with memory support.'''
        # Record interaction
        user_id = ...  # Get from context
        repo_id = ...
        self.memory.set_preference(
            user_id, repo_id,
            "last_query_time",
            {"timestamp": datetime.utcnow().isoformat()}
        )
        
        # ... rest of ReAct loop, using execute_tool instead of direct calls
```
"""

# ============================================================================
# 4. UPDATING PLANNER
# ============================================================================

"""
Update the planner to consider memory when deciding tools:

```python
# In api/agent/planner.py

from api.memory.manager import get_memory_manager
from api.mcp.registry import get_mcp_registry

class RuleBasedPlanner:
    def __init__(self, registry, user_id: str = None, repo_id: str = None):
        self.registry = registry
        self.mcp_registry = get_mcp_registry()
        self.memory = get_memory_manager()
        self.user_id = user_id
        self.repo_id = repo_id
    
    def plan(self, query: str) -> ToolPlan:
        # Check user preferences for context
        if self.user_id and self.repo_id:
            prefs = self.memory.get_preferences(self.user_id, self.repo_id)
            preferred_tools = prefs.get("preferred_tools", [])
        else:
            preferred_tools = []
        
        # ... rest of planning logic, considering preferred_tools
```
"""

# ============================================================================
# 5. API ENDPOINTS FOR MEMORY
# ============================================================================

"""
Add these endpoints to api/api.py for managing memory:

```python
from fastapi import APIRouter, HTTPException
from api.memory.manager import get_memory_manager
from api.mcp.registry import get_mcp_registry

memory_router = APIRouter(prefix="/api/memory", tags=["memory"])

@memory_router.get("/preferences/{user_id}/{repo_id}")
async def get_preferences(user_id: str, repo_id: str):
    '''Get all user preferences.'''
    memory = get_memory_manager()
    prefs = memory.get_preferences(user_id, repo_id)
    return {"preferences": prefs}

@memory_router.post("/preferences/{user_id}/{repo_id}")
async def set_preference(user_id: str, repo_id: str, body: dict):
    '''Set a user preference.'''
    key = body.get("key")
    value = body.get("value")
    if not key:
        raise HTTPException(400, "Missing 'key' field")
    
    memory = get_memory_manager()
    entry = memory.set_preference(user_id, repo_id, key, value)
    return {"id": entry.id, "stored": True}

@memory_router.get("/insights/{user_id}/{repo_id}")
async def get_insights(user_id: str, repo_id: str):
    '''Get memory insights for user.'''
    memory = get_memory_manager()
    stats = memory.get_stats(user_id, repo_id)
    return {
        "total_count": stats.total_count,
        "by_type": stats.by_type,
        "avg_weight": round(stats.avg_weight, 2),
    }

@memory_router.get("/tools")
async def list_tools():
    '''List all available MCP tools.'''
    registry = get_mcp_registry()
    return {
        "tools": [tool.to_dict() for tool in registry.list_all()]
    }

@memory_router.post("/tools/execute/{tool_name}")
async def execute_tool(tool_name: str, input_data: dict):
    '''Execute an MCP tool.'''
    registry = get_mcp_registry()
    try:
        result = await registry.execute(tool_name, input_data)
        return {"result": result}
    except Exception as e:
        raise HTTPException(500, f"Tool execution failed: {e}")

# Add to your FastAPI app
app.include_router(memory_router)
```
"""

# ============================================================================
# 6. EXAMPLE: MEMORY-AWARE AGENT SETUP
# ============================================================================

"""
Complete example of memory-aware agent initialization:

```python
# In your WebSocket or API handler

from api.agent.react import ReActRunner
from api.agent.planner import RuleBasedPlanner
from api.agent.tools.search_tools import build_react_tools
from api.mcp.server import initialize_mcp
from api.memory.manager import get_memory_manager

async def setup_memory_aware_agent(
    user_id: str,
    repo_id: str,
    rag_instance,
    repo_url: str,
) -> tuple[ReActRunner, RuleBasedPlanner]:
    '''Setup agent with memory integration.'''
    
    # Build RAG tools
    tools = build_react_tools(rag_instance, repo_url=repo_url)
    
    # Initialize MCP (safe to call multiple times)
    mcp_server = initialize_mcp(tools)
    
    # Create memory-aware planner
    registry = mcp_server.registry
    planner = RuleBasedPlanner(registry, user_id, repo_id)
    
    # Create ReAct runner with memory manager
    react_runner = ReActRunner(tools, max_iterations=3)
    react_runner.memory = get_memory_manager()
    
    # Record session start
    memory = get_memory_manager()
    memory.set_preference(user_id, repo_id, "session_started", {
        "timestamp": datetime.utcnow().isoformat(),
        "repo_url": repo_url,
    })
    
    return react_runner, planner
```
"""

print("Integration guide loaded - see docstrings for implementation details")
