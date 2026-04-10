"""
API endpoints for Memory, MCP, and Performance monitoring.

Exposes:
- /api/memory/* - User memory operations
- /api/mcp/tools - Tool management
- /api/metrics - Performance metrics
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import json

from api.memory.manager import get_memory_manager
from api.memory.models import MemoryType, MemoryQuery
from api.mcp.registry import get_mcp_registry, ToolCategory
from api.monitoring.performance import get_performance_monitor
from api.tracing import get_tracer

# Create routers
memory_router = APIRouter(prefix="/api/memory", tags=["memory"])
mcp_router = APIRouter(prefix="/api/mcp", tags=["mcp"])
metrics_router = APIRouter(prefix="/api/metrics", tags=["metrics"])
tracing_router = APIRouter(prefix="/api/tracing", tags=["tracing"])


# ============================================================================
# Memory Endpoints
# ============================================================================

@memory_router.get("/preferences/{user_id}/{repo_id}")
async def get_preferences(user_id: str, repo_id: str):
    """Get all user preferences for a repository."""
    try:
        memory = get_memory_manager()
        prefs = memory.get_preferences(user_id, repo_id)
        return {"user_id": user_id, "repo_id": repo_id, "preferences": prefs}
    except Exception as e:
        raise HTTPException(500, f"Error retrieving preferences: {e}")


@memory_router.post("/preferences/{user_id}/{repo_id}")
async def set_preference(user_id: str, repo_id: str, body: Dict[str, Any]):
    """Set or update a user preference."""
    try:
        key = body.get("key")
        value = body.get("value")
        
        if not key:
            raise ValueError("Missing 'key' field")
        
        memory = get_memory_manager()
        entry = memory.set_preference(user_id, repo_id, key, value)
        
        return {
            "success": True,
            "id": entry.id,
            "key": key,
            "saved_at": entry.updated_at.isoformat(),
        }
    except Exception as e:
        raise HTTPException(400, f"Error setting preference: {e}")


@memory_router.get("/stats/{user_id}/{repo_id}")
async def get_memory_stats(user_id: str, repo_id: str):
    """Get memory statistics for a user/repo."""
    try:
        memory = get_memory_manager()
        stats = memory.get_stats(user_id, repo_id)
        
        return {
            "user_id": user_id,
            "repo_id": repo_id,
            "total_memories": stats.total_count,
            "by_type": stats.by_type,
            "by_tier": stats.by_tier,
            "oldest_memory": stats.oldest_memory.isoformat() if stats.oldest_memory else None,
            "newest_memory": stats.newest_memory.isoformat() if stats.newest_memory else None,
            "avg_weight": round(stats.avg_weight, 2),
            "total_weight": round(stats.total_weight, 2),
        }
    except Exception as e:
        raise HTTPException(500, f"Error retrieving stats: {e}")


@memory_router.delete("/cleanup/{user_id}")
async def cleanup_expired(user_id: Optional[str] = None):
    """Clean up expired memories."""
    try:
        memory = get_memory_manager()
        count = memory.cleanup_expired(user_id)
        
        return {
            "deleted_count": count,
            "message": f"Cleaned up {count} expired memories",
        }
    except Exception as e:
        raise HTTPException(500, f"Error during cleanup: {e}")


@memory_router.get("/knowledge/{user_id}/{repo_id}")
async def search_knowledge(user_id: str, repo_id: str, q: str = Query(..., min_length=1), limit: int = 10):
    """Full-text search over the long-term knowledge base."""
    try:
        memory = get_memory_manager()
        entries = memory.search_knowledge(user_id, repo_id, q, limit=min(limit, 50))
        return {
            "user_id": user_id,
            "repo_id": repo_id,
            "query": q,
            "results": [e.to_dict() for e in entries],
        }
    except Exception as e:
        raise HTTPException(500, f"Error searching knowledge: {e}")


@memory_router.get("/insights/{user_id}/{repo_id}")
async def get_user_insights(user_id: str, repo_id: str, limit: int = 20):
    """Get consolidated insights for a user."""
    try:
        memory = get_memory_manager()
        insights = memory.get_user_insights(user_id, repo_id, limit=min(limit, 100))
        return {
            "user_id": user_id,
            "repo_id": repo_id,
            "insights": [e.to_dict() for e in insights],
        }
    except Exception as e:
        raise HTTPException(500, f"Error retrieving insights: {e}")


@memory_router.post("/consolidate")
async def trigger_consolidation():
    """Manually trigger memory consolidation (episodic → long-term)."""
    try:
        memory = get_memory_manager()
        stats = memory.consolidate_now()
        return {"status": "ok", "consolidation_stats": stats}
    except Exception as e:
        raise HTTPException(500, f"Error during consolidation: {e}")


# ============================================================================
# MCP Tool Endpoints
# ============================================================================

@mcp_router.get("/tools")
async def list_tools(category: Optional[str] = None):
    """List all available MCP tools."""
    try:
        registry = get_mcp_registry()
        
        if category:
            try:
                cat = ToolCategory[category.upper()]
                tools = registry.get_by_category(cat)
            except KeyError:
                raise HTTPException(400, f"Unknown category: {category}")
        else:
            tools = registry.list_all()
        
        return {
            "count": len(tools),
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category.value,
                    "status": tool.status.value,
                    "version": tool.version,
                }
                for tool in tools
            ],
        }
    except Exception as e:
        raise HTTPException(500, f"Error listing tools: {e}")


@mcp_router.get("/tools/{tool_name}")
async def get_tool_schema(tool_name: str):
    """Get schema for a specific tool."""
    try:
        registry = get_mcp_registry()
        schema = registry.get_schema(tool_name)
        
        if not schema:
            raise HTTPException(404, f"Tool '{tool_name}' not found")
        
        return schema
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error retrieving tool schema: {e}")


@mcp_router.post("/tools/execute/{tool_name}")
async def execute_tool(tool_name: str, input_data: Dict[str, Any]):
    """Execute an MCP tool."""
    try:
        registry = get_mcp_registry()
        
        if not registry.is_enabled(tool_name):
            raise HTTPException(400, f"Tool '{tool_name}' is disabled")
        
        result = await registry.execute(tool_name, input_data)
        
        return {
            "success": True,
            "tool": tool_name,
            "result": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Tool execution failed: {e}")


# ============================================================================
# Performance Metrics Endpoints
# ============================================================================

@metrics_router.get("/system")
async def get_system_metrics():
    """Get overall system performance metrics."""
    try:
        monitor = get_performance_monitor()
        stats = monitor.get_stats()
        
        return {
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
            "metrics": stats,
        }
    except Exception as e:
        raise HTTPException(500, f"Error retrieving metrics: {e}")


@metrics_router.get("/metric/{metric_name}")
async def get_metric(metric_name: str):
    """Get statistics for a specific metric."""
    try:
        monitor = get_performance_monitor()
        stats = monitor.get_stats(metric_name)
        
        if not stats:
            raise HTTPException(404, f"Metric '{metric_name}' not found")
        
        return {
            "metric": metric_name,
            "stats": stats,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error retrieving metric: {e}")


@metrics_router.get("/session/{user_id}/{repo_id}")
async def get_session_metrics(user_id: str, repo_id: str):
    """Get performance metrics for a specific session."""
    try:
        monitor = get_performance_monitor()
        report = monitor.get_session_report(user_id, repo_id)
        
        return report
    except Exception as e:
        raise HTTPException(500, f"Error retrieving session metrics: {e}")


@metrics_router.get("/export")
async def export_metrics(format: str = Query("json", regex="^(json|text)$")):
    """Export all collected metrics."""
    try:
        monitor = get_performance_monitor()
        
        if format == "json":
            metrics_json = monitor.export_metrics()
            return json.loads(metrics_json)
        else:
            # Text format
            stats = monitor.get_stats()
            lines = ["Performance Metrics Report", "=" * 50]
            for metric, stat in stats.items():
                lines.append(f"\n{metric}:")
                for key, value in stat.items():
                    lines.append(f"  {key}: {value}")
            
            return {"format": "text", "content": "\n".join(lines)}
    except Exception as e:
        raise HTTPException(500, f"Error exporting metrics: {e}")


# ============================================================================
# Tracing / Feedback Endpoints (Langfuse)
# ============================================================================

@tracing_router.post("/feedback")
async def submit_feedback(body: Dict[str, Any]):
    """
    Record user feedback for a trace.

    Body:
        trace_id (str): Langfuse trace ID.
        thumbs (int): 1 or -1.
        rating (int, optional): 1-5 star rating.
        comment (str, optional): Free-text comment.
    """
    trace_id = body.get("trace_id")
    thumbs = body.get("thumbs")

    if not trace_id or thumbs is None:
        raise HTTPException(400, "Missing required fields: trace_id, thumbs")
    if thumbs not in (1, -1):
        raise HTTPException(400, "thumbs must be 1 or -1")

    try:
        tracer = get_tracer()
        tracer.record_user_feedback(
            trace_id=trace_id,
            thumbs=int(thumbs),
            rating=body.get("rating"),
            comment=body.get("comment"),
        )
        return {"success": True, "trace_id": trace_id}
    except Exception as e:
        raise HTTPException(500, f"Error recording feedback: {e}")


@tracing_router.post("/score")
async def submit_score(body: Dict[str, Any]):
    """
    Add a score to a trace (e.g. LLM-as-Judge results).

    Body:
        trace_id (str): Langfuse trace ID.
        name (str): Score name (e.g. "answer_relevance", "faithfulness").
        value (float): Score value.
        comment (str, optional): Free-text comment.
    """
    trace_id = body.get("trace_id")
    name = body.get("name")
    value = body.get("value")

    if not trace_id or not name or value is None:
        raise HTTPException(400, "Missing required fields: trace_id, name, value")

    try:
        tracer = get_tracer()
        tracer.score_by_trace_id(
            trace_id=trace_id,
            name=str(name),
            value=float(value),
            comment=body.get("comment"),
        )
        return {"success": True, "trace_id": trace_id, "score_name": name}
    except Exception as e:
        raise HTTPException(500, f"Error recording score: {e}")


@tracing_router.get("/status")
async def tracing_status():
    """Check whether Langfuse tracing is enabled."""
    tracer = get_tracer()
    return {"enabled": tracer.enabled}


# Summary function to register all routers
def register_routers(app):
    """Register all routers to FastAPI app."""
    app.include_router(memory_router)
    app.include_router(mcp_router)
    app.include_router(metrics_router)
    app.include_router(tracing_router)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info("✓ Registered Memory, MCP, Metrics, and Tracing API endpoints")
