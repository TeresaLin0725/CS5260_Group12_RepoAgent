"""
Performance monitoring and metrics collection for Memory & MCP systems.

Tracks:
- Tool execution time
- Memory operation time
- Cache hit/miss rates
- Agent decision latency
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric measurement."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricStats:
    """Statistics for a metric."""
    count: int = 0
    total: float = 0.0
    min: float = float('inf')
    max: float = 0.0
    avg: float = 0.0
    
    def add(self, value: float) -> None:
        """Add a value to the statistics."""
        self.count += 1
        self.total += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.avg = self.total / self.count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "total": round(self.total, 3),
            "min": round(self.min, 3) if self.min != float('inf') else 0,
            "max": round(self.max, 3),
            "avg": round(self.avg, 3),
        }


class PerformanceMonitor:
    """
    Monitor and collect performance metrics.
    
    Metrics tracked:
    - tool_execution_time: Tool execution latency
    - memory_operation_time: Memory op latency
    - agent_iteration_time: Single ReAct iteration time
    - cache_hit_ratio: Memory cache effectiveness
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.stats: Dict[str, MetricStats] = defaultdict(MetricStats)
        self.session_metrics: Dict[str, Dict[str, Any]] = {}
        self._start_time = time.time()
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a single metric point."""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            metadata=metadata or {},
        )
        self.metrics[metric_name].append(point)
        self.stats[metric_name].add(value)
        
        # Log if value is significantly high (potential bottleneck)
        if metric_name.endswith("_time"):
            stats = self.stats[metric_name]
            if stats.count > 5 and value > stats.avg * 1.5:
                logger.warning(
                    f"Slow {metric_name}: {value:.3f}ms (avg: {stats.avg:.3f}ms)"
                )
    
    def start_timer(self) -> 'TimerContext':
        """Start a timing context."""
        return TimerContext(self)
    
    def get_stats(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for one or all metrics."""
        if metric_name:
            return self.stats[metric_name].to_dict()
        
        return {
            name: stats.to_dict()
            for name, stats in self.stats.items()
        }
    
    def record_session(
        self,
        user_id: str,
        repo_id: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Record session-level metrics."""
        key = f"{user_id}:{repo_id}"
        self.session_metrics[key] = {
            "timestamp": datetime.utcnow().isoformat(),
            **metrics,
        }
    
    def get_session_report(self, user_id: str, repo_id: str) -> Dict[str, Any]:
        """Get performance report for a session."""
        key = f"{user_id}:{repo_id}"
        session = self.session_metrics.get(key, {})
        
        return {
            "session": key,
            "metrics": session,
            "system_stats": self.get_stats(),
            "uptime_seconds": time.time() - self._start_time,
        }
    
    def export_metrics(self) -> str:
        """Export metrics as JSON."""
        return json.dumps({
            "stats": self.get_stats(),
            "sessions": self.session_metrics,
            "collected_at": datetime.utcnow().isoformat(),
        }, indent=2, ensure_ascii=False)


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: PerformanceMonitor):
        """Initialize timer."""
        self.monitor = monitor
        self.start_time = None
        self.metric_name = None
        self.metadata = None
    
    def __call__(
        self,
        metric_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'TimerContext':
        """Configure the timer."""
        self.metric_name = metric_name
        self.metadata = metadata or {}
        return self
    
    def __enter__(self) -> 'TimerContext':
        """Enter context."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and record metric."""
        elapsed_ms = (time.time() - self.start_time) * 1000
        
        if self.metric_name:
            self.monitor.record_metric(
                self.metric_name,
                elapsed_ms,
                self.metadata,
            )


# Global instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
