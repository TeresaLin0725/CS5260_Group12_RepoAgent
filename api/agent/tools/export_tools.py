from api.agent.tools.base import AgentTool, ToolRegistry


def build_export_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        AgentTool(
            name="GENERATE_PDF",
            action_tag="[ACTION:GENERATE_PDF]",
            description="Generate a repository PDF report.",
            keywords=(
                "pdf",
                "report",
                "technical report",
                "报告",
                "pdf报告",
            ),
        )
    )
    registry.register(
        AgentTool(
            name="GENERATE_PPT",
            action_tag="[ACTION:GENERATE_PPT]",
            description="Generate a repository PPT presentation.",
            keywords=(
                "ppt",
                "slides",
                "presentation",
                "deck",
                "演示",
                "幻灯片",
            ),
        )
    )
    registry.register(
        AgentTool(
            name="GENERATE_VIDEO",
            action_tag="[ACTION:GENERATE_VIDEO]",
            description="Generate a repository video overview.",
            keywords=(
                "video",
                "walkthrough",
                "overview video",
                "视频",
            ),
        )
    )
    registry.register(
        AgentTool(
            name="GENERATE_POSTER",
            action_tag="[ACTION:GENERATE_POSTER]",
            description="Generate a repository illustrated poster via NanoBanana.",
            keywords=(
                "poster",
                "illustrated poster",
                "pictorial",
                "infographic",
                "画报",
                "海报",
                "图文海报",
                "画报制作",
            ),
        )
    )
    return registry
