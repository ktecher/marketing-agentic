"""Shared builder: a specialist is a reason -> act -> observe (ReAct) loop over its tools."""

from langchain.agents import create_agent

from core.config import AgentConfig, AppConfig
from core.models import get_model
from tools.registry import build_tools


def build_react_agent(name: str, cfg: AgentConfig, app: AppConfig):
    return create_agent(
        get_model(app.model_for(cfg.model)),
        build_tools(cfg.tools, app.settings),
        system_prompt=app.prompt_text(cfg.prompt) + "\n\n" + app.market_brief(),
        name=name,
    )


def recursion_limit(cfg: AgentConfig) -> int:
    """LangGraph counts model + tool nodes separately: two steps per iteration."""
    return cfg.max_iterations * 2 + 1
