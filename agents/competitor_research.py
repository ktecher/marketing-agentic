"""competitor_research — researches one competitor on the web and writes a cited intelligence report."""

from agents.base import build_react_agent
from agents.registry import register_agent


@register_agent("competitor_research")
def build(name, cfg, app):
    return build_react_agent(name, cfg, app)
