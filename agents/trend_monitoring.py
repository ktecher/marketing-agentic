"""trend_monitoring: recent mentions, sentiment, themes and trends for a brand or topic."""

from agents.base import build_react_agent
from agents.registry import register_agent


@register_agent("trend_monitoring")
def build(name, cfg, app):
    return build_react_agent(name, cfg, app)
