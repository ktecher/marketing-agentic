"""content_drafting — drafts platform-ready marketing copy variations, briefly grounded by search."""

from agents.base import build_react_agent
from agents.registry import register_agent


@register_agent("content_drafting")
def build(name, cfg, app):
    return build_react_agent(name, cfg, app)
