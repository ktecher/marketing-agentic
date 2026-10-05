"""competitor_research agent: registration, prompt, tools, one scripted ReAct run. Offline unless RUN_LIVE."""

import json
import os
import re

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

import agents.base
from agents.base import recursion_limit
from agents.registry import available_agents, build_agent
from core.config import load_config
from core.trace import stream_steps
from tools.mock import MOCK_DIR

SECTIONS = ["Snapshot", "Offering", "Positioning", "Pricing signals", "Recent activity",
            "Strengths", "Weaknesses", "Opportunities for us", "Sources"]


def _fixtures_ok() -> bool:
    try:
        for name in ("search.json", "pages.json"):
            json.loads((MOCK_DIR / name).read_text(encoding="utf-8"))
        return True
    except (OSError, ValueError):
        return False


class _FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):  # create_agent binds tools; the script ignores them
        return self


def test_registered_and_builds_with_configured_tools():
    app = load_config()
    assert "competitor_research" in available_agents()
    graph = build_agent("competitor_research", app)
    bound = graph.nodes["tools"].bound.tools_by_name
    assert sorted(bound) == sorted(app.agents["competitor_research"].tools)


def test_prompt_loads_and_names_every_section():
    app = load_config()
    text = app.prompt_text(app.agents["competitor_research"].prompt)
    for section in SECTIONS:
        assert f"## {section}" in text, section
    assert "_Based on sample data (mock mode)._" in text
    assert "TOOL_ERROR" in text


@pytest.mark.skipif(not _fixtures_ok(), reason="data/mock fixtures missing or not valid JSON yet")
def test_scripted_run_through_real_agent_in_mock_mode(monkeypatch):
    final = "# Competitor report: Kiraleaf\n_Based on sample data (mock mode)._\n\n## Snapshot\nok [1]"

    def fake_model(spec):  # fresh script per build: one tool call, then the final answer
        return _FakeModel(messages=iter([
            AIMessage("", tool_calls=[{"name": "web_search", "args": {"query": "Kiraleaf pricing"}, "id": "c1"}]),
            AIMessage(final),
        ]))

    monkeypatch.setattr(agents.base, "get_model", fake_model)
    app = load_config()
    app.settings.mock_mode = True
    cfg = {"recursion_limit": recursion_limit(app.agents["competitor_research"])}
    request = {"messages": [("user", "Research Kiraleaf")]}

    msgs = build_agent("competitor_research", app).invoke(request, cfg)["messages"]
    [tool_msg] = [m for m in msgs if isinstance(m, ToolMessage)]
    assert tool_msg.name == "web_search" and tool_msg.content.startswith("[mock data]")

    steps = list(stream_steps(build_agent("competitor_research", app), request, cfg,
                              default_agent="competitor_research"))
    assert [s.kind for s in steps] == ["tool_call", "tool_result", "agent_answer"]
    assert steps[-1].detail == final


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("RUN_LIVE"), reason="set RUN_LIVE=1")
def test_live_mock_mode_report():
    app = load_config()
    app.settings.mock_mode = True
    graph = build_agent("competitor_research", app)
    result = graph.invoke({"messages": [("user", "Research our competitor Kiraleaf")]},
                          {"recursion_limit": recursion_limit(app.agents["competitor_research"])})
    msgs = result["messages"]
    assert any(isinstance(m, ToolMessage) and str(m.content).startswith("[mock data]") for m in msgs)
    report = msgs[-1].text
    for section in SECTIONS:
        assert re.search(rf"^#+\s*{re.escape(section)}", report, re.M), section
    assert re.search(r"https?://[^\s)\]]+\.example\b", report)
    assert "_Based on sample data (mock mode)._" in report
