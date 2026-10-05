"""trend_monitoring agent: wiring, prompt, one scripted run through the real compiled graph."""

import os

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import agents.base
from agents.base import recursion_limit
from agents.registry import available_agents, build_agent
from core.config import load_config
from core.trace import stream_steps
from tools import mock

SECTIONS = ["Overall sentiment", "Recurring themes", "Notable trends & signals", "Watch-outs",
            "Suggested actions", "Sources"]


class _FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


def _mock_app():
    app = load_config()
    app.settings.mock_mode = True
    return app


def test_registered_and_builds():
    app = load_config()
    assert "trend_monitoring" in available_agents()
    assert build_agent("trend_monitoring", app) is not None
    assert app.agents["trend_monitoring"].tools == ["web_search", "read_page", "current_date"]


def test_prompt_has_sections_and_rules():
    app = load_config()
    text = app.prompt_text(app.agents["trend_monitoring"].prompt)
    for s in SECTIONS:
        assert s in text, s
    assert "current_date" in text and "[mock data]" in text and "TOOL_ERROR" in text
    assert "_Based on sample data (mock mode)._" in text


def test_scripted_run_through_real_graph(monkeypatch):
    if not (mock.MOCK_DIR / "search.json").exists():  # fixtures are written by another phase; keep the test self-contained
        monkeypatch.setattr(mock, "_load", lambda name: [{"match": [], "results": [
            {"title": "Sample", "url": "https://sample.example/a", "snippet": "s"}]}])
    answer = "# Trend monitor: Aurelle Skin\n_Based on sample data (mock mode)._\n**Overall sentiment:** mixed"
    fake = _FakeModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "web_search", "id": "c1",
                                            "args": {"query": "Aurelle Skin", "kind": "news", "recency": "month"}}]),
        AIMessage(content=answer),
    ]))
    monkeypatch.setattr(agents.base, "get_model", lambda spec: fake)
    app = _mock_app()
    graph = build_agent("trend_monitoring", app)
    steps = list(stream_steps(graph, {"messages": [("user", "Aurelle Skin?")]},
                              {"recursion_limit": recursion_limit(app.agents["trend_monitoring"])},
                              default_agent="trend_monitoring"))
    kinds = [s.kind for s in steps]
    assert kinds == ["tool_call", "tool_result", "agent_answer"], [(s.kind, s.title) for s in steps]
    assert steps[0].data == {"tool": "web_search", "args": {"query": "Aurelle Skin", "kind": "news", "recency": "month"}}
    assert steps[1].detail.startswith("[mock data]")
    assert steps[2].detail == answer


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("RUN_LIVE"), reason="set RUN_LIVE=1")
def test_live_mock_run_has_sections():
    app = _mock_app()
    graph = build_agent("trend_monitoring", app)
    steps = list(stream_steps(graph, {"messages": [("user", "What are people saying about Aurelle Skin lately?")]},
                              {"recursion_limit": recursion_limit(app.agents["trend_monitoring"])},
                              default_agent="trend_monitoring"))
    answer = [s.detail for s in steps if s.kind == "agent_answer"][-1]
    for s in SECTIONS:
        assert s in answer, s
    assert "_Based on sample data (mock mode)._" in answer
