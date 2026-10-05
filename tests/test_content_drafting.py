"""content_drafting specialist: wiring offline, plus one scripted run through the real compiled graph."""

import json
import os
import re

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import agents.base
from agents.base import recursion_limit
from agents.registry import available_agents, build_agent
from core.config import load_config
from core.trace import stream_steps
from tools import mock

NAME = "content_drafting"
REQUEST = "Write 3 Instagram captions for Aurelle Skin launching the Barrier Reset Serum, warm and expert tone"


class FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_registered_and_builds():
    app = load_config()
    assert NAME in available_agents()
    assert build_agent(NAME, app) is not None
    prompt = app.prompt_text(app.agents[NAME].prompt)
    for rule in ("[mock data]", "Grounded in sample data (mock mode)", "Never invent", "language"):
        assert rule in prompt
    assert app.agents[NAME].tools == ["web_search", "read_page"]


def _fixtures_ok() -> bool:
    try:
        json.loads((mock.MOCK_DIR / "search.json").read_text())
        json.loads((mock.MOCK_DIR / "pages.json").read_text())
        return True
    except (OSError, ValueError):
        return False


@pytest.fixture
def mock_world(tmp_path, monkeypatch):
    """Real fixtures when they exist; otherwise a one-entry stand-in so the test stays offline."""
    if not _fixtures_ok():
        (tmp_path / "search.json").write_text(json.dumps([{"match": [], "results": [
            {"title": "Aurelle Skin", "url": "https://aurelle.example/serum", "snippet": "Barrier Reset Serum"}]}]))
        (tmp_path / "pages.json").write_text("{}")
        monkeypatch.setattr(mock, "MOCK_DIR", tmp_path)
    mock._load.cache_clear()
    yield
    mock._load.cache_clear()


def test_scripted_run_through_real_graph(monkeypatch, mock_world):
    answer = "**Brief:** Aurelle Skin · Barrier Reset Serum · Instagram\n_Grounded in sample data (mock mode)._"
    fake = FakeToolModel(messages=iter([
        AIMessage("", tool_calls=[{"name": "web_search", "args": {"query": "Aurelle Barrier Reset Serum"}, "id": "c1"}]),
        AIMessage(answer),
    ]))
    monkeypatch.setattr(agents.base, "get_model", lambda spec: fake)
    app = load_config()
    app.settings.mock_mode = True

    graph = build_agent(NAME, app)
    steps = list(stream_steps(graph, {"messages": [("user", REQUEST)]},
                              {"recursion_limit": recursion_limit(app.agents[NAME])}, default_agent=NAME))

    kinds = [s.kind for s in steps]
    assert kinds == ["tool_call", "tool_result", "agent_answer"], kinds
    assert steps[0].data == {"tool": "web_search", "args": {"query": "Aurelle Barrier Reset Serum"}}
    assert steps[1].detail.startswith(mock.MOCK_LABEL)
    assert steps[2].detail == answer
    assert all(s.agent == NAME for s in steps)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("RUN_LIVE"), reason="set RUN_LIVE=1")
def test_live_mock_run_gives_three_variations():
    app = load_config()
    app.settings.mock_mode = True
    steps = list(stream_steps(build_agent(NAME, app), {"messages": [("user", REQUEST)]},
                              {"recursion_limit": recursion_limit(app.agents[NAME])}, default_agent=NAME))
    answer = [s.detail for s in steps if s.kind == "agent_answer"][-1]
    assert any(s.kind == "tool_result" and s.detail.startswith(mock.MOCK_LABEL) for s in steps)
    assert len(re.findall(r"^###\s*\d", answer, re.M)) >= 3, answer
    assert "Grounded in sample data (mock mode)" in answer
