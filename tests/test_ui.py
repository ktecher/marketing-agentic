"""Streamlit demo UI. Offline: core.supervisor.run is replaced with a scripted trace."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core import supervisor
from core.trace import Step

APP = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")
FINAL = "## Three posts for Aurelle Skin\n\nPost one: calm skin, fewer steps."

SCRIPT = [
    Step("route", "supervisor", "Delegating to competitor_research: research Kiraleaf",
         "Kiraleaf is a named competitor, so research comes first.",
         data={"agent": "competitor_research", "task": "Research Kiraleaf", "reasoning": "Named competitor."}),
    Step("thought", "competitor_research", "Searching first."),
    Step("tool_call", "competitor_research", "web_search(query='Kiraleaf')",
         data={"tool": "web_search", "args": {"query": "Kiraleaf"}}),
    Step("tool_result", "competitor_research", "web_search: [mock data] Kiraleaf sells...",
         "[mock data] Kiraleaf sells serums.", data={"tool": "web_search"}),
    Step("tool_error", "competitor_research", "read_page: TOOL_ERROR",
         "TOOL_ERROR (read_page): TimeoutError: timed out", data={"tool": "read_page"}),
    Step("agent_answer", "competitor_research", "Kiraleaf report", "# Kiraleaf report\n\nSerums."),
    Step("route", "supervisor", "Ready to answer", "Research is done.", data={"reasoning": "Done."}),
    Step("final", "supervisor", "Final answer", FINAL),
]


@pytest.fixture
def app_test(monkeypatch):
    calls = []

    def fake_run(app, request, graph=None):
        calls.append((app, request))
        yield from SCRIPT

    monkeypatch.setattr(supervisor, "run", fake_run)
    at = AppTest.from_file(APP, default_timeout=30)
    at.calls = calls
    return at


def _markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_renders_with_settings(app_test):
    at = app_test.run()
    assert not at.exception
    assert at.toggle(key="mock") is not None
    assert at.selectbox(key="provider").value in {"claude_code", "anthropic", "openai", "ollama", "openai_compatible"}
    assert at.button(key="brief").label == "Brief the team"


def test_example_chip_fills_the_box(app_test):
    at = app_test.run()
    at.toggle(key="mock").set_value(True).run()
    at.button(key="ex_0").click().run()
    assert not at.exception
    assert "Kiraleaf" in at.text_area(key="request").value


def test_scripted_run_shows_final_answer(app_test):
    at = app_test.run()
    at.text_area(key="request").input("Research Kiraleaf and draft posts.")
    at.button(key="brief").click().run()
    assert not at.exception
    assert "Three posts for Aurelle Skin" in _markdown(at)
    app, request = at.calls[0]
    assert request == "Research Kiraleaf and draft posts."
    # Settings from the sidebar reach the run config.
    assert app.settings.llm.provider == at.selectbox(key="provider").value
    # The result survives a rerun (e.g. after clicking download).
    at.run()
    assert "Three posts for Aurelle Skin" in _markdown(at)
    assert len(at.calls) == 1


def test_session_key_reaches_spec_not_env(app_test, monkeypatch):
    import os

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = app_test.run()
    at.selectbox(key="provider").set_value("anthropic").run()
    at.text_input(key="key_anthropic").input("sk-test-123")
    at.text_area(key="request").input("Hi")
    at.button(key="brief").click().run()
    assert not at.exception
    app, _ = at.calls[0]
    assert app.settings.llm.api_key.get_secret_value() == "sk-test-123"
    # .env may define the variable (even as blank); what matters is the session key never lands there.
    assert all("sk-test-123" not in v for v in os.environ.values())


def test_run_failure_renders_as_error(monkeypatch):
    def boom(app, request, graph=None):
        raise RuntimeError("provider 'anthropic' needs ANTHROPIC_API_KEY")
        yield  # pragma: no cover

    monkeypatch.setattr(supervisor, "run", boom)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area(key="request").input("Hi")
    at.button(key="brief").click().run()
    assert not at.exception
    rendered = str([e.proto for e in at.main] + [e.proto for e in at.get("html")])
    assert "No answer this time" in rendered and "ANTHROPIC_API_KEY" in rendered


def test_team_strip_shows_parallel_specialists():
    from core.trace import Step

    # Import only the pure helper: executing the whole script needs a Streamlit runtime.
    src = open(APP).read()
    ns = {}
    exec(compile(src[src.index("def team_status("):src.index("STATUS_WORDS = {")], APP, "exec"),
         {"Step": Step, "TEAM": ["supervisor", "competitor_research", "content_drafting", "trend_monitoring"]}, ns)
    team_status = ns["team_status"]
    route = lambda a: Step("route", "supervisor", "", data={"agent": a})
    steps = [route("trend_monitoring"), route("trend_monitoring"), route("competitor_research")]
    st = team_status(steps, running=True)
    assert st["trend_monitoring"] == st["competitor_research"] == "active" and st["supervisor"] == "idle"
    st = team_status(steps + [Step("agent_answer", "trend_monitoring", "")], running=True)
    assert st["trend_monitoring"] == "active"  # its second task is still running
    st = team_status(steps + [Step("agent_answer", "trend_monitoring", ""), Step("agent_answer", "trend_monitoring", ""),
                              Step("agent_answer", "competitor_research", "")], running=True)
    assert st["trend_monitoring"] == st["competitor_research"] == "done" and st["supervisor"] == "active"
