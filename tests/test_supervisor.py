"""Phase 6: supervisor routing over the registry. Offline: scripted router, fake specialists."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import START, END, MessagesState, StateGraph

from core import supervisor
from core.config import load_config


class ScriptedRouter:
    """Stands in for model.with_structured_output(Decision): returns queued decisions in order."""

    def __init__(self, model, decisions):
        self.model, self.decisions, self.prompts = model, list(decisions), []

    def invoke(self, messages):
        self.prompts.append(messages)
        return self.model(**self.decisions.pop(0))


def _fake_agent(name, reply=None, error=None):
    def node(state):
        if error:
            raise error
        task = state["messages"][-1].content
        return {"messages": [AIMessage(reply or f"{name} did: {task.splitlines()[0]}", name=name)]}

    g = StateGraph(MessagesState)
    g.add_node("model", node)
    g.add_edge(START, "model")
    g.add_edge("model", END)
    return g.compile()


@pytest.fixture
def wire(monkeypatch):
    """Patch the supervisor's model + agents; returns a function that runs a request."""

    def _wire(decisions, agents=None, synth="SYNTHESISED"):
        app = load_config()
        agents = agents or {n: _fake_agent(n) for n in app.enabled_agents}
        monkeypatch.setattr(supervisor, "build_enabled_agents", lambda app: agents)
        router = {}

        class Model:
            def with_structured_output(self, schema):
                router["r"] = ScriptedRouter(schema, decisions)
                return router["r"]

            def invoke(self, messages):
                router["synth_input"] = messages[-1].content
                if isinstance(synth, Exception):
                    raise synth
                return AIMessage(synth)

        monkeypatch.setattr(supervisor, "get_model", lambda spec: Model())
        steps = list(supervisor.run(app, "REQ"))
        return steps, router

    return _wire


def test_single_agent_answer_is_returned_verbatim(wire):
    steps, router = wire([
        {"reasoning": "competitor question", "action": "delegate", "agent": "competitor_research", "task": "Research Kiraleaf"},
        {"reasoning": "done", "action": "finish"},
    ])
    assert [s.kind for s in steps] == ["route", "agent_answer", "route", "final"]
    assert steps[-1].detail == "competitor_research did: Research Kiraleaf"
    assert "synth_input" not in router  # no extra LLM call for a single specialist
    roster = router["r"].prompts[0][0].content
    assert "**trend_monitoring**" in roster and "{agents}" not in roster


def test_multi_step_passes_context_and_synthesises(wire):
    seen = {}

    def content_agent(state):
        seen["prompt"] = state["messages"][-1].content
        return {"messages": [AIMessage("3 posts")]}

    g = StateGraph(MessagesState)
    g.add_node("model", content_agent)
    g.add_edge(START, "model")
    g.add_edge("model", END)
    agents = {"competitor_research": _fake_agent("competitor_research", reply="KIRA REPORT"), "content_drafting": g.compile()}

    steps, router = wire([
        {"reasoning": "research first", "action": "delegate", "agent": "competitor_research", "task": "Research Kiraleaf"},
        {"reasoning": "now draft", "action": "delegate", "agent": "content_drafting", "task": "3 posts vs Kiraleaf"},
        {"reasoning": "all served", "action": "finish"},
    ], agents=agents)

    assert "KIRA REPORT" in seen["prompt"] and seen["prompt"].startswith("3 posts vs Kiraleaf")
    assert steps[-1].detail == "SYNTHESISED"
    assert "KIRA REPORT" in router["synth_input"] and "3 posts" in router["synth_input"]
    assert "KIRA REPORT" in router["r"].prompts[1][1].content  # router sees completed steps


def test_failing_agent_is_reported_and_run_continues(wire):
    agents = {"competitor_research": _fake_agent("competitor_research", error=ConnectionError("boom")),
              "trend_monitoring": _fake_agent("trend_monitoring")}
    steps, router = wire([
        {"reasoning": "r", "action": "delegate", "agent": "competitor_research", "task": "Research X"},
        {"reasoning": "fallback", "action": "delegate", "agent": "trend_monitoring", "task": "Mentions of X"},
        {"reasoning": "done", "action": "finish"},
    ], agents=agents)
    errors = [s for s in steps if s.kind == "error"]
    assert errors and "AGENT_ERROR (competitor_research): ConnectionError: boom" in errors[0].detail
    assert steps[-1].kind == "final"
    assert "FAILED" in router["synth_input"]


def test_no_agent_needed_replies_directly(wire):
    steps, _ = wire([{"reasoning": "greeting", "action": "finish", "reply": "Hi! I can research competitors..."}])
    assert [s.kind for s in steps] == ["route", "final"]
    assert steps[-1].detail.startswith("Hi!")


def test_step_limit_stops_a_looping_router(wire, monkeypatch):
    app = load_config()
    loop = [{"reasoning": "again", "action": "delegate", "agent": "trend_monitoring", "task": "t"}] * 20
    steps, _ = wire(loop)
    assert sum(s.kind == "agent_answer" for s in steps) == app.supervisor.max_steps
    assert steps[-1].kind == "final"


def test_roster_comes_from_config_not_code(wire, monkeypatch, tmp_path):
    """A disabled agent disappears from routing without touching supervisor code."""
    import shutil, yaml
    from core import config as cfgmod

    shutil.copytree(cfgmod.CONFIG_DIR, tmp_path / "config")
    path = tmp_path / "config" / "agents.yaml"
    data = yaml.safe_load(path.read_text())
    data["agents"]["trend_monitoring"]["enabled"] = False
    path.write_text(yaml.safe_dump(data))
    app = cfgmod.load_config(tmp_path / "config")
    seen = {}

    class Model:
        def with_structured_output(self, schema):
            seen["schema"] = schema
            return self

    monkeypatch.setattr(supervisor, "get_model", lambda spec: Model())
    supervisor.build_supervisor(app)  # the REAL builder: registry + config decide the roster
    choices = set(seen["schema"].model_fields["agent"].annotation.__args__[0].__args__)
    assert choices == {"competitor_research", "content_drafting"}


def test_misconfigured_llm_becomes_error_step(monkeypatch):
    from core.config import ModelSpec
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = load_config()
    app.settings.llm = ModelSpec(provider="anthropic", model="claude-sonnet-5-5")
    [step] = list(supervisor.run(app, "hi"))
    assert step.kind == "error" and "ANTHROPIC_API_KEY" in step.detail


def test_router_failure_keeps_finished_work(wire):
    decisions = [{"reasoning": "r", "action": "delegate", "agent": "competitor_research", "task": "Research X"}]
    # second route call raises: ScriptedRouter pops from an empty list
    steps, _ = wire(decisions)
    assert any(s.kind == "error" and "Routing failed" in s.title for s in steps)
    assert steps[-1].kind == "final" and steps[-1].detail == "competitor_research did: Research X"


def test_synthesis_failure_returns_outputs_as_delivered(wire, monkeypatch):
    steps, router = wire([
        {"reasoning": "a", "action": "delegate", "agent": "competitor_research", "task": "Research X"},
        {"reasoning": "b", "action": "delegate", "agent": "content_drafting", "task": "Posts"},
        {"reasoning": "c", "action": "finish"},
    ], synth=RuntimeError("synth down"))
    assert "Could not combine" in steps[-1].detail
    assert "competitor_research did: Research X" in steps[-1].detail and "content_drafting did: Posts" in steps[-1].detail
