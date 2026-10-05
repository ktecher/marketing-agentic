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
        {"reasoning": "competitor question", "action": "delegate", "assignments": [{"agent": "competitor_research", "task": "Research Kiraleaf"}]},
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
        {"reasoning": "research first", "action": "delegate", "assignments": [{"agent": "competitor_research", "task": "Research Kiraleaf"}]},
        {"reasoning": "now draft", "action": "delegate", "assignments": [{"agent": "content_drafting", "task": "3 posts vs Kiraleaf"}]},
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
        {"reasoning": "r", "action": "delegate", "assignments": [{"agent": "competitor_research", "task": "Research X"}]},
        {"reasoning": "fallback", "action": "delegate", "assignments": [{"agent": "trend_monitoring", "task": "Mentions of X"}]},
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
    loop = [{"reasoning": "again", "action": "delegate", "assignments": [{"agent": "trend_monitoring", "task": "t"}]}] * 20
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
    assignment = seen["schema"].model_fields["assignments"].annotation.__args__[0]
    choices = set(assignment.model_fields["agent"].annotation.__args__)
    assert choices == {"competitor_research", "content_drafting"}


def test_misconfigured_llm_becomes_error_step(monkeypatch):
    from core.config import ModelSpec
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = load_config()
    app.settings.llm = ModelSpec(provider="anthropic", model="claude-sonnet-5-5")
    [step] = list(supervisor.run(app, "hi"))
    assert step.kind == "error" and "ANTHROPIC_API_KEY" in step.detail


def test_router_failure_keeps_finished_work(wire):
    decisions = [{"reasoning": "r", "action": "delegate", "assignments": [{"agent": "competitor_research", "task": "Research X"}]}]
    # second route call raises: ScriptedRouter pops from an empty list
    steps, _ = wire(decisions)
    assert any(s.kind == "error" and "Routing failed" in s.title for s in steps)
    assert steps[-1].kind == "final" and steps[-1].detail == "competitor_research did: Research X"


def test_synthesis_failure_returns_outputs_as_delivered(wire, monkeypatch):
    steps, router = wire([
        {"reasoning": "a", "action": "delegate", "assignments": [{"agent": "competitor_research", "task": "Research X"}]},
        {"reasoning": "b", "action": "delegate", "assignments": [{"agent": "content_drafting", "task": "Posts"}]},
        {"reasoning": "c", "action": "finish"},
    ], synth=RuntimeError("synth down"))
    assert "Could not combine" in steps[-1].detail
    assert "competitor_research did: Research X" in steps[-1].detail and "content_drafting did: Posts" in steps[-1].detail


def test_independent_tasks_run_in_parallel_in_one_round(wire):
    """Two specialists in one round: both start before either finishes, and the next round sees both."""
    import threading, time
    started, overlap = [], threading.Event()

    def agent(name):
        def node(state):
            started.append(name)
            if len(started) == 2:
                overlap.set()
            assert overlap.wait(2), f"{name} ran alone: the round was not parallel"
            time.sleep(0.05)  # slow on purpose: parallelism only shows if the calls overlap
            return {"messages": [AIMessage(f"{name.upper()} OUTPUT")]}
        g = StateGraph(MessagesState)
        g.add_node("model", node)
        g.add_edge(START, "model")
        g.add_edge("model", END)
        return g.compile()

    agents = {"trend_monitoring": agent("trend_monitoring"), "competitor_research": agent("competitor_research")}
    steps, router = wire([
        {"reasoning": "independent", "action": "delegate", "assignments": [
            {"agent": "trend_monitoring", "task": "Mentions of A"},
            {"agent": "competitor_research", "task": "Research B"}]},
        {"reasoning": "done", "action": "finish"},
    ], agents=agents)

    assert sorted(started) == ["competitor_research", "trend_monitoring"]
    routes = [s for s in steps if s.kind == "route" and s.data.get("agent")]
    assert [r.data["parallel"] for r in routes] == [2, 2] and {r.data["round"] for r in routes} == {1}
    second_round_input = router["r"].prompts[1][1].content
    assert "TREND_MONITORING OUTPUT" in second_round_input and "COMPETITOR_RESEARCH OUTPUT" in second_round_input
    assert steps[-1].detail == "SYNTHESISED"


def test_same_specialist_twice_in_one_round(wire):
    steps, router = wire([
        {"reasoning": "two brands", "action": "delegate", "assignments": [
            {"agent": "trend_monitoring", "task": "Mentions of A"},
            {"agent": "trend_monitoring", "task": "Mentions of B"}]},
        {"reasoning": "done", "action": "finish"},
    ])
    assert "Mentions of A" in router["synth_input"] and "Mentions of B" in router["synth_input"]


def test_parallel_round_is_capped_by_step_limit(wire):
    app = load_config()
    many = [{"agent": "trend_monitoring", "task": f"t{i}"} for i in range(app.supervisor.max_steps + 3)]
    steps, _ = wire([{"reasoning": "lots", "action": "delegate", "assignments": many},
                     {"reasoning": "done", "action": "finish"}])
    assert sum(s.kind == "route" and bool(s.data.get("agent")) for s in steps) == app.supervisor.max_steps


def test_follow_up_sees_earlier_turns_and_can_answer_directly(wire, monkeypatch):
    """A follow-up answered from earlier results: no specialist runs, the router saw the history."""
    from core import supervisor as sup
    app = load_config()
    seen = {}

    class Router:
        def __init__(self, schema): self.schema = schema
        def invoke(self, messages):
            seen["prompt"] = messages[1].content
            return self.schema(reasoning="already covered", action="finish", reply="Xcite was cheapest (KWD 449.900).")

    class Model:
        def with_structured_output(self, schema): return Router(schema)

    ran = []
    monkeypatch.setattr(sup, "get_model", lambda spec: Model())
    monkeypatch.setattr(sup, "build_enabled_agents", lambda app: {n: _fake_agent(n) for n in app.enabled_agents})
    history = [{"request": "fold 8", "answer": "Xcite lists it at KWD 449.900; Best at KWD 459."}]
    steps = list(sup.run(app, "which retailer was cheapest?", history=history))
    assert steps[-1].detail == "Xcite was cheapest (KWD 449.900)."
    assert not [s for s in steps if s.kind == "agent_answer"]
    assert "Earlier brief 1\nfold 8" in seen["prompt"] and "KWD 449.900" in seen["prompt"]
    assert seen["prompt"].index("Earlier brief") < seen["prompt"].index("which retailer was cheapest?")


def test_follow_up_specialist_receives_earlier_findings(wire):
    seen = {}

    def content(state):
        seen["prompt"] = state["messages"][-1].content
        return {"messages": [AIMessage("Snapchat versions")]}

    g = StateGraph(MessagesState)
    g.add_node("model", content)
    g.add_edge(START, "model")
    g.add_edge("model", END)
    from core import supervisor as sup
    app = load_config()
    decisions = [{"reasoning": "new drafts", "action": "delegate",
                  "assignments": [{"agent": "content_drafting", "task": "Rewrite the 3 posts for Snapchat"}]},
                 {"reasoning": "done", "action": "finish"}]

    class Model:
        def with_structured_output(self, schema): return ScriptedRouter(schema, decisions)
        def invoke(self, m): return AIMessage("unused")

    import pytest
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(sup, "get_model", lambda spec: Model())
        mp.setattr(sup, "build_enabled_agents", lambda app: {"content_drafting": g.compile()})
        steps = list(sup.run(app, "make those posts for Snapchat",
                             history=[{"request": "3 Instagram posts for Aurelle", "answer": "POST ONE / POST TWO / POST THREE"}]))
    finally:
        mp.undo()
    assert steps[-1].detail == "Snapchat versions"
    assert "POST ONE / POST TWO / POST THREE" in seen["prompt"] and seen["prompt"].startswith("Rewrite the 3 posts for Snapchat")


def test_history_is_truncated_per_answer():
    from core.supervisor import _HISTORY_CHARS, _history_block
    block = _history_block([{"request": "r", "answer": "x" * (_HISTORY_CHARS + 500)}])
    assert block.count("x") == _HISTORY_CHARS and block.endswith("[…truncated]")
