"""Supervisor: routes a director's request to specialists, one step at a time, then answers.

Graph:  route ──► <agent node> ──► route ──► ... ──► finish
         └──────────────────────────────────────────►┘

- `route` asks the LLM for a structured Decision (delegate to an agent with a task, or finish).
  The agent list and descriptions come from the registry + agents.yaml, never from this file.
- Each enabled agent is its own node, named after the agent, so the trace can attribute every
  tool call to the right specialist (LangGraph namespaces subgraph events by node name).
- `finish` returns a lone specialist's output verbatim, or synthesises several into one answer.
"""

import operator
from typing import Annotated, Iterator, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, create_model

from agents.base import recursion_limit
from agents.registry import build_enabled_agents
from core.config import AppConfig
from core.models import get_model
from core.trace import Step, stream_steps


class State(TypedDict, total=False):
    request: str
    results: Annotated[list[dict], operator.add]  # [{agent, task, output, ok}]
    next: dict | None                             # {agent, task} chosen by route
    reply: str                                    # direct reply when no agent is needed
    final: str
    trace: list[Step]


def _decision_model(agent_names: list[str]) -> type[BaseModel]:
    return create_model(
        "Decision",
        reasoning=(str, Field(description="One or two sentences: why this next step (shown to the user).")),
        action=(Literal["delegate", "finish"], Field(description="delegate to a specialist, or finish")),
        agent=(Literal[tuple(agent_names)] | None, Field(None, description="specialist to delegate to")),
        task=(str, Field("", description="Self-contained instruction for that specialist")),
        reply=(str, Field("", description="Only when finishing with NO specialist used: the reply to the user")),
    )


def _results_block(results: list[dict]) -> str:
    if not results:
        return "None yet."
    return "\n\n".join(
        f"### Step {i}: {r['agent']} ({'ok' if r['ok'] else 'FAILED'})\nTask: {r['task']}\n\n{r['output']}"
        for i, r in enumerate(results, 1)
    )


def _last_ai_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not m.tool_calls:
            return m.content if isinstance(m.content, str) else m.text
    return ""


def build_supervisor(app: AppConfig):
    agents = build_enabled_agents(app)
    if not agents:
        raise RuntimeError("no enabled agents in config/agents.yaml")
    names = list(agents)
    cfg = app.supervisor
    model = get_model(app.model_for(cfg.model))
    decide = model.with_structured_output(_decision_model(names))
    roster = "\n".join(f"- **{n}**: {app.agents[n].description}" for n in names)
    route_prompt = app.prompt_text(cfg.prompt).replace("{agents}", roster)
    synth_prompt = app.prompt_text(cfg.synthesis_prompt)

    def route(state: State) -> State:
        done = state.get("results", [])
        if len(done) >= cfg.max_steps:
            return {"next": None, "trace": [Step("route", "supervisor", f"Step limit ({cfg.max_steps}) reached — finishing")]}
        try:
            d = decide.invoke([
                SystemMessage(route_prompt),
                HumanMessage(f"## Director's request\n{state['request']}\n\n## Completed steps\n{_results_block(done)}"),
            ])
        except Exception as e:  # keep what the specialists already produced
            return {"next": None, "reply": f"Routing failed ({type(e).__name__}: {e}).",
                    "trace": [Step("error", "supervisor", "Routing failed — finishing with what we have", str(e))]}
        if d.action == "delegate" and d.agent:
            return {
                "next": {"agent": d.agent, "task": d.task or state["request"]},
                "trace": [Step("route", "supervisor", f"Delegating to {d.agent}: {d.task}", d.reasoning,
                               data={"agent": d.agent, "task": d.task, "reasoning": d.reasoning})],
            }
        return {"next": None, "reply": d.reply,
                "trace": [Step("route", "supervisor", "Ready to answer", d.reasoning, data={"reasoning": d.reasoning})]}

    def make_runner(name: str):
        graph = agents[name].with_config(recursion_limit=recursion_limit(app.agents[name]))

        def run(state: State) -> State:
            task = state["next"]["task"]
            prompt = task
            if state.get("results"):
                prompt += "\n\n## Context from previous steps (use it; don't re-research it)\n" + _results_block(state["results"])
            try:
                output, ok = _last_ai_text(graph.invoke({"messages": [HumanMessage(prompt)]})["messages"]), True
                if not output:
                    output, ok = "The agent finished without a final answer.", False
            except GraphRecursionError:
                output, ok = f"{name} ran out of steps before finishing.", False
            except Exception as e:  # one failing specialist must not end the run
                output, ok = f"AGENT_ERROR ({name}): {type(e).__name__}: {e}", False
            # Success is already in the trace (the subgraph's final message streams through);
            # only failures need their own step.
            trace = [] if ok else [Step("error", name, f"{name} failed", output)]
            return {"results": [{"agent": name, "task": task, "output": output, "ok": ok}], "trace": trace}

        return run

    def finish(state: State) -> State:
        results = state.get("results", [])
        ok = [r for r in results if r["ok"]]
        if not results:
            final = state.get("reply") or "I couldn't match this request to a specialist."
        elif len(results) == 1 and ok:
            final = ok[0]["output"]  # one specialist answered: hand it over unchanged
        else:
            try:
                final = model.invoke([
                    SystemMessage(synth_prompt),
                    HumanMessage(f"## Director's request\n{state['request']}\n\n## Specialist outputs\n{_results_block(results)}"),
                ]).content
            except Exception as e:  # never lose finished work to a failed summary
                final = f"_Could not combine the results ({type(e).__name__}); here they are as delivered._\n\n" + \
                    "\n\n---\n\n".join(r["output"] for r in ok)
        return {"final": final, "trace": [Step("final", "supervisor", "Final answer", final)]}

    g = StateGraph(State)
    g.add_node("route", route)
    g.add_node("finish", finish)
    for n in names:
        g.add_node(n, make_runner(n))
        g.add_edge(n, "route")
    g.add_edge(START, "route")
    g.add_conditional_edges("route", lambda s: s["next"]["agent"] if s.get("next") else "finish", [*names, "finish"])
    g.add_edge("finish", END)
    return g.compile()


def run(app: AppConfig, request: str, graph=None) -> Iterator[Step]:
    """Stream the trace for one request. The last Step has kind 'final' (or 'error')."""
    try:
        graph = graph or build_supervisor(app)  # inside try: a missing key/URL becomes an error Step
        yield from stream_steps(graph, {"request": request, "results": []},
                                {"recursion_limit": app.supervisor.max_steps * 2 + 4},
                                default_agent="supervisor")
    except Exception as e:
        yield Step("error", "supervisor", f"Run failed: {type(e).__name__}", str(e))
