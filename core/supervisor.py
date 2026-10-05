"""Supervisor: hands a director's request to specialists in rounds, then answers.

Graph:  route ──► <agent nodes, in parallel> ──► route ──► ... ──► finish
         └──────────────────────────────────────────────────────►┘

- `route` asks the LLM for a structured Decision: the specialists to run NOW (independent tasks
  run side by side), or finish. Work that depends on another result waits for the next round,
  where it receives the earlier outputs as context.
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
from langgraph.types import Send
from pydantic import BaseModel, Field, create_model

from agents.base import recursion_limit
from agents.registry import build_enabled_agents
from core.config import AppConfig
from core.models import get_model
from core.trace import Step, stream_steps


class State(TypedDict, total=False):
    request: str
    history: str                                  # earlier briefs + answers in this conversation
    results: Annotated[list[dict], operator.add]  # [{agent, task, output, ok}]
    assignments: list[dict]                       # [{agent, task}] chosen by route for this round
    assignment: dict                              # the one task a runner received (via Send)
    round: int
    reply: str                                    # direct reply when no agent is needed
    final: str
    trace: Annotated[list[Step], operator.add]   # parallel specialists append in the same step


def _decision_model(agent_names: list[str]) -> type[BaseModel]:
    assignment = create_model(
        "Assignment",
        agent=(Literal[tuple(agent_names)], Field(description="specialist to run")),
        task=(str, Field(description="Self-contained instruction for that specialist")),
    )
    return create_model(
        "Decision",
        reasoning=(str, Field(description="One or two sentences: why these next steps (shown to the user).")),
        action=(Literal["delegate", "finish"], Field(description="delegate this round's work, or finish")),
        assignments=(list[assignment], Field(default_factory=list, description=(
            "Specialists to run NOW, side by side. Include every task that does not need another "
            "task's result; tasks that depend on a result go in a later round."))),
        reply=(str, Field("", description=(
            "Only when finishing with NO specialist used this turn: the full reply to the director. "
            "May answer from earlier answers in the conversation."))),
    )


def _results_block(results: list[dict]) -> str:
    if not results:
        return "None yet."
    return "\n\n".join(
        f"### Step {i}: {r['agent']} ({'ok' if r['ok'] else 'FAILED'})\nTask: {r['task']}\n\n{r['output']}"
        for i, r in enumerate(results, 1)
    )


_HISTORY_CHARS = 6000  # per earlier answer; ponytail: truncation, summarise if long threads matter


def _history_block(history: list[dict] | None) -> str:
    """Earlier turns of this conversation, oldest first, for follow-up questions."""
    if not history:
        return ""
    turns = []
    for i, h in enumerate(history, 1):
        answer = h["answer"] if len(h["answer"]) <= _HISTORY_CHARS else h["answer"][:_HISTORY_CHARS] + "\n[…truncated]"
        turns.append(f"### Earlier brief {i}\n{h['request']}\n\n#### Answer given\n{answer}")
    return "\n\n".join(turns)


def _with_history(text: str, hist: str, label: str) -> str:
    return f"## Earlier in this conversation ({label})\n{hist}\n\n{text}" if hist else text


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
    market = app.market_brief()
    route_prompt = app.prompt_text(cfg.prompt).replace("{agents}", roster) + "\n\n" + market
    synth_prompt = app.prompt_text(cfg.synthesis_prompt) + "\n\n" + market

    def route(state: State) -> State:
        done = state.get("results", [])
        rnd = state.get("round", 0) + 1
        if len(done) >= cfg.max_steps:
            return {"assignments": [], "trace": [Step("route", "supervisor", f"Step limit ({cfg.max_steps}) reached — finishing")]}
        try:
            d = decide.invoke([
                SystemMessage(route_prompt),
                HumanMessage(_with_history(
                    f"## Director's request (now)\n{state['request']}\n\n## Completed steps this turn\n{_results_block(done)}",
                    state.get("history", ""), "the director may be following up on it")),
            ])
        except Exception as e:  # keep what the specialists already produced
            return {"assignments": [], "reply": f"Routing failed ({type(e).__name__}: {e}).",
                    "trace": [Step("error", "supervisor", "Routing failed — finishing with what we have", str(e))]}
        work = [{"agent": a.agent, "task": a.task or state["request"]} for a in d.assignments]
        work = work[: cfg.max_steps - len(done)]
        if d.action == "delegate" and work:
            both = " in parallel" if len(work) > 1 else ""
            return {
                "assignments": work, "round": rnd,
                "trace": [Step("route", "supervisor", f"Delegating to {w['agent']}{both}: {w['task']}", d.reasoning,
                               data={**w, "reasoning": d.reasoning, "round": rnd, "parallel": len(work)})
                          for w in work],
            }
        return {"assignments": [], "reply": d.reply,
                "trace": [Step("route", "supervisor", "Ready to answer", d.reasoning, data={"reasoning": d.reasoning})]}

    def make_runner(name: str):
        graph = agents[name].with_config(recursion_limit=recursion_limit(app.agents[name]))

        def run(state: State) -> State:
            task = state["assignment"]["task"]
            prompt = task
            if state.get("history"):
                prompt += "\n\n## Earlier in this conversation (build on it; don't re-research what's here)\n" + state["history"]
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
                    HumanMessage(_with_history(
                        f"## Director's request (now)\n{state['request']}\n\n## Specialist outputs\n{_results_block(results)}",
                        state.get("history", ""), "for reference; answer the request made now")),
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

    def dispatch(state: State):
        work = state.get("assignments") or []
        if not work:
            return "finish"
        # Send runs each assignment as its own branch in the same step: the specialists work in
        # parallel, and route runs again once all of them have finished.
        return [Send(w["agent"], {"request": state["request"], "history": state.get("history", ""),
                                  "results": state.get("results", []), "assignment": w})
                for w in work]

    g.add_conditional_edges("route", dispatch, [*names, "finish"])
    g.add_edge("finish", END)
    return g.compile()


def run(app: AppConfig, request: str, graph=None, history: list[dict] | None = None) -> Iterator[Step]:
    """Stream the trace for one request. The last Step has kind 'final' (or 'error').
    `history` is the earlier turns of the conversation, [{request, answer}], for follow-ups."""
    try:
        graph = graph or build_supervisor(app)  # inside try: a missing key/URL becomes an error Step
        yield from stream_steps(graph, {"request": request, "results": [], "history": _history_block(history)},
                                {"recursion_limit": app.supervisor.max_steps * 2 + 4},
                                default_agent="supervisor")
    except Exception as e:
        yield Step("error", "supervisor", f"Run failed: {type(e).__name__}", str(e))
