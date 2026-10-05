"""Reasoning trace: turns LangGraph stream updates into flat steps the CLI and UI both render."""

from dataclasses import dataclass, field
from typing import Iterator, Literal

from langchain_core.messages import AIMessage, ToolMessage

Kind = Literal["route", "thought", "tool_call", "tool_result", "tool_error", "agent_answer", "final", "error"]


@dataclass
class Step:
    kind: Kind
    agent: str            # "supervisor" or a specialist name
    title: str            # one line for the trace header
    detail: str = ""      # full text, shown when expanded
    data: dict = field(default_factory=dict)


def _agent_from_namespace(ns: tuple[str, ...], default: str) -> str:
    # Subgraph namespaces look like ("competitor_research:<task id>", ...); the first part names the agent.
    return ns[0].split(":")[0] if ns else default


def _preview(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def message_steps(messages, agent: str) -> Iterator[Step]:
    for m in messages:
        if isinstance(m, AIMessage):
            text = m.content if isinstance(m.content, str) else m.text
            if m.tool_calls:
                if text.strip():
                    yield Step("thought", agent, _preview(text), text)
                for tc in m.tool_calls:
                    args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                    yield Step("tool_call", agent, f"{tc['name']}({_preview(args, 120)})", data={"tool": tc["name"], "args": tc["args"]})
            elif text.strip():
                yield Step("agent_answer", agent, _preview(text), text)
        elif isinstance(m, ToolMessage):
            text = str(m.content)
            kind = "tool_error" if text.startswith("TOOL_ERROR") else "tool_result"
            yield Step(kind, agent, f"{m.name}: {_preview(text, 120)}", text, data={"tool": m.name})


def stream_steps(graph, inputs: dict, config: dict | None = None, default_agent: str = "agent") -> Iterator[Step]:
    """Run a graph and yield Steps as they happen. Nodes may also emit ready-made Steps under the
    `trace` state key (the supervisor does this for routing decisions)."""
    for ns, update in graph.stream(inputs, config or {}, stream_mode="updates", subgraphs=True):
        agent = _agent_from_namespace(ns, default_agent)
        for _node, payload in (update or {}).items():
            if not isinstance(payload, dict):
                continue
            yield from payload.get("trace", [])
            if ns or default_agent != "supervisor":  # specialist internals; supervisor messages come via `trace`
                yield from message_steps(payload.get("messages", []), agent)
