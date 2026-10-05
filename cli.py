"""Command line entry point.

    uv run python cli.py config                  # show resolved config
    uv run python cli.py tools                   # list registered tools
    uv run python cli.py tool current_date '{}'  # run one tool
    uv run python cli.py ping                    # check the configured LLM (text + tool call)
    uv run python cli.py agent competitor_research "Research Kiraleaf" [--mock]
    uv run python cli.py ask "Research Kiraleaf, then draft 3 posts against them" [--mock]
"""

import argparse
import json
import time

from agents.base import recursion_limit
from agents.registry import build_agent
from core.config import load_config
from core.models import get_model
from core.supervisor import run as run_supervisor
from core.trace import stream_steps
from tools.registry import available_tools, build_tools


def cmd_config(app, _):
    print(app.model_dump_json(indent=2, exclude={"config_dir"}))


def cmd_tools(app, _):
    for name, t in sorted(available_tools().items()):
        print(f"{name:16} {t.description}")
        print(f"{'':16} args: {json.dumps(t.args_schema.model_json_schema().get('properties', {}))}")


def cmd_tool(app, args):
    [tool] = build_tools([args.name], app.settings)
    print(tool.invoke(json.loads(args.args)))


def cmd_ping(app, _):
    spec = app.settings.llm
    model = get_model(spec)
    print(f"provider={spec.provider} model={spec.model}")

    t = time.time()
    reply = model.invoke("Reply with exactly: pong")
    print(f"text      ({time.time() - t:4.1f}s): {reply.content!r}")

    t = time.time()
    tools = build_tools(["current_date"], app.settings)
    reply = model.bind_tools(tools).invoke("What is today's date? Use your tool.")
    print(f"tool call ({time.time() - t:4.1f}s): {[(c['name'], c['args']) for c in reply.tool_calls]}")


ICONS = {"route": "⇢", "thought": "…", "tool_call": "→", "tool_result": "←", "tool_error": "✗",
         "agent_answer": "✓", "final": "★", "error": "✗"}


def print_steps(steps) -> str:
    """Print each trace step as it arrives; return the last answer text."""
    answer = ""
    for step in steps:
        print(f"  {ICONS.get(step.kind, '·')} [{step.agent}] {step.title}", flush=True)
        if step.kind in ("agent_answer", "final"):
            answer = step.detail
    return answer


def cmd_agent(app, args):
    t = time.time()
    graph = build_agent(args.name, app)
    print(f"agent={args.name} mock_mode={app.settings.mock_mode} llm={app.model_for(app.agents[args.name].model).provider}\n")
    answer = print_steps(stream_steps(
        graph,
        {"messages": [("user", args.request)]},
        {"recursion_limit": recursion_limit(app.agents[args.name])},
        default_agent=args.name,
    ))
    print(f"\n{'=' * 70}\n{answer}\n{'=' * 70}\n({time.time() - t:.0f}s)")


def cmd_ask(app, args):
    t = time.time()
    print(f"supervisor · mock_mode={app.settings.mock_mode} · agents={list(app.enabled_agents)}\n")
    answer = print_steps(run_supervisor(app, args.request))
    print(f"\n{'=' * 70}\n{answer}\n{'=' * 70}\n({time.time() - t:.0f}s)")


def main():
    parser = argparse.ArgumentParser(description="Marketing Agentic AI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("config").set_defaults(fn=cmd_config)
    sub.add_parser("tools").set_defaults(fn=cmd_tools)
    p = sub.add_parser("tool")
    p.add_argument("name")
    p.add_argument("args", nargs="?", default="{}", help="JSON object of tool arguments")
    p.set_defaults(fn=cmd_tool)
    sub.add_parser("ping").set_defaults(fn=cmd_ping)
    p = sub.add_parser("agent", help="run one specialist on its own")
    p.add_argument("name")
    p.add_argument("request")
    p.set_defaults(fn=cmd_agent)
    p = sub.add_parser("ask", help="send a request to the supervisor (end to end)")
    p.add_argument("request")
    p.set_defaults(fn=cmd_ask)
    for p in sub.choices.values():
        p.add_argument("--mock", action="store_true", help="force mock_mode on (fictional sample data)")

    args = parser.parse_args()
    app = load_config()
    if args.mock:
        app.settings.mock_mode = True
    args.fn(app, args)


if __name__ == "__main__":
    main()
