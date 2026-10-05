"""Command line entry point.

    uv run python cli.py config                  # show resolved config
    uv run python cli.py tools                   # list registered tools
    uv run python cli.py tool current_date '{}'  # run one tool
    uv run python cli.py ping                    # check the configured LLM (text + tool call)
"""

import argparse
import json
import time

from core.config import load_config
from core.models import get_model
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

    args = parser.parse_args()
    args.fn(load_config(), args)


if __name__ == "__main__":
    main()
