"""Tool abstraction: one interface for every tool, assigned to agents by name in config.

A tool is a plain function plus a name, a description and a pydantic input schema:

    class SearchArgs(BaseModel):
        query: str

    @register_tool("web_search", "Search the web...", SearchArgs)
    def web_search(query: str, *, options: dict) -> str: ...

`options` is the tool's block from config/settings.yaml (provider, limits...), so swapping a
free provider for a paid one is a config change. build_tools() wraps every tool so failures come
back to the agent as a TOOL_ERROR string instead of crashing the run.
"""

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from core.config import Settings
from tools.mock import mock_result


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    args_schema: type[BaseModel]
    func: Callable[..., str]


_TOOLS: dict[str, ToolDef] = {}


def register_tool(name: str, description: str, args_schema: type[BaseModel]):
    def decorator(func: Callable[..., str]):
        if name in _TOOLS and _TOOLS[name].func.__module__ != func.__module__:
            raise ValueError(f"tool '{name}' registered by two modules")
        _TOOLS[name] = ToolDef(name, description, args_schema, func)
        return func

    return decorator


def _discover() -> None:
    import tools

    for mod in pkgutil.iter_modules(tools.__path__):
        if mod.name not in ("registry", "mock"):
            importlib.import_module(f"tools.{mod.name}")


def available_tools() -> dict[str, ToolDef]:
    _discover()
    return dict(_TOOLS)


def _wrap(tool: ToolDef, settings: Settings) -> Callable[..., str]:
    options = settings.tools.get(tool.name, {})

    def run(**kwargs) -> str:
        try:
            if settings.mock_mode and (cached := mock_result(tool.name, kwargs)) is not None:
                return cached
            return tool.func(**kwargs, options=options)
        except Exception as e:  # reported to the agent, which decides how to continue
            return f"TOOL_ERROR ({tool.name}): {type(e).__name__}: {e}"

    return run


def build_tools(names: list[str], settings: Settings) -> list[StructuredTool]:
    registry = available_tools()
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise KeyError(f"unknown tool(s) {unknown}; registered: {sorted(registry)}")
    return [
        StructuredTool.from_function(
            func=_wrap(registry[n], settings),
            name=n,
            description=registry[n].description,
            args_schema=registry[n].args_schema,
        )
        for n in names
    ]
