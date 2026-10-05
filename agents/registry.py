"""Agent registry. The supervisor discovers specialists here; it never imports them by name.

An agent module registers a builder:

    @register_agent("competitor_research")
    def build(name: str, cfg: AgentConfig, app: AppConfig):
        return build_react_agent(name, cfg, app)

and is switched on by a matching block in config/agents.yaml (description, model, tools, prompt).
"""

import importlib
import pkgutil
from typing import Callable

from core.config import AgentConfig, AppConfig

Builder = Callable[[str, AgentConfig, AppConfig], object]
_AGENTS: dict[str, Builder] = {}


def register_agent(name: str):
    def decorator(builder: Builder):
        if name in _AGENTS and _AGENTS[name].__module__ != builder.__module__:
            raise ValueError(f"agent '{name}' registered by two modules")
        _AGENTS[name] = builder
        return builder

    return decorator


def available_agents() -> dict[str, Builder]:
    import agents

    for mod in pkgutil.iter_modules(agents.__path__):
        if mod.name not in ("registry", "base"):
            importlib.import_module(f"agents.{mod.name}")
    return dict(_AGENTS)


def build_agent(name: str, app: AppConfig):
    registry = available_agents()
    if name not in app.agents:
        raise KeyError(f"agent '{name}' has no block in config/agents.yaml")
    if name not in registry:
        raise KeyError(f"agent '{name}' is in config but no module registers it; registered: {sorted(registry)}")
    return registry[name](name, app.agents[name], app)


def build_enabled_agents(app: AppConfig) -> dict[str, object]:
    return {name: build_agent(name, app) for name in app.enabled_agents}
