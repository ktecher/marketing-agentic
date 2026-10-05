"""Load and validate config/*.yaml. Every other module gets settings from here."""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

Provider = Literal["claude_code", "anthropic", "openai", "ollama", "openai_compatible"]


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model: str
    base_url: str | None = None
    temperature: float | None = None
    # Set only by the UI for one browser session; never put a key in YAML (use .env).
    api_key: SecretStr | None = Field(None, exclude=True)

    @field_validator("base_url", mode="before")
    @classmethod
    def _blank_is_none(cls, v):
        return v or None


class Market(BaseModel):
    """Where the director works. Localises search results and is briefed to every agent."""

    model_config = ConfigDict(extra="forbid")

    country: str = "Kuwait"
    code: str = "kw"          # ISO 3166 alpha-2, lower case: search engines' country filter
    currency: str = "KWD"
    prompt: str = "prompts/market.md"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mock_mode: bool = False
    market: Market = Field(default_factory=Market)
    llm: ModelSpec
    tools: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # API keys typed in the UI for one browser session, by env var name (e.g. SERPER_API_KEY).
    # They win over .env and are never written to os.environ, disk or a dump.
    secrets: dict[str, SecretStr] = Field(default_factory=dict, exclude=True)


class SupervisorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    synthesis_prompt: str = "prompts/synthesis.md"
    max_steps: int = 6
    model: ModelSpec | None = None


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    description: str
    prompt: str
    tools: list[str] = Field(default_factory=list)
    model: ModelSpec | None = None
    max_iterations: int = 8


class AppConfig(BaseModel):
    settings: Settings
    supervisor: SupervisorConfig
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    config_dir: Path = CONFIG_DIR

    def model_for(self, spec: ModelSpec | None) -> ModelSpec:
        return spec or self.settings.llm

    def prompt_text(self, rel_path: str) -> str:
        return (self.config_dir / rel_path).read_text(encoding="utf-8")

    def market_brief(self) -> str:
        """The market context every agent and the supervisor receive, filled from settings.market."""
        m = self.settings.market
        return self.prompt_text(m.prompt).format(country=m.country, currency=m.currency)

    @property
    def enabled_agents(self) -> dict[str, AgentConfig]:
        return {k: v for k, v in self.agents.items() if v.enabled}


_ENV_REF = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    """Replace ${VAR} / ${VAR:-default} in every string value of the parsed YAML."""
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _read_yaml(path: Path) -> dict:
    return _expand(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def load_config(config_dir: Path = CONFIG_DIR) -> AppConfig:
    load_dotenv(ROOT / ".env")
    agents_yaml = _read_yaml(config_dir / "agents.yaml")
    return AppConfig(
        settings=Settings(**_read_yaml(config_dir / "settings.yaml")),
        supervisor=SupervisorConfig(**agents_yaml["supervisor"]),
        agents=agents_yaml.get("agents") or {},
        config_dir=config_dir,
    )


def ui_can_save() -> bool:
    """Whether the UI may write defaults to .env. Off in shared deployments (ALLOW_UI_SAVE=false),
    so a visitor cannot replace the server's keys."""
    return os.environ.get("ALLOW_UI_SAVE", "true").strip().lower() not in ("false", "0", "no", "off")


def save_env(updates: dict[str, str], path: Path | None = None) -> None:
    """Set KEY=value lines in .env (replacing or appending) and apply them to this process."""
    path = path or ROOT / ".env"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for key, value in updates.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or "\n" in value or '"' in value:
            raise ValueError(f"refusing to write {key!r}")
        # Quote anything dotenv would misread: an unquoted " #" starts a comment.
        line = f'{key}="{value}"' if re.search(r"[\s#'\\]", value) else f"{key}={value}"
        for i, existing in enumerate(lines):
            if re.match(rf"\s*{key}\s*=", existing):
                lines[i] = line
                break
        else:
            lines.append(line)
        os.environ[key] = value
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide config. Tests and the UI call load_config() directly for a fresh copy."""
    return load_config()
