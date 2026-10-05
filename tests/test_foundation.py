"""Phase 1: config, model layer, tool and agent registries. Offline; no LLM calls."""

import json
import subprocess

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from core import models
from core.config import ModelSpec, Settings, load_config
from tools import registry as tool_registry


def test_env_overrides_settings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen3.5:9b")
    monkeypatch.setenv("MOCK_MODE", "true")
    app = load_config()
    assert app.settings.llm.provider == "ollama"
    assert app.settings.llm.model == "qwen3.5:9b"
    assert app.settings.mock_mode is True


def test_defaults_without_env(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "MOCK_MODE", "LLM_BASE_URL"):
        monkeypatch.setenv(var, "")  # blank, so .env cannot fill it either
    app = load_config()
    assert app.settings.llm.provider == "claude_code"
    assert app.settings.llm.base_url is None
    assert app.settings.mock_mode is False


def test_unknown_provider_rejected():
    with pytest.raises(ValidationError):
        ModelSpec(provider="gemini-ish", model="x")


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(models.ModelConfigError, match="ANTHROPIC_API_KEY"):
        models.get_model(ModelSpec(provider="anthropic", model="claude-sonnet-5-5"))


def test_tool_errors_become_text():
    class Args(BaseModel):
        x: int

    @tool_registry.register_tool("_explodes", "test only", Args)
    def _explodes(x: int, *, options: dict) -> str:
        raise ConnectionError("rate limited")

    try:
        [tool] = tool_registry.build_tools(["_explodes"], Settings(llm={"provider": "claude_code", "model": "sonnet"}))
        assert tool.invoke({"x": 1}) == "TOOL_ERROR (_explodes): ConnectionError: rate limited"
    finally:
        tool_registry._TOOLS.pop("_explodes")


def test_unknown_tool_name_fails_loudly():
    with pytest.raises(KeyError, match="nope"):
        tool_registry.build_tools(["nope"], Settings(llm={"provider": "claude_code", "model": "sonnet"}))


class _FakeProc:
    def __init__(self, payload):
        self.stdout, self.stderr, self.returncode = json.dumps(payload), "", 0


def test_claude_code_tool_call_roundtrip(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["input"], seen["env"] = cmd, kw["input"], kw["env"]
        return _FakeProc({"is_error": False, "structured_output": {
            "content": "checking date", "tool_calls": [{"name": "current_date", "args": {}}]}})

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    tools = tool_registry.build_tools(["current_date"], Settings(llm={"provider": "claude_code", "model": "sonnet"}))
    reply = models.ClaudeCodeChat().bind_tools(tools).invoke([
        SystemMessage("You are a researcher."),
        HumanMessage("date?"),
        AIMessage("", tool_calls=[{"name": "current_date", "args": {}, "id": "c1"}]),
        ToolMessage("2026-10-05", tool_call_id="c1", name="current_date"),
    ])

    assert [c["name"] for c in reply.tool_calls] == ["current_date"]
    assert reply.tool_calls[0]["id"].startswith("call_")
    schema = json.loads(seen["cmd"][seen["cmd"].index("--json-schema") + 1])
    assert schema["properties"]["tool_calls"]["items"]["anyOf"][0]["properties"]["name"] == {"const": "current_date"}
    assert "You are a researcher." in seen["cmd"][seen["cmd"].index("--system-prompt") + 1]
    assert "[TOOL RESULT · current_date · id=c1]\n2026-10-05" in seen["input"]
    assert "ANTHROPIC_API_KEY" not in seen["env"]


def test_claude_code_structured_output_forces_a_call(monkeypatch):
    class Pick(BaseModel):
        agent: str

    def fake_run(cmd, **kw):
        schema = json.loads(cmd[cmd.index("--json-schema") + 1])
        assert schema["properties"]["tool_calls"]["minItems"] == 1
        return _FakeProc({"is_error": False, "structured_output": {
            "content": "", "tool_calls": [{"name": "Pick", "args": {"agent": "trend_monitoring"}}]}})

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert models.ClaudeCodeChat().with_structured_output(Pick).invoke("pick") == Pick(agent="trend_monitoring")


def test_claude_code_cli_error_surfaces(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: _FakeProc({"is_error": True, "result": "Not logged in"}))
    with pytest.raises(RuntimeError, match="Not logged in"):
        models.ClaudeCodeChat().invoke("hi")


def _schema_for(monkeypatch, tool_choice):
    seen = {}

    def fake_run(cmd, **kw):
        seen["schema"] = json.loads(cmd[cmd.index("--json-schema") + 1])
        seen["env"] = kw["env"]
        return _FakeProc({"is_error": False, "structured_output": {"content": "", "tool_calls": []}})

    monkeypatch.setattr(subprocess, "run", fake_run)
    tools = tool_registry.build_tools(["current_date"], Settings(llm={"provider": "claude_code", "model": "sonnet"}))
    models.ClaudeCodeChat().bind_tools(tools, tool_choice=tool_choice).invoke("x")
    return seen


@pytest.mark.parametrize("choice", [True, "required", "any", "current_date",
                                    {"type": "function", "function": {"name": "current_date"}}])
def test_forcing_tool_choice_spellings(monkeypatch, choice):
    calls = _schema_for(monkeypatch, choice)["schema"]["properties"]["tool_calls"]
    assert calls["minItems"] == 1 and len(calls["items"]["anyOf"]) == 1


def test_tool_choice_naming_unbound_tool_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="names no bound tool"):
        _schema_for(monkeypatch, "web_search")


def test_all_api_routing_env_is_stripped(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.setenv(var, "x")
    env = _schema_for(monkeypatch, None)["env"]
    assert not {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"} & env.keys()


def test_missing_structured_output_is_an_error_not_an_empty_reply(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: _FakeProc(
        {"is_error": False, "subtype": "error_max_structured_output_retries", "result": "gave up"}))
    tools = tool_registry.build_tools(["current_date"], Settings(llm={"provider": "claude_code", "model": "sonnet"}))
    with pytest.raises(RuntimeError, match="no structured output"):
        models.ClaudeCodeChat().bind_tools(tools).invoke("x")


def test_response_format_is_refused_loudly():
    with pytest.raises(ValueError, match="response_format"):
        models.ClaudeCodeChat().bind_tools([], response_format={"type": "json_schema"})


def test_reimport_after_partial_failure_keeps_real_error(monkeypatch, tmp_path):
    import importlib, sys, tools as tools_pkg
    (tmp_path / "half_broken.py").write_text(
        "from pydantic import BaseModel\nfrom tools.registry import register_tool\n"
        "class A(BaseModel):\n    pass\n"
        "@register_tool('_half', 'x', A)\ndef f(*, options): return ''\n"
        "import does_not_exist_xyz\n")
    monkeypatch.setattr(tools_pkg, "__path__", [*tools_pkg.__path__, str(tmp_path)])
    try:
        for _ in range(2):
            with pytest.raises(ModuleNotFoundError):
                tool_registry.available_tools()
            sys.modules.pop("tools.half_broken", None)
    finally:
        tool_registry._TOOLS.pop("_half", None)
