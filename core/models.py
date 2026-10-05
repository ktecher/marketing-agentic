"""Model layer: get_model(spec) turns a ModelSpec from config into a LangChain chat model.

Switching an agent's model is a config change only. Every provider returns a BaseChatModel
that supports .bind_tools() and .with_structured_output(), so agents never branch on provider.
"""

import json
import os
import subprocess
import tempfile
import uuid
from typing import Any, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from core.config import ModelSpec


class ModelConfigError(RuntimeError):
    """A provider is selected but its key or URL is missing."""


def _require_env(name: str, provider: str, spec: ModelSpec | None = None) -> str:
    value = (spec.api_key.get_secret_value() if spec and spec.api_key else None) or os.environ.get(name)
    if not value:
        raise ModelConfigError(
            f"provider '{provider}' needs {name}. Add it to .env, or set LLM_PROVIDER=claude_code."
        )
    return value


def get_model(spec: ModelSpec) -> BaseChatModel:
    if spec.provider == "claude_code":
        return ClaudeCodeChat(model=spec.model)

    kwargs: dict[str, Any] = {}
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature

    if spec.provider == "anthropic":
        key = _require_env("ANTHROPIC_API_KEY", spec.provider, spec)
        return init_chat_model(spec.model, model_provider="anthropic", api_key=key, **kwargs)
    if spec.provider == "openai":
        key = _require_env("OPENAI_API_KEY", spec.provider, spec)
        return init_chat_model(spec.model, model_provider="openai", api_key=key, **kwargs)
    if spec.provider == "ollama":
        base_url = spec.base_url or os.environ.get("OLLAMA_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        return init_chat_model(spec.model, model_provider="ollama", **kwargs)
    if spec.provider == "openai_compatible":
        if not spec.base_url:
            raise ModelConfigError("provider 'openai_compatible' needs LLM_BASE_URL (or base_url in config).")
        if spec.base_url.rstrip("/") != os.environ.get("LLM_BASE_URL", "").rstrip("/"):
            # A UI visitor typed this endpoint: never send it the server's LLM_API_KEY, and never
            # let it point the server at its own network.
            if not spec.api_key:
                raise ModelConfigError("a custom base URL needs its own API key (the server key only goes to LLM_BASE_URL).")
            from tools.read_page import _check_public

            try:
                _check_public(spec.base_url)
            except (ValueError, OSError) as e:
                raise ModelConfigError(f"base URL refused: {e}") from e
        api_key = _require_env("LLM_API_KEY", spec.provider, spec)
        return init_chat_model(
            spec.model, model_provider="openai", base_url=spec.base_url, api_key=api_key, **kwargs
        )
    raise ModelConfigError(f"unknown provider '{spec.provider}'")


# --------------------------------------------------------------------------------------------
# claude_code: uses the local Claude Code CLI (`claude -p`) and its logged-in subscription.
# No API key. Each call is one isolated, tool-less CLI run. Tool calling is emulated with
# --json-schema structured output, so LangGraph sees ordinary AIMessage.tool_calls.
# ponytail: one subprocess per LLM step (~5s). Swap to provider: anthropic for speed/production.
# --------------------------------------------------------------------------------------------

_TOOL_INSTRUCTIONS = """

## Tools
The tools named in this prompt ({names}) are NOT directly callable here: calling them as tools
fails with "No such tool". Your only callable tool is StructuredOutput. To use a tool, call
StructuredOutput once with that tool listed in `tool_calls` (several at once is fine) and a
one-line note on what you are doing in `content`. Stop there: the results arrive in the next
message. When you have everything you need, call StructuredOutput with an empty `tool_calls`
list and your COMPLETE final answer in `content`. Text written outside StructuredOutput is
discarded, so never put the answer anywhere else or refer to it ("see above", "done")."""


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def _render_transcript(messages: Sequence[BaseMessage]) -> str:
    parts = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, ToolMessage):
            parts.append(f"[TOOL RESULT · {m.name or 'tool'} · id={m.tool_call_id}]\n{_text(m)}")
        elif isinstance(m, AIMessage):
            block = f"[ASSISTANT]\n{_text(m)}".rstrip()
            for tc in m.tool_calls:
                block += f"\n(called {tc['name']} id={tc['id']} args={json.dumps(tc['args'])})"
            parts.append(block)
        else:
            parts.append(f"[USER]\n{_text(m)}")
    return "\n\n".join(parts)


def _normalize_tool_choice(tool_choice: Any) -> str | None:
    """LangChain/OpenAI spell tool_choice several ways; reduce to None | "auto" | "any" | <name>."""
    if tool_choice is True or tool_choice == "required":
        return "any"
    if isinstance(tool_choice, dict):
        return tool_choice.get("function", {}).get("name") or tool_choice.get("name") or "any"
    return tool_choice


def _tool_call_schema(tools: list[dict], tool_choice: str | None) -> dict:
    if tool_choice not in (None, "auto", "any"):
        tools = [t for t in tools if t["function"]["name"] == tool_choice]
        if not tools:
            raise ValueError(f"tool_choice {tool_choice!r} names no bound tool")
    variants = [
        {
            "type": "object",
            "properties": {
                "name": {"const": t["function"]["name"]},
                "args": t["function"].get("parameters") or {"type": "object", "properties": {}},
            },
            "required": ["name", "args"],
        }
        for t in tools
    ]
    calls: dict[str, Any] = {"type": "array", "items": {"anyOf": variants}}
    if tool_choice not in (None, "auto"):
        calls["minItems"] = 1
    return {
        "type": "object",
        "properties": {"content": {"type": "string"}, "tool_calls": calls},
        "required": ["content", "tool_calls"],
    }


_STUB_CHARS = 300  # a `content` shorter than this, with no tool calls, is not a finished answer


def _blocks(events: list[dict]) -> list[dict]:
    return [
        b for e in events if e.get("type") == "assistant"
        for b in e.get("message", {}).get("content", []) if isinstance(b, dict)
    ]


def _call_key(name: str, args: dict) -> tuple[str, str]:
    return name, json.dumps(args, sort_keys=True)


def _native_attempts(events: list[dict], bound: set[str], already_run: set[tuple[str, str]]) -> list[dict]:
    """Bound tools the model tried to call natively in this CLI run, minus repeats of calls the
    agent already executed (habitual re-calls), deduplicated."""
    calls, seen = [], set(already_run)
    for b in _blocks(events):
        if b.get("type") == "tool_use" and b.get("name") in bound:
            key = _call_key(b["name"], b.get("input") or {})
            if key not in seen:
                seen.add(key)
                calls.append({"name": b["name"], "args": b.get("input") or {}})
    return calls


def _recover_answer(content: str, events: list[dict]) -> str:
    """The model sometimes writes the final answer as plain text and puts a stub ("done",
    "report delivered above") in `content`. Only text written after its last real tool attempt
    counts, so an opening preamble is never mistaken for the answer."""
    if len(content.strip()) >= _STUB_CHARS:
        return content
    blocks = _blocks(events)
    last_tool = max((i for i, b in enumerate(blocks) if b.get("type") == "tool_use" and b.get("name") != "StructuredOutput"), default=-1)
    texts = [b.get("text", "") for b in blocks[last_tool + 1:] if b.get("type") == "text"]
    longest = max(texts, key=len, default="")
    return longest if len(longest.strip()) >= _STUB_CHARS else content


def _parse_stream(stdout: str) -> tuple[list[dict], dict]:
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # the CLI may print warnings between events
    results = [e for e in events if isinstance(e, dict) and e.get("type", "result") == "result"]
    if not results:
        raise ValueError("no result event")
    return events, results[-1]


class ClaudeCodeChat(BaseChatModel):
    model: str = "sonnet"
    timeout: int = 300
    cli: str = "claude"

    @property
    def _llm_type(self) -> str:
        return "claude_code"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}

    def bind_tools(self, tools, *, tool_choice: Any = None, **kwargs):
        if kwargs.get("response_format"):
            # LangChain picks ProviderStrategy for some model names (e.g. claude-sonnet-4-5);
            # this shim only does tool-call structured output. Fail loudly instead of ignoring it.
            raise ValueError("claude_code does not support response_format; use ToolStrategy(...)")
        if tool_choice == "none":
            return self.bind(**kwargs)
        return self.bind(
            tools=[convert_to_openai_tool(t) for t in tools],
            tool_choice=_normalize_tool_choice(tool_choice),
            **kwargs,
        )

    def _generate(self, messages, stop=None, run_manager=None, tools=None, tool_choice=None, **kwargs):
        system = "\n\n".join(_text(m) for m in messages if isinstance(m, SystemMessage))
        system = system or "You are a helpful assistant."
        cmd = [
            self.cli, "-p",
            "--model", self.model,
            "--output-format", "stream-json", "--verbose",  # event stream: lets us see native tool attempts
            # Isolation: no user settings/hooks/plugins, no skills, no MCP, no built-in tools.
            "--setting-sources", "",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--tools", "",
            "--no-session-persistence",
        ]
        if tools:
            system += _TOOL_INSTRUCTIONS.format(names=", ".join(t["function"]["name"] for t in tools))
            cmd += ["--json-schema", json.dumps(_tool_call_schema(tools, tool_choice))]
        cmd += ["--system-prompt", system]

        env = dict(os.environ)
        # Any of these would make the CLI bill the API (or route elsewhere) instead of the subscription.
        for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
            env.pop(var, None)
        try:
            proc = subprocess.run(
                cmd,
                input=_render_transcript(messages),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                cwd=tempfile.gettempdir(),  # keep this repo's CLAUDE.md out of the agent's context
            )
        except FileNotFoundError as e:
            raise ModelConfigError("claude CLI not found. Install Claude Code or switch LLM_PROVIDER.") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"claude_code timed out after {self.timeout}s") from e

        try:
            events, data = _parse_stream(proc.stdout)
        except ValueError as e:
            detail = (proc.stderr or proc.stdout).strip()[:500]
            raise RuntimeError(f"claude_code exit {proc.returncode}: {detail}") from e
        if data.get("is_error"):
            raise RuntimeError(f"claude_code error: {str(data.get('result'))[:500]}")

        if tools:
            out = data.get("structured_output")
            if not isinstance(out, dict):
                # Never read this as "no tool calls": an agent would loop on empty replies.
                raise RuntimeError(
                    f"claude_code returned no structured output ({data.get('subtype')}): {str(data.get('result'))[:300]}"
                )
            calls = out.get("tool_calls") or []
            content = _recover_answer(out.get("content", ""), events) if not calls else out.get("content", "")
            if not calls and len(content.strip()) < _STUB_CHARS:
                # The model sometimes calls a bound tool NATIVELY (fails inside the CLI with
                # "No such tool") and then returns no tool calls and no real answer. Its intent was
                # the tool call. A substantial answer means it genuinely finished: leave it alone.
                already_run = {_call_key(c["name"], c["args"]) for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
                calls = _native_attempts(events, {t["function"]["name"] for t in tools}, already_run)
            message = AIMessage(
                content=content,
                tool_calls=[
                    {"name": c["name"], "args": c.get("args") or {}, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}
                    for c in calls
                ],
            )
        else:
            message = AIMessage(content=data.get("result") or "")
        message.response_metadata = {"model": self.model, "duration_ms": data.get("duration_ms"), "num_turns": data.get("num_turns")}
        return ChatResult(generations=[ChatGeneration(message=message)])
