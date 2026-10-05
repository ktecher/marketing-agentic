# Marketing Agentic AI — client demo

A multi-agent system for a marketing agency, built for pitches. A supervisor takes a marketing
director's request, delegates to specialist agents, and returns one answer with a visible trace.
Python 3.13 + LangGraph/LangChain 1.x, managed with `uv`.

Build status and decisions: `notes/progress.md`. Read it at session start; append to it at the
end of each phase.

## Layout

| Path | Role |
|---|---|
| `config/settings.yaml` | Global: `mock_mode`, default `llm`, per-tool provider options |
| `config/agents.yaml` | Supervisor + specialists: enabled, description, model, tools, prompt |
| `config/prompts/*.md` | One system prompt per agent, plus `supervisor.md` (routing, `{agents}` placeholder) and `synthesis.md` |
| `core/config.py` | Loads YAML, expands `${VAR:-default}` from env/.env, validates with pydantic |
| `core/models.py` | `get_model(spec)` for every provider, incl. `ClaudeCodeChat` |
| `core/supervisor.py` | Supervisor graph in rounds: `route` → specialists of this round IN PARALLEL (LangGraph `Send`) → `route` … → `finish`. Dependent work waits a round and gets earlier outputs as context. State keys written by parallel nodes (`results`, `trace`) need reducers. `run(app, request)` yields trace Steps |
| `core/trace.py` | `Step` + `stream_steps`: turns LangGraph stream updates into the trace (CLI and UI share it) |
| `tools/registry.py` | `@register_tool`, `build_tools(names, settings)` with error wrapping + mock-mode switch |
| `tools/web_search.py`, `tools/read_page.py` | Free defaults (ddgs, scrapling with stealth-browser fallback) + paid swaps; `read_page` refuses non-public addresses |
| `tools/mock.py`, `data/mock/` | Mock mode: fixture answers labelled `[mock data]`; fictional brand world (see `data/mock/README.md`) |
| `agents/registry.py` | `@register_agent`, `build_agent`, `build_enabled_agents` |
| `agents/base.py` | `build_react_agent` — shared ReAct loop (`langchain.agents.create_agent`) |
| `agents/<name>.py` | competitor_research, content_drafting, trend_monitoring (standard pattern) |
| `cli.py` | `config`, `tools`, `tool`, `ping`, `agent <name> "<req>"`, `ask "<req>"`; `--mock` on any |
| `ui/app.py`, `ui/theme.css` | Streamlit demo: live trace, team strip, sidebar LLM settings (session-only key) |
| `examples/*.yaml` | Example requests per agent and end to end |
| `Dockerfile`, `docker-compose.yml` | Container defaults to Ollama on the host (claude_code can't run in a container) |
| `tests/` | Offline by default; `live` marker for real LLM/network |

## Conventions

- **Config, not code.** Models, tools, prompts and enabled agents live in YAML. Never hardcode
  a model id or provider in Python.
- **Registries discover modules.** Any module in `tools/` or `agents/` is imported automatically.
  The supervisor reads agents from the registry + `agents.yaml`, never by import.
- **Tools never raise into the agent.** `build_tools` turns exceptions into
  `TOOL_ERROR (<name>): <Type>: <msg>` text so the agent can report and continue.
- **Secrets from env or the UI session only.** Add a new key to `.env.example` and to `API_KEYS` in
  `ui/app.py`, never to YAML. Read keys with `tools.registry.secret(name, options)`: it prefers the
  UI session's key (`settings.secrets`, never written to `os.environ`) over `.env`.
- Tests never read the real `.env` (`tests/conftest.py` disables it); set env explicitly per test.
- Verify LangChain/LangGraph APIs against the installed package (`inspect.signature`), not
  memory. LangChain 1.x moved things: use `langchain.agents.create_agent`, `langchain.chat_models.init_chat_model`.

## LLM providers

`settings.llm.provider` (env `LLM_PROVIDER`) is one of:

| provider | needs | notes |
|---|---|---|
| `claude_code` | `claude` CLI logged in | Default. Uses the local Claude Code subscription, no API key. ~3-5s per call. Not for client deployment. |
| `anthropic` | `ANTHROPIC_API_KEY` | Use for client deployments. |
| `openai` | `OPENAI_API_KEY` | |
| `ollama` | local Ollama (`OLLAMA_BASE_URL` optional) | Fully offline; small models route multi-step requests poorly. |
| `openai_compatible` | `LLM_BASE_URL` + `LLM_API_KEY` | Groq, OpenRouter, Together, LM Studio... |

Per-agent override in `agents.yaml`: `model: { provider: anthropic, model: claude-sonnet-5-5 }`.

`ClaudeCodeChat` runs `claude -p` isolated (`--setting-sources ''`, `--disable-slash-commands`,
`--strict-mcp-config`, `--tools ''`, cwd = temp dir) so this machine's CLAUDE.md, hooks and
skills never reach the agents. It strips `ANTHROPIC_API_KEY` from the subprocess env so the CLI
uses the subscription. Tool calls are emulated with `--json-schema`; `with_structured_output`
works through the same path. A test guards each of these.

It reads `--output-format stream-json` because two failure modes only show in the event stream,
both silent (exit 0, well-formed output) and both seen in ~50% of runs before the fixes:
- the model calls a bound tool NATIVELY ("No such tool") then returns no tool calls → those
  attempts are converted into the tool calls (`_native_attempts`);
- the model writes the final answer as plain text and puts a stub ("done", "see above") in
  `content` → the longest assistant text is used when `content` is short (`_recover_answer`).
`_TOOL_INSTRUCTIONS` tells the model both rules; the recovery code is the backstop. When an
agent's output looks wrong on claude_code, re-run with stream-json and read every event before
changing a prompt.

## How to add a tool

1. New module `tools/<name>.py`: a pydantic args model and a function
   `def fn(**args, *, options: dict) -> str`, decorated with `@register_tool(name, description, Args)`.
   `options` is `settings.tools.<name>` — read `options.get("provider")` to support swaps.
2. Add `<name>: {...}` under `tools:` in `config/settings.yaml`.
3. List it in an agent's `tools:` in `config/agents.yaml`.
4. Check: `uv run python cli.py tool <name> '{"arg": "..."}'`.

## How to add an agent

1. `agents/<name>.py`:
   ```python
   from agents.base import build_react_agent
   from agents.registry import register_agent

   @register_agent("<name>")
   def build(name, cfg, app):
       return build_react_agent(name, cfg, app)
   ```
2. `config/prompts/<name>.md` — the system prompt.
3. Block under `agents:` in `config/agents.yaml` with `description` (the supervisor routes on
   this — write it as "what requests to send here"), `prompt`, `tools`, optional `model`.
4. No edits to the supervisor or other agents.

## Commands

```bash
uv sync                          # install
uv run pytest -q                 # offline tests
RUN_LIVE=1 uv run pytest -q      # include live LLM/network tests
uv run python cli.py ping        # check the configured LLM
uv run python cli.py agent trend_monitoring "What are people saying about Aurelle Skin?" --mock
uv run python cli.py ask "Research Kiraleaf, then draft 3 posts against them" --mock
uv run streamlit run ui/app.py   # demo UI on :8501
docker compose up --build        # container on :8501 (Ollama on host by default)
```

Runs on claude_code take ~30-60s per specialist and ~2-3 min for a two-agent request.
Parallel agents/sessions: never `git stash`/checkout for a baseline (use `git show HEAD:<file>`).
