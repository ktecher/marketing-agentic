# Progress log

Read at session start. Append one dated entry per phase or checkpoint.

## Decisions (2026-10-05)

- Default LLM is `claude_code` (local Claude Code CLI, subscription, no API key). The API
  provider is selectable via `LLM_PROVIDER` + key in `.env`; the UI settings panel comes in phase 7.
- Mock data uses a fictional client brand and fictional competitors. Never fabricate intel about real companies.
- English only for now. Arabic variations are planned: keep `language` a prompt/request parameter, not hardcoded.
- Repo: private GitHub `ktecher/marketing-agentic`; commit at the end of each phase.
- Python 3.13 (3.14 is installed but ecosystem wheels lag).
- Supervisor: custom LangGraph node with structured output, not the `langgraph-supervisor` package (needs a visible reasoning field for the trace).
- Docker image can't use `claude_code` (no CLI login inside); it defaults to an API provider.

## 2026-10-05 — Phase 1: skeleton, config, models, tool + agent registries

Built: `core/config.py`, `core/models.py` (incl. `ClaudeCodeChat`), `tools/registry.py` +
`current_date`, `agents/registry.py` + `agents/base.py`, `cli.py`, 9 offline tests.

Verified: `cli.py ping` on claude_code gave a text reply in 3.1s and a correct tool call in 4.0s.
A full `create_agent` ReAct loop (tool call, then result, then final answer) took 9.2s.
A mutation check (removing the API-key strip) turned the guard test red.

Next: phase 2, `web_search` (ddgs / tavily / brave) and `read_page` (trafilatura / jina / firecrawl) with mock mode.

Review (separate code-reviewer agent): 6 findings, 0 blocking, all fixed with tests (19 passing):
tool_choice spellings (True / dict / unknown name), missing structured_output now raises instead
of looping on empty replies, `response_format` refused loudly, re-import after a half-failed
module keeps the real error, null `result`, and ANTHROPIC_AUTH_TOKEN/BASE_URL also stripped.
Phase 3+ note: agents must not pass `response_format=` with claude_code; use ToolStrategy.


## 2026-10-05 — Phases 2-8 (autonomous run, parallel agents)

Built: tools (web_search ddgs/tavily/brave, read_page trafilatura/jina/firecrawl with SSRF
guard, mock mode + 32-entry/45-page fictional fixture world), three specialists, supervisor
(route → agent nodes → finish; single specialist passed through verbatim, several synthesised),
shared trace, CLI `agent`/`ask`, Streamlit UI (team strip, live trace, session-only API key),
Docker (Ollama on host by default), README.

Bugs found and fixed:
- ClaudeCodeChat: native tool attempts → ungrounded answers (0/3 → 5/5 clean after instruction
  rewrite) and stub final answers ("done") in ~50% of runs (7/7 full reports after
  stream-json recovery). Both silent, both caught only by reading the CLI event stream.
- Supervisor: specialist answers duplicated in the trace; build errors escaped `run()`.
- Mock lookup split "Nuvéa" into "nuv"+"a" (accent stripping added); `_load` cache thrashed.
- Content drafts' header lines ran together in markdown (prompt now asks for blank lines).

Verified: 54 offline tests; live e2e `ask` Kiraleaf→3 Instagram posts in mock mode (145s,
2 specialists, synthesised); UI run in Chrome (36s, chips + trace + answer); the README's
4th-agent tutorial followed literally in a scratch copy — supervisor routed to `seo_audit`
with no code edits (44s); Docker image healthy, Ollama ping + tool call from inside it.

Next ideas: Arabic variations (language is already a prompt parameter in content_drafting);
`anthropic` provider for client deployments (much faster than claude_code).
