# Marketing Agentic AI

## What it is

A multi-agent demo for a marketing agency. A marketing director types a request. A supervisor
splits it among three specialists: competitor research, content drafting and trend
monitoring. Each specialist is its own agent with its own tools and reason-act-observe loop.
Independent parts run in parallel; a part that needs another's result (posts based on research)
waits for it. The supervisor then returns one answer. The UI shows the
supervisor's reasoning and every tool call as it happens. The system produces drafts and reports
only; it never publishes anything.

```
                    director's request
                           |
                      SUPERVISOR   (routes step by step, then answers)
                           |
        +------------------+-------------------+
        |                  |                   |
 competitor_research  content_drafting   trend_monitoring
        |                  |                   |
        +--------+---------+---------+---------+
                 |                   |
            web_search           read_page          current_date
        (ddgs/tavily/brave) (scrapling/trafilatura/jina/firecrawl)
```

Built on Python 3.13, LangGraph/LangChain 1.x and Streamlit, managed with `uv`.

## Quick start (local)

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run scrapling install   # one-time: browser for the page reader's stealth mode
cp .env.example .env
uv run streamlit run ui/app.py
```

The default LLM is Claude Code: the `claude` CLI, logged in on this machine. It needs no API
key. Check it works before a demo:

```bash
uv run python cli.py ping
```

For an offline pitch, turn on **Sample data mode** in the UI sidebar. Search and page reads then
answer from a fictional brand world (see [Sample data mode](#sample-data-mode)).

## Running from the command line

Run from the project root.

| Command | What it does |
|---|---|
| `uv run python cli.py ping` | Checks the configured LLM with a text reply and a tool call |
| `uv run python cli.py agent <name> "<request>" [--mock]` | Runs one specialist on its own |
| `uv run python cli.py ask "<request>" [--mock]` | Sends a request to the supervisor, end to end |
| `uv run python cli.py tools` | Lists registered tools and their arguments |
| `uv run python cli.py tool <name> '<json>'` | Runs one tool with JSON arguments |
| `uv run python cli.py config` | Prints the resolved configuration |

Examples:

```bash
uv run python cli.py agent competitor_research "Research our competitor Kiraleaf" --mock
uv run python cli.py ask "Research our competitor Kiraleaf, then draft 3 Instagram posts positioning Aurelle Skin against them." --mock
uv run python cli.py tool current_date '{}'
```

Agent names are the keys in `config/agents.yaml`: `competitor_research`, `content_drafting`,
`trend_monitoring`.

Ready-made requests, with notes on what to expect, are in `examples/`:

- `examples/supervisor.yaml`
- `examples/competitor_research.yaml`
- `examples/content_drafting.yaml`
- `examples/trend_monitoring.yaml`

Entries marked `mock: true` use the fictional sample data; run them with `--mock`.

## Choosing the LLM service

Set the default for every agent in `.env` or in `config/settings.yaml` (the `llm:` block).

| Provider | Needs | Notes |
|---|---|---|
| `claude_code` | `claude` CLI, logged in | Default. No API key. Local demos only. |
| `anthropic` | `ANTHROPIC_API_KEY` | Use for client deployments. |
| `openai` | `OPENAI_API_KEY` | |
| `ollama` | A running Ollama; `OLLAMA_BASE_URL` optional (default `http://localhost:11434`) | Fully offline. |
| `openai_compatible` | `LLM_BASE_URL` and `LLM_API_KEY` | Groq, OpenRouter, Together, LM Studio and similar. |

Environment variables (all in `.env.example`):

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | One of the providers above |
| `LLM_MODEL` | For `claude_code`: `sonnet`, `opus` or `haiku`. Others: the provider's model id, for example `claude-sonnet-5-5` |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | Keys for those providers |
| `LLM_BASE_URL`, `LLM_API_KEY` | For `openai_compatible` |
| `OLLAMA_BASE_URL` | For a non-default Ollama address |

Example `.env` for a client deployment:

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-5-5
ANTHROPIC_API_KEY=<the client's key>
```

**In the UI.** The sidebar has an LLM service section: pick the service, model and base URL, and
paste an API key. The key is kept for that browser session only. Leave it blank to use the
server's key from `.env`.

**Per-agent override.** Give one agent, or the supervisor, a different model in
`config/agents.yaml`:

```yaml
agents:
  content_drafting:
    # ...
    model: { provider: anthropic, model: claude-sonnet-5-5 }
```

The same `model:` line works under `supervisor:`. Agents without it use the global default.

Be aware of these trade-offs:

- `claude_code` costs about 3-5 seconds per LLM step, and a full supervisor run takes many steps.
  It runs on the developer's own Claude Code subscription. Use it for local demos only.
- Client deployments should use `anthropic` with the client's own API key.
- Small local Ollama models route multi-step requests less reliably.

## Tools and free-first defaults

Each specialist uses tools listed in `config/agents.yaml`. The defaults need no API key.

| Tool | What it does | Providers |
|---|---|---|
| `web_search` | Web and news search | `ddgs` (default, free), `tavily` (`TAVILY_API_KEY`), `brave` (`BRAVE_API_KEY`) |
| `read_page` | Returns a page's main text | `scrapling` (default, free, local; stealth browser for blocked or JavaScript pages), `trafilatura` (free, plain HTTP), `jina` (free tier), `firecrawl` (`FIRECRAWL_API_KEY`) |
| `current_date` | Today's date (UTC), to judge how recent a source is | none |

Swap a provider in `config/settings.yaml`:

```yaml
tools:
  web_search:
    provider: tavily
  read_page:
    provider: jina
```

Or set it in `.env`, which overrides the default in `settings.yaml`:

```bash
SEARCH_PROVIDER=tavily
TAVILY_API_KEY=<key>
READER_PROVIDER=jina
```

`.env.example` also lists `JINA_API_KEY`, used by the `jina` reader.

Trade-offs:

- DuckDuckGo (`ddgs`) is free but can rate-limit. If it does, switch on sample data mode or use
  `tavily`.
- `scrapling` first fetches with a real-browser fingerprint (about 1s). Only if the page is
  blocked or renders client-side (e.g. TikTok profiles) does it open its stealth browser (about
  10s). The browser needs a one-time `uv run scrapling install`; set `READER_BROWSER=false` to
  skip it. Sites behind enterprise bot protection (Forbes, for example) still refuse every free
  reader.
- `trafilatura` is plain HTTP: fastest, but it cannot render JavaScript.
- `read_page` refuses loopback, private and link-local addresses, because the LLM chooses the
  URLs.

## Sample data mode

Sample data mode (mock mode) makes `web_search` and `read_page` answer from files in
`data/mock/` instead of the network. Use it when the venue's internet is unreliable or the
search API fails mid-pitch.

- The LLM still runs for real and reasons over the sample data.
- Every mock answer starts with `[mock data]`.
- Everything is fictional: the client brand Aurelle Skin, competitors Kiraleaf and Nuvéa Labs,
  and all sources use the reserved `.example` domain. See [data/mock/README.md](data/mock/README.md)
  for the full cast and how to add entries.

Turn it on any of these ways:

| Way | How |
|---|---|
| UI | Sample data mode toggle in the sidebar |
| CLI | `--mock` on `agent` or `ask` |
| Environment | `MOCK_MODE=true` in `.env` (default in `settings.yaml` is `false`) |

## How to add a tool

A tool is a plain function plus a name, description and pydantic argument model. Modules in
`tools/` are imported automatically. This example adds a `keyword_count` tool for SEO checks.

1. Create `tools/keyword_count.py`:

   ```python
   from pydantic import BaseModel, Field

   from tools.registry import register_tool


   class KeywordCountArgs(BaseModel):
       text: str = Field(description="The text to scan.")
       keyword: str = Field(description="Word or phrase to count, case-insensitive.")


   @register_tool(
       "keyword_count",
       "Count how often a keyword or phrase appears in a text. Use it to check keyword use on a page.",
       KeywordCountArgs,
   )
   def keyword_count(text: str, keyword: str, *, options: dict) -> str:
       count = text.lower().count(keyword.lower())
       words = max(len(text.split()), 1)
       density = round(100 * count * len(keyword.split()) / words, max(options.get("decimals", 1), 0))
       return f"'{keyword}' appears {count} times in {words} words ({density}%)."
   ```

   The function receives the model's arguments as keyword arguments plus `options`, the tool's
   block from `settings.yaml`. It must return a string. If it raises, the agent receives
   `TOOL_ERROR (<name>): <Type>: <message>` instead of the run crashing. Read
   `options.get("provider")` if you want the tool to support swappable providers.

2. Add its options to `config/settings.yaml` under `tools:`. The block is optional; without it
   `options` is `{}`.

   ```yaml
   tools:
     keyword_count:
       decimals: 1
   ```

3. Give it to an agent in `config/agents.yaml`:

   ```yaml
   trend_monitoring:
     tools: [web_search, read_page, current_date, keyword_count]
   ```

4. Test it:

   ```bash
   uv run python cli.py tools
   uv run python cli.py tool keyword_count '{"text": "Barrier serum for barrier repair. A serum.", "keyword": "barrier"}'
   ```

   Expected output: `'barrier' appears 2 times in 7 words (28.6%).`

## How to add a fourth agent

The supervisor and the three existing agents need no edits. The supervisor reads its roster from
the agent registry and `config/agents.yaml`, and the UI lists whatever agents are enabled, so a
new agent shows up in both automatically.

This tutorial adds `seo_audit`, which audits one page for SEO issues. It reuses `web_search` and
`read_page` (and `keyword_count` if you added it above).

1. **Module.** Create `agents/seo_audit.py`:

   ```python
   """seo_audit: audits one page for on-page SEO issues and writes a short, cited report."""

   from agents.base import build_react_agent
   from agents.registry import register_agent


   @register_agent("seo_audit")
   def build(name, cfg, app):
       return build_react_agent(name, cfg, app)
   ```

   The name in `@register_agent` must match the key in `agents.yaml`.

2. **Prompt.** Create `config/prompts/seo_audit.md`. Model it on
   `config/prompts/competitor_research.md`: say who the agent is, how to work, what the output
   sections are, and what to do when a tool returns `TOOL_ERROR`. Keep the "state only facts
   that appear in tool results" rule.

3. **Config.** Add a block under `agents:` in `config/agents.yaml`:

   ```yaml
   seo_audit:
     enabled: true
     description: >-
       Audits one web page (by URL) for on-page SEO: title, headings, keyword use and content
       gaps, with fixes ranked by impact. Use for "audit this page", "why isn't this page
       ranking", "SEO review of <URL>".
     prompt: prompts/seo_audit.md
     tools: [web_search, read_page]   # add keyword_count if you built it above
     max_iterations: 8
   ```

   **`description` is what the supervisor routes on.** Write it as the kind of requests to send
   to this agent. A vague description sends the wrong requests, or none. Optional keys: `model`
   (per-agent LLM) and `enabled: false` to switch the agent off.

4. **Run it alone:**

   ```bash
   uv run python cli.py agent seo_audit "Audit aurelle.example for SEO" --mock
   ```

5. **Run it through the supervisor:**

   ```bash
   uv run python cli.py ask "Audit aurelle.example for SEO, then draft 2 LinkedIn posts that fix the biggest gap." --mock
   ```

   The trace should show the supervisor delegating to `seo_audit`, then to `content_drafting`.

6. **Add a test.** Copy `tests/test_competitor_research.py` to `tests/test_seo_audit.py` and
   change the names. It has four parts to keep:
   - registration: the agent is in `available_agents()` and `build_agent` binds exactly the
     tools listed in its `agents.yaml` block;
   - prompt: it loads, and names every output section you require;
   - a scripted run: a fake model makes one tool call and then answers, so the test runs offline
     in mock mode;
   - a `@pytest.mark.live` test, skipped unless `RUN_LIVE=1`.

   Run it with `uv run pytest -q tests/test_seo_audit.py`.

## Docker

```bash
docker compose up --build
```

Open http://localhost:8501.

`claude_code` cannot run in the container, because the Claude Code CLI login lives on the host.
The container therefore defaults to Ollama running on the host machine
(`OLLAMA_BASE_URL` defaults to `http://host.docker.internal:11434`, model `qwen3.5:9b`).

Switch the container to another provider in `.env`:

```bash
DOCKER_LLM_PROVIDER=anthropic
DOCKER_LLM_MODEL=claude-sonnet-5-5
ANTHROPIC_API_KEY=<key>
```

Or pick a provider and paste a key in the UI sidebar (kept for that browser session only).
`MOCK_MODE` from `.env` is passed through to the container.

## Tests

```bash
uv run pytest -q                 # offline; no LLM or network
RUN_LIVE=1 uv run pytest -q      # also runs tests marked live (real LLM and network)
```

## Project layout

```
config/settings.yaml    Global settings: mock_mode, default llm, per-tool provider options
config/agents.yaml      Supervisor and specialists: enabled, description, model, tools, prompt
config/prompts/*.md     One system prompt per agent, plus supervisor routing and synthesis
core/config.py          Loads YAML, expands ${VAR:-default} from env/.env, validates
core/models.py          get_model(spec) for every provider, including ClaudeCodeChat
core/supervisor.py      LangGraph supervisor: route, run specialist, repeat, then answer
core/trace.py           Turns graph events into the visible trace steps
agents/registry.py      @register_agent and agent discovery
agents/base.py          Shared ReAct loop used by every specialist
agents/*.py             One module per specialist
tools/registry.py       @register_tool, build_tools with TOOL_ERROR wrapping
tools/*.py              web_search, read_page, current_date; mock.py serves sample data
data/mock/              Fictional brand world for sample data mode
examples/*.yaml         Example requests for the CLI
ui/app.py               Streamlit UI
tests/                  Offline tests; live ones need RUN_LIVE=1
notes/progress.md       Build log and decisions
Dockerfile, docker-compose.yml
```

Conventions for upgrading: models, tools, prompts and enabled agents live in YAML, not Python.
Secrets come from the environment only; add a new key to `.env.example`, never to YAML.

## Troubleshooting

**`claude` not found, or not logged in.** `uv run python cli.py ping` fails when using
`claude_code`. Install the Claude Code CLI, run `claude` once and log in, then ping again. Or set
`LLM_PROVIDER` to another provider and add its key.

**`provider 'anthropic' needs ANTHROPIC_API_KEY`.** The selected provider has no key. Add the
key to `.env`, paste it in the UI sidebar, or set `LLM_PROVIDER=claude_code`. The same error
format appears for `openai` (`OPENAI_API_KEY`) and `openai_compatible` (`LLM_API_KEY`;
`LLM_BASE_URL` is also required).

**DuckDuckGo rate limiting.** Searches return `TOOL_ERROR` or thin results. Switch on Sample
data mode, or set `SEARCH_PROVIDER=tavily` with `TAVILY_API_KEY`.

**Slow runs on `claude_code`.** Each LLM step is a separate CLI call of about 3-5 seconds, and
multi-agent requests take many steps. Use sample data mode to remove search delays, run a single
specialist with `cli.py agent`, or switch to `anthropic`.
