"""Briefing room: a director hands the team a brief and watches the agents work it, live.

Run: uv run streamlit run ui/app.py
"""

import html
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `streamlit run ui/app.py` puts ui/ on sys.path, not the project root
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
import yaml  # noqa: E402

import core.supervisor as supervisor  # noqa: E402  (module attribute, so tests can patch run)
from pydantic import SecretStr  # noqa: E402

from core.config import ModelSpec, load_config, save_env, ui_can_save  # noqa: E402
from core.trace import Step  # noqa: E402

AGENT_COLOURS = {
    "supervisor": "#1B2340",
    "competitor_research": "#0F7C80",
    "content_drafting": "#8A3FA0",
    "trend_monitoring": "#B7791F",
}
FALLBACK_COLOURS = ["#2F5FD0", "#B2471B", "#4D7C0F", "#B0245E", "#0B6E99", "#6B4FBB"]
TOOL_ICONS = {"web_search": "🔍", "read_page": "📄", "current_date": "📅"}
PROVIDERS = {
    "claude_code": "Claude Code (local, no key)",
    "anthropic": "Anthropic API",
    "openai": "OpenAI API",
    "ollama": "Ollama (local, offline)",
    "openai_compatible": "OpenAI-compatible (Groq, OpenRouter, LM Studio)",
}
DEFAULT_MODELS = {"claude_code": "sonnet", "anthropic": "claude-sonnet-5-5", "openai": "gpt-5.1",
                  "ollama": "qwen3.5:9b", "openai_compatible": ""}
LLM_KEY_VAR = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "openai_compatible": "LLM_API_KEY"}
SEARCH_SOURCES = {"ddgs": "Bing + Yahoo (free)", "serper": "Google (Serper)", "tavily": "Tavily",
                  "brave": "Brave Search"}
READERS = {"scrapling": "Scrapling (free)", "trafilatura": "Plain HTTP (free, fastest)",
           "jina": "Jina Reader", "firecrawl": "Firecrawl"}
# (env var, label, where to get it, the source/reader/LLM service that needs it, required?)
API_KEYS = [
    ("SERPER_API_KEY", "Google (Serper)", "https://serper.dev/api-keys", "serper", True),
    ("TAVILY_API_KEY", "Tavily", "https://app.tavily.com", "tavily", True),
    ("BRAVE_API_KEY", "Brave Search", "https://api-dashboard.search.brave.com", "brave", True),
    ("FIRECRAWL_API_KEY", "Firecrawl", "https://www.firecrawl.dev/app/api-keys", "firecrawl", True),
    ("JINA_API_KEY", "Jina Reader (optional, raises limits)", "https://jina.ai/reader", "jina", False),
    ("ANTHROPIC_API_KEY", "Anthropic", "https://console.anthropic.com/settings/keys", "anthropic", True),
    ("OPENAI_API_KEY", "OpenAI", "https://platform.openai.com/api-keys", "openai", True),
    ("LLM_API_KEY", "OpenAI-compatible service", "", "openai_compatible", True),
]
NEEDS_URL = {"openai_compatible"}  # ollama uses the server's OLLAMA_BASE_URL: a visitor-typed URL would make the server call any host
MOCK_PREFIX = "[mock data]"

st.set_page_config(page_title="Briefing room", page_icon="🗂️", layout="wide")
base = load_config()
TEAM = ["supervisor", *base.enabled_agents]
SUPERVISOR_ROLE = "Reads the brief, picks a specialist and says why, then puts the answer together."


# ── helpers ──────────────────────────────────────────────────────────────────────────────────

def esc(s) -> str:
    return html.escape(str(s))


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower())


def colour(name: str) -> str:
    if name in AGENT_COLOURS:
        return AGENT_COLOURS[name]
    extras = [a for a in TEAM if a not in AGENT_COLOURS]
    i = extras.index(name) if name in extras else len(extras)
    return FALLBACK_COLOURS[i % len(FALLBACK_COLOURS)]


def human(name: str) -> str:
    return name.replace("_", " ").strip().capitalize() or "Agent"


def initials(name: str) -> str:
    words = human(name).split()
    return "".join(w[0] for w in words[:2]).upper()


def clock(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s}s" if s < 60 else f"{s // 60}m {s % 60:02d}s"


def compact_args(args: dict) -> str:
    vals = [str(v) for v in (args or {}).values()]
    text = vals[0] if len(vals) == 1 else ", ".join(f"{k}: {v}" for k, v in (args or {}).items())
    return text if len(text) <= 90 else text[:89] + "…"


def preview(text: str, n: int = 150) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def css() -> str:
    agent_vars = "".join(
        f'.ag-{slug(a)}, [class*="-ag-{slug(a)}-x"] {{ --agent: {colour(a)}; }}\n' for a in TEAM
    )
    return f"<style>{(ROOT / 'ui' / 'theme.css').read_text(encoding='utf-8')}\n{agent_vars}</style>"


def team_status(steps: list[Step], running: bool) -> dict[str, str]:
    """idle | active | done | failed for each team member, from the steps so far.
    Specialists in the same round work in parallel, so several can be active at once (and one
    specialist can hold two tasks): count open tasks per agent instead of tracking one."""
    status = {a: "idle" for a in TEAM}
    open_tasks = {a: 0 for a in TEAM}
    if running:
        status["supervisor"] = "active"

    def refresh_supervisor():
        busy = any(open_tasks[a] for a in TEAM if a != "supervisor")
        status["supervisor"] = "idle" if busy else "active"

    for s in steps:
        if s.kind == "route":
            target = s.data.get("agent")
            if target in status:
                open_tasks[target] += 1
                status[target] = "active"
            refresh_supervisor()
        elif s.kind in ("agent_answer", "error") and s.agent in status and s.agent != "supervisor":
            open_tasks[s.agent] = max(0, open_tasks[s.agent] - 1)
            if s.kind == "error":
                status[s.agent] = "failed"
            elif not open_tasks[s.agent] and status[s.agent] != "failed":
                status[s.agent] = "done"
            refresh_supervisor()
        elif s.kind == "error":  # supervisor-level failure
            status["supervisor"] = "failed"
        elif s.kind == "final":
            status = {a: ("done" if v == "active" else v) for a, v in status.items()}
            status["supervisor"] = "done"
    return status


STATUS_WORDS = {"idle": "Standing by", "active": "Working", "done": "Done", "failed": "Failed"}


def strip_html(status: dict[str, str]) -> str:
    chips = []
    for i, a in enumerate(TEAM):
        desc = SUPERVISOR_ROLE if a == "supervisor" else base.agents[a].description
        chips.append(
            f'<div class="chip {status[a]} ag-{slug(a)}" title="{esc(desc)}">'
            f'<span class="avatar" aria-hidden="true">{esc(initials(a))}</span>'
            f'<span><div class="chip-name">{esc(human(a))}</div>'
            f'<div class="chip-status">{STATUS_WORDS[status[a]]}</div></span></div>'
        )
        if i == 0:
            chips.append('<span class="team-gap" aria-hidden="true"></span>')
    return f'<div class="team" role="status" aria-label="Team activity">{"".join(chips)}</div>'


def error_advice(step: Step) -> str:
    if step.agent != "supervisor":
        return "The supervisor carries on with the rest of the brief."
    text = f"{step.title} {step.detail}".lower()
    if "modelconfigerror" in text or "api key" in text or "api_key" in text or "needs" in text:
        return "Check the LLM service settings in the sidebar (model, base URL, API key), then brief the team again."
    return "Brief the team again. If the venue's internet is unreliable, switch on Sample data mode in the sidebar."


# ── trace rendering ──────────────────────────────────────────────────────────────────────────

def render_step(i: int, step: Step, t: float) -> None:
    a = step.agent
    who = human(a)
    detail = title = note = ""
    cls = step.kind
    expander = None

    if step.kind == "route":
        target = step.data.get("agent")
        head = f"Delegating to {human(target)}" if target else step.title
        task = step.data.get("task", "")
        reason = step.data.get("reasoning") or step.detail
        body = f'<div class="step-title">{esc(head)}</div>'
        if task:
            body += f'<div class="step-detail">Task: {esc(preview(task, 220))}</div>'
        if reason:
            body += f'<div class="step-detail">{esc(reason)}</div>'
        title = f'<div class="bubble">{body}</div>'
    elif step.kind == "thought":
        title = f'<div class="step-title">{esc(preview(step.detail or step.title, 220))}</div>'
    elif step.kind == "tool_call":
        tool = step.data.get("tool") or step.title.split("(")[0]
        icon = TOOL_ICONS.get(tool, "🛠️")
        title = (f'<div class="step-title">{icon} {esc(human(tool))} '
                 f'<q>{esc(compact_args(step.data.get("args", {})))}</q></div>')
    elif step.kind == "tool_result":
        tool = step.data.get("tool", "tool")
        text = step.detail or step.title
        mock = text.lstrip().startswith(MOCK_PREFIX)
        shown = text.lstrip()[len(MOCK_PREFIX):] if mock else text
        badge = '<span class="badge">sample data</span>' if mock else ""
        title = f'<div class="step-title">{esc(human(tool))} returned{badge}</div>'
        detail = f'<div class="step-detail">{esc(preview(shown))}</div>'
        expander = ("Full result", text)
    elif step.kind == "tool_error":
        cls = "tool_error bad"
        tool = step.data.get("tool", "tool")
        msg = (step.detail or step.title).removeprefix(f"TOOL_ERROR ({tool}): ")
        title = f'<div class="step-title">{esc(human(tool))} failed</div>'
        detail = f'<div class="step-detail">{esc(preview(msg, 240))}</div>'
        note = '<div class="step-note">The agent will try another way.</div>'
    elif step.kind == "agent_answer":
        title = f'<div class="step-title">{esc(who)} finished</div>'
        expander = ("Read the answer", step.detail or step.title)
    elif step.kind == "error":
        cls = "error bad"
        title = f'<div class="step-title">{esc(step.title)}</div>'
        detail = f'<div class="step-detail">{esc(preview(step.detail, 400))}</div>'
        note = f'<div class="step-note">{esc(error_advice(step))}</div>'
    elif step.kind == "final":
        title = '<div class="step-title">Final answer ready</div>'
    else:
        title = f'<div class="step-title">{esc(step.title)}</div>'

    with st.container(key=f"s{i}-k-{step.kind}-ag-{slug(a)}-x"):
        st.html(
            f'<div class="step {cls}"><span class="step-num">{i}</span>'
            f'<span class="avatar" aria-hidden="true">{esc(initials(a))}</span>'
            f'<div class="step-main"><div class="step-meta"><span class="step-who">{esc(who)}</span>'
            f'<span class="step-time">{clock(t)}</span></div>{title}{detail}{note}</div></div>'
        )
        if expander:
            with st.expander(expander[0]):
                st.markdown(expander[1])


def render_answer(slot, request: str, steps: list[Step], running: bool, active: str | None = None) -> None:
    final = next((s for s in reversed(steps) if s.kind == "final"), None)
    with slot.container():
        if final:
            with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
                st.subheader("The answer", anchor=False, width="content")
                st.download_button(
                    "Download as Markdown", f"# Brief\n\n{request}\n\n---\n\n{final.detail}\n",
                    file_name="briefing-room-answer.md", mime="text/markdown", on_click="ignore",
                    key=f"download_{len(steps)}_{running}",
                )
            st.markdown(final.detail or "_The team returned an empty answer._")
        elif running:
            who = human(active) if active else "The team"
            st.html(f'<div class="answer-wait">{esc(who)} is working on it. The answer appears here '
                    'when the team finishes.</div>')
        else:
            err = next((s for s in reversed(steps) if s.kind == "error"), None)
            what = esc(err.title) if err else "The run ended without an answer."
            advice = esc(error_advice(err)) if err else "Brief the team again."
            st.html(f'<div class="answer-error"><strong>No answer this time.</strong> {what}<br>{advice}</div>')


def safe_steps(app, request: str):
    """The live run, with every failure turned into an error Step so the page never crashes."""
    ended = False
    try:
        for step in supervisor.run(app, request):
            ended = ended or step.kind == "final"
            yield step
    except Exception as e:  # e.g. ModelConfigError while building the agents
        ended = True
        yield Step("error", "supervisor", f"Run failed: {type(e).__name__}", str(e))
    if not ended:
        yield Step("error", "supervisor", "The team stopped without a final answer", "")


def show_run(request: str, timed: list[tuple[Step, float]], strip, live_iter=None) -> None:
    """Render a stored run, or stream a live one (live_iter) into the same layout."""
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader("How the team worked", anchor=False)
        status_line = st.empty()
    answer_slot = right.container(key="answer").empty()

    def refresh(running: bool, elapsed: float):
        steps = [s for s, _ in timed]
        status = team_status(steps, running)
        strip.html(strip_html(status))
        n = f"{len(steps)} step{'s' if len(steps) != 1 else ''}"
        line = f"Working for {clock(elapsed)}, {n} so far" if running else f"Finished in {clock(elapsed)}, {n}"
        status_line.html(f'<p class="run-status">{line}</p>')
        active = next((a for a in TEAM[1:] if status[a] == "active"), None)
        render_answer(answer_slot, request, steps, running, active)

    if live_iter is None:
        with left:
            for i, (s, t) in enumerate(timed, 1):
                render_step(i, s, t)
        refresh(False, timed[-1][1] if timed else 0)
        return

    t0 = time.monotonic()
    refresh(True, 0)
    with left:
        for step in live_iter:
            t = time.monotonic() - t0
            timed.append((step, t))
            render_step(len(timed), step, t)
            refresh(True, t)
    refresh(False, time.monotonic() - t0)


# ── sidebar ──────────────────────────────────────────────────────────────────────────────────

def pick_example(text: str) -> None:
    st.session_state["request"] = text


def default_settings() -> dict:
    """This machine's defaults (config + .env), as the Settings pop-up edits them."""
    llm, tools = base.settings.llm, base.settings.tools
    search = [p.strip() for p in str(tools.get("web_search", {}).get("provider", "ddgs")).split(",")]
    reader = tools.get("read_page", {}).get("provider", "scrapling")
    return {"provider": llm.provider, "model": llm.model, "base_url": llm.base_url or "",
            "search": [p for p in search if p in SEARCH_SOURCES] or ["ddgs"],
            "reader": reader if reader in READERS else "scrapling", "keys": {}}


def current() -> dict:
    """The settings in force for this browser session (the pop-up's last Apply, else defaults).
    Kept outside widget state: widgets inside a dialog are forgotten when it closes."""
    return st.session_state.setdefault("settings", default_settings())


def missing_keys(cfg: dict) -> list[str]:
    in_use = {cfg["provider"], cfg["reader"], *cfg["search"]}
    return [label for var, label, _, service, required in API_KEYS
            if required and service in in_use and not cfg["keys"].get(var) and not os.environ.get(var)]


def summary(cfg: dict) -> str:
    sources = ", ".join(SEARCH_SOURCES[p].replace(" (free)", "") for p in cfg["search"])
    return f"{PROVIDERS[cfg['provider']].split(' (')[0]} · {cfg['model']}  \nSearch: {sources}  \nReader: {READERS[cfg['reader']]}"


@st.dialog("Settings", width="large")
def settings_dialog() -> None:
    cfg = current()
    ss = st.session_state
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("LLM service", anchor=False)
        names = list(PROVIDERS)
        provider = st.selectbox("Service", names, index=names.index(cfg["provider"]), format_func=PROVIDERS.get,
                                key="set_provider")
        model = st.text_input("Model", key=f"set_model_{provider}",
                              value=cfg["model"] if provider == cfg["provider"] else DEFAULT_MODELS[provider],
                              placeholder="Model id from your provider")
        base_url = ""
        if provider in NEEDS_URL:
            base_url = st.text_input("Base URL", key=f"set_url_{provider}", placeholder="https://api.groq.com/openai/v1",
                                     value=cfg["base_url"] if provider == cfg["provider"] else "")
        st.subheader("Search and page reading", anchor=False)
        search = st.multiselect("Search sources", list(SEARCH_SOURCES), default=cfg["search"],
                                format_func=SEARCH_SOURCES.get, key="set_search",
                                help="Several sources are searched at the same time and the results merged.")
        reader = st.selectbox("Page reader", list(READERS), index=list(READERS).index(cfg["reader"]),
                              format_func=READERS.get, key="set_reader")
    with right:
        st.subheader("API keys", anchor=False)
        st.caption("A key typed here is used for this browser session. "
                   "Save as defaults to keep it on this machine.")
        typed = {}
        for var, label, url, service, _ in API_KEYS:
            saved = bool(os.environ.get(var))
            value = st.text_input(label, type="password", key=f"set_key_{var}", value=cfg["keys"].get(var, ""),
                                  placeholder="Saved on this machine" if saved else "Not set",
                                  help=(f"Get one at {url}. " if url else "") + "Type a new key to replace the saved one.")
            if value.strip():
                typed[var] = value.strip()

    chosen = {"provider": provider, "model": model.strip(), "base_url": base_url.strip(),
              "search": search or ["ddgs"], "reader": reader, "keys": typed}
    if gaps := missing_keys(chosen):
        st.warning("Needs a key: " + ", ".join(gaps) + ".")

    apply_col, save_col = st.columns(2)
    if apply_col.button("Apply", type="primary", key="set_apply", use_container_width=True):
        ss["settings"] = chosen
        st.rerun()
    if ui_can_save() and save_col.button("Save as defaults on this machine", key="set_save", use_container_width=True):
        updates = {"LLM_PROVIDER": chosen["provider"], "LLM_MODEL": chosen["model"],
                   "SEARCH_PROVIDER": ",".join(chosen["search"]), "READER_PROVIDER": chosen["reader"], **typed}
        if chosen["provider"] in NEEDS_URL:
            updates["LLM_BASE_URL"] = chosen["base_url"]
        try:
            save_env(updates)
        except (OSError, ValueError) as e:
            st.error(f"Couldn't save: {e}")
            return
        ss["settings"] = {**chosen, "keys": {}}  # saved keys now come from .env
        ss["saved_msg"] = "Saved. These are now the defaults on this machine."
        st.rerun()


def sidebar() -> dict:
    cfg = current()
    with st.sidebar:
        st.header("Settings", anchor=False)
        mock = st.toggle("Sample data mode", value=base.settings.mock_mode, key="mock")
        st.caption("Uses a fictional brand world instead of the live web. "
                   "Use it if the venue's internet is unreliable.")
        st.caption(summary(cfg))
        if st.button("Settings", icon=":material/tune:", key="open_settings", use_container_width=True):
            settings_dialog()
        if gaps := missing_keys(cfg):
            st.warning("Needs a key: " + ", ".join(gaps) + ". Open Settings to add it.")
        if msg := st.session_state.pop("saved_msg", None):
            st.success(msg)

        st.subheader("Team", anchor=False)
        rows = [("supervisor", SUPERVISOR_ROLE)] + [(n, a.description) for n, a in base.enabled_agents.items()]
        st.html('<div class="roster">' + "".join(
            f'<div class="roster-row ag-{slug(n)}"><span class="avatar" aria-hidden="true">{esc(initials(n))}</span>'
            f'<div><div class="roster-name">{esc(human(n))}</div><div class="roster-desc">{esc(d)}</div></div></div>'
            for n, d in rows) + "</div>")

    spec = ModelSpec(provider=cfg["provider"], model=cfg["model"], base_url=cfg["base_url"] or None,
                     temperature=base.settings.llm.temperature,
                     api_key=cfg["keys"].get(LLM_KEY_VAR.get(cfg["provider"], "")) or None)
    return {"mock": mock, "spec": spec, "search": cfg["search"], "reader": cfg["reader"],
            "secrets": {k: SecretStr(v) for k, v in cfg["keys"].items()}}


def examples(mock: bool) -> list[str]:
    try:
        items = yaml.safe_load((ROOT / "examples" / "supervisor.yaml").read_text(encoding="utf-8")) or []
    except OSError:
        return []
    return [e["request"] for e in items if isinstance(e, dict) and bool(e.get("mock")) == mock and e.get("request")]


# ── page ─────────────────────────────────────────────────────────────────────────────────────

st.markdown(css(), unsafe_allow_html=True)
choices = sidebar()
mock = choices["mock"]

st.title("What should the team work on?", anchor=False)
st.html('<p class="lead">Write the brief as you would to your agency. The supervisor hands it to the '
        'right specialists, and you watch every step as they research and write.</p>')
request = st.text_area("Your brief", key="request", height=110, label_visibility="collapsed",
                       placeholder="For example: research our competitor Kiraleaf, then draft three "
                                   "Instagram posts that position Aurelle Skin against them.")
clicked = st.button("Brief the team", type="primary", key="brief")
strip = st.empty()

if clicked and not request.strip():
    st.warning("Write a brief first, or pick one of the examples below.")
    clicked = False

if clicked:
    app = load_config()
    app.settings.mock_mode = mock
    app.settings.llm = choices["spec"]
    app.settings.secrets = choices["secrets"]  # session-only; never os.environ
    app.settings.tools.setdefault("web_search", {})["provider"] = ",".join(choices["search"])
    app.settings.tools.setdefault("read_page", {})["provider"] = choices["reader"]
    timed: list[tuple[Step, float]] = []
    st.session_state["result"] = None
    show_run(request.strip(), timed, strip, live_iter=safe_steps(app, request.strip()))
    st.session_state["result"] = {"request": request.strip(), "timed": timed}
elif st.session_state.get("result"):
    r = st.session_state["result"]
    show_run(r["request"], r["timed"], strip)
else:
    strip.html(strip_html(team_status([], running=False)))
    reqs = examples(mock)
    if reqs:
        st.subheader("Try one of these", anchor=False)
        with st.container(key="examples", horizontal=True, gap="small"):
            for n, text in enumerate(reqs):
                st.button(text, key=f"ex_{n}", on_click=pick_example, args=(text,))
