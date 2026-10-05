"""Briefing room: a director hands the team a brief and watches the agents work it, live.

Run: uv run streamlit run ui/app.py
"""

import html
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

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


# ── network view: the agency as a living cell ────────────────────────────────────────────────

NET_COLOURS = {"competitor_research": "#2BB5B9", "content_drafting": "#C27BD6", "trend_monitoring": "#E2A743"}
NET_W, NET_H = 1000, 500
NUCLEUS = (500, 232)
NET_WORDS = {"idle": "standing by", "active": "working", "done": "done", "failed": "failed"}
NUCLEUS_WORDS = {"thinking": "thinking", "waiting": "waiting on the team", "final": "answer ready",
                 "failed": "hit a problem", "idle": "standing by"}


def glow_colour(name: str) -> str:
    """The agent's hue, brightened for the dark stage."""
    if name in NET_COLOURS:
        return NET_COLOURS[name]
    c = colour(name).lstrip("#")
    return "#" + "".join(f"{round(int(c[i:i + 2], 16) * 0.65 + 255 * 0.35):02X}" for i in (0, 2, 4))


def svg_text(s, n: int) -> str:
    # The SVG goes through st.markdown (st.html strips <svg>): one line, escaped, no `$` for math.
    return esc(preview(s, n)).replace("$", "&#36;")


def agent_caption(steps: list[Step], agent: str) -> str:
    caption = "Standing by"
    for s in steps:
        if s.kind == "route" and s.data.get("agent") == agent:
            caption = f"Briefed: {s.data.get('task') or 'new task'}"
        elif s.agent != agent:
            continue
        elif s.kind == "tool_call":
            tool, args = s.data.get("tool", ""), s.data.get("args") or {}
            if tool == "web_search":
                caption = f"Searching: {args.get('query') or compact_args(args)}"
            elif tool == "read_page":
                host = urlparse(str(args.get("url", ""))).netloc.removeprefix("www.")
                caption = f"Reading {host or 'a page'}"
            else:
                caption = f"Using {human(tool).lower()}"
        elif s.kind == "thought":
            caption = s.detail or s.title
        elif s.kind == "agent_answer":
            caption = "Done"
        elif s.kind == "tool_error":
            caption = "Hit an error, trying another way"
        elif s.kind == "error":
            caption = "Hit an error"
    return caption


def supervisor_caption(steps: list[Step], status: dict[str, str], running: bool) -> str:
    if any(s.kind == "final" for s in steps):
        return "Answer ready"
    if status["supervisor"] == "failed":
        return "Hit a problem"
    if not running:
        return "Stopped" if steps else "Waiting for a brief"
    busy = [human(a) for a in TEAM[1:] if status[a] == "active"]
    if status["supervisor"] != "active" and busy:
        return f"Delegating to {busy[0]}" if len(busy) == 1 else f"Delegating to {len(busy)} specialists in parallel"
    last_route = next((s for s in reversed(steps) if s.kind == "route"), None)
    if last_route and not last_route.data.get("agent") and steps[-1] is last_route:
        return "Writing the answer"
    return "Thinking…"


def _bokeh() -> str:
    """Depth: a few blurred specks and a faint layer of tiny dots. Fixed seed, so it never jumps."""
    rnd = random.Random(7)
    big = "".join(f'<circle cx="{rnd.uniform(20, NET_W - 20):.0f}" cy="{rnd.uniform(15, NET_H - 15):.0f}" '
                  f'r="{rnd.uniform(4, 22):.0f}" opacity="{rnd.uniform(0.05, 0.14):.2f}"/>' for _ in range(16))
    dust = "".join(f'<circle cx="{rnd.uniform(0, NET_W):.0f}" cy="{rnd.uniform(0, NET_H):.0f}" '
                   f'r="{rnd.uniform(0.8, 1.8):.1f}" opacity="{rnd.uniform(0.08, 0.22):.2f}"/>' for _ in range(70))
    return f'<g class="bokeh" filter="url(#cell-soft)">{big}</g><g class="dust">{dust}</g>'


BOKEH = _bokeh()
CELL_TEXTURE = (  # bubbly surface for every sphere: rings of varied size, tiled
    '<pattern id="cell-tex" width="46" height="46" patternUnits="userSpaceOnUse">'
    '<g fill="#FFFFFF" fill-opacity="0.07" stroke="#FFFFFF" stroke-opacity="0.32" stroke-width="1">'
    + "".join(f'<circle cx="{x}" cy="{y}" r="{r}"/>' for x, y, r in
              [(8, 9, 6), (27, 6, 4), (40, 20, 5), (18, 26, 8), (5, 38, 4), (33, 37, 6), (44, 44, 3), (24, 44, 2.5)])
    + '</g></pattern>'
)


def sphere_defs(gid: str, c: str) -> str:
    """A translucent glassy sphere tinted with c: highlight off-centre top-left, darker body, lit rim."""
    return (f'<radialGradient id="{gid}" cx="36%" cy="32%" r="72%">'
            f'<stop offset="0" stop-color="#FFFFFF" stop-opacity="0.9"/>'
            f'<stop offset="0.22" stop-color="{c}" stop-opacity="0.8"/>'
            f'<stop offset="0.72" stop-color="{c}" stop-opacity="0.32"/>'
            f'<stop offset="0.93" stop-color="{c}" stop-opacity="0.5"/>'
            f'<stop offset="1" stop-color="#FFFFFF" stop-opacity="0.55"/></radialGradient>')


def sphere(cls: str, gid: str, x: float, y: float, r: float) -> str:
    return (f'<circle class="{cls}-glow" cx="{x:.1f}" cy="{y:.1f}" r="{r * 1.45:.1f}" filter="url(#cell-blur)"/>'
            f'<circle class="{cls}-core" cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="url(#{gid})"/>'
            f'<circle class="sphere-tex" cx="{x:.1f}" cy="{y:.1f}" r="{r - 1}" fill="url(#cell-tex)"/>'
            f'<ellipse class="sphere-shine" cx="{x - r * 0.34:.1f}" cy="{y - r * 0.4:.1f}" rx="{r * 0.34:.1f}" '
            f'ry="{r * 0.2:.1f}" transform="rotate(-35 {x - r * 0.34:.1f} {y - r * 0.4:.1f})"/>'
            f'<circle class="sphere-rim" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>')


def tube_paths(cx: float, cy: float, x: float, y: float, r0: float) -> list[str]:
    """Two lobes from nucleus to agent: they leave the nucleus a little apart, bow out like petals,
    and meet at the agent. Both run nucleus -> agent, so one dash animation means 'outward'."""
    dx, dy = x - cx, y - cy
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    paths = []
    for sign in (1, -1):
        a = math.atan2(uy, ux) + sign * 0.42
        sx, sy = cx + r0 * math.cos(a), cy + r0 * math.sin(a)
        bow = sign * length * 0.26
        c1 = (sx + dx * 0.3 - uy * bow, sy + dy * 0.3 + ux * bow)
        c2 = (cx + dx * 0.75 - uy * bow * 0.7, cy + dy * 0.75 + ux * bow * 0.7)
        paths.append(f"M{sx:.1f},{sy:.1f} C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {x:.1f},{y:.1f}")
    return paths


def network_svg(steps: list[Step], running: bool) -> str:
    """The run as living cells: the supervisor nucleus in the middle, one cell per specialist on an
    orbit, joined by twin tubes whose particles flow out (working) and back (answer delivered).
    Pure: no Streamlit calls. All motion lives in theme.css, keyed on the classes set here."""
    status = team_status(steps, running)
    final = any(s.kind == "final" for s in steps)
    sup = status["supervisor"]
    nucleus_state = ("final" if final else "failed" if sup == "failed" else "thinking" if sup == "active"
                     else "waiting" if running else "idle")
    cx, cy = NUCLEUS
    rx, ry = 360, 172
    nr, ar = 64, 36  # nucleus and agent cell radii
    team = TEAM[1:]
    last_answer = {s.agent: i for i, s in enumerate(steps) if s.kind == "agent_answer"}

    defs = [CELL_TEXTURE, sphere_defs("sph-nucleus", "#BCD3FF")]
    tubes, nodes, aria = [], [], [f"Supervisor {NUCLEUS_WORDS[nucleus_state]}"]
    sup_caption = supervisor_caption(steps, status, running)
    # Narrow screens: the canvas keeps the motion, and this list (hidden on wide ones) carries the words.
    legend = [f'<div class="leg-row {sup}" style="--c:#C9D6F2"><span class="leg-dot"></span>'
              f'<b>Supervisor</b><span>{svg_text(sup_caption, 60)}</span></div>']
    for i, a in enumerate(team):
        ang = -math.pi / 2 + 2 * math.pi * i / max(len(team), 1)
        x, y = cx + rx * math.cos(ang), cy + ry * math.sin(ang)
        st_a = status[a]
        # Results flowing back: the agent's answer is among the last few steps.
        back = " returning" if st_a == "done" and last_answer.get(a, -9) >= len(steps) - 3 else ""
        c = glow_colour(a)
        gid = f"sph-{slug(a)}"
        defs.append(sphere_defs(gid, c))
        lobes = "".join(f'<path class="tube-wall" d="{d}"/><path class="tube-core" d="{d}"/>'
                        f'<path class="tube-flow" d="{d}"/>' for d in tube_paths(cx, cy, x, y, nr * 0.8))
        tubes.append(f'<g class="axon {st_a}{back}" color="{c}">{lobes}</g>')

        searches = sum(1 for s in steps if s.agent == a and s.kind == "tool_call" and s.data.get("tool") == "web_search")
        pages = sum(1 for s in steps if s.agent == a and s.kind == "tool_call" and s.data.get("tool") == "read_page")
        if abs(math.cos(ang)) < 0.35:  # top or bottom of the orbit: labels beside the cell
            tx, ty, anchor = x + ar + 20, y - 18, "start"
        else:  # labels under the cell
            tx, ty, anchor = x, y + ar + 30, "middle"
        nodes.append(
            f'<g class="node {st_a}" color="{c}">{sphere("node", gid, x, y, ar)}'
            f'<text class="node-initials" x="{x:.1f}" y="{y:.1f}">{esc(initials(a))}</text>'
            f'<text class="node-name" x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}">{esc(human(a))}</text>'
            f'<text class="node-caption" x="{tx:.1f}" y="{ty + 25:.1f}" text-anchor="{anchor}">'
            f'{svg_text(agent_caption(steps, a), 38)}</text>'
            f'<text class="node-count" x="{tx:.1f}" y="{ty + 49:.1f}" text-anchor="{anchor}">'
            f'🔍 {searches} · 📄 {pages}</text></g>'
        )
        aria.append(f"{human(a)} {NET_WORDS[st_a]}")
        legend.append(f'<div class="leg-row {st_a}" style="--c:{c}"><span class="leg-dot"></span>'
                      f'<b>{esc(human(a))}</b><span>{svg_text(agent_caption(steps, a), 60)}</span></div>')

    nucleus = (
        f'<g class="nucleus {nucleus_state}" color="#BCD3FF">'
        f'<circle class="nucleus-halo" cx="{cx}" cy="{cy}" r="{nr * 2}" fill="url(#cell-halo)"/>'
        f'<circle class="membrane" cx="{cx}" cy="{cy}" r="{nr + 26}"/>'
        f'<circle class="membrane membrane-inner" cx="{cx}" cy="{cy}" r="{nr + 16}"/>'
        f'{sphere("nucleus", "sph-nucleus", cx, cy, nr)}'
        f'<text class="nucleus-name" x="{cx}" y="{cy + nr + 56}">Supervisor</text>'
        f'<text class="nucleus-caption" x="{cx}" y="{cy + nr + 81}">{svg_text(sup_caption, 42)}</text></g>'
    )
    defs_html = (
        '<defs>'
        '<radialGradient id="cell-bg" cx="50%" cy="46%" r="70%"><stop offset="0" stop-color="#26315C"/>'
        '<stop offset="1" stop-color="#141A30"/></radialGradient>'
        '<radialGradient id="cell-halo"><stop offset="0.4" stop-color="#9FC3FF" stop-opacity="0.5"/>'
        '<stop offset="1" stop-color="#9FC3FF" stop-opacity="0"/></radialGradient>'
        '<filter id="cell-blur" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="8"/></filter>'
        '<filter id="cell-soft" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="5"/></filter>'
        f'{"".join(defs)}</defs>'
    )
    return (
        f'<div class="cell-stage"><svg class="cell{" running" if running else ""}" viewBox="0 0 {NET_W} {NET_H}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" aria-label="{esc("; ".join(aria))}">{defs_html}'
        f'<rect width="{NET_W}" height="{NET_H}" fill="url(#cell-bg)"/>{BOKEH}'
        f'{"".join(tubes)}{nucleus}{"".join(nodes)}</svg>'
        f'<div class="cell-legend" aria-hidden="true">{"".join(legend)}</div></div>'
    )


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


def render_answer(slot, request: str, steps: list[Step], running: bool, active: list[str] | None = None,
                  where: str = "") -> None:
    final = next((s for s in reversed(steps) if s.kind == "final"), None)
    with slot.container():
        if final:
            with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
                st.subheader("The answer", anchor=False, width="content")
                st.download_button(
                    "Download as Markdown", f"# Brief\n\n{request}\n\n---\n\n{final.detail}\n",
                    file_name="briefing-room-answer.md", mime="text/markdown", on_click="ignore",
                    key=f"download{where}_{len(steps)}_{running}",
                )
            st.markdown(final.detail or "_The team returned an empty answer._")
        elif running:
            names = [human(a) for a in active or []]
            if names:
                who = " and ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 1 else names[0]
                line = f"{who} {'are' if len(names) > 1 else 'is'} working on it."
            elif any(s.kind == "agent_answer" for s in steps):
                line = "The supervisor is putting the answer together."
            else:
                line = "The supervisor is reading the brief."
            st.html(f'<div class="answer-wait">{esc(line)} The answer appears here when the team finishes.</div>')
        else:
            err = next((s for s in reversed(steps) if s.kind == "error"), None)
            what = esc(err.title) if err else "The run ended without an answer."
            advice = esc(error_advice(err)) if err else "Brief the team again."
            st.html(f'<div class="answer-error"><strong>No answer this time.</strong> {what}<br>{advice}</div>')


def views():
    """Network | Timeline. st.tabs, not a segmented control: a widget rerun would stop a live run,
    while a tab switch stays in the browser."""
    return st.tabs(["Network", "Timeline"], key="view", default="Network")


def safe_steps(app, request: str, history: list[dict] | None = None):
    """The live run, with every failure turned into an error Step so the page never crashes."""
    ended = False
    try:
        for step in supervisor.run(app, request, history=history) if history else supervisor.run(app, request):
            ended = ended or step.kind == "final"
            yield step
    except Exception as e:  # e.g. ModelConfigError while building the agents
        ended = True
        yield Step("error", "supervisor", f"Run failed: {type(e).__name__}", str(e))
    if not ended:
        yield Step("error", "supervisor", "The team stopped without a final answer", "")


def show_run(request: str, timed: list[tuple[Step, float]], strip, live_iter=None) -> None:
    """Render a stored run, or stream a live one (live_iter) into the same layout. Both views are
    filled on every step, so switching tabs is client-side and never interrupts a live run."""
    status_line = st.empty()
    network, timeline = views()
    net_slot = network.empty()
    net_answer = network.container(key="answer_net").empty()
    with timeline:
        left, right = st.columns([2, 3], gap="large")
        with left:
            st.subheader("How the team worked", anchor=False)
    answer_slot = right.container(key="answer").empty()

    def refresh(running: bool, elapsed: float):
        steps = [s for s, _ in timed]
        status = team_status(steps, running)
        strip.html(strip_html(status))
        n = f"{len(steps)} step{'s' if len(steps) != 1 else ''}"
        line = f"Working for {clock(elapsed)}, {n} so far" if running else f"Finished in {clock(elapsed)}, {n}"
        status_line.html(f'<p class="run-status">{line}</p>')
        net_slot.markdown(network_svg(steps, running), unsafe_allow_html=True)
        active = [a for a in TEAM[1:] if status[a] == "active"]  # parallel rounds: several at once
        render_answer(net_answer, request, steps, running, active, where="_net")
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

def make_app():
    app = load_config()
    app.settings.mock_mode = mock
    app.settings.llm = choices["spec"]
    app.settings.secrets = choices["secrets"]  # session-only; never os.environ
    app.settings.tools.setdefault("web_search", {})["provider"] = ",".join(choices["search"])
    app.settings.tools.setdefault("read_page", {})["provider"] = choices["reader"]
    return app


def final_text(turn: dict) -> str:
    return next((s.detail for s, _ in reversed(turn["timed"]) if s.kind == "final"), "")


def run_turn(text: str) -> None:
    """Run one brief or follow-up live, with the earlier turns as context, and keep it."""
    thread = st.session_state.setdefault("thread", [])
    history = [{"request": t["request"], "answer": final_text(t)} for t in thread if final_text(t)]
    timed: list[tuple[Step, float]] = []
    show_run(text, timed, strip, live_iter=safe_steps(make_app(), text, history))
    thread.append({"request": text, "timed": timed})


def new_brief() -> None:
    st.session_state["thread"] = []
    st.session_state["request"] = ""


thread = st.session_state.setdefault("thread", [])
followup = st.chat_input("Ask a follow-up about these results", key="followup") if thread else None

if clicked:
    st.session_state["thread"] = thread = []  # a new brief starts a new conversation
    run_turn(request.strip())
    st.rerun()  # redraw as a conversation, so the follow-up box appears straight away
elif thread:
    for n, turn in enumerate(thread[:-1] if not followup else thread, 1):
        with st.expander(f"Brief {n}: {preview(turn['request'], 90)}", expanded=False):
            st.markdown(final_text(turn) or "_No answer this time._")
    if followup and followup.strip():
        st.html(f'<p class="followup-q"><strong>Follow-up:</strong> {esc(followup.strip())}</p>')
        run_turn(followup.strip())
    else:
        last = thread[-1]
        show_run(last["request"], last["timed"], strip)
    st.button("New brief", key="new_brief", on_click=new_brief, icon=":material/restart_alt:",
              help="Start over: the next brief won't see this conversation.")
else:
    strip.html(strip_html(team_status([], running=False)))
    network, timeline = views()
    network.markdown(network_svg([], running=False), unsafe_allow_html=True)
    timeline.html('<p class="quiet">Every step the team takes appears here once you brief them.</p>')
    reqs = examples(mock)
    if reqs:
        st.subheader("Try one of these", anchor=False)
        with st.container(key="examples", horizontal=True, gap="small"):
            for n, text in enumerate(reqs):
                st.button(text, key=f"ex_{n}", on_click=pick_example, args=(text,))
