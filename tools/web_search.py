"""web_search — free default: DuckDuckGo via `ddgs` (no key). Paid swaps: tavily, brave."""

import os
import re
import threading
import time
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from tools.registry import register_tool

Recency = Literal["any", "day", "week", "month", "year"]
_AD_URLS = ("bing.com/aclick", "duckduckgo.com/y.js")  # sponsored results ddgs passes through

# ddgs scrapes public search pages, and the engines refuse bursts: agents batch several searches
# into one step, which ran 6 at once and failed ~25% of them (measured). Queue them instead and
# retry a refusal after a pause.
# ponytail: per-process limit; several app replicas on one IP would need a shared limiter.
_DDGS_SLOTS = threading.Semaphore(2)
_DDGS_ATTEMPTS = 3


class SearchArgs(BaseModel):
    query: str = Field(description="Search query, e.g. 'Acme Corp pricing plans'")
    kind: Literal["web", "news"] = Field("web", description="'news' for recent articles with dates")
    recency: Recency = Field("any", description="Limit results to this recent period")


def _format(results: list[dict]) -> str:
    if not results:
        return "No results."
    lines = []
    for i, r in enumerate(results, 1):
        date = r.get("date") or ""
        date = date[:10] if re.match(r"\d{4}-\d{2}-\d{2}T", date) else date  # trim ISO timestamps only ("4 weeks ago" stays whole)
        meta = " · ".join(x for x in (r.get("source"), date) if x)
        title = r["title"] if len(r["title"]) <= 120 else r["title"][:117] + "..."  # engines sometimes merge carousel titles
        lines.append(f"{i}. {title}\n   {r['url']}" + (f"\n   [{meta}]" if meta else ""))
        if r.get("snippet"):
            lines.append(f"   {r['snippet'][:400]}")
    return "\n".join(lines)


def _ddgs(query: str, kind: str, recency: Recency, n: int, backends: dict | None = None) -> list[dict]:
    from ddgs.exceptions import DDGSException

    backend = (backends or {}).get(kind, "auto")
    for attempt in range(1, _DDGS_ATTEMPTS + 1):
        try:
            with _DDGS_SLOTS:
                return _ddgs_once(query, kind, recency, n, backend)
        except DDGSException as e:
            if attempt == _DDGS_ATTEMPTS:
                raise RuntimeError(
                    f"free search engines returned nothing after {attempt} tries ({e}). "
                    "They may be throttling; try a broader query, or switch on sample data mode."
                ) from e
            time.sleep(1.5 * attempt)
    return []


def _ddgs_once(query: str, kind: str, recency: Recency, n: int, backend: str = "auto") -> list[dict]:
    from ddgs import DDGS

    kwargs = {"max_results": n, "backend": backend}
    if recency != "any":
        kwargs["timelimit"] = recency[0]  # d / w / m / y
    with DDGS() as d:
        if kind == "news":
            return [
                {"title": r["title"], "url": r["url"], "snippet": r.get("body"), "date": r.get("date"), "source": r.get("source")}
                for r in d.news(query, **kwargs)
            ]
        return [
            {"title": r["title"], "url": r["href"], "snippet": r.get("body")}
            for r in d.text(query, **kwargs)
            if not any(ad in r["href"] for ad in _AD_URLS)
        ]


def _tavily(query: str, kind: str, recency: Recency, n: int, **_) -> list[dict]:
    body = {"query": query, "max_results": n, "topic": "news" if kind == "news" else "general"}
    if recency != "any":
        body["time_range"] = recency
    r = httpx.post(
        "https://api.tavily.com/search",
        json=body,
        headers={"Authorization": f"Bearer {_key('TAVILY_API_KEY')}"},
        timeout=30,
    )
    r.raise_for_status()
    return [
        {"title": x["title"], "url": x["url"], "snippet": x.get("content"), "date": x.get("published_date")}
        for x in r.json().get("results", [])
    ]


def _brave(query: str, kind: str, recency: Recency, n: int, **_) -> list[dict]:
    endpoint = "news" if kind == "news" else "web"
    params = {"q": query, "count": n}
    if recency != "any":
        params["freshness"] = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}[recency]
    r = httpx.get(
        f"https://api.search.brave.com/res/v1/{endpoint}/search",
        params=params,
        headers={"X-Subscription-Token": _key("BRAVE_API_KEY"), "Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    items = data.get("results", []) if kind == "news" else data.get("web", {}).get("results", [])
    return [
        {"title": x["title"], "url": x["url"], "snippet": x.get("description"), "date": x.get("page_age") or x.get("age")}
        for x in items
    ]


def _serper(query: str, kind: str, recency: Recency, n: int, **_) -> list[dict]:
    """Google results via Serper (serper.dev): 2,500 free searches, then prepaid credits."""
    body = {"q": query, "num": n}
    if recency != "any":
        body["tbs"] = f"qdr:{recency[0]}"  # d / w / m / y
    r = httpx.post(
        f"https://google.serper.dev/{'news' if kind == 'news' else 'search'}",
        json=body,
        headers={"X-API-KEY": _key("SERPER_API_KEY"), "Content-Type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    items = r.json().get("news" if kind == "news" else "organic", [])
    return [
        {"title": x["title"], "url": x["link"], "snippet": x.get("snippet"), "date": x.get("date"), "source": x.get("source")}
        for x in items
    ]


def _key(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set (needed by this search provider). Add it to .env or use provider: ddgs")
    return value


PROVIDERS = {"ddgs": _ddgs, "serper": _serper, "tavily": _tavily, "brave": _brave}
LABELS = {"ddgs": "Bing/Yahoo (free)", "serper": "Google", "tavily": "Tavily", "brave": "Brave"}


def _search_one(provider: str, query: str, kind: str, recency: Recency, n: int, options: dict) -> tuple[list[dict], str]:
    """Results from one provider, plus a note for the agent ('' when nothing to say)."""
    if provider != "ddgs":
        return PROVIDERS[provider](query, kind, recency, n), ""
    engines = {"web": options.get("web_engines", "auto"), "news": options.get("news_engines", "auto")}
    try:
        return PROVIDERS["ddgs"](query, kind, recency, n, backends=engines), ""
    except RuntimeError:
        if recency == "any":
            raise
        # Date filters make the scraped engines refuse far more often (measured). Fall back to an
        # unfiltered search and say so, so the agent checks dates itself instead of assuming them.
        results = PROVIDERS["ddgs"](query, kind, "any", n, backends=engines)
        return results, (f"[No results with the '{recency}' filter; these are from ANY date. Check each "
                         f"date before treating it as recent.]")


@register_tool(
    "web_search",
    "Search the web. Returns titles, URLs and snippets. Use kind='news' plus recency for recent "
    "coverage and mentions. Follow up with read_page on the most relevant URLs.",
    SearchArgs,
)
def web_search(query: str, kind: str = "web", recency: Recency = "any", *, options: dict) -> str:
    # `provider` may list several ("ddgs,serper"): they are searched in parallel and merged.
    providers = [p.strip() for p in str(options.get("provider", "ddgs")).split(",") if p.strip()]
    unknown = [p for p in providers if p not in PROVIDERS]
    if unknown or not providers:
        raise ValueError(f"unknown web_search provider {unknown or providers}; choose from {sorted(PROVIDERS)}")
    n = int(options.get("max_results", 6))
    if len(providers) == 1:
        results, note = _search_one(providers[0], query, kind, recency, n, options)
        return "\n".join(filter(None, [note, _format(results)]))

    from concurrent.futures import ThreadPoolExecutor

    def attempt(p):
        try:
            return p, *_search_one(p, query, kind, recency, n, options), None
        except Exception as e:
            return p, [], "", e

    with ThreadPoolExecutor(len(providers)) as pool:
        outcomes = list(pool.map(attempt, providers))
    if all(err for *_, err in outcomes):
        raise RuntimeError("; ".join(f"{LABELS[p]}: {type(e).__name__}: {e}" for p, _, _, e in outcomes))

    # Interleave the sources (best result of each first), dropping URLs already listed.
    merged, seen = [], set()
    for rank in range(n):
        for p, results, _, _ in outcomes:
            if rank < len(results) and (url := results[rank]["url"].rstrip("/")) not in seen:
                seen.add(url)
                merged.append({**results[rank], "source": results[rank].get("source") or LABELS[p]})
    notes = [note for _, _, note, _ in outcomes if note]
    notes += [f"[{LABELS[p]} search failed: {type(e).__name__}: {e}]" for p, _, _, e in outcomes if e]
    return "\n".join([*notes, _format(merged)])
