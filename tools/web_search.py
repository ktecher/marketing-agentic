"""web_search — free default: DuckDuckGo via `ddgs` (no key). Paid swaps: tavily, brave."""

import os
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
        meta = " · ".join(x for x in (r.get("source"), (r.get("date") or "")[:10]) if x)
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


def _key(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set (needed by this search provider). Add it to .env or use provider: ddgs")
    return value


PROVIDERS = {"ddgs": _ddgs, "tavily": _tavily, "brave": _brave}


@register_tool(
    "web_search",
    "Search the web. Returns titles, URLs and snippets. Use kind='news' plus recency for recent "
    "coverage and mentions. Follow up with read_page on the most relevant URLs.",
    SearchArgs,
)
def web_search(query: str, kind: str = "web", recency: Recency = "any", *, options: dict) -> str:
    provider = options.get("provider", "ddgs")
    if provider not in PROVIDERS:
        raise ValueError(f"unknown web_search provider '{provider}'; choose from {sorted(PROVIDERS)}")
    n = int(options.get("max_results", 6))
    if provider != "ddgs":
        return _format(PROVIDERS[provider](query, kind, recency, n))
    engines = {"web": options.get("web_engines", "auto"), "news": options.get("news_engines", "auto")}
    try:
        return _format(PROVIDERS["ddgs"](query, kind, recency, n, backends=engines))
    except RuntimeError:
        if recency == "any":
            raise
        # Date filters make the scraped engines refuse far more often (measured). Fall back to an
        # unfiltered search and say so, so the agent checks dates itself instead of assuming them.
        results = PROVIDERS["ddgs"](query, kind, "any", n, backends=engines)
        return (f"[No results with the '{recency}' filter; these are from ANY date. Check each date "
                f"before treating it as recent.]\n" + _format(results))
