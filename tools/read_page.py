"""read_page — free default: scrapling (browser-grade fetch, stealth browser for blocked or
JavaScript pages). Other options: trafilatura (plain HTTP), jina (free tier), firecrawl (paid)."""

import ipaddress
import os
import socket
from functools import lru_cache
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, Field

from tools.registry import register_tool

_MAX_BYTES = 5_000_000  # stop reading huge responses; the text is truncated to max_chars anyway
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128 Safari/537.36"


class PageArgs(BaseModel):
    url: str = Field(description="Full http(s) URL of the page to read")


def _check_public(url: str) -> None:
    """Refuse loopback/private/link-local targets: the LLM chooses URLs, and a page can tell it to.
    ponytail: resolve-then-fetch leaves a DNS-rebinding window; pin the resolved IP if this is exposed publicly."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"not an http(s) URL: {url!r}")
    for info in socket.getaddrinfo(parsed.hostname, parsed.port or 443):
        ip = ipaddress.ip_address(info[4][0])
        if not all(a.is_global for a in _with_embedded_v4(ip)):
            raise ValueError(f"refusing non-public address {ip} for {parsed.hostname}")


_EMBEDS_V4 = [ipaddress.ip_network(n) for n in ("::/96", "64:ff9b::/96")]  # IPv4-compatible, NAT64


def _with_embedded_v4(ip) -> list:
    """An IPv6 address can carry an IPv4 one that is_global ignores (::127.0.0.1, 64:ff9b::7f00:1)."""
    found = [ip]
    if ip.version == 6:
        found += [a for a in (ip.ipv4_mapped, ip.sixtofour) if a]
        if any(ip in net for net in _EMBEDS_V4):
            found.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    return found


def _extract(html: str, url: str) -> str | None:
    import trafilatura

    return trafilatura.extract(html, url=url, include_tables=True, with_metadata=True)


@lru_cache(maxsize=256)
def _host_is_public(url: str) -> bool:
    try:
        _check_public(url)
        return True
    except (ValueError, OSError):
        return False


def _scrapling(url: str, options: dict) -> str:
    """Fast path: curl_cffi impersonating Chrome (beats most bot checks, no browser needed).
    Escalates to Scrapling's stealth browser when the page is blocked or renders client-side."""
    from scrapling.fetchers import Fetcher

    # Follow redirects by hand so every hop passes the public-address check.
    status, html, final = 0, "", url
    for _ in range(6):
        r = Fetcher.get(final, impersonate="chrome", stealthy_headers=True, follow_redirects=False, timeout=20)
        location = r.headers.get("location") or r.headers.get("Location")
        if 300 <= r.status < 400 and location:
            final = urljoin(final, location)
            _check_public(final)
            continue
        status, html = r.status, r.html_content or ""
        break
    text = _extract(html, final) if status < 400 else None
    if text and len(text) >= 400:
        return text
    if str(options.get("browser", True)).strip().lower() in ("false", "0", "no", "off"):  # env values arrive as strings
        raise RuntimeError(f"HTTP {status}; page blocked or needs JavaScript (enable read_page.browser)")

    from scrapling.fetchers import StealthyFetcher

    def guard(page):
        # Every request the browser makes (redirects, JS navigation, sub-resources) must be public.
        page.route("**/*", lambda route: route.continue_() if _host_is_public(route.request.url) else route.abort())

    try:
        page = StealthyFetcher.fetch(final, headless=True, network_idle=True, timeout=45000, page_setup=guard)
    except Exception as e:  # e.g. browsers not installed (`uv run scrapling install`)
        raise RuntimeError(f"HTTP {status} on fast fetch; browser fallback failed: {type(e).__name__}: {e}") from e
    if page.status >= 400:
        raise RuntimeError(f"HTTP {page.status}: the site blocks automated readers")
    text = _extract(page.html_content or "", final) or " ".join(str(page.get_all_text(ignore_tags=("script", "style"))).split())
    if not text:
        raise RuntimeError("page rendered but had no readable text")
    return text


def _trafilatura(url: str, options: dict) -> str:
    import trafilatura

    # The hook re-checks every redirect hop, not just the first URL.
    hooks = {"request": [lambda req: _check_public(str(req.url))]}
    with httpx.Client(headers={"User-Agent": _UA}, timeout=20, follow_redirects=True, event_hooks=hooks) as c:
        with c.stream("GET", url) as r:
            r.raise_for_status()
            body = bytearray()
            for chunk in r.iter_bytes():
                body += chunk
                if len(body) >= _MAX_BYTES:
                    break
            html = bytes(body).decode(r.encoding or "utf-8", errors="replace")
    text = _extract(html, url)
    if not text:
        raise RuntimeError("no readable main text (page may need JavaScript; try provider: scrapling)")
    return text


def _jina(url: str, options: dict) -> str:
    headers = {"Accept": "text/plain"}
    if key := os.environ.get("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    r = httpx.get(f"https://r.jina.ai/{url}", headers=headers, timeout=60)
    r.raise_for_status()
    return r.text


def _firecrawl(url: str, options: dict) -> str:
    key = os.environ.get("FIRECRAWL_API_KEY")
    if not key:
        raise RuntimeError("FIRECRAWL_API_KEY is not set. Add it to .env or use provider: trafilatura")
    r = httpx.post(
        "https://api.firecrawl.dev/v1/scrape",
        json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
        headers={"Authorization": f"Bearer {key}"},
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["data"]["markdown"]


PROVIDERS = {"scrapling": _scrapling, "trafilatura": _trafilatura, "jina": _jina, "firecrawl": _firecrawl}


@register_tool(
    "read_page",
    "Read a web page and return its main text (navigation and ads stripped). Use on URLs from web_search.",
    PageArgs,
)
def read_page(url: str, *, options: dict) -> str:
    _check_public(url)
    provider = options.get("provider", "scrapling")
    if provider not in PROVIDERS:
        raise ValueError(f"unknown read_page provider '{provider}'; choose from {sorted(PROVIDERS)}")
    text = PROVIDERS[provider](url, options).strip()
    limit = int(options.get("max_chars", 8000))
    return text if len(text) <= limit else text[:limit] + f"\n\n[truncated at {limit} of {len(text)} chars]"
