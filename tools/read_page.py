"""read_page — free default: trafilatura (local extraction, no key). Swaps: jina (free tier), firecrawl (paid)."""

import ipaddress
import os
import socket
from urllib.parse import urlparse

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


def _trafilatura(url: str) -> str:
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
    text = trafilatura.extract(html, url=url, include_tables=True, with_metadata=True)
    if not text:
        raise RuntimeError("no readable main text (page may need JavaScript; try provider: jina)")
    return text


def _jina(url: str) -> str:
    headers = {"Accept": "text/plain"}
    if key := os.environ.get("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    r = httpx.get(f"https://r.jina.ai/{url}", headers=headers, timeout=60)
    r.raise_for_status()
    return r.text


def _firecrawl(url: str) -> str:
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


PROVIDERS = {"trafilatura": _trafilatura, "jina": _jina, "firecrawl": _firecrawl}


@register_tool(
    "read_page",
    "Read a web page and return its main text (navigation and ads stripped). Use on URLs from web_search.",
    PageArgs,
)
def read_page(url: str, *, options: dict) -> str:
    _check_public(url)
    provider = options.get("provider", "trafilatura")
    if provider not in PROVIDERS:
        raise ValueError(f"unknown read_page provider '{provider}'; choose from {sorted(PROVIDERS)}")
    text = PROVIDERS[provider](url).strip()
    limit = int(options.get("max_chars", 8000))
    return text if len(text) <= limit else text[:limit] + f"\n\n[truncated at {limit} of {len(text)} chars]"
