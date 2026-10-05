"""Phase 2: web_search / read_page providers, safety checks, mock mode. Offline."""

import json
import os
import socket

import pytest

from core.config import Settings
from tools import mock, read_page, web_search
from tools.registry import build_tools


def _settings(**kw):
    return Settings(llm={"provider": "claude_code", "model": "sonnet"}, **kw)


def test_format_lists_title_url_meta_snippet():
    out = web_search._format([{"title": "T", "url": "https://a.example", "snippet": "S", "date": "2026-09-01T10:00", "source": "Src"}])
    assert out == "1. T\n   https://a.example\n   [Src · 2026-09-01]\n   S"
    assert web_search._format([]) == "No results."


def test_ddgs_drops_sponsored_results(monkeypatch):
    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def text(self, q, **kw):
            return [{"title": "Ad", "href": "https://www.bing.com/aclick?ld=x", "body": ""},
                    {"title": "Real", "href": "https://real.example", "body": "b"}]

    import ddgs
    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    assert [r["title"] for r in web_search._ddgs("q", "web", "any", 5)] == ["Real"]


def test_unknown_provider_and_missing_key_are_tool_errors(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    [bad] = build_tools(["web_search"], _settings(tools={"web_search": {"provider": "bing"}}))
    assert bad.invoke({"query": "x"}).startswith("TOOL_ERROR (web_search): ValueError: unknown web_search provider")
    [tav] = build_tools(["web_search"], _settings(tools={"web_search": {"provider": "tavily"}}))
    assert "TAVILY_API_KEY is not set" in tav.invoke({"query": "x"})


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1",
                                "::ffff:127.0.0.1", "::127.0.0.1", "64:ff9b::7f00:1", "2002:7f00:1::1"])
def test_read_page_refuses_non_public_targets(monkeypatch, ip):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, (ip, 0))])
    with pytest.raises(ValueError, match="non-public"):
        read_page._check_public("https://innocent-looking.example/page")


def test_read_page_allows_public_and_truncates(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("93.184.216.34", 0))])
    monkeypatch.setitem(read_page.PROVIDERS, "scrapling", lambda url, options: "x" * 50)
    out = read_page.read_page("https://site.example", options={"max_chars": 10})
    assert out.startswith("x" * 10) and "[truncated at 10 of 50 chars]" in out


def test_read_page_rejects_non_http():
    [tool] = build_tools(["read_page"], _settings())
    assert "not an http(s) URL" in tool.invoke({"url": "file:///etc/passwd"})


@pytest.fixture
def fixtures(tmp_path, monkeypatch):
    (tmp_path / "search.json").write_text(json.dumps([
        {"match": [], "kind": "any", "results": [{"title": "General", "url": "https://g.example"}]},
        {"match": ["kiraleaf", "pricing"], "kind": "web", "results": [{"title": "Kira pricing", "url": "https://kiraleaf.example/pricing"}]},
        {"match": ["kiraleaf"], "kind": "news", "results": [{"title": "Kira news", "url": "https://n.example", "date": "2026-09-01", "source": "BW"}]},
    ]))
    (tmp_path / "pages.json").write_text(json.dumps({"https://kiraleaf.example/pricing": "Plans: $18"}))
    monkeypatch.setattr(mock, "MOCK_DIR", tmp_path)
    mock._load.cache_clear()
    yield
    mock._load.cache_clear()


def test_mock_mode_never_touches_network(fixtures, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network called in mock mode")

    monkeypatch.setitem(web_search.PROVIDERS, "ddgs", boom)
    monkeypatch.setitem(read_page.PROVIDERS, "trafilatura", boom)
    monkeypatch.setitem(read_page.PROVIDERS, "scrapling", boom)
    search, page = build_tools(["web_search", "read_page"], _settings(mock_mode=True))

    assert "Kira pricing" in search.invoke({"query": "Kiraleaf pricing plans"})
    assert "Kira news" in search.invoke({"query": "Kiraleaf", "kind": "news"})
    assert "General" in search.invoke({"query": "totally unrelated"})
    assert page.invoke({"url": "https://kiraleaf.example/pricing/"}) == f"{mock.MOCK_LABEL}\nPlans: $18"
    assert "no cached sample page" in page.invoke({"url": "https://other.example"})


def test_mock_mode_off_uses_real_provider(fixtures, monkeypatch):
    monkeypatch.setitem(web_search.PROVIDERS, "ddgs", lambda *a: [{"title": "LIVE", "url": "https://l.example"}])
    [search] = build_tools(["web_search"], _settings(mock_mode=False))
    assert "LIVE" in search.invoke({"query": "Kiraleaf pricing"})


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("RUN_LIVE"), reason="set RUN_LIVE=1")
def test_live_search_and_read():
    search, page = build_tools(["web_search", "read_page"], _settings(tools={"web_search": {"provider": "ddgs"}}))
    assert "http" in search.invoke({"query": "content marketing"})
    assert len(page.invoke({"url": "https://en.wikipedia.org/wiki/Content_marketing"})) > 500


def test_scrapling_checks_every_redirect_hop(monkeypatch):
    """A public URL that redirects to an internal one must be refused before it is fetched."""
    import scrapling.fetchers as sf

    class R:
        def __init__(self, status, headers, html=""):
            self.status, self.headers, self.html_content = status, headers, html

    fetched = []

    def fake_get(url, **kw):
        fetched.append(url)
        return R(302, {"location": "http://169.254.169.254/latest/meta-data"})

    monkeypatch.setattr(sf.Fetcher, "get", staticmethod(fake_get))
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: [
        (None, None, None, None, (host if host[0].isdigit() else "93.184.216.34", 0))])
    with pytest.raises(ValueError, match="non-public"):
        read_page._scrapling("https://innocent.example/go", {})
    assert fetched == ["https://innocent.example/go"]  # the internal hop was never requested


def test_scrapling_escalates_to_browser_when_blocked(monkeypatch):
    import scrapling.fetchers as sf

    class R:
        def __init__(self, status, html):
            self.status, self.headers, self.html_content = status, {}, html
        def get_all_text(self, **kw):
            return "rendered text " * 50

    monkeypatch.setattr(sf.Fetcher, "get", staticmethod(lambda url, **kw: R(403, "<html>blocked</html>")))
    used = {}

    def fake_browser(url, **kw):
        used["guard"] = kw.get("page_setup")
        return R(200, "<html><body></body></html>")

    monkeypatch.setattr(sf.StealthyFetcher, "fetch", staticmethod(fake_browser))
    assert "rendered text" in read_page._scrapling("https://site.example", {})
    assert callable(used["guard"])  # browser requests go through the public-address guard
    with pytest.raises(RuntimeError, match="HTTP 403"):
        read_page._scrapling("https://site.example", {"browser": False})
    with pytest.raises(RuntimeError, match="HTTP 403"):
        read_page._scrapling("https://site.example", {"browser": "false"})  # as it arrives from READER_BROWSER


def test_ddgs_retries_refusals_then_reports_clearly(monkeypatch):
    from ddgs.exceptions import DDGSException
    monkeypatch.setattr(web_search.time, "sleep", lambda s: None)
    calls = []

    def flaky(*a):
        calls.append(1)
        if len(calls) < 3:
            raise DDGSException("No results found.")
        return [{"title": "T", "url": "https://u.example"}]

    monkeypatch.setattr(web_search, "_ddgs_once", flaky)
    assert web_search._ddgs("q", "web", "any", 5)[0]["title"] == "T" and len(calls) == 3

    monkeypatch.setattr(web_search, "_ddgs_once", lambda *a: (_ for _ in ()).throw(DDGSException("No results found.")))
    with pytest.raises(RuntimeError, match="after 3 tries"):
        web_search._ddgs("q", "web", "any", 5)


def test_ddgs_never_runs_more_than_two_at_once(monkeypatch):
    import threading as th
    from concurrent.futures import ThreadPoolExecutor
    live, peak, lock = [0], [0], th.Lock()

    def slow(*a):
        with lock:
            live[0] += 1; peak[0] = max(peak[0], live[0])
        import time; time.sleep(0.05)  # slow on purpose: concurrency only shows if calls overlap
        with lock:
            live[0] -= 1
        return []

    monkeypatch.setattr(web_search, "_ddgs_once", slow)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda _: web_search._ddgs("q", "web", "any", 5), range(6)))
    assert peak[0] == 2
