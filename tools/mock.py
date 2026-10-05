"""Mock mode: tools answer from data/mock/ fixtures instead of the network.

Fixtures describe a FICTIONAL brand world (see data/mock/README.md). Every mock answer starts
with MOCK_LABEL so the trace and the agent both know it is sample data.
"""

import json
import re
import unicodedata
from functools import lru_cache

from core.config import ROOT

MOCK_DIR = ROOT / "data" / "mock"
MOCK_LABEL = "[mock data]"


@lru_cache(maxsize=None)
def _load(name: str):
    return json.loads((MOCK_DIR / name).read_text(encoding="utf-8"))


def _words(text: str) -> set[str]:
    # Strip accents first so "Nuvéa" matches "nuvea" instead of splitting into "nuv" + "a".
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return set(re.findall(r"[a-z0-9]+", plain.lower()))


def _search(query: str, kind: str = "web", **_) -> str:
    from tools.web_search import _format

    q = _words(query)
    entries = [e for e in _load("search.json") if e.get("kind", "any") in ("any", kind)]
    # Best keyword overlap wins; an entry with no keywords is the catch-all.
    best = max(entries, key=lambda e: (len(q & _words(" ".join(e["match"]))), not e["match"]))
    return f"{MOCK_LABEL}\n" + _format(best["results"])


def _page(url: str, **_) -> str:
    pages = _load("pages.json")
    key = url.rstrip("/")
    if key not in pages:
        raise LookupError(f"no cached sample page for {url}; cached: {sorted(pages)}")
    return f"{MOCK_LABEL}\n" + pages[key]


HANDLERS = {"web_search": _search, "read_page": _page}


def mock_result(tool_name: str, kwargs: dict) -> str | None:
    """Fixture answer for this call, or None if the tool has no mock (it then runs for real)."""
    handler = HANDLERS.get(tool_name)
    return handler(**kwargs) if handler else None
