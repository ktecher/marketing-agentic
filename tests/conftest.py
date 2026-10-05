import pytest


@pytest.fixture(autouse=True)
def _no_real_dotenv(monkeypatch):
    """Tests must never read the developer's real .env (its keys and provider choices would
    leak in and make results depend on the machine). Tests set what they need explicitly."""
    monkeypatch.setattr("core.config.load_dotenv", lambda *a, **k: False)
