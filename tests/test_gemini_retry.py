from types import SimpleNamespace

import pytest

from gemclip import gemini as gemini_mod


class _Models:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item)


def _client(outcomes):
    return SimpleNamespace(models=_Models(outcomes))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(gemini_mod.time, "sleep", lambda _s: None)


def test_transient_error_is_retried_once():
    client = _client([RuntimeError("503 UNAVAILABLE"), '{"ok": 1}'])
    response = gemini_mod.generate_with_retry(client, model="m", contents="c", config=None)
    assert response.text == '{"ok": 1}'
    assert client.models.calls == 2


def test_gives_up_after_one_retry():
    client = _client([RuntimeError("429 RESOURCE_EXHAUSTED"), RuntimeError("503 UNAVAILABLE")])
    with pytest.raises(RuntimeError, match="503"):
        gemini_mod.generate_with_retry(client, model="m", contents="c", config=None)
    assert client.models.calls == 2


def test_permanent_error_is_not_retried():
    client = _client([RuntimeError("403 PERMISSION_DENIED")])
    with pytest.raises(RuntimeError, match="403"):
        gemini_mod.generate_with_retry(client, model="m", contents="c", config=None)
    assert client.models.calls == 1


def test_transient_classification():
    assert gemini_mod.is_transient(RuntimeError("504 deadline exceeded"))
    assert gemini_mod.is_transient(RuntimeError("request timed out"))
    assert not gemini_mod.is_transient(RuntimeError("400 INVALID_ARGUMENT"))


def test_make_client_sets_http_timeout(monkeypatch):
    seen = {}

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            seen.update(api_key=api_key, http_options=http_options)

    monkeypatch.setattr(gemini_mod.genai, "Client", FakeClient)
    gemini_mod.make_client("kunci")
    assert seen["api_key"] == "kunci"
    assert seen["http_options"].timeout == gemini_mod.GEMINI_TIMEOUT_MS
