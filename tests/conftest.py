import pytest

from gemclip import pipeline


@pytest.fixture(autouse=True)
def no_youtube_captions(monkeypatch):
    """Tanpa jaringan, caption YouTube dimatikan di semua tes. Tes caption memasang fungsi tiruannya sendiri."""
    monkeypatch.setattr(pipeline, "fetch_youtube_captions", lambda *args, **kwargs: None)
