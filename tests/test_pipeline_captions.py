import pytest

from gemclip import pipeline
from gemclip.cache import CAPTION_SOURCE, transcript_path, video_cache_dir
from gemclip.config import Settings
from gemclip.models import Segment, VideoInfo, Word

URL = "https://youtu.be/dQw4w9WgXcQ"


def _settings(tmp_path, use_captions=True):
    return Settings(
        gemini_api_key="test-key",
        gemini_model="gemini-test",
        work_dir=tmp_path,
        clip_min_sec=5,
        clip_max_sec=60,
        default_clip_count=2,
        max_height=720,
        use_youtube_captions=use_captions,
    )


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    """Tanpa unduhan dan tanpa Gemini. Caption dan transkripsi dicatat jumlah panggilannya."""
    calls = {"captions": 0, "transcribe": 0}
    caption_words = [Word(0.0, 0.5, "kalimat"), Word(0.5, 1.0, "dari"), Word(1.0, 1.5, "caption")]
    caption_segments = [Segment(0.0, 1.5, "kalimat dari caption")]

    def fake_download(url, out_dir, max_height):
        return VideoInfo(path=tmp_path / "x.mp4", title="Video", duration=20.0)

    def fake_captions(url, subs_dir, lo=0.0, hi=float("inf")):
        calls["captions"] += 1
        return caption_segments, caption_words

    def fake_transcribe(**kwargs):
        calls["transcribe"] += 1
        return [Segment(0.0, 1.0, "dari gemini")], [Word(0.0, 1.0, "dari")]

    monkeypatch.setattr(pipeline, "download_video", fake_download)
    monkeypatch.setattr(pipeline, "fetch_youtube_captions", fake_captions)
    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "find_moments", lambda **kwargs: [])
    monkeypatch.setattr(pipeline, "check_ffmpeg", lambda: None)
    return calls


def _run(settings, job_name):
    return pipeline.run_job(
        URL,
        clip_count=1,
        settings=settings,
        job_dir=settings.work_dir / job_name,
        progress=lambda s, f: None,
    )


def test_captions_are_used_before_gemini_and_cached(tmp_path, stubs):
    with pytest.raises(RuntimeError, match="tidak menemukan momen"):
        _run(_settings(tmp_path), "job1")

    assert stubs["captions"] == 1
    assert stubs["transcribe"] == 0, "Gemini tidak boleh dipanggil jika caption tersedia"

    cache_dir = video_cache_dir(tmp_path / "cache", "dQw4w9WgXcQ")
    assert transcript_path(cache_dir, CAPTION_SOURCE, 0.0, 20.0).exists()


def test_second_job_reuses_caption_cache(tmp_path, stubs):
    settings = _settings(tmp_path)
    for name in ("job1", "job2"):
        with pytest.raises(RuntimeError, match="tidak menemukan momen"):
            _run(settings, name)
    assert stubs["captions"] == 1, "caption hanya diambil sekali per video"


def test_captions_can_be_switched_off(tmp_path, stubs):
    with pytest.raises(RuntimeError, match="tidak menemukan momen"):
        _run(_settings(tmp_path, use_captions=False), "job3")
    assert stubs["captions"] == 0
    assert stubs["transcribe"] == 1


def test_missing_captions_fall_back_to_gemini(tmp_path, stubs, monkeypatch):
    monkeypatch.setattr(pipeline, "fetch_youtube_captions", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="tidak menemukan momen"):
        _run(_settings(tmp_path), "job4")
    assert stubs["transcribe"] == 1
