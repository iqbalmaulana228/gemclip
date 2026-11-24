import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from gemclip import transcribe as transcribe_mod
from gemclip.chunks import extract_audio
from gemclip.transcribe import parse_pieces, split_words, transcribe

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe tidak tersedia",
)


def test_split_words_shares_time_by_word_length():
    words = split_words("satu dua tiga", 0.0, 11.0)  # panjang 4, 3, 4 dari total 11
    assert [w.text for w in words] == ["satu", "dua", "tiga"]
    assert [(w.start, w.end) for w in words] == [(0.0, 4.0), (4.0, 7.0), (7.0, 11.0)]


def test_split_words_handles_zero_length_sentence():
    words = split_words("a b", 5.0, 5.0)
    assert len(words) == 2
    assert all(w.end > w.start >= 5.0 for w in words)


def test_split_words_empty_text():
    assert split_words("   ", 0.0, 1.0) == []


def test_parse_pieces_rejects_invalid_json():
    with pytest.raises(RuntimeError, match="tidak valid"):
        parse_pieces("bukan json")


class _Files:
    def __init__(self):
        self.uploaded: list[tuple[str, str]] = []
        self.deleted: list[str] = []

    def upload(self, file, config):
        self.uploaded.append((file, config.mime_type))
        return SimpleNamespace(name=f"files/a{len(self.uploaded)}", state=SimpleNamespace(name="ACTIVE"))

    def delete(self, name):
        self.deleted.append(name)


class _Models:
    def __init__(self, payloads: list[str]):
        self.payloads = payloads
        self.calls: list[dict] = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text=self.payloads.pop(0))


@pytest.fixture
def fake_gemini(monkeypatch):
    state = {"payloads": [], "clients": [], "extracted": []}

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.files = _Files()
            self.models = _Models(state["payloads"])
            state["clients"].append(self)

    def fake_extract(source, start, duration, dest):
        state["extracted"].append((round(start, 3), round(duration, 3)))
        dest.write_bytes(b"mp3")
        return dest

    monkeypatch.setattr(transcribe_mod.genai, "Client", FakeClient)
    monkeypatch.setattr(transcribe_mod, "extract_audio", fake_extract)
    return state


def _pieces(*items) -> str:
    return json.dumps([{"start_sec": s, "end_sec": e, "text": t} for s, e, t in items])


def test_transcribe_maps_relative_times_to_video_time(tmp_path, fake_gemini):
    fake_gemini["payloads"].append(_pieces((1.0, 4.0, "halo dunia"), (5.0, 9.0, "apa kabar")))

    segments, words = transcribe(
        video_path=tmp_path / "source.mp4",
        api_key="test-key",
        model="gemini-test",
        work_dir=tmp_path,
        duration=600.0,
        clip_range=(100.0, 130.0),
    )

    assert fake_gemini["extracted"] == [(100.0, 30.0)]
    assert [(s.start, s.end, s.text) for s in segments] == [
        (101.0, 104.0, "halo dunia"),
        (105.0, 109.0, "apa kabar"),
    ]
    assert [w.text for w in words] == ["halo", "dunia", "apa", "kabar"]
    assert words[0].start == 101.0

    client = fake_gemini["clients"][0]
    assert client.files.uploaded[0][1] == "audio/mpeg", "audio harus diunggah sebagai mp3"
    assert client.files.deleted == ["files/a1"], "berkas audio di Gemini harus dihapus"
    assert client.models.calls[0]["model"] == "gemini-test"
    assert client.models.calls[0]["config"].response_mime_type == "application/json"
    assert not (tmp_path / "audio_part_01.mp3").exists(), "audio sementara harus dihapus"


def test_transcribe_splits_long_video_into_windows(tmp_path, fake_gemini):
    fake_gemini["payloads"].extend(
        [_pieces((0.0, 1.0, "satu")), _pieces((0.0, 1.0, "dua")), _pieces((0.0, 1.0, "tiga"))]
    )

    segments, _words = transcribe(
        video_path=tmp_path / "source.mp4",
        api_key="test-key",
        model="gemini-test",
        work_dir=tmp_path,
        duration=2000.0,
    )

    assert fake_gemini["extracted"] == [(0.0, 900.0), (900.0, 900.0), (1800.0, 200.0)]
    assert [s.start for s in segments] == [0.0, 900.0, 1800.0]
    assert [s.text for s in segments] == ["satu", "dua", "tiga"]


def test_transcribe_empty_answer_gives_no_words(tmp_path, fake_gemini):
    fake_gemini["payloads"].append("[]")
    segments, words = transcribe(
        video_path=tmp_path / "source.mp4",
        api_key="test-key",
        model="gemini-test",
        work_dir=tmp_path,
        duration=60.0,
    )
    assert segments == [] and words == []


def test_transcribe_reports_progress(tmp_path, fake_gemini):
    fake_gemini["payloads"].append(_pieces((0.0, 1.0, "ya")))
    steps = []
    transcribe(
        video_path=tmp_path / "source.mp4",
        api_key="test-key",
        model="gemini-test",
        work_dir=tmp_path,
        duration=60.0,
        progress=lambda msg, frac: steps.append(frac),
    )
    assert steps[0] == 0.0 and steps[-1] == 1.0


@needs_ffmpeg
def test_extract_audio_makes_small_mono_mp3(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=10",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(source),
        ],
        check=True,
    )
    dest = tmp_path / "audio.mp3"
    extract_audio(source, 2.0, 3.0, dest)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,channels,sample_rate:format=duration",
         "-of", "json", str(dest)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(probe.stdout)
    stream = info["streams"][0]
    assert stream["codec_name"] == "mp3"
    assert stream["channels"] == 1
    assert stream["sample_rate"] == "16000"
    assert 2.5 <= float(info["format"]["duration"]) <= 3.5
