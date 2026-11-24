import json
from types import SimpleNamespace

import pytest

from gemclip import gemini as gemini_mod
from gemclip import moments as moments_mod
from gemclip.models import Segment, Word


class _FakeFiles:
    def __init__(self, final_state="ACTIVE"):
        self.final_state = final_state
        self.deleted = []
        self.uploaded = []

    def upload(self, file, config):
        assert config.mime_type == "video/mp4"
        self.uploaded.append(file)
        return SimpleNamespace(name=f"files/fake{len(self.uploaded)}", state=SimpleNamespace(name="PROCESSING"))

    def get(self, name):
        return SimpleNamespace(name=name, state=SimpleNamespace(name=self.final_state))

    def delete(self, name):
        self.deleted.append(name)


class _FakeModels:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text=self.payload)


class _FakeClient:
    payload = "[]"
    final_state = "ACTIVE"
    instances: list["_FakeClient"] = []

    def __init__(self, api_key):
        assert api_key == "test-key"
        self.files = _FakeFiles(self.final_state)
        self.models = _FakeModels(self.payload)
        _FakeClient.instances.append(self)


@pytest.fixture
def fake_genai(monkeypatch, tmp_path):
    _FakeClient.instances = []
    _FakeClient.payload = "[]"
    _FakeClient.final_state = "ACTIVE"
    monkeypatch.setattr(moments_mod.genai, "Client", _FakeClient)
    monkeypatch.setattr(gemini_mod.time, "sleep", lambda _s: None)

    def fake_cut(source, start, duration, dest):
        dest.write_bytes(b"chunk")
        return dest

    monkeypatch.setattr(moments_mod, "cut_chunk", fake_cut)
    return _FakeClient


def _payload():
    return json.dumps(
        [
            {
                "start_sec": 12.0,
                "end_sec": 48.0,
                "viral_score": 91,
                "title": "Rahasia produktivitas",
                "hook": "Ini yang sering dilewatkan",
                "caption": "Satu kebiasaan kecil yang mengubah hari.",
                "hashtags": ["produktivitas", "#tips"],
            },
            {
                "start_sec": 60.0,
                "end_sec": 95.0,
                "viral_score": 70,
                "title": "Cerita kegagalan",
                "hook": "Saya hampir menyerah",
                "caption": "Momen yang menentukan.",
                "hashtags": ["motivasi"],
            },
        ]
    )


def _video(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake")
    return video


def _words(n=200):
    return [Word(start=i * 0.5, end=i * 0.5 + 0.4, text="kata") for i in range(n)]


def test_find_moments_full_flow(fake_genai, tmp_path):
    fake_genai.payload = _payload()
    segments = [Segment(start=0.0, end=100.0, text="contoh transkrip")]

    result = moments_mod.find_moments(
        api_key="test-key",
        model="gemini-test",
        video_path=_video(tmp_path),
        segments=segments,
        words=_words(),
        duration=120.0,
        count=5,
        min_sec=30,
        max_sec=60,
    )

    client = fake_genai.instances[0]
    call = client.models.calls[0]
    assert call["model"] == "gemini-test"
    assert call["config"].response_mime_type == "application/json"
    assert "[0.0-100.0] contoh transkrip" in call["contents"][1]
    assert client.files.deleted == ["files/fake1"]
    assert not list(tmp_path.glob("gemini_part_*.mp4")), "potongan sementara harus dihapus"

    assert [m.score for m in result] == [91, 70]
    assert result[0].hashtags == ("#produktivitas", "#tips")
    assert result[0].end - result[0].start >= 30


def test_find_moments_splits_long_video_into_windows(fake_genai, tmp_path):
    fake_genai.payload = _payload()
    moments_mod.find_moments(
        api_key="test-key",
        model="gemini-test",
        video_path=_video(tmp_path),
        segments=[],
        words=_words(),
        duration=2000.0,
        count=5,
        min_sec=30,
        max_sec=60,
    )
    client = fake_genai.instances[0]
    # 2000 detik dengan potongan 900 detik dan tumpang tindih 60 detik = 3 potongan
    assert len(client.models.calls) == 3
    assert len(client.files.uploaded) == 3


def test_find_moments_shifts_times_by_window_offset(fake_genai, tmp_path):
    fake_genai.payload = _payload()
    result = moments_mod.find_moments(
        api_key="test-key",
        model="gemini-test",
        video_path=_video(tmp_path),
        segments=[],
        words=_words(4000),
        duration=2000.0,
        count=10,
        min_sec=30,
        max_sec=60,
    )
    # potongan kedua mulai di 840 detik, jadi kandidat dari sana harus berada di atas 840
    assert any(m.start >= 840 for m in result)
    assert all(0 <= m.start < m.end <= 2000 for m in result)


def test_find_moments_respects_range_and_instruction(fake_genai, tmp_path):
    fake_genai.payload = _payload()
    result = moments_mod.find_moments(
        api_key="test-key",
        model="gemini-test",
        video_path=_video(tmp_path),
        segments=[Segment(start=650.0, end=700.0, text="bagian yang dipilih")],
        words=_words(4000),
        duration=3600.0,
        count=5,
        min_sec=30,
        max_sec=60,
        range_start=600.0,
        range_end=1100.0,
        instruction="fokus pada tips keuangan",
    )
    client = fake_genai.instances[0]
    assert len(client.models.calls) == 1
    prompt = client.models.calls[0]["contents"][1]
    assert "fokus pada tips keuangan" in prompt
    assert "[50.0-100.0] bagian yang dipilih" in prompt  # transkrip relatif terhadap awal potongan (600 detik)
    assert all(600 <= m.start and m.end <= 1100 for m in result)


def test_find_moments_fails_when_gemini_processing_fails(fake_genai, tmp_path):
    fake_genai.final_state = "FAILED"
    with pytest.raises(RuntimeError, match="gagal memproses"):
        moments_mod.find_moments(
            api_key="test-key",
            model="gemini-test",
            video_path=_video(tmp_path),
            segments=[],
            words=[],
            duration=60.0,
            count=3,
            min_sec=30,
            max_sec=60,
        )


def test_find_moments_rejects_empty_range(fake_genai, tmp_path):
    with pytest.raises(RuntimeError, match="Rentang waktu"):
        moments_mod.find_moments(
            api_key="test-key",
            model="gemini-test",
            video_path=_video(tmp_path),
            segments=[],
            words=[],
            duration=600.0,
            count=3,
            min_sec=30,
            max_sec=60,
            range_start=300.0,
            range_end=100.0,
        )


def test_transcript_mode_sends_single_text_request(fake_genai, tmp_path):
    fake_genai.payload = json.dumps(
        [
            {"start_sec": 600.0, "end_sec": 640.0, "viral_score": 88, "title": "A", "hook": "H",
             "caption": "C", "hashtags": ["a"]},
            {"start_sec": 700.0, "end_sec": 740.0, "viral_score": 60, "title": "B", "hook": "H",
             "caption": "C", "hashtags": ["b"]},
        ]
    )
    segments = [
        Segment(start=600.0, end=650.0, text="kalimat pertama"),
        Segment(start=1500.0, end=1520.0, text="di luar rentang"),
    ]
    result = moments_mod.find_moments(
        api_key="test-key",
        model="gemini-test",
        video_path=_video(tmp_path),
        segments=segments,
        words=_words(4000),
        duration=3600.0,
        count=5,
        min_sec=30,
        max_sec=60,
        range_start=500.0,
        range_end=1000.0,
        instruction="fokus pada tips",
        analysis="transcript",
    )
    client = fake_genai.instances[0]
    assert len(client.models.calls) == 1
    assert client.files.uploaded == [], "mode transkrip tidak boleh mengunggah video"
    contents = client.models.calls[0]["contents"]
    assert isinstance(contents, str)
    assert "[600.0-650.0] kalimat pertama" in contents  # waktu tetap absolut
    assert "di luar rentang" not in contents
    assert "fokus pada tips" in contents
    assert [m.score for m in result] == [88, 60]
    assert all(500 <= m.start and m.end <= 1000 for m in result)


def test_find_moments_rejects_unknown_analysis_mode(fake_genai, tmp_path):
    with pytest.raises(ValueError):
        moments_mod.find_moments(
            api_key="test-key",
            model="gemini-test",
            video_path=_video(tmp_path),
            segments=[],
            words=[],
            duration=60.0,
            count=3,
            min_sec=30,
            max_sec=60,
            analysis="gambar",
        )
