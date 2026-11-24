from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from gemclip import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    test_settings = replace(server.settings, work_dir=tmp_path, gemini_api_key="test-key")
    monkeypatch.setattr(server, "settings", test_settings)
    return TestClient(server.app)


def _body(**overrides):
    body = {"url": "https://youtu.be/abc123", "clips": 5}
    body.update(overrides)
    return body


def test_rejects_non_youtube_url(client):
    res = client.post("/api/jobs", json=_body(url="https://example.com/video"))
    assert res.status_code == 400


def test_rejects_missing_api_key(client, monkeypatch):
    monkeypatch.setattr(server, "settings", replace(server.settings, gemini_api_key=""))
    res = client.post("/api/jobs", json=_body())
    assert res.status_code == 400
    assert "GEMINI_API_KEY" in res.json()["detail"]


def test_rejects_clip_count_out_of_range(client):
    assert client.post("/api/jobs", json=_body(clips=50)).status_code == 422


def test_rejects_min_duration_greater_than_max(client):
    res = client.post("/api/jobs", json=_body(clip_min_sec=60, clip_max_sec=30))
    assert res.status_code == 400
    assert "minimum" in res.json()["detail"].lower()


def test_rejects_end_before_start(client):
    res = client.post("/api/jobs", json=_body(start="10:00", end="05:00"))
    assert res.status_code == 400


def test_rejects_malformed_timecode(client):
    res = client.post("/api/jobs", json=_body(start="sepuluh menit"))
    assert res.status_code == 400


def test_rejects_too_long_instruction(client):
    assert client.post("/api/jobs", json=_body(instruction="x" * 501)).status_code == 422


def test_rejects_unknown_analysis_mode(client):
    assert client.post("/api/jobs", json=_body(analysis="gambar")).status_code == 422


def test_rejects_unknown_ratio_and_caption_style(client):
    assert client.post("/api/jobs", json=_body(ratio="4:3")).status_code == 422
    assert client.post("/api/jobs", json=_body(caption_style="pelangi")).status_code == 422


def test_accepts_supported_ratio_and_style(client, monkeypatch):
    monkeypatch.setattr(server, "_worker", lambda *args, **kwargs: None)
    res = client.post("/api/jobs", json=_body(ratio="16:9", caption_style="neon"))
    assert res.status_code == 200
    job = client.get(f"/api/jobs/{res.json()['job_id']}").json()
    assert job["ratio"] == "16:9"
    assert job["caption_style"] == "neon"


def test_srt_files_are_served(client, tmp_path):
    job_dir = server.settings.work_dir / "abcdef012345"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "clip_01.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhalo\n", encoding="utf-8")
    res = client.get("/files/abcdef012345/clip_01.srt")
    assert res.status_code == 200
    assert "halo" in res.text


def test_unknown_job_returns_404(client):
    assert client.get("/api/jobs/000000000000").status_code == 404
    assert client.get("/api/jobs/../../etc").status_code == 404


def test_file_route_blocks_path_traversal(client):
    assert client.get("/files/000000000000/..%2F..%2Fsecret.txt").status_code == 404
    assert client.get("/files/000000000000/clips.json").status_code == 404


def test_config_endpoint(client):
    body = client.get("/api/config").json()
    assert body["max_clips"] == 20
    assert body["max_clip_sec"] == 180
    assert body["has_api_key"] is True


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("", None),
        ("90", 90.0),
        ("1:30", 90.0),
        ("1:02:03", 3723.0),
        ("10:00.5", 600.5),
    ],
)
def test_parse_timecode(value, expected):
    assert server.parse_timecode(value) == expected


@pytest.mark.parametrize("value", ["abc", "1:2:3:4", "-5", "1:x"])
def test_parse_timecode_rejects_invalid(value):
    with pytest.raises(ValueError):
        server.parse_timecode(value)
